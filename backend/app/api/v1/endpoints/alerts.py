import asyncio
import json
import logging
import re
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.di import get_db_session
from app.db.session import async_session_maker
from app.services.embeddings import EmbeddingService
from app.services.llm_service import LLMService
from app.services.memory import MemorySystem, get_memory_system
from app.services.observability import ObservabilityService, get_observability_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])

_ALERT_ANALYSIS_SYSTEM_PROMPT = """You are a senior SRE analyzing a Prometheus alert.
Return ONLY valid JSON with these exact keys:
- root_cause (str): what is likely causing this alert
- impact (str): what systems/functionality are affected
- severity (str): critical|high|medium|low
- remediation_steps (list of str): actionable step-by-step remediation
- prevention (list of str): steps to prevent recurrence
- confidence (float): 0.0 to 1.0"""


def _extract_json(text: str) -> str:
    """Extract JSON from LLM text that may be wrapped in markdown."""
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip()


async def _process_alert_background(alert_body: str, incident_summary: str, description: str):
    """Background analysis of a Prometheus alert using LLM.

    Runs in a fire-and-forget task after the webhook returns.
    Stores structured analysis in semantic memory.
    """
    try:
        llm = LLMService()
        raw = await asyncio.wait_for(
            llm.generate(_ALERT_ANALYSIS_SYSTEM_PROMPT, alert_body),
            timeout=120.0,
        )
        parsed = json.loads(_extract_json(raw))
    except asyncio.TimeoutError:
        logger.warning("Alert analysis timed out for: %s", incident_summary)
        return
    except (json.JSONDecodeError, ValueError):
        logger.warning("Alert analysis returned non-JSON for: %s", incident_summary)
        return
    except Exception as e:
        logger.warning("Alert analysis LLM call failed: %s", e)
        return

    try:
        async with async_session_maker() as db:
            embeddings = EmbeddingService(db)
            memory = MemorySystem(db, embeddings)
            analysis_text = (
                f"Impact: {parsed.get('impact', 'unknown')}\n"
                f"Severity: {parsed.get('severity', 'medium')}\n"
                f"Confidence: {parsed.get('confidence', 0.5)}\n"
                f"Remediation: {'; '.join(parsed.get('remediation_steps', []))}\n"
                f"Prevention: {'; '.join(parsed.get('prevention', []))}"
            )
            emb = (await embeddings.generate_embeddings([analysis_text]))[0]
            if emb:
                await memory.store_semantic(
                    text=f"Alert Analysis: {parsed.get('root_cause', incident_summary)}",
                    embedding=emb,
                    metadata={
                        "type": "alert_analysis",
                        "alert_summary": incident_summary,
                        "root_cause": parsed.get("root_cause", ""),
                        "severity": parsed.get("severity", "medium"),
                        "confidence": parsed.get("confidence", 0.5),
                        "has_remediation": bool(parsed.get("remediation_steps")),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
                logger.info("Alert analysis stored for: %s", incident_summary)
    except Exception as e:
        logger.warning("Failed to store alert analysis: %s", e)


class Alert(BaseModel):
    status: str
    labels: dict
    annotations: dict
    startsAt: str
    endsAt: str = ""
    generatorURL: str = ""
    fingerprint: str = ""


class AlertmanagerWebhook(BaseModel):
    receiver: str = ""
    status: str
    alerts: List[Alert]
    groupLabels: dict = {}
    commonLabels: dict = {}
    commonAnnotations: dict = {}
    externalURL: str = ""


@router.post("/webhook")  # public: secured via ALERT_WEBHOOK_SECRET (see config.py)
async def alert_webhook(
    payload: AlertmanagerWebhook,
    x_webhook_secret: str | None = Header(None, alias="X-Webhook-Secret"),
    db: AsyncSession = Depends(get_db_session),
    obs: ObservabilityService = Depends(get_observability_service),
    memory: MemorySystem = Depends(get_memory_system),
):
    """Receive Prometheus Alertmanager webhook payload.

    If ALERT_WEBHOOK_SECRET is set (prod), require X-Webhook-Secret header.
    If empty (dev), webhook is open — compose/k8s should set it in prod.
    """
    if settings.ALERT_WEBHOOK_SECRET:
        if x_webhook_secret != settings.ALERT_WEBHOOK_SECRET:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook secret")
    # Idempotency: deduplicate Alertmanager retries by fingerprints within 5m
    try:
        from app.core.redis import get_redis
        r = get_redis()
        fps = sorted(a.fingerprint for a in payload.alerts if a.fingerprint)
        if fps:
            import hashlib
            dedup_key = "alert_fps:" + hashlib.md5(",".join(fps).encode()).hexdigest()
            if await r.get(dedup_key):
                return {"status": "duplicate", "alert_count": len(payload.alerts), "dedup": True}
            await r.setex(dedup_key, 300, "1")
    except Exception:
        pass
    try:
        await obs.record_request("POST", "/alerts/webhook", 200, 0)
    except Exception:
        pass

    incident_summary = f"Alert {payload.status}: {len(payload.alerts)} alert(s)"
    if payload.commonAnnotations:
        summary = payload.commonAnnotations.get("summary", "")
        if summary:
            incident_summary = summary

    description = payload.commonAnnotations.get("description", incident_summary)
    alert_names = [a.labels.get("alertname", "unknown") for a in payload.alerts]

    body = (
        f"Alertmanager Notification — Status: {payload.status}\n"
        f"Alerts ({len(payload.alerts)}):\n" + "\n".join(
            f"  - {a.labels.get('alertname', 'unknown')}: {a.annotations.get('summary', 'no summary')}"
            for a in payload.alerts[:20]
        )
    )

    # Store raw alert in semantic memory
    try:
        emb = (await memory.embeddings.generate_embeddings([body]))[0]
        if emb:
            await memory.store_semantic(
                text=f"Alert: {incident_summary}\n{description}",
                embedding=emb,
                metadata={
                    "type": "alert",
                    "status": payload.status,
                    "alert_names": alert_names,
                    "alert_count": len(payload.alerts),
                    "timestamp": datetime.utcnow().isoformat(),
                    "processing": "pending",
                },
            )
    except Exception as e:
        logger.warning("Failed to store alert in memory: %s", e)

    # Fire background analysis immediately after storing
    asyncio.create_task(_process_alert_background(body, incident_summary, description))

    return {
        "status": "received",
        "alert_count": len(payload.alerts),
        "status_group": payload.status,
        "stored": True,
        "processing": "in_background",
    }





@router.get("/history")
async def alert_history(
    limit: int = 50,
    db: AsyncSession = Depends(get_db_session),
    memory: MemorySystem = Depends(get_memory_system),
):
    """List recent alerts from semantic memory."""
    try:
        results = await memory.search_semantic("alert notification incident", limit=limit, threshold=0.0)
        return {"alerts": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def alert_stats(
    db: AsyncSession = Depends(get_db_session),
    memory: MemorySystem = Depends(get_memory_system),
):
    """Get alert statistics by querying semantic memory."""
    try:
        results = await memory.search_semantic("alert notification incident", limit=200, threshold=0.0)
        firing = sum(1 for r in results if r.get("metadata", {}).get("status") == "firing")
        resolved = sum(1 for r in results if r.get("metadata", {}).get("status") == "resolved")
        return {
            "total_alerts": len(results),
            "firing": firing,
            "resolved": resolved,
            "unknown": len(results) - firing - resolved,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
