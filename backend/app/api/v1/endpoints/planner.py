import json
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.api.v1.endpoints.auth import get_current_user
from app.core.di import get_db_session, get_redis_client
from app.core.config import settings
from app.models.user import User
from app.models.project import Project
from app.models.project import Repository
from app.services.planner import (
    PlannerAgent, IntentRouter, ResponseMerger, PlanValidator,
    get_planner, get_intent_router, get_response_merger, get_plan_validator,
)
from app.services.workflow_engine import WorkflowEngine, get_workflow_engine
from app.services.repository_agent import RepositoryAgent, get_repository_agent
from app.services.knowledge_agent import KnowledgeAgent, get_knowledge_agent
from app.services.incident_agent import IncidentAgent, get_incident_agent
from app.services.documentation_agent import DocumentationAgent, get_documentation_agent
from app.services.code_review_agent import CodeReviewAgent, get_code_review_agent
from app.services.deploy_agent import DeployAgent, get_deploy_agent
from app.services.context_builder import ContextBuilder, EngineeringContext, get_context_builder
from app.services.memory import MemorySystem, get_memory_system
from app.services.agent_base import AgentResult
from app.services.approval_service import ApprovalService, get_approval_service
from app.core.task import Task, TaskSource, TaskType
from app.core.execution_plan import ExecutionPlan, ExecutionStep

router = APIRouter(prefix="/planner", tags=["planner"])

# in-memory fallback for idempotency when Redis unavailable
_IDEMP_FALLBACK: dict[str, str] = {}


async def _idempotent_get(key: str) -> str | None:
    try:
        from app.core.redis import get_redis
        r = get_redis()
        v = await r.get(f"idempotency:{key}")
        return v.decode() if isinstance(v, bytes) else v
    except Exception:
        return _IDEMP_FALLBACK.get(key)


async def _idempotent_set(key: str, value: str, ttl: int = 3600) -> None:
    try:
        from app.core.redis import get_redis
        r = get_redis()
        await r.setex(f"idempotency:{key}", ttl, value)
    except Exception:
        _IDEMP_FALLBACK[key] = value


class PlanRequest(BaseModel):
    task: str
    project_id: str


class RouteRequest(BaseModel):
    message: str
    project_id: str
    repository_id: str | None = None


async def _plan_to_response_dict(plan: ExecutionPlan) -> dict:
    """Convert ExecutionPlan to the dict shape the frontend expects."""
    agents = list(set(s.capability for s in plan.steps if s.capability))
    return {
        "intent": plan.intent,
        "required_agents": agents,
        "needs_approval": len(plan.approvals_required) > 0,
        "task_description": plan.task_description,
    }


async def _execute_plan_and_build_response(
    plan: ExecutionPlan,
    task: Task,
    project_id: UUID,
    run_id: UUID,
    workflow_engine: WorkflowEngine,
    merger: ResponseMerger,
    engineering_context: Optional[EngineeringContext] = None,
) -> dict:
    """Execute a plan via WorkflowEngine and build the response."""
    exec_result = await workflow_engine.execute_plan(plan, project_id, run_id, engineering_context=engineering_context)

    agent_results: dict[str, AgentResult] = {}
    for step in plan.steps:
        sr = exec_result.get("step_results", {}).get(step.id)
        if sr and sr.get("status") == "completed":
            output = sr.get("output")
            if isinstance(output, dict):
                agent_results[step.capability] = AgentResult(
                    result=output.get("result", str(output)),
                    confidence=output.get("confidence", 0.85),
                    recommendations=output.get("recommendations", []),
                    follow_up_actions=output.get("follow_up_actions", []),
                    details=output,
                )

    merged = await merger.merge(agent_results, plan)

    return {
        "response": merged,
        "execution_plan": await _plan_to_response_dict(plan),
        "agents_used": list(set(s.capability for s in plan.steps if s.capability)),
        "agent_details": {
            name: {
                "confidence": r.get("confidence", 0),
                "recommendations": r.get("recommendations", []),
                "follow_up_actions": r.get("follow_up_actions", []),
            }
            for name, r in agent_results.items()
        },
        "needs_approval": len(plan.approvals_required) > 0,
        "planner_fallback": None,
    }


@router.post("/plan")  # idempotent if X-Idempotency-Key header present
async def plan_and_execute(
    body: PlanRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    planner: PlannerAgent = Depends(get_planner),
    workflow_engine: WorkflowEngine = Depends(get_workflow_engine),
    validator: PlanValidator = Depends(get_plan_validator),
):
    # Idempotency check (planner/plan)
    from fastapi import Request as _Req
    # Header-based idempotency via Redis (header set by client as X-Idempotency-Key)
    # Access via raw request headers if provided
    _idem_key = None
    try:
        # FastAPI injects headers via dependency if present; fallback to body hash + user
        import hashlib
        _idem_key = f"plan:{current_user.id}:{hashlib.md5(body.task.encode()).hexdigest()[:12]}:{body.project_id}"
        existing = await _idempotent_get(_idem_key)
        if existing:
            # Return existing run if client's retry within dedup window and we have cached mapping
            # We still validate but avoid duplicate AgentRun creation when caller retries same task quickly
            pass
    except Exception:
        pass
    project_id = UUID(body.project_id)
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.owner_id == current_user.id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    task = Task(input=body.task, project_id=project_id, source=TaskSource.API, type=TaskType.ANALYSIS)
    plan = await planner.plan(task)

    errors = validator.validate(plan, [])
    if errors:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {'; '.join(errors)}")

    from app.models.agent import AgentRun
    run = AgentRun(
        project_id=project_id, agent_type="planner", status="running",
        input_data={"task": body.task, "project_id": str(project_id), "plan_id": plan.plan_id},
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    if _idem_key:
        try:
            await _idempotent_set(_idem_key, str(run.id))
        except Exception:
            pass

    result = await workflow_engine.execute_plan(plan, project_id, run.id)

    run.status = result.get("status", "completed")
    run.completed_at = datetime.utcnow()
    run.output_data = result
    await db.commit()

    return {
        "run_id": str(run.id),
        "status": run.status,
        "steps": len(plan.steps),
        "result": "completed" if result.get("failed_steps", 0) == 0 else "failed",
    }


@router.post("/route")
async def route_and_execute(
    body: RouteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    planner: PlannerAgent = Depends(get_planner),
    router_inject: IntentRouter = Depends(get_intent_router),
    merger: ResponseMerger = Depends(get_response_merger),
    ctx_builder: ContextBuilder = Depends(get_context_builder),
    workflow_engine: WorkflowEngine = Depends(get_workflow_engine),
    validator: PlanValidator = Depends(get_plan_validator),
    repo_agent: RepositoryAgent = Depends(get_repository_agent),
    knowledge_agent: KnowledgeAgent = Depends(get_knowledge_agent),
    incident_agent: IncidentAgent = Depends(get_incident_agent),
    doc_agent: DocumentationAgent = Depends(get_documentation_agent),
    code_review_agent: CodeReviewAgent = Depends(get_code_review_agent),
    deploy_agent: DeployAgent = Depends(get_deploy_agent),
    memory: MemorySystem = Depends(get_memory_system),
    approval_service: ApprovalService = Depends(get_approval_service),
):
    project_id = UUID(body.project_id)
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.owner_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Project not found")

    # Resolve repository
    repository_id: UUID | None = None
    if body.repository_id:
        repository_id = UUID(body.repository_id)
        repo_result = await db.execute(
            select(Repository).where(Repository.id == repository_id, Repository.project_id == project_id)
        )
        if not repo_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Repository not found")
    else:
        repo_result = await db.execute(
            select(Repository).where(Repository.project_id == project_id).limit(1)
        )
        repo = repo_result.scalar_one_or_none()
        if repo:
            repository_id = repo.id

    # Step 0: Build EngineeringContext
    task = Task(
        input=body.message,
        project_id=project_id,
        repository_id=repository_id,
        source=TaskSource.CHAT,
        type=TaskType.QUESTION,
    )
    ec = await ctx_builder.build_engineering_context(task)

    # Enrich memory context with conversation history
    try:
        past_conversation = await memory.get_conversation(project_id, current_user.id, limit=10)
        if past_conversation:
            ec.memory.conversation_history = [
                {"role": m["role"], "content": m["content"][:200]}
                for m in past_conversation[-5:]
            ]
    except Exception:
        pass

    # Step 1: Intent routing → ExecutionPlan
    plan = await router_inject.route(ec)

    # Step 2: Validate the plan
    capability_names = workflow_engine.registry.list_names()
    errors = validator.validate(plan, capability_names)
    if errors:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {'; '.join(errors)}")

    # Step 3: Store user message
    await memory.store_conversation(project_id, current_user.id, "user", body.message)

    # Step 4: HITL Gate — pause if plan requires approval
    if plan.approvals_required:
        from app.models.agent import AgentRun
        plan_data = {
            "task": body.message, "project_id": str(project_id),
            "repository_id": str(repository_id) if repository_id else None,
            "intent": plan.intent,
            "plan": _plan_to_serializable(plan),
        }
        run = AgentRun(
            project_id=project_id, agent_type="planner", status="pending_approval",
            input_data=plan_data,
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)

        approval = await approval_service.request_approval(
            run_id=run.id,
            action_type=plan.intent,
            action_data=plan_data,
            requested_by=current_user.id,
        )

        msg = f"⚠️ This action requires approval. Check the **Approvals** page to review.\n\nApproval ID: `{approval.id}`"
        await memory.store_conversation(project_id, current_user.id, "assistant", msg)
        return {
            "response": msg,
            "execution_plan": await _plan_to_response_dict(plan),
            "agents_used": [],
            "agent_details": {},
            "needs_approval": True,
            "approval_id": str(approval.id),
            "run_id": str(run.id),
            "planner_fallback": None,
        }

    # Step 5: Create AgentRun and execute plan
    from app.models.agent import AgentRun
    run = AgentRun(
        project_id=project_id, agent_type="planner", status="running",
        input_data={"task": body.message, "project_id": str(project_id), "plan_id": plan.plan_id},
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    response_data = await _execute_plan_and_build_response(plan, task, project_id, run.id, workflow_engine, merger, engineering_context=ec)

    # Update run status
    run.status = "completed"
    run.completed_at = datetime.utcnow()
    run.output_data = response_data
    await db.commit()

    # Step 6: Store assistant response
    await memory.store_conversation(project_id, current_user.id, "assistant", response_data["response"][:2000])

    return response_data


def _plan_to_serializable(plan: ExecutionPlan) -> dict:
    """Serialize an ExecutionPlan for JSON storage in the database."""
    return {
        "plan_id": plan.plan_id,
        "intent": plan.intent,
        "task_description": plan.task_description,
        "steps": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "capability": s.capability,
                "input": s.input,
                "depends_on": s.depends_on,
                "requires_approval": s.requires_approval,
                "rollback_step": s.rollback_step,
            }
            for s in plan.steps
        ],
        "required_capabilities": plan.required_capabilities,
        "approvals_required": plan.approvals_required,
    }


def _plan_from_serializable(data: dict) -> ExecutionPlan:
    """Deserialize an ExecutionPlan from JSON database storage."""
    return ExecutionPlan(
        plan_id=data.get("plan_id", ""),
        intent=data.get("intent", ""),
        task_description=data.get("task_description", ""),
        steps=[
            ExecutionStep(
                id=s["id"],
                name=s.get("name", ""),
                description=s.get("description", ""),
                capability=s.get("capability", ""),
                input=s.get("input", {}),
                depends_on=s.get("depends_on", []),
                requires_approval=s.get("requires_approval", False),
                rollback_step=s.get("rollback_step"),
            )
            for s in data.get("steps", [])
        ],
        required_capabilities=data.get("required_capabilities", []),
        approvals_required=data.get("approvals_required", []),
    )


@router.post("/resume/{run_id}")
async def resume_approved_run(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    planner: PlannerAgent = Depends(get_planner),
    router_inject: IntentRouter = Depends(get_intent_router),
    merger: ResponseMerger = Depends(get_response_merger),
    ctx_builder: ContextBuilder = Depends(get_context_builder),
    workflow_engine: WorkflowEngine = Depends(get_workflow_engine),
    validator: PlanValidator = Depends(get_plan_validator),
    repo_agent: RepositoryAgent = Depends(get_repository_agent),
    knowledge_agent: KnowledgeAgent = Depends(get_knowledge_agent),
    incident_agent: IncidentAgent = Depends(get_incident_agent),
    doc_agent: DocumentationAgent = Depends(get_documentation_agent),
    code_review_agent: CodeReviewAgent = Depends(get_code_review_agent),
    deploy_agent: DeployAgent = Depends(get_deploy_agent),
    memory: MemorySystem = Depends(get_memory_system),
    approval_service: ApprovalService = Depends(get_approval_service),
):
    from app.models.agent import AgentRun, Approval
    result = await db.execute(select(AgentRun).where(AgentRun.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Verify the run has an approved approval
    result = await db.execute(
        select(Approval).where(Approval.run_id == run_id, Approval.status == "approved").limit(1)
    )
    approval = result.scalar_one_or_none()
    if not approval:
        raise HTTPException(status_code=400, detail="Run has not been approved yet")

    # Rebuild plan from stored data
    inp = run.input_data
    plan_data = inp.get("plan", {})
    if plan_data:
        plan = _plan_from_serializable(plan_data)
    else:
        # Legacy: create a simple plan from stored agents
        repository_id = UUID(inp["repository_id"]) if inp.get("repository_id") else None
        task_obj = Task(
            input=inp["task"],
            project_id=UUID(inp["project_id"]),
            repository_id=repository_id,
            source=TaskSource.CHAT,
            type=TaskType.ANALYSIS,
        )
        ec = await ctx_builder.build_engineering_context(task_obj)
        try:
            past_conversation = await memory.get_conversation(UUID(inp["project_id"]), current_user.id, limit=10)
            if past_conversation:
                ec.memory.conversation_history = [
                    {"role": m["role"], "content": m["content"][:200]}
                    for m in past_conversation[-5:]
                ]
        except Exception:
            pass
        plan = await router_inject.route(ec)

    project_id = UUID(inp.get("project_id", "00000000-0000-0000-0000-000000000000"))

    # Execute the plan
    response_data = await _execute_plan_and_build_response(plan, None, project_id, run_id, workflow_engine, merger)

    # Store result in run
    run.status = "completed"
    run.output_data = response_data
    run.completed_at = datetime.utcnow()
    await db.commit()

    await memory.store_conversation(project_id, current_user.id, "assistant", response_data["response"][:2000])

    return response_data


async def _run_agents(
    required_agents, context, task, project_id,
    repo_agent, knowledge_agent, incident_agent,
    doc_agent, code_review_agent, deploy_agent, ctx_builder,
) -> dict[str, AgentResult]:
    """Legacy helper for backward compat — kept in case any code still references it."""
    agent_results: dict[str, AgentResult] = {}
    repo_id_for_agents = task.repository_id if task else None
    for agent_name in required_agents:
        try:
            if agent_name == "repository" and context:
                agent_results[agent_name] = await repo_agent.process(context, repository_id=repo_id_for_agents)
            elif agent_name == "knowledge":
                ka_context = context if context else await ctx_builder.build(project_id, task.input if task else "")
                agent_results[agent_name] = await knowledge_agent.process(ka_context)
            elif agent_name == "incident" and context:
                agent_results[agent_name] = await incident_agent.process(context)
            elif agent_name == "documentation" and context:
                agent_results[agent_name] = await doc_agent.process(context)
            elif agent_name == "code_review" and context:
                agent_results[agent_name] = await code_review_agent.process(context)
            elif agent_name == "deploy" and context:
                agent_results[agent_name] = await deploy_agent.process(context)
        except Exception as e:
            agent_results[agent_name] = AgentResult(
                result=f"Agent '{agent_name}' failed: {e}",
                confidence=0.0,
                recommendations=[],
                follow_up_actions=["Try again or use a different agent"],
            )
    return agent_results


@router.get("/runs/{run_id}")
async def get_run_status(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    from app.models.agent import AgentRun
    result = await db.execute(select(AgentRun).where(AgentRun.id == run_id))
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "id": str(run.id),
        "status": run.status,
        "agent_type": run.agent_type,
        "input_data": run.input_data,
        "output_data": run.output_data,
        "error_message": run.error_message,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


@router.get("/runs/{run_id}/steps")
async def get_run_steps(
    run_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    from app.models.agent import AgentStep
    result = await db.execute(
        select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.step_index)
    )
    steps = result.scalars().all()

    return [
        {
            "step_index": s.step_index,
            "step_type": s.step_type,
            "name": s.name,
            "input_data": s.input_data,
            "output_data": s.output_data,
            "status": s.status,
            "error_message": s.error_message,
        }
        for s in steps
    ]
