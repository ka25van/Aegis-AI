import logging
from typing import Any, Dict, Optional

import httpx

from app.mcp.registry import registry, ToolRegistry

logger = logging.getLogger(__name__)

PROMETHEUS_URL = "http://prometheus:9090"
QUERY_TIMEOUT = 15


async def _prom_query(query: str) -> dict:
    """Execute a PromQL instant query and return parsed results."""
    try:
        async with httpx.AsyncClient(timeout=QUERY_TIMEOUT) as client:
            resp = await client.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": query},
            )
            resp.raise_for_status()
            data = resp.json()
            if data["status"] != "success":
                return {"status": "error", "error": data.get("error", "unknown Prometheus error")}
            return {"status": "success", "data": data["data"]}
    except httpx.ConnectError:
        return {"status": "error", "error": f"Cannot connect to Prometheus at {PROMETHEUS_URL}"}
    except httpx.TimeoutException:
        return {"status": "error", "error": "Prometheus query timed out"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _format_metric_result(data: dict) -> list:
    """Normalize Prometheus result vector into a list of dicts."""
    result_type = data.get("resultType", "")
    results = data.get("result", [])
    if result_type == "vector":
        return [
            {
                "metric": r["metric"],
                "value": float(r["value"][1]) if r.get("value") else None,
                "timestamp": r["value"][0] if r.get("value") else None,
            }
            for r in results
        ]
    if result_type == "matrix":
        return [
            {
                "metric": r["metric"],
                "values": [
                    {"timestamp": v[0], "value": float(v[1])} for v in r.get("values", [])
                ],
            }
            for r in results
        ]
    if result_type == "scalar":
        if results:
            return [{"value": float(results[1]), "timestamp": results[0]}]
        return []
    return results


def _summarize_metric(entries: list, label_key: str = "metric") -> dict:
    """Summarize a list of metric results into a readable dict."""
    if not entries:
        return {"summary": "No data available", "values": []}
    values = [e.get("value") for e in entries if e.get("value") is not None]
    if not values:
        return {"summary": "No numeric values", "values": entries}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "average": sum(values) / len(values),
        "total": sum(values),
        "values": entries,
    }


class PrometheusMCP:
    """MCP adapter for Prometheus metrics queries.

    Provides both raw PromQL access and higher-level convenience wrappers
    for common operational queries.
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry

    async def query(self, args: dict, context: dict = None) -> dict:
        """Execute an arbitrary PromQL query and return results."""
        query_str = args.get("query", "")
        if not query_str:
            return {"status": "completed", "result": {"error": "PromQL query is required"}}

        raw = await _prom_query(query_str)
        if raw["status"] == "error":
            return {"status": "completed", "result": {"error": raw["error"]}}

        entries = _format_metric_result(raw["data"])
        return {
            "status": "completed",
            "result": {
                "query": query_str,
                "result_type": raw["data"].get("resultType", ""),
                "count": len(entries),
                "results": entries,
            },
        }

    async def http_requests(self, args: dict, context: dict = None) -> dict:
        """Get HTTP request count and rate."""
        duration = args.get("duration", "5m")
        query = f'sum(rate(http_requests_total[{duration}]))'
        raw = await _prom_query(query)
        if raw["status"] == "error":
            return {"status": "completed", "result": {"error": raw["error"]}}

        entries = _format_metric_result(raw["data"])
        rate_value = entries[0]["value"] if entries else 0
        return {
            "status": "completed",
            "result": {
                "metric": "http_requests_total",
                "duration": duration,
                "request_rate_per_second": rate_value,
                "query": query,
            },
        }

    async def error_rate(self, args: dict, context: dict = None) -> dict:
        """Get HTTP error rate (5xx responses) as a percentage of total requests."""
        duration = args.get("duration", "5m")
        total_q = f'sum(rate(http_requests_total[{duration}]))'
        error_q = f'sum(rate(http_requests_total{{status=~"5.."}}[{duration}]))'

        total_raw = await _prom_query(total_q)
        error_raw = await _prom_query(error_q)
        if total_raw["status"] == "error":
            return {"status": "completed", "result": {"error": total_raw["error"]}}

        total_entries = _format_metric_result(total_raw["data"])
        error_entries = _format_metric_result(error_raw["data"])
        total_rate = total_entries[0]["value"] if total_entries else 0
        error_rate_val = error_entries[0]["value"] if error_entries else 0
        percentage = (error_rate_val / total_rate * 100) if total_rate > 0 else 0

        return {
            "status": "completed",
            "result": {
                "metric": "http_5xx_rate",
                "duration": duration,
                "total_requests_per_second": total_rate,
                "error_requests_per_second": error_rate_val,
                "error_percentage": round(percentage, 2),
            },
        }

    async def latency(self, args: dict, context: dict = None) -> dict:
        """Get request latency percentiles (p50, p95, p99)."""
        duration = args.get("duration", "5m")
        quantiles = {}
        for label, q in [("p50", 0.50), ("p95", 0.95), ("p99", 0.99)]:
            query = f'histogram_quantile({q}, sum(rate(http_request_duration_seconds_bucket[{duration}])) by (le))'
            raw = await _prom_query(query)
            if raw["status"] == "error":
                quantiles[label] = None
            else:
                entries = _format_metric_result(raw["data"])
                quantiles[label] = entries[0]["value"] if entries else None

        return {
            "status": "completed",
            "result": {
                "metric": "http_request_duration_seconds",
                "duration": duration,
                "p50_seconds": quantiles.get("p50"),
                "p95_seconds": quantiles.get("p95"),
                "p99_seconds": quantiles.get("p99"),
            },
        }

    async def active_runs(self, args: dict, context: dict = None) -> dict:
        """Get the current number of active agent runs."""
        raw = await _prom_query("active_runs")
        if raw["status"] == "error":
            return {"status": "completed", "result": {"error": raw["error"]}}

        entries = _format_metric_result(raw["data"])
        value = entries[0]["value"] if entries else 0
        return {
            "status": "completed",
            "result": {
                "metric": "active_runs",
                "value": int(value) if value else 0,
            },
        }

    async def total_errors(self, args: dict, context: dict = None) -> dict:
        """Get total error count by error type."""
        raw = await _prom_query('total_errors')
        if raw["status"] == "error":
            return {"status": "completed", "result": {"error": raw["error"]}}

        entries = _format_metric_result(raw["data"])
        by_type = {}
        for e in entries:
            err_type = e.get("metric", {}).get("type", "unknown")
            by_type[err_type] = e.get("value", 0)

        return {
            "status": "completed",
            "result": {
                "metric": "total_errors",
                "by_type": by_type,
                "total": sum(by_type.values()),
            },
        }

    async def resource_usage(self, args: dict, context: dict = None) -> dict:
        """Query container resource usage (CPU/memory) if kubelet metrics are available.

        Requires cAdvisor or kubelet metrics to be scraped by Prometheus.
        Returns a helpful message if those metrics are not available.
        """
        queries = {
            "cpu": 'sum(rate(container_cpu_usage_seconds_total[5m])) by (namespace, pod)',
            "memory": 'sum(container_memory_usage_bytes) by (namespace, pod)',
        }
        results = {}
        for name, q in queries.items():
            raw = await _prom_query(q)
            if raw["status"] == "success":
                entries = _format_metric_result(raw["data"])
                if entries:
                    results[name] = _summarize_metric(entries)
                else:
                    results[name] = {"summary": f"No {name} metrics found"}
            else:
                results[name] = {"summary": f"{name} metrics not available. Configure cAdvisor/kubelet scraping.", "error": raw.get("error")}

        return {
            "status": "completed",
            "result": {
                "available": bool(results.get("cpu", {}).get("values") or results.get("memory", {}).get("values")),
                "resources": results,
            },
        }


prom_adapter = PrometheusMCP(registry)

# ---- Register tools ----

registry.register(
    "prom_query",
    "Execute an arbitrary PromQL query against Prometheus",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Valid PromQL query string"},
        },
        "required": ["query"],
    },
    prom_adapter.query,
)

registry.register(
    "prom_http_requests",
    "Get HTTP request rate (requests per second) from the backend",
    {
        "type": "object",
        "properties": {
            "duration": {"type": "string", "default": "5m", "description": "Time window (e.g. 5m, 1h, 1d)"},
        },
    },
    prom_adapter.http_requests,
)

registry.register(
    "prom_error_rate",
    "Get HTTP 5xx error rate as percentage of total requests",
    {
        "type": "object",
        "properties": {
            "duration": {"type": "string", "default": "5m", "description": "Time window (e.g. 5m, 1h, 1d)"},
        },
    },
    prom_adapter.error_rate,
)

registry.register(
    "prom_latency",
    "Get request latency percentiles (p50, p95, p99) from the backend",
    {
        "type": "object",
        "properties": {
            "duration": {"type": "string", "default": "5m", "description": "Time window (e.g. 5m, 1h, 1d)"},
        },
    },
    prom_adapter.latency,
)

registry.register(
    "prom_active_runs",
    "Get the current number of active agent runs tracked by Prometheus",
    {
        "type": "object",
        "properties": {},
    },
    prom_adapter.active_runs,
)

registry.register(
    "prom_total_errors",
    "Get total error count broken down by error type",
    {
        "type": "object",
        "properties": {},
    },
    prom_adapter.total_errors,
)

registry.register(
    "prom_resource_usage",
    "Query container CPU/memory usage if kubelet or cAdvisor metrics are available",
    {
        "type": "object",
        "properties": {},
    },
    prom_adapter.resource_usage,
)


def get_prometheus_mcp() -> PrometheusMCP:
    return prom_adapter
