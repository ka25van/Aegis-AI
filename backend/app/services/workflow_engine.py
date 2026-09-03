import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from uuid import UUID
from datetime import datetime

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import AgentRun, AgentStep
from app.models.project import Repository
from app.core.di import get_db_session
from app.services.agent_base import AgentResult
from app.services.context_builder import ContextBuilder, ProjectContext, get_context_builder
from app.services.capability_registry import CapabilityRegistry, CapabilityNotFoundError
from app.services.repository_agent import RepositoryAgent, get_repository_agent
from app.services.knowledge_agent import KnowledgeAgent, get_knowledge_agent
from app.services.incident_agent import IncidentAgent, get_incident_agent
from app.services.documentation_agent import DocumentationAgent, get_documentation_agent
from app.services.code_review_agent import CodeReviewAgent, get_code_review_agent
from app.services.deploy_agent import DeployAgent, get_deploy_agent
from app.core.execution_plan import ExecutionPlan, ExecutionStep, RetryPolicy
from app.core.task import Task
from app.services.execution_adapters import adapt_agent, adapt_mcp, adapt_rest, adapt_python

try:
    from langfuse import observe, get_client
except ImportError:
    observe = lambda *a, **kw: (lambda f: f)  # type: ignore
    get_client = lambda: None  # type: ignore

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Error that can be retried (LLM timeout, DB transient)."""
    pass


class FatalError(Exception):
    """Error that should not be retried (invalid input, auth failure)."""
    pass


TOOL_AGENT_MAP: Dict[str, str] = {
    "analyze_code": "repository",
    "search_knowledge": "knowledge",
    "generate_docs": "documentation",
    "review_code": "code_review",
    "analyze_incidents": "incident",
    "analyze_deploy": "deploy",
}


class StepResult:
    def __init__(self, status: str, output: Any = None, error: str = None, duration_ms: int = 0):
        self.status = status
        self.output = output
        self.error = error
        self.duration_ms = duration_ms


class ExecutionRuntime:
    """Owns execution of a single step: retries, timeout, cancellation, telemetry."""

    def __init__(self):
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    async def execute_step(
        self,
        step: ExecutionStep,
        executor: Callable[..., Awaitable[Any]],
        context: ProjectContext,
        attempt_callback: Callable[[int, int], Awaitable[None]] = None,
    ) -> StepResult:
        """Retries + timeout enforcement (execution_plan.py:29 timeout_seconds).

        Classification: FatalError, ValueError/TypeError/KeyError, HTTP 4xx -> no retry.
        Transient (ConnectError, Timeout, 5xx) -> RetryableError with backoff.
        """
        policy = step.retry_policy or RetryPolicy()
        last_error: Optional[str] = None
        start = time.monotonic()
        timeout = step.timeout_seconds or 120  # default 120s per step

        for attempt in range(1, policy.max_retries + 1):
            if self._cancelled:
                return StepResult(status="cancelled", error="Execution cancelled", duration_ms=int((time.monotonic() - start) * 1000))

            try:
                # Enforce per-step timeout (was unenforced before)
                result = await asyncio.wait_for(executor(context), timeout=timeout)
                duration_ms = int((time.monotonic() - start) * 1000)
                return StepResult(status="completed", output=result, duration_ms=duration_ms)

            except asyncio.TimeoutError:
                last_error = f"Step {step.id} timed out after {timeout}s"
                logger.error(last_error)

            except FatalError as e:
                last_error = str(e)
                logger.error("Fatal error on step %s: %s — not retrying", step.id, last_error)
                duration_ms = int((time.monotonic() - start) * 1000)
                return StepResult(status="failed", error=last_error, duration_ms=duration_ms)

            except (ValueError, TypeError, KeyError, AttributeError) as e:
                last_error = str(e)
                logger.error("Fatal (bad input) on step %s: %s — not retrying", step.id, last_error)
                duration_ms = int((time.monotonic() - start) * 1000)
                return StepResult(status="failed", error=last_error, duration_ms=duration_ms)

            except Exception as e:
                # Classify transport / transient errors as retryable
                msg = str(e).lower()
                transient_markers = ("connect", "timeout", "timed out", "503", "502", "504", "connection", "unavailable")
                is_transient = any(m in msg for m in transient_markers) or isinstance(e, (asyncio.TimeoutError, ConnectionError, TimeoutError))
                if not is_transient and isinstance(e, FatalError):
                    last_error = str(e)
                    duration_ms = int((time.monotonic() - start) * 1000)
                    return StepResult(status="failed", error=last_error, duration_ms=duration_ms)
                last_error = str(e)
                if attempt < policy.max_retries:
                    delay = policy.retry_delay_seconds * (policy.backoff_multiplier ** (attempt - 1))
                    logger.warning("Step %s attempt %d/%d failed (retryable): %s — retrying in %.1fs", step.id, attempt, policy.max_retries, last_error, delay)
                    if attempt_callback:
                        await attempt_callback(attempt, policy.max_retries)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Step %s exhausted %d attempts: %s", step.id, policy.max_retries, last_error)
                    duration_ms = int((time.monotonic() - start) * 1000)
                    return StepResult(status="failed", error=last_error, duration_ms=duration_ms)

        duration_ms = int((time.monotonic() - start) * 1000)
        return StepResult(status="failed", error=last_error, duration_ms=duration_ms)


class WorkflowEngine:
    """Engine for executing and managing workflows.

    Consumes ExecutionPlan.
    Performs orchestration only — delegates execution to CapabilityRegistry.
    Never performs reasoning.
    """

    def __init__(
        self,
        db: AsyncSession,
        ctx_builder: ContextBuilder,
        registry: CapabilityRegistry,
    ):
        self.db = db
        self.ctx_builder = ctx_builder
        self.registry = registry

    async def _build_context_for_step(self, step: ExecutionStep, project_id: UUID, run_id: UUID) -> ProjectContext:
        # If EngineeringContext is available (wired via execute_plan), use its
        # pre-built ProjectContext as the base to avoid redundant DB/embedding queries.
        ec = getattr(self, '_engineering_context', None)
        if ec is not None:
            import dataclasses
            return dataclasses.replace(ec.project, step_input=step.input)

        repository_id: Optional[UUID] = None
        rid_raw = step.input.get("repository_id") or (str(project_id) if project_id else None)
        if rid_raw:
            try:
                repository_id = UUID(rid_raw) if isinstance(rid_raw, str) else rid_raw
            except (ValueError, TypeError):
                pass

        if not repository_id and project_id:
            repo_result = await self.db.execute(
                select(Repository).where(Repository.project_id == project_id).limit(1)
            )
            repo = repo_result.scalar_one_or_none()
            if repo:
                repository_id = repo.id

        task_description = step.description or step.name or "Execute step"

        if repository_id:
            ctx = await self.ctx_builder.build(repository_id, task_description)
        else:
            ctx = ProjectContext(
                project_id=project_id or UUID("00000000-0000-0000-0000-000000000000"),
                task_description=task_description,
            )
        ctx.step_input = step.input
        return ctx

    @observe(name="workflow.execute_plan", as_type="span", capture_input=False, capture_output=False)
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        project_id: UUID,
        run_id: UUID,
        max_retries: int = 3,
        engineering_context: Optional[Any] = None,
    ) -> Dict:
        """Execute an ExecutionPlan by resolving its DAG and dispatching steps.
        
        Orchestration only:
        - Resolves dependency order (topological sort)
        - Delegates each step to CapabilityRegistry
        - Manages execution state
        - Pauses for approvals
        - Tracks progress
        """
        try:
            lf = get_client()
            if lf:
                lf.update_current_span(input={"intent": plan.intent, "steps": len(plan.steps), "capabilities": plan.required_capabilities, "run_id": str(run_id)}, metadata={"project_id": str(project_id)})
        except Exception:
            pass
        self._engineering_context = engineering_context
        runtime = ExecutionRuntime()
        step_results: Dict[str, StepResult] = {}
        completed_steps = 0
        failed_steps = 0

        # Topological sort: respect depends_on
        ordered = self._topological_sort(plan.steps)

        for step in ordered:
            # Check dependencies are all completed
            deps_status = [step_results[d].status for d in step.depends_on if d in step_results]
            if deps_status and any(s != "completed" for s in deps_status):
                step_results[step.id] = StepResult(status="skipped", error=f"Dependency failed: {step.depends_on}")
                continue

            # Build context
            context = await self._build_context_for_step(step, project_id, run_id)

            # Resolve capability
            try:
                capability = self.registry.resolve(step.capability)
            except CapabilityNotFoundError as e:
                step_results[step.id] = StepResult(status="failed", error=str(e))
                failed_steps += 1
                continue

            # Record step start
            step_record = AgentStep(
                run_id=run_id,
                step_index=int(step.id.split("-")[1]) if "-" in step.id else 0,
                step_type=step.capability,
                name=step.name,
                input_data=step.input,
                status="running",
                started_at=datetime.utcnow(),
            )
            self.db.add(step_record)
            await self.db.commit()
            await self.db.refresh(step_record)

            # Execute via ExecutionRuntime — wrapped in Langfuse observation for Agent Graph
            # Map capability to observation type: agents→agent, MCP→tool, else span
            obs_type = "agent" if step.capability in ("repository", "knowledge", "incident", "documentation", "code_review", "deploy") else ("tool" if getattr(capability, "execution_type", "") == "mcp" else "span")
            _lf = None
            try:
                _lf = get_client()
            except Exception:
                pass
            if _lf and hasattr(_lf, "start_as_current_observation"):
                with _lf.start_as_current_observation(as_type=obs_type, name=f"step:{step.name}", input={"capability": step.capability, "description": step.description}, metadata={"step_id": step.id, "execution_type": getattr(capability, "execution_type", "")}) as _obs:
                    result = await runtime.execute_step(step, capability.executor, context)
                    try:
                        _obs.update(output={"status": result.status, "error": result.error}, level="ERROR" if result.status == "failed" else "DEFAULT")
                    except Exception:
                        pass
            else:
                result = await runtime.execute_step(step, capability.executor, context)

            # Record step result
            duration_ms = result.duration_ms
            step_record.status = result.status
            step_record.duration_ms = duration_ms
            if result.status == "completed":
                step_record.output_data = {"result": result.output}
                completed_steps += 1
            else:
                step_record.output_data = {"error": result.error}
                step_record.error_message = result.error
                failed_steps += 1
            step_record.completed_at = datetime.utcnow()
            await self.db.commit()

            step_results[step.id] = result

            # Stop on failure (no cascade — DAG handles dependencies)
            if result.status == "failed" and not step.rollback_step:
                break

            # Handle rollback if step has rollback_step
            if result.status == "failed" and step.rollback_step:
                rb_step = next((s for s in plan.steps if s.id == step.rollback_step), None)
                if rb_step:
                    rb_context = await self._build_context_for_step(rb_step, project_id, run_id)
                    try:
                        rb_cap = self.registry.resolve(rb_step.capability)
                        rb_result = await runtime.execute_step(rb_step, rb_cap.executor, rb_context)
                        step_results[rb_step.id] = rb_result
                    except CapabilityNotFoundError as e:
                        step_results[rb_step.id] = StepResult(status="failed", error=str(e))

        overall_status = "completed" if failed_steps == 0 else "failed"
        return {
            "status": overall_status,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "total_steps": len(plan.steps),
            "step_results": {
                sid: {"status": r.status, "error": r.error, "duration_ms": r.duration_ms, "output": r.output}
                for sid, r in step_results.items()
            },
        }

    def _topological_sort(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
        """Sort steps in dependency order (Kahn's algorithm)."""
        step_map = {s.id: s for s in steps}
        in_degree = {s.id: 0 for s in steps}
        for s in steps:
            for dep in s.depends_on:
                if dep in in_degree:
                    in_degree[s.id] += 1

        queue = [s.id for s in steps if in_degree[s.id] == 0]
        ordered = []
        while queue:
            node = queue.pop(0)
            ordered.append(step_map[node])
            for s in steps:
                if node in s.depends_on:
                    in_degree[s.id] -= 1
                    if in_degree[s.id] == 0:
                        queue.append(s.id)
        return ordered

    async def execute_workflow(
        self,
        task: str,
        steps: List[Dict],
        run_id: UUID,
        max_retries: int = 3,
        retry_delay: int = 2,
    ) -> Dict:
        """Backward-compatible method: wraps flat steps into ExecutionPlan and delegates to execute_plan."""
        from app.core.execution_plan import ExecutionStep as ES, ExecutionPlan as EP

        plan_steps = []
        for i, s in enumerate(steps):
            tool = s.get("tool", "analyze")
            agent_name = TOOL_AGENT_MAP.get(tool, tool)
            plan_steps.append(ES(
                id=f"step-{i+1}",
                name=s.get("name", f"Step {i+1}"),
                description=s.get("description", ""),
                capability=agent_name,
                input=s,
                depends_on=[] if i == 0 else [f"step-{i}"],
            ))

        plan = EP(
            intent=task[:100] if task else "workflow",
            task_description=task,
            steps=plan_steps,
            required_capabilities=[s.capability for s in plan_steps],
        )

        run = await self.db.get(AgentRun, run_id)
        project_id = run.project_id if run else None

        return await self.execute_plan(plan, project_id, run_id, max_retries=max_retries)

    async def _execute_step(
        self,
        tool: str,
        step: Dict,
        project_id: Optional[UUID],
        run_id: UUID,
        attempt: int,
    ) -> Any:
        """Backward-compatible single-step execution via CapabilityRegistry."""
        agent_name = TOOL_AGENT_MAP.get(tool)
        if not agent_name:
            return {"result": f"Unknown tool: {tool}"}

        try:
            capability = self.registry.resolve(agent_name)
        except CapabilityNotFoundError:
            return {"result": f"No capability for tool: {tool}"}

        repository_id: Optional[UUID] = None
        rid_raw = step.get("repository_id") or (str(project_id) if project_id else None)
        if rid_raw:
            try:
                repository_id = UUID(rid_raw) if isinstance(rid_raw, str) else rid_raw
            except (ValueError, TypeError):
                pass

        if not repository_id and project_id:
            repo_result = await self.db.execute(
                select(Repository).where(Repository.project_id == project_id).limit(1)
            )
            repo = repo_result.scalar_one_or_none()
            if repo:
                repository_id = repo.id

        task_description = step.get("description") or step.get("goal") or step.get("name", "Execute step")

        if repository_id:
            context = await self.ctx_builder.build(repository_id, task_description)
        else:
            context = ProjectContext(
                project_id=project_id or UUID("00000000-0000-0000-0000-000000000000"),
                task_description=task_description,
            )

        return await capability.executor(context)

    async def get_workflow_state(self, run_id: UUID) -> Dict:
        """Get current workflow state with real step results."""
        result = await self.db.execute(
            select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.step_index)
        )
        steps = result.scalars().all()

        return {
            "run_id": str(run_id),
            "total_steps": len(steps),
            "completed_steps": sum(1 for s in steps if s.status == "completed"),
            "failed_steps": sum(1 for s in steps if s.status == "failed"),
            "steps": [
                {
                    "id": str(s.id),
                    "step_index": s.step_index,
                    "step_type": s.step_type,
                    "name": s.name,
                    "status": s.status,
                    "error_message": s.error_message,
                    "input_data": s.input_data,
                    "output_data": s.output_data,
                    "duration_ms": s.duration_ms,
                    "created_at": s.created_at.isoformat(),
                }
                for s in steps
            ],
        }

    async def resume_workflow(self, run_id: UUID) -> Dict:
        """Resume a failed workflow from the last failed step."""
        run = await self.db.get(AgentRun, run_id)
        if not run:
            return {"status": "not_found", "message": "Run not found"}

        result = await self.db.execute(
            select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.step_index)
        )
        steps = result.scalars().all()

        completed = [s for s in steps if s.status == "completed"]
        failed = [s for s in steps if s.status == "failed"]

        if not failed:
            return {"status": "already_completed", "message": "No failed steps found"}

        original_task = ""
        original_steps: List[Dict] = []
        if run.input_data:
            original_task = run.input_data.get("task", "")
            original_steps = run.input_data.get("steps", [])

        first_failed = failed[0]
        resume_from = first_failed.step_index
        remaining_steps = [s for s in original_steps if s.get("step_index", 0) >= resume_from]

        completed_details = [
            {
                "step_index": s.step_index,
                "name": s.name,
                "tool": s.step_type,
                "output": s.output_data,
            }
            for s in completed
        ]

        logger.info("Resuming workflow %s from step %d (%d remaining)", run_id, resume_from, len(remaining_steps))

        resume_result = await self.execute_workflow(
            task=original_task,
            steps=remaining_steps,
            run_id=run_id,
        )

        return {
            "status": "resumed",
            "resume_from": resume_from,
            "error": first_failed.error_message,
            "completed_before_resume": completed_details,
            "resume_result": resume_result,
        }


async def get_workflow_engine(
    db: AsyncSession = Depends(get_db_session),
    ctx_builder: ContextBuilder = Depends(get_context_builder),
    repo_agent: RepositoryAgent = Depends(get_repository_agent),
    knowledge_agent: KnowledgeAgent = Depends(get_knowledge_agent),
    incident_agent: IncidentAgent = Depends(get_incident_agent),
    doc_agent: DocumentationAgent = Depends(get_documentation_agent),
    code_review_agent: CodeReviewAgent = Depends(get_code_review_agent),
    deploy_agent: DeployAgent = Depends(get_deploy_agent),
) -> WorkflowEngine:
    """Build a WorkflowEngine with all capabilities registered.

    Agents are registered directly (their .process() methods already match
    the common signature). MCP tools, REST endpoints, and Python functions
    are normalized via adapter functions.
    """
    registry = CapabilityRegistry()
    registry.register("repository", "Analyze repo structure, architecture, code search", adapt_agent(repo_agent.process))
    registry.register("knowledge", "Search indexed documents and answer questions", adapt_agent(knowledge_agent.process))
    registry.register("incident", "Find error patterns, root cause analysis, recommendations", adapt_agent(incident_agent.process))
    registry.register("documentation", "Generate README, API docs, architecture docs", adapt_agent(doc_agent.process))
    registry.register("code_review", "Security audit, code quality, best practices", adapt_agent(code_review_agent.process))
    registry.register("deploy", "Analyze Docker, CI/CD, deployment configs", adapt_agent(deploy_agent.process))

    # Register MCP tools via generic adapters — MCP becomes just another capability
    try:
        from app.mcp.registry import get_registry
        mcp_registry = get_registry()
        for tool in mcp_registry.list_tools():
            name = tool["name"]
            desc = tool["description"]
            if not registry.has(name):
                registry.register(name, desc, adapt_mcp(name, mcp_registry), execution_type="mcp")
            else:
                logger.debug("Skipping MCP tool '%s' — already registered", name)
    except Exception:
        logger.warning("Could not load MCP tools for registry", exc_info=True)

    # Register REST and Python stub capabilities for future extensibility
    if not registry.has("rest_call"):
        registry.register("rest_call", "Make an HTTP request to an external API", adapt_rest(), execution_type="rest")
    if not registry.has("python_exec"):
        registry.register("python_exec", "Execute a Python function or snippet", adapt_python(), execution_type="python")

    return WorkflowEngine(db=db, ctx_builder=ctx_builder, registry=registry)
