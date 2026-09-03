import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import AgentRun, AgentStep
from app.services.llm_service import LLMService, get_llm_service
from app.services.memory import MemorySystem, get_memory_system
from app.services.agent_base import AgentResult
from app.services.context_builder import EngineeringContext, get_context_builder
from app.core.task import Task
from app.mcp.registry import ToolRegistry, get_registry
from app.core.di import get_db_session
from app.core.config import settings
from app.core.execution_plan import ExecutionPlan, ExecutionStep, RetryPolicy, RollbackStrategy

try:
    from langfuse import observe, get_client
except ImportError:
    observe = lambda *a, **kw: (lambda f: f)  # type: ignore
    get_client = lambda: None  # type: ignore


AGENT_NAMES = {
    "repository": "RepositoryAgent",
    "knowledge": "KnowledgeAgent",
    "incident": "IncidentAgent",
    "documentation": "DocumentationAgent",
    "code_review": "CodeReviewAgent",
    "deploy": "DeployAgent",
}


class PlannerAgent:
    """Planner performs reasoning only — produces ExecutionPlan, never executes."""

    def __init__(
        self,
        db: AsyncSession,
        llm: LLMService,
        memory: Optional[MemorySystem] = None,
        mcp: Optional[ToolRegistry] = None,
    ):
        self.db = db
        self.llm = llm
        self.memory = memory
        self.mcp = mcp

    async def _inject_memory_context(self, task: str) -> str:
        if not self.memory:
            return ""
        results = await self.memory.search_semantic(task, limit=3, threshold=settings.SIMILARITY_THRESHOLD)
        if not results:
            return ""
        parts = []
        for r in results:
            text = r.get("text", "")
            if text:
                parts.append(text[:500])
        return "\nRelated past context:\n" + "\n---\n".join(parts) if parts else ""

    @observe(name="planner.plan", as_type="span", capture_input=False, capture_output=False)
    async def plan(self, task: Task) -> ExecutionPlan:
        """Produce an ExecutionPlan by decomposing the task.
        
        No side effects — no DB writes, no agent calls, no LLM execution.
        Only reasoning.
        """
        try:
            lf = get_client()
            if lf:
                lf.update_current_span(input={"task": task.input, "project_id": str(getattr(task, 'project_id', ''))})
        except Exception:
            pass
        task_str = task.input
        memory_context = await self._inject_memory_context(task_str) if self.memory else ""
        system_prompt = f"Break this software engineering task into 2-4 steps. For each step, describe what needs to be done. Format as JSON: [{{'step': 1, 'name': '...', 'description': '...', 'capability': 'repository|knowledge|incident|documentation|code_review|deploy'}}]{memory_context}"
        result = await self.llm.generate(system_prompt, task_str)

        try:
            parsed = json.loads(result)
            steps_data = parsed if isinstance(parsed, list) else [{"step": 1, "name": "Execute", "description": task_str, "capability": "repository"}]
        except json.JSONDecodeError:
            steps_data = [{"step": 1, "name": "Execute", "description": task_str, "capability": "repository"}]

        steps = []
        capabilities = set()
        for i, s in enumerate(steps_data):
            cap = s.get("capability", "repository")
            capabilities.add(cap)
            steps.append(ExecutionStep(
                id=f"step-{i+1}",
                name=s.get("name", f"Step {i+1}"),
                description=s.get("description", ""),
                capability=cap,
                input={"task": task_str, "step_description": s.get("description", "")},
                depends_on=[] if i == 0 else [f"step-{i}"],
                retry_policy=RetryPolicy(max_retries=3, retry_delay_seconds=2.0),
            ))

        return ExecutionPlan(
            intent=steps_data[0].get("description", task_str)[:100] if steps_data else task_str[:100],
            task_description=task_str,
            steps=steps,
            required_capabilities=list(capabilities),
        )

    async def plan_and_execute(self, task: Task) -> Dict:
        """Backward-compatible method: produces a plan and executes via LLM.
        
        Keeps the original return shape: {run_id, status, steps, result}.
        Internally uses plan() then delegates execution to LLM per step.
        """
        plan = await self.plan(task)
        task_str = task.input
        project_id = task.project_id

        run = AgentRun(
            project_id=project_id, agent_type="planner", status="running",
            input_data={"task": task_str, "project_id": str(project_id), "plan_id": plan.plan_id},
        )
        self.db.add(run)
        await self.db.commit()
        await self.db.refresh(run)

        for step_def in plan.steps:
            step_prompt = f"Execute step: {step_def.description}\nTask context: {task_str}"
            step_result = await self.llm.generate(step_prompt, "Execute this step")

            step = AgentStep(
                run_id=run.id, step_index=int(step_def.id.split("-")[1]), step_type="llm_call",
                name=step_def.name,
                input_data={"task": task_str, "step_description": step_def.description},
                output_data={"result": step_result}, status="completed",
            )
            self.db.add(step)

        run.status = "completed"
        run.completed_at = datetime.utcnow()
        run.output_data = {"total_steps": len(plan.steps), "result": "Task completed via LLM planning and execution"}
        await self.db.commit()
        await self.db.refresh(run)

        if self.memory:
            try:
                emb = (await self.memory.embeddings.generate_embeddings([task_str]))[0]
                if emb:
                    await self.memory.store_semantic(
                        text=f"Task: {task_str}\nPlan: {json.dumps([s.__dict__ for s in plan.steps])}",
                        embedding=emb,
                        metadata={"type": "planner_plan", "run_id": str(run.id)},
                    )
            except Exception:
                pass

        return {
            "run_id": str(run.id),
            "status": run.status,
            "steps": len(plan.steps),
            "result": "completed",
        }


class IntentRouter:
    """Classifies user input into an ExecutionPlan.
    
    Uses keyword-based routing as the primary method (reliable, fast).
    Falls back to LLM classification for ambiguous queries.
    Returns a structured ExecutionPlan dataclass with DAG steps.
    """

    def __init__(self, llm: LLMService):
        self.llm = llm

    AGENT_DESCRIPTIONS = {
        "repository": "Analyze repo structure, architecture, code search",
        "knowledge": "Search indexed documents and answer questions",
        "incident": "Find error patterns, root cause analysis, recommendations",
        "documentation": "Generate README, API docs, architecture docs",
        "code_review": "Security audit, code quality, best practices",
        "deploy": "Analyze Docker, CI/CD, deployment configs",
    }

    KEYWORD_RULES = [
        {
            "keywords": ["architecture", "structure", "codebase", "code structure", "how is", "project layout", "tech stack", "design pattern", "folder", "module"],
            "agents": ["repository"],
            "intent": "Understand codebase architecture and structure",
            "approval": False,
        },
        {
            "keywords": ["search", "find", "look for", "where is", "locate", "query"],
            "agents": ["knowledge", "repository"],
            "intent": "Search codebase and documents for information",
            "approval": False,
        },
        {
            "keywords": ["error", "bug", "issue", "crash", "exception", "fail", "broken", "problem", "incident", "root cause"],
            "agents": ["incident"],
            "intent": "Analyze errors and find root cause",
            "approval": False,
        },
        {
            "keywords": ["readme", "documentation", "docs", "api doc", "generate doc", "document"],
            "agents": ["documentation"],
            "intent": "Generate or review documentation",
            "approval": False,
        },
        {
            "keywords": ["security", "vulnerability", "audit", "review code", "code review", "best practice", "quality"],
            "agents": ["code_review"],
            "intent": "Perform code review and security audit",
            "approval": False,
        },
        {
            "keywords": ["deploy", "docker", "ci/cd", "pipeline", "kubernetes", "k8s", "infrastructure", "nginx", "ci config"],
            "agents": ["deploy"],
            "intent": "Analyze deployment and infrastructure configuration",
            "approval": False,
        },
        {
            "keywords": ["what is", "explain", "describe", "tell me", "overview", "summary"],
            "agents": ["repository", "knowledge"],
            "intent": "Provide an overview or explanation of the project",
            "approval": False,
        },
    ]

    def _agents_to_steps(self, agent_names: List[str], user_input: str, approval: bool = False) -> List[ExecutionStep]:
        """Convert agent name list to ExecutionStep DAG.
        
        First step has no dependencies.
        Subsequent steps depend on the previous step.
        """
        steps = []
        for i, name in enumerate(agent_names):
            depends = [] if i == 0 else [f"step-{i}"]
            steps.append(ExecutionStep(
                id=f"step-{i+1}",
                name=name,
                capability=name,
                input={"task": user_input},
                depends_on=depends,
                requires_approval=approval and name == "deploy",
                retry_policy=RetryPolicy(max_retries=3, retry_delay_seconds=2.0),
            ))
        return steps

    def _keyword_route(self, user_input: str) -> ExecutionPlan:
        """Route based on keyword matching (reliable, no LLM needed)."""
        lower = user_input.lower()
        for rule in self.KEYWORD_RULES:
            if any(kw in lower for kw in rule["keywords"]):
                return ExecutionPlan(
                    intent=rule["intent"],
                    task_description=user_input,
                    steps=self._agents_to_steps(rule["agents"], user_input, rule.get("approval", False)),
                    required_capabilities=rule["agents"],
                )
        return ExecutionPlan(
            intent="general_question",
            task_description=user_input,
            steps=self._agents_to_steps(["repository", "knowledge"], user_input),
            required_capabilities=["repository", "knowledge"],
        )

    @observe(name="intent-router.route", as_type="span", capture_input=False, capture_output=False)
    async def route(self, ec: EngineeringContext) -> ExecutionPlan:
        """Classify user input and produce an ExecutionPlan."""
        try:
            lf = get_client()
            if lf:
                lf.update_current_span(input={"user_input": ec.task.input, "project_id": str(getattr(ec.task, 'project_id', ''))})
        except Exception:
            pass
        user_input = ec.task.input
        lower = user_input.lower()

        # Check for destructive/deployment actions first
        destructive_keywords = ["delete", "remove", "destroy", "drop", "truncate", "deploy to production", "push to main"]
        if any(kw in lower for kw in destructive_keywords):
            step1 = ExecutionStep(
                id="step-1",
                name="deploy",
                capability="deploy",
                input={"task": user_input},
                requires_approval=True,
                rollback_step="step-2",
                retry_policy=RetryPolicy(max_retries=3, retry_delay_seconds=2.0),
            )
            step2 = ExecutionStep(
                id="step-2",
                name="rollback",
                capability="deploy",
                input={"action": "rollback"},
                depends_on=["step-1"],
                retry_policy=RetryPolicy(max_retries=3, retry_delay_seconds=2.0),
            )
            return ExecutionPlan(
                intent="destructive_action",
                task_description=user_input,
                steps=[step1, step2],
                required_capabilities=["deploy"],
                approvals_required=["step-1"],
                rollback_strategy=RollbackStrategy(steps=["step-2"], automatic=True),
            )

        # Primary: keyword-based routing (fast, reliable)
        keyword_plan = self._keyword_route(user_input)

        # Secondary: try LLM for better classification (best effort)
        try:
            conversation_context = ""
            if ec.memory.conversation_history:
                conv_lines = [
                    f"{m['role']}: {m['content']}"
                    for m in ec.memory.conversation_history
                ]
                conversation_context = "Recent conversation:\n" + "\n".join(conv_lines[-3:]) + "\n\n"
            agent_list = "\n".join(f"  - {k}: {v}" for k, v in self.AGENT_DESCRIPTIONS.items())
            prompt = f"""Given the user request below, decide which agents should handle it.

Available agents:
{agent_list}

Return ONLY valid JSON with these keys:
- intent (str): one-sentence summary
- required_agents (list of str): agent names that should run
- needs_approval (bool): true only for destructive actions

{conversation_context}User request: {user_input}"""

            result = await self.llm.generate(
                "Return ONLY valid JSON. No explanation.",
                prompt,
            )
            plan = json.loads(result)
            required = plan.get("required_agents", [])
            if required and all(a in self.AGENT_DESCRIPTIONS for a in required):
                routed = ExecutionPlan(
                    intent=plan.get("intent", keyword_plan.intent),
                    task_description=user_input,
                    steps=self._agents_to_steps(required, user_input, plan.get("needs_approval", False)),
                    required_capabilities=required,
                    approvals_required=["step-1"] if plan.get("needs_approval") else [],
                )
                try:
                    lf = get_client()
                    if lf:
                        lf.update_current_span(output={"intent": routed.intent, "required_capabilities": routed.required_capabilities, "via": "llm"})
                except Exception:
                    pass
                return routed
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

        # Fallback to keyword routing
        try:
            lf = get_client()
            if lf:
                lf.update_current_span(output={"intent": keyword_plan.intent, "required_capabilities": keyword_plan.required_capabilities, "via": "keyword"})
        except Exception:
            pass
        return keyword_plan


class ResponseMerger:
    """Combines multiple AgentResult dicts into one coherent response."""

    def __init__(self, llm: LLMService):
        self.llm = llm

    async def merge(
        self,
        agent_results: Dict[str, AgentResult],
        execution_plan: Any,
    ) -> str:
        if not agent_results:
            return "No agent results to merge."

        task_desc = ""
        if hasattr(execution_plan, "task_description"):
            task_desc = execution_plan.task_description
        elif isinstance(execution_plan, dict):
            task_desc = execution_plan.get("task_description", "")

        if len(agent_results) == 1:
            name, result = next(iter(agent_results.items()))
            text = result.get("result", "")
            recs = result.get("recommendations", [])
            footer = "\n\n**Recommendations:**\n" + "\n".join(f"- {r}" for r in recs) if recs else ""
            return text + footer

        parts = []
        for agent_name, result in agent_results.items():
            display_name = AGENT_NAMES.get(agent_name, agent_name)
            text = result.get("result", "")
            confidence = result.get("confidence", 0.5)
            recs = result.get("recommendations", [])
            parts.append(f"## {display_name} (confidence: {confidence:.0%})\n\n{text}")
            if recs:
                parts.append("**Recommendations:**\n" + "\n".join(f"- {r}" for r in recs))

        combined = "\n\n---\n\n".join(parts)

        system_prompt = "Merge these agent outputs into a single coherent response. Remove redundancy, preserve unique insights from each agent."
        merged = await self.llm.generate(system_prompt, f"Task: {task_desc}\n\n{combined[:6000]}")
        return merged


class PlanValidator:
    """Validates an ExecutionPlan before execution.
    
    Checks:
    - No cycles in dependency graph
    - All step IDs referenced in depends_on exist
    - All capabilities are registered
    - Rollback step IDs exist
    - At least one step
    """

    def validate(self, plan: ExecutionPlan, available_capabilities: List[str] = None) -> List[str]:
        errors = []

        if not plan.steps:
            errors.append("ExecutionPlan has no steps")
            return errors

        step_ids = {s.id for s in plan.steps}

        # Check all step IDs are unique
        if len(step_ids) != len(plan.steps):
            ids = [s.id for s in plan.steps]
            duplicates = [id for id in ids if ids.count(id) > 1]
            errors.append(f"Duplicate step IDs: {set(duplicates)}")

        # Check dependency references
        for s in plan.steps:
            for dep in s.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step '{s.id}' depends on '{dep}' which does not exist")

        # Check rollback references
        for s in plan.steps:
            if s.rollback_step and s.rollback_step not in step_ids:
                errors.append(f"Step '{s.id}' references rollback step '{s.rollback_step}' which does not exist")

        # Check for cycles (simple DFS)
        visited = set()
        in_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            in_stack.add(node_id)
            step = next((s for s in plan.steps if s.id == node_id), None)
            if step:
                for dep in step.depends_on:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in in_stack:
                        return True
            in_stack.discard(node_id)
            return False

        for s in plan.steps:
            if s.id not in visited:
                if has_cycle(s.id):
                    errors.append("ExecutionPlan contains a cycle in the dependency graph")
                    break

        # Check capabilities
        if available_capabilities:
            for s in plan.steps:
                if s.capability and s.capability not in available_capabilities:
                    errors.append(f"Step '{s.id}' requires capability '{s.capability}' which is not available")

        # Check approval step IDs exist
        for aid in plan.approvals_required:
            if aid not in step_ids:
                errors.append(f"Approval required for step '{aid}' which does not exist")

        return errors


async def get_planner(
    db: AsyncSession = Depends(get_db_session),
    llm: LLMService = Depends(get_llm_service),
    memory: Optional[MemorySystem] = Depends(get_memory_system),
) -> PlannerAgent:
    return PlannerAgent(db, llm, memory=memory, mcp=get_registry())


async def get_intent_router(
    llm: LLMService = Depends(get_llm_service),
) -> IntentRouter:
    return IntentRouter(llm)


async def get_response_merger(
    llm: LLMService = Depends(get_llm_service),
) -> ResponseMerger:
    return ResponseMerger(llm)


async def get_plan_validator() -> PlanValidator:
    return PlanValidator()
