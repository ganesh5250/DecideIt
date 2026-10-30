"""Bounded DecideIt orchestration engine and stable backend-facing interface."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import json

from .memory import ExecutionMemory
from .planner import Planner
from .prompt_builder import build_messages
from .llm import LLMClient, LLMError


@dataclass(frozen=True)
class AgentConfig:
    max_iterations: int = 12
    max_tool_calls: int = 20
    json_mode: bool = True

    def __post_init__(self):
        if not 1 <= self.max_iterations <= 50:
            raise ValueError("max_iterations must be between 1 and 50")
        if not 0 <= self.max_tool_calls <= 100:
            raise ValueError("max_tool_calls must be between 0 and 100")


@dataclass
class AgentResult:
    status: str
    output: dict[str, Any]
    plan: list[dict[str, Any]]
    history: list[dict[str, Any]]
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "output": self.output, "plan": self.plan,
                "history": self.history, "errors": self.errors}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class DecideItAgent:
    """Call evaluate() from a FastAPI service; auth/database remain backend concerns."""

    def __init__(self, llm: Any | None = None, *, config: AgentConfig | None = None,
                 tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None):
        self.llm = llm
        self.config = config or AgentConfig()
        self.planner = Planner()
        self.tools = tools or {}

    def evaluate(self, task: dict[str, Any]) -> AgentResult:
        self._validate_task(task)
        plan = self.planner.build(task)
        memory = ExecutionMemory()
        errors: list[dict[str, str]] = []
        context: dict[str, Any] = {"plan": self.planner.to_dict(plan), "tool_results": {}}

        # Deterministic steps are delegated to registered tools in later stages.
        # Unknown tools are recorded, never dynamically imported or executed.
        for step in plan[:self.config.max_iterations]:
            memory.record(action=step.action, status="planned")
            tool = self.tools.get(step.action)
            if tool is None:
                memory.record(action=step.action, status="not_configured")
                continue
            if len(context["tool_results"]) >= self.config.max_tool_calls:
                errors.append({"stage": step.action, "error": "Tool-call limit reached."})
                break
            if memory.has_repeated_action(step.action):
                errors.append({"stage": step.action, "error": "Repeated action blocked."})
                continue
            try:
                result = tool(task)
                if not isinstance(result, dict):
                    raise TypeError("Tool must return a dictionary.")
                context["tool_results"][step.action] = result
                memory.record(action=step.action, status="success")
            except Exception:
                errors.append({"stage": step.action, "error": "Tool execution failed."})
                memory.record(action=step.action, status="error")

        if self.llm is None:
            output = {
                "project_title": task.get("project_title"),
                "problem_analysis": {"status": "pending_llm_or_research",
                                     "problem_description": task["problem_description"]},
                "tool_results": context["tool_results"],
                "missing_information": ["Configure an LLM and the research/evaluation tools."],
            }
            status = "partial"
        else:
            try:
                messages = build_messages(task, context, json_mode=self.config.json_mode)
                raw = self.llm.generate(messages, json_mode=self.config.json_mode)
                parsed = json.loads(raw) if self.config.json_mode else {"analysis": raw}
                if not isinstance(parsed, dict):
                    raise ValueError("LLM output must be a JSON object.")
                output, status = parsed, "completed"
            except (LLMError, ValueError, TypeError):
                output = {"project_title": task.get("project_title"),
                          "error": "The analysis response could not be validated.",
                          "tool_results": context["tool_results"]}
                errors.append({"stage": "synthesis", "error": "LLM or output validation failed."})
                status = "partial"
        return AgentResult(status, output, self.planner.to_dict(plan),
                           memory.export(), errors)

    @staticmethod
    def _validate_task(task: dict[str, Any]) -> None:
        if not isinstance(task, dict):
            raise ValueError("Task must be a JSON object.")
        title = task.get("project_title", "")
        problem = task.get("problem_description", "")
        if not isinstance(title, str) or len(title) > 300:
            raise ValueError("project_title must be a string of at most 300 characters.")
        if not isinstance(problem, str) or not problem.strip() or len(problem) > 20000:
            raise ValueError("problem_description is required and must be at most 20,000 characters.")
        if task.get("proposed_solution") is not None and not isinstance(task["proposed_solution"], str):
            raise ValueError("proposed_solution must be a string or null.")
