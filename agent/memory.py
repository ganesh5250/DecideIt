import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


SECRET_PATTERNS = [
    re.compile(r"(AIza[0-9A-Za-z\-_]{35})"),
    re.compile(r"(gsk_[0-9A-Za-z]{40,})"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*['\"]?[^\s'\"]+['\"]?)"),
]


def scrub_secrets(value: Any) -> Any:
    """Recursively scrub potential API keys or secrets from logged payloads."""
    if isinstance(value, str):
        cleaned = value
        for pattern in SECRET_PATTERNS:
            cleaned = pattern.sub("***REDACTED***", cleaned)
        return cleaned
    if isinstance(value, dict):
        return {
            k: ("***REDACTED***" if "key" in str(k).lower() or "secret" in str(k).lower() else scrub_secrets(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [scrub_secrets(item) for item in value]
    return value


class ExecutionRecord(BaseModel):
    """Single step or tool call record in the agent's execution history."""

    step_number: int
    stage: str
    action_type: str  # "plan", "tool_call", "evaluator", "llm_synthesis", "error"
    name: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    status: str  # "success", "error", "skipped", "blocked"
    output_summary: Optional[Any] = None
    error_message: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ExecutionMemory:
    """Tracks run-local execution history, enforces bounds, and detects repeated actions."""

    def __init__(self, max_iterations: int = 12, max_tool_calls: int = 20) -> None:
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.iterations_count: int = 0
        self.tool_calls_count: int = 0
        self.records: List[ExecutionRecord] = []
        self._action_signatures: Set[str] = set()
        self.tool_results: Dict[str, Any] = {}

    @staticmethod
    def _compute_signature(action_name: str, inputs: Dict[str, Any]) -> str:
        canonical = json.dumps(
            {"action": action_name, "inputs": inputs},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_repeated_action(self, action_name: str, inputs: Dict[str, Any]) -> bool:
        """Return True if the exact same action and input combination was already executed."""
        sig = self._compute_signature(action_name, inputs)
        return sig in self._action_signatures

    def check_iteration_limit(self) -> bool:
        """Return True if within allowed iteration bounds."""
        return self.iterations_count < self.max_iterations

    def increment_iteration(self) -> None:
        """Increment reasoning iteration counter or raise RuntimeError if limit reached."""
        if not self.check_iteration_limit():
            raise RuntimeError(
                f"Maximum reasoning iterations ({self.max_iterations}) exceeded."
            )
        self.iterations_count += 1

    def check_tool_call_limit(self) -> bool:
        """Return True if within allowed tool call bounds."""
        return self.tool_calls_count < self.max_tool_calls

    def record_action(
        self,
        stage: str,
        action_type: str,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        status: str = "success",
        output_summary: Optional[Any] = None,
        error_message: Optional[str] = None,
        register_signature: bool = True,
    ) -> ExecutionRecord:
        """Log an execution action safely with secret scrubbing."""
        safe_inputs = scrub_secrets(inputs or {})
        safe_output = scrub_secrets(output_summary)
        safe_error = scrub_secrets(error_message) if error_message else None

        if action_type == "tool_call" and status not in {"skipped", "blocked"}:
            self.tool_calls_count += 1

        if register_signature and status == "success":
            sig = self._compute_signature(name, safe_inputs)
            self._action_signatures.add(sig)

        record = ExecutionRecord(
            step_number=len(self.records) + 1,
            stage=stage,
            action_type=action_type,
            name=name,
            inputs=safe_inputs,
            status=status,
            output_summary=safe_output,
            error_message=safe_error,
        )
        self.records.append(record)
        return record

    def store_tool_result(self, key: str, result: Any) -> None:
        """Save structured tool output for downstream evaluators and report generation."""
        self.tool_results[key] = scrub_secrets(result)

    def to_history_list(self) -> List[Dict[str, Any]]:
        """Serialize execution records for the final JSON output."""
        return [r.model_dump() for r in self.records]