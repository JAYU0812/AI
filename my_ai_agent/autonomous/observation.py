"""Observation model for Step 3D Autonomous Engine.

Captures the factual outcome of an executed action and allows the agent
to reason about what occurred before choosing the next action.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
from my_ai_agent.autonomous.action import ActionType


@dataclass
class ActionObservation:
    """Factual outcome and evaluation of an autonomous action."""

    action_id: str
    action_type: ActionType
    success: bool
    output: str = ""
    error: str = ""
    exit_code: Optional[int] = None
    modified_files: List[str] = field(default_factory=list)
    diffs: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Optional[Dict[str, Any]] = None
    evaluation: str = ""  # The agent's structured reasoning on the outcome
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value if isinstance(self.action_type, ActionType) else str(self.action_type),
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "modified_files": self.modified_files,
            "diffs": self.diffs,
            "diagnostics": self.diagnostics,
            "evaluation": self.evaluation,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionObservation":
        raw_type = data.get("action_type", "READ_FILE")
        try:
            a_type = ActionType(raw_type)
        except ValueError:
            a_type = ActionType.READ_FILE

        return cls(
            action_id=data.get("action_id", ""),
            action_type=a_type,
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error", ""),
            exit_code=data.get("exit_code"),
            modified_files=data.get("modified_files", []),
            diffs=data.get("diffs", []),
            diagnostics=data.get("diagnostics"),
            evaluation=data.get("evaluation", ""),
            timestamp=data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
        )
