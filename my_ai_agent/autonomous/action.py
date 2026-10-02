"""Autonomous Action definitions for Step 3D.

Explicit, structured actions with granular risk assessment:
- LOW: read file, search code, create test, add function, run tests
- MEDIUM: modify configuration, install dependency, database model change
- HIGH: delete file, database migration, destructive command, system change
"""

from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import Any, Dict, Optional


class ActionType(str, Enum):
    INSPECT_PROJECT = "INSPECT_PROJECT"
    READ_FILE = "READ_FILE"
    SEARCH_CODE = "SEARCH_CODE"
    CREATE_FILE = "CREATE_FILE"
    PATCH_FILE = "PATCH_FILE"
    DELETE_FILE = "DELETE_FILE"
    RUN_COMMAND = "RUN_COMMAND"
    RUN_TESTS = "RUN_TESTS"
    INSTALL_DEPENDENCY = "INSTALL_DEPENDENCY"
    VERIFY = "VERIFY"
    ASK_USER = "ASK_USER"
    FINISH = "FINISH"


class ActionRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class AutonomousAction:
    """Represents a discrete, reasoned action planned by the agent."""

    type: ActionType
    target: str
    reason: str
    risk: ActionRisk = ActionRisk.LOW
    params: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    requires_approval: bool = False
    approved: bool = False
    status: str = "PENDING"  # PENDING, APPROVED, REJECTED, EXECUTING, COMPLETED, FAILED

    def __post_init__(self):
        # Automatically mark high-risk actions as requiring user approval
        if self.risk == ActionRisk.HIGH and not self.requires_approval:
            self.requires_approval = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, ActionType) else str(self.type),
            "target": self.target,
            "reason": self.reason,
            "risk": self.risk.value if isinstance(self.risk, ActionRisk) else str(self.risk),
            "params": self.params,
            "requires_approval": self.requires_approval,
            "approved": self.approved,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutonomousAction":
        raw_type = data.get("type", "READ_FILE")
        try:
            a_type = ActionType(raw_type)
        except ValueError:
            a_type = ActionType.READ_FILE

        raw_risk = data.get("risk", "LOW")
        try:
            a_risk = ActionRisk(raw_risk)
        except ValueError:
            a_risk = ActionRisk.LOW

        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            type=a_type,
            target=data.get("target", ""),
            reason=data.get("reason", ""),
            risk=a_risk,
            params=data.get("params", {}),
            requires_approval=data.get("requires_approval", False),
            approved=data.get("approved", False),
            status=data.get("status", "PENDING"),
        )
