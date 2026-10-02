"""Autonomous Task Model for Step 3D.

Encapsulates complete task lifecycle, state machine, execution budget,
action plan, observation logs, and verifiable outcomes.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set
import uuid

from my_ai_agent.autonomous.action import AutonomousAction
from my_ai_agent.autonomous.budget import TaskBudget
from my_ai_agent.autonomous.observation import ActionObservation
from my_ai_agent.autonomous.task_state import TaskStateMachine, TaskStatus


@dataclass
class AutonomousTask:
    """Explicit autonomous task tracking unit."""

    user_request: str
    project_id: str = "default"
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    state_machine: TaskStateMachine = field(default_factory=TaskStateMachine)
    budget: TaskBudget = field(default_factory=TaskBudget)

    plan: List[AutonomousAction] = field(default_factory=list)
    completed_actions: List[AutonomousAction] = field(default_factory=list)
    observations: List[ActionObservation] = field(default_factory=list)
    pending_approval_action: Optional[AutonomousAction] = None

    modified_files: Set[str] = field(default_factory=set)
    attempt: int = 1
    max_attempts: int = 5
    final_summary: str = ""

    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    ended_at: Optional[str] = None

    @property
    def status(self) -> TaskStatus:
        return self.state_machine.current_state

    def transition_to(self, new_state: TaskStatus, reason: str = "") -> TaskStatus:
        res = self.state_machine.transition_to(new_state, reason=reason)
        if self.is_terminal() and not self.ended_at:
            self.ended_at = time.strftime("%Y-%m-%d %H:%M:%S")
        return res

    def is_terminal(self) -> bool:
        return self.state_machine.is_terminal()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "user_request": self.user_request,
            "status": self.status.value,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "plan": [a.to_dict() for a in self.plan],
            "completed_actions": [a.to_dict() for a in self.completed_actions],
            "observations_count": len(self.observations),
            "pending_approval": self.pending_approval_action.to_dict() if self.pending_approval_action else None,
            "modified_files": sorted(list(self.modified_files)),
            "budget": self.budget.to_dict(),
            "final_summary": self.final_summary,
            "state_history": self.state_machine.history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutonomousTask":
        task = cls(
            user_request=data.get("user_request", ""),
            project_id=data.get("project_id", "default"),
            task_id=data.get("task_id", str(uuid.uuid4())[:12]),
            attempt=data.get("attempt", 1),
            max_attempts=data.get("max_attempts", 5),
            started_at=data.get("started_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            ended_at=data.get("ended_at"),
            final_summary=data.get("final_summary", ""),
            modified_files=set(data.get("modified_files", [])),
        )
        # Restore status
        raw_status = data.get("status", "QUEUED")
        try:
            target_status = TaskStatus(raw_status)
            task.state_machine._current_state = target_status
        except ValueError:
            pass

        # Restore plan and actions
        task.plan = [AutonomousAction.from_dict(a) for a in data.get("plan", [])]
        task.completed_actions = [AutonomousAction.from_dict(a) for a in data.get("completed_actions", [])]
        if data.get("pending_approval"):
            task.pending_approval_action = AutonomousAction.from_dict(data["pending_approval"])

        return task
