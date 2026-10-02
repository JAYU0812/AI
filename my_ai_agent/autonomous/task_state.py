"""Task State Machine for Step 3D Autonomous Engine.

Defines all formal task lifecycle statuses and guarantees valid state transitions.
Never simulates or fakes statuses; each state corresponds to actual execution.
"""

from enum import Enum
from typing import Dict, List, Optional, Set


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    EXECUTING = "EXECUTING"
    TESTING = "TESTING"
    DIAGNOSING = "DIAGNOSING"
    REPAIRING = "REPAIRING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Valid state transitions graph
VALID_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.QUEUED: {
        TaskStatus.UNDERSTANDING,
        TaskStatus.CANCELLED,
    },
    TaskStatus.UNDERSTANDING: {
        TaskStatus.PLANNING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.PLANNING: {
        TaskStatus.EXECUTING,
        TaskStatus.WAITING_FOR_APPROVAL,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.WAITING_FOR_APPROVAL: {
        TaskStatus.EXECUTING,
        TaskStatus.PLANNING,
        TaskStatus.CANCELLED,
        TaskStatus.FAILED,
    },
    TaskStatus.EXECUTING: {
        TaskStatus.EXECUTING,
        TaskStatus.TESTING,
        TaskStatus.DIAGNOSING,
        TaskStatus.WAITING_FOR_APPROVAL,
        TaskStatus.VERIFYING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.TESTING: {
        TaskStatus.VERIFYING,
        TaskStatus.DIAGNOSING,
        TaskStatus.EXECUTING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.DIAGNOSING: {
        TaskStatus.REPAIRING,
        TaskStatus.PLANNING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.REPAIRING: {
        TaskStatus.TESTING,
        TaskStatus.EXECUTING,
        TaskStatus.DIAGNOSING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.VERIFYING: {
        TaskStatus.COMPLETED,
        TaskStatus.PARTIALLY_COMPLETED,
        TaskStatus.DIAGNOSING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.COMPLETED: set(),
    TaskStatus.PARTIALLY_COMPLETED: {
        TaskStatus.PLANNING,  # Allow user continuation
    },
    TaskStatus.FAILED: {
        TaskStatus.PLANNING,  # Allow retry
        TaskStatus.QUEUED,
    },
    TaskStatus.CANCELLED: {
        TaskStatus.QUEUED,    # Allow restart
    },
}


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class TaskStateMachine:
    """Manages state transitions and transition history for an autonomous task."""

    def __init__(self, initial_state: TaskStatus = TaskStatus.QUEUED):
        self._current_state: TaskStatus = initial_state
        self._history: List[Dict[str, str]] = []
        self._record_transition(None, initial_state, "Task initialized")

    @property
    def current_state(self) -> TaskStatus:
        return self._current_state

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def transition_to(self, new_state: TaskStatus, reason: str = "") -> TaskStatus:
        """Transitions to new_state if valid according to transition rules."""
        if new_state == self._current_state:
            return self._current_state

        valid_targets = VALID_TRANSITIONS.get(self._current_state, set())
        if new_state not in valid_targets:
            raise StateTransitionError(
                f"Invalid transition: cannot move from {self._current_state.value} to {new_state.value}. "
                f"Allowed targets: {[s.value for s in valid_targets]}"
            )

        old_state = self._current_state
        self._current_state = new_state
        self._record_transition(old_state, new_state, reason)
        return self._current_state

    def is_terminal(self) -> bool:
        """Returns True if current state is final (cannot transition further without explicit restart)."""
        return self._current_state in (
            TaskStatus.COMPLETED,
            TaskStatus.PARTIALLY_COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    def _record_transition(self, from_state: Optional[TaskStatus], to_state: TaskStatus, reason: str):
        import time
        self._history.append({
            "from_state": from_state.value if from_state else "NONE",
            "to_state": to_state.value,
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
