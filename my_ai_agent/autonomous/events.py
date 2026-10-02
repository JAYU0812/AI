"""Real-time Agent Events and Streaming Broadcaster for Step 3D.

Every event corresponds to a factual execution action.
Never generates artificial progress or dummy animation events.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional


class AgentEventType(str, Enum):
    TASK_STARTED = "TASK_STARTED"
    UNDERSTANDING_PROJECT = "UNDERSTANDING_PROJECT"
    PROJECT_ANALYZED = "PROJECT_ANALYZED"
    PLAN_CREATED = "PLAN_CREATED"
    ACTION_STARTED = "ACTION_STARTED"
    ACTION_COMPLETED = "ACTION_COMPLETED"
    COMMAND_STARTED = "COMMAND_STARTED"
    COMMAND_COMPLETED = "COMMAND_COMPLETED"
    TEST_STARTED = "TEST_STARTED"
    TEST_PASSED = "TEST_PASSED"
    TEST_FAILED = "TEST_FAILED"
    ERROR_DIAGNOSED = "ERROR_DIAGNOSED"
    REPAIR_STARTED = "REPAIR_STARTED"
    PATCH_APPLIED = "PATCH_APPLIED"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_GRANTED = "APPROVAL_GRANTED"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    RETRYING = "RETRYING"
    VERIFICATION_STARTED = "VERIFICATION_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_PARTIALLY_COMPLETED = "TASK_PARTIALLY_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_CANCELLED = "TASK_CANCELLED"


@dataclass
class AgentEvent:
    """A factual milestone event emitted during autonomous execution."""

    task_id: str
    event_type: AgentEventType
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "event_type": self.event_type.value if isinstance(self.event_type, AgentEventType) else str(self.event_type),
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event string."""
        return f"event: {self.event_type.value}\ndata: {json.dumps(self.to_dict())}\n\n"


class EventBroadcaster:
    """Thread-safe event dispatcher and pub/sub broadcaster for SSE and callbacks."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._subscribers = defaultdict(list)  # task_id -> list of Queue
                cls._global_listeners: List[Callable[[AgentEvent], None]] = []
                cls._event_history = defaultdict(list)  # task_id -> list of AgentEvent
            return cls._instance

    def subscribe(self, task_id: str) -> "queue.Queue[AgentEvent]":
        """Subscribe to real-time events for a task (used by SSE)."""
        q: queue.Queue[AgentEvent] = queue.Queue()
        with self._lock:
            # Replay historical events for this task so client doesn't miss early events
            for ev in self._event_history.get(task_id, []):
                q.put(ev)
            self._subscribers[task_id].append(q)
        return q

    def unsubscribe(self, task_id: str, q: "queue.Queue[AgentEvent]"):
        """Remove a subscriber queue."""
        with self._lock:
            if task_id in self._subscribers and q in self._subscribers[task_id]:
                self._subscribers[task_id].remove(q)

    def add_listener(self, listener: Callable[[AgentEvent], None]):
        """Register a global event listener callback."""
        with self._lock:
            self._global_listeners.append(listener)

    def emit(self, event: AgentEvent):
        """Emit an event to all subscribers and global listeners."""
        with self._lock:
            self._event_history[event.task_id].append(event)
            # Push to task subscriber queues
            subscribers = list(self._subscribers.get(event.task_id, []))
            listeners = list(self._global_listeners)

        for q in subscribers:
            try:
                q.put_nowait(event)
            except Exception:
                pass

        for fn in listeners:
            try:
                fn(event)
            except Exception:
                pass

    def get_events(self, task_id: str) -> List[AgentEvent]:
        """Retrieve stored event trajectory for a task."""
        with self._lock:
            return list(self._event_history.get(task_id, []))

    def clear(self):
        """Clear all queues and history (useful for testing)."""
        with self._lock:
            self._subscribers.clear()
            self._event_history.clear()
            self._global_listeners.clear()


# Global singleton instance
broadcaster = EventBroadcaster()
