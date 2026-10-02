"""Task Budget for Step 3D Autonomous Engine.

Enforces configurable operational ceilings on:
- MAX_STEPS (default: 30)
- MAX_REPAIR_ATTEMPTS (default: 5)
- MAX_COMMANDS (default: 20)
- MAX_FILES_CHANGED (default: 50)
- MAX_EXECUTION_TIME (default: 300.0s)

Prevents runaway iterations or resource exhaustion.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, Optional, Set


@dataclass
class TaskBudget:
    """Configurable execution budget and real-time usage meters."""

    max_steps: int = 30
    max_repair_attempts: int = 5
    max_commands: int = 20
    max_files_changed: int = 50
    max_execution_time_seconds: float = 300.0

    # Current usage meters
    steps_taken: int = 0
    repair_attempts: int = 0
    commands_run: int = 0
    files_changed: Set[str] = field(default_factory=set)
    start_time: float = field(default_factory=time.time)

    def record_step(self):
        """Increment execution step counter."""
        self.steps_taken += 1

    def record_repair(self):
        """Increment auto-repair attempt counter."""
        self.repair_attempts += 1

    def record_command(self):
        """Increment terminal command execution counter."""
        self.commands_run += 1

    def record_file_changed(self, file_path: str):
        """Record a file modification."""
        if file_path:
            self.files_changed.add(str(file_path))

    @property
    def elapsed_time(self) -> float:
        """Elapsed execution time in seconds."""
        return time.time() - self.start_time

    def check_budget(self) -> Optional[str]:
        """Returns exhaustion reason if any budget limit is exceeded, else None."""
        if self.steps_taken >= self.max_steps:
            return f"Exceeded maximum allowed steps ({self.steps_taken}/{self.max_steps})"

        if self.repair_attempts >= self.max_repair_attempts:
            return f"Exceeded maximum auto-repair attempts ({self.repair_attempts}/{self.max_repair_attempts})"

        if self.commands_run >= self.max_commands:
            return f"Exceeded maximum allowed command executions ({self.commands_run}/{self.max_commands})"

        if len(self.files_changed) >= self.max_files_changed:
            return f"Exceeded maximum number of files modified ({len(self.files_changed)}/{self.max_files_changed})"

        if self.elapsed_time >= self.max_execution_time_seconds:
            return f"Exceeded maximum execution time ({self.elapsed_time:.1f}s/{self.max_execution_time_seconds:.1f}s)"

        return None

    def to_dict(self) -> Dict:
        return {
            "max_steps": self.max_steps,
            "max_repair_attempts": self.max_repair_attempts,
            "max_commands": self.max_commands,
            "max_files_changed": self.max_files_changed,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "steps_taken": self.steps_taken,
            "repair_attempts": self.repair_attempts,
            "commands_run": self.commands_run,
            "files_changed_count": len(self.files_changed),
            "files_changed": sorted(list(self.files_changed)),
            "elapsed_seconds": round(self.elapsed_time, 2),
        }
