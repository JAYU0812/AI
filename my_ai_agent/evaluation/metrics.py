"""Factual Benchmark Metrics.

Tracks real latencies, token usages, resource footprints, and action counts.
Strictly returns null for unmeasured attributes rather than fabricated numbers.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, Optional


@dataclass
class TaskMetrics:
    """Factual execution metrics collected during an evaluation run."""
    model_load_latency_seconds: Optional[float] = None
    inference_latency_seconds: Optional[float] = None
    task_completion_latency_seconds: Optional[float] = None
    streaming_latency_seconds: Optional[float] = None
    
    model_calls: int = 0
    tool_calls: int = 0
    repair_attempts: int = 0
    files_modified: int = 0
    tests_executed: int = 0
    
    ram_usage_mb: Optional[float] = None
    context_size_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes metrics with explicit None values preserved as null."""
        return {
            "model_load_latency_seconds": self.model_load_latency_seconds,
            "inference_latency_seconds": self.inference_latency_seconds,
            "task_completion_latency_seconds": self.task_completion_latency_seconds,
            "streaming_latency_seconds": self.streaming_latency_seconds,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "repair_attempts": self.repair_attempts,
            "files_modified": self.files_modified,
            "tests_executed": self.tests_executed,
            "ram_usage_mb": self.ram_usage_mb,
            "context_size_tokens": self.context_size_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def capture_memory_mb(cls) -> Optional[float]:
        """Safely captures current process memory working set in MB, or None if unavailable."""
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            return round(proc.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return None
