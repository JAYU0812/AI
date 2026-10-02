"""Provenance metadata for training examples.

Ensures 100% end-to-end traceability from Step 6A benchmark task -> Step 6B failure
diagnosis -> Step 6C training example and augmentation tracking.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ExampleProvenance:
    """Provenance tracking for an individual training example."""
    source_task_id: str
    source_benchmark_version: str = "v1.0"
    source_step: str = "6B"
    source_root_cause: str = "MODEL_CAPABILITY"
    source_category: str = "code_generation"
    source_failure_type: str = "runtime_error"
    derived_from: Optional[str] = None
    augmentation_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert provenance metadata to dictionary."""
        res = {
            "source_task_id": self.source_task_id,
            "source_benchmark_version": self.source_benchmark_version,
            "source_step": self.source_step,
            "source_root_cause": self.source_root_cause,
            "source_category": self.source_category,
            "source_failure_type": self.source_failure_type,
        }
        if self.derived_from is not None:
            res["derived_from"] = self.derived_from
        if self.augmentation_type is not None:
            res["augmentation_type"] = self.augmentation_type
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExampleProvenance":
        """Instantiate provenance metadata from dictionary."""
        return cls(
            source_task_id=data.get("source_task_id", ""),
            source_benchmark_version=data.get("source_benchmark_version", "v1.0"),
            source_step=data.get("source_step", "6B"),
            source_root_cause=data.get("source_root_cause", "MODEL_CAPABILITY"),
            source_category=data.get("source_category", "code_generation"),
            source_failure_type=data.get("source_failure_type", "runtime_error"),
            derived_from=data.get("derived_from"),
            augmentation_type=data.get("augmentation_type"),
        )
