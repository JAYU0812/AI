"""Root-Cause Taxonomy and Data Models for Baseline Weakness Analysis (Step 6B).

Defines the centralized taxonomy of root causes, confidence levels, fix types,
and structured root-cause diagnostic records.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RootCauseCategory(str, Enum):
    """Exhaustive categorization of reasons why an agent or model task failed or partially passed."""
    MODEL_CAPABILITY = "MODEL_CAPABILITY"
    PROMPT_DESIGN = "PROMPT_DESIGN"
    CONTEXT_ENGINEERING = "CONTEXT_ENGINEERING"
    AGENT_ARCHITECTURE = "AGENT_ARCHITECTURE"
    TOOL_INTERFACE = "TOOL_INTERFACE"
    STRUCTURED_OUTPUT_INTERFACE = "STRUCTURED_OUTPUT_INTERFACE"
    PROJECT_UNDERSTANDING = "PROJECT_UNDERSTANDING"
    PATCH_ENGINE = "PATCH_ENGINE"
    AUTONOMOUS_ENGINE = "AUTONOMOUS_ENGINE"
    VERIFICATION = "VERIFICATION"
    SAFETY_BOUNDARY = "SAFETY_BOUNDARY"
    DATASET_QUALITY = "DATASET_QUALITY"
    EVALUATOR_ERROR = "EVALUATOR_ERROR"
    CONFIGURATION = "CONFIGURATION"
    PERFORMANCE = "PERFORMANCE"
    UNKNOWN = "UNKNOWN"


class ConfidenceLevel(str, Enum):
    """Confidence level assigned to a root-cause diagnostic."""
    HIGH = "HIGH"        # Evidence is conclusive (e.g. direct syntax failure, schema mismatch, or assertion trace)
    MEDIUM = "MEDIUM"    # Evidence strongly indicates cause but secondary factors exist
    LOW = "LOW"          # Ambiguous or multi-factorial issue requiring targeted ablation experiments


class FixType(str, Enum):
    """Actionable remediation category for engineering roadmap planning."""
    PROMPT_FIX = "PROMPT_FIX"
    CONTEXT_FIX = "CONTEXT_FIX"
    AGENT_ARCHITECTURE_FIX = "AGENT_ARCHITECTURE_FIX"
    TOOL_INTERFACE_FIX = "TOOL_INTERFACE_FIX"
    STRUCTURED_OUTPUT_FIX = "STRUCTURED_OUTPUT_FIX"
    MODEL_SELECTION = "MODEL_SELECTION"
    DATASET_FIX = "DATASET_FIX"
    EVALUATOR_FIX = "EVALUATOR_FIX"
    FINE_TUNING_CANDIDATE = "FINE_TUNING_CANDIDATE"
    MODEL_TRAINING_CANDIDATE = "MODEL_TRAINING_CANDIDATE"
    NO_ACTION = "NO_ACTION"


@dataclass
class RootCauseRecord:
    """Diagnostic root-cause determination for a specific benchmark task."""
    task_id: str
    status: str
    category: str
    difficulty: str
    score_percentage: float
    failure_type: str
    primary_root_cause: RootCauseCategory
    secondary_causes: List[RootCauseCategory] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    recommended_action: str = ""
    fix_type: FixType = FixType.NO_ACTION
    is_training_candidate: bool = False
    non_training_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "category": self.category,
            "difficulty": self.difficulty,
            "score_percentage": self.score_percentage,
            "failure_type": self.failure_type,
            "primary_root_cause": self.primary_root_cause.value,
            "secondary_causes": [c.value for c in self.secondary_causes],
            "evidence": self.evidence,
            "confidence": round(self.confidence, 2),
            "confidence_level": self.confidence_level.value,
            "recommended_action": self.recommended_action,
            "fix_type": self.fix_type.value,
            "is_training_candidate": self.is_training_candidate,
            "non_training_reason": self.non_training_reason,
        }


@dataclass
class FailureCluster:
    """Group of tasks sharing identical failure modes and common root causes."""
    cluster_id: str
    name: str
    description: str
    task_ids: List[str]
    dominant_root_cause: RootCauseCategory
    fix_type: FixType
    priority: str  # P0, P1, P2, P3
    impact_task_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "description": self.description,
            "task_ids": self.task_ids,
            "dominant_root_cause": self.dominant_root_cause.value,
            "fix_type": self.fix_type.value,
            "priority": self.priority,
            "impact_task_count": len(self.task_ids),
        }
