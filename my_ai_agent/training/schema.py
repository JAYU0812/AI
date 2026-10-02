"""Training Example Schema and Data Structures.

Defines the canonical, stable schema for all Step 6C training dataset examples
with comprehensive serialization, deserialization, and structural validation.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Union

from my_ai_agent.training.provenance import ExampleProvenance


class TaskType(str, Enum):
    """Core dataset task domains."""
    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    STRUCTURED_OUTPUT = "structured_output"
    INSTRUCTION_FOLLOWING = "instruction_following"


class OutputType(str, Enum):
    """Output format categories."""
    CODE = "code"
    DIFF = "diff"
    JSON = "json"


@dataclass
class QualityMetadata:
    """Quality and filtering assessment for an individual training example."""
    validated: bool = False
    duplicate: bool = False
    leakage: bool = False
    score: float = 0.0
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "validated": self.validated,
            "duplicate": self.duplicate,
            "leakage": self.leakage,
            "score": round(self.score, 4),
        }
        if self.rejection_reason:
            res["rejection_reason"] = self.rejection_reason
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityMetadata":
        return cls(
            validated=data.get("validated", False),
            duplicate=data.get("duplicate", False),
            leakage=data.get("leakage", False),
            score=float(data.get("score", 0.0)),
            rejection_reason=data.get("rejection_reason"),
        )


@dataclass
class TrainingExample:
    """Canonical training example adhering to Step 6C dataset specification."""
    id: str
    instruction: str
    task_type: str
    target_output: str
    provenance: ExampleProvenance
    version: str = "1.0"
    context: Dict[str, Any] = field(default_factory=lambda: {"project_type": "python", "files": []})
    constraints: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    output_type: str = "code"
    tools_allowed: List[str] = field(default_factory=list)
    verification: List[Dict[str, Any]] = field(default_factory=list)
    difficulty: str = "medium"
    quality: QualityMetadata = field(default_factory=QualityMetadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert training example to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "version": self.version,
            "task_type": self.task_type,
            "instruction": self.instruction,
            "context": self.context,
            "constraints": self.constraints,
            "expected_behavior": self.expected_behavior,
            "target_output": self.target_output,
            "output_type": self.output_type,
            "tools_allowed": self.tools_allowed,
            "verification": self.verification,
            "provenance": self.provenance.to_dict() if hasattr(self.provenance, "to_dict") else self.provenance,
            "difficulty": self.difficulty,
            "quality": self.quality.to_dict() if hasattr(self.quality, "to_dict") else self.quality,
        }

    def to_json(self) -> str:
        """Serialize example to single-line JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingExample":
        """Instantiate training example from dictionary."""
        prov_raw = data.get("provenance", {})
        prov = prov_raw if isinstance(prov_raw, ExampleProvenance) else ExampleProvenance.from_dict(prov_raw)

        qual_raw = data.get("quality", {})
        qual = qual_raw if isinstance(qual_raw, QualityMetadata) else QualityMetadata.from_dict(qual_raw)

        return cls(
            id=data.get("id", ""),
            version=data.get("version", "1.0"),
            task_type=data.get("task_type", TaskType.CODE_GENERATION.value),
            instruction=data.get("instruction", ""),
            context=data.get("context", {"project_type": "python", "files": []}),
            constraints=data.get("constraints", []),
            expected_behavior=data.get("expected_behavior", ""),
            target_output=data.get("target_output", ""),
            output_type=data.get("output_type", "code"),
            tools_allowed=data.get("tools_allowed", []),
            verification=data.get("verification", []),
            provenance=prov,
            difficulty=data.get("difficulty", "medium"),
            quality=qual,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TrainingExample":
        """Parse example from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
