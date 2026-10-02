"""Model Capabilities and Health Status Models for Step 4.

Records factual capabilities and real health states. Zero dummy metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class ModelHealthStatus(str, Enum):
    """Factual operational status of a model."""
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    LOADING = "LOADING"
    ERROR = "ERROR"


@dataclass
class ModelHealthReport:
    """Detailed health check report for an AI model."""
    status: ModelHealthStatus
    message: str = "Model operational"
    last_checked: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "message": self.message,
            "last_checked": self.last_checked,
            "details": self.details,
        }


@dataclass
class ModelCapabilities:
    """Actual supported capabilities of an AI model.
    
    Never invent capabilities or fake benchmark scores.
    """
    model_id: str
    intent_classification: bool = True
    planning: bool = True
    code_generation: bool = True
    code_modification: bool = True
    diagnosis: bool = True
    repair: bool = True
    structured_output: bool = True
    streaming: bool = True
    context_window: int = 0  # 0 indicates rule-based / unbounded / not applicable
    supported_languages: List[str] = field(default_factory=lambda: ["python", "javascript", "typescript", "json", "markdown"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "capabilities": {
                "intent_classification": self.intent_classification,
                "planning": self.planning,
                "code_generation": self.code_generation,
                "code_modification": self.code_modification,
                "diagnosis": self.diagnosis,
                "repair": self.repair,
                "structured_output": self.structured_output,
                "streaming": self.streaming,
            },
            "context_window": self.context_window,
            "supported_languages": self.supported_languages,
        }
