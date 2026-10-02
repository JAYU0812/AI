"""Model Response and Token Metrics Data Models for Step 4 Intelligence Layer.

Provider- and model-agnostic response representations.
CRITICAL: Never fabricate token counts when real tokenization is not performed.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, Optional


class FinishReason(str, Enum):
    """Reason the model generation finished."""
    STOP = "STOP"
    LENGTH = "LENGTH"
    TOOL_CALL = "TOOL_CALL"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class TokenMetrics:
    """Token consumption and context metrics.
    
    If real tokenization is not performed (e.g. deterministic rule engine),
    values are None and is_available is False.
    """
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    context_size: Optional[int] = None
    is_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "context_size": self.context_size,
            "is_available": self.is_available,
        }


@dataclass
class ModelResponse:
    """Standardized response from any AI model provider."""
    text: str
    structured: Optional[Dict[str, Any]] = None
    model: str = "deterministic"
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    finish_reason: FinishReason = FinishReason.STOP
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "structured": self.structured,
            "model": self.model,
            "tokens": self.tokens.to_dict(),
            "finish_reason": self.finish_reason.value,
            "latency_seconds": round(self.latency_seconds, 4),
            "metadata": self.metadata,
        }
