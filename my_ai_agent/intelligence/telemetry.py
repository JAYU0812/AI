"""Safe Telemetry and Performance Metrics for Step 4.

Records model execution metrics without leaking sensitive information.
CRITICAL: Strips credentials, API keys, passwords, private keys, and
sensitive code tokens before storing or displaying telemetry records.
"""

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, List, Optional
import uuid

from my_ai_agent.intelligence.model_response import TokenMetrics


@dataclass
class TelemetryRecord:
    """Safe model execution telemetry record."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    model_name: str = "deterministic"
    request_id: str = ""
    task_type: str = "general"
    latency_seconds: float = 0.0
    success: bool = True
    error_type: Optional[str] = None
    structured_valid: bool = True
    token_metrics: Dict[str, Any] = field(default_factory=dict)
    sanitized_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "request_id": self.request_id,
            "task_type": self.task_type,
            "latency_seconds": round(self.latency_seconds, 4),
            "success": self.success,
            "error_type": self.error_type,
            "structured_valid": self.structured_valid,
            "token_metrics": self.token_metrics,
            "metadata": self.sanitized_metadata,
        }


class ModelTelemetry:
    """Collects and sanitizes execution telemetry."""

    SENSITIVE_KEY_PATTERNS = [
        r"(?i)password",
        r"(?i)secret",
        r"(?i)api[_-]?key",
        r"(?i)token",
        r"(?i)authorization",
        r"(?i)private[_-]?key",
    ]

    def __init__(self, max_records: int = 100):
        self._records: List[TelemetryRecord] = []
        self._max_records = max_records

    def record(
        self,
        model_name: str,
        latency_seconds: float,
        success: bool = True,
        request_id: str = "",
        task_type: str = "general",
        error_type: Optional[str] = None,
        structured_valid: bool = True,
        token_metrics: Optional[TokenMetrics] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetryRecord:
        """Records an execution event after stripping sensitive fields."""
        sanitized = self._sanitize_dict(metadata or {})
        rec = TelemetryRecord(
            model_name=model_name,
            request_id=request_id,
            task_type=task_type,
            latency_seconds=latency_seconds,
            success=success,
            error_type=error_type,
            structured_valid=structured_valid,
            token_metrics=token_metrics.to_dict() if token_metrics else {},
            sanitized_metadata=sanitized,
        )
        self._records.append(rec)
        if len(self._records) > self._max_records:
            self._records.pop(0)
        return rec

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns recent telemetry records."""
        return [r.to_dict() for r in self._records[-limit:]]

    def _sanitize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Redacts sensitive values matching credential patterns."""
        clean = {}
        for k, v in d.items():
            is_sensitive = any(re.search(pat, k) for pat in self.SENSITIVE_KEY_PATTERNS)
            if is_sensitive:
                clean[k] = "[REDACTED]"
            elif isinstance(v, dict):
                clean[k] = self._sanitize_dict(v)
            elif isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)[:100]
        return clean


# Global singleton instance
telemetry = ModelTelemetry()
