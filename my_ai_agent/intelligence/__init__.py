"""Step 4 Intelligence Layer & Model Integration Foundation Package.

Provides a unified model abstraction, structured output validation, context management,
safety boundary enforcement, and capability-specific engines.
"""

from my_ai_agent.intelligence.backend import (
    LocalInferenceBackend,
    NullBackend,
    OllamaBackend,
    TransformersBackend,
)
from my_ai_agent.intelligence.capabilities import ModelCapabilities, ModelHealthReport, ModelHealthStatus
from my_ai_agent.intelligence.context_manager import AssembledContext, ContextBudget, ContextManager
from my_ai_agent.intelligence.deterministic_adapter import DeterministicModel
from my_ai_agent.intelligence.engines import (
    BaseIntelligenceEngine,
    CodingEngine,
    DiagnosisIntelligence,
    ExplanationEngine,
    IntentEngine,
    PlanningEngine,
    RepairIntelligence,
)
from my_ai_agent.intelligence.local_provider import LocalModelProvider
from my_ai_agent.intelligence.model import AIModel
from my_ai_agent.intelligence.model_registry import ModelRegistry, registry
from my_ai_agent.intelligence.model_response import FinishReason, ModelResponse, TokenMetrics
from my_ai_agent.intelligence.model_router import ModelRouter, router
from my_ai_agent.intelligence.prompt_builder import PromptBuilder
from my_ai_agent.intelligence.reasoning import ReasoningSummary
from my_ai_agent.intelligence.safety_boundary import ModelSafetyBoundary, SafetyViolation
from my_ai_agent.intelligence.structured_output import SchemaType, StructuredOutputValidator
from my_ai_agent.intelligence.telemetry import ModelTelemetry, TelemetryRecord, telemetry

__all__ = [
    "AIModel",
    "ModelResponse",
    "TokenMetrics",
    "FinishReason",
    "ModelCapabilities",
    "ModelHealthStatus",
    "ModelHealthReport",
    "DeterministicModel",
    "LocalModelProvider",
    "ModelRegistry",
    "registry",
    "ModelRouter",
    "router",
    "ContextBudget",
    "ContextManager",
    "AssembledContext",
    "PromptBuilder",
    "ReasoningSummary",
    "SchemaType",
    "StructuredOutputValidator",
    "ModelSafetyBoundary",
    "SafetyViolation",
    "ModelTelemetry",
    "TelemetryRecord",
    "telemetry",
    "BaseIntelligenceEngine",
    "IntentEngine",
    "PlanningEngine",
    "CodingEngine",
    "DiagnosisIntelligence",
    "RepairIntelligence",
    "ExplanationEngine",
]
