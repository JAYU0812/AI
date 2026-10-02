"""Standard AI Model Interface for Step 4 Model Abstraction.

All models (deterministic, local, fine-tuned, or custom trained) implement
this standard interface. The rest of the Agent layer depends exclusively on
this abstraction rather than specific providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional

from my_ai_agent.intelligence.capabilities import ModelCapabilities, ModelHealthReport
from my_ai_agent.intelligence.model_response import ModelResponse


class AIModel(ABC):
    """Abstract base class for all AI intelligence engines."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for the model (e.g. 'deterministic', 'local_codellama')."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Declared actual capabilities of this model."""
        pass

    @property
    @abstractmethod
    def health(self) -> ModelHealthReport:
        """Real-time operational health of the model."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate unstructured natural language response."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate a response strictly validated against a JSON schema."""
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream response chunks in real-time."""
        pass
