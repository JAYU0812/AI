"""Model Registry for Step 4 Model Management.

Thread-safe registry managing available AI models and active defaults.
Allows seamlessly swapping models without modifying agent logic.
"""

import threading
from typing import Any, Dict, List, Optional

from my_ai_agent.config import config
from my_ai_agent.intelligence.deterministic_adapter import DeterministicModel
from my_ai_agent.intelligence.model import AIModel


class ModelRegistry:
    """Thread-safe registry for AI models."""

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._models = {}
                cls._instance._default_model_id = "deterministic"
                # Register deterministic model as baseline
                det_model = DeterministicModel()
                cls._instance._models[det_model.model_id] = det_model
                # Register local model provider
                from my_ai_agent.intelligence.local_provider import LocalModelProvider
                local_m = LocalModelProvider(model_id="local")
                cls._instance._models["local"] = local_m
                cls._instance._models["local_model"] = local_m
            return cls._instance

    def register(
        self,
        name: str,
        model: AIModel,
        make_default: bool = False,
    ) -> None:
        """Register a model instance by name."""
        with self._lock:
            self._models[name] = model
            if make_default:
                self._default_model_id = name
                config.AI_MODEL_NAME = name

    def get(self, name: str) -> Optional[AIModel]:
        """Retrieve model by name."""
        with self._lock:
            return self._models.get(name)

    def get_default(self) -> AIModel:
        """Retrieve the currently designated default model."""
        with self._lock:
            provider = getattr(config, "AI_MODEL_PROVIDER", "deterministic").lower()
            if provider == "local" and "local" in self._models:
                return self._models["local"]
            if self._default_model_id in self._models:
                return self._models[self._default_model_id]
            configured_name = getattr(config, "AI_MODEL_NAME", "deterministic")
            if configured_name in self._models:
                return self._models[configured_name]
            # Fallback
            det = DeterministicModel()
            self._models[det.model_id] = det
            return det

    def set_default(self, name: str) -> bool:
        """Sets default model by name if registered."""
        with self._lock:
            if name in self._models:
                self._default_model_id = name
                config.AI_MODEL_NAME = name
                return True
            return False

    def unregister(self, name: str) -> bool:
        """Removes a model from the registry."""
        with self._lock:
            if name == "deterministic":
                # Deterministic adapter is baseline and should not be removed
                return False
            if name in self._models:
                del self._models[name]
                if self._default_model_id == name:
                    self._default_model_id = "deterministic"
                return True
            return False

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models and their capabilities/health."""
        with self._lock:
            res = {}
            for name, m in self._models.items():
                res[name] = {
                    "model_id": m.model_id,
                    "is_default": (name == self._default_model_id),
                    "capabilities": m.capabilities.to_dict(),
                    "health": m.health.to_dict(),
                }
            return res


# Global singleton instance
registry = ModelRegistry()
