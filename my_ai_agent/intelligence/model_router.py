"""Model Router for Step 4 Model Dispatch.

Directs requests to appropriate models based on capability requirements,
task type, and system configuration. Currently defaults all requests cleanly
to the DeterministicModel without artificial or fake model selection.
"""

from typing import Any, Dict, Optional

from my_ai_agent.config import config
from my_ai_agent.intelligence.model import AIModel
from my_ai_agent.intelligence.model_registry import ModelRegistry, registry


class ModelRouter:
    """Routes intelligence requests to appropriate AIModel instances."""

    def __init__(self, model_registry: Optional[ModelRegistry] = None):
        self.registry = model_registry or registry

    def route(
        self,
        task_type: str = "general",
        request: str = "",
        context: Optional[Dict[str, Any]] = None,
        required_capability: Optional[str] = None,
    ) -> AIModel:
        """Determines the appropriate model for the task.
        
        Currently routes to configured default model (DeterministicModel).
        Future expansions can route fast tasks to small models and complex tasks
        to heavier reasoning models.
        """
        provider = getattr(config, "AI_MODEL_PROVIDER", "deterministic").lower()
        fallback_mode = getattr(config, "AI_MODEL_FALLBACK", "deterministic").lower()

        # 1. If provider is explicitly configured as 'local'
        if provider == "local":
            local_model = self.registry.get("local")
            if local_model:
                from my_ai_agent.intelligence.capabilities import ModelHealthStatus
                if local_model.health.status == ModelHealthStatus.AVAILABLE:
                    return local_model
                elif fallback_mode == "none":
                    # Surface unavailability honestly
                    return local_model
                else:
                    # Explicit fallback to deterministic
                    det = self.registry.get("deterministic")
                    if det:
                        return det

        # 2. Check if user configured a specific model by name
        configured_name = getattr(config, "AI_MODEL_NAME", "deterministic")
        target_model = self.registry.get(configured_name)

        # 3. Check capability requirements if specified
        if required_capability:
            from my_ai_agent.intelligence.capabilities import ModelHealthStatus
            def has_cap(m: AIModel, cap: str) -> bool:
                if m.health.status != ModelHealthStatus.AVAILABLE:
                    return False
                caps_dict = m.capabilities.to_dict().get("capabilities", {})
                if caps_dict.get(cap, False):
                    return True
                val = getattr(m.capabilities, cap, None)
                if isinstance(val, bool):
                    return val
                if isinstance(val, (int, float)) and val > 0:
                    return True
                return False

            if not target_model or not has_cap(target_model, required_capability):
                for name, info in self.registry.list_models().items():
                    m = self.registry.get(name)
                    if m and has_cap(m, required_capability):
                        return m

        if target_model:
            return target_model

        # Fallback to registry default
        return self.registry.get_default()


# Global singleton instance
router = ModelRouter()
