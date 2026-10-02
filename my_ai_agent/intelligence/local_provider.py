"""Local Model Provider Implementation for Step 5.

Connects real local open-source models (via TransformersBackend, OllamaBackend, etc.)
to the AIModel interface. Enforces genuine inference, factual token metrics,
and truthful model health reporting.

CRITICAL INTEGRITY RULES:
1. Never fake inference or return dummy responses.
2. If the local runtime or weights are unavailable, truthfully report UNAVAILABLE health.
3. The LLM is an intelligence provider, NOT a tool-execution or security authority.
"""

from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Generator, Optional

from my_ai_agent.config import config
from my_ai_agent.intelligence.backend import (
    LocalInferenceBackend,
    NullBackend,
    OllamaBackend,
    TransformersBackend,
)
from my_ai_agent.intelligence.capabilities import ModelCapabilities, ModelHealthReport, ModelHealthStatus
from my_ai_agent.intelligence.model import AIModel
from my_ai_agent.intelligence.model_response import FinishReason, ModelResponse, TokenMetrics
from my_ai_agent.intelligence.structured_output import SchemaType, StructuredOutputValidator


class LocalModelProvider(AIModel):
    """Real local model provider implementing the standard AIModel interface."""

    def __init__(
        self,
        model_id: str = "local_model",
        backend: Optional[LocalInferenceBackend] = None,
        model_path: Optional[Path | str] = None,
        context_window: int = 2048,
    ):
        self._model_id = model_id
        self.model_path = Path(model_path).resolve() if model_path else None
        self._context_window = context_window

        if backend is not None:
            self.backend: LocalInferenceBackend = backend
        elif self.model_path and not self.model_path.exists():
            self.backend = NullBackend(message=f"Model path does not exist: {self.model_path}")
        else:
            # Build backend from configuration
            backend_type = getattr(config, "AI_MODEL_BACKEND", "transformers").lower()
            model_target = str(self.model_path) if self.model_path else getattr(config, "AI_MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
            device = getattr(config, "AI_MODEL_DEVICE", "cpu")

            if backend_type == "transformers":
                self.backend = TransformersBackend(
                    model_name_or_path=model_target,
                    device=device,
                    context_length=context_window,
                )
            elif backend_type == "ollama":
                self.backend = OllamaBackend(
                    model_name=model_target,
                    context_length=context_window,
                )
            else:
                self.backend = NullBackend(message=f"Unknown backend type: '{backend_type}'")

        self._capabilities = ModelCapabilities(
            model_id=self._model_id,
            intent_classification=True,
            planning=True,
            code_generation=True,
            code_modification=True,
            diagnosis=True,
            repair=True,
            structured_output=True,
            streaming=True,
            context_window=self._context_window,
            supported_languages=["python", "javascript", "typescript", "c", "cpp", "json", "html"],
        )

    def load(self) -> bool:
        """Loads underlying backend weights."""
        return self.backend.load()

    def unload(self) -> bool:
        """Unloads underlying backend weights to free memory."""
        return self.backend.unload()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def health(self) -> ModelHealthReport:
        """Verifies real operational status of the underlying local inference backend."""
        is_ready = self.backend.is_available()
        meta = self.backend.get_metadata()
        if is_ready:
            return ModelHealthReport(
                status=ModelHealthStatus.AVAILABLE,
                message="Local inference backend active",
                last_checked=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                details={
                    "backend": meta.get("backend", "local"),
                    "model": meta.get("model_name_or_path") or meta.get("model_name"),
                    "device": meta.get("device", "cpu"),
                    "context_length": meta.get("context_length", self._context_window),
                },
            )
        return ModelHealthReport(
            status=ModelHealthStatus.UNAVAILABLE,
            message="Local model weights not loaded or runtime unconfigured",
            last_checked=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            details={
                "model_path": str(self.model_path) if self.model_path else None,
                "is_loaded": False,
                "metadata": meta,
            },
        )

    def load_model(self, backend: Optional[LocalInferenceBackend] = None) -> bool:
        """Loads weights and prepares inference backend."""
        if backend:
            self.backend = backend
        return self.backend.load()

    def unload_model(self) -> bool:
        """Frees model weights from memory."""
        return self.backend.unload()

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ModelResponse:
        """Executes real local inference."""
        if not self.backend.is_available():
            # Attempt loading once if configured
            if not self.load_model():
                raise RuntimeError(
                    f"Cannot generate from '{self._model_id}': local model weights are not loaded. "
                    "No fake inference permitted."
                )

        m_tokens = max_tokens or getattr(config, "AI_MODEL_MAX_TOKENS", 512)
        temp = temperature if temperature is not None else getattr(config, "AI_MODEL_TEMPERATURE", 0.2)

        res = self.backend.generate(
            prompt=prompt,
            max_tokens=m_tokens,
            temperature=temp,
            **kwargs,
        )

        return ModelResponse(
            text=res.get("text", ""),
            structured=None,
            model=self._model_id,
            tokens=res.get("tokens", TokenMetrics(is_available=False)),
            finish_reason=res.get("finish_reason", FinishReason.STOP),
            latency_seconds=res.get("latency_seconds", 0.0),
            metadata=res.get("metadata", {}),
        )

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any] | SchemaType | str,
        context: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ModelResponse:
        """Executes real local inference and validates result against JSON schema."""
        if not self.backend.is_available():
            if not self.load_model():
                raise RuntimeError(
                    f"Cannot generate structured from '{self._model_id}': local model weights are not loaded."
                )

        schema_key = schema.value if isinstance(schema, SchemaType) else str(schema).lower()

        # Format prompt with strict JSON formatting instruction
        structured_prompt = (
            f"{prompt}\n\n"
            f"CRITICAL REQUIREMENT: Output MUST be a valid JSON object matching the '{schema_key}' schema.\n"
            "Do not include any introductory or concluding text. Output JSON only.\n"
        )

        resp = self.generate(
            prompt=structured_prompt,
            context=context,
            max_tokens=max_tokens or 512,
            temperature=temperature or 0.1,  # Low temperature for strict schema adherence
            **kwargs,
        )

        raw_text = resp.text.strip()
        # Parse JSON from output
        json_obj = self._extract_json(raw_text)
        if json_obj is None:
            return ModelResponse(
                text=raw_text,
                structured=None,
                model=self._model_id,
                tokens=resp.tokens,
                finish_reason=FinishReason.ERROR,
                latency_seconds=resp.latency_seconds,
                metadata={"error": "Model response did not contain valid JSON object"},
            )

        # Validate against schema
        valid, err_msg, validated_dict = StructuredOutputValidator.validate(json_obj, schema_key)
        if not valid:
            return ModelResponse(
                text=raw_text,
                structured=None,
                model=self._model_id,
                tokens=resp.tokens,
                finish_reason=FinishReason.ERROR,
                latency_seconds=resp.latency_seconds,
                metadata={"error": f"Schema validation error: {err_msg}"},
            )

        return ModelResponse(
            text=raw_text,
            structured=validated_dict,
            model=self._model_id,
            tokens=resp.tokens,
            finish_reason=FinishReason.STOP,
            latency_seconds=resp.latency_seconds,
            metadata={"validated_schema": schema_key},
        )

    def stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Streams real token chunks incrementally from the local model."""
        if not self.backend.is_available():
            if not self.load_model():
                raise RuntimeError(
                    f"Cannot stream from '{self._model_id}': local model weights are not loaded."
                )

        m_tokens = max_tokens or getattr(config, "AI_MODEL_MAX_TOKENS", 512)
        temp = temperature if temperature is not None else getattr(config, "AI_MODEL_TEMPERATURE", 0.2)

        yield from self.backend.stream(
            prompt=prompt,
            max_tokens=m_tokens,
            temperature=temp,
            **kwargs,
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON object from markdown code fences or raw string."""
        # 1. Try markdown ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        # 2. Try outermost curly braces { ... }
        brace_match = re.search(r"(\{[\s\S]*\})", text)
        if brace_match:
            candidate = brace_match.group(1).strip()
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        # 3. Direct parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        return None
