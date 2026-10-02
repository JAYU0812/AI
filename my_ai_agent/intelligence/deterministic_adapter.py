"""Deterministic Model Adapter for Step 4.

Wraps the existing rule-based and deterministic engines behind the standard
AIModel interface. This ensures the entire existing system continues running
unaltered while establishing the exact same interface future LLMs will implement.

CRITICAL: TokenMetrics.is_available is False and token counts are None.
Never fabricates token metrics for deterministic generation.
"""

from datetime import datetime
import json
import time
from typing import Any, Dict, Generator, Optional

from my_ai_agent.autonomous.action import ActionRisk, ActionType
from my_ai_agent.intelligence.capabilities import ModelCapabilities, ModelHealthReport, ModelHealthStatus
from my_ai_agent.intelligence.model import AIModel
from my_ai_agent.intelligence.model_response import FinishReason, ModelResponse, TokenMetrics
from my_ai_agent.intelligence.structured_output import SchemaType, StructuredOutputValidator
from my_ai_agent.nlp import IntentType, NaturalLanguageUnderstanding


class DeterministicModel(AIModel):
    """Rule-based deterministic implementation of AIModel."""

    def __init__(self, model_id: str = "deterministic"):
        self._model_id = model_id
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
            context_window=0,  # Rule-based has no fixed token context window
            supported_languages=["python", "javascript", "typescript", "json", "markdown"],
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def health(self) -> ModelHealthReport:
        return ModelHealthReport(
            status=ModelHealthStatus.AVAILABLE,
            message="Deterministic rule engine operational",
            last_checked=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            details={"engine": "deterministic_adapter", "version": "1.0.0"},
        )

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generates natural language response from prompt and context."""
        start_time = time.time()
        nlu = NaturalLanguageUnderstanding()
        parse_res = nlu.parse(prompt)
        intent = parse_res.intent
        entities = parse_res.entities

        # Factual summary generation
        target_name = entities.filename or "project workspace"
        text_response = (
            f"Understood request '{prompt}' with intent '{intent.value}'. "
            f"Target entities: {[target_name]}. "
            f"Ready to execute atomic actions."
        )

        latency = time.time() - start_time
        return ModelResponse(
            text=text_response,
            structured=None,
            model=self._model_id,
            tokens=TokenMetrics(is_available=False),
            finish_reason=FinishReason.STOP,
            latency_seconds=latency,
            metadata={"intent": intent.value, "confidence": parse_res.confidence},
        )

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any] | SchemaType | str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generates structured decision conforming to the requested schema."""
        start_time = time.time()
        ctx = context or {}
        schema_name = schema.value if isinstance(schema, SchemaType) else str(schema).lower()

        nlu = NaturalLanguageUnderstanding()
        parse_res = nlu.parse(prompt)
        intent = parse_res.intent
        entities = parse_res.entities
        target_file = entities.filename or "main.py"

        structured_data: Dict[str, Any] = {}

        if "action_decision" in schema_name:
            # Map deterministic intent to autonomous action
            action_map = {
                IntentType.READ_FILE: "READ_FILE",
                IntentType.CREATE_FILE: "CREATE_FILE",
                IntentType.BUILD_FILE: "CREATE_FILE",
                IntentType.BUILD_AND_RUN: "PATCH_FILE",
                IntentType.REFACTOR_FILE: "PATCH_FILE",
                IntentType.WRITE_FILE: "PATCH_FILE",
                IntentType.RUN_COMMAND: "RUN_COMMAND",
                IntentType.UNDERSTAND_PROJECT: "INSPECT_PROJECT",
            }
            action_type = action_map.get(intent, "PATCH_FILE")
            risk = "LOW"
            if action_type in ("RUN_COMMAND", "INSTALL_DEPENDENCY"):
                risk = "MEDIUM"
            elif action_type == "DELETE_FILE":
                risk = "HIGH"

            structured_data = {
                "action": action_type,
                "target": target_file,
                "reason": f"Fulfilling '{prompt}' via deterministic strategy",
                "risk": risk,
                "params": {"intent": intent.value},
            }

        elif "plan" in schema_name:
            structured_data = {
                "goal": prompt,
                "steps": [
                    {
                        "description": f"Inspect project for {prompt}",
                        "action": "INSPECT_PROJECT",
                        "target": "workspace",
                        "reason": "Understand architecture",
                        "risk": "LOW",
                    },
                    {
                        "description": f"Implement changes in {target_file}",
                        "action": "PATCH_FILE",
                        "target": target_file,
                        "reason": f"Fulfill {prompt}",
                        "risk": "LOW",
                    },
                    {
                        "description": "Run tests and verify",
                        "action": "RUN_TESTS",
                        "target": "tests",
                        "reason": "Verify correctness",
                        "risk": "LOW",
                    },
                ],
            }

        elif "diagnosis" in schema_name:
            structured_data = {
                "category": ctx.get("category", "RUNTIME_ERROR"),
                "error_message": ctx.get("error_message", "Error diagnosed"),
                "affected_file": ctx.get("affected_file"),
                "affected_line": ctx.get("affected_line"),
                "failing_symbol": ctx.get("failing_symbol"),
                "probable_cause": ctx.get("probable_cause", "Deterministic rule match"),
                "evidence": ctx.get("evidence", "Output analysis"),
            }

        else:
            structured_data = {
                "intent": intent.value,
                "target": entities.files[0] if entities.files else "workspace",
            }

        latency = time.time() - start_time
        return ModelResponse(
            text=json.dumps(structured_data, indent=2),
            structured=structured_data,
            model=self._model_id,
            tokens=TokenMetrics(is_available=False),
            finish_reason=FinishReason.STOP,
            latency_seconds=latency,
            metadata={"schema": schema_name},
        )

    def stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Streams deterministic response chunks."""
        response = self.generate(prompt, context=context, **kwargs)
        words = response.text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.01)
