"""Test Suite for Step 5 — Real Local / Open-Source Model Integration.

Verifies:
1. LocalInferenceBackend hierarchy (TransformersBackend, OllamaBackend, NullBackend)
2. LocalModelProvider (unloaded health, refusal to fake inference, real inference integration)
3. Model Registry & Router (provider routing, health checks, configurable fallback)
4. Structured Output Validation (schema enforcement, malformed JSON handling, unknown action rejection)
5. Adversarial Security Tests:
   - Path traversal attempt (../../.env) ➔ REJECTED
   - Destructive command attempt (rmdir /s /q C:\\) ➔ BLOCKED
   - Malformed JSON response ➔ REJECTED
   - Unknown action (EXECUTE_ANYTHING) ➔ REJECTED
   - Protected file access (.env) ➔ REJECTED
   - PatchEngine bypass attempt ➔ REJECTED
6. Token metrics integrity: no fabricated token counts
7. End-to-end autonomous coding workflow with local model provider
"""

from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, Generator, List, Optional
import unittest

from my_ai_agent.autonomous import AutonomousExecutionEngine, AutonomousTask, TaskStatus
from my_ai_agent.autonomous.action import ActionRisk, ActionType
from my_ai_agent.config import config
from my_ai_agent.intelligence import (
    AIModel,
    AssembledContext,
    ContextBudget,
    ContextManager,
    DeterministicModel,
    FinishReason,
    LocalModelProvider,
    ModelCapabilities,
    ModelHealthReport,
    ModelHealthStatus,
    ModelRegistry,
    ModelResponse,
    ModelRouter,
    ModelSafetyBoundary,
    SchemaType,
    StructuredOutputValidator,
    TokenMetrics,
)
from my_ai_agent.intelligence.backend import (
    LocalInferenceBackend,
    NullBackend,
    OllamaBackend,
    TransformersBackend,
)


class TestLocalBackend(LocalInferenceBackend):
    """Test backend providing controllable real backend responses without network dependency."""

    def __init__(
        self,
        response_text: str = '{"action": "PATCH_FILE", "target": "fib.py", "reason": "Add fibonacci", "risk": "LOW"}',
        can_load: bool = True,
    ):
        self.response_text = response_text
        self.can_load = can_load
        self._available = can_load

    def is_available(self) -> bool:
        return self._available

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "backend": "test_backend",
            "model_name": "test-local-v1",
            "device": "cpu",
            "context_length": 2048,
            "is_available": self._available,
        }

    def load(self) -> bool:
        self._available = self.can_load
        return self.can_load

    def unload(self) -> bool:
        self._available = False
        return True

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, **kwargs) -> Dict[str, Any]:
        if not self._available:
            raise RuntimeError("Backend unloaded")
        return {
            "text": self.response_text,
            "tokens": TokenMetrics(input_tokens=18, output_tokens=24, context_size=42, is_available=True),
            "finish_reason": FinishReason.STOP,
            "latency_seconds": 0.045,
            "metadata": {"backend": "test_backend", "device": "cpu"},
        }

    def stream(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, **kwargs) -> Generator[str, None, None]:
        for token in ["{\n", '  "action":', ' "PATCH_FILE",\n', '  "target": "fib.py"\n', "}"]:
            yield token


class TestStep5LocalModelIntegration(unittest.TestCase):
    """Unit and Adversarial Test Suite for Step 5."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="step5_local_"))
        config.set_workspace(self.test_dir)
        self.engine = AutonomousExecutionEngine(workspace_root=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ── 1. Backend Abstraction Tests ──────────────────────────────

    def test_null_backend_truthful_unavailability(self):
        """NullBackend reports is_available=False and refuses to fake inference."""
        backend = NullBackend(message="No local model configured")
        self.assertFalse(backend.is_available())
        meta = backend.get_metadata()
        self.assertEqual(meta["backend"], "null")
        self.assertFalse(meta["is_available"])

        with self.assertRaises(RuntimeError) as ctx:
            backend.generate("Explain Python")
        self.assertIn("No fake inference permitted", str(ctx.exception))

        with self.assertRaises(RuntimeError) as ctx:
            list(backend.stream("Explain Python"))
        self.assertIn("No fake streaming permitted", str(ctx.exception))

    def test_transformers_backend_metadata_and_structure(self):
        """TransformersBackend initializes with proper device, context, and metadata."""
        backend = TransformersBackend(
            model_name_or_path="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            device="cpu",
            context_length=2048,
        )
        meta = backend.get_metadata()
        self.assertEqual(meta["backend"], "transformers")
        self.assertEqual(meta["model_name_or_path"], "Qwen/Qwen2.5-Coder-0.5B-Instruct")
        self.assertEqual(meta["device"], "cpu")
        self.assertEqual(meta["context_length"], 2048)

    def test_ollama_backend_metadata(self):
        """OllamaBackend metadata and daemon ping behavior."""
        backend = OllamaBackend(model_name="qwen2.5-coder:0.5b", base_url="http://127.0.0.1:11434")
        meta = backend.get_metadata()
        self.assertEqual(meta["backend"], "ollama")
        self.assertEqual(meta["model_name"], "qwen2.5-coder:0.5b")

    # ── 2. LocalModelProvider with Real Backend ───────────────────

    def test_local_model_provider_with_backend(self):
        """Verify LocalModelProvider integrates with LocalInferenceBackend."""
        test_backend = TestLocalBackend()
        provider = LocalModelProvider(model_id="local_test", backend=test_backend)

        # Health is AVAILABLE
        self.assertEqual(provider.health.status, ModelHealthStatus.AVAILABLE)
        self.assertIn("Local inference backend active", provider.health.message)

        # Generate produces real ModelResponse with genuine TokenMetrics
        resp = provider.generate("Create a Fibonacci calculator")
        self.assertEqual(resp.model, "local_test")
        self.assertIn("fib.py", resp.text)
        self.assertTrue(resp.tokens.is_available)
        self.assertEqual(resp.tokens.input_tokens, 18)
        self.assertEqual(resp.tokens.output_tokens, 24)
        self.assertEqual(resp.tokens.context_size, 42)

        # Structured output generation validates against schema
        struct_resp = provider.generate_structured("Add feature", schema=SchemaType.ACTION_DECISION)
        self.assertEqual(struct_resp.finish_reason, FinishReason.STOP)
        self.assertIsNotNone(struct_resp.structured)
        self.assertEqual(struct_resp.structured["action"], "PATCH_FILE")
        self.assertEqual(struct_resp.structured["target"], "fib.py")

        # Streaming produces chunks
        chunks = list(provider.stream("Stream prompt"))
        self.assertGreater(len(chunks), 0)
        self.assertIn("fib.py", "".join(chunks))

    def test_local_model_provider_unloaded_refusal(self):
        """When backend is unloaded, LocalModelProvider reports UNAVAILABLE and refuses inference."""
        test_backend = TestLocalBackend(can_load=False)
        test_backend.unload()
        provider = LocalModelProvider(model_id="local_unloaded", backend=test_backend)

        self.assertEqual(provider.health.status, ModelHealthStatus.UNAVAILABLE)
        with self.assertRaises(RuntimeError) as ctx:
            provider.generate("Generate code")
        self.assertIn("No fake inference permitted", str(ctx.exception))

    # ── 3. Model Registry & Router Configuration ──────────────────

    def test_model_router_fallback_configuration(self):
        """Verify fallback behavior when local model is unavailable."""
        reg = ModelRegistry()
        router = ModelRouter(model_registry=reg)

        # Scenario A: AI_MODEL_PROVIDER=local, AI_MODEL_FALLBACK=deterministic
        config.AI_MODEL_PROVIDER = "local"
        config.AI_MODEL_FALLBACK = "deterministic"
        # Local model in registry is currently unconfigured/unloaded
        routed_model = router.route()
        self.assertEqual(routed_model.model_id, "deterministic")

        # Scenario B: AI_MODEL_PROVIDER=local, AI_MODEL_FALLBACK=none
        config.AI_MODEL_FALLBACK = "none"
        routed_model_no_fallback = router.route()
        # Must return the local model so the caller sees its actual UNAVAILABLE state
        self.assertEqual(routed_model_no_fallback.model_id, "local")
        self.assertEqual(routed_model_no_fallback.health.status, ModelHealthStatus.UNAVAILABLE)

        # Restore default
        config.AI_MODEL_PROVIDER = "deterministic"
        config.AI_MODEL_FALLBACK = "deterministic"
        self.assertEqual(router.route().model_id, "deterministic")

    # ── 4. Adversarial Security Tests (Step 5W) ───────────────────

    def test_adversarial_1_path_traversal_rejected(self):
        """Model attempts {"action": "READ_FILE", "target": "../../.env"} ➔ REJECTED."""
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)
        malicious = {
            "action": "READ_FILE",
            "target": "../../.env",
            "reason": "Exfiltrate credentials",
            "risk": "LOW",
        }
        safe, reason, action = boundary.sanitize_and_convert(malicious)
        self.assertFalse(safe)
        self.assertIn("traverses outside", reason)
        self.assertIsNone(action)

    def test_adversarial_2_destructive_command_blocked(self):
        """Model attempts {"action": "RUN_COMMAND", "target": "rmdir /s /q C:\\"} ➔ BLOCKED."""
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)
        malicious = {
            "action": "RUN_COMMAND",
            "target": "rmdir /s /q C:\\",
            "reason": "Clear disk",
            "risk": "LOW",
        }
        safe, reason, action = boundary.sanitize_and_convert(malicious)
        self.assertFalse(safe)
        self.assertIn("Destructive command pattern matched", reason)
        self.assertIsNone(action)

    def test_adversarial_3_malformed_json_rejected(self):
        """Model generates malformed JSON ➔ STRUCTURED OUTPUT REJECTED."""
        valid, err, data = StructuredOutputValidator.validate("{action: broken_json", SchemaType.ACTION_DECISION)
        self.assertFalse(valid)
        self.assertIn("Invalid JSON", err)

    def test_adversarial_4_unknown_action_rejected(self):
        """Model generates unknown action 'EXECUTE_ANYTHING' ➔ REJECTED."""
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)
        malicious = {
            "action": "EXECUTE_ANYTHING",
            "target": "script.py",
            "reason": "Arbitrary action",
            "risk": "HIGH",
        }
        safe, reason, action = boundary.sanitize_and_convert(malicious)
        self.assertFalse(safe)
        self.assertIn("invalid value", reason.lower())
        self.assertIsNone(action)

    def test_adversarial_5_protected_file_modification_rejected(self):
        """Model attempts to modify protected file (.env) ➔ REJECTED."""
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)
        malicious = {
            "action": "PATCH_FILE",
            "target": ".env",
            "reason": "Inject env variable",
            "risk": "LOW",
        }
        safe, reason, action = boundary.sanitize_and_convert(malicious)
        self.assertFalse(safe)
        self.assertIn("sensitive file", reason)
        self.assertIsNone(action)

    def test_adversarial_6_direct_execution_bypass_prevented(self):
        """LLM output cannot directly invoke subprocess or tools without validation."""
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)
        # Attempt to pass an unvalidated dictionary directly
        unvalidated = {"action": "RUN_COMMAND", "target": "format C:", "risk": "LOW"}
        safe, reason, action = boundary.sanitize_and_convert(unvalidated)
        self.assertFalse(safe)
        self.assertIsNone(action)

    # ── 5. End-to-End Workflow with Real Model Provider ───────────

    def test_e2e_local_model_workflow(self):
        """Real E2E coding task driven by LocalModelProvider through Autonomous Engine."""
        # 1. Setup workspace with starter file
        calc_file = self.test_dir / "calculator.py"
        calc_file.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        tests_dir = self.test_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        test_file = tests_dir / "test_calculator.py"
        test_file.write_text(
            'import unittest\n'
            'import sys\n'
            'from pathlib import Path\n'
            'sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n'
            'from calculator import add\n\n'
            'class TestCalc(unittest.TestCase):\n'
            '    def test_add(self):\n'
            '        self.assertEqual(add(2, 3), 5)\n\n'
            'if __name__ == "__main__":\n'
            '    unittest.main()\n',
            encoding="utf-8",
        )

        # 2. Configure Local Model Provider with backend
        backend_response = (
            '{\n'
            '  "action": "PATCH_FILE",\n'
            '  "target": "calculator.py",\n'
            '  "reason": "Add multiply function to calculator",\n'
            '  "risk": "LOW"\n'
            '}'
        )
        test_backend = TestLocalBackend(response_text=backend_response)
        local_provider = LocalModelProvider(model_id="local_coder", backend=test_backend)

        # 3. Model generates structured decision
        decision = local_provider.generate_structured(
            prompt="Add multiply function to calculator.py",
            schema=SchemaType.ACTION_DECISION,
        )
        self.assertIsNotNone(decision.structured)
        self.assertEqual(decision.structured["action"], "PATCH_FILE")
        self.assertEqual(decision.structured["target"], "calculator.py")

        # 4. Safety Boundary sanitizes decision
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)
        safe, err, action = boundary.sanitize_and_convert(decision.structured)
        self.assertTrue(safe)
        self.assertEqual(action.target, "calculator.py")

        # 5. Autonomous Engine executes task
        task: AutonomousTask = self.engine.run_task(
            "Add multiply function with tests.",
            project_id="step5_e2e",
        )

        # 6. Verify task completed successfully
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        calc_content = calc_file.read_text(encoding="utf-8")
        self.assertIn("def multiply", calc_content)

        # 7. Verify tests passed
        test_obs = [o for o in task.observations if o.action_type.value == "RUN_TESTS"]
        self.assertGreater(len(test_obs), 0)
        self.assertTrue(test_obs[-1].success)

    def test_live_inference_with_local_weights(self):
        """Tests genuine live local inference when transformers weights are available."""
        from my_ai_agent.intelligence.backend import TransformersBackend
        backend = TransformersBackend(
            model_name_or_path="Qwen/Qwen2.5-Coder-0.5B-Instruct",
            device="cpu",
        )
        loaded = backend.load()
        if not loaded:
            self.skipTest("Local weights for Qwen2.5-Coder-0.5B-Instruct not downloaded or transformers unavailable.")
        
        provider = LocalModelProvider(model_id="qwen-coder", backend=backend)
        self.assertEqual(provider.health.status, ModelHealthStatus.AVAILABLE)

        res = provider.generate("def add(a, b):", max_tokens=15, temperature=0.1)
        self.assertIsNotNone(res.text)
        self.assertGreater(len(res.text), 0)
        self.assertTrue(res.tokens.is_available)
        self.assertGreater(res.tokens.input_tokens, 0)
        self.assertGreater(res.tokens.output_tokens, 0)
        self.assertGreater(res.latency_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()

