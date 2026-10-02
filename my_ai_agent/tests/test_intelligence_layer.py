"""Unit and Adversarial Test Suite for Step 4 Intelligence Layer.

Covers:
1. AIModel interface, capabilities, health, and TokenMetrics
2. DeterministicModel adapter (zero fabricated token counts)
3. ModelRegistry (register, get, list, set_default, unregister)
4. ModelRouter (default deterministic routing and capability dispatch)
5. StructuredOutputValidator (valid/invalid schemas, required fields, enums)
6. ContextManager & ContextBudget (relevance prioritization, byte limits, sensitive masking)
7. ModelSafetyBoundary (path traversal, system directory, destructive commands, sensitive files)
8. ModelTelemetry (privacy redaction for passwords, API keys, secrets)
9. ReasoningSummary (safe user-facing explanations)
10. LocalModelProvider (unloaded state reporting, refusal to fake inference)
11. Model Swap Test (MockModel swapped into registry without agent modifications)
"""

from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, Generator, Optional
import unittest

from my_ai_agent.autonomous.action import ActionRisk, ActionType
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
    ModelTelemetry,
    PromptBuilder,
    ReasoningSummary,
    SchemaType,
    StructuredOutputValidator,
    TokenMetrics,
)


class MockModel(AIModel):
    """Mock model solely for testing the AIModel interface and model swap capability."""

    def __init__(self, model_id: str = "mock_model"):
        self._model_id = model_id
        self._caps = ModelCapabilities(
            model_id=model_id,
            intent_classification=True,
            planning=True,
            code_generation=True,
            code_modification=True,
            diagnosis=True,
            repair=True,
            structured_output=True,
            streaming=True,
            context_window=8192,
            supported_languages=["python"],
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._caps

    @property
    def health(self) -> ModelHealthReport:
        return ModelHealthReport(
            status=ModelHealthStatus.AVAILABLE,
            message="Mock model active for testing",
        )

    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> ModelResponse:
        return ModelResponse(
            text=f"Mock response for: {prompt}",
            model=self._model_id,
            tokens=TokenMetrics(input_tokens=10, output_tokens=5, context_size=15, is_available=True),
        )

    def generate_structured(self, prompt: str, schema: Dict[str, Any] | str, context: Optional[Dict[str, Any]] = None, **kwargs) -> ModelResponse:
        decision = {
            "action": "PATCH_FILE",
            "target": "auth.py",
            "reason": "Mock patch decision",
            "risk": "LOW",
        }
        return ModelResponse(
            text="{}",
            structured=decision,
            model=self._model_id,
            tokens=TokenMetrics(input_tokens=15, output_tokens=8, context_size=23, is_available=True),
        )

    def stream(self, prompt: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Generator[str, None, None]:
        for word in ["Mock", "streaming", "tokens"]:
            yield word + " "


class TestIntelligenceLayer(unittest.TestCase):
    """Comprehensive Step 4 intelligence tests."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_intel_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ── 1. AIModel & Deterministic Adapter Tests ──────────────────

    def test_deterministic_model_interface(self):
        """Verify DeterministicModel implements full AIModel interface."""
        model = DeterministicModel()
        self.assertEqual(model.model_id, "deterministic")
        self.assertTrue(model.capabilities.intent_classification)
        self.assertEqual(model.health.status, ModelHealthStatus.AVAILABLE)

        # Generate response
        resp = model.generate("Add user authentication")
        self.assertIn("authentication", resp.text.lower())
        self.assertEqual(resp.model, "deterministic")
        self.assertEqual(resp.finish_reason, FinishReason.STOP)
        self.assertFalse(resp.tokens.is_available)
        self.assertIsNone(resp.tokens.input_tokens)  # No fake token counts

        # Structured response
        struct_resp = model.generate_structured("Create file auth.py", schema=SchemaType.ACTION_DECISION)
        self.assertIsNotNone(struct_resp.structured)
        self.assertEqual(struct_resp.structured.get("action"), "CREATE_FILE")
        self.assertEqual(struct_resp.structured.get("target"), "auth.py")

        # Stream
        chunks = list(model.stream("Hello agent"))
        self.assertGreater(len(chunks), 0)

    # ── 2. Model Registry Tests ───────────────────────────────────

    def test_model_registry(self):
        """Test registration, retrieval, listing, and default model handling."""
        reg = ModelRegistry()
        # Initial default is deterministic
        default_m = reg.get_default()
        self.assertEqual(default_m.model_id, "deterministic")

        # Register custom mock
        mock = MockModel("test_mock")
        reg.register("test_mock", mock, make_default=True)

        self.assertEqual(reg.get("test_mock").model_id, "test_mock")
        self.assertEqual(reg.get_default().model_id, "test_mock")

        # List models
        models = reg.list_models()
        self.assertIn("deterministic", models)
        self.assertIn("test_mock", models)
        self.assertTrue(models["test_mock"]["is_default"])

        # Unregister
        unreg_res = reg.unregister("test_mock")
        self.assertTrue(unreg_res)
        self.assertIsNone(reg.get("test_mock"))
        self.assertEqual(reg.get_default().model_id, "deterministic")

        # Deterministic cannot be unregistered
        self.assertFalse(reg.unregister("deterministic"))

    # ── 3. Model Router Tests ─────────────────────────────────────

    def test_model_router_default_and_capability(self):
        """Verify ModelRouter dispatches to default and matches capabilities."""
        reg = ModelRegistry()
        router = ModelRouter(model_registry=reg)

        # Default dispatch
        routed = router.route(task_type="coding", request="Build a feature")
        self.assertEqual(routed.model_id, "deterministic")

        # Capability dispatch
        mock = MockModel("special_model")
        reg.register("special_model", mock)
        routed_cap = router.route(required_capability="context_window")
        self.assertEqual(routed_cap.model_id, "special_model")
        reg.unregister("special_model")

    # ── 4. Structured Output Validation ───────────────────────────

    def test_structured_output_validation_valid(self):
        """Verify valid payloads pass schema validation."""
        action_payload = {
            "action": "PATCH_FILE",
            "target": "services/auth.py",
            "reason": "Add logout method",
            "risk": "LOW",
        }
        valid, err, data = StructuredOutputValidator.validate(action_payload, SchemaType.ACTION_DECISION)
        self.assertTrue(valid)
        self.assertIsNone(err)

        plan_payload = {
            "goal": "Add logout",
            "steps": [
                {"description": "Inspect auth", "action": "READ_FILE"},
                {"description": "Patch auth", "action": "PATCH_FILE"},
            ],
        }
        valid, err, data = StructuredOutputValidator.validate(plan_payload, SchemaType.PLAN)
        self.assertTrue(valid)

    def test_structured_output_validation_invalid(self):
        """Verify invalid payloads (missing required, bad enum, invalid types) are rejected."""
        # Missing required field 'target'
        bad_action = {"action": "PATCH_FILE", "reason": "Missing target", "risk": "LOW"}
        valid, err, _ = StructuredOutputValidator.validate(bad_action, SchemaType.ACTION_DECISION)
        self.assertFalse(valid)
        self.assertIn("target", err)

        # Invalid enum value for action
        bad_enum = {"action": "DESTROY_HARD_DRIVE", "target": "auth.py", "reason": "Bad", "risk": "LOW"}
        valid, err, _ = StructuredOutputValidator.validate(bad_enum, SchemaType.ACTION_DECISION)
        self.assertFalse(valid)
        self.assertIn("invalid value", err)

        # Invalid JSON string
        valid, err, _ = StructuredOutputValidator.validate("{bad json", SchemaType.ACTION_DECISION)
        self.assertFalse(valid)

    # ── 5. Context Manager & Budget Prioritization ────────────────

    def test_context_manager_prioritization_and_budget(self):
        """Verify ContextManager prioritizes relevant files and enforces ContextBudget."""
        # Create test workspace files
        auth_file = self.test_dir / "auth.py"
        auth_file.write_text("class AuthService:\n    def login(self):\n        pass\n", encoding="utf-8")

        calc_file = self.test_dir / "calc.py"
        calc_file.write_text("def add(a, b): return a + b\n", encoding="utf-8")

        secret_file = self.test_dir / ".env"
        secret_file.write_text("DATABASE_PASSWORD=secret123\n", encoding="utf-8")

        budget = ContextBudget(max_files=5, max_file_bytes=1000, max_context_bytes=5000)
        mgr = ContextManager(workspace_root=self.test_dir, budget=budget)

        ctx: AssembledContext = mgr.assemble_context(user_request="Add logout to auth service")

        # 1. auth.py should be selected because it's relevant to 'auth'
        included_paths = [f["path"] for f in ctx.relevant_files]
        self.assertIn("auth.py", included_paths)

        # 2. Sensitive file (.env) MUST NEVER be included
        self.assertNotIn(".env", included_paths)

        # 3. Context markdown formatting
        md = ctx.to_markdown()
        self.assertIn("# SYSTEM ROLE", md)
        self.assertIn("auth.py", md)
        self.assertNotIn("secret123", md)

    # ── 6. Safety Boundary & Adversarial Tests ────────────────────

    def test_adversarial_path_traversal_rejected(self):
        """Model attempting path traversal outside workspace must be rejected."""
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)

        malicious_decision = {
            "action": "READ_FILE",
            "target": "../../.env",
            "reason": "Exfiltrate parent environment",
            "risk": "LOW",
        }
        safe, reason, action = boundary.sanitize_and_convert(malicious_decision)
        self.assertFalse(safe)
        self.assertIn("traverses outside", reason)
        self.assertIsNone(action)

    def test_adversarial_system_path_rejected(self):
        """Model attempting to modify system directory must be rejected."""
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)

        malicious_decision = {
            "action": "PATCH_FILE",
            "target": r"C:\Windows\System32\drivers\etc\hosts",
            "reason": "Modify system hosts",
            "risk": "LOW",
        }
        safe, reason, action = boundary.sanitize_and_convert(malicious_decision)
        self.assertFalse(safe)
        self.assertIn("protected system directory", reason)
        self.assertIsNone(action)

    def test_adversarial_destructive_command_blocked(self):
        """Model attempting destructive command must be intercepted."""
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)

        malicious_decision = {
            "action": "RUN_COMMAND",
            "target": "rmdir /s /q C:\\",
            "reason": "Cleanup files",
            "risk": "LOW",
        }
        safe, reason, action = boundary.sanitize_and_convert(malicious_decision)
        self.assertFalse(safe)
        self.assertIn("Destructive command", reason)
        self.assertIsNone(action)

    def test_adversarial_sensitive_file_rejected(self):
        """Model attempting to access sensitive files directly must be rejected."""
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)

        malicious_decision = {
            "action": "DELETE_FILE",
            "target": ".env",
            "reason": "Delete secrets",
            "risk": "HIGH",
        }
        safe, reason, action = boundary.sanitize_and_convert(malicious_decision)
        self.assertFalse(safe)
        self.assertIn("sensitive file", reason)
        self.assertIsNone(action)

    def test_safe_action_converted_with_approval_check(self):
        """Valid actions are converted to safe AutonomousActions with proper approval flags."""
        boundary = ModelSafetyBoundary(workspace_root=self.test_dir)

        valid_high_risk = {
            "action": "DELETE_FILE",
            "target": "temp_cache.py",
            "reason": "Remove obsolete cache file",
            "risk": "HIGH",
        }
        safe, reason, action = boundary.sanitize_and_convert(valid_high_risk)
        self.assertTrue(safe)
        self.assertIsNotNone(action)
        self.assertEqual(action.type, ActionType.DELETE_FILE)
        self.assertEqual(action.risk, ActionRisk.HIGH)
        self.assertTrue(action.requires_approval)

    # ── 7. Telemetry & Privacy Redaction Tests ───────────────────

    def test_telemetry_redacts_sensitive_data(self):
        """Verify telemetry records metrics while redacting credentials and keys."""
        tel = ModelTelemetry()
        rec = tel.record(
            model_name="deterministic",
            latency_seconds=0.015,
            success=True,
            metadata={
                "operation": "auth_update",
                "api_key": "sk-secret-12345",
                "user_password": "super_secret_password",
                "safe_param": "auth.py",
            },
        )
        self.assertEqual(rec.sanitized_metadata["api_key"], "[REDACTED]")
        self.assertEqual(rec.sanitized_metadata["user_password"], "[REDACTED]")
        self.assertEqual(rec.sanitized_metadata["safe_param"], "auth.py")

    # ── 8. Local Model Provider Abstraction Tests ─────────────────

    def test_local_model_provider_unloaded_health(self):
        """Verify LocalModelProvider reports UNAVAILABLE health and does not fake inference."""
        provider = LocalModelProvider(model_id="local_llama", model_path="weights.bin")
        self.assertEqual(provider.health.status, ModelHealthStatus.UNAVAILABLE)
        self.assertIn("not loaded", provider.health.message)

        # Must refuse to generate when unloaded
        with self.assertRaises(RuntimeError):
            provider.generate("Test prompt")

    # ── 9. Model Swap Test ────────────────────────────────────────

    def test_model_swap_without_code_changes(self):
        """Demonstrate swapping model via registry and routing without altering agent logic."""
        reg = ModelRegistry()
        router = ModelRouter(model_registry=reg)

        # Baseline: routes to deterministic
        self.assertEqual(router.route().model_id, "deterministic")

        # Swap in MockModel
        swap_model = MockModel("swapped_model")
        reg.register("swapped_model", swap_model, make_default=True)

        # Router now resolves to swapped_model
        active_model = router.route()
        self.assertEqual(active_model.model_id, "swapped_model")

        # Test generation through interface
        resp = active_model.generate("Test swapped model")
        self.assertIn("Mock response", resp.text)
        self.assertTrue(resp.tokens.is_available)
        self.assertEqual(resp.tokens.input_tokens, 10)

        # Clean up
        reg.unregister("swapped_model")
        self.assertEqual(router.route().model_id, "deterministic")

    # ── 10. Reasoning Summary Tests ───────────────────────────────

    def test_reasoning_summary_synthesis(self):
        """Verify user-facing action-oriented reasoning summary generation."""
        summary = ReasoningSummary.synthesize_from_context(
            objective="Add logout function",
            project_type="FastAPI",
            identified_files=["auth.py", "routes.py"],
            identified_symbols=["AuthService", "login"],
            has_tests=True,
        )
        msg = summary.to_user_message()
        self.assertIn("Detected FastAPI architecture", msg)
        self.assertIn("auth.py", msg)
        self.assertIn("Strategy:", msg)
        self.assertIn("Next Steps:", msg)


if __name__ == "__main__":
    unittest.main()
