"""End-to-End Test for Step 4 Intelligence Layer & Model Abstraction.

Verifies Section 31 end-to-end integration:
User ➔ Intelligence Interface ➔ Model (Deterministic) ➔ Structured Decision ➔
Safety Boundary ➔ Project Context ➔ Planner ➔ PatchEngine ➔ Tests ➔ Verification.
"""

from pathlib import Path
import shutil
import tempfile
import unittest

from my_ai_agent.autonomous import AutonomousExecutionEngine, AutonomousTask, TaskStatus
from my_ai_agent.config import config
from my_ai_agent.intelligence import (
    AssembledContext,
    ContextBudget,
    ContextManager,
    DeterministicModel,
    ModelSafetyBoundary,
    PromptBuilder,
    ReasoningSummary,
    SchemaType,
    StructuredOutputValidator,
    router,
)


class TestE2EIntelligenceWorkflow(unittest.TestCase):
    """Verifies end-to-end autonomous coding driven through the intelligence layer."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="e2e_intel_"))
        config.set_workspace(self.test_dir)
        self.engine = AutonomousExecutionEngine(workspace_root=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_e2e_logout_feature_workflow(self):
        """E2E Scenario: 'Add a logout function to the authentication service.'"""
        # 1. Setup real project workspace
        auth_file = self.test_dir / "auth.py"
        auth_file.write_text(
            'class AuthService:\n'
            '    def __init__(self):\n'
            '        self.sessions = {}\n\n'
            '    def login(self, username, password):\n'
            '        if password == "secret":\n'
            '            self.sessions[username] = True\n'
            '            return True\n'
            '        return False\n',
            encoding="utf-8",
        )

        tests_dir = self.test_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        test_file = tests_dir / "test_auth.py"
        test_file.write_text(
            'import unittest\n'
            'import sys\n'
            'from pathlib import Path\n'
            'sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n'
            'from auth import AuthService\n\n'
            'class TestAuthService(unittest.TestCase):\n'
            '    def test_login(self):\n'
            '        auth = AuthService()\n'
            '        self.assertTrue(auth.login("alice", "secret"))\n\n'
            'if __name__ == "__main__":\n'
            '    unittest.main()\n',
            encoding="utf-8",
        )

        request = "Add a logout function to the authentication service."

        # 2. Intelligence Interface: Context Gathering & Prioritization
        context_mgr = ContextManager(workspace_root=self.test_dir)
        assembled_ctx: AssembledContext = context_mgr.assemble_context(user_request=request)

        # Verify context prioritized auth.py
        relevant_file_names = [f["path"] for f in assembled_ctx.relevant_files]
        self.assertIn("auth.py", relevant_file_names)

        # 3. Model Reasoning & Structured Decision
        active_model = router.route(request=request)
        self.assertEqual(active_model.model_id, "deterministic")

        # Generate structured decision
        decision_resp = active_model.generate_structured(
            prompt=f"Add logout function to {auth_file.name}",
            schema=SchemaType.ACTION_DECISION,
            context={"assembled": assembled_ctx},
        )
        self.assertIsNotNone(decision_resp.structured)

        # 4. Safety Boundary Enforcement
        safety_boundary = ModelSafetyBoundary(workspace_root=self.test_dir)
        is_safe, violation, safe_action = safety_boundary.sanitize_and_convert(decision_resp.structured)
        self.assertTrue(is_safe)
        self.assertIsNone(violation)
        self.assertIsNotNone(safe_action)
        self.assertEqual(safe_action.target, "auth.py")

        # 5. Safe User-Facing Reasoning Summary
        summary = ReasoningSummary.synthesize_from_context(
            objective=request,
            project_type="Python",
            identified_files=["auth.py"],
            identified_symbols=["AuthService", "login"],
            has_tests=True,
        )
        self.assertIn("auth.py", summary.to_user_message())

        # 6. Execute Autonomous Task through Engine
        task: AutonomousTask = self.engine.run_task(request, project_id="e2e_intel")

        # 7. Verification: Factual Task Outcome
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertGreater(len(task.completed_actions), 0)

        # 8. Verification: Real File Modification
        patched_code = auth_file.read_text(encoding="utf-8")
        self.assertIn("def logout", patched_code)
        self.assertIn("def login", patched_code)  # Preserved original login

        # 9. Verification: Tests passed
        test_obs = [o for o in task.observations if o.action_type.value == "RUN_TESTS"]
        self.assertGreater(len(test_obs), 0)
        self.assertTrue(test_obs[-1].success)
