"""End-to-End Tests for Step 3D Autonomous Coding & Debugging Engine.

Verifies:
1. Scenario 34 — Autonomous Coding:
   Temporary project with add() -> Request: "Add subtract, multiply and divide functions with tests."
   Agent understands, plans, modifies calculator.py, modifies tests, executes tests, verifies, completes.
2. Scenario 35 — Autonomous Debugging:
   Intentional bug in divide() returning a * b -> Request: "Fix the failing tests."
   Agent observes failure, diagnoses logic error, patches divide(), re-tests, completes with evidence.
3. Scenario 36 — Multi-Step Python CLI:
   "Build a small Python CLI that stores tasks in a JSON file and supports add, list and complete commands."
   Agent implements CLI and tests end-to-end, all tests pass.
4. Scenario 37 — Safe Failure Handling:
   Unresolvable failure stops safely within budget, does not fabricate success, reports evidence.
5. Scenario 38 — REST & Streaming API Endpoints:
   POST /api/tasks/run, GET /api/tasks/{id}, SSE streaming, GET /api/projects/{id}/tasks.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

from config import config
from memory import AgentMemory
from my_ai_agent.autonomous.events import broadcaster
from my_ai_agent.autonomous.execution_loop import AutonomousExecutionEngine
from my_ai_agent.autonomous.task import AutonomousTask
from my_ai_agent.autonomous.task_state import TaskStatus
from web_ide.server import app, handle_agent_turn


class TestE2EAutonomousCoding(unittest.TestCase):
    """End-to-end autonomous coding and debugging test suite."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="e2e_autonomous_"))
        config.set_workspace(self.test_dir)
        self.engine = AutonomousExecutionEngine(workspace_root=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_scenario_34_autonomous_coding_calculator(self):
        """Scenario 34: Autonomous feature expansion with unit tests."""
        # 1. Initial project on disk
        calc_file = self.test_dir / "calculator.py"
        calc_file.write_text(
            'def add(a, b):\n'
            '    return a + b\n',
            encoding="utf-8",
        )

        tests_dir = self.test_dir / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_calculator.py"
        test_file.write_text(
            'import unittest\n'
            'import sys\n'
            'from pathlib import Path\n'
            'sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n'
            'from calculator import add\n\n'
            'class TestCalculator(unittest.TestCase):\n'
            '    def test_add(self):\n'
            '        self.assertEqual(add(2, 3), 5)\n\n'
            'if __name__ == "__main__":\n'
            '    unittest.main()\n',
            encoding="utf-8",
        )

        # 2. Run Autonomous Task
        request = "Add subtract, multiply and divide functions with tests."
        task: AutonomousTask = self.engine.run_task(request, project_id="e2e_calc")

        # 3. Verify task outcome
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertGreater(len(task.completed_actions), 0)

        # 4. Verify calculator.py contains all requested functions
        code = calc_file.read_text(encoding="utf-8")
        self.assertIn("def add", code)
        self.assertIn("def subtract", code)
        self.assertIn("def multiply", code)
        self.assertIn("def divide", code)

        # 5. Verify test_calculator.py has tests for all functions
        test_code = test_file.read_text(encoding="utf-8")
        self.assertIn("test_subtract", test_code)
        self.assertIn("test_multiply", test_code)
        self.assertIn("test_divide", test_code)

        # 6. Verify tests were executed and passed
        test_observations = [o for o in task.observations if o.action_type.value == "RUN_TESTS"]
        self.assertGreater(len(test_observations), 0)
        self.assertTrue(test_observations[-1].success)
        self.assertEqual(test_observations[-1].exit_code, 0)

    def test_scenario_35_autonomous_debugging_bug_fix(self):
        """Scenario 35: Autonomous detection, diagnosis, repair, and re-test of an intentional bug."""
        # 1. Project with deliberate bug: divide() uses * instead of /
        calc_file = self.test_dir / "calculator.py"
        calc_file.write_text(
            'def divide(a, b):\n'
            '    return a * b\n',  # BUG: multiplication instead of division!
            encoding="utf-8",
        )

        tests_dir = self.test_dir / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_calculator.py"
        test_file.write_text(
            'import unittest\n'
            'import sys\n'
            'from pathlib import Path\n'
            'sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n'
            'from calculator import divide\n\n'
            'class TestCalculator(unittest.TestCase):\n'
            '    def test_divide(self):\n'
            '        self.assertEqual(divide(10, 2), 5)\n\n'
            'if __name__ == "__main__":\n'
            '    unittest.main()\n',
            encoding="utf-8",
        )

        # 2. Run Autonomous Debugging Task
        request = "Fix the failing tests."
        task: AutonomousTask = self.engine.run_task(request, project_id="e2e_debug")

        # 3. Verify task completed successfully
        self.assertEqual(task.status, TaskStatus.COMPLETED)

        # 4. Verify calculator.py was patched to use division
        repaired_code = calc_file.read_text(encoding="utf-8")
        self.assertIn("return a / b", repaired_code)
        self.assertNotIn("return a * b", repaired_code)

        # 5. Verify the diagnosis happened and repair succeeded
        state_history = [h["to_state"] for h in task.state_machine.history]
        self.assertIn("DIAGNOSING", state_history)
        self.assertIn("REPAIRING", state_history)
        self.assertIn("TESTING", state_history)

    def test_scenario_36_multi_step_python_cli(self):
        """Scenario 36: Build a small Python CLI task manager with JSON storage and tests."""
        request = "Build a small Python CLI that stores tasks in a JSON file and supports add, list and complete commands."
        task: AutonomousTask = self.engine.run_task(request, project_id="e2e_cli")

        self.assertEqual(task.status, TaskStatus.COMPLETED)

        # Verify files created on disk
        tasks_file = self.test_dir / "tasks.py"
        self.assertTrue(tasks_file.exists(), "tasks.py must be created")
        tasks_code = tasks_file.read_text(encoding="utf-8")
        self.assertIn("def add_task", tasks_code)
        self.assertIn("def list_tasks", tasks_code)
        self.assertIn("def complete_task", tasks_code)

        # Verify tests exist and pass
        test_file = self.test_dir / "tests" / "test_tasks.py"
        self.assertTrue(test_file.exists(), "test_tasks.py must be created")

        test_observations = [o for o in task.observations if o.action_type.value == "RUN_TESTS"]
        self.assertGreater(len(test_observations), 0)
        self.assertTrue(test_observations[-1].success)

    def test_scenario_37_failure_without_dummy_data(self):
        """Scenario 37: Unresolvable failure stops safely within budget without fabricating success."""
        # Project with syntax error that cannot be resolved automatically
        broken_file = self.test_dir / "broken.py"
        broken_file.write_text("def impossible(x)))): return ??\n", encoding="utf-8")

        tests_dir = self.test_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_broken.py").write_text(
            "import broken\n",
            encoding="utf-8",
        )

        request = "Fix the failing tests."
        task: AutonomousTask = self.engine.run_task(request, project_id="e2e_fail")

        # Must report FAILED or PARTIALLY_COMPLETED, NEVER false COMPLETED
        self.assertIn(task.status, (TaskStatus.FAILED, TaskStatus.PARTIALLY_COMPLETED))
        self.assertNotEqual(task.status, TaskStatus.COMPLETED)
        self.assertTrue(len(task.final_summary) > 0)

    def test_scenario_38_api_endpoints_and_streaming(self):
        """Scenario 38: Verify REST and SSE streaming endpoints."""
        from fastapi.testclient import TestClient
        client = TestClient(app)

        # 1. Start autonomous task via POST /api/tasks/run
        resp = client.post(
            "/api/tasks/run",
            json={"message": "Add subtract, multiply and divide functions with tests.", "project_id": "test_api_proj"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        task_id = data["task_id"]
        self.assertTrue(task_id)

        # 2. Get task state via GET /api/tasks/{task_id}
        resp_get = client.get(f"/api/tasks/{task_id}")
        self.assertEqual(resp_get.status_code, 200)
        self.assertEqual(resp_get.json()["task_id"], task_id)

        # 3. Get project tasks via GET /api/projects/{project_id}/tasks
        resp_list = client.get("/api/projects/test_api_proj/tasks")
        self.assertEqual(resp_list.status_code, 200)
        self.assertGreater(resp_list.json()["count"], 0)

        # 4. SSE streaming endpoint GET /api/tasks/{task_id}/events
        events = broadcaster.get_events(task_id)
        self.assertGreater(len(events), 0)
        event_types = [e.event_type.value for e in events]
        self.assertIn("TASK_STARTED", event_types)


if __name__ == "__main__":
    unittest.main()
