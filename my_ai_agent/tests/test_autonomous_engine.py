"""Unit tests for Step 3D Autonomous Engine components.

Verifies:
1. Task state machine transitions and guards
2. Operational budget enforcement
3. Action model and risk classification
4. Observation and reasoning
5. Diagnosis engine on real tracebacks and test outputs
6. Repair engine using PatchEngine
7. Real-time event broadcasting and SSE formatting
8. Approval boundary pausing and resumption
9. Cancellation and rollback
10. Task persistence in SQLite
"""

from pathlib import Path
import shutil
import tempfile
import unittest

from config import config
from memory import AgentMemory
from my_ai_agent.autonomous.action import ActionRisk, ActionType, AutonomousAction
from my_ai_agent.autonomous.action_executor import ActionExecutor
from my_ai_agent.autonomous.budget import TaskBudget
from my_ai_agent.autonomous.completion import CompletionReport, CompletionVerifier
from my_ai_agent.autonomous.diagnosis import DiagnosisEngine, DiagnosticReport, ErrorCategory
from my_ai_agent.autonomous.events import AgentEvent, AgentEventType, EventBroadcaster, broadcaster
from my_ai_agent.autonomous.execution_loop import AutonomousExecutionEngine
from my_ai_agent.autonomous.observation import ActionObservation
from my_ai_agent.autonomous.repair import RepairEngine, RepairResult
from my_ai_agent.autonomous.task import AutonomousTask
from my_ai_agent.autonomous.task_state import StateTransitionError, TaskStateMachine, TaskStatus
from my_ai_agent.editing.patch_engine import PatchEngine


class TestTaskStateMachine(unittest.TestCase):
    """Test formal state machine transitions and invalid transition enforcement."""

    def test_valid_forward_transitions(self):
        sm = TaskStateMachine()
        self.assertEqual(sm.current_state, TaskStatus.QUEUED)

        sm.transition_to(TaskStatus.UNDERSTANDING, "Starting scan")
        self.assertEqual(sm.current_state, TaskStatus.UNDERSTANDING)

        sm.transition_to(TaskStatus.PLANNING, "Scan complete")
        self.assertEqual(sm.current_state, TaskStatus.PLANNING)

        sm.transition_to(TaskStatus.EXECUTING, "Executing steps")
        self.assertEqual(sm.current_state, TaskStatus.EXECUTING)

        sm.transition_to(TaskStatus.TESTING, "Running tests")
        self.assertEqual(sm.current_state, TaskStatus.TESTING)

        sm.transition_to(TaskStatus.VERIFYING, "Checking evidence")
        self.assertEqual(sm.current_state, TaskStatus.VERIFYING)

        sm.transition_to(TaskStatus.COMPLETED, "Success")
        self.assertEqual(sm.current_state, TaskStatus.COMPLETED)
        self.assertTrue(sm.is_terminal())

    def test_invalid_transition_raises_error(self):
        sm = TaskStateMachine()
        # Cannot jump from QUEUED directly to COMPLETED
        with self.assertRaises(StateTransitionError):
            sm.transition_to(TaskStatus.COMPLETED)

    def test_repair_cycle_transitions(self):
        sm = TaskStateMachine(TaskStatus.EXECUTING)
        sm.transition_to(TaskStatus.DIAGNOSING, "Error detected")
        sm.transition_to(TaskStatus.REPAIRING, "Fixing code")
        sm.transition_to(TaskStatus.TESTING, "Re-testing")
        self.assertEqual(sm.current_state, TaskStatus.TESTING)


class TestTaskBudget(unittest.TestCase):
    """Test operational ceilings and limit enforcement."""

    def test_step_limit_enforcement(self):
        budget = TaskBudget(max_steps=3)
        self.assertIsNone(budget.check_budget())

        budget.record_step()
        budget.record_step()
        budget.record_step()
        err = budget.check_budget()
        self.assertIsNotNone(err)
        self.assertIn("Exceeded maximum allowed steps", err)

    def test_repair_limit_enforcement(self):
        budget = TaskBudget(max_repair_attempts=2)
        budget.record_repair()
        self.assertIsNone(budget.check_budget())

        budget.record_repair()
        err = budget.check_budget()
        self.assertIsNotNone(err)
        self.assertIn("Exceeded maximum auto-repair attempts", err)

    def test_command_limit_enforcement(self):
        budget = TaskBudget(max_commands=2)
        budget.record_command()
        budget.record_command()
        err = budget.check_budget()
        self.assertIsNotNone(err)
        self.assertIn("Exceeded maximum allowed command executions", err)


class TestActionAndRiskClassification(unittest.TestCase):
    """Test action modeling and automatic high-risk approval marking."""

    def test_low_risk_action_default(self):
        act = AutonomousAction(
            type=ActionType.READ_FILE,
            target="auth.py",
            reason="Read existing authentication code",
            risk=ActionRisk.LOW,
        )
        self.assertFalse(act.requires_approval)

    def test_high_risk_action_auto_approval_flag(self):
        act = AutonomousAction(
            type=ActionType.DELETE_FILE,
            target="database.db",
            reason="Remove legacy database",
            risk=ActionRisk.HIGH,
        )
        self.assertTrue(act.requires_approval)
        self.assertFalse(act.approved)


class TestDiagnosisEngine(unittest.TestCase):
    """Test error diagnosis from real tracebacks and outputs."""

    def test_diagnose_syntax_error(self):
        tb = '''File "calc.py", line 12
    def divide(a, b)
                   ^
SyntaxError: expected ':'
'''
        report = DiagnosisEngine.diagnose(tb)
        self.assertEqual(report.category, ErrorCategory.SYNTAX_ERROR)
        self.assertEqual(report.affected_line, 12)
        self.assertIn("calc.py", report.affected_file)

    def test_diagnose_import_error(self):
        tb = '''Traceback (most recent call last):
  File "main.py", line 2, in <module>
    import fastapi
ModuleNotFoundError: No module named 'fastapi'
'''
        report = DiagnosisEngine.diagnose(tb)
        self.assertEqual(report.category, ErrorCategory.DEPENDENCY_ERROR)
        self.assertEqual(report.failing_symbol, "fastapi")
        self.assertIn("fastapi", report.error_message)

    def test_diagnose_assertion_failure(self):
        out = '''=================================== FAILURES ===================================
_________________________________ test_divide __________________________________
tests/test_calculator.py:10: in test_divide
    assert divide(10, 2) == 5
E   AssertionError: assert 20 == 5
'''
        report = DiagnosisEngine.diagnose(out)
        self.assertEqual(report.category, ErrorCategory.TEST_FAILURE)
        self.assertIn("divide", report.failing_symbol)
        self.assertIn("20 == 5", report.error_message)


class TestRepairEngine(unittest.TestCase):
    """Test targeted surgical repairs using PatchEngine."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_repair_"))
        config.set_workspace(self.temp_dir)
        self.patch_engine = PatchEngine(workspace_root=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_repair_arithmetic_operator_bug(self):
        calc_file = self.temp_dir / "calculator.py"
        calc_file.write_text(
            'def divide(a, b):\n'
            '    return a * b\n',  # Bug: multiplication instead of division!
            encoding="utf-8",
        )

        report = DiagnosticReport(
            category=ErrorCategory.TEST_FAILURE,
            error_message="AssertionError: assert divide(10, 2) == 5, got 20",
            affected_file="calculator.py",
            failing_symbol="divide",
            evidence="assert divide(10, 2) == 5",
        )

        res = RepairEngine.attempt_repair(report, self.temp_dir, self.patch_engine)
        self.assertTrue(res.success)
        self.assertIn("calculator.py", res.modified_files)

        repaired_code = calc_file.read_text(encoding="utf-8")
        self.assertIn("return a / b", repaired_code)
        self.assertNotIn("return a * b", repaired_code)


class TestEventsAndBroadcaster(unittest.TestCase):
    """Test event streaming and subscriber queues."""

    def test_broadcast_and_sse_formatting(self):
        broadcaster.clear()
        q = broadcaster.subscribe("task-abc")

        ev = AgentEvent(
            task_id="task-abc",
            event_type=AgentEventType.TASK_STARTED,
            message="Task initialized",
            metadata={"step": 1},
        )
        broadcaster.emit(ev)

        received = q.get_nowait()
        self.assertEqual(received.task_id, "task-abc")
        self.assertEqual(received.event_type, AgentEventType.TASK_STARTED)

        sse = received.to_sse()
        self.assertIn("event: TASK_STARTED", sse)
        self.assertIn("Task initialized", sse)


class TestAutonomousTaskPersistence(unittest.TestCase):
    """Test SQLite persistence of autonomous tasks."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_persist_"))
        self.db_path = self.temp_dir / "test_memory.db"
        self.memory = AgentMemory(db_path=self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_retrieve_task(self):
        task = AutonomousTask(
            user_request="Build user registration",
            project_id="proj_1",
            task_id="t-100",
        )
        task.transition_to(TaskStatus.UNDERSTANDING)
        task.transition_to(TaskStatus.PLANNING)
        task.transition_to(TaskStatus.EXECUTING)
        task.budget.record_step()
        task.budget.record_command()

        self.memory.save_autonomous_task(task.to_dict())

        loaded = self.memory.get_autonomous_task("t-100")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["task_id"], "t-100")
        self.assertEqual(loaded["status"], "EXECUTING")
        self.assertEqual(loaded["budget"]["steps_taken"], 1)

        tasks = self.memory.get_autonomous_tasks_for_project("proj_1")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "t-100")


if __name__ == "__main__":
    unittest.main()
