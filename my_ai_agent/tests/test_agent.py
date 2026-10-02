"""Comprehensive Unit Test Suite for AI Agent Engine (Step 1 + Step 2)."""

import unittest
from pathlib import Path
import shutil
import tempfile
import sys

# Ensure package directory is on path
PACKAGE_DIR = Path(__file__).resolve().parent.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from config import config
from memory import AgentMemory
from tools import create_file, read_file, write_file, list_files, run_command, registry
from nlp import NaturalLanguageUnderstanding, IntentType, ExtractedEntities
from synthesizer import CodeSynthesizer
from planner import MultiStepPlanner, PlanState
from verifier import SelfVerifier
from agent import AgentEngine


class TestAgentEngine(unittest.TestCase):

    def setUp(self):
        # Create isolated temporary workspace and database
        self.test_dir = tempfile.mkdtemp()
        self.temp_workspace = Path(self.test_dir) / "workspace"
        self.temp_workspace.mkdir(parents=True, exist_ok=True)
        self.temp_db = Path(self.test_dir) / "test_memory.db"

        self.orig_workspace = config.WORKSPACE_DIR
        self.orig_db = config.DB_PATH
        config.WORKSPACE_DIR = self.temp_workspace
        config.DB_PATH = self.temp_db

        self.memory = AgentMemory(db_path=self.temp_db)
        self.agent = AgentEngine(tool_registry=registry, memory=self.memory, verbose=False)

    def tearDown(self):
        config.WORKSPACE_DIR = self.orig_workspace
        config.DB_PATH = self.orig_db
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. Tools Tests (Step 1 Foundation)
    # -------------------------------------------------------------
    def test_tools_operations(self):
        # create
        c_res = create_file("calc.py", "# empty")
        self.assertTrue(c_res.success)
        self.assertTrue((self.temp_workspace / "calc.py").exists())

        # write
        w_res = write_file("calc.py", "x = 10\nprint(x * 2)")
        self.assertTrue(w_res.success)

        # read
        r_res = read_file("calc.py")
        self.assertTrue(r_res.success)
        self.assertIn("print(x * 2)", r_res.output)

        # run
        cmd_res = run_command("python calc.py")
        self.assertTrue(cmd_res.success)
        self.assertEqual(cmd_res.output.strip(), "20")

        # list
        l_res = list_files()
        self.assertTrue(l_res.success)
        self.assertIn("calc.py", l_res.output)

    # -------------------------------------------------------------
    # 2. NLP: Intent Classification & Entity Extraction (Step 2)
    # -------------------------------------------------------------
    def test_nlp_entity_extraction_and_classification(self):
        nlu = NaturalLanguageUnderstanding()

        # Complex compound prompt
        query = "Can you make me a Python file that calculates the average of three numbers and then run it?"
        res = nlu.parse(query)

        self.assertEqual(res.intent, IntentType.BUILD_AND_RUN)
        self.assertEqual(res.entities.language, "python")
        self.assertEqual(res.entities.filename, "average_calculator.py")
        self.assertIn("average", res.entities.purpose.lower())
        self.assertEqual(res.entities.inputs, [10, 20, 30])
        self.assertTrue(res.entities.run_after)

    def test_nlp_atomic_variations(self):
        nlu = NaturalLanguageUnderstanding()

        # Atomic commands
        r1 = nlu.parse("Create test.py")
        self.assertEqual(r1.intent, IntentType.CREATE_FILE)
        self.assertEqual(r1.entities.filename, "test.py")

        r2 = nlu.parse('Write print("Hello") into test.py')
        self.assertEqual(r2.intent, IntentType.WRITE_FILE)
        self.assertEqual(r2.entities.filename, "test.py")
        self.assertEqual(r2.entities.raw_slots["content"], 'print("Hello")')

        r3 = nlu.parse("Run test.py")
        self.assertEqual(r3.intent, IntentType.RUN_COMMAND)

        r4 = nlu.parse("List files")
        self.assertEqual(r4.intent, IntentType.LIST_FILES)

    # -------------------------------------------------------------
    # 3. Code Synthesizer Tests (Step 2)
    # -------------------------------------------------------------
    def test_code_synthesizer(self):
        synth = CodeSynthesizer()

        ent = ExtractedEntities(
            language="python",
            purpose="calculate average of three numbers",
            inputs=[10, 20, 30],
        )
        code = synth.synthesize(ent)
        self.assertIn("calculate_average", code)
        self.assertIn("[10, 20, 30]", code)
        self.assertIn("def main():", code)

    # -------------------------------------------------------------
    # 4. Multi-Step Planner Tests (Step 2)
    # -------------------------------------------------------------
    def test_multi_step_planner(self):
        planner = MultiStepPlanner()
        nlu = NaturalLanguageUnderstanding()

        query = "Can you make me a Python file that calculates the average of three numbers and then run it?"
        nlu_res = nlu.parse(query)

        plan = planner.create_plan(nlu_res)
        self.assertEqual(len(plan.steps), 3)
        self.assertEqual(plan.steps[0].action_type, "CREATE_FILE")
        self.assertEqual(plan.steps[1].action_type, "WRITE_FILE")
        self.assertEqual(plan.steps[2].action_type, "RUN_COMMAND")

    # -------------------------------------------------------------
    # 5. Self-Verification & Error Diagnostics Tests (Step 2)
    # -------------------------------------------------------------
    def test_self_verifier(self):
        verifier = SelfVerifier()

        # Success case
        v1 = verifier.verify_execution("Calculated Average: 20.00", 0, purpose="average calculator")
        self.assertTrue(v1.passed)
        self.assertEqual(v1.status, "VERIFIED")

        # Failure case with SyntaxError
        err_output = "Traceback (most recent call last):\n  File 'test.py', line 1\n    x =\nSyntaxError: invalid syntax"
        v2 = verifier.verify_execution(err_output, 1)
        self.assertFalse(v2.passed)
        self.assertIn("Syntax error", v2.diagnostics)
        self.assertIsNotNone(v2.suggested_fix)

    # -------------------------------------------------------------
    # 6. End-to-End Cognitive Cycle for Step 2 Target Query
    # -------------------------------------------------------------
    def test_step2_compound_query_end_to_end(self):
        query = "Can you make me a Python file that calculates the average of three numbers and then run it?"
        response = self.agent.run_cycle(query)

        # Assert final synthesized response
        self.assertIn("Successfully built and executed 'average_calculator.py'", response)
        self.assertIn("Calculated Average: 20.00", response)
        self.assertIn("Self-Verification: ✅ PASSED", response)

        # Assert file was actually created in workspace
        target_file = self.temp_workspace / "average_calculator.py"
        self.assertTrue(target_file.exists())

        # Assert plan was persisted to SQLite
        plans = self.memory.get_recent_plans(1)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["intent"], "BUILD_AND_RUN")
        self.assertEqual(plans[0]["total_steps"], 3)
        self.assertEqual(plans[0]["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
