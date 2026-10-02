"""Step 6D Tests: Model comparison, evaluation scoring, experiment result determination."""

import json
import unittest
from unittest.mock import MagicMock


class TestOutputScoring(unittest.TestCase):
    """Tests code and JSON output scoring functions."""

    def test_valid_function_scores_high(self):
        from my_ai_agent.fine_tuning.evaluator import score_code_output
        code = 'def multiply(a: int, b: int) -> int:\n    """Multiplies."""\n    return a * b\n'
        result = score_code_output(code, "code_generation", ["def multiply"])
        self.assertGreaterEqual(result["score"], 0.7)

    def test_empty_output_scores_zero(self):
        from my_ai_agent.fine_tuning.evaluator import score_code_output
        result = score_code_output("", "code_generation", ["def f"])
        self.assertEqual(result["score"], 0.0)

    def test_syntax_error_reduces_score(self):
        from my_ai_agent.fine_tuning.evaluator import score_code_output
        result = score_code_output("def broken(:\n    pass", "code_generation", ["def broken"])
        self.assertLess(result["score"], 0.5)

    def test_valid_json_scores_high(self):
        from my_ai_agent.fine_tuning.evaluator import score_json_output
        output = '{"action": "INSPECT_PROJECT", "target": ".", "reason": "check", "risk": "LOW"}'
        result = score_json_output(output)
        self.assertGreaterEqual(result["score"], 0.8)

    def test_json_with_markdown_wrapper_scores_zero(self):
        from my_ai_agent.fine_tuning.evaluator import score_json_output
        output = '```json\n{"action": "FINISH", "target": ""}\n```'
        result = score_json_output(output)
        self.assertEqual(result["score"], 0.0)

    def test_json_with_preamble_scores_zero(self):
        from my_ai_agent.fine_tuning.evaluator import score_json_output
        output = 'Sure! Here is the JSON:\n{"action": "FINISH", "target": ""}'
        result = score_json_output(output)
        self.assertEqual(result["score"], 0.0)

    def test_invalid_json_scores_zero(self):
        from my_ai_agent.fine_tuning.evaluator import score_json_output
        result = score_json_output("{bad json}")
        self.assertEqual(result["score"], 0.0)

    def test_coherent_response_scores_well(self):
        from my_ai_agent.fine_tuning.evaluator import score_conversational_output
        output = "I'm an AI assistant and I can help you with coding tasks, explain concepts, and answer questions."
        result = score_conversational_output(output)
        self.assertTrue(result["coherent"])
        self.assertGreaterEqual(result["score"], 0.5)

    def test_empty_response_is_incoherent(self):
        from my_ai_agent.fine_tuning.evaluator import score_conversational_output
        result = score_conversational_output("")
        self.assertFalse(result["coherent"])
        self.assertEqual(result["score"], 0.0)


class TestStep6CTestEvaluator(unittest.TestCase):
    """Tests the Step 6C test split evaluator."""

    def _make_test_records(self):
        return [
            {
                "id": "test_001",
                "task_type": "code_generation",
                "instruction": "Implement multiply(a, b).",
                "target_output": "def multiply(a, b):\n    return a * b\n",
                "output_type": "code",
            },
            {
                "id": "test_002",
                "task_type": "structured_output",
                "instruction": "Generate action decision.",
                "target_output": '{"action": "FINISH", "target": "", "reason": "done", "risk": "LOW"}',
                "output_type": "json",
            },
        ]

    def test_evaluator_returns_correct_structure(self):
        from my_ai_agent.fine_tuning.evaluator import Step6CTestEvaluator
        evaluator = Step6CTestEvaluator()

        def mock_inference(instruction):
            return {"output": "def multiply(a, b):\n    return a * b\n"}

        records = self._make_test_records()
        result = evaluator.evaluate_on_test_split(mock_inference, records, label="test")

        self.assertIn("label", result)
        self.assertIn("total_examples", result)
        self.assertIn("average_score", result)
        self.assertIn("results", result)
        self.assertEqual(result["total_examples"], 2)

    def test_good_code_output_gets_high_score(self):
        from my_ai_agent.fine_tuning.evaluator import Step6CTestEvaluator
        evaluator = Step6CTestEvaluator()
        records = [self._make_test_records()[0]]

        def mock_good(instruction):
            return {"output": "def multiply(a, b):\n    return a * b\n"}

        result = evaluator.evaluate_on_test_split(mock_good, records)
        self.assertGreater(result["average_score"], 0.5)

    def test_bad_output_gets_low_score(self):
        from my_ai_agent.fine_tuning.evaluator import Step6CTestEvaluator
        evaluator = Step6CTestEvaluator()
        records = [self._make_test_records()[0]]

        def mock_bad(instruction):
            return {"output": ""}

        result = evaluator.evaluate_on_test_split(mock_bad, records)
        self.assertEqual(result["average_score"], 0.0)

    def test_generalization_probes_execute(self):
        from my_ai_agent.fine_tuning.evaluator import Step6CTestEvaluator, GENERALIZATION_PROBES
        evaluator = Step6CTestEvaluator()

        def mock_inference(instruction):
            return {"output": "def is_palindrome(s): return s == s[::-1]\n"}

        result = evaluator.run_generalization_probes(mock_inference)
        self.assertEqual(result["total_probes"], len(GENERALIZATION_PROBES))
        self.assertGreaterEqual(result["average_score"], 0.0)

    def test_casual_conversation_probes_execute(self):
        from my_ai_agent.fine_tuning.evaluator import Step6CTestEvaluator, CASUAL_CONVERSATION_PROBES
        evaluator = Step6CTestEvaluator()

        def mock_inference(instruction):
            return {"output": "I'm a helpful coding assistant. I can help you with Python and more."}

        result = evaluator.run_casual_conversation(mock_inference)
        self.assertEqual(result["total_probes"], len(CASUAL_CONVERSATION_PROBES))
        self.assertIn("verdict", result)
        self.assertIn("coherence_rate", result)


class TestComparisonAnalysis(unittest.TestCase):
    """Tests the comparison and result determination logic."""

    def test_task_comparison_correctly_classifies(self):
        from my_ai_agent.fine_tuning.comparison import compare_task_results
        baseline = [
            {"id": "t1", "passed": True, "score": 1.0},
            {"id": "t2", "passed": False, "score": 0.0},
            {"id": "t3", "passed": True, "score": 1.0},
        ]
        finetuned = [
            {"id": "t1", "passed": True, "score": 1.0},   # PASS→PASS
            {"id": "t2", "passed": True, "score": 1.0},   # FAIL→PASS (improvement)
            {"id": "t3", "passed": False, "score": 0.0},  # PASS→FAIL (regression)
        ]
        result = compare_task_results(baseline, finetuned)
        self.assertEqual(result["improvement_count"], 1)
        self.assertEqual(result["regression_count"], 1)
        self.assertEqual(result["unchanged_pass_count"], 1)

    def test_category_comparison_calculates_delta(self):
        from my_ai_agent.fine_tuning.comparison import build_category_comparison
        baseline = {"Code Generation": 36.5, "Debugging": 93.8}
        finetuned = {"Code Generation": 55.0, "Debugging": 90.0}
        rows = build_category_comparison(baseline, finetuned)
        code_gen = next(r for r in rows if r["category"] == "Code Generation")
        debugging = next(r for r in rows if r["category"] == "Debugging")
        self.assertAlmostEqual(code_gen["delta"], 18.5, places=1)
        self.assertAlmostEqual(debugging["delta"], -3.8, places=1)
        self.assertEqual(code_gen["direction"], "improved")
        self.assertEqual(debugging["direction"], "regressed")

    def test_success_result_when_weakness_improved(self):
        from my_ai_agent.fine_tuning.comparison import determine_experiment_result
        weakness_analysis = {
            "meaningful_improvement": True,
            "weakness_improved_count": 4,
            "weakness_total": 5,
            "regression_count": 0,
        }
        safety_cmp = {"experiment_fail": False}
        result = determine_experiment_result(62.5, 68.0, weakness_analysis, safety_cmp)
        self.assertIn(result["result"], ["SUCCESS", "PARTIAL_SUCCESS"])

    def test_regression_result_when_safety_fails(self):
        from my_ai_agent.fine_tuning.comparison import determine_experiment_result
        weakness_analysis = {
            "meaningful_improvement": True,
            "weakness_improved_count": 5,
            "weakness_total": 5,
            "regression_count": 0,
        }
        safety_cmp = {"experiment_fail": True, "verdict": "SAFETY REGRESSION"}
        result = determine_experiment_result(62.5, 68.0, weakness_analysis, safety_cmp)
        self.assertEqual(result["result"], "REGRESSION")
        self.assertIn("SAFETY", result["verdict"])

    def test_no_improvement_result_when_no_change(self):
        from my_ai_agent.fine_tuning.comparison import determine_experiment_result
        weakness_analysis = {
            "meaningful_improvement": False,
            "weakness_improved_count": 0,
            "weakness_total": 5,
            "regression_count": 0,
        }
        safety_cmp = {"experiment_fail": False}
        result = determine_experiment_result(62.5, 62.5, weakness_analysis, safety_cmp)
        self.assertIn(result["result"], ["NO_SIGNIFICANT_IMPROVEMENT", "PARTIAL_SUCCESS"])

    def test_weakness_analysis_identifies_improvements(self):
        from my_ai_agent.fine_tuning.comparison import analyze_weakness_improvement
        baseline = {
            "Intent Understanding": 16.7,
            "Structured Output": 16.7,
            "Code Generation": 36.5,
            "Debugging": 93.8,
            "Safety": 65.8,
        }
        finetuned = {
            "Intent Understanding": 40.0,   # +23.3 improved
            "Structured Output": 60.0,       # +43.3 improved
            "Code Generation": 50.0,         # +13.5 improved
            "Debugging": 90.0,               # -3.8 mild drop (within tolerance)
            "Safety": 65.0,                  # ~unchanged
        }
        analysis = analyze_weakness_improvement(baseline, finetuned)
        self.assertGreater(analysis["weakness_improved_count"], 0)

    def test_weakness_analysis_detects_regression(self):
        from my_ai_agent.fine_tuning.comparison import analyze_weakness_improvement
        baseline = {"Debugging": 93.8, "Verification": 91.7}
        finetuned = {"Debugging": 50.0, "Verification": 40.0}  # Severe regression
        analysis = analyze_weakness_improvement(baseline, finetuned)
        self.assertGreater(analysis["regression_count"], 0)


class TestBaselineResults(unittest.TestCase):
    """Tests that baseline constants match the pre-recorded Step 6A results."""

    def test_baseline_overall_score_is_62_5(self):
        from my_ai_agent.fine_tuning.comparison import BASELINE_6A_RESULTS
        self.assertAlmostEqual(BASELINE_6A_RESULTS["overall_score"], 62.5)

    def test_baseline_categories_present(self):
        from my_ai_agent.fine_tuning.comparison import BASELINE_6A_RESULTS
        expected_cats = [
            "Debugging", "Project Understanding", "Verification",
            "Code Generation", "Structured Output", "Tool Selection",
            "Intent Understanding", "Instruction Following",
        ]
        for cat in expected_cats:
            self.assertIn(cat, BASELINE_6A_RESULTS["categories"])

    def test_baseline_known_weak_categories(self):
        from my_ai_agent.fine_tuning.comparison import BASELINE_6A_RESULTS
        cats = BASELINE_6A_RESULTS["categories"]
        # These should be the lowest-scoring (Step 6B identified)
        self.assertAlmostEqual(cats["Intent Understanding"], 16.7, places=0)
        self.assertAlmostEqual(cats["Structured Output"], 16.7, places=0)
        self.assertAlmostEqual(cats["Tool Selection"], 16.7, places=0)
        self.assertLess(cats["Code Generation"], 50.0)


if __name__ == "__main__":
    unittest.main()
