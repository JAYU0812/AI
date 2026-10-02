"""Tests for Root-Cause Classification, Categories, and Pattern Detection (Step 6B).

Verifies root cause assignments, category aggregation metrics, difficulty analysis,
and failure clustering logic.
"""

import json
from pathlib import Path
import unittest

from my_ai_agent.analysis.category_analyzer import CategoryAnalyzer
from my_ai_agent.analysis.failure_analyzer import TaskFailureAnalyzer
from my_ai_agent.analysis.pattern_detector import PatternDetector
from my_ai_agent.analysis.root_cause import RootCauseCategory


BASELINE_FILE = Path(__file__).resolve().parent.parent / "evaluation" / "results" / "baseline_v1.json"


class TestRootCauseAndPatterns(unittest.TestCase):
    """Verifies root cause taxonomy, category diagnostics, and clustering."""

    @classmethod
    def setUpClass(cls):
        raw_baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        cls.tasks = raw_baseline.get("tasks", [])
        cls.records = [TaskFailureAnalyzer.analyze_task(t) for t in cls.tasks]

    def test_root_cause_categories_assigned(self):
        """Ensures all non-passing tasks are assigned a specific root-cause category."""
        non_passing = [r for r in self.records if r.status != "PASS"]
        self.assertEqual(len(non_passing), 54)  # 23 partial + 31 failed = 54
        for r in non_passing:
            self.assertIn(
                r.primary_root_cause,
                [
                    RootCauseCategory.MODEL_CAPABILITY,
                    RootCauseCategory.STRUCTURED_OUTPUT_INTERFACE,
                    RootCauseCategory.PROMPT_DESIGN,
                    RootCauseCategory.AUTONOMOUS_ENGINE,
                    RootCauseCategory.EVALUATOR_ERROR,
                    RootCauseCategory.UNKNOWN,
                ],
            )
            self.assertGreater(r.confidence, 0.0)

    def test_category_aggregation(self):
        """Verifies CategoryAnalyzer aggregates all 14 categories with correct counts."""
        cats = CategoryAnalyzer.analyze_categories(self.records)
        self.assertEqual(len(cats), 14)
        for cat_name, info in cats.items():
            self.assertGreater(info["total_tasks"], 0)
            self.assertEqual(info["passed"] + info["partial"] + info["failed"], info["total_tasks"])
            self.assertGreaterEqual(info["average_score"], 0.0)
            self.assertLessEqual(info["average_score"], 100.0)
            self.assertTrue(info["primary_root_cause"])

    def test_difficulty_pattern_analysis(self):
        """Verifies difficulty breakdown for Easy, Medium, and Hard."""
        diff_res = PatternDetector.analyze_by_difficulty(self.records)
        self.assertIn("easy", diff_res)
        self.assertIn("medium", diff_res)
        self.assertIn("hard", diff_res)
        total_diff_tasks = sum(d["total_tasks"] for d in diff_res.values())
        self.assertEqual(total_diff_tasks, 104)

    def test_failure_clustering(self):
        """Verifies that non-passing tasks are grouped into prioritized clusters."""
        clusters = PatternDetector.cluster_failures(self.records)
        self.assertGreater(len(clusters), 0)
        for c in clusters:
            self.assertTrue(c.cluster_id)
            self.assertTrue(c.name)
            self.assertGreater(len(c.task_ids), 0)
            self.assertIn(c.priority, ["P0", "P1", "P2", "P3"])


if __name__ == "__main__":
    unittest.main()
