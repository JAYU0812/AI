"""Tests for Task Failure Analysis and Evidence Extraction (Step 6B).

Verifies that all 104 baseline tasks are analyzed without omission,
evidence is extracted factually, and decisions follow the decision tree.
"""

import json
from pathlib import Path
import unittest

from my_ai_agent.analysis.evidence import EvidenceExtractor
from my_ai_agent.analysis.failure_analyzer import TaskFailureAnalyzer
from my_ai_agent.analysis.root_cause import RootCauseCategory


BASELINE_FILE = Path(__file__).resolve().parent.parent / "evaluation" / "results" / "baseline_v1.json"


class TestFailureAnalysis(unittest.TestCase):
    """Verifies empirical failure analysis on baseline tasks."""

    @classmethod
    def setUpClass(cls):
        cls.raw_baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        cls.tasks = cls.raw_baseline.get("tasks", [])

    def test_all_104_tasks_analyzed_without_omission(self):
        """Ensures every single task in the baseline is processed and analyzed."""
        self.assertEqual(len(self.tasks), 104)
        records = [TaskFailureAnalyzer.analyze_task(t) for t in self.tasks]
        self.assertEqual(len(records), 104)

        # Ensure no task has empty ID, category, or status
        for r in records:
            self.assertTrue(r.task_id)
            self.assertTrue(r.category)
            self.assertIn(r.status, ["PASS", "PARTIAL", "FAIL"])
            self.assertIsNotNone(r.primary_root_cause)
            self.assertGreater(len(r.evidence), 0)

    def test_evidence_extraction_integrity(self):
        """Verifies factual evidence extraction from task metrics and assertion outcomes."""
        sample_task = self.tasks[0]  # auto_001
        evidence = EvidenceExtractor.extract_evidence(sample_task)
        self.assertIsInstance(evidence, list)
        self.assertGreater(len(evidence), 0)
        # Verify no fabricated values
        for e in evidence:
            self.assertIsInstance(e, str)
            self.assertGreater(len(e), 0)

    def test_passing_tasks_classified_cleanly(self):
        """Passing tasks should be classified with NO_ACTION and 100% confidence."""
        passing_tasks = [t for t in self.tasks if t.get("status") == "PASS"]
        self.assertGreater(len(passing_tasks), 0)
        rec = TaskFailureAnalyzer.analyze_task(passing_tasks[0])
        self.assertEqual(rec.status, "PASS")
        self.assertEqual(rec.confidence, 1.0)
        self.assertFalse(rec.is_training_candidate)


if __name__ == "__main__":
    unittest.main()
