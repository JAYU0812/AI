"""Tests for Evaluation Dataset and Task Specifications (Step 6A).

Verifies task schema validation, duplicate ID detection, dataset splitting,
secret auditing, and dataset filtering.
"""

from pathlib import Path
import tempfile
import unittest

from my_ai_agent.evaluation.assertions import FileExistsAssertion
from my_ai_agent.evaluation.categories import DatasetSplit, Difficulty, TaskCategory
from my_ai_agent.evaluation.dataset import BenchmarkDataset, DatasetQualityAuditor
from my_ai_agent.evaluation.task import EvaluationTask


class TestDatasetAndTasks(unittest.TestCase):
    """Verifies dataset integrity, schema validation, and auditing."""

    def test_task_schema_validation(self):
        """Validates that valid tasks pass and invalid tasks report errors."""
        valid_task = EvaluationTask(
            id="test_001",
            category=TaskCategory.CODE_GENERATION,
            difficulty=Difficulty.EASY,
            prompt="Write a function.",
            assertions=[FileExistsAssertion(file="main.py")],
        )
        self.assertEqual(valid_task.validate(), [])

        invalid_task = EvaluationTask(
            id="",
            category=TaskCategory.CODE_GENERATION,
            difficulty=Difficulty.EASY,
            prompt="",
            assertions=[],
        )
        errors = invalid_task.validate()
        self.assertGreater(len(errors), 0)

    def test_duplicate_task_id_detection(self):
        """Verifies that DatasetQualityAuditor catches duplicate task IDs."""
        task1 = EvaluationTask(
            id="dup_001",
            category=TaskCategory.GENERAL_UNDERSTANDING,
            difficulty=Difficulty.EASY,
            prompt="Explain project.",
            assertions=[FileExistsAssertion(file="main.py")],
        )
        task2 = EvaluationTask(
            id="dup_001",
            category=TaskCategory.INTENT_UNDERSTANDING,
            difficulty=Difficulty.MEDIUM,
            prompt="Another prompt.",
            assertions=[FileExistsAssertion(file="main.py")],
        )
        res = DatasetQualityAuditor.audit_dataset([task1, task2])
        self.assertFalse(res["valid"])
        self.assertIn("dup_001", res["issues"])

    def test_secret_detection_in_prompts(self):
        """Verifies that prompts containing accidental secrets are rejected."""
        leaky_task = EvaluationTask(
            id="leak_001",
            category=TaskCategory.CODE_GENERATION,
            difficulty=Difficulty.EASY,
            prompt="Use API key sk-1234567890abcdef to authenticate.",
            assertions=[FileExistsAssertion(file="main.py")],
        )
        res = DatasetQualityAuditor.audit_dataset([leaky_task])
        self.assertFalse(res["valid"])
        self.assertTrue(any("secret" in err.lower() for err in res["issues"]["leak_001"]))

    def test_dataset_v1_loading_and_filtering(self):
        """Tests loading official v1 dataset and filtering by attributes."""
        dataset = BenchmarkDataset(version="v1")
        self.assertGreater(len(dataset.tasks), 60, "Dataset v1 should contain at least 60 tasks")

        # Category filter
        safety_tasks = dataset.filter(category=TaskCategory.SAFETY)
        self.assertGreaterEqual(len(safety_tasks), 10, "Should have at least 10 safety tasks")
        for t in safety_tasks:
            self.assertEqual(t.category, TaskCategory.SAFETY)

        # Difficulty filter
        easy_tasks = dataset.filter(difficulty=Difficulty.EASY)
        self.assertGreater(len(easy_tasks), 0)
        for t in easy_tasks:
            self.assertEqual(t.difficulty, Difficulty.EASY)

        # Split filter
        dev_tasks = dataset.filter(split=DatasetSplit.DEVELOPMENT)
        self.assertGreater(len(dev_tasks), 0)
        for t in dev_tasks:
            self.assertEqual(t.split, DatasetSplit.DEVELOPMENT)

    def test_dataset_quality_audit_on_v1(self):
        """Verifies that official Dataset v1 passes 100% of quality audits."""
        dataset = BenchmarkDataset(version="v1")
        audit_res = dataset.audit()
        self.assertTrue(audit_res["valid"], f"Dataset v1 audit failed: {audit_res['issues']}")
        self.assertEqual(len(audit_res["issues"]), 0)


if __name__ == "__main__":
    unittest.main()
