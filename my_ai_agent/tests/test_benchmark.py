"""Tests for Benchmark Runner, Reports, and Baseline Comparisons (Step 6A).

Verifies multi-task execution, JSON serialization, Markdown report formatting,
and regression comparison logic.
"""

from pathlib import Path
import tempfile
import unittest

from my_ai_agent.evaluation.assertions import FileExistsAssertion
from my_ai_agent.evaluation.baseline import BaselineManager
from my_ai_agent.evaluation.categories import DatasetSplit, Difficulty, TaskCategory
from my_ai_agent.evaluation.dataset import BenchmarkDataset
from my_ai_agent.evaluation.report import BenchmarkReportGenerator
from my_ai_agent.evaluation.runner import BenchmarkRunner
from my_ai_agent.evaluation.task import EvaluationTask
from my_ai_agent.intelligence import DeterministicModel


class TestBenchmarkEngine(unittest.TestCase):
    """Verifies benchmark runner execution, metrics, and reporting."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name).resolve()
        self.dataset = BenchmarkDataset(version="v1")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_benchmark_runner_single_category(self):
        """Runs benchmark against a small category subset (safety) to verify metrics."""
        runner = BenchmarkRunner(dataset=self.dataset)
        model = DeterministicModel()

        res = runner.run_benchmark(
            model=model,
            category=TaskCategory.SAFETY,
            runs=1,
            verbose=False,
        )

        self.assertNotIn("error", res)
        self.assertGreater(res["total_tasks"], 0)
        self.assertGreaterEqual(res["overall_score_percentage"], 0.0)
        self.assertIn("categories", res)
        self.assertEqual(res["safety_violations"], 0)

        # Verify JSON saving
        out_json = self.test_root / "res.json"
        saved = runner.save_result(res, output_path=out_json)
        self.assertTrue(saved.exists())

    def test_markdown_report_generation(self):
        """Tests that Markdown report renders all sections properly."""
        sample_result = {
            "dataset_version": "v1",
            "model_id": "deterministic",
            "timestamp": "2026-10-01 12:00:00",
            "total_tasks": 10,
            "passed": 8,
            "partial": 1,
            "failed": 1,
            "safety_violations": 0,
            "overall_score_percentage": 85.0,
            "total_latency_seconds": 12.4,
            "categories": {
                "safety": {"count": 5, "average_score": 100.0},
                "code_generation": {"count": 5, "average_score": 70.0},
            },
            "failure_distribution": {
                "test_failure": 1,
            },
            "tasks": [],
        }

        md_text = BenchmarkReportGenerator.generate_report(sample_result)
        self.assertIn("Agent Intelligence Benchmark Report", md_text)
        self.assertIn("85.0%", md_text)
        self.assertIn("Executive Summary", md_text)
        self.assertIn("Capability Category Scores", md_text)
        self.assertIn("Safety Evaluation", md_text)

        # Save report
        out_md = self.test_root / "report.md"
        saved = BenchmarkReportGenerator.save_markdown_report(md_text, filename=str(out_md))
        self.assertTrue(saved.exists())

    def test_baseline_delta_comparison(self):
        """Tests baseline comparison logic between baseline and candidate models."""
        base_res = {
            "model_id": "baseline_qwen",
            "overall_score_percentage": 65.0,
            "categories": {
                "code_generation": {"average_score": 60.0},
                "debugging": {"average_score": 50.0},
            },
        }
        cand_res = {
            "model_id": "finetuned_candidate",
            "overall_score_percentage": 78.0,
            "categories": {
                "code_generation": {"average_score": 80.0},
                "debugging": {"average_score": 70.0},
            },
        }

        delta = BaselineManager.compare(cand_res, base_res)
        self.assertTrue(delta["is_improvement"])
        self.assertEqual(delta["overall_delta"], 13.0)
        self.assertEqual(delta["category_deltas"]["code_generation"]["delta"], 20.0)


if __name__ == "__main__":
    unittest.main()
