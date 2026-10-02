"""Tests for Baseline Analysis Pipeline, Recommendations, and Baseline Immutability (Step 6B).

Verifies that baseline data remains immutable, priority matrices are constructed,
controlled ablation experiments execute cleanly, and reports render all required sections.
"""

import json
from pathlib import Path
import tempfile
import unittest

from my_ai_agent.analysis.baseline_analysis import BaselineAnalysisPipeline
from my_ai_agent.analysis.experiments import AblationExperimentRunner
from my_ai_agent.analysis.recommendations import RecommendationEngine
from my_ai_agent.analysis.report import AnalysisReportGenerator
from my_ai_agent.analysis.root_cause import RootCauseRecord


BASELINE_FILE = Path(__file__).resolve().parent.parent / "evaluation" / "results" / "baseline_v1.json"


class TestBaselineAnalysisPipeline(unittest.TestCase):
    """Verifies baseline analysis pipeline and immutability guarantees."""

    def test_baseline_file_immutability(self):
        """CRITICAL: Baseline file content must remain completely unchanged after analysis runs."""
        initial_content = BASELINE_FILE.read_text(encoding="utf-8")
        pipeline = BaselineAnalysisPipeline()
        pipeline.run_analysis()
        after_content = BASELINE_FILE.read_text(encoding="utf-8")
        self.assertEqual(initial_content, after_content, "Baseline file was illegally modified!")

    def test_recommendation_and_candidate_separation(self):
        """Verifies separation of legitimate training candidates from non-training issues."""
        pipeline = BaselineAnalysisPipeline()
        data = pipeline.run_analysis()
        training_cands = data.get("training_candidates", [])
        non_train = data.get("non_training_issues", [])

        self.assertGreater(len(training_cands), 0)
        self.assertGreater(len(non_train), 0)

        # Ensure no task is both a training candidate and a non-training issue
        train_ids = {c["task_id"] for c in training_cands}
        non_train_ids = {n["task_id"] for n in non_train}
        intersection = train_ids.intersection(non_train_ids)
        self.assertEqual(len(intersection), 0, f"Overlapping candidate tasks: {intersection}")

    def test_ablation_experiments_execution(self):
        """Verifies prompt, context, and architecture ablation experiments."""
        p_res = AblationExperimentRunner.run_prompt_ablation()
        self.assertEqual(p_res["experiment"], "prompt_ablation")
        self.assertTrue(p_res["results"]["explicit_schema_prompt"]["valid"])

        c_res = AblationExperimentRunner.run_context_ablation()
        self.assertEqual(c_res["experiment"], "context_ablation")
        self.assertFalse(c_res["results"]["relevant_context"]["sensitive_files_exposed"])

        m_res = AblationExperimentRunner.run_model_vs_architecture()
        self.assertEqual(m_res["experiment"], "model_vs_architecture")
        self.assertIn("raw_model", m_res["results"])
        self.assertIn("agent_pipeline", m_res["results"])

    def test_markdown_report_sections(self):
        """Verifies that generated baseline analysis report contains all required headers."""
        pipeline = BaselineAnalysisPipeline()
        data = pipeline.run_analysis()
        report_text = AnalysisReportGenerator.generate_report(data)

        required_sections = [
            "Executive Summary",
            "Baseline Overview",
            "Category Analysis",
            "Difficulty Analysis",
            "Failure Taxonomy",
            "Root-Cause Distribution",
            "Model vs Architecture Findings",
            "Capability Deep-Dives",
            "Engineering Priority Matrix",
            "Training Candidates",
            "Next Step Recommendation",
        ]
        for sec in required_sections:
            self.assertIn(sec, report_text, f"Missing section: '{sec}' in report")


if __name__ == "__main__":
    unittest.main()
