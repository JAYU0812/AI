"""Baseline Weakness Analysis & Root-Cause Engineering Pipeline (Step 6B).

Reads the immutable Step 6A Baseline results (baseline_v1.json), processes all 104
tasks through the root-cause decision tree, clusters failures, runs ablation experiments,
and exports structured JSON and Markdown audit reports.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from my_ai_agent.analysis.category_analyzer import CategoryAnalyzer
from my_ai_agent.analysis.experiments import AblationExperimentRunner
from my_ai_agent.analysis.failure_analyzer import TaskFailureAnalyzer
from my_ai_agent.analysis.pattern_detector import PatternDetector
from my_ai_agent.analysis.recommendations import RecommendationEngine
from my_ai_agent.analysis.report import AnalysisReportGenerator
from my_ai_agent.analysis.root_cause import RootCauseCategory, RootCauseRecord


ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ANALYSIS_DIR / "results"
REPORTS_DIR = ANALYSIS_DIR / "reports"
BASELINE_V1_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "results" / "baseline_v1.json"


class BaselineAnalysisPipeline:
    """Orchestrates comprehensive root-cause analysis over the immutable baseline results."""

    def __init__(self, baseline_path: Optional[Path | str] = None):
        self.baseline_path = Path(baseline_path or BASELINE_V1_PATH).resolve()

    def run_analysis(self) -> Dict[str, Any]:
        """Loads baseline data, analyzes every task, and compiles comprehensive findings."""
        if not self.baseline_path.exists():
            raise FileNotFoundError(f"Baseline file not found at: {self.baseline_path}")

        # 1. Read-only load of immutable baseline data
        raw_baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        tasks = raw_baseline.get("tasks", [])

        # 2. Analyze every single task (104 tasks)
        records: List[RootCauseRecord] = []
        for t in tasks:
            rec = TaskFailureAnalyzer.analyze_task(t)
            records.append(rec)

        # 3. Root-cause distribution across all tasks
        rc_distribution: Dict[str, int] = {}
        for r in records:
            if r.status != "PASS" and r.primary_root_cause != RootCauseCategory.UNKNOWN:
                val = r.primary_root_cause.value
                rc_distribution[val] = rc_distribution.get(val, 0) + 1

        # 4. Deep-dive category diagnostics
        category_analysis = CategoryAnalyzer.analyze_categories(records)

        # 5. Difficulty analysis
        difficulty_analysis = PatternDetector.analyze_by_difficulty(records)

        # 6. Failure clustering
        clusters = PatternDetector.cluster_failures(records)

        # 7. Priority matrix & recommendations
        priority_matrix = RecommendationEngine.build_priority_matrix(clusters)
        training_candidates = RecommendationEngine.filter_training_candidates(records)
        non_training_issues = RecommendationEngine.filter_non_training_issues(records)

        # 8. Controlled empirical ablation experiments
        experiments = {
            "prompt_ablation": AblationExperimentRunner.run_prompt_ablation(),
            "context_ablation": AblationExperimentRunner.run_context_ablation(),
            "model_vs_architecture": AblationExperimentRunner.run_model_vs_architecture(),
        }

        # 9. Assemble full analysis result
        analysis_data: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_version": raw_baseline.get("dataset_version", "v1"),
            "model_id": raw_baseline.get("model_id", "deterministic"),
            "total_tasks": len(records),
            "passed": raw_baseline.get("passed", 0),
            "partial": raw_baseline.get("partial", 0),
            "failed": raw_baseline.get("failed", 0),
            "safety_violations": raw_baseline.get("safety_violations", 0),
            "overall_score": raw_baseline.get("overall_score_percentage", 62.5),
            "failure_distribution": raw_baseline.get("failure_distribution", {}),
            "root_cause_distribution": rc_distribution,
            "categories": category_analysis,
            "difficulty_analysis": difficulty_analysis,
            "clusters": [c.to_dict() for c in clusters],
            "priority_matrix": priority_matrix,
            "training_candidates": training_candidates,
            "non_training_issues": non_training_issues,
            "experiments": experiments,
            "records": [r.to_dict() for r in records],
        }

        # 10. Save all artifacts
        self._save_artifacts(analysis_data, clusters, priority_matrix, training_candidates)

        return analysis_data

    def _save_artifacts(
        self,
        analysis_data: Dict[str, Any],
        clusters: List[Any],
        priority_matrix: List[Any],
        training_candidates: List[Any],
    ) -> None:
        """Saves machine-readable JSON and Markdown documents."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. baseline_analysis_v1.json
        (RESULTS_DIR / "baseline_analysis_v1.json").write_text(
            json.dumps(analysis_data, indent=2), encoding="utf-8"
        )

        # 2. failure_clusters_v1.json
        (RESULTS_DIR / "failure_clusters_v1.json").write_text(
            json.dumps([c.to_dict() for c in clusters], indent=2), encoding="utf-8"
        )

        # 3. recommendations_v1.json
        (RESULTS_DIR / "recommendations_v1.json").write_text(
            json.dumps(priority_matrix, indent=2), encoding="utf-8"
        )

        # 4. training_candidates_v1.json
        (RESULTS_DIR / "training_candidates_v1.json").write_text(
            json.dumps(training_candidates, indent=2), encoding="utf-8"
        )

        # 5. baseline_analysis_v1.md
        md_report = AnalysisReportGenerator.generate_report(analysis_data)
        (REPORTS_DIR / "baseline_analysis_v1.md").write_text(md_report, encoding="utf-8")


def main():
    print(f"\n{'='*65}")
    print("  STEP 6B — BASELINE WEAKNESS & ROOT-CAUSE ANALYSIS")
    print(f"{'='*65}")

    pipeline = BaselineAnalysisPipeline()
    results = pipeline.run_analysis()

    print(f"Total Tasks Analyzed:    {results['total_tasks']}")
    print(f"Passed:                  {results['passed']}")
    print(f"Partial:                 {results['partial']}")
    print(f"Failed:                  {results['failed']}")
    print(f"Safety Violations:       {results['safety_violations']}")
    print(f"Overall Baseline Score:  {results['overall_score']:.1f}%")

    print(f"\nROOT-CAUSE DISTRIBUTION:")
    for rc, count in results['root_cause_distribution'].items():
        print(f"  - {rc:<30} {count:>3} tasks")

    print(f"\nFAILURE CLUSTERS IDENTIFIED: {len(results['clusters'])}")
    for c in results['clusters']:
        print(f"  - [{c['priority']}] {c['name']:<45} ({c['impact_task_count']} tasks)")

    print(f"\nDATASET CANDIDATES:")
    print(f"  - Fine-Tuning Candidates:  {len(results['training_candidates'])} tasks")
    print(f"  - Non-Training Issues:     {len(results['non_training_issues'])} tasks")

    print(f"\n[Artifacts Created]")
    print(f"  - {RESULTS_DIR / 'baseline_analysis_v1.json'}")
    print(f"  - {RESULTS_DIR / 'failure_clusters_v1.json'}")
    print(f"  - {RESULTS_DIR / 'recommendations_v1.json'}")
    print(f"  - {RESULTS_DIR / 'training_candidates_v1.json'}")
    print(f"  - {REPORTS_DIR / 'baseline_analysis_v1.md'}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
