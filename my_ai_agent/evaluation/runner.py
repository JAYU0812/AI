"""Benchmark Runner.

CLI and programmatic runner for executing evaluation benchmarks against AIModel
instances, computing multi-run statistics, and outputting structured JSON results.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from my_ai_agent.evaluation.categories import DatasetSplit, Difficulty, FailureType, TaskCategory
from my_ai_agent.evaluation.dataset import BenchmarkDataset
from my_ai_agent.evaluation.evaluator import TaskEvaluationResult, TaskEvaluator
from my_ai_agent.evaluation.task import EvaluationTask
from my_ai_agent.intelligence import AIModel, DeterministicModel, LocalModelProvider, registry, router


RESULTS_DIR = Path(__file__).resolve().parent / "results"


class BenchmarkRunner:
    """Orchestrates benchmark runs across tasks, models, and repetitions."""

    def __init__(
        self,
        dataset: Optional[BenchmarkDataset] = None,
        evaluator: Optional[TaskEvaluator] = None,
    ):
        self.dataset = dataset or BenchmarkDataset(version="v1")
        self.evaluator = evaluator or TaskEvaluator()

    def run_benchmark(
        self,
        model: Optional[AIModel] = None,
        model_name: str = "deterministic",
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        split: Optional[str] = None,
        task_id: Optional[str] = None,
        runs: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Runs benchmark suite and aggregates comprehensive results."""
        tasks = self.dataset.filter(
            category=category,
            difficulty=difficulty,
            split=split,
            task_id=task_id,
        )

        if not tasks:
            return {"error": "No tasks matched the specified criteria"}

        model = model or router.route()
        start_time = time.time()
        task_results: List[Dict[str, Any]] = []

        total_tasks = len(tasks)
        passed_count = 0
        partial_count = 0
        failed_count = 0
        safety_violations = 0

        category_scores: Dict[str, List[float]] = {}
        failure_distribution: Dict[str, int] = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"BENCHMARK RUN: {self.dataset.version.upper()} — {len(tasks)} Tasks")
            print(f"Model: {model.model_id} ({getattr(model, 'backend', 'adapter')})")
            print(f"{'='*60}")

        for idx, task in enumerate(tasks, start=1):
            run_outcomes: List[TaskEvaluationResult] = []
            for r in range(runs):
                res = self.evaluator.evaluate_task(task, model=model)
                run_outcomes.append(res)

            # Average overall score across repetitions
            avg_score = sum(r.scorecard.overall_percentage for r in run_outcomes) / len(run_outcomes)
            last_res = run_outcomes[-1]
            last_card = last_res.scorecard

            if last_card.safety_violation:
                safety_violations += 1
                status = "SAFETY_VIOLATION"
                failed_count += 1
            elif last_card.passed:
                passed_count += 1
                status = "PASS"
            elif last_card.partial:
                partial_count += 1
                status = "PARTIAL"
            else:
                failed_count += 1
                status = "FAIL"

            cat_key = task.category.value if isinstance(task.category, TaskCategory) else str(task.category)
            category_scores.setdefault(cat_key, []).append(avg_score)

            if last_card.failed:
                f_type = last_card.failure_type.value if isinstance(last_card.failure_type, FailureType) else str(last_card.failure_type)
                failure_distribution[f_type] = failure_distribution.get(f_type, 0) + 1

            if verbose:
                lat = last_res.metrics.task_completion_latency_seconds or 0.0
                print(f"[{idx:03d}/{total_tasks:03d}] {task.id:<18} {status:<16} {avg_score:>5.1f}%  ({lat:.2f}s)")

            task_results.append({
                "task_id": task.id,
                "category": cat_key,
                "difficulty": task.difficulty.value if isinstance(task.difficulty, Difficulty) else str(task.difficulty),
                "split": task.split.value if isinstance(task.split, DatasetSplit) else str(task.split),
                "status": status,
                "score_percentage": round(avg_score, 2),
                "dimensions": last_card.dimensions,
                "safety_violation": last_card.safety_violation,
                "failure_type": last_card.failure_type.value if isinstance(last_card.failure_type, FailureType) else str(last_card.failure_type),
                "metrics": last_res.metrics.to_dict(),
                "assertion_results": last_card.assertion_results,
            })

        total_elapsed = round(time.time() - start_time, 2)
        overall_score = round(sum(t["score_percentage"] for t in task_results) / len(task_results), 2) if task_results else 0.0

        # Category summaries
        category_summary = {}
        for cat, scores in category_scores.items():
            category_summary[cat] = {
                "count": len(scores),
                "average_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            }

        benchmark_result = {
            "dataset_version": self.dataset.version,
            "model_id": model.model_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_tasks": total_tasks,
            "runs_per_task": runs,
            "passed": passed_count,
            "partial": partial_count,
            "failed": failed_count,
            "safety_violations": safety_violations,
            "overall_score_percentage": overall_score,
            "total_latency_seconds": total_elapsed,
            "categories": category_summary,
            "failure_distribution": failure_distribution,
            "tasks": task_results,
        }

        return benchmark_result

    def save_result(self, result: Dict[str, Any], output_path: Optional[Path | str] = None) -> Path:
        """Saves benchmark results to a JSON file."""
        if not output_path:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = result.get("model_id", "unknown").replace("/", "_")
            out_file = RESULTS_DIR / f"benchmark_{model_id}_{ts}.json"
        else:
            out_file = Path(output_path).resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)

        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return out_file


def main():
    parser = argparse.ArgumentParser(description="AI Agent Benchmark Runner (Step 6A)")
    parser.add_argument("--dataset", default="v1", help="Dataset version (e.g. v1)")
    parser.add_argument("--model", default="deterministic", choices=["deterministic", "local"], help="Model provider")
    parser.add_argument("--category", default=None, help="Filter by category")
    parser.add_argument("--difficulty", default=None, help="Filter by difficulty (easy, medium, hard)")
    parser.add_argument("--split", default=None, help="Filter by split (development, validation, test)")
    parser.add_argument("--task", default=None, help="Execute specific task ID")
    parser.add_argument("--runs", type=int, default=1, help="Repetitions per task")
    parser.add_argument("--output", default=None, help="JSON output file path")
    args = parser.parse_args()

    # Model selection
    if args.model == "local":
        from my_ai_agent.config import config
        config.AI_MODEL_PROVIDER = "local"
        m = registry.get("local")
        if not m or m.health.status.value != "AVAILABLE":
            print("[BenchmarkRunner] Loading local model weights...")
            if m and hasattr(m, "load"):
                m.load()
    else:
        m = DeterministicModel()

    dataset = BenchmarkDataset(version=args.dataset)
    runner = BenchmarkRunner(dataset=dataset)
    res = runner.run_benchmark(
        model=m,
        model_name=args.model,
        category=args.category,
        difficulty=args.difficulty,
        split=args.split,
        task_id=args.task,
        runs=args.runs,
        verbose=True,
    )

    out_path = runner.save_result(res, output_path=args.output)
    print(f"\n[BenchmarkRunner] Result saved to: {out_path}")
    print(f"Overall Score: {res.get('overall_score_percentage', 0.0)}%  (Passed: {res.get('passed')}/{res.get('total_tasks')})")


if __name__ == "__main__":
    main()
