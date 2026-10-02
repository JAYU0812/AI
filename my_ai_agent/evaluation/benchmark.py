"""High-Level Benchmark Runner and Terminal Report Engine.

Provides standard one-line execution command for benchmarks, renders formatted
terminal scorecard, and automatically outputs both JSON results and Markdown reports.
"""

import argparse
from pathlib import Path
from typing import Optional

from my_ai_agent.evaluation.dataset import BenchmarkDataset
from my_ai_agent.evaluation.report import BenchmarkReportGenerator
from my_ai_agent.evaluation.runner import BenchmarkRunner
from my_ai_agent.intelligence import DeterministicModel, registry


def run_benchmark_suite(
    dataset_version: str = "v1",
    model_name: str = "deterministic",
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    split: Optional[str] = None,
    runs: int = 1,
    output_report: Optional[str] = None,
):
    """Executes benchmark and prints terminal summary."""
    print(f"\n{'='*65}")
    print(f"      AGENT INTELLIGENCE BENCHMARK SUITE ({dataset_version.upper()})")
    print(f"{'='*65}")

    # Model resolution
    if model_name == "local":
        from my_ai_agent.config import config
        config.AI_MODEL_PROVIDER = "local"
        m = registry.get("local")
        if not m or m.health.status.value != "AVAILABLE":
            print("[Benchmark] Loading local model weights...")
            if m and hasattr(m, "load"):
                m.load()
    else:
        m = DeterministicModel()

    dataset = BenchmarkDataset(version=dataset_version)
    runner = BenchmarkRunner(dataset=dataset)

    results = runner.run_benchmark(
        model=m,
        model_name=model_name,
        category=category,
        difficulty=difficulty,
        split=split,
        runs=runs,
        verbose=True,
    )

    # Save JSON result
    json_path = runner.save_result(results)

    # Generate and save Markdown report
    report_md = BenchmarkReportGenerator.generate_report(results)
    report_filename = output_report or f"baseline_{dataset_version}.md"
    report_path = BenchmarkReportGenerator.save_markdown_report(report_md, filename=report_filename)

    # Print Clean Terminal Summary
    print(f"\n{'='*65}")
    print("                    BENCHMARK SCORECARD")
    print(f"{'='*65}")
    print(f"Model:              {results.get('model_id')}")
    print(f"Dataset Version:    {dataset_version.upper()}")
    print(f"Total Tasks:        {results.get('total_tasks')}")
    print(f"Passed:             {results.get('passed')}")
    print(f"Partial:            {results.get('partial')}")
    print(f"Failed:             {results.get('failed')}")
    print(f"Safety Violations:  {results.get('safety_violations')}")
    print(f"Total Latency:      {results.get('total_latency_seconds'):.1f}s")
    print(f"OVERALL SCORE:      {results.get('overall_score_percentage'):.1f}%")
    print(f"{'='*65}")

    print("\nCATEGORY RESULTS:")
    for cat_name, data in results.get("categories", {}).items():
        clean_name = cat_name.replace("_", " ").title()
        print(f"  - {clean_name:<26} {data.get('average_score', 0.0):>5.1f}%  ({data.get('count')} tasks)")

    print(f"\n[Artifacts Saved]")
    print(f"  - JSON Results:     {json_path}")
    print(f"  - Markdown Report:  {report_path}")
    print(f"{'='*65}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Agent Intelligence Benchmark Suite")
    parser.add_argument("--dataset", default="v1", help="Dataset version")
    parser.add_argument("--model", default="deterministic", choices=["deterministic", "local"], help="Model provider")
    parser.add_argument("--category", default=None, help="Filter category")
    parser.add_argument("--difficulty", default=None, help="Filter difficulty")
    parser.add_argument("--split", default=None, help="Filter split (development, validation, test)")
    parser.add_argument("--runs", type=int, default=1, help="Repetitions")
    parser.add_argument("--report", default=None, help="Markdown report filename")
    args = parser.parse_args()

    run_benchmark_suite(
        dataset_version=args.dataset,
        model_name=args.model,
        category=args.category,
        difficulty=args.difficulty,
        split=args.split,
        runs=args.runs,
        output_report=args.report,
    )


if __name__ == "__main__":
    main()
