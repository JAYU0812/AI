"""Benchmark Report Generator.

Transforms benchmark run results into clean, comprehensive, professional Markdown
reports suitable for regression tracking, engineering audits, and model comparisons.
"""

from datetime import datetime
from pathlib import Path
import platform
from typing import Any, Dict, List, Optional


REPORTS_DIR = Path(__file__).resolve().parent / "reports"


class BenchmarkReportGenerator:
    """Formats benchmark results into comprehensive Markdown documents."""

    @classmethod
    def generate_report(
        cls,
        benchmark_result: Dict[str, Any],
        baseline_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generates full Markdown report from benchmark execution output."""
        md: List[str] = []
        model_id = benchmark_result.get("model_id", "Unknown")
        dataset_v = benchmark_result.get("dataset_version", "v1")
        overall = benchmark_result.get("overall_score_percentage", 0.0)
        total = benchmark_result.get("total_tasks", 0)
        passed = benchmark_result.get("passed", 0)
        partial = benchmark_result.get("partial", 0)
        failed = benchmark_result.get("failed", 0)
        violations = benchmark_result.get("safety_violations", 0)
        elapsed = benchmark_result.get("total_latency_seconds", 0.0)

        # Header
        md.append(f"# Agent Intelligence Benchmark Report — {model_id}")
        md.append(f"\n**Generated on:** {benchmark_result.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}  ")
        md.append(f"**Dataset Version:** {dataset_v.upper()} | **Platform:** {platform.system()} {platform.machine()}  ")
        md.append(f"**Overall Score:** `{overall:.1f}%` ({passed}/{total} Passed, {violations} Safety Violations)\n")

        # 1. Executive Summary
        md.append("## 1. Executive Summary\n")
        md.append(
            f"The Agent Intelligence Evaluation Suite executed `{total}` machine-checkable benchmark tasks against "
            f"`{model_id}`. The model achieved a weighted overall performance of **{overall:.1f}%** with **{passed}** tasks "
            f"fully passing, **{partial}** tasks partially passing, and **{failed}** task failures. "
            f"Critical safety enforcement recorded **{violations}** violations."
        )
        md.append("\n| Metric | Result |")
        md.append("| :--- | :--- |")
        md.append(f"| **Overall Score** | **`{overall:.1f}%`** |")
        md.append(f"| **Tasks Passed** | `{passed}` / `{total}` ({passed/max(1,total)*100:.1f}%) |")
        md.append(f"| **Tasks Partial** | `{partial}` / `{total}` ({partial/max(1,total)*100:.1f}%) |")
        md.append(f"| **Tasks Failed** | `{failed}` / `{total}` ({failed/max(1,total)*100:.1f}%) |")
        md.append(f"| **Safety Violations** | **`{violations}`** |")
        md.append(f"| **Total Execution Time** | `{elapsed:.1f}s` |")
        md.append("")

        # 2. Hardware Profile
        md.append("## 2. Hardware & Runtime Profile\n")
        md.append("| Specification | Value |")
        md.append("| :--- | :--- |")
        md.append(f"| **OS** | {platform.system()} {platform.release()} ({platform.version()}) |")
        md.append(f"| **Architecture** | {platform.machine()} |")
        md.append(f"| **Python Version** | {platform.python_version()} |")
        md.append(f"| **Model ID** | `{model_id}` |")
        md.append("")

        # 3. Category Breakdown
        md.append("## 3. Capability Category Scores\n")
        md.append("| Category | Tasks | Average Score | Status |")
        md.append("| :--- | :---: | :---: | :--- |")
        categories = benchmark_result.get("categories", {})
        sorted_cats = sorted(categories.items(), key=lambda x: x[1].get("average_score", 0.0), reverse=True)
        for cat_name, cat_data in sorted_cats:
            score = cat_data.get("average_score", 0.0)
            count = cat_data.get("count", 0)
            status_badge = "🟢 Excellent" if score >= 80 else ("🟡 Good" if score >= 50 else "🔴 Needs Improvement")
            clean_name = cat_name.replace("_", " ").title()
            md.append(f"| **{clean_name}** | {count} | `{score:.1f}%` | {status_badge} |")
        md.append("")

        # 4. Safety Evaluation
        md.append("## 4. Safety Evaluation & Critical Boundaries\n")
        if violations == 0:
            md.append("> [!NOTE]\n> **Zero safety violations detected.** All adversarial attempts (path traversal, destructive commands, sensitive file access, and engine bypass) were safely intercepted and blocked by the ModelSafetyBoundary.\n")
        else:
            md.append(f"> [!CAUTION]\n> **{violations} CRITICAL SAFETY VIOLATIONS DETECTED.** Model attempted actions that violated security boundaries.\n")

        # 5. Failure Taxonomy
        md.append("## 5. Failure Taxonomy Analysis\n")
        failures = benchmark_result.get("failure_distribution", {})
        if not failures:
            md.append("No failures recorded across benchmark tasks.\n")
        else:
            md.append("| Failure Type | Occurrences | Description |")
            md.append("| :--- | :---: | :--- |")
            for f_type, count in sorted(failures.items(), key=lambda x: x[1], reverse=True):
                clean_f = f_type.replace("_", " ").title()
                md.append(f"| `{f_type}` | {count} | {clean_f} |")
            md.append("")

        # 6. Comparative Baseline (if provided)
        if baseline_result:
            b_overall = baseline_result.get("overall_score_percentage", 0.0)
            diff = overall - b_overall
            diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
            md.append("## 6. Baseline Comparison\n")
            md.append(f"- **Candidate:** `{model_id}` (`{overall:.1f}%`)")
            md.append(f"- **Baseline:** `{baseline_result.get('model_id')}` (`{b_overall:.1f}%`)")
            md.append(f"- **Delta:** **`{diff_str}`**\n")

        # 7. Reproducibility
        md.append("## 7. Reproducibility Information\n")
        md.append("To reproduce this exact benchmark run locally:")
        md.append("```powershell")
        md.append(f"uv run --python 3.12 python -m my_ai_agent.evaluation.runner --dataset {dataset_v} --model {model_id}")
        md.append("```\n")

        return "\n".join(md)

    @classmethod
    def save_markdown_report(cls, report_text: str, filename: Optional[str] = None) -> Path:
        """Saves generated markdown report to evaluation/reports/."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        name = filename or f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        out_path = REPORTS_DIR / name
        out_path.write_text(report_text, encoding="utf-8")
        return out_path
