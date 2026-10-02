"""Step 6D — Baseline vs Fine-Tuned Comparison.

Implements per-task comparison analysis:
  - PASS → PASS: unchanged
  - PASS → FAIL: regression
  - FAIL → PASS: improvement
  - FAIL → FAIL: unchanged failure

Also produces the category comparison table and weakness improvement analysis.

IMPORTANT: The official Step 6A benchmark comparison is NOT run here.
It is run via the existing my_ai_agent/evaluation/ benchmark runner.
This module only processes and formats the comparison results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Baseline scores from Step 6A (ground truth, never to be overwritten)
BASELINE_6A_RESULTS = {
    "overall_score": 62.5,
    "passed": 50,
    "partial": 23,
    "failed": 31,
    "safety_violations": 0,
    "categories": {
        "Debugging": 93.8,
        "Project Understanding": 91.7,
        "Verification": 91.7,
        "General Understanding": 86.1,
        "Context Usage": 85.8,
        "Autonomous Completion": 74.0,
        "Code Modification": 66.9,
        "Safety": 65.8,
        "Multi-Step": 64.6,
        "Instruction Following": 49.0,
        "Code Generation": 36.5,
        "Intent Understanding": 16.7,
        "Structured Output": 16.7,
        "Tool Selection": 16.7,
    }
}


def compare_task_results(
    baseline_results: List[Dict],
    finetuned_results: List[Dict],
) -> Dict[str, Any]:
    """Per-task comparison: PASS/FAIL changes (Step 6D.21).

    Args:
        baseline_results: List of task result dicts with 'id', 'passed', 'score'.
        finetuned_results: Same structure for the fine-tuned model.

    Returns:
        dict with improvements, regressions, unchanged, and change matrix.
    """
    baseline_by_id = {r["id"]: r for r in baseline_results}
    finetuned_by_id = {r["id"]: r for r in finetuned_results}

    improvements = []
    regressions = []
    unchanged_pass = []
    unchanged_fail = []

    all_ids = set(baseline_by_id) | set(finetuned_by_id)

    for task_id in sorted(all_ids):
        b = baseline_by_id.get(task_id, {})
        f = finetuned_by_id.get(task_id, {})

        b_pass = b.get("passed", False)
        f_pass = f.get("passed", False)
        b_score = b.get("score", 0.0)
        f_score = f.get("score", 0.0)

        entry = {
            "id": task_id,
            "baseline_passed": b_pass,
            "finetuned_passed": f_pass,
            "baseline_score": b_score,
            "finetuned_score": f_score,
            "delta": round(f_score - b_score, 4),
        }

        if not b_pass and f_pass:
            improvements.append(entry)
        elif b_pass and not f_pass:
            regressions.append(entry)
        elif b_pass and f_pass:
            unchanged_pass.append(entry)
        else:
            unchanged_fail.append(entry)

    total = len(all_ids)
    return {
        "total_tasks": total,
        "improvements": improvements,
        "regressions": regressions,
        "unchanged_pass": unchanged_pass,
        "unchanged_fail": unchanged_fail,
        "improvement_count": len(improvements),
        "regression_count": len(regressions),
        "unchanged_pass_count": len(unchanged_pass),
        "unchanged_fail_count": len(unchanged_fail),
    }


def build_category_comparison(
    baseline_categories: Dict[str, float],
    finetuned_categories: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Build the category comparison table (Step 6D.19).

    Args:
        baseline_categories: {category_name: score_pct} for baseline.
        finetuned_categories: {category_name: score_pct} for fine-tuned.

    Returns:
        Sorted list of category comparison dicts.
    """
    all_cats = set(baseline_categories) | set(finetuned_categories)
    rows = []

    for cat in sorted(all_cats):
        b_score = baseline_categories.get(cat, None)
        f_score = finetuned_categories.get(cat, None)
        delta = None
        if b_score is not None and f_score is not None:
            delta = round(f_score - b_score, 1)

        rows.append({
            "category": cat,
            "baseline": b_score,
            "fine_tuned": f_score,
            "delta": delta,
            "direction": (
                "improved" if delta and delta > 0 else
                "regressed" if delta and delta < 0 else
                "unchanged"
            ),
        })

    # Sort by delta descending (improvements first)
    rows.sort(key=lambda x: x.get("delta") or 0, reverse=True)
    return rows


def analyze_weakness_improvement(
    baseline_categories: Dict[str, float],
    finetuned_categories: Dict[str, float],
) -> Dict[str, Any]:
    """Analyze improvement in the Step 6B-identified weakness categories (Step 6D.20).

    Target weaknesses:
        Intent Understanding (~17%)
        Structured Output (~17%)
        Tool Selection (~17%)
        Code Generation (~37%)
        Instruction Following (~49%)

    Also checks for regressions in strong categories:
        Debugging (94%)
        Project Understanding (92%)
        Verification (92%)
        Safety (66%)
    """
    TARGET_WEAKNESSES = [
        "Intent Understanding",
        "Structured Output",
        "Tool Selection",
        "Code Generation",
        "Code Modification",
        "Instruction Following",
    ]
    STRONG_CATEGORIES = [
        "Debugging",
        "Project Understanding",
        "Verification",
        "General Understanding",
        "Safety",
    ]

    weakness_analysis = []
    for cat in TARGET_WEAKNESSES:
        b = baseline_categories.get(cat)
        f = finetuned_categories.get(cat)
        if b is not None and f is not None:
            delta = round(f - b, 1)
            weakness_analysis.append({
                "category": cat,
                "baseline": b,
                "fine_tuned": f,
                "delta": delta,
                "improved": delta > 0,
                "target_met": delta > 5,  # >5% improvement = meaningful
            })

    regression_analysis = []
    for cat in STRONG_CATEGORIES:
        b = baseline_categories.get(cat)
        f = finetuned_categories.get(cat)
        if b is not None and f is not None:
            delta = round(f - b, 1)
            regression_analysis.append({
                "category": cat,
                "baseline": b,
                "fine_tuned": f,
                "delta": delta,
                "regressed": delta < -5,  # >5% drop = concerning regression
            })

    improved_count = sum(1 for w in weakness_analysis if w.get("improved"))
    regressed_count = sum(1 for r in regression_analysis if r.get("regressed"))

    return {
        "weakness_improvements": weakness_analysis,
        "regression_checks": regression_analysis,
        "weakness_improved_count": improved_count,
        "weakness_total": len(weakness_analysis),
        "regression_count": regressed_count,
        "meaningful_improvement": improved_count >= len(weakness_analysis) * 0.4,
    }


def determine_experiment_result(
    overall_baseline: float,
    overall_finetuned: float,
    weakness_analysis: Dict[str, Any],
    safety_comparison: Dict[str, Any],
    pilot_result: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Determine the final experiment result category (Step 6D.31-6D.33).

    Returns:
        dict with 'result', 'verdict', 'rationale'.
    """
    # Check for infrastructure failure
    if pilot_result and pilot_result.get("status") != "PASS":
        return {
            "result": "FAILED_INFRASTRUCTURE",
            "verdict": "EXPERIMENT FAIL — Pilot run failed",
            "rationale": f"Pilot failed: {pilot_result.get('errors', [])}",
        }

    # Check for safety regression (hard constraint)
    if safety_comparison.get("experiment_fail", False):
        return {
            "result": "REGRESSION",
            "verdict": "EXPERIMENT FAIL — SAFETY REGRESSION",
            "rationale": "Fine-tuned model introduced safety violations not present in baseline",
        }

    delta = overall_finetuned - overall_baseline
    weakness_improved = weakness_analysis.get("meaningful_improvement", False)
    regression_count = weakness_analysis.get("regression_count", 0)

    if delta < -5 or regression_count >= 3:
        return {
            "result": "REGRESSION",
            "verdict": "REGRESSION",
            "rationale": (
                f"Fine-tuned model overall score dropped by {abs(delta):.1f}% "
                f"with {regression_count} strong category regressions."
            ),
        }

    if weakness_improved and delta >= 0:
        return {
            "result": "SUCCESS",
            "verdict": "SUCCESS",
            "rationale": (
                f"Fine-tuned model improved overall by {delta:.1f}% "
                f"with {weakness_analysis['weakness_improved_count']}/{weakness_analysis['weakness_total']} "
                f"target weaknesses improved."
            ),
        }

    if weakness_improved or (delta > 0 and delta < 5):
        return {
            "result": "PARTIAL_SUCCESS",
            "verdict": "PARTIAL SUCCESS",
            "rationale": (
                f"Delta: {delta:+.1f}%. "
                f"{weakness_analysis['weakness_improved_count']}/{weakness_analysis['weakness_total']} weaknesses improved. "
                "Improvement is present but not conclusive."
            ),
        }

    return {
        "result": "NO_SIGNIFICANT_IMPROVEMENT",
        "verdict": "NO SIGNIFICANT IMPROVEMENT",
        "rationale": (
            f"Delta: {delta:+.1f}%. "
            f"{weakness_analysis['weakness_improved_count']}/{weakness_analysis['weakness_total']} weaknesses improved. "
            "Fine-tuning on 32 examples with CPU-only training produced no significant effect."
        ),
    }


def build_comparison_dict(
    baseline_overall: float,
    finetuned_overall: float,
    baseline_categories: Dict[str, float],
    finetuned_categories: Dict[str, float],
    baseline_task_results: Optional[List[Dict]] = None,
    finetuned_task_results: Optional[List[Dict]] = None,
    safety_baseline: Optional[Dict] = None,
    safety_finetuned: Optional[Dict] = None,
    training_metrics: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build the complete comparison dictionary."""
    from .safety import compare_safety

    task_comparison = {}
    if baseline_task_results and finetuned_task_results:
        task_comparison = compare_task_results(baseline_task_results, finetuned_task_results)

    category_comparison = build_category_comparison(baseline_categories, finetuned_categories)
    weakness_analysis = analyze_weakness_improvement(baseline_categories, finetuned_categories)

    safety_comparison = {}
    if safety_baseline and safety_finetuned:
        safety_comparison = compare_safety(safety_baseline, safety_finetuned)
    else:
        safety_comparison = {"experiment_fail": False, "verdict": "SAFETY PROBES NOT RUN"}

    experiment_result = determine_experiment_result(
        overall_baseline=baseline_overall,
        overall_finetuned=finetuned_overall,
        weakness_analysis=weakness_analysis,
        safety_comparison=safety_comparison,
    )

    return {
        "experiment_id": "experiment_001",
        "baseline_model": BASELINE_6A_RESULTS,
        "overall": {
            "baseline": baseline_overall,
            "fine_tuned": finetuned_overall,
            "delta": round(finetuned_overall - baseline_overall, 1),
        },
        "category_comparison": category_comparison,
        "weakness_analysis": weakness_analysis,
        "task_comparison": task_comparison,
        "safety": safety_comparison,
        "training_metrics": training_metrics or {},
        "experiment_result": experiment_result,
    }


def save_comparison(comparison: Dict[str, Any], output_path: str) -> str:
    """Save comparison dict to JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)
    return str(p)
