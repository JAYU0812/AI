"""Baseline Model Manager & Comparator.

Loads, saves, and compares baseline benchmark outputs to evaluate model regressions
and measure incremental performance across iterations.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_BASELINE_FILE = RESULTS_DIR / "baseline_v1.json"


class BaselineManager:
    """Manages persistent baseline results and delta comparisons."""

    @classmethod
    def load_baseline(cls, path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
        """Loads baseline result JSON if exists."""
        target = Path(path).resolve() if path else DEFAULT_BASELINE_FILE
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return None

    @classmethod
    def save_baseline(cls, result: Dict[str, Any], path: Optional[Path | str] = None) -> Path:
        """Saves benchmark output as official baseline."""
        target = Path(path).resolve() if path else DEFAULT_BASELINE_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return target

    @classmethod
    def compare(
        cls,
        candidate_result: Dict[str, Any],
        baseline_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculates delta between candidate and baseline across categories and overall score."""
        c_score = candidate_result.get("overall_score_percentage", 0.0)
        b_score = baseline_result.get("overall_score_percentage", 0.0)
        delta_overall = round(c_score - b_score, 2)

        c_cats = candidate_result.get("categories", {})
        b_cats = baseline_result.get("categories", {})
        all_cats = set(c_cats.keys()).union(b_cats.keys())

        category_deltas: Dict[str, Dict[str, float]] = {}
        for cat in all_cats:
            c_val = c_cats.get(cat, {}).get("average_score", 0.0)
            b_val = b_cats.get(cat, {}).get("average_score", 0.0)
            category_deltas[cat] = {
                "candidate": c_val,
                "baseline": b_val,
                "delta": round(c_val - b_val, 2),
            }

        return {
            "candidate_model": candidate_result.get("model_id"),
            "baseline_model": baseline_result.get("model_id"),
            "candidate_overall": c_score,
            "baseline_overall": b_score,
            "overall_delta": delta_overall,
            "category_deltas": category_deltas,
            "is_improvement": delta_overall > 0,
        }
