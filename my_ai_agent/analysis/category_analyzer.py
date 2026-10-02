"""Category Deep-Dive Analyzer (Step 6B).

Aggregates task diagnostics per capability category, computing mean/median scores,
failure rates, dominant root causes, and tailored engineering recommendations.
"""

import statistics
from typing import Any, Dict, List

from my_ai_agent.analysis.root_cause import RootCauseCategory, RootCauseRecord


class CategoryAnalyzer:
    """Produces deep-dive diagnostic profiles across all 14 capability categories."""

    @classmethod
    def analyze_categories(cls, records: List[RootCauseRecord]) -> Dict[str, Dict[str, Any]]:
        """Groups records by category and produces detailed diagnostic metrics."""
        category_map: Dict[str, List[RootCauseRecord]] = {}
        for rec in records:
            category_map.setdefault(rec.category, []).append(rec)

        results: Dict[str, Dict[str, Any]] = {}

        for cat_name, recs in category_map.items():
            scores = [r.score_percentage for r in recs]
            total = len(recs)
            passed = sum(1 for r in recs if r.status == "PASS")
            partial = sum(1 for r in recs if r.status == "PARTIAL")
            failed = sum(1 for r in recs if r.status == "FAIL")

            avg_score = round(statistics.mean(scores), 2) if scores else 0.0
            median_score = round(statistics.median(scores), 2) if scores else 0.0
            failure_rate = round(((failed + partial) / max(1, total)) * 100.0, 2)

            # Failure causes distribution
            causes_count: Dict[str, int] = {}
            for r in recs:
                if r.status != "PASS" and r.primary_root_cause != RootCauseCategory.UNKNOWN:
                    c_val = r.primary_root_cause.value
                    causes_count[c_val] = causes_count.get(c_val, 0) + 1

            sorted_causes = sorted(causes_count.items(), key=lambda x: x[1], reverse=True)
            primary_cause = sorted_causes[0][0] if sorted_causes else "NONE"
            secondary_cause = sorted_causes[1][0] if len(sorted_causes) > 1 else "NONE"

            # Primary recommendation
            rec_actions = [r.recommended_action for r in recs if r.recommended_action and r.status != "PASS"]
            rep_recommendation = rec_actions[0] if rec_actions else "Maintain existing architecture and verify regression suite."

            results[cat_name] = {
                "category": cat_name,
                "total_tasks": total,
                "passed": passed,
                "partial": partial,
                "failed": failed,
                "average_score": avg_score,
                "median_score": median_score,
                "failure_rate": failure_rate,
                "primary_root_cause": primary_cause,
                "secondary_root_cause": secondary_cause,
                "recommended_action": rep_recommendation,
                "task_ids": [r.task_id for r in recs],
            }

        return results
