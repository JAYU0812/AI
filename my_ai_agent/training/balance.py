"""Dataset Balance Analyzer and Distribution Auditor.

Tracks and assesses balance across task types, difficulty levels,
output types, and provenance root causes.
"""

from collections import Counter
from typing import Any, Dict, List

from my_ai_agent.training.schema import TrainingExample


class DatasetBalanceAuditor:
    """Audits distribution metrics to ensure balanced representation."""

    @staticmethod
    def analyze_distribution(examples: List[TrainingExample]) -> Dict[str, Any]:
        """Calculates multi-dimensional distribution stats for dataset."""
        total = len(examples)
        if total == 0:
            return {
                "total": 0,
                "by_task_type": {},
                "by_difficulty": {},
                "by_output_type": {},
                "by_root_cause": {},
                "is_balanced": False,
            }

        by_task_type = Counter(e.task_type for e in examples)
        by_difficulty = Counter(e.difficulty for e in examples)
        by_output_type = Counter(e.output_type for e in examples)
        by_root_cause = Counter(
            e.provenance.source_root_cause for e in examples if e.provenance
        )

        # Check that all 4 primary domains are represented
        required_domains = {
            "code_generation",
            "code_modification",
            "structured_output",
            "instruction_following",
        }
        present_domains = set(by_task_type.keys())
        has_all_domains = required_domains.issubset(present_domains)

        # Check no domain accounts for more than 50% of the dataset
        max_ratio = max(by_task_type.values()) / total if total > 0 else 0
        is_balanced = has_all_domains and (max_ratio <= 0.50)

        return {
            "total": total,
            "by_task_type": {k: {"count": v, "percentage": round(v / total * 100, 2)} for k, v in by_task_type.items()},
            "by_difficulty": {k: {"count": v, "percentage": round(v / total * 100, 2)} for k, v in by_difficulty.items()},
            "by_output_type": {k: {"count": v, "percentage": round(v / total * 100, 2)} for k, v in by_output_type.items()},
            "by_root_cause": {k: {"count": v, "percentage": round(v / total * 100, 2)} for k, v in by_root_cause.items()},
            "is_balanced": is_balanced,
            "max_domain_share_percentage": round(max_ratio * 100, 2),
        }
