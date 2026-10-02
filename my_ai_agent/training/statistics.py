"""Comprehensive Training Dataset Statistics Engine.

Computes overall metrics, quality distributions, split counts, rejection reasons,
and provenance coverage for the generated dataset.
"""

from collections import Counter
from typing import Any, Dict, List

from my_ai_agent.training.balance import DatasetBalanceAuditor
from my_ai_agent.training.schema import TrainingExample


class DatasetStatisticsEngine:
    """Calculates granular statistics across accepted and rejected dataset partitions."""

    @classmethod
    def compute_statistics(
        cls,
        all_candidates: List[TrainingExample],
        accepted_examples: List[TrainingExample],
        train_examples: List[TrainingExample],
        val_examples: List[TrainingExample],
        test_examples: List[TrainingExample],
        rejected_examples: List[TrainingExample],
    ) -> Dict[str, Any]:
        """Generates comprehensive dataset statistics dictionary."""
        total_accepted = len(accepted_examples)
        total_rejected = len(rejected_examples)
        total_generated = len(all_candidates)

        scores = [e.quality.score for e in accepted_examples if hasattr(e, "quality")]
        avg_quality = round(sum(scores) / len(scores), 4) if scores else 0.0

        # Score bands: [0.85-0.90), [0.90-0.95), [0.95-1.00]
        quality_distribution = {
            "0.85-0.89": sum(1 for s in scores if 0.85 <= s < 0.90),
            "0.90-0.94": sum(1 for s in scores if 0.90 <= s < 0.95),
            "0.95-1.00": sum(1 for s in scores if 0.95 <= s <= 1.00),
        }

        # Rejection reasons breakdown
        rejection_reasons = Counter()
        for rej in rejected_examples:
            reason = rej.quality.rejection_reason or "Unknown"
            # Extract key prefix
            if "Secret" in reason:
                rejection_reasons["secrets"] += 1
            elif "duplicate" in reason.lower():
                rejection_reasons["duplicates"] += 1
            elif "leakage" in reason.lower():
                rejection_reasons["leakage"] += 1
            elif "Quality score" in reason:
                rejection_reasons["low_quality"] += 1
            elif "Code validation" in reason or "Syntax" in reason:
                rejection_reasons["invalid_code"] += 1
            elif "JSON" in reason:
                rejection_reasons["invalid_json"] += 1
            else:
                rejection_reasons["other"] += 1

        # Domain distribution
        balance = DatasetBalanceAuditor.analyze_distribution(accepted_examples)

        # Source task coverage
        unique_sources = len({e.provenance.source_task_id for e in accepted_examples if e.provenance})

        # Augmentation distribution
        augmentations = Counter(
            e.provenance.augmentation_type or "canonical"
            for e in accepted_examples
            if e.provenance
        )

        return {
            "summary": {
                "total_generated": total_generated,
                "total_accepted": total_accepted,
                "total_rejected": total_rejected,
                "acceptance_rate_percentage": round((total_accepted / total_generated * 100), 2) if total_generated > 0 else 0.0,
                "average_quality_score": avg_quality,
            },
            "splits": {
                "train_count": len(train_examples),
                "train_percentage": round(len(train_examples) / total_accepted * 100, 2) if total_accepted > 0 else 0.0,
                "validation_count": len(val_examples),
                "validation_percentage": round(len(val_examples) / total_accepted * 100, 2) if total_accepted > 0 else 0.0,
                "test_count": len(test_examples),
                "test_percentage": round(len(test_examples) / total_accepted * 100, 2) if total_accepted > 0 else 0.0,
            },
            "quality_distribution": quality_distribution,
            "rejections": dict(rejection_reasons),
            "domains": balance.get("by_task_type", {}),
            "difficulties": balance.get("by_difficulty", {}),
            "output_types": balance.get("by_output_type", {}),
            "root_causes": balance.get("by_root_cause", {}),
            "provenance": {
                "unique_source_candidates_covered": unique_sources,
                "coverage_of_40_candidates_percentage": round(unique_sources / 40 * 100, 2),
                "augmentation_breakdown": dict(augmentations),
            },
            "balance_status": {
                "is_balanced": balance.get("is_balanced", False),
                "max_domain_share_percentage": balance.get("max_domain_share_percentage", 0.0),
            },
        }
