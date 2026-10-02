"""Remediation Recommendations & Priority Matrix Generator (Step 6B).

Synthesizes diagnostic findings into an actionable engineering priority matrix,
and cleanly distinguishes fine-tuning training candidates from architectural fixes.
"""

from typing import Any, Dict, List

from my_ai_agent.analysis.root_cause import FailureCluster, FixType, RootCauseRecord


class RecommendationEngine:
    """Generates remediation priority matrices and filters training candidates."""

    @classmethod
    def build_priority_matrix(cls, clusters: List[FailureCluster]) -> List[Dict[str, Any]]:
        """Constructs an engineering priority matrix ranked by impact and severity."""
        matrix: List[Dict[str, Any]] = []
        for cluster in clusters:
            matrix.append({
                "problem": cluster.name,
                "impact_tasks": len(cluster.task_ids),
                "dominant_root_cause": cluster.dominant_root_cause.value,
                "fix_type": cluster.fix_type.value,
                "priority": cluster.priority,
                "description": cluster.description,
                "sample_tasks": cluster.task_ids[:5],
            })
        return matrix

    @classmethod
    def filter_training_candidates(cls, records: List[RootCauseRecord]) -> List[Dict[str, Any]]:
        """Extracts legitimate fine-tuning training candidates (unambiguous, model-improvable)."""
        candidates: List[Dict[str, Any]] = []
        for r in records:
            if r.is_training_candidate:
                candidates.append({
                    "task_id": r.task_id,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "score_percentage": r.score_percentage,
                    "primary_root_cause": r.primary_root_cause.value,
                    "recommended_action": r.recommended_action,
                    "evidence": r.evidence[:3],
                })
        return candidates

    @classmethod
    def filter_non_training_issues(cls, records: List[RootCauseRecord]) -> List[Dict[str, Any]]:
        """Extracts failures that must NOT be solved with model training (architecture/evaluator bugs)."""
        non_training: List[Dict[str, Any]] = []
        for r in records:
            if r.status != "PASS" and not r.is_training_candidate:
                non_training.append({
                    "task_id": r.task_id,
                    "category": r.category,
                    "primary_root_cause": r.primary_root_cause.value,
                    "non_training_reason": r.non_training_reason or "Architectural, evaluator, or harness issue",
                    "fix_type": r.fix_type.value,
                    "recommended_action": r.recommended_action,
                })
        return non_training
