"""Cross-Category Pattern Detector and Failure Clustering (Step 6B).

Detects recurrent failure patterns across diverse benchmark categories, analyzes
difficulty correlations, and groups related failures into prioritized engineering clusters.
"""

from typing import Any, Dict, List

from my_ai_agent.analysis.root_cause import FailureCluster, FixType, RootCauseCategory, RootCauseRecord


class PatternDetector:
    """Discovers cross-category correlations, difficulty thresholds, and failure clusters."""

    @classmethod
    def analyze_by_difficulty(cls, records: List[RootCauseRecord]) -> Dict[str, Dict[str, Any]]:
        """Compares performance across Easy, Medium, and Hard difficulty levels."""
        diff_groups: Dict[str, List[RootCauseRecord]] = {}
        for r in records:
            diff_groups.setdefault(r.difficulty, []).append(r)

        analysis: Dict[str, Dict[str, Any]] = {}
        for diff_level, group in diff_groups.items():
            total = len(group)
            passed = sum(1 for r in group if r.status == "PASS")
            partial = sum(1 for r in group if r.status == "PARTIAL")
            failed = sum(1 for r in group if r.status == "FAIL")
            scores = [r.score_percentage for r in group]
            avg_score = round(sum(scores) / max(1, total), 2)
            pass_rate = round((passed / max(1, total)) * 100.0, 2)

            analysis[diff_level] = {
                "difficulty": diff_level,
                "total_tasks": total,
                "passed": passed,
                "partial": partial,
                "failed": failed,
                "average_score": avg_score,
                "pass_rate": pass_rate,
                "failure_rate": round(100.0 - pass_rate, 2),
            }

        return analysis

    @classmethod
    def cluster_failures(cls, records: List[RootCauseRecord]) -> List[FailureCluster]:
        """Groups failed and partial tasks into actionable failure clusters."""
        non_passing = [r for r in records if r.status != "PASS"]

        clusters: List[FailureCluster] = [
            FailureCluster(
                cluster_id="CLUSTER_1_STRUCTURED_INTERFACE",
                name="Direct Structured JSON Generation Failure",
                description="Direct invocation of generate_structured() returns unstructured text when rule engine lacks pattern match.",
                task_ids=[r.task_id for r in non_passing if r.primary_root_cause == RootCauseCategory.STRUCTURED_OUTPUT_INTERFACE],
                dominant_root_cause=RootCauseCategory.STRUCTURED_OUTPUT_INTERFACE,
                fix_type=FixType.STRUCTURED_OUTPUT_FIX,
                priority="P0",
            ),
            FailureCluster(
                cluster_id="CLUSTER_2_OPEN_ENDED_CODE_SYNTHESIS",
                name="Novel Logic & Arbitrary Module Generation Deficit",
                description="Model unable to invent arbitrary logic (e.g. factorial, reverse_string, converters) without pre-programmed rules.",
                task_ids=[r.task_id for r in non_passing if r.primary_root_cause == RootCauseCategory.MODEL_CAPABILITY and r.category == "code_generation"],
                dominant_root_cause=RootCauseCategory.MODEL_CAPABILITY,
                fix_type=FixType.FINE_TUNING_CANDIDATE,
                priority="P0",
            ),
            FailureCluster(
                cluster_id="CLUSTER_3_MULTI_STEP_COORDINATION",
                name="Multi-File Test Synchronization Incomplete",
                description="Autonomous engine executes single-file modification but leaves corresponding test updates incomplete.",
                task_ids=[r.task_id for r in non_passing if r.primary_root_cause == RootCauseCategory.AUTONOMOUS_ENGINE],
                dominant_root_cause=RootCauseCategory.AUTONOMOUS_ENGINE,
                fix_type=FixType.AGENT_ARCHITECTURE_FIX,
                priority="P1",
            ),
            FailureCluster(
                cluster_id="CLUSTER_4_INSTRUCTION_CONSTRAINT_VIOLATION",
                name="Explicit Single-File Constraint Non-Compliance",
                description="Agent creates default greeting files rather than honoring strict single-file named constraints.",
                task_ids=[r.task_id for r in non_passing if r.primary_root_cause == RootCauseCategory.PROMPT_DESIGN],
                dominant_root_cause=RootCauseCategory.PROMPT_DESIGN,
                fix_type=FixType.PROMPT_FIX,
                priority="P1",
            ),
            FailureCluster(
                cluster_id="CLUSTER_5_SAFETY_EVALUATOR_SIMULATION",
                name="Safety Evaluator Payload Simulation Deficit",
                description="Harness tests failed to trigger mock payload execution for abstract adversarial prompts (id_rsa, system hosts).",
                task_ids=[r.task_id for r in non_passing if r.primary_root_cause == RootCauseCategory.EVALUATOR_ERROR],
                dominant_root_cause=RootCauseCategory.EVALUATOR_ERROR,
                fix_type=FixType.EVALUATOR_FIX,
                priority="P2",
            ),
        ]

        # Filter out empty clusters and sort by impact
        active_clusters = [c for c in clusters if c.task_ids]
        active_clusters.sort(key=lambda c: len(c.task_ids), reverse=True)
        return active_clusters
