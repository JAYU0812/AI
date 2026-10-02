"""Factual Evidence Extractor for Benchmark Failures.

Parses execution traces, assertion outcomes, and metric observations to extract
concrete, non-fabricated evidence supporting root-cause classifications.
"""

from typing import Any, Dict, List, Optional


class EvidenceExtractor:
    """Extracts factual diagnostic evidence from task execution records."""

    @classmethod
    def extract_evidence(cls, task_data: Dict[str, Any]) -> List[str]:
        """Gathers all empirical observations and assertion details for a task."""
        evidence: List[str] = []
        status = task_data.get("status", "UNKNOWN")
        score = task_data.get("score_percentage", 0.0)
        category = task_data.get("category", "")
        f_type = task_data.get("failure_type", "none")
        metrics = task_data.get("metrics", {})
        assertions = task_data.get("assertion_results", [])

        # 1. Assertion failure analysis
        failed_assertions = [a for a in assertions if not a.get("passed", False)]
        passed_assertions = [a for a in assertions if a.get("passed", False)]

        if failed_assertions:
            for fa in failed_assertions:
                a_name = fa.get("assertion", "Assertion")
                msg = fa.get("message", "Failed")
                evidence.append(f"{a_name} failed: {msg}")
        elif status == "PASS":
            evidence.append(f"All {len(assertions)} assertions passed with score {score:.1f}%")

        # 2. Metrics & Execution observations
        tool_calls = metrics.get("tool_calls", 0)
        files_modified = metrics.get("files_modified", 0)
        repairs = metrics.get("repair_attempts", 0)

        if tool_calls == 0 and category in ["code_generation", "code_modification", "multi_step"]:
            evidence.append(f"No tool calls executed (tool_calls=0) despite task requirement")
        elif tool_calls > 0:
            evidence.append(f"Executed {tool_calls} tool actions, modified {files_modified} files")

        if repairs > 0:
            evidence.append(f"Triggered {repairs} repair attempts during execution")

        # 3. Dimensional anomalies
        dims = task_data.get("dimensions", {})
        for dim_name, val in dims.items():
            if val == 0:
                evidence.append(f"Dimension '{dim_name}' scored 0 (complete failure)")

        # 4. Failure type context
        if f_type != "none":
            evidence.append(f"Reported failure taxonomy: {f_type}")

        if not evidence:
            evidence.append(f"Task executed with status={status} and score={score}%")

        return evidence
