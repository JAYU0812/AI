"""Safe User-Facing Reasoning Summaries for Step 4.

CRITICAL: Do NOT expose raw/hidden chain-of-thought tokens or hallucinated
private rationale. Instead produce concise, action-oriented, factual explanations
of what was inspected, why a path was chosen, and what actions will be taken.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningSummary:
    """Safe, user-facing, action-oriented explanation of agent intent and strategy."""
    objective: str
    findings: List[str] = field(default_factory=list)
    decision: str = ""
    planned_actions: List[str] = field(default_factory=list)

    def to_user_message(self) -> str:
        """Format as a clear, polite, transparent agent explanation."""
        lines = []
        if self.findings:
            lines.append("I inspected your project:")
            for f in self.findings:
                lines.append(f"  • {f}")
        if self.decision:
            lines.append(f"\nStrategy: {self.decision}")
        if self.planned_actions:
            lines.append("\nNext Steps:")
            for idx, a in enumerate(self.planned_actions, 1):
                lines.append(f"  {idx}. {a}")
        return "\n".join(lines).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "findings": self.findings,
            "decision": self.decision,
            "planned_actions": self.planned_actions,
        }

    @classmethod
    def synthesize_from_context(
        cls,
        objective: str,
        project_type: str,
        identified_files: List[str],
        identified_symbols: List[str],
        has_tests: bool,
    ) -> "ReasoningSummary":
        """Generate a factual reasoning summary from verified project context."""
        findings = []
        if project_type:
            findings.append(f"Detected {project_type} architecture")
        if identified_files:
            file_names = ", ".join(f"`{f}`" for f in identified_files[:3])
            findings.append(f"Located relevant files: {file_names}")
        if identified_symbols:
            sym_names = ", ".join(f"`{s}`" for s in identified_symbols[:3])
            findings.append(f"Located key symbols: {sym_names}")
        if has_tests:
            findings.append("Existing test suite identified for automated verification")

        decision = (
            f"Modifying existing codebase structures in {', '.join(identified_files[:2])} "
            f"to satisfy '{objective}' while preserving existing architecture."
            if identified_files else
            f"Creating new module structure to implement '{objective}' with unit tests."
        )

        steps = [
            f"Update implementation in {identified_files[0] if identified_files else 'target file'}",
            "Verify syntax and compatibility via AST analysis",
            "Run project test suite",
            "Perform evidence-based verification",
        ]

        return cls(
            objective=objective,
            findings=findings,
            decision=decision,
            planned_actions=steps,
        )
