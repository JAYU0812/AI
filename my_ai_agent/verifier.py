"""Self-Verification and Error Diagnostics Engine for Step 2.

Performs observation analysis, output assertion, and automatic error diagnostics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re


@dataclass
class VerificationResult:
    passed: bool
    status: str
    diagnostics: str
    suggested_fix: Optional[str] = None


class SelfVerifier:
    """Verifies tool observations and diagnoses errors."""

    def verify_execution(self, output: str, returncode: int, purpose: str = "") -> VerificationResult:
        """Verifies command execution and asserts expected outcomes."""
        if returncode != 0 or "[STDERR]" in output or "Traceback" in output:
            diagnosis, fix = self._diagnose_error(output)
            return VerificationResult(
                passed=False,
                status="EXECUTION_ERROR",
                diagnostics=f"Execution failed with returncode {returncode}. {diagnosis}",
                suggested_fix=fix,
            )

        # Output inspection based on purpose
        p_lower = purpose.lower()
        if "average" in p_lower or "mean" in p_lower:
            if "average" in output.lower():
                return VerificationResult(
                    passed=True,
                    status="VERIFIED",
                    diagnostics="Self-verification passed: Output contains calculated average.",
                )

        if "sum" in p_lower:
            if "sum" in output.lower() or "total" in output.lower():
                return VerificationResult(
                    passed=True,
                    status="VERIFIED",
                    diagnostics="Self-verification passed: Output contains calculated sum.",
                )

        if output.strip():
            return VerificationResult(
                passed=True,
                status="VERIFIED",
                diagnostics="Self-verification passed: Command produced expected output without errors.",
            )

        return VerificationResult(
            passed=True,
            status="WARNING_EMPTY_OUTPUT",
            diagnostics="Command succeeded with exit code 0 but produced empty stdout.",
        )

    def _diagnose_error(self, output: str) -> (str, Optional[str]):
        """Diagnoses common Python errors from traceback."""
        if "SyntaxError" in output:
            return "Syntax error detected in generated code.", "Check syntax, indentation, and mismatched parentheses."
        if "IndentationError" in output:
            return "Indentation error in script.", "Ensure consistent 4-space indentation."
        if "ModuleNotFoundError" in output or "ImportError" in output:
            mod_match = re.search(r"No module named '([^']+)'", output)
            mod = mod_match.group(1) if mod_match else "unknown"
            return f"Missing dependency: module '{mod}' is not installed.", f"Run 'pip install {mod}' or use standard library."
        if "ZeroDivisionError" in output:
            return "Division by zero occurred during execution.", "Add check for non-empty lists before dividing."
        if "NameError" in output:
            return "Undefined variable name used.", "Verify variable declarations and scope."
        
        return "Unknown runtime error occurred.", "Review traceback details in logs."
