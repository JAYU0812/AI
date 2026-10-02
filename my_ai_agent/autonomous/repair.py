"""Repair Engine for Step 3D.

Reuses the Step 3C Patch Engine to construct surgical, evidence-backed
code repairs from DiagnosticReports.
Never blindly overwrites whole files; generates targeted EditOperations.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from my_ai_agent.autonomous.diagnosis import DiagnosticReport, ErrorCategory
from my_ai_agent.editing.code_locator import CodeLocator
from my_ai_agent.editing.diff_generator import DiffResult
from my_ai_agent.editing.edit_operations import (
    EditOperation,
    EditOperationType,
    EditProposal,
    FileEditProposal,
)
from my_ai_agent.editing.patch_engine import PatchEngine, TransactionResult


@dataclass
class RepairResult:
    """Outcome of an autonomous repair attempt."""

    success: bool
    modified_files: List[str] = field(default_factory=list)
    diffs: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "modified_files": self.modified_files,
            "diffs": self.diffs,
            "error": self.error,
            "summary": self.summary,
        }


class RepairEngine:
    """Synthesizes and applies surgical code corrections using Step 3C PatchEngine."""

    @classmethod
    def attempt_repair(
        cls,
        report: DiagnosticReport,
        workspace_root: Path,
        patch_engine: PatchEngine,
    ) -> RepairResult:
        """Create and apply a corrective patch based on the diagnosis."""
        root = Path(workspace_root).resolve()

        # 1. Determine target file to repair
        target_file = cls._identify_target_file(report, root)
        if not target_file:
            return RepairResult(
                success=False,
                error="Could not locate source file requiring repair",
            )

        full_path = root / target_file
        if not full_path.exists():
            return RepairResult(
                success=False,
                error=f"Target file '{target_file}' does not exist on disk",
            )

        code = full_path.read_text(encoding="utf-8", errors="replace")

        # 2. Formulate targeted EditProposal based on diagnosis
        proposal = cls._formulate_repair_proposal(report, target_file, code)
        if not proposal or not proposal.files:
            return RepairResult(
                success=False,
                error=f"Could not construct a deterministic repair for {report.category.value}: {report.error_message}",
            )

        # 3. Apply via Step 3C PatchEngine (guarantees validation and rollback)
        tx_res: TransactionResult = patch_engine.apply_proposal(proposal)

        diffs_dict = [d.to_dict() for d in tx_res.diffs] if tx_res.diffs else []
        return RepairResult(
            success=tx_res.success,
            modified_files=tx_res.modified_files,
            diffs=diffs_dict,
            error="; ".join(tx_res.errors) if tx_res.errors else "",
            summary=tx_res.summary,
        )

    # ── Internal Repair Strategies ────────────────────────────────

    @classmethod
    def _identify_target_file(cls, report: DiagnosticReport, root: Path) -> Optional[str]:
        """Distinguish between test file (where assertion failed) and implementation file (where bug lives)."""
        aff = report.affected_file
        if not aff:
            # Search workspace for source files matching failing symbol
            if report.failing_symbol:
                for py_file in root.rglob("*.py"):
                    if "test" not in py_file.name.lower():
                        content = py_file.read_text(encoding="utf-8", errors="ignore")
                        if f"def {report.failing_symbol}" in content:
                            return str(py_file.relative_to(root)).replace("\\", "/")
            return None

        aff_path = Path(aff)
        # If affected file is a test file, find corresponding implementation file
        if "test" in aff_path.name.lower():
            # E.g. tests/test_calculator.py -> calculator.py
            stem = aff_path.stem
            clean_stem = re.sub(r"(?i)^test_?", "", stem)
            clean_stem = re.sub(r"(?i)_?test$", "", clean_stem)
            for candidate in root.rglob(f"{clean_stem.lower()}.py"):
                if "test" not in candidate.name.lower():
                    return str(candidate.relative_to(root)).replace("\\", "/")
            for candidate in root.rglob(f"{clean_stem}.py"):
                if "test" not in candidate.name.lower():
                    return str(candidate.relative_to(root)).replace("\\", "/")

        # Check if failing symbol is in aff file; if not, find the file that contains it
        if report.failing_symbol:
            full_aff = root / aff
            if full_aff.exists():
                code = full_aff.read_text(encoding="utf-8", errors="ignore")
                if f"def {report.failing_symbol}" in code:
                    return aff

            for py_file in root.rglob("*.py"):
                if "test" not in py_file.name.lower():
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    if f"def {report.failing_symbol}" in content:
                        return str(py_file.relative_to(root)).replace("\\", "/")

        return aff

    @classmethod
    def _formulate_repair_proposal(
        cls,
        report: DiagnosticReport,
        target_file: str,
        code: str,
    ) -> Optional[EditProposal]:
        """Formulate a targeted surgical edit for the diagnosed bug."""
        ops: List[EditOperation] = []

        # Case 1: Test Failure on Arithmetic / Logic operation (e.g. divide using * instead of /)
        if report.category == ErrorCategory.TEST_FAILURE:
            # Check for common operator bugs in the failing function
            fn_name = report.failing_symbol
            if not fn_name:
                # Try finding from error message
                m = re.search(r"(\w+)\(", report.error_message)
                if m:
                    fn_name = m.group(1)

            if fn_name:
                loc = CodeLocator.locate_target(code, target_symbol=fn_name)
                if loc.found and loc.matched_text:
                    orig_func = loc.matched_text
                    fixed_func = cls._repair_function_body(orig_func, fn_name, report.evidence)
                    if fixed_func and fixed_func != orig_func:
                        ops.append(EditOperation(
                            operation_type=EditOperationType.REPLACE,
                            target_file=target_file,
                            target_symbol=fn_name,
                            content=fixed_func,
                            description=f"Fix logic in {fn_name} based on test failure",
                        ))

        # Case 2: Import / Module Error
        elif report.category in (ErrorCategory.IMPORT_ERROR, ErrorCategory.DEPENDENCY_ERROR):
            mod = report.failing_symbol
            if mod:
                ops.append(EditOperation(
                    operation_type=EditOperationType.INSERT,
                    target_file=target_file,
                    content="",
                    required_imports=[f"import {mod}"],
                    description=f"Add missing import for {mod}",
                ))

        # Case 3: Syntax Error (repair line or fix unclosed parenthesis/quotes)
        elif report.category == ErrorCategory.SYNTAX_ERROR and report.affected_line:
            lines = code.splitlines()
            line_idx = report.affected_line - 1
            if 0 <= line_idx < len(lines):
                bad_line = lines[line_idx]
                fixed_line = cls._repair_syntax_line(bad_line)
                if fixed_line != bad_line:
                    ops.append(EditOperation(
                        operation_type=EditOperationType.REPLACE,
                        target_file=target_file,
                        line_number=report.affected_line,
                        content=fixed_line,
                        description="Fix syntax error on line",
                    ))

        # Case 4: Attribute / Name Error (e.g. missing attribute initialization)
        elif report.category == ErrorCategory.RUNTIME_ERROR:
            attr_m = re.search(r"has no attribute ['\"]([^'\"]+)['\"]", report.error_message)
            if attr_m:
                attr_name = attr_m.group(1)
                # If inside a class __init__, initialize attribute
                if "def __init__" in code:
                    ops.append(EditOperation(
                        operation_type=EditOperationType.INSERT,
                        target_file=target_file,
                        target_symbol="__init__",
                        content=f"self.{attr_name} = None",
                        position="after",
                        description=f"Initialize missing attribute self.{attr_name}",
                    ))

        if not ops:
            return None

        return EditProposal(
            intent="AUTO_REPAIR",
            description=f"Autonomous fix for {report.category.value} in {target_file}",
            files=[FileEditProposal(path=target_file, operations=ops)],
        )

    @classmethod
    def _repair_function_body(cls, func_code: str, func_name: str, evidence: str) -> str:
        """Fix logical operator in function code based on semantic intent and evidence."""
        # 1. Division bug: def divide(a, b): return a * b -> return a / b or a // b
        if "divide" in func_name.lower() or "div" in func_name.lower():
            if "*" in func_code:
                # Replace * with /
                return re.sub(r"([a-zA-Z0-9_]+)\s*\*\s*([a-zA-Z0-9_]+)", r"\1 / \2", func_code)
            if "+" in func_code:
                return re.sub(r"([a-zA-Z0-9_]+)\s*\+\s*([a-zA-Z0-9_]+)", r"\1 / \2", func_code)

        # 2. Subtraction bug: def subtract(a, b): return a + b -> return a - b
        if "sub" in func_name.lower() or "minus" in func_name.lower():
            if "+" in func_code:
                return re.sub(r"([a-zA-Z0-9_]+)\s*\+\s*([a-zA-Z0-9_]+)", r"\1 - \2", func_code)

        # 3. Multiplication bug: def multiply(a, b): return a + b -> return a * b
        if "mul" in func_name.lower() or "times" in func_name.lower():
            if "+" in func_code:
                return re.sub(r"([a-zA-Z0-9_]+)\s*\+\s*([a-zA-Z0-9_]+)", r"\1 * \2", func_code)

        # 4. Addition bug: def add(a, b): return a - b -> return a + b
        if "add" in func_name.lower() or "sum" in func_name.lower():
            if "-" in func_code:
                return re.sub(r"([a-zA-Z0-9_]+)\s*-\s*([a-zA-Z0-9_]+)", r"\1 + \2", func_code)

        return func_code

    @classmethod
    def _repair_syntax_line(cls, line: str) -> str:
        """Attempt common syntax repairs on a single line."""
        # Missing colon on def / class / if / for / while
        s = line.rstrip()
        if re.match(r"^\s*(?:def|class|if|elif|else|for|while|try|except|finally|with)\b", s):
            if not s.endswith(":"):
                return s + ":"
        return line
