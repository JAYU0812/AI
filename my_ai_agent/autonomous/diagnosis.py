"""Error Diagnosis Engine for Step 3D.

Performs deterministic, evidence-based error classification and diagnostic extraction
from tracebacks, test outputs, compiler errors, and runtime failures.
Never invents causes without concrete textual evidence.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


class ErrorCategory(str, Enum):
    SYNTAX_ERROR = "SYNTAX_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    TYPE_ERROR = "TYPE_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TEST_FAILURE = "TEST_FAILURE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PERMISSION_ERROR = "PERMISSION_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class DiagnosticReport:
    """Structured, evidence-backed diagnostic analysis of an error."""

    category: ErrorCategory
    error_message: str
    affected_file: Optional[str] = None
    affected_line: Optional[int] = None
    failing_symbol: Optional[str] = None
    probable_cause: str = ""
    evidence: str = ""
    suggested_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value if isinstance(self.category, ErrorCategory) else str(self.category),
            "error_message": self.error_message,
            "affected_file": self.affected_file,
            "affected_line": self.affected_line,
            "failing_symbol": self.failing_symbol,
            "probable_cause": self.probable_cause,
            "evidence": self.evidence,
            "suggested_action": self.suggested_action,
        }


class DiagnosisEngine:
    """Parses real error outputs, tracebacks, and test failures into DiagnosticReports."""

    @classmethod
    def diagnose(cls, error_output: str, workspace_root: Optional[Path] = None) -> DiagnosticReport:
        """Analyze combined stdout/stderr to categorize error and pinpoint cause."""
        text = error_output.strip()
        if not text:
            return DiagnosticReport(
                category=ErrorCategory.UNKNOWN_ERROR,
                error_message="Empty error output",
                probable_cause="Process failed with empty error stream",
            )

        # 1. Syntax Errors
        syntax_match = re.search(r'File "([^"]+)", line (\d+)(?:, in (.+))?\n(?:\s+.*\n)?\s*(\w*SyntaxError.*)', text)
        if syntax_match or "SyntaxError:" in text:
            file_p = syntax_match.group(1) if syntax_match else cls._find_file(text, workspace_root)
            line_no = int(syntax_match.group(2)) if syntax_match else cls._find_line(text)
            err_line = syntax_match.group(4) if syntax_match else cls._extract_error_line(text, "SyntaxError")
            return DiagnosticReport(
                category=ErrorCategory.SYNTAX_ERROR,
                error_message=err_line,
                affected_file=file_p,
                affected_line=line_no,
                probable_cause="Malformed code or invalid Python grammar",
                evidence=err_line,
                suggested_action="Repair syntax at specified line",
            )

        # 2. Import / Dependency Errors
        import_match = re.search(r"(?:ModuleNotFoundError|ImportError):\s*([^\n]+)", text)
        if import_match or "ModuleNotFoundError:" in text:
            err_msg = import_match.group(0) if import_match else "Module import failure"
            missing_module = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
            mod_name = missing_module.group(1) if missing_module else ""
            file_p = cls._find_file(text, workspace_root)
            line_no = cls._find_line(text)

            cat = ErrorCategory.DEPENDENCY_ERROR if mod_name and not mod_name.startswith(".") else ErrorCategory.IMPORT_ERROR
            return DiagnosticReport(
                category=cat,
                error_message=err_msg,
                affected_file=file_p,
                affected_line=line_no,
                failing_symbol=mod_name,
                probable_cause=f"Required package '{mod_name}' is not installed or import path is invalid" if mod_name else "Import statement failed",
                evidence=err_msg,
                suggested_action=f"Install dependency '{mod_name}' or fix import path",
            )

        # 3. Test Failures & Assertion Errors
        if "AssertionError" in text or "FAILED (" in text or "FAIL:" in text or "assert " in text:
            file_p = cls._find_test_file(text, workspace_root)
            line_no = cls._find_line(text)
            assert_detail = cls._extract_assertion_detail(text)
            failing_func = cls._find_failing_function(text)

            return DiagnosticReport(
                category=ErrorCategory.TEST_FAILURE,
                error_message=assert_detail or "Assertion failed during test execution",
                affected_file=file_p,
                affected_line=line_no,
                failing_symbol=failing_func,
                probable_cause=f"Logical mismatch in '{failing_func or 'function'}': assertion failed" if assert_detail else "Test assertion failed",
                evidence=assert_detail or cls._extract_snippet(text),
                suggested_action="Modify function implementation to produce expected return value",
            )

        # 4. File Not Found Errors
        fnf_match = re.search(r"(?:FileNotFoundError|NoSuchFileException):\s*([^\n]+)", text)
        if fnf_match:
            return DiagnosticReport(
                category=ErrorCategory.FILE_NOT_FOUND,
                error_message=fnf_match.group(0),
                affected_file=cls._find_file(text, workspace_root),
                probable_cause="Target file does not exist on disk",
                evidence=fnf_match.group(0),
                suggested_action="Create missing file or verify path",
            )

        # 5. Type Errors
        type_match = re.search(r"TypeError:\s*([^\n]+)", text)
        if type_match:
            return DiagnosticReport(
                category=ErrorCategory.TYPE_ERROR,
                error_message=type_match.group(0),
                affected_file=cls._find_file(text, workspace_root),
                affected_line=cls._find_line(text),
                probable_cause="Function received incompatible operand or argument types",
                evidence=type_match.group(0),
                suggested_action="Fix argument types or type conversions",
            )

        # 6. General Runtime Errors
        runtime_match = re.search(r"(?:NameError|ValueError|KeyError|AttributeError|ZeroDivisionError):\s*([^\n]+)", text)
        if runtime_match:
            file_p = cls._find_file(text, workspace_root)
            line_no = cls._find_line(text)
            return DiagnosticReport(
                category=ErrorCategory.RUNTIME_ERROR,
                error_message=runtime_match.group(0),
                affected_file=file_p,
                affected_line=line_no,
                probable_cause=f"Runtime exception: {runtime_match.group(0)}",
                evidence=runtime_match.group(0),
                suggested_action="Fix variable name, attribute access, or runtime condition",
            )

        # 7. Fallback Unknown
        return DiagnosticReport(
            category=ErrorCategory.UNKNOWN_ERROR,
            error_message=text.splitlines()[-1] if text.splitlines() else "Unknown error",
            evidence=text[:300],
            probable_cause="Process exited with non-zero code",
            suggested_action="Inspect full execution output for clues",
        )

    # ── Internal extraction helpers ───────────────────────────────

    @classmethod
    def _find_file(cls, text: str, workspace_root: Optional[Path]) -> Optional[str]:
        # Look for File "path", line X
        matches = re.findall(r'File "([^"]+)", line \d+', text)
        if matches:
            # Prefer last file in traceback that is within workspace or a local file
            for f in reversed(matches):
                if "<" not in f and "site-packages" not in f and "lib\\" not in f.lower():
                    if workspace_root:
                        try:
                            rel = Path(f).resolve().relative_to(Path(workspace_root).resolve())
                            return str(rel).replace("\\", "/")
                        except Exception:
                            pass
                    return Path(f).name
            return Path(matches[-1]).name
        return None

    @classmethod
    def _find_test_file(cls, text: str, workspace_root: Optional[Path]) -> Optional[str]:
        file_from_tb = cls._find_file(text, workspace_root)
        if file_from_tb and "test" in file_from_tb.lower():
            return file_from_tb

        # Check pytest filename:line
        pytest_match = re.search(r"([\w_/-]+\.py):(\d+): in ", text)
        if pytest_match:
            return pytest_match.group(1)

        # Check unittest FAIL: test_xxx (test_file.Class)
        m = re.search(r"FAIL:\s+\w+\s+\(([\w\.]+)\)", text)
        if m:
            parts = m.group(1).split(".")
            for p in parts:
                if p.lower().startswith("test_") or p.lower().endswith("_test"):
                    return f"tests/{p}.py"
            mod = parts[0]
            return f"tests/{mod}.py" if not mod.endswith(".py") else mod

        return file_from_tb

    @classmethod
    def _find_line(cls, text: str) -> Optional[int]:
        lines = re.findall(r'line (\d+)', text)
        if lines:
            return int(lines[-1])
        py_lines = re.findall(r'\.py:(\d+):', text)
        if py_lines:
            return int(py_lines[-1])
        return None

    @classmethod
    def _extract_error_line(cls, text: str, error_name: str) -> str:
        for line in text.splitlines():
            if error_name in line:
                return line.strip()
        return error_name

    @classmethod
    def _extract_assertion_detail(cls, text: str) -> str:
        for line in text.splitlines():
            s = line.strip()
            if "AssertionError" in s:
                return s
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("assert ") or "assert False" in s:
                return s
        return ""

    @classmethod
    def _find_failing_function(cls, text: str) -> Optional[str]:
        # 1. Look for function call inside assert: self.assertEqual(divide(10, 2), 5) or assert divide(10, 2) == 5
        m_call = re.search(r"(?:assertEqual|assertTrue|assertFalse)\s*\(\s*([a-zA-Z0-9_]+)\(", text)
        if m_call and m_call.group(1) not in ("self", "assertEqual", "assertTrue", "assertFalse"):
            return m_call.group(1)

        m_assert = re.search(r"assert\s+([a-zA-Z0-9_]+)\(", text)
        if m_assert:
            return m_assert.group(1)

        # 2. Look for test method name: in test_xxx or FAIL: test_xxx
        m_fail = re.search(r"FAIL:\s+test_([a-zA-Z0-9_]+)", text)
        if m_fail:
            return m_fail.group(1)

        m = re.search(r"in (?:test_)?([a-zA-Z0-9_]+)", text)
        if m:
            return m.group(1)
        return None

    @classmethod
    def _extract_snippet(cls, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines[-4:]) if len(lines) >= 4 else "\n".join(lines)
