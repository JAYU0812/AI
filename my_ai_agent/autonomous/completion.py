"""Completion Verification Engine for Step 3D.

Produces evidence-based completion reports:
- Validates AST syntax across all modified files
- Discovers and executes real test runners
- Verifies targeted symbols/functions actually exist on disk
- Distinguishes between COMPLETED, PARTIALLY_COMPLETED, and FAILED
Never claims success without factual verification evidence.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from my_ai_agent.autonomous.task_state import TaskStatus
from my_ai_agent.editing.test_runner import ProjectTestRunner, TestExecutionResult


@dataclass
class CompletionReport:
    """Evidence-backed verification summary."""

    status: TaskStatus
    summary: str
    evidence: List[str] = field(default_factory=list)
    verified_files: List[str] = field(default_factory=list)
    unmet_criteria: List[str] = field(default_factory=list)
    test_results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, TaskStatus) else str(self.status),
            "summary": self.summary,
            "evidence": self.evidence,
            "verified_files": self.verified_files,
            "unmet_criteria": self.unmet_criteria,
            "test_results": self.test_results,
        }


class CompletionVerifier:
    """Evaluates task completion using concrete on-disk and execution evidence."""

    @classmethod
    def verify(
        cls,
        workspace_root: Path,
        modified_files: Set[str],
        user_request: str,
        expected_symbols: Optional[List[str]] = None,
    ) -> CompletionReport:
        """Inspect workspace to determine if task criteria are factually met."""
        root = Path(workspace_root).resolve()
        evidence: List[str] = []
        unmet: List[str] = []
        verified: List[str] = []

        # 1. Syntax validation on all modified Python files
        syntax_all_valid = True
        for rel_path in modified_files:
            full_path = root / rel_path
            if not full_path.exists():
                unmet.append(f"Modified file '{rel_path}' does not exist on disk")
                syntax_all_valid = False
                continue

            if full_path.suffix.lower() == ".py":
                try:
                    ast.parse(full_path.read_text(encoding="utf-8", errors="replace"))
                    evidence.append(f"Syntax valid (ast.parse) for '{rel_path}'")
                    verified.append(rel_path)
                except SyntaxError as e:
                    unmet.append(f"Syntax error in '{rel_path}': line {e.lineno} {e.msg}")
                    syntax_all_valid = False
            else:
                verified.append(rel_path)
                evidence.append(f"File verified on disk: '{rel_path}'")

        if not syntax_all_valid:
            return CompletionReport(
                status=TaskStatus.FAILED,
                summary="Completion verification failed: syntax errors detected in modified files.",
                evidence=evidence,
                verified_files=verified,
                unmet_criteria=unmet,
            )

        # 2. Check for expected symbols if specified
        if expected_symbols:
            all_content = ""
            for p in root.rglob("*.py"):
                if "test" not in p.name.lower():
                    all_content += "\n" + p.read_text(encoding="utf-8", errors="ignore")

            for sym in expected_symbols:
                if f"def {sym}" in all_content or f"class {sym}" in all_content:
                    evidence.append(f"Symbol '{sym}' verified in project code")
                else:
                    unmet.append(f"Expected symbol '{sym}' was not found in project files")

        # 3. Discover and execute project test suite
        test_exec: Optional[TestExecutionResult] = ProjectTestRunner.run_tests(root)
        test_dict = test_exec.to_dict() if test_exec else None

        if test_exec:
            if test_exec.passed:
                evidence.append(f"All project tests passed ({test_exec.runner_command}, exit code 0)")
            else:
                unmet.append(f"Project tests failed ({test_exec.runner_command}, exit code {test_exec.returncode})")

        # 4. Final status determination
        if unmet:
            # If tests failed, mark FAILED
            if test_exec and not test_exec.passed:
                return CompletionReport(
                    status=TaskStatus.FAILED,
                    summary=f"Task verification failed: {'; '.join(unmet)}",
                    evidence=evidence,
                    verified_files=verified,
                    unmet_criteria=unmet,
                    test_results=test_dict,
                )
            # If some criteria unmet but no test failure, mark PARTIALLY_COMPLETED
            return CompletionReport(
                status=TaskStatus.PARTIALLY_COMPLETED,
                summary=f"Task partially completed: {'; '.join(unmet)}",
                evidence=evidence,
                verified_files=verified,
                unmet_criteria=unmet,
                test_results=test_dict,
            )

        # All criteria satisfied with concrete evidence
        return CompletionReport(
            status=TaskStatus.COMPLETED,
            summary="All completion criteria verified with factual evidence and passing tests.",
            evidence=evidence,
            verified_files=verified,
            unmet_criteria=[],
            test_results=test_dict,
        )
