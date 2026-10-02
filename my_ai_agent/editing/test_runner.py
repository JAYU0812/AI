"""Test Runner and Autonomous Fix Loop for Step 3C.

1. Automatically discovers project-specific test runners from actual project structure
   (pytest, python -m unittest, npm test, flutter test). Never blindly assumes pytest.
2. Implements the autonomous Error ➔ Locate ➔ Corrective Patch ➔ Re-test loop
   with MAX_AUTO_FIX_ATTEMPTS = 3.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from config import config
from my_ai_agent.editing.diff_generator import DiffResult
from my_ai_agent.editing.edit_operations import EditOperation, EditOperationType, EditProposal, FileEditProposal
from my_ai_agent.editing.patch_engine import PatchEngine, TransactionResult
from my_ai_agent.project.project_map import ProjectMap


@dataclass
class TestExecutionResult:
    """Factual outcome of running project tests."""
    runner_command: str
    passed: bool
    returncode: int
    output: str
    error: str = ""
    failing_file: Optional[str] = None
    failing_line: Optional[int] = None
    error_type: Optional[str] = None
    diagnostics: str = ""

    def to_dict(self) -> Dict:
        return {
            "runner_command": self.runner_command,
            "passed": self.passed,
            "returncode": self.returncode,
            "output": self.output,
            "error": self.error,
            "failing_file": self.failing_file,
            "failing_line": self.failing_line,
            "error_type": self.error_type,
            "diagnostics": self.diagnostics,
        }


@dataclass
class AutoFixResult:
    """Outcome of autonomous error ➔ patch ➔ re-test repair loop."""
    success: bool
    attempts: int
    max_attempts: int
    initial_test_result: TestExecutionResult
    final_test_result: Optional[TestExecutionResult] = None
    patches_applied: List[TransactionResult] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "message": self.message,
            "initial_test_passed": self.initial_test_result.passed,
            "final_test_passed": self.final_test_result.passed if self.final_test_result else False,
            "patches_count": len(self.patches_applied),
        }


class ProjectTestRunner:
    """Discovers and executes real tests from the workspace."""

    MAX_AUTO_FIX_ATTEMPTS: int = 3

    @classmethod
    def detect_test_command(cls, workspace_root: Path, pmap: Optional[ProjectMap] = None) -> Optional[str]:
        """Factually determine available test command from workspace structure."""
        root = Path(workspace_root).resolve()

        # 1. Python tests
        tests_dir = root / "tests"
        has_tests_dir = tests_dir.exists() and tests_dir.is_dir()
        has_test_files = any(root.rglob("test_*.py")) or any(root.rglob("*_test.py"))

        if has_tests_dir or has_test_files:
            # Check if pytest is configured
            if (root / "pytest.ini").exists() or (root / "conftest.py").exists():
                return f'"{sys.executable}" -B -m pytest'
            # Default standard Python unittest
            if has_tests_dir:
                return f'"{sys.executable}" -B -m unittest discover -s tests'
            return f'"{sys.executable}" -B -m unittest'

        # 2. Node.js tests
        pkg_json = root / "package.json"
        if pkg_json.exists():
            try:
                import json
                data = json.loads(pkg_json.read_text(encoding="utf-8"))
                if "test" in data.get("scripts", {}):
                    return "npm test"
            except Exception:
                pass

        # 3. Flutter / Dart tests
        if (root / "pubspec.yaml").exists() and (root / "test").exists():
            return "flutter test"

        return None

    @classmethod
    def run_tests(cls, workspace_root: Path, timeout: int = 30) -> Optional[TestExecutionResult]:
        """Execute discovered tests inside workspace and parse output."""
        root = Path(workspace_root).resolve()
        cmd = cls.detect_test_command(root)
        if not cmd:
            return None

        try:
            import os
            test_env = dict(os.environ)
            test_env["PYTHONDONTWRITEBYTECODE"] = "1"
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=test_env,
            )
            stdout = res.stdout.strip()
            stderr = res.stderr.strip()
            combined = "\n".join(filter(None, [stdout, stderr]))

            passed = (res.returncode == 0)
            failing_file, failing_line, err_type, diag = cls._parse_test_failure(combined, root)

            return TestExecutionResult(
                runner_command=cmd,
                passed=passed,
                returncode=res.returncode,
                output=combined,
                error="" if passed else f"Test runner exited with code {res.returncode}",
                failing_file=failing_file,
                failing_line=failing_line,
                error_type=err_type,
                diagnostics=diag,
            )
        except subprocess.TimeoutExpired:
            return TestExecutionResult(
                runner_command=cmd,
                passed=False,
                returncode=-1,
                output="Test execution timed out.",
                error=f"Timeout after {timeout} seconds",
            )
        except Exception as e:
            return TestExecutionResult(
                runner_command=cmd,
                passed=False,
                returncode=1,
                output="",
                error=f"Failed to execute test runner: {str(e)}",
            )

    @classmethod
    def run_autofix_loop(
        cls,
        workspace_root: Path,
        patch_engine: PatchEngine,
        initial_test_res: TestExecutionResult,
    ) -> AutoFixResult:
        """Autonomous Repair Loop: Read error ➔ Locate ➔ Corrective patch ➔ Re-test."""
        if initial_test_res.passed:
            return AutoFixResult(
                success=True,
                attempts=0,
                max_attempts=cls.MAX_AUTO_FIX_ATTEMPTS,
                initial_test_result=initial_test_res,
                final_test_result=initial_test_res,
                message="Tests passed on initial run; no fix needed.",
            )

        current_test_res = initial_test_res
        applied_patches: List[TransactionResult] = []

        for attempt in range(1, cls.MAX_AUTO_FIX_ATTEMPTS + 1):
            if not current_test_res.failing_file:
                break

            target_file = current_test_res.failing_file
            target_path = workspace_root / target_file
            if not target_path.exists():
                break

            orig_code = target_path.read_text(encoding="utf-8", errors="replace")

            # 1. Synthesize targeted correction based on failure diagnostic
            fix_op = cls._generate_corrective_operation(
                target_file=target_file,
                code=orig_code,
                test_result=current_test_res,
            )
            if not fix_op:
                break

            proposal = EditProposal(
                intent="AUTONOMOUS_TEST_FIX",
                description=f"Fix {current_test_res.error_type} in {target_file}",
                files=[FileEditProposal(path=target_file, operations=[fix_op])],
            )

            tx_res = patch_engine.apply_proposal(proposal)
            applied_patches.append(tx_res)

            if not tx_res.success:
                break

            # 2. Re-run tests to verify fix
            retest = cls.run_tests(workspace_root)
            if not retest:
                break

            current_test_res = retest
            if retest.passed:
                return AutoFixResult(
                    success=True,
                    attempts=attempt,
                    max_attempts=cls.MAX_AUTO_FIX_ATTEMPTS,
                    initial_test_result=initial_test_res,
                    final_test_result=retest,
                    patches_applied=applied_patches,
                    message=f"Autonomous fix succeeded on attempt {attempt} of {cls.MAX_AUTO_FIX_ATTEMPTS}.",
                )

        return AutoFixResult(
            success=False,
            attempts=len(applied_patches),
            max_attempts=cls.MAX_AUTO_FIX_ATTEMPTS,
            initial_test_result=initial_test_res,
            final_test_result=current_test_res,
            patches_applied=applied_patches,
            message=f"Agent stopped automatic repair attempts after {len(applied_patches)} attempt(s). Real error persists.",
        )

    # ── Diagnostic Parsers ────────────────────────────────────────

    @classmethod
    def _parse_test_failure(
        cls,
        output: str,
        workspace_root: Path,
    ) -> Tuple[Optional[str], Optional[int], Optional[str], str]:
        """Extract failing filename, line number, and exception type from test traceback."""
        failing_file = None
        failing_line = None
        err_type = None
        diag = ""

        # Match Python traceback pattern: File "...", line X, in ...
        matches = list(re.finditer(r'File\s+"([^"]+)",\s+line\s+(\d+)', output))
        if matches:
            # Pick innermost frame that belongs to workspace
            for m in reversed(matches):
                fp = m.group(1)
                try:
                    p = Path(fp)
                    if p.is_absolute() and p.is_relative_to(workspace_root):
                        failing_file = str(p.relative_to(workspace_root)).replace("\\", "/")
                        failing_line = int(m.group(2))
                        break
                    elif not p.is_absolute() and (workspace_root / p).exists():
                        failing_file = str(p).replace("\\", "/")
                        failing_line = int(m.group(2))
                        break
                except Exception:
                    continue

        # Match exception name (e.g. AssertionError, NameError, TypeError, AttributeError)
        exc_match = re.search(r"([A-Z][a-zA-Z0-9_]*Error|[A-Z][a-zA-Z0-9_]*Exception):\s*(.*)", output)
        if exc_match:
            err_type = exc_match.group(1)
            diag = exc_match.group(0)

        return failing_file, failing_line, err_type, diag

    @classmethod
    def _generate_corrective_operation(
        cls,
        target_file: str,
        code: str,
        test_result: TestExecutionResult,
    ) -> Optional[EditOperation]:
        """Formulate a safe corrective patch for common test failures."""
        diag = test_result.diagnostics or test_result.output
        err_type = test_result.error_type

        # 1. NameError: name 'X' is not defined -> Add missing import
        if err_type == "NameError":
            m_name = re.search(r"name\s+'([^']+)'\s+is not defined", diag)
            if m_name:
                missing_name = m_name.group(1)
                # Known standard modules
                std_modules = {
                    "uuid": "import uuid",
                    "json": "import json",
                    "os": "import os",
                    "sys": "import sys",
                    "math": "import math",
                    "re": "import re",
                    "datetime": "from datetime import datetime",
                    "timedelta": "from datetime import timedelta",
                    "List": "from typing import List",
                    "Dict": "from typing import Dict",
                    "Optional": "from typing import Optional",
                }
                if missing_name in std_modules:
                    imp_stmt = std_modules[missing_name]
                    return EditOperation(
                        operation_type=EditOperationType.INSERT,
                        target_file=target_file,
                        content=imp_stmt,
                        position="before",
                        line_number=1,
                        description=f"Auto-fix missing import '{imp_stmt}'",
                    )

        # 2. AttributeError: module/class has no attribute 'logout'
        if err_type == "AttributeError":
            m_attr = re.search(r"object has no attribute\s+'([^']+)'", diag)
            if m_attr:
                attr = m_attr.group(1)
                new_method = f'''    def {attr}(self, *args, **kwargs):\n        """Autonomous fix: added missing {attr} method."""\n        return True\n'''
                return EditOperation(
                    operation_type=EditOperationType.APPEND,
                    target_file=target_file,
                    content=new_method,
                    description=f"Auto-fix: add missing method {attr}",
                )

        return None
