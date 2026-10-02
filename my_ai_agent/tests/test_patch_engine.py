"""Comprehensive Unit Tests for Step 3C — Precise Code Editing & Patch Engine."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Setup paths for tests
agent_root = Path(__file__).resolve().parent.parent
workspace_root = agent_root.parent
if str(agent_root) not in sys.path:
    sys.path.insert(0, str(agent_root))
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from config import config
from my_ai_agent.editing.code_locator import CodeLocator
from my_ai_agent.editing.diff_generator import DiffGenerator
from my_ai_agent.editing.edit_operations import (
    EditOperation,
    EditOperationType,
    EditProposal,
    FileEditProposal,
)
from my_ai_agent.editing.patch_engine import PatchEngine
from my_ai_agent.editing.patch_validator import PatchValidator
from my_ai_agent.editing.test_runner import ProjectTestRunner


class TestCodeLocator(unittest.TestCase):
    """Test locating classes, methods, functions, imports, and exact text."""

    def test_locate_python_symbols(self):
        code = '''
class AuthService:
    def __init__(self):
        self.session = None

    def login(self, username, password):
        return username == "admin"

def helper():
    return 42
'''
        # 1. Locate class
        res_class = CodeLocator.locate_target(code, target_symbol="AuthService")
        self.assertTrue(res_class.found)
        self.assertEqual(res_class.symbol_type, "class")
        self.assertEqual(res_class.start_line, 2)

        # 2. Locate method inside class
        res_method = CodeLocator.locate_target(code, target_symbol="login")
        self.assertTrue(res_method.found)
        self.assertEqual(res_method.symbol_type, "function")

        # 3. Locate top-level function
        res_fn = CodeLocator.locate_target(code, target_symbol="helper")
        self.assertTrue(res_fn.found)
        self.assertEqual(res_fn.symbol_type, "function")

    def test_class_insertion_point(self):
        code = '''class Calculator:
    def add(self, a, b):
        return a + b
'''
        loc = CodeLocator.find_class_insertion_point(code, "Calculator")
        self.assertTrue(loc.found)
        self.assertEqual(loc.strategy, "class_body")
        self.assertEqual(loc.indentation, "    ")

    def test_import_insertion_point(self):
        code = '''"""Module docstring."""

import os
import sys

def main():
    pass
'''
        loc = CodeLocator.find_import_insertion_point(code)
        self.assertTrue(loc.found)
        self.assertGreaterEqual(loc.start_line, 4)

    def test_exact_and_normalized_text(self):
        code = "line 1\n  special_marker = 100\nline 3\n"
        # Exact
        res_exact = CodeLocator.locate_target(code, target_text="special_marker = 100")
        self.assertTrue(res_exact.found)
        self.assertEqual(res_exact.start_line, 2)

        # Normalized (different whitespace)
        res_norm = CodeLocator.locate_target(code, target_text="special_marker   =   100")
        self.assertTrue(res_norm.found)
        self.assertEqual(res_norm.start_line, 2)


class TestPatchValidator(unittest.TestCase):
    """Test syntax validation, duplicate prevention, and preservation."""

    def test_valid_and_invalid_syntax(self):
        valid_code = "def valid():\n    return True\n"
        valid, err = PatchValidator.validate_python_syntax(valid_code)
        self.assertTrue(valid)
        self.assertEqual(err, "")

        invalid_code = "def broken(x\n    return x\n"
        valid, err = PatchValidator.validate_python_syntax(invalid_code)
        self.assertFalse(valid)
        self.assertIn("SyntaxError", err)

    def test_duplicate_symbol_detection(self):
        code = '''class User:
    def logout(self):
        pass
'''
        # Duplicate method in class
        is_dup, msg = PatchValidator.check_duplicate_symbol(code, "logout", enclosing_class="User")
        self.assertTrue(is_dup)
        self.assertIn("already exists", msg)

        # Non-duplicate method
        is_dup2, _ = PatchValidator.check_duplicate_symbol(code, "login", enclosing_class="User")
        self.assertFalse(is_dup2)

    def test_duplicate_import_detection(self):
        code = "import os\nfrom datetime import datetime\n"
        self.assertTrue(PatchValidator.check_duplicate_import(code, "import os"))
        self.assertTrue(PatchValidator.check_duplicate_import(code, "from datetime import datetime"))
        self.assertFalse(PatchValidator.check_duplicate_import(code, "import uuid"))


class TestDiffGenerator(unittest.TestCase):
    """Test unified diff calculation and preservation metrics."""

    def test_unified_diff_and_stats(self):
        orig = "line 1\nline 2\nline 3\n"
        mod = "line 1\nline 2 (modified)\nline 3\nline 4 (added)\n"

        diff_res = DiffGenerator.generate_diff(orig, mod, file_path="test.txt")
        self.assertTrue(diff_res.has_changes)
        self.assertGreater(diff_res.lines_added, 0)
        self.assertIn("--- a/test.txt", diff_res.unified_diff)
        self.assertIn("+++ b/test.txt", diff_res.unified_diff)
        self.assertGreater(diff_res.preservation_percentage, 50.0)


class TestPatchEngine(unittest.TestCase):
    """Test atomic patch operations, import management, and multi-file transactions."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_patch_"))
        config.set_workspace(self.temp_dir)
        self.engine = PatchEngine(workspace_root=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_insert_method_into_class(self):
        orig = '''class AuthService:
    def login(self, username, password):
        return username == "admin"
'''
        op = EditOperation(
            operation_type=EditOperationType.INSERT,
            target_file="auth.py",
            target_symbol="AuthService",
            content='def logout(self):\n    self.session = None\n    return True\n',
        )

        res = self.engine.apply_file_operations("auth.py", orig, [op])
        self.assertTrue(res.success, f"Patch failed: {res.error}")
        self.assertIn("def logout(self):", res.patched_code)
        self.assertIn("def login(self, username, password):", res.patched_code)
        # Verify valid Python syntax
        valid, err = PatchValidator.validate_python_syntax(res.patched_code)
        self.assertTrue(valid, err)

    def test_replace_operation(self):
        orig = '''def calculate(x):
    return x * 1
'''
        op = EditOperation(
            operation_type=EditOperationType.REPLACE,
            target_file="calc.py",
            target_symbol="calculate",
            content='def calculate(x):\n    return x * 2\n',
        )

        res = self.engine.apply_file_operations("calc.py", orig, [op])
        self.assertTrue(res.success)
        self.assertIn("return x * 2", res.patched_code)
        self.assertNotIn("return x * 1", res.patched_code)

    def test_automatic_import_management(self):
        orig = '''def generate_id():
    return str(uuid.uuid4())
'''
        op = EditOperation(
            operation_type=EditOperationType.INSERT,
            target_file="ident.py",
            content="",
            required_imports=["import uuid"],
        )

        res = self.engine.apply_file_operations("ident.py", orig, [op])
        self.assertTrue(res.success)
        self.assertTrue(res.patched_code.startswith("import uuid\n"))
        # Second time: verify duplicate import is not added
        res2 = self.engine.apply_file_operations("ident.py", res.patched_code, [op])
        self.assertEqual(res2.patched_code.count("import uuid"), 1)

    def test_multi_file_transaction_rollback(self):
        # Create 2 initial files on disk
        f1 = self.temp_dir / "file1.py"
        f1.write_text("x = 10\n", encoding="utf-8")
        f2 = self.temp_dir / "file2.py"
        f2.write_text("y = 20\n", encoding="utf-8")

        # Proposal where file 1 is valid, but file 2 introduces broken syntax
        op1 = EditOperation(
            operation_type=EditOperationType.APPEND,
            target_file="file1.py",
            content="z = 30\n",
        )
        op2 = EditOperation(
            operation_type=EditOperationType.APPEND,
            target_file="file2.py",
            content="def broken_syntax(\n",  # Syntax error!
        )

        proposal = EditProposal(
            intent="TEST_TRANSACTION",
            description="Multi-file edit with deliberate error in file 2",
            files=[
                FileEditProposal(path="file1.py", operations=[op1]),
                FileEditProposal(path="file2.py", operations=[op2]),
            ],
        )

        tx_res = self.engine.apply_proposal(proposal)

        # Must fail and rollback
        self.assertFalse(tx_res.success)
        self.assertTrue(tx_res.rolled_back)

        # Verify file1.py was ROLLED BACK to original content
        self.assertEqual(f1.read_text(encoding="utf-8"), "x = 10\n")
        # Verify file2.py was unchanged
        self.assertEqual(f2.read_text(encoding="utf-8"), "y = 20\n")

    def test_checkpoint_and_restore(self):
        f = self.temp_dir / "app.py"
        f.write_text("original_state = True\n", encoding="utf-8")

        self.engine.create_checkpoint("v1")

        # Modify file
        f.write_text("corrupted_state = False\n", encoding="utf-8")
        self.assertEqual(f.read_text(encoding="utf-8"), "corrupted_state = False\n")

        # Restore checkpoint
        restored = self.engine.restore_checkpoint("v1")
        self.assertTrue(restored)
        self.assertEqual(f.read_text(encoding="utf-8"), "original_state = True\n")

    def test_safety_boundary_path_traversal(self):
        op = EditOperation(
            operation_type=EditOperationType.APPEND,
            target_file="../../outside.py",
            content="evil = True",
        )
        proposal = EditProposal(
            intent="EVIL",
            description="Path traversal",
            files=[FileEditProposal(path="../../outside.py", operations=[op])],
        )
        tx_res = self.engine.apply_proposal(proposal)
        self.assertFalse(tx_res.success)
        self.assertIn("Security boundary", tx_res.errors[0])

    def test_safety_boundary_sensitive_file(self):
        op = EditOperation(
            operation_type=EditOperationType.APPEND,
            target_file=".env",
            content="SECRET=123",
        )
        proposal = EditProposal(
            intent="SENSITIVE_EDIT",
            description="Modify .env",
            files=[FileEditProposal(path=".env", operations=[op])],
        )
        tx_res = self.engine.apply_proposal(proposal)
        self.assertFalse(tx_res.success)
        self.assertIn("Security boundary", tx_res.errors[0])


if __name__ == "__main__":
    unittest.main()
