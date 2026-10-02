"""End-to-End Test for Step 3C — Precise Code Editing & Patch Engine.

Verifies:
1. Real temporary project on disk:
   test_edit_project/
   ├── main.py
   ├── auth.py
   ├── database.py
   └── tests/
       └── test_auth.py
2. Query 1: "Add a logout method to AuthService."
   - auth.py modified precisely
   - logout() inserted inside AuthService class
   - existing login() preserved
   - ast.parse validation passes
   - real unified diff generated
   - project tests executed and verified
3. Query 2: "Add a uuid based session ID to login."
   - import uuid added exactly once
   - login() modified precisely with session_id
   - existing code preserved
   - tests executed
4. Failure Scenario:
   - Deliberate syntax/patch failure in multi-file proposal
   - Complete rollback triggered
   - Original files on disk restored without partial modifications
5. API Endpoint:
   - POST /api/projects/{id}/patch succeeds with real diff
"""

import ast
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

agent_root = Path(__file__).resolve().parent.parent
workspace_root = agent_root.parent
if str(agent_root) not in sys.path:
    sys.path.insert(0, str(agent_root))
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from config import config
from my_ai_agent.editing.diff_generator import DiffGenerator
from my_ai_agent.editing.edit_operations import (
    EditOperation,
    EditOperationType,
    EditProposal,
    FileEditProposal,
)
from my_ai_agent.editing.patch_engine import PatchEngine
from my_ai_agent.editing.test_runner import ProjectTestRunner
from web_ide.server import app, handle_agent_turn


class TestE2EPreciseEditing(unittest.TestCase):
    """End-to-End tests for Step 3C Precise Code Editing."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="e2e_edit_proj_"))
        
        # 1. main.py
        (self.test_dir / "main.py").write_text(
            'from auth import AuthService\n\n'
            'def main():\n'
            '    service = AuthService()\n'
            '    print("Logged in:", service.login("admin", "secret"))\n\n'
            'if __name__ == "__main__":\n'
            '    main()\n',
            encoding="utf-8",
        )

        # 2. auth.py
        (self.test_dir / "auth.py").write_text(
            'class AuthService:\n\n'
            '    def login(self, username, password):\n'
            '        return username == "admin"\n',
            encoding="utf-8",
        )

        # 3. database.py
        (self.test_dir / "database.py").write_text(
            'def get_connection():\n'
            '    return {"connected": True}\n',
            encoding="utf-8",
        )

        # 4. tests/test_auth.py
        tests_dir = self.test_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_auth.py").write_text(
            'import unittest\n'
            'import sys\n'
            'from pathlib import Path\n'
            'sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n'
            'from auth import AuthService\n\n'
            'class TestAuth(unittest.TestCase):\n'
            '    def test_login(self):\n'
            '        auth = AuthService()\n'
            '        self.assertTrue(auth.login("admin", "secret"))\n'
            '        self.assertFalse(auth.login("guest", "wrong"))\n\n'
            'if __name__ == "__main__":\n'
            '    unittest.main()\n',
            encoding="utf-8",
        )

        config.set_workspace(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_add_logout_method_to_auth_service(self):
        """Query 1: 'Add a logout method to AuthService.'"""
        query = "Add a logout method to AuthService."
        result = handle_agent_turn(query, project_id="e2e_edit")

        # 1. Verify auth.py content on disk
        auth_code = (self.test_dir / "auth.py").read_text(encoding="utf-8")
        
        # Must be valid Python syntax
        tree = ast.parse(auth_code)
        self.assertIsNotNone(tree)

        # Must have logout method inside AuthService
        class_node = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "AuthService":
                class_node = node
                break
        self.assertIsNotNone(class_node, "AuthService class must exist")
        
        method_names = [n.name for n in class_node.body if isinstance(n, ast.FunctionDef)]
        self.assertIn("logout", method_names, "logout must be a method inside AuthService")
        self.assertIn("login", method_names, "Existing login method must be preserved inside AuthService")

        # 2. Verify diffs were produced and attached
        diffs = result.get("diffs", [])
        self.assertGreater(len(diffs), 0, "Real diff must be attached to result")
        auth_diff = next((d for d in diffs if "auth.py" in d.get("file_path", "")), None)
        self.assertIsNotNone(auth_diff, "Diff for auth.py must be present")
        self.assertGreater(auth_diff["lines_added"], 0)
        self.assertIn("+", auth_diff["unified_diff"])
        self.assertGreater(auth_diff["preservation_percentage"], 50.0)

        # 3. Verify tests were discovered and executed
        test_res = result.get("test_results")
        self.assertIsNotNone(test_res, "Project tests must be run automatically")
        self.assertTrue(test_res["passed"], f"Tests should pass, output: {test_res.get('output')}")
        self.assertIn("unittest", test_res["runner_command"])

    def test_02_add_uuid_session_id_to_login(self):
        """Query 2: 'Add a uuid based session ID to login.'"""
        query = "Add a uuid based session ID to login."
        result = handle_agent_turn(query, project_id="e2e_edit")

        auth_code = (self.test_dir / "auth.py").read_text(encoding="utf-8")

        # 1. Valid Python syntax
        tree = ast.parse(auth_code)
        self.assertIsNotNone(tree)

        # 2. Check import uuid was added exactly once
        self.assertEqual(auth_code.count("import uuid"), 1, "import uuid must be present exactly once")

        # 3. Check login method has session_id
        self.assertIn("session_id", auth_code, "login method must set session_id")
        self.assertIn("uuid.uuid4()", auth_code, "login method must use uuid.uuid4()")
        self.assertIn('return username == "admin"', auth_code, "Original return logic preserved")

        # 4. Check diff
        diffs = result.get("diffs", [])
        self.assertGreater(len(diffs), 0)
        auth_diff = next((d for d in diffs if "auth.py" in d.get("file_path", "")), None)
        self.assertIsNotNone(auth_diff)
        self.assertIn("+import uuid", auth_diff["unified_diff"])

        # 5. Verify tests run and pass
        test_res = result.get("test_results")
        self.assertIsNotNone(test_res)
        self.assertTrue(test_res["passed"])

    def test_03_transactional_rollback_on_failure(self):
        """Test that a failed multi-file operation rolls back all files completely."""
        auth_before = (self.test_dir / "auth.py").read_text(encoding="utf-8")
        db_before = (self.test_dir / "database.py").read_text(encoding="utf-8")

        engine = PatchEngine(workspace_root=self.test_dir)

        # Proposal with valid auth.py edit and syntax-broken database.py edit
        proposal = EditProposal(
            intent="UPDATE_SYSTEM",
            description="Multi-file change with deliberate failure in file 2",
            files=[
                FileEditProposal(
                    path="auth.py",
                    operations=[
                        EditOperation(
                            operation_type=EditOperationType.INSERT,
                            target_file="auth.py",
                            target_symbol="AuthService",
                            content="def temp_method(self):\n    return True\n",
                        )
                    ],
                ),
                FileEditProposal(
                    path="database.py",
                    operations=[
                        EditOperation(
                            operation_type=EditOperationType.APPEND,
                            target_file="database.py",
                            content="def broken_syntax(\n",  # Syntax error!
                        )
                    ],
                ),
            ],
        )

        tx_res = engine.apply_proposal(proposal)

        # 1. Transaction must report failure and rollback
        self.assertFalse(tx_res.success)
        self.assertTrue(tx_res.rolled_back)
        self.assertIn("database.py", "; ".join(tx_res.errors))

        # 2. Disk contents must match the exact pre-transaction state
        auth_after = (self.test_dir / "auth.py").read_text(encoding="utf-8")
        db_after = (self.test_dir / "database.py").read_text(encoding="utf-8")

        self.assertEqual(auth_after, auth_before, "auth.py was NOT rolled back!")
        self.assertEqual(db_after, db_before, "database.py was NOT rolled back!")

    def test_04_patch_api_endpoint(self):
        """Test POST /api/projects/{id}/patch endpoint."""
        from fastapi.testclient import TestClient
        client = TestClient(app)

        payload = {
            "intent": "ADD_HELPER",
            "description": "Add helper to database.py",
            "files": [
                {
                    "path": "database.py",
                    "operations": [
                        {
                            "operation_type": "APPEND",
                            "target_file": "database.py",
                            "content": "\ndef ping():\n    return 'pong'\n",
                        }
                    ],
                }
            ],
        }

        resp = client.post("/api/projects/test_proj/patch", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertFalse(data["rolled_back"])
        self.assertIn("database.py", data["modified_files"])
        self.assertGreater(len(data["diffs"]), 0)

        # Check file on disk
        db_code = (self.test_dir / "database.py").read_text(encoding="utf-8")
        self.assertIn("def ping():", db_code)


if __name__ == "__main__":
    unittest.main()
