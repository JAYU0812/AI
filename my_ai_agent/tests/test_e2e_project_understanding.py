"""End-to-End Test for Step 3B — Real Project Understanding Engine.

Simulates the exact user scenario:
1. Sets up real test project:
   test_project/
   ├── main.py
   ├── auth.py
   ├── database.py
   ├── tests/
   │   └── test_auth.py
   └── requirements.txt

2. Query 1: "Understand this project and tell me how authentication currently works."
   - Must scan project
   - Must detect Python
   - Must build project map
   - Must identify authentication files (auth.py, tests/test_auth.py)
   - Must inspect code and produce factual explanation
   - Must NOT modify any project files!

3. Query 2: "Add a logout function."
   - Must identify relevant file (auth.py)
   - Must inspect code before modifying
   - Must refactor auth.py to add logout
   - Must verify auth.py actually contains logout function
   - Must verify project map refreshes
"""

import os
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
from my_ai_agent.agent import AgentEngine
from my_ai_agent.project.project_understanding import ProjectUnderstandingEngine
from web_ide.server import handle_agent_turn


class TestE2EProjectUnderstanding(unittest.TestCase):
    """Real End-to-End test for Step 3B on disk."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="e2e_project_"))
        
        # 1. Populate test project files
        (self.test_dir / "main.py").write_text(
            'from auth import authenticate_user\n\n'
            'def main():\n'
            '    print("Application starting...")\n'
            '    user = authenticate_user("admin", "secret")\n'
            '    print(f"User: {user}")\n\n'
            'if __name__ == "__main__":\n'
            '    main()\n',
            encoding="utf-8",
        )
        
        (self.test_dir / "auth.py").write_text(
            '"""Authentication module."""\n\n'
            'def authenticate_user(username: str, password_hash: str) -> dict:\n'
            '    """Authenticate a user with username and password."""\n'
            '    if username == "admin" and password_hash == "secret":\n'
            '        return {"username": "admin", "role": "admin", "token": "xyz_tok"}\n'
            '    return {}\n\n'
            'def verify_token(token: str) -> bool:\n'
            '    return token == "xyz_tok"\n',
            encoding="utf-8",
        )

        (self.test_dir / "database.py").write_text(
            '"""Database connection module."""\n\n'
            'def get_db():\n'
            '    return {"connected": True}\n',
            encoding="utf-8",
        )

        tests_dir = self.test_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_auth.py").write_text(
            'from auth import authenticate_user, verify_token\n\n'
            'def test_authenticate():\n'
            '    res = authenticate_user("admin", "secret")\n'
            '    assert res.get("role") == "admin"\n',
            encoding="utf-8",
        )

        (self.test_dir / "requirements.txt").write_text(
            'pytest>=7.0.0\n',
            encoding="utf-8",
        )

        # Set workspace to test directory
        config.set_workspace(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_e2e_understanding_and_read_before_modify(self):
        # -------------------------------------------------------------
        # Part 1: Understand project without modifying it
        # -------------------------------------------------------------
        # Record mtimes before query
        files_before = {
            p.name: p.stat().st_mtime for p in self.test_dir.rglob("*.py")
        }

        query1 = "Understand this project and tell me how authentication currently works."
        result1 = handle_agent_turn(query1, project_id="test_proj_e2e")

        # 1. Verify intent is UNDERSTAND_PROJECT
        self.assertEqual(result1["intent"]["name"], "UNDERSTAND_PROJECT")
        
        # 2. Verify factual response mentions authenticate_user or auth.py
        resp1 = result1["response"]
        self.assertTrue(
            "authenticate_user" in resp1 or "auth.py" in resp1,
            f"Expected factual analysis mentioning auth components, got: {resp1}"
        )

        # 3. Verify project understanding metadata is attached
        pu = result1.get("project_understanding", {})
        self.assertEqual(pu.get("primary_type"), "Python")
        self.assertEqual(pu.get("total_files"), 5)

        # 4. Strict check: NO files modified during inquiry
        for p in self.test_dir.rglob("*.py"):
            self.assertEqual(
                p.stat().st_mtime,
                files_before[p.name],
                f"File {p.name} was modified during an inquiry task!"
            )

        # -------------------------------------------------------------
        # Part 2: "Add a logout function" (Read before modify)
        # -------------------------------------------------------------
        query2 = "Add a logout function."
        result2 = handle_agent_turn(query2, project_id="test_proj_e2e")

        # Verify steps executed
        executed = result2.get("executed_steps", [])
        action_types = [s["action_type"] for s in executed]

        # Must have performed READ_FILE before WRITE_FILE
        self.assertIn("READ_FILE", action_types, "Agent must read existing code before modifying it")
        self.assertIn("WRITE_FILE", action_types, "Agent must write refactored code")

        read_idx = action_types.index("READ_FILE")
        write_idx = action_types.index("WRITE_FILE")
        self.assertLess(read_idx, write_idx, "READ_FILE must occur BEFORE WRITE_FILE")

        # Verify auth.py on disk actually has the new logout function
        auth_content = (self.test_dir / "auth.py").read_text(encoding="utf-8")
        self.assertIn("def logout", auth_content, "auth.py must contain the new logout function")
        # Must still preserve existing functions
        self.assertIn("def authenticate_user", auth_content, "Existing code was clobbered!")
        self.assertIn("def verify_token", auth_content, "Existing code was clobbered!")

        print("\n[E2E Verification PASS] Both Part 1 (Inquiry) and Part 2 (Read-Before-Modify) succeeded flawlessly.")


if __name__ == "__main__":
    unittest.main()
