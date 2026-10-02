"""Comprehensive automated tests for Step 3B — Real Project Understanding Engine."""

import shutil
import tempfile
import unittest
from pathlib import Path

from my_ai_agent.project.file_classifier import FileCategory, FileClassifier
from my_ai_agent.project.scanner import ProjectScanner
from my_ai_agent.project.dependency_analyzer import DependencyAnalyzer
from my_ai_agent.project.symbol_analyzer import SymbolAnalyzer
from my_ai_agent.project.analyzer import ProjectAnalyzer
from my_ai_agent.project.project_map import ProjectMapCache
from my_ai_agent.project.relevance import RelevanceEngine
from my_ai_agent.project.context_builder import ContextBuilder
from my_ai_agent.project.project_understanding import ProjectUnderstandingEngine
from my_ai_agent.tools import read_file, run_command
from config import config


class TestFileClassifier(unittest.TestCase):
    """Test deterministic file classification."""

    def test_source_files(self):
        c1 = FileClassifier.classify("app/main.py")
        self.assertEqual(c1.category, FileCategory.SOURCE)
        self.assertTrue(c1.is_code)

        c2 = FileClassifier.classify("src/index.ts")
        self.assertEqual(c2.category, FileCategory.SOURCE)

        c3 = FileClassifier.classify("lib/main.dart")
        self.assertEqual(c3.category, FileCategory.SOURCE)

    def test_test_files(self):
        c1 = FileClassifier.classify("tests/test_auth.py")
        self.assertEqual(c1.category, FileCategory.TEST)

        c2 = FileClassifier.classify("src/components/button.test.tsx")
        self.assertEqual(c2.category, FileCategory.TEST)

    def test_dependency_manifests(self):
        c1 = FileClassifier.classify("requirements.txt")
        self.assertEqual(c1.category, FileCategory.DEPENDENCY)

        c2 = FileClassifier.classify("package.json")
        self.assertEqual(c2.category, FileCategory.DEPENDENCY)

        c3 = FileClassifier.classify("pubspec.yaml")
        self.assertEqual(c3.category, FileCategory.DEPENDENCY)

    def test_config_and_docs(self):
        c1 = FileClassifier.classify("tsconfig.json")
        self.assertEqual(c1.category, FileCategory.CONFIG)

        c2 = FileClassifier.classify("README.md")
        self.assertEqual(c2.category, FileCategory.DOCUMENTATION)

    def test_sensitive_files(self):
        c1 = FileClassifier.classify(".env")
        self.assertEqual(c1.category, FileCategory.SENSITIVE)
        self.assertTrue(c1.is_sensitive)

        c2 = FileClassifier.classify(".env.production")
        self.assertTrue(c2.is_sensitive)

        c3 = FileClassifier.classify("server.key")
        self.assertTrue(c3.is_sensitive)


class TestProjectScanner(unittest.TestCase):
    """Test recursive filesystem scanner and ignore rules."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_scanner_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scanner_with_ignores_and_nesting(self):
        # Create standard structure
        (self.temp_dir / "src").mkdir()
        (self.temp_dir / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
        (self.temp_dir / "src" / "utils.py").write_text("def helper(): pass", encoding="utf-8")
        (self.temp_dir / "tests").mkdir()
        (self.temp_dir / "tests" / "test_main.py").write_text("assert True", encoding="utf-8")
        (self.temp_dir / ".env").write_text("SECRET=123", encoding="utf-8")

        # Create ignored folders
        (self.temp_dir / ".git").mkdir()
        (self.temp_dir / ".git" / "config").write_text("git config", encoding="utf-8")
        (self.temp_dir / "node_modules").mkdir()
        (self.temp_dir / "node_modules" / "pkg.js").write_text("pkg", encoding="utf-8")
        (self.temp_dir / "__pycache__").mkdir()
        (self.temp_dir / "__pycache__" / "c.pyc").write_bytes(b"pyc")

        scanner = ProjectScanner()
        scanned_files = scanner.scan(self.temp_dir)

        discovered_paths = [f.path for f in scanned_files]

        # Verify ignored paths not present
        self.assertFalse(any(p.startswith(".git") for p in discovered_paths))
        self.assertFalse(any(p.startswith("node_modules") for p in discovered_paths))
        self.assertFalse(any(p.startswith("__pycache__") for p in discovered_paths))

        # Verify legitimate files present
        self.assertIn("src/main.py", discovered_paths)
        self.assertIn("src/utils.py", discovered_paths)
        self.assertIn("tests/test_main.py", discovered_paths)
        self.assertIn(".env", discovered_paths)

        # Verify file count
        self.assertEqual(len(scanned_files), 4)


class TestDependencyAnalyzer(unittest.TestCase):
    """Test manifest parsing without running package managers."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_deps_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_python_requirements(self):
        req_file = self.temp_dir / "requirements.txt"
        req_file.write_text(
            "fastapi>=0.100.0\nuvicorn==0.23.2\npydantic\n# comment\n", encoding="utf-8"
        )
        report = DependencyAnalyzer.analyze_file(req_file, "requirements.txt")
        self.assertIsNotNone(report)
        self.assertEqual(report.ecosystem, "python")
        self.assertIn("fastapi", report.dependencies)
        self.assertIn("uvicorn", report.dependencies)
        self.assertIn("pydantic", report.dependencies)

    def test_node_package_json(self):
        pkg_file = self.temp_dir / "package.json"
        pkg_file.write_text(
            '{"dependencies": {"express": "^4.18.2"}, "devDependencies": {"jest": "^29.0.0"}}',
            encoding="utf-8",
        )
        report = DependencyAnalyzer.analyze_file(pkg_file, "package.json")
        self.assertIsNotNone(report)
        self.assertIn("express", report.dependencies)
        self.assertIn("jest", report.dev_dependencies)


class TestSymbolAnalyzer(unittest.TestCase):
    """Test Python AST class, method, function, and import detection."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_symbols_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_python_ast(self):
        code = '''
import os
from datetime import datetime

class UserService:
    def authenticate(self, user: str, pwd: str) -> bool:
        return True

def hash_password(pwd: str) -> str:
    return "hash"

if __name__ == "__main__":
    print("running")
'''
        file_path = self.temp_dir / "auth.py"
        file_path.write_text(code, encoding="utf-8")

        summary = SymbolAnalyzer.analyze_file(file_path, "auth.py")
        self.assertEqual(len(summary.classes), 1)
        self.assertEqual(summary.classes[0].name, "UserService")
        method_names = [m.name for m in summary.classes[0].methods]
        self.assertIn("authenticate", method_names)

        func_names = [f.name for f in summary.functions]
        self.assertIn("hash_password", func_names)

        self.assertIn("os", summary.imports)
        self.assertIn("datetime", summary.imports)
        self.assertTrue(any("main" in sig for sig in summary.entry_point_signals))


class TestProjectAnalyzer(unittest.TestCase):
    """Test project type detection and entry point discovery."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_analyzer_"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_python_fastapi_detection(self):
        (self.temp_dir / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()", encoding="utf-8")
        req_file = self.temp_dir / "requirements.txt"
        req_file.write_text("fastapi\nuvicorn", encoding="utf-8")

        scanner = ProjectScanner()
        scanned_files = scanner.scan(self.temp_dir)
        dep_info = DependencyAnalyzer.analyze_file(req_file, "requirements.txt")
        deps = [dep_info] if dep_info else []
        symbols = {"main.py": SymbolAnalyzer.analyze_file(self.temp_dir / "main.py", "main.py")}

        types = ProjectAnalyzer.detect_project_types(scanned_files, deps, symbols)
        type_names = [t.name for t in types]
        self.assertIn("FastAPI", type_names)
        self.assertIn("Python", type_names)

        entry_points = ProjectAnalyzer.detect_entry_points(scanned_files, symbols)
        self.assertTrue(any(ep.path == "main.py" for ep in entry_points))


class TestRelevanceEngine(unittest.TestCase):
    """Test deterministic relevance ranking against user requests."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_relevance_"))
        (self.temp_dir / "auth.py").write_text("def authenticate(): pass\ndef login(): pass", encoding="utf-8")
        (self.temp_dir / "database.py").write_text("def connect(): pass", encoding="utf-8")
        (self.temp_dir / "styles.css").write_text("body { color: red; }", encoding="utf-8")
        (self.temp_dir / "README.md").write_text("# Project Docs", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_authentication_query(self):
        engine = ProjectUnderstandingEngine()
        pmap = engine.understand_project(self.temp_dir)

        ranked = RelevanceEngine.rank_files(pmap, "Fix the authentication and login bug")
        self.assertTrue(len(ranked) > 0)
        # auth.py must be the top ranked file
        self.assertEqual(ranked[0].path, "auth.py")
        self.assertGreater(ranked[0].score, 0.5)


class TestContextBuilderAndSafety(unittest.TestCase):
    """Test context creation and security boundary enforcement."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_safety_"))
        (self.temp_dir / "main.py").write_text("print('app')", encoding="utf-8")
        (self.temp_dir / ".env").write_text("DB_PASSWORD=supersecret_key_12345", encoding="utf-8")
        config.set_workspace(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sensitive_file_masked_in_context(self):
        engine = ProjectUnderstandingEngine()
        ctx = engine.get_relevant_context(self.temp_dir, "Check database password")

        # Verify .env contents are never in file_contents
        self.assertNotIn(".env", ctx.file_contents)
        self.assertNotIn("supersecret_key_12345", ctx.to_formatted_prompt_block())

    def test_sensitive_file_blocked_in_tools(self):
        res = read_file(".env")
        self.assertFalse(res.success)
        self.assertIn("Safety boundary", res.error)

    def test_dangerous_command_blocked(self):
        res = run_command("rm -rf /")
        self.assertFalse(res.success)
        self.assertIn("Safety boundary", res.error)
        self.assertIn("WAITING_FOR_APPROVAL", res.error)


if __name__ == "__main__":
    unittest.main()
