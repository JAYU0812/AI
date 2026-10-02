"""Step 6C Tests: Secret Scanning, Code Validation, and Quality Scoring.

Verifies the quality module's security scanning regexes, Python AST validation,
structured JSON schema validation, constraint checking, and quality scoring formula.
"""

import json
import unittest

from my_ai_agent.training.provenance import ExampleProvenance
from my_ai_agent.training.quality import (
    DEFAULT_QUALITY_THRESHOLD,
    scan_for_pii,
    scan_for_secrets,
    score_example_quality,
    validate_constraints,
    validate_python_code,
    validate_structured_json,
)
from my_ai_agent.training.schema import OutputType, QualityMetadata, TaskType, TrainingExample


def _make_code_example(code: str, constraints=None) -> TrainingExample:
    """Helper to construct a code example for testing."""
    return TrainingExample(
        id="test_001",
        instruction="Add function multiply to utils.py.",
        task_type=TaskType.CODE_GENERATION.value,
        target_output=code,
        output_type=OutputType.CODE.value,
        constraints=constraints or [],
        provenance=ExampleProvenance(source_task_id="code_gen_001"),
        verification=[{"call": "multiply(2, 3)", "expected": 6}],
        expected_behavior="Multiplies two integers",
        context={"project_type": "python", "files": []},
    )


class TestSecretScanning(unittest.TestCase):
    """Tests that the secret scanner reliably detects credential patterns."""

    def test_clean_code_no_secrets(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        findings = scan_for_secrets(code)
        self.assertEqual(findings, [])

    def test_detects_private_key_header(self):
        code = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
        findings = scan_for_secrets(code)
        self.assertGreater(len(findings), 0)
        self.assertTrue(any("key" in f.lower() or "Key" in f for f in findings))

    def test_detects_aws_access_key_id(self):
        code = "aws_key = 'AKIAIOSFODNN7EXAMPLE'"
        findings = scan_for_secrets(code)
        self.assertGreater(len(findings), 0)

    def test_detects_hardcoded_password(self):
        code = "password = 'myS3cretP@ssword'"
        findings = scan_for_secrets(code)
        self.assertGreater(len(findings), 0)

    def test_detects_bearer_token(self):
        code = "headers = {'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0'}"
        findings = scan_for_secrets(code)
        self.assertGreater(len(findings), 0)

    def test_ignores_docstrings_and_comments(self):
        code = '"""Helper module for math utilities."""\ndef add(a, b):\n    # No credentials here\n    return a + b\n'
        findings = scan_for_secrets(code)
        self.assertEqual(findings, [])


class TestPIIScanning(unittest.TestCase):
    """Tests PII detection capabilities."""

    def test_clean_code_no_pii(self):
        code = "def factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)\n"
        findings = scan_for_pii(code)
        self.assertEqual(findings, [])

    def test_example_email_ignored(self):
        # Standard dummy emails like user@example.com should be filtered out
        code = "contact = 'user@example.com'"
        findings = scan_for_pii(code)
        self.assertEqual(findings, [])

    def test_detects_ssn(self):
        code = "ssn = '123-45-6789'"
        findings = scan_for_pii(code)
        self.assertGreater(len(findings), 0)


class TestPythonCodeValidation(unittest.TestCase):
    """Tests AST-based Python code correctness validation."""

    def test_valid_function_passes(self):
        code = (
            "def multiply(a: int, b: int) -> int:\n"
            '    """Multiplies a and b."""\n'
            "    return a * b\n"
        )
        valid, err = validate_python_code(code)
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_syntax_error_detected(self):
        code = "def broken(:\n    pass"
        valid, err = validate_python_code(code)
        self.assertFalse(valid)
        self.assertIsNotNone(err)
        self.assertIn("Syntax error", err)

    def test_empty_code_fails(self):
        valid, err = validate_python_code("")
        self.assertFalse(valid)
        self.assertIsNotNone(err)

    def test_whitespace_only_fails(self):
        valid, err = validate_python_code("   \n   ")
        self.assertFalse(valid)

    def test_valid_class_passes(self):
        code = (
            "class Calculator:\n"
            "    def __init__(self, value: float = 0.0):\n"
            "        self.value = value\n\n"
            "    def power(self, exp: float) -> float:\n"
            '        """Raises value to exp."""\n'
            "        self.value = self.value ** exp\n"
            "        return self.value\n"
        )
        valid, err = validate_python_code(code)
        self.assertTrue(valid)

    def test_import_and_function_passes(self):
        code = "import math\n\ndef distance(x: float, y: float) -> float:\n    return math.hypot(x, y)\n"
        valid, err = validate_python_code(code)
        self.assertTrue(valid)


class TestStructuredJSONValidation(unittest.TestCase):
    """Tests JSON schema validation against canonical agent schemas."""

    def test_valid_action_decision_passes(self):
        decision = {
            "action": "INSPECT_PROJECT",
            "target": ".",
            "reason": "Inspect project structure before making edits.",
            "risk": "LOW",
        }
        valid, err = validate_structured_json(json.dumps(decision))
        self.assertTrue(valid, f"Valid action_decision rejected: {err}")

    def test_missing_required_field_fails(self):
        decision = {
            "action": "READ_FILE",
            "target": "main.py",
            # Missing required: "reason" and "risk"
        }
        valid, err = validate_structured_json(json.dumps(decision))
        self.assertFalse(valid)

    def test_invalid_action_enum_fails(self):
        decision = {
            "action": "DO_SOMETHING_WRONG",
            "target": ".",
            "reason": "Test",
            "risk": "LOW",
        }
        valid, err = validate_structured_json(json.dumps(decision))
        self.assertFalse(valid)

    def test_markdown_wrapper_rejected(self):
        text = '```json\n{"action": "INSPECT_PROJECT", "target": ".", "reason": "test", "risk": "LOW"}\n```'
        valid, err = validate_structured_json(text)
        self.assertFalse(valid)
        self.assertIn("markdown", err.lower())

    def test_conversational_preamble_rejected(self):
        text = 'Sure, here is the JSON:\n{"action": "FINISH", "target": "", "reason": "done", "risk": "LOW"}'
        valid, err = validate_structured_json(text)
        self.assertFalse(valid)

    def test_empty_json_fails(self):
        valid, err = validate_structured_json("")
        self.assertFalse(valid)

    def test_invalid_json_syntax_fails(self):
        valid, err = validate_structured_json("{bad: json}")
        self.assertFalse(valid)
        self.assertIn("Invalid JSON", err)

    def test_pure_json_all_valid_actions(self):
        valid_actions = [
            "INSPECT_PROJECT", "READ_FILE", "SEARCH_CODE", "CREATE_FILE",
            "PATCH_FILE", "RUN_TESTS", "VERIFY", "FINISH",
        ]
        for action in valid_actions:
            decision = {"action": action, "target": "x", "reason": "r", "risk": "LOW"}
            valid, err = validate_structured_json(json.dumps(decision))
            self.assertTrue(valid, f"Action '{action}' was incorrectly rejected: {err}")


class TestQualityScoring(unittest.TestCase):
    """Tests the quality scoring formula coverage."""

    def test_perfect_code_example_gets_high_score(self):
        code = (
            'def multiply(a: int, b: int) -> int:\n'
            '    """Multiplies two integers.\n\n    Args:\n        a: First.\n        b: Second.\n\n'
            '    Returns:\n        Product.\n    """\n'
            '    return a * b\n'
        )
        ex = _make_code_example(code)
        ex.context = {"project_type": "python", "files": []}
        ex.expected_behavior = "Returns product of a * b"
        score = score_example_quality(ex)
        self.assertGreaterEqual(score, DEFAULT_QUALITY_THRESHOLD)

    def test_secret_in_code_reduces_score(self):
        code = (
            'import os\n'
            'API_KEY = "sk-AKIA0000000000000000"\n\n'
            'def fetch(url: str) -> str:\n'
            '    """Fetches a URL."""\n'
            '    return url\n'
        )
        ex = _make_code_example(code)
        score = score_example_quality(ex)
        # Secret detection should reduce the security score component
        self.assertLess(score, 1.0)

    def test_syntax_error_code_reduces_score(self):
        code = "def broken(:\n    pass"
        ex = _make_code_example(code)
        score = score_example_quality(ex)
        # Target correctness component should fail
        self.assertLess(score, DEFAULT_QUALITY_THRESHOLD)

    def test_valid_json_example_gets_high_score(self):
        decision = {"action": "INSPECT_PROJECT", "target": ".", "reason": "Inspect project.", "risk": "LOW"}
        ex = TrainingExample(
            id="test_json_001",
            instruction="Generate structured action decision.",
            task_type=TaskType.STRUCTURED_OUTPUT.value,
            target_output=json.dumps(decision, indent=2),
            output_type=OutputType.JSON.value,
            constraints=[],
            provenance=ExampleProvenance(source_task_id="struct_001"),
            verification=[{"schema_validation": "ACTION_DECISION"}],
            expected_behavior="Emits structured JSON decision",
            context={"project_type": "python", "files": []},
        )
        score = score_example_quality(ex)
        self.assertGreaterEqual(score, DEFAULT_QUALITY_THRESHOLD)

    def test_score_between_zero_and_one(self):
        ex = _make_code_example("def add(a, b): return a + b\n")
        score = score_example_quality(ex)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestConstraintValidation(unittest.TestCase):
    """Tests constraint adherence verification."""

    def test_valid_constraints_pass(self):
        ex = _make_code_example(
            'def multiply(a: int, b: int) -> int:\n    """Multiplies."""\n    return a * b\n',
            constraints=["Include docstring", "Return type annotated"],
        )
        ok, violations = validate_constraints(ex)
        self.assertTrue(ok)
        self.assertEqual(violations, [])

    def test_pip_install_violates_no_deps_constraint(self):
        code = (
            "import subprocess\n"
            "subprocess.run(['pip', 'install', 'requests'])\n"
            "def fetch(): pass\n"
        )
        ex = _make_code_example(code, constraints=["Do not install external dependencies"])
        ok, violations = validate_constraints(ex)
        self.assertFalse(ok)
        self.assertGreater(len(violations), 0)


class TestValidatorIntegration(unittest.TestCase):
    """Integration tests for the full TrainingExampleValidator pipeline."""

    @classmethod
    def setUpClass(cls):
        from my_ai_agent.training.validator import TrainingExampleValidator
        cls.validator = TrainingExampleValidator()

    def test_valid_code_example_passes_validator(self):
        code = (
            'def multiply(a: int, b: int) -> int:\n'
            '    """Multiplies two integers."""\n'
            '    return a * b\n'
        )
        ex = TrainingExample(
            id="val_001",
            instruction="Implement multiply function in utils.py.",
            task_type=TaskType.CODE_GENERATION.value,
            target_output=code,
            output_type=OutputType.CODE.value,
            constraints=[],
            provenance=ExampleProvenance(source_task_id="code_gen_001"),
            verification=[{"call": "multiply(2, 3)", "expected": 6}],
            expected_behavior="Returns a * b",
            context={"project_type": "python", "files": []},
        )
        is_valid, errors, score = self.validator.validate(ex)
        self.assertTrue(is_valid, f"Valid example rejected: {errors}")
        self.assertGreaterEqual(score, DEFAULT_QUALITY_THRESHOLD)

    def test_missing_instruction_fails_validator(self):
        ex = TrainingExample(
            id="val_002",
            instruction="",
            task_type=TaskType.CODE_GENERATION.value,
            target_output="def f(): pass\n",
            output_type=OutputType.CODE.value,
            provenance=ExampleProvenance(source_task_id="code_gen_001"),
        )
        is_valid, errors, score = self.validator.validate(ex)
        self.assertFalse(is_valid)
        self.assertTrue(any("Instruction" in e for e in errors))

    def test_missing_provenance_fails_validator(self):
        from my_ai_agent.training.provenance import ExampleProvenance
        ex = TrainingExample(
            id="val_003",
            instruction="Add multiply to utils.py.",
            task_type=TaskType.CODE_GENERATION.value,
            target_output="def multiply(a, b): return a * b\n",
            output_type=OutputType.CODE.value,
            provenance=ExampleProvenance(source_task_id=""),  # Empty source ID
        )
        is_valid, errors, score = self.validator.validate(ex)
        self.assertFalse(is_valid)
        self.assertTrue(any("source_task_id" in e or "Provenance" in e for e in errors))

    def test_secret_in_example_fails_validator(self):
        code = "API_KEY = 'sk-AKIA0000000000000000'\ndef process(): pass\n"
        ex = TrainingExample(
            id="val_004",
            instruction="Create a simple processor.",
            task_type=TaskType.CODE_GENERATION.value,
            target_output=code,
            output_type=OutputType.CODE.value,
            provenance=ExampleProvenance(source_task_id="code_gen_001"),
        )
        is_valid, errors, score = self.validator.validate(ex)
        self.assertFalse(is_valid)
        self.assertTrue(any("Secret" in e for e in errors))

    def test_all_42_generated_examples_pass_validator(self):
        from my_ai_agent.training.generator import TrainingDatasetGenerator
        generator = TrainingDatasetGenerator()
        examples = generator.generate_all()
        failures = []
        for ex in examples:
            is_valid, errors, score = self.validator.validate(ex)
            if not is_valid:
                failures.append((ex.id, errors))
        self.assertEqual(failures, [], f"Generated examples failed validation: {failures}")


if __name__ == "__main__":
    unittest.main()
