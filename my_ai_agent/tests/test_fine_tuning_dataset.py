"""Step 6D Tests: Fine-Tuning Dataset — formatting, preflight, token analysis.

Integration tests against the real Step 6C dataset files.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class TestFineTuningDatasetIntegration(unittest.TestCase):
    """Integration tests against actual Step 6C dataset files."""

    @classmethod
    def setUpClass(cls):
        from my_ai_agent.fine_tuning.config import TRAINING_JSONL, VALIDATION_JSONL, TEST_JSONL
        cls.has_data = (
            TRAINING_JSONL.exists() and
            VALIDATION_JSONL.exists() and
            TEST_JSONL.exists()
        )

    def _skip_if_no_data(self):
        if not self.has_data:
            self.skipTest("Step 6C dataset files not found")

    def test_training_file_has_32_examples(self):
        self._skip_if_no_data()
        from my_ai_agent.fine_tuning.config import TRAINING_JSONL
        from my_ai_agent.fine_tuning.dataset import load_jsonl
        records = load_jsonl(str(TRAINING_JSONL))
        self.assertEqual(len(records), 32)

    def test_validation_file_has_5_examples(self):
        self._skip_if_no_data()
        from my_ai_agent.fine_tuning.config import VALIDATION_JSONL
        from my_ai_agent.fine_tuning.dataset import load_jsonl
        records = load_jsonl(str(VALIDATION_JSONL))
        self.assertEqual(len(records), 5)

    def test_test_file_has_5_examples(self):
        self._skip_if_no_data()
        from my_ai_agent.fine_tuning.config import TEST_JSONL
        from my_ai_agent.fine_tuning.dataset import load_jsonl
        records = load_jsonl(str(TEST_JSONL))
        self.assertEqual(len(records), 5)

    def test_all_training_records_have_provenance(self):
        self._skip_if_no_data()
        from my_ai_agent.fine_tuning.config import TRAINING_JSONL
        from my_ai_agent.fine_tuning.dataset import load_jsonl
        records = load_jsonl(str(TRAINING_JSONL))
        for rec in records:
            prov = rec.get("provenance", {})
            self.assertTrue(prov.get("source_task_id"), f"Record {rec.get('id')} missing source_task_id")

    def test_all_training_records_have_target_output(self):
        self._skip_if_no_data()
        from my_ai_agent.fine_tuning.config import TRAINING_JSONL
        from my_ai_agent.fine_tuning.dataset import load_jsonl
        records = load_jsonl(str(TRAINING_JSONL))
        for rec in records:
            self.assertTrue(rec.get("target_output", "").strip(),
                            f"Record {rec.get('id')} has empty target_output")

    def test_chatml_format_for_all_training_records(self):
        self._skip_if_no_data()
        from my_ai_agent.fine_tuning.config import TRAINING_JSONL
        from my_ai_agent.fine_tuning.dataset import load_jsonl, format_as_chatml
        records = load_jsonl(str(TRAINING_JSONL))
        for rec in records:
            text = format_as_chatml(rec)
            self.assertIn("<|im_start|>assistant", text)
            self.assertIn(rec["target_output"].strip()[:20], text)

    def test_no_step_6a_benchmark_tasks_in_training(self):
        """Training data must not contain Step 6A held-out benchmark prompts."""
        self._skip_if_no_data()
        from my_ai_agent.fine_tuning.config import TRAINING_JSONL
        from my_ai_agent.fine_tuning.dataset import load_jsonl
        from my_ai_agent.training.leakage import BenchmarkLeakageDetector

        detector = BenchmarkLeakageDetector()
        records = load_jsonl(str(TRAINING_JSONL))

        leaky = []
        for rec in records:
            # Simulate a minimal TrainingExample-like object for the detector
            class FakeExample:
                pass
            ex = FakeExample()
            ex.instruction = rec.get("instruction", "")
            ex.provenance = MagicMock()
            ex.provenance.source_task_id = rec.get("provenance", {}).get("source_task_id", "")
            has_leakage, reason = detector.check_example_leakage(ex)
            if has_leakage:
                leaky.append((rec.get("id"), reason))

        self.assertEqual(leaky, [], f"Step 6A leakage found: {leaky}")

    def test_no_secrets_in_training_data(self):
        """Training targets must not contain secrets."""
        self._skip_if_no_data()
        import re
        from my_ai_agent.fine_tuning.config import TRAINING_JSONL
        from my_ai_agent.fine_tuning.dataset import load_jsonl

        SECRET_PATTERN = re.compile(
            r"(-----BEGIN [A-Z ]+KEY-----|AKIA[0-9A-Z]{16}|"
            r"(?:password|passwd|pwd)\s*=\s*['\"][^'\"]{6,}['\"])",
            re.IGNORECASE
        )
        records = load_jsonl(str(TRAINING_JSONL))
        violations = []
        for rec in records:
            target = rec.get("target_output", "")
            if SECRET_PATTERN.search(target):
                violations.append(rec.get("id"))
        self.assertEqual(violations, [])

    def test_source_task_coverage_is_40(self):
        """All 40 Step 6B candidates must be covered in training data."""
        self._skip_if_no_data()
        from my_ai_agent.fine_tuning.config import TRAINING_JSONL, VALIDATION_JSONL, TEST_JSONL
        from my_ai_agent.fine_tuning.dataset import load_jsonl

        all_records = (
            load_jsonl(str(TRAINING_JSONL)) +
            load_jsonl(str(VALIDATION_JSONL)) +
            load_jsonl(str(TEST_JSONL))
        )
        sources = {rec.get("provenance", {}).get("source_task_id") for rec in all_records}
        sources.discard(None)
        sources.discard("")
        self.assertEqual(len(sources), 40, f"Expected 40 unique source tasks, got {len(sources)}")


class TestChatMLFormatting(unittest.TestCase):
    """Unit tests for ChatML prompt formatting."""

    def test_instruction_text_includes_context(self):
        from my_ai_agent.fine_tuning.dataset import _build_instruction_text
        rec = {
            "instruction": "Add a multiply function.",
            "context": {"project_type": "python", "files": [{"path": "utils.py"}]},
            "constraints": ["No side effects"],
            "expected_behavior": "Returns product",
        }
        text = _build_instruction_text(rec)
        self.assertIn("Add a multiply function.", text)
        self.assertIn("python", text)
        self.assertIn("No side effects", text)
        self.assertIn("Returns product", text)

    def test_chatml_system_prompt_present(self):
        from my_ai_agent.fine_tuning.dataset import format_as_chatml, _SYSTEM_PROMPT
        rec = {"instruction": "Write a function.", "target_output": "def f(): pass"}
        text = format_as_chatml(rec)
        self.assertIn(_SYSTEM_PROMPT[:30], text)

    def test_inference_prompt_ends_with_assistant_start(self):
        from my_ai_agent.fine_tuning.dataset import format_as_inference_prompt
        prompt = format_as_inference_prompt("Explain recursion.")
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_chatml_target_is_in_assistant_turn(self):
        from my_ai_agent.fine_tuning.dataset import format_as_chatml
        target = "def add(a, b):\n    return a + b\n"
        rec = {"instruction": "Write add function.", "target_output": target}
        text = format_as_chatml(rec)
        # Find assistant section
        assistant_start = text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
        assistant_section = text[assistant_start:]
        self.assertIn("def add", assistant_section)

    def test_chatml_no_double_escaped_characters(self):
        from my_ai_agent.fine_tuning.dataset import format_as_chatml
        rec = {
            "instruction": 'Generate {"action": "FINISH", "target": ""}.',
            "target_output": '{"action": "FINISH", "target": "", "reason": "done", "risk": "LOW"}',
        }
        text = format_as_chatml(rec)
        # Should not have double-escaped quotes
        self.assertNotIn('\\"', text.replace("\\\\", ""))


class TestDatasetPreflight(unittest.TestCase):
    """Tests the preflight validation with mocked tokenizer."""

    def _make_mock_tokenizer(self, token_len=100):
        mock = MagicMock()
        mock.encode.return_value = list(range(token_len))
        return mock

    def _make_record(self, idx=1, empty_instruction=False):
        return {
            "id": f"train_{idx:06d}",
            "instruction": "" if empty_instruction else f"Implement function_{idx}.",
            "target_output": f"def f_{idx}(): pass\n",
            "task_type": "code_generation",
            "output_type": "code",
            "provenance": {"source_task_id": f"code_gen_{idx:03d}"},
            "constraints": [],
        }

    def test_preflight_passes_valid_records(self):
        from my_ai_agent.fine_tuning.dataset import run_preflight
        train = [self._make_record(i) for i in range(1, 5)]
        val = [self._make_record(i) for i in range(5, 7)]
        mock_tok = self._make_mock_tokenizer(token_len=100)
        result = run_preflight(train, val, mock_tok, max_seq_length=512)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["critical_errors"]), 0)

    def test_preflight_fails_on_empty_instruction(self):
        from my_ai_agent.fine_tuning.dataset import run_preflight
        train = [self._make_record(1, empty_instruction=True)]
        val = [self._make_record(2)]
        mock_tok = self._make_mock_tokenizer(50)
        result = run_preflight(train, val, mock_tok, max_seq_length=512)
        self.assertEqual(result["status"], "FAIL")

    def test_preflight_warns_on_length_overflow(self):
        from my_ai_agent.fine_tuning.dataset import run_preflight
        train = [self._make_record(1)]
        val = [self._make_record(2)]
        mock_tok = self._make_mock_tokenizer(token_len=1000)  # Exceeds max_seq_length=512
        result = run_preflight(train, val, mock_tok, max_seq_length=512)
        self.assertGreater(len(result["warnings"]), 0)
        self.assertGreater(result["truncation_count"], 0)

    def test_preflight_detects_secret_in_target(self):
        from my_ai_agent.fine_tuning.dataset import run_preflight
        rec = self._make_record(1)
        rec["target_output"] = "-----BEGIN RSA PRIVATE KEY-----\nABC123\n-----END RSA PRIVATE KEY-----"
        mock_tok = self._make_mock_tokenizer(100)
        result = run_preflight([rec], [], mock_tok, max_seq_length=512)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("secret" in e.lower() for e in result["critical_errors"]))


if __name__ == "__main__":
    unittest.main()
