"""Step 6D Tests: Core infrastructure — config, dataset, tokenizer, preflight.

Tests:
    - ExperimentConfig defaults and serialization
    - Dataset loading and manifest verification
    - ChatML formatting
    - Preflight validation
    - Token length statistics
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestExperimentConfig(unittest.TestCase):
    """Tests ExperimentConfig defaults, serialization, and pilot config."""

    def test_default_config_has_correct_model(self):
        from my_ai_agent.fine_tuning.config import get_default_config
        cfg = get_default_config()
        self.assertEqual(cfg.base_model_name, "Qwen/Qwen2.5-Coder-0.5B-Instruct")

    def test_default_config_cpu_appropriate(self):
        from my_ai_agent.fine_tuning.config import get_default_config
        cfg = get_default_config()
        self.assertEqual(cfg.torch_dtype, "float32")
        self.assertLessEqual(cfg.max_seq_length, 512)
        self.assertEqual(cfg.per_device_train_batch_size, 1)

    def test_lora_config_defaults(self):
        from my_ai_agent.fine_tuning.config import get_default_config
        cfg = get_default_config()
        self.assertEqual(cfg.lora_r, 8)
        self.assertEqual(cfg.lora_alpha, 16)
        self.assertIn("q_proj", cfg.lora_target_modules)
        self.assertIn("v_proj", cfg.lora_target_modules)

    def test_seed_is_42(self):
        from my_ai_agent.fine_tuning.config import get_default_config
        cfg = get_default_config()
        self.assertEqual(cfg.seed, 42)

    def test_config_to_dict_contains_all_fields(self):
        from my_ai_agent.fine_tuning.config import get_default_config
        cfg = get_default_config()
        d = cfg.to_dict()
        self.assertIn("base_model_name", d)
        self.assertIn("lora_r", d)
        self.assertIn("num_train_epochs", d)
        self.assertIn("seed", d)
        self.assertIn("max_seq_length", d)

    def test_config_save_and_load_roundtrip(self):
        from my_ai_agent.fine_tuning.config import ExperimentConfig, get_default_config
        cfg = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = cfg.save(str(Path(tmpdir) / "config.json"))
            loaded = ExperimentConfig.load(saved_path)
            self.assertEqual(cfg.base_model_name, loaded.base_model_name)
            self.assertEqual(cfg.lora_r, loaded.lora_r)
            self.assertEqual(cfg.seed, loaded.seed)

    def test_pilot_config_has_minimal_epochs(self):
        from my_ai_agent.fine_tuning.config import get_pilot_config
        cfg = get_pilot_config()
        self.assertEqual(cfg.num_train_epochs, 1)

    def test_create_dirs_works(self):
        from my_ai_agent.fine_tuning.config import get_default_config
        cfg = get_default_config()
        cfg.create_dirs()
        self.assertTrue(Path(cfg.experiment_dir).exists())
        self.assertTrue(Path(cfg.checkpoint_dir).exists())


class TestDatasetLoading(unittest.TestCase):
    """Tests JSONL loading, ChatML formatting, and manifest verification."""

    def _make_temp_jsonl(self, records, tmpdir) -> str:
        p = Path(tmpdir) / "data.jsonl"
        with open(p, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return str(p)

    def test_load_jsonl_basic(self):
        from my_ai_agent.fine_tuning.dataset import load_jsonl
        with tempfile.TemporaryDirectory() as tmpdir:
            p = self._make_temp_jsonl([{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}], tmpdir)
            records = load_jsonl(p)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["id"], "1")

    def test_load_jsonl_skips_blank_lines(self):
        from my_ai_agent.fine_tuning.dataset import load_jsonl
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "data.jsonl"
            with open(p, "w") as f:
                f.write('{"id": "1"}\n\n{"id": "2"}\n')
            records = load_jsonl(str(p))
            self.assertEqual(len(records), 2)

    def test_load_jsonl_missing_file_raises(self):
        from my_ai_agent.fine_tuning.dataset import load_jsonl
        with self.assertRaises(FileNotFoundError):
            load_jsonl("/nonexistent/path.jsonl")

    def test_format_as_chatml_has_correct_structure(self):
        from my_ai_agent.fine_tuning.dataset import format_as_chatml
        rec = {
            "instruction": "Write a function add(a, b).",
            "target_output": "def add(a, b):\n    return a + b\n",
            "constraints": ["Include return type annotation"],
            "expected_behavior": "Returns sum of a and b",
        }
        text = format_as_chatml(rec)
        self.assertIn("<|im_start|>system", text)
        self.assertIn("<|im_start|>user", text)
        self.assertIn("<|im_start|>assistant", text)
        self.assertIn("Write a function add(a, b).", text)
        self.assertIn("def add(a, b):", text)
        self.assertIn("<|im_end|>", text)

    def test_format_as_inference_prompt_no_target(self):
        from my_ai_agent.fine_tuning.dataset import format_as_inference_prompt
        prompt = format_as_inference_prompt("Write factorial function.")
        self.assertIn("<|im_start|>assistant", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("def factorial", prompt)

    def test_chatml_includes_constraints(self):
        from my_ai_agent.fine_tuning.dataset import format_as_chatml
        rec = {
            "instruction": "Add multiply function.",
            "target_output": "def multiply(a, b): return a * b",
            "constraints": ["No side effects", "Include docstring"],
        }
        text = format_as_chatml(rec)
        self.assertIn("No side effects", text)
        self.assertIn("Include docstring", text)

    def test_preflight_record_catches_missing_instruction(self):
        from my_ai_agent.fine_tuning.dataset import preflight_record
        rec = {
            "id": "test_001",
            "instruction": "",
            "target_output": "def f(): pass",
            "task_type": "code_generation",
            "output_type": "code",
        }
        errors = preflight_record(rec, 0)
        self.assertTrue(any("empty instruction" in e for e in errors))

    def test_preflight_record_catches_missing_fields(self):
        from my_ai_agent.fine_tuning.dataset import preflight_record
        rec = {"id": "test_001", "instruction": "Do something."}  # Missing many fields
        errors = preflight_record(rec, 0)
        self.assertTrue(len(errors) > 0)

    def test_preflight_record_passes_valid(self):
        from my_ai_agent.fine_tuning.dataset import preflight_record
        rec = {
            "id": "test_001",
            "instruction": "Write a function.",
            "target_output": "def f(): pass",
            "task_type": "code_generation",
            "output_type": "code",
            "provenance": {"source_task_id": "code_gen_001"},
        }
        errors = preflight_record(rec, 0)
        self.assertEqual(errors, [])

    def test_real_training_file_loads(self):
        """Integration: actual Step 6C training file loads without error."""
        from my_ai_agent.fine_tuning.config import TRAINING_JSONL
        if not TRAINING_JSONL.exists():
            self.skipTest("training_dataset_v1.jsonl not found")
        from my_ai_agent.fine_tuning.dataset import load_jsonl
        records = load_jsonl(str(TRAINING_JSONL))
        self.assertGreaterEqual(len(records), 30)

    def test_real_training_records_have_required_fields(self):
        """Integration: all training records have required fields."""
        from my_ai_agent.fine_tuning.config import TRAINING_JSONL
        if not TRAINING_JSONL.exists():
            self.skipTest("training_dataset_v1.jsonl not found")
        from my_ai_agent.fine_tuning.dataset import load_jsonl, preflight_record
        records = load_jsonl(str(TRAINING_JSONL))
        all_errors = []
        for i, rec in enumerate(records):
            errors = preflight_record(rec, i)
            all_errors.extend(errors)
        self.assertEqual(all_errors, [], f"Preflight errors: {all_errors[:3]}")


class TestTokenizerUtilities(unittest.TestCase):
    """Tests tokenizer utilities without loading the actual model."""

    def test_collate_fn_pads_correctly(self):
        import torch
        from my_ai_agent.fine_tuning.tokenizer import collate_fn
        # Two items of different lengths
        item1 = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
        }
        item2 = {
            "input_ids": torch.tensor([4, 5]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([4, 5]),
        }
        batch = collate_fn([item1, item2], pad_token_id=0, max_seq_length=10)
        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertEqual(batch["attention_mask"][1, 2].item(), 0)  # Padded position
        self.assertEqual(batch["labels"][1, 2].item(), -100)  # Padded label = -100

    def test_compute_token_stats_correct_values(self):
        """Use a mock tokenizer that returns known token lengths."""
        from my_ai_agent.fine_tuning.tokenizer import compute_token_stats
        mock_tok = MagicMock()
        # 3 texts of lengths 10, 20, 30
        mock_tok.encode.side_effect = [list(range(10)), list(range(20)), list(range(30))]
        stats = compute_token_stats(["a", "b", "c"], mock_tok)
        self.assertEqual(stats["min"], 10)
        self.assertEqual(stats["max"], 30)
        self.assertAlmostEqual(stats["mean"], 20.0)
        self.assertEqual(stats["median"], 20)

    def test_compute_token_stats_empty(self):
        from my_ai_agent.fine_tuning.tokenizer import compute_token_stats
        mock_tok = MagicMock()
        stats = compute_token_stats([], mock_tok)
        self.assertEqual(stats["count"], 0)

    def test_tokenize_with_labels_masks_prompt(self):
        """Test that labels for the prompt portion are -100."""
        import torch
        from my_ai_agent.fine_tuning.tokenizer import tokenize_with_labels

        # Build a mock tokenizer
        mock_tok = MagicMock()
        # Full sequence: 5 prompt tokens + 3 response tokens = 8 total
        full_text = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\nR<|im_end|>"
        prompt_text = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"

        full_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        prompt_ids = [1, 2, 3, 4, 5]

        def encode_side_effect(text, truncation=False, max_length=None):
            if "<|im_start|>assistant\n" in text and text.endswith("<|im_end|>"):
                return full_ids[:max_length] if max_length else full_ids
            else:
                return prompt_ids[:max_length] if max_length else prompt_ids

        mock_tok.encode.side_effect = encode_side_effect
        result = tokenize_with_labels(full_text, mock_tok, max_seq_length=512)

        # Labels for positions 0-4 (prompt) should be -100
        labels = result["labels"]
        for i in range(5):
            self.assertEqual(labels[i].item(), -100, f"Position {i} should be masked")
        # Labels for positions 5-7 (assistant) should be actual token ids
        for i in range(5, 8):
            self.assertEqual(labels[i].item(), full_ids[i], f"Position {i} should have label")


class TestResourceMonitor(unittest.TestCase):
    """Tests ResourceMonitor and environment info collection."""

    def test_monitor_starts_and_stops(self):
        from my_ai_agent.fine_tuning.resource_monitor import ResourceMonitor
        mon = ResourceMonitor(interval_sec=1.0)
        mon.start()
        import time
        time.sleep(0.1)
        mon.stop()
        summary = mon.get_summary()
        self.assertIn("total_duration_sec", summary)
        self.assertGreaterEqual(summary["total_duration_sec"], 0)

    def test_monitor_records_event(self):
        from my_ai_agent.fine_tuning.resource_monitor import ResourceMonitor
        mon = ResourceMonitor(interval_sec=60.0)
        mon._start_time = __import__("time").time()
        mon.record_event("test_event")
        self.assertTrue(any("event" in s for s in mon._samples))

    def test_environment_info_has_required_keys(self):
        from my_ai_agent.fine_tuning.resource_monitor import get_environment_info
        env = get_environment_info()
        for key in ["os", "python", "torch", "transformers", "peft", "cuda_available"]:
            self.assertIn(key, env)

    def test_environment_info_cuda_is_false(self):
        from my_ai_agent.fine_tuning.resource_monitor import get_environment_info
        env = get_environment_info()
        # We know from hardware inspection that CUDA is not available
        self.assertFalse(env["cuda_available"])


if __name__ == "__main__":
    unittest.main()
