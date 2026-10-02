"""Step 6C Tests: Benchmark Leakage and Cross-Split Isolation.

Verifies:
1. No Step 6A held-out benchmark test tasks enter training.
2. No cross-split instruction, target, or source-task leakage.
3. The BenchmarkLeakageDetector correctly identifies violations.
4. The verify_split_isolation method accurately flags violations.
"""

import json
import unittest
from typing import List

from my_ai_agent.training.deduplicator import DatasetDeduplicator, compute_code_ast_fingerprint, normalize_text
from my_ai_agent.training.generator import TrainingDatasetGenerator
from my_ai_agent.training.leakage import BenchmarkLeakageDetector
from my_ai_agent.training.provenance import ExampleProvenance
from my_ai_agent.training.schema import OutputType, TaskType, TrainingExample
from my_ai_agent.training.splitter import DatasetSplitter


def _make_example(ex_id: str, instruction: str, src_id: str, task_type: str = "code_generation") -> TrainingExample:
    return TrainingExample(
        id=ex_id,
        instruction=instruction,
        task_type=task_type,
        target_output="def f(): pass\n",
        output_type=OutputType.CODE.value,
        provenance=ExampleProvenance(source_task_id=src_id),
    )


class TestBenchmarkLeakageDetector(unittest.TestCase):
    """Tests that benchmark test set leakage is detected and prevented."""

    @classmethod
    def setUpClass(cls):
        cls.detector = BenchmarkLeakageDetector()

    def test_detector_loads_held_out_tasks(self):
        """Held-out task prompts should be indexed."""
        self.assertGreater(len(self.detector.held_out_prompts), 0)

    def test_benchmark_test_task_ids_loaded(self):
        """Must have held-out test task IDs registered."""
        self.assertGreater(len(self.detector.held_out_ids), 0)

    def test_clean_novel_instruction_passes(self):
        """Novel instructions from the generator should not leak benchmark prompts."""
        ex = _make_example("t001", "Implement a function scale_vector in math_ops.py.", "code_gen_001")
        has_leakage, reason = self.detector.check_example_leakage(ex)
        self.assertFalse(has_leakage, f"Leakage falsely detected: {reason}")

    def test_no_generator_examples_leak_benchmark_prompts(self):
        """Integration: all 42 generated examples must pass the leakage check."""
        gen = TrainingDatasetGenerator()
        examples = gen.generate_all()
        leaky = []
        for ex in examples:
            has_leakage, reason = self.detector.check_example_leakage(ex)
            if has_leakage:
                leaky.append((ex.id, ex.instruction, reason))
        self.assertEqual(leaky, [], f"Leakage detected in generated examples: {leaky}")

    def test_verbatim_benchmark_prompt_triggers_leakage(self):
        """An exact copy of a held-out prompt must be flagged."""
        # Pick a held-out benchmark prompt directly from the loaded set
        if not self.detector.held_out_tasks:
            self.skipTest("No held-out tasks loaded")
        prompt = self.detector.held_out_tasks[0].prompt
        ex = _make_example("leaky_001", prompt, "code_gen_001")
        has_leakage, reason = self.detector.check_example_leakage(ex)
        self.assertTrue(has_leakage, "Verbatim benchmark prompt was NOT flagged as leakage")

    def test_normalized_benchmark_prompt_triggers_leakage(self):
        """A lightly reformatted version of a held-out prompt must also be caught."""
        if not self.detector.held_out_tasks:
            self.skipTest("No held-out tasks loaded")
        original = self.detector.held_out_tasks[0].prompt
        # Add minor whitespace and capitalisation variation
        reformatted = original.upper().strip() + "   "
        ex = _make_example("leaky_002", reformatted, "code_gen_001")
        has_leakage, reason = self.detector.check_example_leakage(ex)
        self.assertTrue(has_leakage, "Reformatted benchmark prompt was NOT flagged as leakage")


class TestSplitIsolation(unittest.TestCase):
    """Tests cross-split isolation guarantees."""

    def setUp(self):
        self.splitter = DatasetSplitter()

    def test_verify_isolation_passes_on_clean_splits(self):
        """Clean splits with no overlap must pass verification."""
        train = [_make_example(f"t{i}", f"Task {i}", f"src_{i}") for i in range(8)]
        val = [_make_example(f"v{i}", f"Val task {i}", f"val_src_{i}") for i in range(2)]
        test = [_make_example(f"te{i}", f"Test task {i}", f"test_src_{i}") for i in range(2)]
        is_isolated, violations = BenchmarkLeakageDetector.verify_split_isolation(train, val, test)
        self.assertTrue(is_isolated)
        self.assertEqual(violations, [])

    def test_verify_isolation_fails_on_source_overlap(self):
        """Split with same source_task_id in both train and val must fail."""
        shared_src = "code_gen_001"
        train = [_make_example("t1", "Train task unique.", shared_src)]
        val = [_make_example("v1", "Val task unique.", shared_src)]
        test = [_make_example("te1", "Test task unique.", "another_src")]
        is_isolated, violations = BenchmarkLeakageDetector.verify_split_isolation(train, val, test)
        self.assertFalse(is_isolated)
        self.assertGreater(len(violations), 0)

    def test_verify_isolation_fails_on_instruction_overlap(self):
        """Same instruction appearing in train and test must fail."""
        shared_instruction = "Implement multiply in utils.py."
        train = [_make_example("t1", shared_instruction, "src_001")]
        val = [_make_example("v1", "Different val instruction.", "src_002")]
        test = [_make_example("te1", shared_instruction, "src_003")]
        is_isolated, violations = BenchmarkLeakageDetector.verify_split_isolation(train, val, test)
        self.assertFalse(is_isolated)

    def test_generated_examples_split_isolation(self):
        """Full pipeline: generated examples split must be isolated."""
        gen = TrainingDatasetGenerator()
        examples = gen.generate_all()
        train, val, test = self.splitter.split(examples)
        is_isolated, violations = BenchmarkLeakageDetector.verify_split_isolation(train, val, test)
        self.assertTrue(is_isolated, f"Isolation failures: {violations}")

    def test_empty_splits_pass_isolation(self):
        is_isolated, violations = BenchmarkLeakageDetector.verify_split_isolation([], [], [])
        self.assertTrue(is_isolated)
        self.assertEqual(violations, [])


class TestDeduplicator(unittest.TestCase):
    """Tests exact and normalized duplicate detection."""

    def setUp(self):
        self.dedup = DatasetDeduplicator()

    def test_fresh_example_not_duplicate(self):
        ex = _make_example("t1", "Unique instruction for scale_vector.", "code_gen_001")
        ex.target_output = "def scale_vector(v, f): return [x*f for x in v]\n"
        is_dup, reason = self.dedup.is_duplicate(ex)
        self.assertFalse(is_dup)

    def test_exact_duplicate_detected(self):
        ex1 = _make_example("t1", "Same instruction.", "code_gen_001")
        ex1.target_output = "def add(a, b): return a + b\n"
        ex2 = _make_example("t2", "Same instruction.", "code_gen_001")
        ex2.target_output = "def add(a, b): return a + b\n"
        self.dedup.register(ex1)
        is_dup, reason = self.dedup.is_duplicate(ex2)
        self.assertTrue(is_dup)
        self.assertIn("duplicate", reason.lower())

    def test_normalized_whitespace_duplicate_detected(self):
        ex1 = _make_example("t1", "Add function multiply to utils.py.", "code_gen_001")
        ex1.target_output = "def multiply(a, b): return a * b\n"
        ex2 = _make_example("t2", "Add  function  multiply  to  utils.py.", "code_gen_001")
        ex2.target_output = "def multiply(a, b): return a * b\n"
        self.dedup.register(ex1)
        is_dup, reason = self.dedup.is_duplicate(ex2)
        self.assertTrue(is_dup)

    def test_different_instructions_not_duplicate(self):
        ex1 = _make_example("t1", "Implement scale_vector.", "code_gen_001")
        ex1.target_output = "def scale_vector(v, f): pass\n"
        ex2 = _make_example("t2", "Implement compute_product.", "code_gen_002")
        ex2.target_output = "def compute_product(nums): pass\n"
        self.dedup.register(ex1)
        is_dup, reason = self.dedup.is_duplicate(ex2)
        self.assertFalse(is_dup)

    def test_all_42_generated_examples_unique(self):
        """Integration: none of the 42 generated examples should be duplicate."""
        gen = TrainingDatasetGenerator()
        examples = gen.generate_all()
        dedup = DatasetDeduplicator()
        duplicates = []
        for ex in examples:
            is_dup, reason = dedup.is_duplicate(ex)
            if is_dup:
                duplicates.append((ex.id, reason))
            else:
                dedup.register(ex)
        self.assertEqual(duplicates, [], f"Unexpected duplicates: {duplicates}")


class TestNormalizationAndFingerprinting(unittest.TestCase):
    """Tests normalization and AST fingerprinting utilities."""

    def test_normalize_text_strips_punctuation(self):
        result = normalize_text("Hello, World! 2026.")
        self.assertNotIn(",", result)
        self.assertNotIn(".", result)

    def test_normalize_text_collapses_spaces(self):
        result = normalize_text("multiple  spaces   here")
        self.assertEqual(result, "multiple spaces here")

    def test_ast_fingerprint_semantic_equivalence(self):
        code1 = "def add(a, b):\n    return a + b\n"
        code2 = "def add(x, y):\n    return x + y\n"  # Same AST structure
        fp1 = compute_code_ast_fingerprint(code1)
        fp2 = compute_code_ast_fingerprint(code2)
        # Both functions have identical AST node type sequences
        self.assertEqual(fp1, fp2)

    def test_ast_fingerprint_different_structures(self):
        code1 = "def add(a, b):\n    return a + b\n"
        code2 = "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
        fp1 = compute_code_ast_fingerprint(code1)
        fp2 = compute_code_ast_fingerprint(code2)
        self.assertNotEqual(fp1, fp2)

    def test_ast_fingerprint_syntax_error_fallback(self):
        # Should not raise, returns fallback hash
        fp = compute_code_ast_fingerprint("def broken(:\n    pass")
        self.assertIsInstance(fp, str)
        self.assertEqual(len(fp), 64)  # SHA-256 hex digest length


if __name__ == "__main__":
    unittest.main()
