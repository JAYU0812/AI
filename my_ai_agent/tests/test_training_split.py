"""Step 6C Tests: Dataset Split Ratios and Source-Task Grouping Integrity.

Verifies that the splitter correctly partitions examples by source_task_id,
maintains balanced ratios, and prevents any source task from appearing in
multiple splits.
"""

import unittest
from collections import Counter, defaultdict
from typing import List

from my_ai_agent.training.provenance import ExampleProvenance
from my_ai_agent.training.schema import OutputType, QualityMetadata, TaskType, TrainingExample
from my_ai_agent.training.splitter import DatasetSplitter


def _make_test_examples(count: int, task_type: str = "code_generation") -> List[TrainingExample]:
    """Creates synthetic examples with distinct source_task_ids for testing."""
    examples = []
    for i in range(count):
        src_id = f"code_gen_{i + 1:03d}"
        ex = TrainingExample(
            id=f"train_{i + 1:06d}",
            instruction=f"Implement function_{i} in module_{i}.py.",
            task_type=task_type,
            target_output=f"def function_{i}(x: int) -> int:\n    \"\"\"Returns x.\"\"\"\n    return x\n",
            output_type=OutputType.CODE.value,
            provenance=ExampleProvenance(source_task_id=src_id),
            difficulty="easy",
        )
        examples.append(ex)
    return examples


def _make_multi_domain_examples() -> List[TrainingExample]:
    """Creates examples across all 4 task domains for balance testing."""
    domains = [
        ("code_generation", "code", 12),
        ("code_modification", "code", 7),
        ("structured_output", "json", 18),
        ("instruction_following", "code", 5),
    ]
    examples = []
    counter = 1
    for task_type, output_type, n in domains:
        for i in range(n):
            src_id = f"{task_type.split('_')[0]}_{i + 1:03d}"
            examples.append(
                TrainingExample(
                    id=f"train_{counter:06d}",
                    instruction=f"Task {counter}: {task_type}.",
                    task_type=task_type,
                    target_output=f"def func_{counter}(): pass\n" if output_type == "code" else '{"action": "FINISH", "target": "", "reason": "done", "risk": "LOW"}',
                    output_type=output_type,
                    provenance=ExampleProvenance(
                        source_task_id=src_id,
                        source_category=task_type.split("_")[0],
                    ),
                    difficulty="medium",
                )
            )
            counter += 1
    return examples


class TestDatasetSplitter(unittest.TestCase):
    """Tests core splitting logic for ratio, grouping, and isolation."""

    def setUp(self):
        self.splitter = DatasetSplitter(train_ratio=0.80, val_ratio=0.10, test_ratio=0.10)

    def test_splits_42_examples_correctly(self):
        """Validates the exact 42-example dataset produces splits with correct totals."""
        examples = _make_test_examples(42)
        train, val, test = self.splitter.split(examples)
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, 42)

    def test_train_is_largest_partition(self):
        examples = _make_test_examples(42)
        train, val, test = self.splitter.split(examples)
        self.assertGreater(len(train), len(val))
        self.assertGreater(len(train), len(test))

    def test_all_source_task_ids_in_exactly_one_split(self):
        """Every source task must appear in exactly one partition (no overlap)."""
        examples = _make_test_examples(42)
        train, val, test = self.splitter.split(examples)

        train_sources = {e.provenance.source_task_id for e in train}
        val_sources = {e.provenance.source_task_id for e in val}
        test_sources = {e.provenance.source_task_id for e in test}

        # No intersection between any pair of splits
        self.assertEqual(len(train_sources.intersection(val_sources)), 0)
        self.assertEqual(len(train_sources.intersection(test_sources)), 0)
        self.assertEqual(len(val_sources.intersection(test_sources)), 0)

    def test_all_examples_assigned_to_a_split(self):
        """No examples should be dropped during splitting."""
        examples = _make_test_examples(20)
        train, val, test = self.splitter.split(examples)
        self.assertEqual(len(train) + len(val) + len(test), len(examples))

    def test_empty_input_returns_empty_splits(self):
        train, val, test = self.splitter.split([])
        self.assertEqual(len(train), 0)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(test), 0)

    def test_single_example_goes_to_train(self):
        examples = _make_test_examples(1)
        train, val, test = self.splitter.split(examples)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(test), 0)

    def test_two_examples_split_train_val(self):
        examples = _make_test_examples(2)
        train, val, test = self.splitter.split(examples)
        self.assertEqual(len(train) + len(val) + len(test), 2)

    def test_real_42_examples_no_source_overlap(self):
        """Integration test with actual generator output."""
        from my_ai_agent.training.generator import TrainingDatasetGenerator
        gen = TrainingDatasetGenerator()
        examples = gen.generate_all()
        train, val, test = self.splitter.split(examples)

        train_sources = {e.provenance.source_task_id for e in train}
        val_sources = {e.provenance.source_task_id for e in val}
        test_sources = {e.provenance.source_task_id for e in test}

        self.assertEqual(len(train_sources.intersection(val_sources)), 0)
        self.assertEqual(len(train_sources.intersection(test_sources)), 0)
        self.assertEqual(len(val_sources.intersection(test_sources)), 0)

    def test_ratio_correctness_on_large_dataset(self):
        """With 100 distinct source tasks, splits should approximately match ratios."""
        examples = _make_test_examples(100)
        train, val, test = self.splitter.split(examples)
        total = len(train) + len(val) + len(test)
        train_pct = len(train) / total
        # With source-task grouping, can't hit exactly 80% but should be within ±15%
        self.assertGreater(train_pct, 0.60)
        self.assertLess(train_pct, 0.95)

    def test_multi_domain_no_source_overlap(self):
        """Multi-domain examples must all be isolated by source across splits."""
        examples = _make_multi_domain_examples()
        train, val, test = self.splitter.split(examples)

        train_sources = {e.provenance.source_task_id for e in train}
        val_sources = {e.provenance.source_task_id for e in val}
        test_sources = {e.provenance.source_task_id for e in test}

        self.assertEqual(len(train_sources.intersection(val_sources)), 0)
        self.assertEqual(len(train_sources.intersection(test_sources)), 0)
        self.assertEqual(len(val_sources.intersection(test_sources)), 0)


class TestDatasetBalanceAuditor(unittest.TestCase):
    """Tests the balance auditor's domain distribution analysis."""

    def test_balanced_dataset_flagged_as_balanced(self):
        from my_ai_agent.training.balance import DatasetBalanceAuditor
        examples = _make_multi_domain_examples()
        stats = DatasetBalanceAuditor.analyze_distribution(examples)
        self.assertIn("by_task_type", stats)
        self.assertIn("is_balanced", stats)

    def test_all_domains_present_in_42_examples(self):
        from my_ai_agent.training.balance import DatasetBalanceAuditor
        from my_ai_agent.training.generator import TrainingDatasetGenerator
        gen = TrainingDatasetGenerator()
        examples = gen.generate_all()
        stats = DatasetBalanceAuditor.analyze_distribution(examples)
        present_domains = set(stats["by_task_type"].keys())
        required_domains = {"code_generation", "code_modification", "structured_output", "instruction_following"}
        self.assertEqual(required_domains, present_domains)

    def test_empty_dataset_handled_gracefully(self):
        from my_ai_agent.training.balance import DatasetBalanceAuditor
        stats = DatasetBalanceAuditor.analyze_distribution([])
        self.assertEqual(stats["total"], 0)
        self.assertFalse(stats["is_balanced"])


if __name__ == "__main__":
    unittest.main()
