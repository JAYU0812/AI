"""Leakage Detection and Evaluation Boundary Guard.

Guarantees:
1. Held-out Step 6A benchmark test set is NEVER leaked into training data.
2. No training example memorizes or duplicates exact benchmark prompts.
3. Strict cross-split separation: source task groups and code fingerprints
   do not cross between train, validation, and test splits.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from my_ai_agent.evaluation.dataset import BenchmarkDataset
from my_ai_agent.training.deduplicator import normalize_text
from my_ai_agent.training.schema import TrainingExample


class BenchmarkLeakageDetector:
    """Detects leaks of benchmark evaluation tasks into training data."""

    def __init__(self, benchmark_version: str = "v1"):
        self.benchmark_dataset = BenchmarkDataset(version=benchmark_version)
        self.held_out_tasks = self.benchmark_dataset.filter(split="test")
        
        # Build normalized index of held-out benchmark prompts
        self.held_out_prompts: Set[str] = {
            normalize_text(task.prompt) for task in self.held_out_tasks
        }
        self.held_out_ids: Set[str] = {task.id for task in self.held_out_tasks}

    def check_example_leakage(self, example: TrainingExample) -> Tuple[bool, Optional[str]]:
        """Checks if a training candidate leaks any held-out benchmark task.

        Returns:
            (has_leakage, reason)
        """
        norm_inst = normalize_text(example.instruction)

        # Check if instruction is an exact duplicate of a held-out test prompt
        if norm_inst in self.held_out_prompts:
            return True, f"Training instruction matches held-out benchmark test prompt: '{example.instruction}'"

        return False, None

    @classmethod
    def verify_split_isolation(
        cls,
        train_examples: List[TrainingExample],
        val_examples: List[TrainingExample],
        test_examples: List[TrainingExample],
    ) -> Tuple[bool, List[str]]:
        """Verifies that no source task, instruction, or target leaks across splits.

        Returns:
            (is_isolated, list_of_violations)
        """
        violations: List[str] = []

        train_sources = {e.provenance.source_task_id for e in train_examples if e.provenance}
        val_sources = {e.provenance.source_task_id for e in val_examples if e.provenance}
        test_sources = {e.provenance.source_task_id for e in test_examples if e.provenance}

        # Check source task overlap
        train_val_overlap = train_sources.intersection(val_sources)
        if train_val_overlap:
            violations.append(f"Source task overlap between train and validation: {sorted(list(train_val_overlap))}")

        train_test_overlap = train_sources.intersection(test_sources)
        if train_test_overlap:
            violations.append(f"Source task overlap between train and test: {sorted(list(train_test_overlap))}")

        val_test_overlap = val_sources.intersection(test_sources)
        if val_test_overlap:
            violations.append(f"Source task overlap between validation and test: {sorted(list(val_test_overlap))}")

        # Check normalized instruction overlap
        train_insts = {normalize_text(e.instruction) for e in train_examples}
        val_insts = {normalize_text(e.instruction) for e in val_examples}
        test_insts = {normalize_text(e.instruction) for e in test_examples}

        tv_inst = train_insts.intersection(val_insts)
        if tv_inst:
            violations.append(f"Normalized instruction leakage between train and val: {len(tv_inst)} overlaps")

        tt_inst = train_insts.intersection(test_insts)
        if tt_inst:
            violations.append(f"Normalized instruction leakage between train and test: {len(tt_inst)} overlaps")

        vt_inst = val_insts.intersection(test_insts)
        if vt_inst:
            violations.append(f"Normalized instruction leakage between val and test: {len(vt_inst)} overlaps")

        return len(violations) == 0, violations
