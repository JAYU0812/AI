"""Deterministic Source-Task Grouped Dataset Splitter.

Enforces zero-leakage splits across Train (80%), Validation (10%), and Test (10%).
Guarantees that all variants and augmentations originating from the same source
candidate remain strictly co-located within the same split.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from my_ai_agent.training.schema import TrainingExample


class DatasetSplitter:
    """Partitions training examples into leakage-free train/val/test splits."""

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
    ):
        total = train_ratio + val_ratio + test_ratio
        self.train_ratio = train_ratio / total
        self.val_ratio = val_ratio / total
        self.test_ratio = test_ratio / total

    def split(
        self, examples: List[TrainingExample]
    ) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
        """Splits examples by grouping by source_task_id to prevent data leakage.

        Returns:
            (train_examples, val_examples, test_examples)
        """
        if not examples:
            return [], [], []

        # 1. Group by source_task_id
        groups: Dict[str, List[TrainingExample]] = defaultdict(list)
        for ex in examples:
            src_id = ex.provenance.source_task_id if ex.provenance else "unknown"
            groups[src_id].append(ex)

        total_examples = len(examples)
        target_val_count = max(1, round(total_examples * self.val_ratio))
        target_test_count = max(1, round(total_examples * self.test_ratio))

        # 2. Sort group keys deterministically
        sorted_keys = sorted(groups.keys())

        # Group by category to ensure balanced representation across splits
        category_to_keys: Dict[str, List[str]] = defaultdict(list)
        for k in sorted_keys:
            cat = groups[k][0].task_type
            category_to_keys[cat].append(k)

        train_keys: List[str] = []
        val_keys: List[str] = []
        test_keys: List[str] = []

        # Allocate deterministically per category
        for cat, keys in sorted(category_to_keys.items()):
            n = len(keys)
            if n == 1:
                train_keys.append(keys[0])
            elif n == 2:
                train_keys.append(keys[0])
                val_keys.append(keys[1])
            else:
                # E.g. for n >= 3, assign ~80% train, ~10% val, ~10% test
                n_val = max(1, round(n * 0.10)) if n >= 5 else 1
                n_test = max(1, round(n * 0.10)) if n >= 5 else 1
                n_train = n - n_val - n_test
                if n_train < 1:
                    n_train = 1
                    if n_val > 1:
                        n_val -= 1
                    elif n_test > 1:
                        n_test -= 1

                train_keys.extend(keys[:n_train])
                val_keys.extend(keys[n_train : n_train + n_val])
                test_keys.extend(keys[n_train + n_val :])

        train_examples = [ex for k in train_keys for ex in groups[k]]
        val_examples = [ex for k in val_keys for ex in groups[k]]
        test_examples = [ex for k in test_keys for ex in groups[k]]

        return train_examples, val_examples, test_examples
