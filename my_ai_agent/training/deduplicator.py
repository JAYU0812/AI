"""Deduplication and Collision Detection Engine.

Detects exact duplicates, normalized text duplicates, and code/AST fingerprint
collisions to prevent redundant samples from entering the dataset.
"""

import ast
import hashlib
import json
import re
from typing import Dict, List, Set, Tuple

from my_ai_agent.training.schema import TrainingExample


def normalize_text(text: str) -> str:
    """Normalizes whitespace, casing, and punctuation for similarity comparison."""
    if not text:
        return ""
    # Lowercase, strip non-alphanumeric except spaces
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    # Collapse multiple spaces
    return " ".join(cleaned.split())


def compute_code_ast_fingerprint(code: str) -> str:
    """Computes an AST node-type fingerprint ignoring variable names and whitespace."""
    try:
        tree = ast.parse(code)
        node_types = [type(node).__name__ for node in ast.walk(tree)]
        return hashlib.sha256("".join(node_types).encode("utf-8")).hexdigest()
    except Exception:
        # Fallback to normalized text hash
        return hashlib.sha256(normalize_text(code).encode("utf-8")).hexdigest()


def compute_json_fingerprint(json_str: str) -> str:
    """Computes a canonical fingerprint for structured JSON."""
    try:
        data = json.loads(json_str)
        canonical_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(normalize_text(json_str).encode("utf-8")).hexdigest()


class DatasetDeduplicator:
    """Detects and isolates duplicate training examples."""

    def __init__(self):
        self.exact_hashes: Set[str] = set()
        self.instruction_target_hashes: Set[str] = set()
        self.ast_fingerprints: Dict[str, str] = {}  # fingerprint -> first_seen_id

    def is_duplicate(self, example: TrainingExample) -> Tuple[bool, str]:
        """Checks if example is an exact or normalized duplicate of an existing example.

        Returns:
            (is_duplicate, reason)
        """
        # 1. Exact instruction + target hash
        combined = f"{example.instruction.strip()}###{example.target_output.strip()}"
        exact_h = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        if exact_h in self.exact_hashes:
            return True, "Exact duplicate instruction and target"

        # 2. Normalized instruction + target hash
        norm_combined = f"{normalize_text(example.instruction)}###{normalize_text(example.target_output)}"
        norm_h = hashlib.sha256(norm_combined.encode("utf-8")).hexdigest()
        if norm_h in self.instruction_target_hashes:
            return True, "Normalized text duplicate of existing example"

        # 3. Code AST duplicate check (only if exact same instruction intent and exact same AST)
        if example.output_type == "code":
            fp = compute_code_ast_fingerprint(example.target_output)
            norm_inst = normalize_text(example.instruction)
            key = f"{norm_inst}::{fp}"
            if key in self.ast_fingerprints:
                return True, f"Identical AST structure and instruction to example {self.ast_fingerprints[key]}"

        return False, ""

    def register(self, example: TrainingExample) -> None:
        """Registers accepted example into tracking sets."""
        combined = f"{example.instruction.strip()}###{example.target_output.strip()}"
        self.exact_hashes.add(hashlib.sha256(combined.encode("utf-8")).hexdigest())

        norm_combined = f"{normalize_text(example.instruction)}###{normalize_text(example.target_output)}"
        self.instruction_target_hashes.add(hashlib.sha256(norm_combined.encode("utf-8")).hexdigest())

        if example.output_type == "code":
            fp = compute_code_ast_fingerprint(example.target_output)
            norm_inst = normalize_text(example.instruction)
            key = f"{norm_inst}::{fp}"
            self.ast_fingerprints[key] = example.id
