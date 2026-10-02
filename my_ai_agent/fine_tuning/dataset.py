"""Step 6D — Dataset Loading and Prompt Formatting.

Loads Step 6C JSONL datasets and converts each TrainingExample into the
chat-formatted prompt+response string that the Qwen tokenizer expects.

IMPORTANT:
    - Only training_dataset_v1.jsonl and validation_dataset_v1.jsonl are used
      for training/validation.
    - test_dataset_v1.jsonl is used ONLY as a Step 6C internal test evaluation.
    - The Step 6A held-out benchmark is NEVER loaded here.
    - The manifest SHA-256 hashes are verified before any data is loaded.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ExperimentConfig, get_default_config

# ── Qwen ChatML format ────────────────────────────────────────────────────────
# Qwen2 uses ChatML: <|im_start|>role\ncontent<|im_end|>\n
_SYSTEM_PROMPT = (
    "You are an expert coding assistant. "
    "When asked to write or modify code, produce only the correct, complete Python code. "
    "When asked for structured JSON output, produce only valid JSON. "
    "Follow all constraints exactly."
)


# ── Manifest Verification ─────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset_manifest(manifest_path: str) -> Tuple[bool, List[str]]:
    """Verify dataset file integrity via SHA-256 hashes from the manifest.

    Returns:
        (all_ok, list_of_errors)
    """
    mpath = Path(manifest_path)
    if not mpath.exists():
        return False, [f"Manifest not found: {manifest_path}"]

    with open(mpath, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    errors = []
    hashes = manifest.get("file_hashes", {})
    base_dir = mpath.parent

    for fname, expected_hash in hashes.items():
        fpath = base_dir / fname
        if not fpath.exists():
            errors.append(f"File missing: {fname}")
            continue
        actual = _sha256_file(fpath)
        if actual != expected_hash:
            errors.append(f"Hash mismatch for {fname}: expected {expected_hash[:8]}... got {actual[:8]}...")

    return len(errors) == 0, errors


# ── Raw Loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []
    with open(p, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON error at line {lineno} in {path}: {e}")
    return records


# ── Prompt Formatting ─────────────────────────────────────────────────────────

def _build_instruction_text(record: Dict[str, Any]) -> str:
    """Construct the full instruction from a training record.

    Incorporates: instruction, context, constraints, expected_behavior.
    """
    parts = [record.get("instruction", "").strip()]

    context = record.get("context", {})
    if context:
        project_type = context.get("project_type", "")
        files = context.get("files", [])
        if project_type:
            parts.append(f"Project type: {project_type}")
        if files:
            # Summarize existing files
            file_summary = ", ".join(str(f.get("path", f)) for f in files[:5])
            parts.append(f"Existing files: {file_summary}")

    constraints = record.get("constraints", [])
    if constraints:
        constraint_text = " ".join(f"[{i+1}] {c}" for i, c in enumerate(constraints))
        parts.append(f"Constraints: {constraint_text}")

    expected = record.get("expected_behavior", "")
    if expected:
        parts.append(f"Expected: {expected}")

    return "\n".join(parts)


def format_as_chatml(record: Dict[str, Any]) -> str:
    """Format a training record as Qwen2 ChatML for training.

    Format:
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {instruction}<|im_end|>
        <|im_start|>assistant
        {target_output}<|im_end|>

    The entire string becomes the training sequence.
    Loss is computed on the assistant turn only (masked in tokenizer.py).
    """
    instruction = _build_instruction_text(record)
    target = record.get("target_output", "").strip()

    text = (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{target}<|im_end|>"
    )
    return text


def format_as_inference_prompt(instruction: str) -> str:
    """Format an instruction as an inference-only prompt (no target).

    Used during evaluation to get model predictions.
    """
    text = (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return text


# ── Preflight Validation ──────────────────────────────────────────────────────

_REQUIRED_FIELDS = {"id", "instruction", "target_output", "task_type", "output_type"}

def preflight_record(record: Dict[str, Any], idx: int) -> List[str]:
    """Validate a single record before training. Returns list of error strings."""
    errors = []
    missing = _REQUIRED_FIELDS - set(record.keys())
    if missing:
        errors.append(f"Record {idx} ({record.get('id', '?')}): missing fields {missing}")
    if not record.get("instruction", "").strip():
        errors.append(f"Record {idx} ({record.get('id', '?')}): empty instruction")
    if not record.get("target_output", "").strip():
        errors.append(f"Record {idx} ({record.get('id', '?')}): empty target_output")
    prov = record.get("provenance", {})
    if not prov or not prov.get("source_task_id", ""):
        errors.append(f"Record {idx} ({record.get('id', '?')}): missing provenance.source_task_id")
    return errors


def run_preflight(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> Dict[str, Any]:
    """Full dataset preflight check per Step 6D.8 requirements.

    Checks:
        - Schema validation (required fields)
        - Empty example check
        - Provenance check
        - Tokenization (no errors)
        - Maximum length check
        - Duplicate instruction check (within this split)
        - Secret token check (no leaked secrets in formatted text)

    Returns a preflight report dict.
    """
    import re as _re

    report: Dict[str, Any] = {
        "status": "PASS",
        "critical_errors": [],
        "warnings": [],
        "train_count": len(train_records),
        "val_count": len(val_records),
        "token_stats": {},
        "truncation_count": 0,
    }

    # ── Schema validation ────────────────────────────────────────────────────
    schema_errors = []
    for i, rec in enumerate(train_records + val_records):
        errs = preflight_record(rec, i)
        schema_errors.extend(errs)
    if schema_errors:
        report["critical_errors"].extend(schema_errors)
        report["status"] = "FAIL"

    # ── Token length analysis ─────────────────────────────────────────────────
    all_lengths = []
    truncations = 0
    for rec in train_records + val_records:
        text = format_as_chatml(rec)
        tokens = tokenizer.encode(text, truncation=False)
        length = len(tokens)
        all_lengths.append(length)
        if length > max_seq_length:
            truncations += 1
            report["warnings"].append(
                f"Record {rec.get('id', '?')}: {length} tokens > max_seq_length={max_seq_length} (will be truncated)"
            )

    if all_lengths:
        import statistics as _stat
        sorted_lengths = sorted(all_lengths)
        n = len(all_lengths)
        report["token_stats"] = {
            "min": sorted_lengths[0],
            "max": sorted_lengths[-1],
            "mean": round(sum(all_lengths) / n, 1),
            "median": sorted_lengths[n // 2],
            "p95": sorted_lengths[int(n * 0.95)],
            "max_seq_length_configured": max_seq_length,
        }
    report["truncation_count"] = truncations

    # ── Duplicate check ──────────────────────────────────────────────────────
    seen_instructions = {}
    for i, rec in enumerate(train_records):
        inst = rec.get("instruction", "").strip().lower()
        if inst in seen_instructions:
            report["warnings"].append(
                f"Possible duplicate: record {i} matches record {seen_instructions[inst]}"
            )
        seen_instructions[inst] = i

    # ── Secret check ─────────────────────────────────────────────────────────
    secret_pattern = _re.compile(
        r"(-----BEGIN [A-Z ]+KEY-----|AKIA[0-9A-Z]{16}|(?:password|passwd|pwd)\s*=\s*['\"][^'\"]{6,}['\"])",
        _re.IGNORECASE
    )
    for rec in train_records + val_records:
        text = format_as_chatml(rec)
        if secret_pattern.search(text):
            report["critical_errors"].append(
                f"Record {rec.get('id', '?')}: secret pattern detected in formatted text"
            )
            report["status"] = "FAIL"

    if report["critical_errors"] and report["status"] != "FAIL":
        report["status"] = "FAIL"

    return report


# ── Public Dataset API ────────────────────────────────────────────────────────

class FineTuningDataset:
    """Step 6D dataset loader.

    Loads, verifies, and formats training/validation data.
    Keeps strict separation from the Step 6A benchmark.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or get_default_config()
        self._train_records: Optional[List[Dict]] = None
        self._val_records: Optional[List[Dict]] = None
        self._test_records: Optional[List[Dict]] = None

    def load_all(self, verify_manifest: bool = True) -> None:
        """Load train, validation, and test splits. Optionally verify hashes."""
        if verify_manifest:
            ok, errors = verify_dataset_manifest(self.config.dataset_manifest)
            if not ok:
                raise ValueError(f"Dataset manifest verification FAILED: {errors}")

        self._train_records = load_jsonl(self.config.training_jsonl)
        self._val_records = load_jsonl(self.config.validation_jsonl)
        self._test_records = load_jsonl(self.config.test_jsonl)

    @property
    def train_records(self) -> List[Dict]:
        if self._train_records is None:
            raise RuntimeError("Call load_all() first")
        return self._train_records

    @property
    def val_records(self) -> List[Dict]:
        if self._val_records is None:
            raise RuntimeError("Call load_all() first")
        return self._val_records

    @property
    def test_records(self) -> List[Dict]:
        if self._test_records is None:
            raise RuntimeError("Call load_all() first")
        return self._test_records

    def get_formatted_train(self) -> List[str]:
        """Return ChatML-formatted training strings."""
        return [format_as_chatml(r) for r in self.train_records]

    def get_formatted_val(self) -> List[str]:
        """Return ChatML-formatted validation strings."""
        return [format_as_chatml(r) for r in self.val_records]

    def get_inference_prompts(self, split: str = "test") -> List[Tuple[str, str, str]]:
        """Return (id, instruction_text, target_output) for inference evaluation."""
        records = {"train": self._train_records, "val": self._val_records, "test": self._test_records}[split]
        if records is None:
            raise RuntimeError("Call load_all() first")
        result = []
        for r in records:
            inst = _build_instruction_text(r)
            prompt = format_as_inference_prompt(inst)
            result.append((r.get("id", ""), prompt, r.get("target_output", "")))
        return result

    def summary(self) -> Dict[str, Any]:
        """Return dataset summary statistics."""
        return {
            "train_count": len(self._train_records) if self._train_records else 0,
            "val_count": len(self._val_records) if self._val_records else 0,
            "test_count": len(self._test_records) if self._test_records else 0,
            "train_path": self.config.training_jsonl,
            "val_path": self.config.validation_jsonl,
            "test_path": self.config.test_jsonl,
            "manifest_path": self.config.dataset_manifest,
        }
