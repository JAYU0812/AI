"""Step 6D — Experiment Configuration.

All hyperparameters, paths, and experiment settings are defined here.
Every value is documented. The experiment is reproducible given this config.

Hardware-aware defaults chosen for:
    CPU: Intel Core i7-8550U (Coffee Lake)
    RAM: ~16 GB
    GPU: NONE (Intel UHD 620 — no CUDA/VRAM)
    Device: CPU ONLY

CPU-only LoRA constraints:
    - float32 for all operations (bfloat16 unsupported without CUDA)
    - Very small batch size (1) + gradient accumulation
    - Small max_seq_length to limit peak RAM
    - Few epochs — dataset only has 32 training examples
    - Conservative LoRA rank (r=8) to limit adapter size
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ── Root directories ────────────────────────────────────────────────────────

_PACKAGE_ROOT = Path(__file__).parent.parent  # my_ai_agent/
_FINE_TUNING_ROOT = Path(__file__).parent      # my_ai_agent/fine_tuning/
_RESULTS_ROOT = _FINE_TUNING_ROOT / "results"
_TRAINING_DATA_ROOT = _PACKAGE_ROOT / "training" / "results"

# ── Model ────────────────────────────────────────────────────────────────────

BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
"""The baseline model. Never overwritten. Loaded read-only from HF cache."""

# ── Dataset paths ────────────────────────────────────────────────────────────

TRAINING_JSONL = _TRAINING_DATA_ROOT / "training_dataset_v1.jsonl"
VALIDATION_JSONL = _TRAINING_DATA_ROOT / "validation_dataset_v1.jsonl"
TEST_JSONL = _TRAINING_DATA_ROOT / "test_dataset_v1.jsonl"
DATASET_MANIFEST = _TRAINING_DATA_ROOT / "dataset_manifest_v1.json"

# ── Experiment paths ─────────────────────────────────────────────────────────

EXPERIMENT_ID = "experiment_001"
EXPERIMENT_DIR = _RESULTS_ROOT / EXPERIMENT_ID
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
METRICS_DIR = EXPERIMENT_DIR / "metrics"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
ADAPTER_DIR = EXPERIMENT_DIR / "adapter"


@dataclass
class ExperimentConfig:
    """Complete reproducible experiment configuration for Step 6D.

    All values documented with rationale for CPU-only hardware constraints.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    experiment_id: str = "experiment_001"
    experiment_version: str = "1.0"
    seed: int = 42
    """Reproducibility seed. Fixed to 42 for all runs."""

    # ── Model ────────────────────────────────────────────────────────────────
    base_model_name: str = BASE_MODEL_NAME
    model_revision: str = "main"
    torch_dtype: str = "float32"
    """float32 required — CPU-only, bfloat16 unsupported without hardware support."""

    # ── LoRA Configuration ───────────────────────────────────────────────────
    lora_r: int = 8
    """LoRA rank. 8 is conservative — small dataset, CPU-only. Higher ranks waste memory."""
    lora_alpha: int = 16
    """LoRA alpha. Typically 2x rank. Controls effective learning rate scaling."""
    lora_dropout: float = 0.05
    """LoRA dropout. Small regularization to reduce overfitting on 32 examples."""
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    """Target only query and value projections. Minimum for learning; avoids OOM."""
    lora_bias: str = "none"

    # ── Training ─────────────────────────────────────────────────────────────
    num_train_epochs: int = 3
    """3 epochs on 32 examples = 96 gradient steps (with batch_size=1, grad_accum=4).
    Conservative to limit overfitting. Dataset is too small for more epochs."""
    per_device_train_batch_size: int = 1
    """Batch size 1 — CPU RAM constraint. Effective batch = batch_size × grad_accum."""
    gradient_accumulation_steps: int = 4
    """Effective batch size = 4. Simulates mini-batch training without OOM."""
    learning_rate: float = 2e-4
    """2e-4 is standard for LoRA fine-tuning. Higher than full fine-tuning."""
    weight_decay: float = 0.01
    """Light L2 regularization. Important given tiny dataset."""
    warmup_ratio: float = 0.1
    """10% warmup. With 96 steps, ~9-10 warmup steps."""
    lr_scheduler_type: str = "cosine"
    """Cosine decay from peak LR. Preferred for small datasets."""
    max_grad_norm: float = 1.0
    """Gradient clipping. Standard value."""

    # ── Sequence Length ──────────────────────────────────────────────────────
    max_seq_length: int = 512
    """512 tokens. CPU RAM constraint. Model supports 32768 but that would OOM.
    All 42 dataset examples should fit within 512 tokens (verified in preflight)."""

    # ── Checkpointing ────────────────────────────────────────────────────────
    save_strategy: str = "epoch"
    """Save at end of each epoch. 3 epochs = 3 checkpoints + 1 best."""
    save_total_limit: int = 2
    """Keep only the best + most recent checkpoint. Disk space conservative."""
    load_best_model_at_end: bool = True
    eval_strategy: str = "epoch"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # ── Pilot Run ────────────────────────────────────────────────────────────
    pilot_max_steps: int = 3
    """Pilot run: 3 steps only. Purpose: verify pipeline, not train."""

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_batch_size: int = 1
    max_new_tokens: int = 256
    """Max tokens generated during evaluation."""
    eval_temperature: float = 0.0
    """Greedy decoding during evaluation for determinism."""
    do_sample: bool = False

    # ── Resource Monitoring ──────────────────────────────────────────────────
    monitor_interval_sec: float = 10.0
    """Poll CPU/RAM every N seconds during training."""

    # ── Paths ────────────────────────────────────────────────────────────────
    training_jsonl: str = str(TRAINING_JSONL)
    validation_jsonl: str = str(VALIDATION_JSONL)
    test_jsonl: str = str(TEST_JSONL)
    dataset_manifest: str = str(DATASET_MANIFEST)
    experiment_dir: str = str(EXPERIMENT_DIR)
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    metrics_dir: str = str(METRICS_DIR)
    reports_dir: str = str(REPORTS_DIR)
    adapter_dir: str = str(ADAPTER_DIR)

    def to_dict(self) -> dict:
        """Serialize configuration for reproducibility artifact."""
        import dataclasses
        d = dataclasses.asdict(self)
        d["base_model_name"] = self.base_model_name
        return d

    def save(self, path: Optional[str] = None) -> str:
        """Save config to JSON for reproducibility."""
        import json
        p = Path(path) if path else Path(self.experiment_dir) / "experiment_config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return str(p)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Filter only known fields
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def create_dirs(self) -> None:
        """Ensure all experiment directories exist."""
        for p in [
            self.experiment_dir,
            self.checkpoint_dir,
            self.metrics_dir,
            self.reports_dir,
            self.adapter_dir,
        ]:
            Path(p).mkdir(parents=True, exist_ok=True)


def get_default_config() -> ExperimentConfig:
    """Return the default Step 6D experiment configuration."""
    return ExperimentConfig()


def get_pilot_config() -> ExperimentConfig:
    """Return a minimal pilot-run configuration (3 steps only)."""
    cfg = ExperimentConfig()
    cfg.experiment_id = "pilot_001"
    cfg.num_train_epochs = 1
    cfg.experiment_dir = str(_RESULTS_ROOT / "pilot_001")
    cfg.checkpoint_dir = str(_RESULTS_ROOT / "pilot_001" / "checkpoints")
    cfg.metrics_dir = str(_RESULTS_ROOT / "pilot_001" / "metrics")
    cfg.reports_dir = str(_RESULTS_ROOT / "pilot_001" / "reports")
    cfg.adapter_dir = str(_RESULTS_ROOT / "pilot_001" / "adapter")
    return cfg
