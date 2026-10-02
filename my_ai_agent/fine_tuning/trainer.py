"""Step 6D — LoRA Fine-Tuning Trainer.

CPU-only LoRA training loop for Qwen2.5-Coder-0.5B-Instruct.

Key design decisions for CPU-only training:
    1. No DataLoader workers (num_workers=0 on Windows CPU)
    2. float32 throughout (bfloat16 not reliably supported without hardware)
    3. Very conservative batch size (1) + gradient accumulation (4)
    4. Full training loss + validation loss tracked per epoch
    5. Best checkpoint saved based on validation loss
    6. Overfitting detection: train_loss << val_loss triggers warning
    7. Early stopping NOT implemented (only 3 epochs, not enough to warrant it)

Separation guarantee:
    This module NEVER touches the Step 6A benchmark data.
    It only uses training_dataset_v1.jsonl and validation_dataset_v1.jsonl.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from .config import ExperimentConfig, get_default_config
from .tokenizer import FineTuningTokenizedDataset, collate_fn
from .resource_monitor import ResourceMonitor


# ── LoRA injection ────────────────────────────────────────────────────────────

def build_lora_model(config: ExperimentConfig) -> Tuple[Any, Any]:
    """Load the base model and inject LoRA adapters.

    Returns:
        (peft_model, base_model) — base_model is kept for recovery if needed.

    Notes:
        - Base model weights are NEVER modified. LoRA adapters are separate.
        - float32 used throughout (CPU-only constraint).
        - The original Qwen HF cache is read-only.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"[Trainer] Loading base model: {config.base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.train()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias=config.lora_bias,
        inference_mode=False,
    )

    peft_model = get_peft_model(model, lora_cfg)
    trainable, total = peft_model.get_nb_trainable_parameters()
    print(f"[Trainer] LoRA injected: {trainable:,} trainable params ({100*trainable/total:.3f}%)")
    return peft_model, model


# ── Training Loop ─────────────────────────────────────────────────────────────

class LoRATrainer:
    """Step 6D controlled fine-tuning trainer.

    Implements the training loop with:
        - Per-step loss tracking
        - Per-epoch validation loss
        - Overfitting detection
        - Checkpoint management
        - Resource monitoring
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
    ):
        self.config = config or get_default_config()
        self.monitor = resource_monitor
        self.metrics: Dict[str, Any] = {
            "experiment_id": self.config.experiment_id,
            "train_losses": [],  # list of (step, loss)
            "epoch_train_losses": [],
            "epoch_val_losses": [],
            "best_val_loss": float("inf"),
            "best_epoch": -1,
            "total_steps": 0,
            "total_tokens": 0,
            "training_time_sec": 0.0,
            "overfitting_detected": False,
        }
        self._best_checkpoint_path: Optional[str] = None

    def _get_dataloader(self, dataset: FineTuningTokenizedDataset, shuffle: bool) -> DataLoader:
        tok_pad_id = self.config.experiment_id  # placeholder; set at call site
        return dataset  # Yielded manually below (no DataLoader for simplicity)

    def run_pilot(
        self,
        peft_model: Any,
        tokenizer: PreTrainedTokenizer,
        train_texts: List[str],
    ) -> Dict[str, Any]:
        """Pilot run: verify the full pipeline in 3 micro-steps (Step 6D.12).

        Returns:
            dict with pilot results (success/failure + loss values).
        """
        print(f"\n[Pilot] Starting pilot run ({self.config.pilot_max_steps} steps)...")
        pilot_result: Dict[str, Any] = {
            "status": "FAIL",
            "steps_completed": 0,
            "losses": [],
            "errors": [],
        }

        torch.manual_seed(self.config.seed)

        # Build tiny dataset (just first 3 examples, possibly repeated)
        pilot_texts = (train_texts * 10)[:self.config.pilot_max_steps]
        dataset = FineTuningTokenizedDataset(pilot_texts, tokenizer, self.config.max_seq_length)

        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

        peft_model.train()
        step = 0

        for item in dataset:
            if step >= self.config.pilot_max_steps:
                break
            try:
                input_ids = item["input_ids"].unsqueeze(0)
                attention_mask = item["attention_mask"].unsqueeze(0)
                labels = item["labels"].unsqueeze(0)

                # Forward pass
                outputs = peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                if loss is None or torch.isnan(loss):
                    pilot_result["errors"].append(f"Step {step}: loss is None or NaN")
                    break

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_val = loss.item()
                pilot_result["losses"].append(loss_val)
                step += 1
                print(f"[Pilot] Step {step}/{self.config.pilot_max_steps} — loss: {loss_val:.4f}")

            except Exception as e:
                pilot_result["errors"].append(f"Step {step}: {type(e).__name__}: {e}")
                break

        pilot_result["steps_completed"] = step
        if step >= self.config.pilot_max_steps and not pilot_result["errors"]:
            pilot_result["status"] = "PASS"
            print("[Pilot] PASS — all micro-steps completed without error")
        else:
            print(f"[Pilot] FAIL — {pilot_result['errors']}")

        return pilot_result

    def train(
        self,
        peft_model: Any,
        tokenizer: PreTrainedTokenizer,
        train_texts: List[str],
        val_texts: List[str],
    ) -> Dict[str, Any]:
        """Full training loop.

        Args:
            peft_model: LoRA-injected model.
            tokenizer: Loaded tokenizer.
            train_texts: ChatML-formatted training strings.
            val_texts: ChatML-formatted validation strings.

        Returns:
            Metrics dictionary.
        """
        print(f"\n[Trainer] Starting training — {len(train_texts)} train, {len(val_texts)} val")
        print(f"[Trainer] Config: epochs={self.config.num_train_epochs}, "
              f"lr={self.config.learning_rate}, "
              f"grad_accum={self.config.gradient_accumulation_steps}")

        torch.manual_seed(self.config.seed)
        self.config.create_dirs()

        train_dataset = FineTuningTokenizedDataset(train_texts, tokenizer, self.config.max_seq_length)
        val_dataset = FineTuningTokenizedDataset(val_texts, tokenizer, self.config.max_seq_length)

        pad_id = tokenizer.pad_token_id

        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Total steps for scheduler
        steps_per_epoch = max(1, len(train_dataset) // self.config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.config.num_train_epochs
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps)
        )

        if self.monitor:
            self.monitor.record_event("training_start")

        global_step = 0
        train_start = time.time()

        for epoch in range(1, self.config.num_train_epochs + 1):
            peft_model.train()
            epoch_loss_sum = 0.0
            epoch_steps = 0
            accumulated_loss = 0.0
            accum_count = 0

            print(f"\n[Trainer] === Epoch {epoch}/{self.config.num_train_epochs} ===")

            # Shuffle training indices
            import random
            indices = list(range(len(train_dataset)))
            random.shuffle(indices)

            optimizer.zero_grad()

            for i, idx in enumerate(indices):
                item = train_dataset[idx]
                input_ids = item["input_ids"].unsqueeze(0)
                attention_mask = item["attention_mask"].unsqueeze(0)
                labels = item["labels"].unsqueeze(0)

                outputs = peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                if loss is None or torch.isnan(loss):
                    print(f"[Trainer] Warning: NaN loss at epoch={epoch}, i={i}")
                    continue

                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                accumulated_loss += loss.item()
                accum_count += 1

                if accum_count >= self.config.gradient_accumulation_steps or i == len(indices) - 1:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in peft_model.parameters() if p.requires_grad],
                        self.config.max_grad_norm,
                    )
                    optimizer.step()
                    if global_step >= warmup_steps:
                        scheduler.step()
                    optimizer.zero_grad()

                    avg_step_loss = accumulated_loss / accum_count
                    self.metrics["train_losses"].append((global_step, avg_step_loss))
                    epoch_loss_sum += avg_step_loss
                    epoch_steps += 1
                    global_step += 1
                    accumulated_loss = 0.0
                    accum_count = 0

                    current_lr = optimizer.param_groups[0]["lr"]
                    if global_step % 5 == 0 or global_step <= 3:
                        print(f"[Trainer] Step {global_step} | loss: {avg_step_loss:.4f} | lr: {current_lr:.2e}")

            # Epoch train loss
            epoch_train_loss = epoch_loss_sum / max(1, epoch_steps)
            self.metrics["epoch_train_losses"].append(epoch_train_loss)

            # Validation
            val_loss = self._evaluate_loss(peft_model, val_dataset)
            self.metrics["epoch_val_losses"].append(val_loss)

            print(f"[Trainer] Epoch {epoch}: train_loss={epoch_train_loss:.4f}, val_loss={val_loss:.4f}")

            # Overfitting detection
            if epoch > 1:
                prev_val = self.metrics["epoch_val_losses"][-2]
                if val_loss > prev_val * 1.1 and epoch_train_loss < prev_val * 0.9:
                    self.metrics["overfitting_detected"] = True
                    print(f"[Trainer] WARNING: Overfitting detected at epoch {epoch} "
                          f"(train↓ {epoch_train_loss:.4f}, val↑ {val_loss:.4f})")

            # Save best checkpoint
            if val_loss < self.metrics["best_val_loss"]:
                self.metrics["best_val_loss"] = val_loss
                self.metrics["best_epoch"] = epoch
                self._save_checkpoint(peft_model, tokenizer, epoch, is_best=True)
            else:
                self._save_checkpoint(peft_model, tokenizer, epoch, is_best=False)

            if self.monitor:
                self.monitor.record_event(f"epoch_{epoch}_complete")

        training_sec = time.time() - train_start
        self.metrics["total_steps"] = global_step
        self.metrics["training_time_sec"] = round(training_sec, 1)
        self.metrics["training_time_hms"] = _format_duration(training_sec)

        if self.monitor:
            self.monitor.record_event("training_complete")

        print(f"\n[Trainer] Training complete in {self.metrics['training_time_hms']}")
        print(f"[Trainer] Best val_loss: {self.metrics['best_val_loss']:.4f} (epoch {self.metrics['best_epoch']})")
        print(f"[Trainer] Overfitting detected: {self.metrics['overfitting_detected']}")

        return self.metrics

    def _evaluate_loss(
        self,
        peft_model: Any,
        val_dataset: FineTuningTokenizedDataset,
    ) -> float:
        """Compute average loss on validation set."""
        peft_model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for item in val_dataset:
                input_ids = item["input_ids"].unsqueeze(0)
                attention_mask = item["attention_mask"].unsqueeze(0)
                labels = item["labels"].unsqueeze(0)
                outputs = peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                if outputs.loss is not None and not torch.isnan(outputs.loss):
                    total_loss += outputs.loss.item()
                    count += 1

        peft_model.train()
        return total_loss / max(1, count)

    def _save_checkpoint(
        self,
        peft_model: Any,
        tokenizer: PreTrainedTokenizer,
        epoch: int,
        is_best: bool,
    ) -> str:
        """Save LoRA adapter checkpoint."""
        label = "best" if is_best else f"epoch_{epoch}"
        ckpt_path = Path(self.config.checkpoint_dir) / label
        peft_model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))
        meta = {
            "epoch": epoch,
            "val_loss": self.metrics["epoch_val_losses"][-1],
            "train_loss": self.metrics["epoch_train_losses"][-1],
            "is_best": is_best,
        }
        with open(ckpt_path / "checkpoint_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if is_best:
            self._best_checkpoint_path = str(ckpt_path)
            print(f"[Trainer] Best checkpoint saved: {ckpt_path}")
        return str(ckpt_path)

    def save_metrics(self, path: Optional[str] = None) -> str:
        """Save training metrics to JSON."""
        p = Path(path or self.config.metrics_dir) / "training_metrics.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, default=str)
        return str(p)

    @property
    def best_checkpoint_path(self) -> Optional[str]:
        return self._best_checkpoint_path


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"
