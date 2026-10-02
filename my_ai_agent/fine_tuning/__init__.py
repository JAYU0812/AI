"""Fine-tuning package for Step 6D — Controlled Fine-Tuning Experiment.

This package implements the infrastructure for the Step 6D controlled experiment
comparing Qwen2.5-Coder-0.5B-Instruct baseline vs LoRA fine-tuned model.

Architecture:
    config.py          -- Experiment configuration and hyperparameters
    dataset.py         -- Dataset loading and prompt formatting
    tokenizer.py       -- Tokenizer utilities and length analysis
    trainer.py         -- LoRA fine-tuning loop
    checkpoint.py      -- Checkpoint management
    evaluator.py       -- Step 6C test split evaluation
    comparison.py      -- Baseline vs fine-tuned comparison
    resource_monitor.py -- CPU/RAM usage tracking
    safety.py          -- Safety regression checks
    inference.py       -- Inference backend for fine-tuned adapter
    experiment.py      -- End-to-end experiment orchestrator
    report.py          -- Report generation

IMPORTANT:
    - ZERO model weight changes to Qwen cache
    - Step 6A held-out benchmark protected (never used for training)
    - LoRA only: full fine-tuning NOT feasible on CPU-only hardware
    - Baseline preserved and recoverable at all times
"""
