"""Step 6D — Checkpoint Management.

Handles:
  - Saving LoRA adapter + tokenizer to isolated experiment directory
  - Loading best checkpoint for evaluation
  - Verifying adapter loads and inference works
  - Never touching the original Qwen HF cache
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def save_final_adapter(
    peft_model: Any,
    tokenizer: Any,
    adapter_dir: str,
    config_dict: Optional[Dict] = None,
) -> str:
    """Save the final LoRA adapter and tokenizer.

    Args:
        peft_model: The trained PEFT model.
        tokenizer: The tokenizer used for training.
        adapter_dir: Target directory (isolated from HF cache).
        config_dict: Optional metadata to save alongside adapter.

    Returns:
        Path to saved adapter directory.
    """
    p = Path(adapter_dir)
    p.mkdir(parents=True, exist_ok=True)

    peft_model.save_pretrained(str(p))
    tokenizer.save_pretrained(str(p))

    if config_dict:
        with open(p / "experiment_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)

    # Write adapter metadata
    adapter_meta = {
        "adapter_type": "LoRA",
        "base_model": peft_model.base_model.config._name_or_path if hasattr(peft_model, "base_model") else "unknown",
        "adapter_path": str(p),
        "files": [x.name for x in p.iterdir() if x.is_file()],
    }
    with open(p / "adapter_metadata.json", "w", encoding="utf-8") as f:
        json.dump(adapter_meta, f, indent=2)

    print(f"[Checkpoint] Final adapter saved to: {p}")
    return str(p)


def load_adapter_for_inference(
    base_model_name: str,
    adapter_path: str,
    device: str = "cpu",
) -> Tuple[Any, Any]:
    """Load a saved LoRA adapter on top of the base model.

    Args:
        base_model_name: HF model name for the frozen base.
        adapter_path: Path to saved LoRA adapter directory.
        device: Target device ("cpu").

    Returns:
        (peft_model, tokenizer) ready for inference.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Checkpoint] Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
        device_map=device,
    )

    print(f"[Checkpoint] Loading adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    print("[Checkpoint] Adapter loaded and attached successfully")
    return peft_model, tokenizer


def verify_adapter_inference(
    peft_model: Any,
    tokenizer: Any,
    test_prompt: str = "Write a Python function that adds two numbers.",
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """Verify that the loaded adapter can perform inference (Step 6D.16).

    Returns:
        dict with status, output, and any errors.
    """
    from .dataset import format_as_inference_prompt

    result: Dict[str, Any] = {
        "status": "FAIL",
        "prompt": test_prompt,
        "output": "",
        "tokens_generated": 0,
        "error": None,
    }

    try:
        prompt = format_as_inference_prompt(test_prompt)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)

        peft_model.eval()
        with torch.no_grad():
            output_ids = peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated portion
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)

        result["output"] = decoded.strip()
        result["tokens_generated"] = len(generated)
        result["status"] = "PASS" if decoded.strip() else "EMPTY_OUTPUT"
        print(f"[Checkpoint] Inference verification: {result['status']}")
        print(f"[Checkpoint] Sample output (first 200 chars): {decoded[:200]}")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"[Checkpoint] Inference verification FAILED: {e}")

    return result


def get_checkpoint_metadata(checkpoint_dir: str) -> Dict[str, Any]:
    """Read checkpoint metadata if available."""
    p = Path(checkpoint_dir) / "checkpoint_meta.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "No checkpoint_meta.json found"}


def list_checkpoints(checkpoints_root: str) -> list:
    """List all checkpoints in the experiment directory."""
    root = Path(checkpoints_root)
    if not root.exists():
        return []
    return sorted([str(p) for p in root.iterdir() if p.is_dir()])
