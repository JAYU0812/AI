"""Step 6D — Inference Backend for Fine-Tuned and Baseline Models.

Provides a unified interface for running inference on:
  1. The baseline Qwen2.5-Coder-0.5B-Instruct (read-only from HF cache)
  2. The fine-tuned model (base + LoRA adapter)

Both produce output using the same parameters so that the only variable
between baseline and fine-tuned evaluation is the model weights.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from .config import ExperimentConfig, get_default_config
from .dataset import format_as_inference_prompt, _build_instruction_text


class BaselineInference:
    """Inference against the original unmodified Qwen2.5-Coder-0.5B-Instruct.

    The model is loaded in float32, read-only from the HF cache.
    No gradient tracking. No modification.
    """

    def __init__(self, model_name: str, config: Optional[ExperimentConfig] = None):
        self.model_name = model_name
        self.config = config or get_default_config()
        self._model: Optional[Any] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._loaded = False

    def load(self) -> None:
        """Load baseline model. Call once before inference."""
        print(f"[BaselineInference] Loading: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            device_map="cpu",
        )
        self._model.eval()
        self._loaded = True
        print("[BaselineInference] Model loaded (baseline, unmodified)")

    def unload(self) -> None:
        """Free model memory."""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False
        import gc
        gc.collect()

    def generate(self, instruction: str) -> Dict[str, Any]:
        """Generate a response for the given instruction."""
        if not self._loaded:
            raise RuntimeError("Call load() first")
        return _run_inference(
            model=self._model,
            tokenizer=self._tokenizer,
            instruction=instruction,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
        )

    def generate_batch(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Generate responses for a list of instructions."""
        return [self.generate(inst) for inst in instructions]


class FineTunedInference:
    """Inference against the LoRA fine-tuned model.

    Loads the base model + LoRA adapter.
    Same inference parameters as BaselineInference to isolate the weight effect.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_path: str,
        config: Optional[ExperimentConfig] = None,
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.config = config or get_default_config()
        self._model: Optional[Any] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._loaded = False

    def load(self) -> None:
        """Load fine-tuned model (base + LoRA adapter)."""
        from .checkpoint import load_adapter_for_inference
        self._model, self._tokenizer = load_adapter_for_inference(
            base_model_name=self.base_model_name,
            adapter_path=self.adapter_path,
        )
        self._loaded = True
        print(f"[FineTunedInference] Model + adapter loaded from: {self.adapter_path}")

    def unload(self) -> None:
        """Free model memory."""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False
        import gc
        gc.collect()

    def generate(self, instruction: str) -> Dict[str, Any]:
        """Generate a response for the given instruction."""
        if not self._loaded:
            raise RuntimeError("Call load() first")
        return _run_inference(
            model=self._model,
            tokenizer=self._tokenizer,
            instruction=instruction,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
        )

    def generate_batch(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Generate responses for a list of instructions."""
        return [self.generate(inst) for inst in instructions]


def _run_inference(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> Dict[str, Any]:
    """Core inference function.

    Uses greedy decoding (do_sample=False, temperature=None) for determinism.
    Identical parameters for baseline and fine-tuned to isolate weight effect.

    Returns:
        dict with 'output', 'tokens_generated', 'latency_sec'.
    """
    prompt = format_as_inference_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=None,   # Greedy when do_sample=False
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0

    # Decode only the generated portion (not the prompt)
    generated_ids = output_ids[0][input_len:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "output": decoded.strip(),
        "tokens_generated": len(generated_ids),
        "latency_sec": round(latency, 3),
        "tokens_per_sec": round(len(generated_ids) / max(latency, 0.001), 2),
    }
