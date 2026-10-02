"""Local Inference Backend Abstraction for Step 5.

Provides a pluggable runtime interface for local open-source models:
- TransformersBackend: HuggingFace Transformers + PyTorch (CPU/CUDA)
- OllamaBackend: Local Ollama daemon REST API
- NullBackend: Safe unconfigured/unloaded fallback

CRITICAL: Never simulate inference. Tokens, latencies, and generation text
must come directly from the underlying runtime execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, Generator, List, Optional

from my_ai_agent.intelligence.model_response import FinishReason, TokenMetrics


class LocalInferenceBackend(ABC):
    """Abstract interface for local model execution runtimes."""

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True only if the backend runtime and weights are loaded and ready."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns factual runtime metadata (backend name, model id, device, context size)."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """Loads model weights into memory."""
        pass

    @abstractmethod
    def unload(self) -> bool:
        """Unloads model weights from memory."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Executes real local inference.
        
        Returns dict with:
        - text: generated text
        - tokens: TokenMetrics
        - finish_reason: FinishReason
        - latency_seconds: float
        - metadata: dict
        """
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Streams real token chunks incrementally from the model."""
        pass


class NullBackend(LocalInferenceBackend):
    """Fallback backend when no local runtime is configured or weights are missing."""

    def __init__(self, message: str = "Local model backend is not loaded"):
        self.message = message

    def is_available(self) -> bool:
        return False

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "backend": "null",
            "is_available": False,
            "message": self.message,
        }

    def load(self) -> bool:
        return False

    def unload(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        raise RuntimeError(
            f"Cannot execute inference: {self.message}. "
            "No fake inference permitted. Switch to deterministic mode or configure weights."
        )

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        raise RuntimeError(
            f"Cannot execute streaming: {self.message}. "
            "No fake streaming permitted."
        )


class TransformersBackend(LocalInferenceBackend):
    """Inference backend powered by HuggingFace Transformers and PyTorch."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        device: str = "cpu",
        context_length: int = 2048,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.context_length = context_length
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return self._is_loaded and self._model is not None and self._tokenizer is not None

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "backend": "transformers",
            "model_name_or_path": self.model_name_or_path,
            "device": self.device,
            "context_length": self.context_length,
            "is_available": self._is_loaded,
        }

    def load(self) -> bool:
        """Loads weights and tokenizer using transformers."""
        with self._lock:
            if self._is_loaded:
                return True
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                target_device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
                self.device = target_device

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    dtype=torch.float32 if target_device == "cpu" else torch.float16,
                    trust_remote_code=True,
                ).to(target_device)

                self._model.eval()
                self._is_loaded = True
                return True
            except Exception as e:
                self._is_loaded = False
                self._model = None
                self._tokenizer = None
                return False

    def unload(self) -> bool:
        """Frees model weights from memory."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._is_loaded = False
            import gc
            gc.collect()
            return True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Executes real local inference via PyTorch."""
        if not self.is_available():
            if not self.load():
                raise RuntimeError(
                    f"Transformers model '{self.model_name_or_path}' could not be loaded."
                )

        import torch

        start_time = time.time()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        input_token_count = int(inputs["input_ids"].shape[1])

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = max(0.01, temperature)
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_tokens = self._model.generate(**inputs, **gen_kwargs)

        output_ids = output_tokens[0]
        total_token_count = len(output_ids)
        new_token_ids = output_ids[input_token_count:]
        output_token_count = len(new_token_ids)

        generated_text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
        latency = time.time() - start_time

        # Real factual token metrics
        tokens = TokenMetrics(
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            context_size=total_token_count,
            is_available=True,
        )

        return {
            "text": generated_text,
            "tokens": tokens,
            "finish_reason": FinishReason.STOP,
            "latency_seconds": latency,
            "metadata": {
                "device": self.device,
                "model": self.model_name_or_path,
                "backend": "transformers",
            },
        }

    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Streams real token chunks incrementally via TextIteratorStreamer."""
        if not self.is_available():
            if not self.load():
                raise RuntimeError(
                    f"Transformers model '{self.model_name_or_path}' could not be loaded."
                )

        from transformers import TextIteratorStreamer
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "streamer": streamer,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "do_sample": (temperature > 0.0),
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = max(0.01, temperature)

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        for chunk in streamer:
            if chunk:
                yield chunk

        thread.join()


class OllamaBackend(LocalInferenceBackend):
    """Inference backend connecting to local Ollama daemon REST API."""

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:0.5b",
        base_url: str = "http://localhost:11434",
        context_length: int = 4096,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.context_length = context_length

    def is_available(self) -> bool:
        """Pings local Ollama service to verify daemon and model availability."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # Check if requested model or model prefix is installed
                    return any(self.model_name in m for m in models)
        except Exception:
            pass
        return False

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "backend": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "context_length": self.context_length,
            "is_available": self.is_available(),
        }

    def load(self) -> bool:
        return self.is_available()

    def unload(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Executes real local inference via Ollama HTTP API."""
        import urllib.request

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        start_time = time.time()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Ollama inference error: {e}")

        latency = time.time() - start_time
        gen_text = data.get("response", "")
        prompt_tokens = data.get("prompt_eval_count")
        eval_tokens = data.get("eval_count")

        tokens = TokenMetrics(
            input_tokens=prompt_tokens,
            output_tokens=eval_tokens,
            context_size=(prompt_tokens + eval_tokens) if (prompt_tokens and eval_tokens) else None,
            is_available=(prompt_tokens is not None),
        )

        return {
            "text": gen_text,
            "tokens": tokens,
            "finish_reason": FinishReason.STOP if data.get("done") else FinishReason.UNKNOWN,
            "latency_seconds": latency,
            "metadata": {"backend": "ollama", "model": self.model_name},
        }

    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Streams real token chunks from Ollama API."""
        import urllib.request

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if line:
                    chunk_data = json.loads(line.decode("utf-8"))
                    chunk_text = chunk_data.get("response", "")
                    if chunk_text:
                        yield chunk_text
                    if chunk_data.get("done", False):
                        break
