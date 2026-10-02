"""Step 6D — Tokenizer Utilities and Token Length Analysis.

Handles:
  - Tokenizer loading (Qwen2Tokenizer)
  - ChatML sequence tokenization with loss-masking (labels)
  - Token length statistics (Step 6D.9)
  - Collation for CPU-only training (no DataLoader workers)

Loss masking strategy:
    The full ChatML string is tokenized. The label for every token in the
    system/user turns is set to -100 (ignored). Only tokens in the
    <|im_start|>assistant ... <|im_end|> portion have valid labels.
    This teaches the model to predict the assistant response, not to
    predict the prompt or special tokens.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load the Qwen2 tokenizer.

    Notes:
        - PAD token: <|endoftext|> (id=151643). We use it for left-padding.
        - EOS token: <|im_end|> (id=151645).
        - The tokenizer has a built-in chat template.
    """
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Ensure pad token is set (Qwen2 has it as 151643)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    # Use left padding for decoder-only models during inference
    tok.padding_side = "right"  # right-pad during training (standard)
    return tok


def tokenize_with_labels(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize a full ChatML sequence and create masked labels.

    Labels for the system + user turns are set to -100 (ignored in loss).
    Labels for the assistant turn are set to the actual token ids.

    Args:
        text: Full ChatML string (system + user + assistant).
        tokenizer: Loaded tokenizer.
        max_seq_length: Hard truncation limit.

    Returns:
        dict with 'input_ids', 'attention_mask', 'labels' as 1D tensors.
    """
    # Split at assistant start to find the boundary
    ASSISTANT_START = "<|im_start|>assistant\n"
    if ASSISTANT_START in text:
        prompt_part = text[:text.index(ASSISTANT_START) + len(ASSISTANT_START)]
    else:
        prompt_part = text

    # Tokenize the full text
    full_ids = tokenizer.encode(text, truncation=True, max_length=max_seq_length)
    # Tokenize only the prompt portion to find where to unmask
    prompt_ids = tokenizer.encode(prompt_part, truncation=True, max_length=max_seq_length)
    prompt_len = len(prompt_ids)

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()
    # Mask everything up to (but not including) the assistant response
    # The assistant's first token starts at prompt_len
    labels[:prompt_len] = -100

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def compute_token_stats(texts: List[str], tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """Compute token length statistics for the dataset (Step 6D.9).

    Returns:
        dict with min, max, mean, median, p95, p99, and per-item lengths.
    """
    lengths = []
    for text in texts:
        ids = tokenizer.encode(text, truncation=False)
        lengths.append(len(ids))

    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0, "p95": 0, "p99": 0}

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def percentile(p: float) -> int:
        idx = min(int(math.ceil(n * p / 100)) - 1, n - 1)
        return lengths_sorted[max(0, idx)]

    return {
        "count": n,
        "min": lengths_sorted[0],
        "max": lengths_sorted[-1],
        "mean": round(sum(lengths) / n, 1),
        "median": lengths_sorted[n // 2],
        "p95": percentile(95),
        "p99": percentile(99),
        "lengths": lengths,
    }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """Collate a list of tokenized examples into a padded batch.

    Pads input_ids and attention_mask on the right.
    Labels are padded with -100 (ignored in loss).

    Args:
        batch: List of dicts with 'input_ids', 'attention_mask', 'labels'.
        pad_token_id: Token id used for padding input_ids.
        max_seq_length: Maximum sequence length (hard cap).
    """
    max_len = min(max(item["input_ids"].shape[0] for item in batch), max_seq_length)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        ids = item["input_ids"][:max_len]
        mask = item["attention_mask"][:max_len]
        labs = item["labels"][:max_len]

        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)])

        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        labels_list.append(labs)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


class FineTuningTokenizedDataset:
    """A simple list-backed dataset for CPU-only training.

    Compatible with torch.utils.data.DataLoader.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
    ):
        self._items: List[Dict[str, torch.Tensor]] = []
        for text in texts:
            item = tokenize_with_labels(text, tokenizer, max_seq_length)
            self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._items[idx]
