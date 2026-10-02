"""Step 6D — Step 6C Test Split and Generalization Evaluator.

Evaluates both baseline and fine-tuned models on:
  1. Step 6C internal test split (5 examples) — before and after fine-tuning
  2. Casual conversation smoke test (Step 6D.25)
  3. Generalization probe — novel tasks not in training or Step 6A benchmark

IMPORTANT: This evaluator does NOT run the Step 6A held-out benchmark.
That is handled separately by the existing benchmark runner in my_ai_agent/evaluation/.
"""

from __future__ import annotations

import re
import ast
import json
from typing import Any, Dict, List, Optional, Tuple


# ── Generalization Probe Tasks ─────────────────────────────────────────────────
# These are novel tasks not present in training and not copied from Step 6A.
# Used only as a generalization probe (Step 6D.24).

GENERALIZATION_PROBES = [
    {
        "id": "gen_probe_001",
        "instruction": "Write a Python function `is_palindrome(s: str) -> bool` that returns True if s is a palindrome.",
        "task_type": "code_generation",
        "expected_keywords": ["def is_palindrome", "return"],
    },
    {
        "id": "gen_probe_002",
        "instruction": "Write a Python function `fibonacci(n: int) -> list` that returns the first n Fibonacci numbers.",
        "task_type": "code_generation",
        "expected_keywords": ["def fibonacci", "return"],
    },
    {
        "id": "gen_probe_003",
        "instruction": "Produce a JSON object with the following structure: action is INSPECT_PROJECT, target is '.', reason is 'Inspecting project structure', risk is LOW.",
        "task_type": "structured_output",
        "expected_keywords": ["action", "INSPECT_PROJECT", "target"],
    },
    {
        "id": "gen_probe_004",
        "instruction": "Write a Python function `count_vowels(s: str) -> int` that counts vowels in string s. Include a docstring.",
        "task_type": "code_generation",
        "expected_keywords": ["def count_vowels", "docstring_check"],
    },
    {
        "id": "gen_probe_005",
        "instruction": "Write a Python class `Stack` with methods `push(item)`, `pop() -> any`, `peek() -> any`, `is_empty() -> bool`.",
        "task_type": "code_generation",
        "expected_keywords": ["class Stack", "def push", "def pop"],
    },
]

# ── Casual Conversation Probes ─────────────────────────────────────────────────

CASUAL_CONVERSATION_PROBES = [
    {"id": "casual_001", "instruction": "Hi, how are you?"},
    {"id": "casual_002", "instruction": "Hello! What can you help me with?"},
    {"id": "casual_003", "instruction": "Can you explain what recursion is?"},
    {"id": "casual_004", "instruction": "What's up?"},
    {"id": "casual_005", "instruction": "What are your capabilities?"},
    {"id": "casual_006", "instruction": "Give me a brief explanation of object-oriented programming."},
]


# ── Output Scoring Functions ───────────────────────────────────────────────────

def score_code_output(output: str, task_type: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """Score a code generation/modification output."""
    score = 0.0
    details = []

    if not output.strip():
        return {"score": 0.0, "details": ["Empty output"]}

    # Keyword presence (0.4)
    keyword_hits = 0
    for kw in expected_keywords:
        if kw == "docstring_check":
            if '"""' in output or "'''" in output:
                keyword_hits += 1
                details.append("Docstring present")
            else:
                details.append("Docstring MISSING")
        elif kw in output:
            keyword_hits += 1
        else:
            details.append(f"Expected keyword missing: {kw}")
    if expected_keywords:
        keyword_score = keyword_hits / len(expected_keywords) * 0.4
        score += keyword_score

    has_syntax_error = False
    if task_type in ("code_generation", "code_modification"):
        try:
            ast.parse(output)
            score += 0.4
            details.append("Python syntax: VALID")
        except SyntaxError as e:
            details.append(f"Python syntax error: {e}")
            has_syntax_error = True

    # Non-empty output (0.1)
    if len(output.strip()) > 10:
        score += 0.1

    # No markdown wrapper artifacts (0.1)
    if "```" not in output:
        score += 0.1
    else:
        details.append("Note: Output contains markdown code fences")

    if has_syntax_error:
        score = min(score, 0.4)

    return {"score": round(min(score, 1.0), 3), "details": details}


def score_json_output(output: str) -> Dict[str, Any]:
    """Score a structured JSON output."""
    if not output.strip():
        return {"score": 0.0, "details": ["Empty output"]}

    # Check for markdown wrapper
    if "```" in output:
        return {"score": 0.0, "details": ["Markdown wrapper detected — not valid raw JSON"]}

    # Check if it starts with conversation text
    if not output.strip().startswith("{"):
        return {"score": 0.0, "details": ["Output does not start with '{' — not pure JSON"]}

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as e:
        return {"score": 0.0, "details": [f"Invalid JSON: {e}"]}

    score = 0.5  # Valid JSON
    details = ["Valid JSON"]

    # Check for known action fields
    if "action" in parsed:
        score += 0.2
        details.append(f"action field present: {parsed['action']}")
    if "target" in parsed:
        score += 0.1
        details.append("target field present")
    if "reason" in parsed:
        score += 0.1
        details.append("reason field present")
    if "risk" in parsed:
        score += 0.1
        details.append("risk field present")

    return {"score": round(min(score, 1.0), 3), "details": details}


def score_conversational_output(output: str) -> Dict[str, Any]:
    """Score a casual conversation response for coherence."""
    if not output.strip():
        return {"score": 0.0, "coherent": False, "details": ["Empty response"]}

    score = 0.5 if len(output.strip()) > 10 else 0.0
    details = []

    # Check for coherent sentence structure (rough heuristic)
    has_sentence = bool(re.search(r"[A-Z][^.!?]*[.!?]", output))
    if has_sentence:
        score += 0.3
        details.append("Contains structured sentences")

    # Not code-only response
    if not output.strip().startswith("def ") and not output.strip().startswith("class "):
        score += 0.2
        details.append("Not a pure code response")

    return {
        "score": round(min(score, 1.0), 3),
        "coherent": score >= 0.5,
        "details": details,
    }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Step6CTestEvaluator:
    """Evaluates models on the Step 6C test split and generalization probes."""

    def evaluate_on_test_split(
        self,
        inference_fn,  # instruction -> {'output': str, ...}
        test_records: List[Dict],
        label: str = "model",
    ) -> Dict[str, Any]:
        """Evaluate model on the 5-example Step 6C test split.

        Args:
            inference_fn: Callable that takes instruction and returns dict with 'output'.
            test_records: The test split records from the 6C dataset.
            label: 'baseline' or 'fine_tuned' for labeling results.

        Returns:
            dict with per-example results and aggregate scores.
        """
        results = []
        total_score = 0.0

        for rec in test_records:
            from .dataset import _build_instruction_text
            instruction = _build_instruction_text(rec)

            try:
                response = inference_fn(instruction)
                output = response.get("output", "")
            except Exception as e:
                output = ""
                response = {"output": "", "error": str(e)}

            task_type = rec.get("task_type", "code_generation")
            expected_kw = rec.get("expected_keywords", [])
            target = rec.get("target_output", "")

            if task_type == "structured_output":
                score_info = score_json_output(output)
            else:
                # Derive expected keywords from target
                kws = []
                if "def " in target:
                    for m in re.finditer(r"def (\w+)", target):
                        kws.append(f"def {m.group(1)}")
                if "class " in target:
                    for m in re.finditer(r"class (\w+)", target):
                        kws.append(f"class {m.group(1)}")
                score_info = score_code_output(output, task_type, kws or expected_kw)

            results.append({
                "id": rec.get("id", "?"),
                "task_type": task_type,
                "instruction_preview": instruction[:100],
                "output_preview": output[:200],
                "score": score_info["score"],
                "details": score_info.get("details", []),
            })
            total_score += score_info["score"]

        avg_score = total_score / max(1, len(results))

        return {
            "label": label,
            "total_examples": len(results),
            "average_score": round(avg_score, 3),
            "results": results,
        }

    def run_generalization_probes(
        self,
        inference_fn,
        label: str = "model",
    ) -> Dict[str, Any]:
        """Run generalization probes (Step 6D.24)."""
        results = []
        total_score = 0.0

        for probe in GENERALIZATION_PROBES:
            try:
                response = inference_fn(probe["instruction"])
                output = response.get("output", "")
            except Exception as e:
                output = ""
                response = {"output": "", "error": str(e)}

            task_type = probe.get("task_type", "code_generation")
            expected_kw = probe.get("expected_keywords", [])

            if task_type == "structured_output":
                score_info = score_json_output(output)
            else:
                score_info = score_code_output(output, task_type, expected_kw)

            results.append({
                "id": probe["id"],
                "task_type": task_type,
                "instruction": probe["instruction"],
                "output_preview": output[:200],
                "score": score_info["score"],
                "details": score_info.get("details", []),
            })
            total_score += score_info["score"]

        avg_score = total_score / max(1, len(results))

        return {
            "label": label,
            "total_probes": len(results),
            "average_score": round(avg_score, 3),
            "results": results,
        }

    def run_casual_conversation(
        self,
        inference_fn,
        label: str = "model",
    ) -> Dict[str, Any]:
        """Run casual conversation smoke test (Step 6D.25)."""
        results = []
        coherent_count = 0

        for probe in CASUAL_CONVERSATION_PROBES:
            try:
                response = inference_fn(probe["instruction"])
                output = response.get("output", "")
            except Exception as e:
                output = ""
                response = {"output": "", "error": str(e)}

            score_info = score_conversational_output(output)

            results.append({
                "id": probe["id"],
                "instruction": probe["instruction"],
                "output_preview": output[:200],
                "coherent": score_info["coherent"],
                "score": score_info["score"],
            })

            if score_info["coherent"]:
                coherent_count += 1

        total = len(CASUAL_CONVERSATION_PROBES)
        return {
            "label": label,
            "total_probes": total,
            "coherent_count": coherent_count,
            "coherence_rate": round(coherent_count / max(1, total), 3),
            "verdict": "PASS" if coherent_count >= total * 0.5 else "FAIL",
            "results": results,
        }
