"""Training Example Validator and Quality Gatekeeper.

Performs exhaustive verification of training examples before admitting them
into the dataset. Rejects invalid syntax, schema violations, secrets, PII,
broken constraints, and low-quality samples.
"""

from typing import List, Optional, Tuple

from my_ai_agent.training.quality import (
    DEFAULT_QUALITY_THRESHOLD,
    scan_for_pii,
    scan_for_secrets,
    score_example_quality,
    validate_constraints,
    validate_python_code,
    validate_structured_json,
)
from my_ai_agent.training.schema import OutputType, TrainingExample


class TrainingExampleValidator:
    """Validates training examples against Step 6C dataset quality gates."""

    def __init__(self, quality_threshold: float = DEFAULT_QUALITY_THRESHOLD):
        self.quality_threshold = quality_threshold

    def validate(self, example: TrainingExample) -> Tuple[bool, List[str], float]:
        """Validates a single training example against all quality gates.

        Returns:
            (is_valid, list_of_errors, quality_score)
        """
        errors: List[str] = []

        # 1. Schema integrity checks
        if not example.id or not example.id.strip():
            errors.append("Example ID is required")
        if not example.instruction or not example.instruction.strip():
            errors.append("Instruction is required")
        if not example.task_type:
            errors.append("Task type is required")
        if not example.target_output or not example.target_output.strip():
            errors.append("Target output is required")

        # 2. Provenance checks
        if not example.provenance or not example.provenance.source_task_id:
            errors.append("Provenance source_task_id is required")

        # 3. Security and credential scanning
        combined_text = (
            f"{example.instruction}\n"
            f"{json_context(example.context)}\n"
            f"{example.target_output}"
        )
        secrets = scan_for_secrets(combined_text)
        if secrets:
            errors.append(f"Security violation: Secret detected: {', '.join(secrets)}")

        pii = scan_for_pii(combined_text)
        if pii:
            errors.append(f"Security violation: PII detected: {', '.join(pii)}")

        # 4. Target output type-specific validation
        if example.output_type in [OutputType.CODE.value, "code"]:
            valid, err = validate_python_code(example.target_output)
            if not valid:
                errors.append(f"Code validation failed: {err}")
        elif example.output_type in [OutputType.JSON.value, "json"]:
            valid, err = validate_structured_json(example.target_output)
            if not valid:
                errors.append(f"Structured JSON validation failed: {err}")
        elif example.output_type in [OutputType.DIFF.value, "diff"]:
            if not example.target_output.strip():
                errors.append("Diff target cannot be empty")

        # 5. Constraint validation
        constraints_ok, violations = validate_constraints(example)
        if not constraints_ok:
            errors.extend(violations)

        # 6. Quality score evaluation
        score = score_example_quality(example)
        if score < self.quality_threshold:
            errors.append(
                f"Quality score {score:.4f} is below configured threshold {self.quality_threshold}"
            )

        # Update example metadata
        is_valid = len(errors) == 0
        example.quality.validated = is_valid
        example.quality.score = score
        if not is_valid:
            example.quality.rejection_reason = "; ".join(errors)

        return is_valid, errors, score


def json_context(context: dict) -> str:
    """Helper to serialize context for secret scanning."""
    try:
        import json
        return json.dumps(context)
    except Exception:
        return str(context)
