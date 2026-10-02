"""Quality Assessment, Security Scanning, and Code/JSON Target Validation.

Implements rigorous gate checks:
- Secret & credential scanning (.env, API keys, private keys, database URLs, JWTs)
- PII scanning (emails, SSNs, credit cards)
- Python AST syntax validation
- Strict structured JSON schema validation (using Agent canonical schemas)
- Instruction constraint verification
- Quality scoring algorithm (0.0 to 1.0)
"""

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from my_ai_agent.intelligence.structured_output import SchemaType, StructuredOutputValidator
from my_ai_agent.training.schema import OutputType, TaskType, TrainingExample

DEFAULT_QUALITY_THRESHOLD = 0.85

# ── Secret Scanning Patterns ─────────────────────────────────────────

SECRET_PATTERNS = [
    (r"-----BEGIN [A-Z ]+KEY-----", "Private cryptographic key"),
    (r"(?:api[_-]?key|apikey|secret[_-]?key)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]", "API or Secret Key assignment"),
    (r"Bearer\s+[A-Za-z0-9_\-\.]{25,}", "Bearer Token"),
    (r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}", "JSON Web Token (JWT)"),
    (r"(?:postgres|postgresql|mysql|mongodb|redis):\/\/[A-Za-z0-9_\-]+:[^@\s]+@[A-Za-z0-9_\-\.]+:\d+", "Database Connection URI with credentials"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]", "Hardcoded password assignment"),
    (r"(?:^|\n)[A-Z0-9_]+=\s*['\"]?[A-Za-z0-9_\-]{20,}['\"]?(?:\n|$)", "Environment credential definition (.env style)"),
]

# ── PII Patterns ─────────────────────────────────────────────────────

PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "Email address"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "Social Security Number (SSN)"),
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", "Credit Card Number"),
]


def scan_for_secrets(text: str) -> List[str]:
    """Scans text for any hardcoded credentials, keys, or secret patterns."""
    if not text:
        return []
    findings: List[str] = []
    for pattern, desc in SECRET_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            findings.append(desc)
    return findings


def scan_for_pii(text: str) -> List[str]:
    """Scans text for personally identifiable information."""
    if not text:
        return []
    findings: List[str] = []
    for pattern, desc in PII_PATTERNS:
        # Ignore dummy test emails like user@example.com
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            if isinstance(m, str) and ("example.com" in m.lower() or "test.com" in m.lower()):
                continue
            findings.append(desc)
    return findings


def validate_python_code(code: str) -> Tuple[bool, Optional[str]]:
    """Validates that python code parses into a valid AST without syntax errors."""
    if not code or not code.strip():
        return False, "Code target is empty"
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def validate_structured_json(json_str: str, expected_schema: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Validates that text is pure, valid JSON conforming to an Agent schema."""
    if not json_str or not json_str.strip():
        return False, "Structured JSON target is empty"

    stripped = json_str.strip()

    # Detect conversational preamble/markdown wrapping
    if stripped.startswith("```") or stripped.endswith("```"):
        return False, "Target contains markdown code block wrappers; pure JSON required"
    if stripped.startswith("Sure") or stripped.startswith("Here is") or "I have" in stripped[:30]:
        return False, "Target contains conversational prose; pure JSON required"

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON syntax: {e.msg}"

    if not isinstance(data, dict):
        return False, f"Expected JSON object, got {type(data).__name__}"

    # Validate against agent schema if specified
    if expected_schema:
        valid, err, _ = StructuredOutputValidator.validate(data, expected_schema)
        if not valid:
            return False, f"Agent schema '{expected_schema}' validation failed: {err}"
    else:
        # Check against common known schemas: action_decision, plan, edit_proposal, diagnosis
        validated = False
        last_err = None
        for st in [SchemaType.ACTION_DECISION, SchemaType.PLAN, SchemaType.EDIT_PROPOSAL, SchemaType.DIAGNOSIS]:
            v, err, _ = StructuredOutputValidator.validate(data, st)
            if v:
                validated = True
                break
            last_err = err
        if not validated:
            return False, f"Structured output did not match any canonical agent schema. Last error: {last_err}"

    return True, None


def validate_constraints(example: TrainingExample) -> Tuple[bool, List[str]]:
    """Validates that the target output respects explicit instruction constraints."""
    violations: List[str] = []
    target = example.target_output.lower()

    for constraint in example.constraints:
        c_lower = constraint.lower()

        # Check single-file constraint
        if "only" in c_lower and "file" in c_lower:
            # Must not mention modifying external files in output
            pass

        # Check negative constraint: do not install / no external dependencies
        if "do not install" in c_lower or "no dependencies" in c_lower or "no external" in c_lower:
            if "pip install" in target or "subprocess" in target and "pip" in target:
                violations.append(f"Constraint violated: '{constraint}' but target executes pip install")

        # Check docstring requirement
        if "docstring" in c_lower and ("add" in c_lower or "require" in c_lower):
            if '"""' not in example.target_output and "'''" not in example.target_output:
                violations.append(f"Constraint violated: '{constraint}' but no docstring found in target code")

    return len(violations) == 0, violations


def score_example_quality(example: TrainingExample) -> float:
    """Calculates overall quality score between 0.0 and 1.0.

    Evaluation weights:
    - Target correctness (AST / JSON schema): 0.35
    - Security / Secret clean: 0.15
    - Constraint adherence: 0.20
    - Structure completeness (docstring/typing/context): 0.15
    - Verification defined: 0.10
    - Provenance completeness: 0.05
    """
    score = 0.0

    # 1. Target correctness (0.35)
    if example.output_type in [OutputType.CODE.value, "code"]:
        valid, _ = validate_python_code(example.target_output)
        if valid:
            score += 0.35
    elif example.output_type in [OutputType.JSON.value, "json"]:
        valid, _ = validate_structured_json(example.target_output)
        if valid:
            score += 0.35
    elif example.output_type in [OutputType.DIFF.value, "diff"]:
        # Unified diff format validation
        if "@@" in example.target_output or "---" in example.target_output or "+++" in example.target_output:
            score += 0.35
        elif example.target_output.strip():
            score += 0.30

    # 2. Security clean (0.15)
    full_text = f"{example.instruction}\n{example.target_output}"
    secrets = scan_for_secrets(full_text)
    pii = scan_for_pii(full_text)
    if not secrets and not pii:
        score += 0.15

    # 3. Constraint adherence (0.20)
    constraints_ok, _ = validate_constraints(example)
    if constraints_ok:
        score += 0.20

    # 4. Structure completeness (0.15)
    completeness = 0.0
    if example.context and example.context.get("project_type"):
        completeness += 0.05
    if example.expected_behavior:
        completeness += 0.05
    if example.output_type == "code":
        # Check type annotations or docstrings
        if "->" in example.target_output or '"""' in example.target_output:
            completeness += 0.05
    else:
        completeness += 0.05
    score += completeness

    # 5. Verification tests defined (0.10)
    if example.verification and len(example.verification) > 0:
        score += 0.10

    # 6. Provenance completeness (0.05)
    if (
        example.provenance
        and example.provenance.source_task_id
        and example.provenance.source_root_cause
        and example.provenance.source_step
    ):
        score += 0.05

    return round(score, 4)
