"""Structured Output Validation and Schema Definitions for Step 4.

Ensures any model decision strictly conforms to validated schemas before
the autonomous agent or tools can execute it. NEVER trusts raw model output.
"""

from enum import Enum
import json
from typing import Any, Dict, List, Optional, Tuple


class SchemaType(str, Enum):
    """Supported structured output schema categories."""
    ACTION_DECISION = "action_decision"
    PLAN = "plan"
    EDIT_PROPOSAL = "edit_proposal"
    DIAGNOSIS = "diagnosis"
    REPAIR_PROPOSAL = "repair_proposal"


# ── Canonical Schemas ─────────────────────────────────────────────

ACTION_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["action", "target", "reason", "risk"],
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "INSPECT_PROJECT", "READ_FILE", "SEARCH_CODE", "CREATE_FILE",
                "PATCH_FILE", "DELETE_FILE", "RUN_COMMAND", "RUN_TESTS",
                "INSTALL_DEPENDENCY", "VERIFY", "ASK_USER", "FINISH",
            ],
        },
        "target": {"type": "string"},
        "reason": {"type": "string"},
        "risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "params": {"type": "object"},
        "operations": {"type": "array"},
    },
}

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["goal", "steps"],
    "properties": {
        "goal": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description", "action"],
                "properties": {
                    "description": {"type": "string"},
                    "action": {"type": "string"},
                    "target": {"type": "string"},
                    "reason": {"type": "string"},
                    "risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                },
            },
        },
    },
}

EDIT_PROPOSAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["intent", "files"],
    "properties": {
        "intent": {"type": "string"},
        "description": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "operations"],
                "properties": {
                    "path": {"type": "string"},
                    "operations": {"type": "array"},
                },
            },
        },
    },
}

DIAGNOSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["category", "error_message"],
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "SYNTAX_ERROR", "IMPORT_ERROR", "TYPE_ERROR", "RUNTIME_ERROR",
                "TEST_FAILURE", "CONFIGURATION_ERROR", "DEPENDENCY_ERROR",
                "FILE_NOT_FOUND", "PERMISSION_ERROR", "NETWORK_ERROR", "UNKNOWN_ERROR",
            ],
        },
        "error_message": {"type": "string"},
        "affected_file": {"type": ["string", "null"]},
        "affected_line": {"type": ["integer", "null"]},
        "failing_symbol": {"type": ["string", "null"]},
        "probable_cause": {"type": "string"},
        "evidence": {"type": "string"},
    },
}


class StructuredOutputValidator:
    """Validates structured model outputs against canonical schemas."""

    SCHEMAS: Dict[str, Dict[str, Any]] = {
        SchemaType.ACTION_DECISION.value: ACTION_DECISION_SCHEMA,
        SchemaType.PLAN.value: PLAN_SCHEMA,
        SchemaType.EDIT_PROPOSAL.value: EDIT_PROPOSAL_SCHEMA,
        SchemaType.DIAGNOSIS.value: DIAGNOSIS_SCHEMA,
    }

    @classmethod
    def validate(
        cls,
        data: Any,
        schema_type: SchemaType | str,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validates payload against schema.
        
        Returns:
            (is_valid, error_message, validated_dict)
        """
        if isinstance(schema_type, SchemaType):
            st_key = schema_type.value
        else:
            st_key = str(schema_type).lower()

        schema = cls.SCHEMAS.get(st_key)
        if not schema:
            return False, f"Unknown schema type: '{schema_type}'", None

        # Parse string JSON if needed
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception as e:
                return False, f"Invalid JSON string: {e}", None

        if not isinstance(data, dict):
            return False, f"Payload must be a dict/object, got {type(data).__name__}", None

        # Validate required fields
        required_fields: List[str] = schema.get("required", [])
        for req in required_fields:
            if req not in data:
                return False, f"Missing required property: '{req}'", None

        # Validate property constraints
        props: Dict[str, Any] = schema.get("properties", {})
        for prop_name, prop_spec in props.items():
            if prop_name not in data:
                continue

            val = data[prop_name]
            expected_type = prop_spec.get("type")

            # Check type
            if expected_type == "string" and not isinstance(val, str):
                return False, f"Property '{prop_name}' must be string, got {type(val).__name__}", None
            elif expected_type == "integer" and not isinstance(val, int) and val is not None:
                return False, f"Property '{prop_name}' must be integer, got {type(val).__name__}", None
            elif expected_type == "array" and not isinstance(val, list):
                return False, f"Property '{prop_name}' must be array/list, got {type(val).__name__}", None
            elif expected_type == "object" and not isinstance(val, dict):
                return False, f"Property '{prop_name}' must be object/dict, got {type(val).__name__}", None

            # Check enum
            allowed_enums = prop_spec.get("enum")
            if allowed_enums and val not in allowed_enums:
                return False, f"Property '{prop_name}' has invalid value '{val}'. Allowed: {allowed_enums}", None

        # Additional specific schema validations
        if st_key == SchemaType.ACTION_DECISION.value:
            # Action must have non-empty target
            if not data.get("target") and data.get("action") not in ("FINISH", "VERIFY"):
                return False, "Action target cannot be empty", None

        elif st_key == SchemaType.PLAN.value:
            steps = data.get("steps", [])
            if not isinstance(steps, list) or len(steps) == 0:
                return False, "Plan must contain at least one step", None
            for idx, s in enumerate(steps):
                if not isinstance(s, dict) or "description" not in s or "action" not in s:
                    return False, f"Step {idx} missing description or action", None

        return True, None, data
