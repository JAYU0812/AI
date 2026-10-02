"""Model-to-Action Safety Boundary for Step 4.

CRITICAL ARCHITECTURAL BOUNDARY:
The model is an advisory decision component. It is NEVER the execution boundary.
No model output is permitted to directly invoke shell, subprocess, filesystem,
database, or network. All model decisions undergo strict path validation,
sensitive file protection, destructive command interception, and permission checks.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from my_ai_agent.autonomous.action import ActionRisk, ActionType, AutonomousAction
from my_ai_agent.intelligence.structured_output import SchemaType, StructuredOutputValidator


class SafetyViolation(Exception):
    """Raised when model output violates security or safety policies."""
    pass


class ModelSafetyBoundary:
    """Enforces strict security boundaries between model outputs and agent actions."""

    # Intercept destructive system commands
    DESTRUCTIVE_COMMAND_PATTERNS = [
        r"\brm\s+-rf\s+[/~]",
        r"\brmdir\s+/[sS]\s+/[qQ]\s+[cC]:\\",
        r"\bdel\s+/[fF]\s+/[sS]\s+/[qQ]\s+[cC]:\\",
        r"\bmkfs\b",
        r"\bformat\s+[cC]:",
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # Fork bomb
        r"\bdrop\s+database\b",
        r"\btruncate\s+table\b",
        r"\bdelete\s+everything\b",
    ]

    # Sensitive files that can never be read, modified, or deleted by model
    SENSITIVE_PATTERNS = [
        r"^\.env(\..+)?$",
        r".*\.key$",
        r".*\.pem$",
        r".*id_rsa.*",
        r".*secrets?\.(json|yaml|yml)$",
        r".*credentials?\.(json|yaml|yml)$",
    ]

    # Dangerous system directories
    SYSTEM_DIR_PATTERNS = [
        r"^[a-zA-Z]:\\windows",
        r"^[a-zA-Z]:\\program files",
        r"^[a-zA-Z]:\\system32",
        r"^/etc/",
        r"^/usr/",
        r"^/bin/",
        r"^/sbin/",
        r"^/var/run",
    ]

    def __init__(self, workspace_root: Path | str):
        self.workspace_root = Path(workspace_root).resolve()

    def sanitize_and_convert(
        self,
        raw_decision: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[AutonomousAction]]:
        """Validates a raw model action decision and converts it to a safe AutonomousAction.
        
        Returns:
            (is_safe, violation_reason, action)
        """
        # 1. Schema Validation
        valid, err, validated_data = StructuredOutputValidator.validate(
            raw_decision, SchemaType.ACTION_DECISION
        )
        if not valid or not validated_data:
            return False, f"Schema validation failed: {err}", None

        action_str = validated_data["action"]
        target = validated_data["target"]
        reason = validated_data.get("reason", "")
        declared_risk = validated_data.get("risk", "LOW").upper()
        params = validated_data.get("params", {})
        if "operations" in validated_data:
            params["operations"] = validated_data["operations"]

        try:
            action_type = ActionType(action_str)
        except ValueError:
            return False, f"Invalid action type: '{action_str}'", None

        # 2. Path Traversal & System Directory Check
        if action_type in (
            ActionType.READ_FILE, ActionType.CREATE_FILE, ActionType.PATCH_FILE,
            ActionType.DELETE_FILE, ActionType.SEARCH_CODE,
        ):
            path_safe, path_err = self._validate_file_path(target)
            if not path_safe:
                return False, f"Path security violation: {path_err}", None

        # 3. Sensitive File Check
        target_name = Path(target).name
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, target_name, re.IGNORECASE):
                return False, f"Access to sensitive file '{target}' is strictly forbidden", None

        # 4. Command Safety Check
        if action_type == ActionType.RUN_COMMAND:
            cmd = target
            cmd_safe, cmd_err, requires_approval = self._validate_command(cmd)
            if not cmd_safe:
                return False, f"Command security violation: {cmd_err}", None
            if requires_approval:
                declared_risk = "HIGH"

        # 5. Determine Final Risk and Approval Requirements
        risk = ActionRisk.LOW
        if declared_risk == "HIGH" or action_type == ActionType.DELETE_FILE:
            risk = ActionRisk.HIGH
        elif declared_risk == "MEDIUM" or action_type in (ActionType.INSTALL_DEPENDENCY, ActionType.RUN_COMMAND):
            risk = ActionRisk.MEDIUM

        action = AutonomousAction(
            type=action_type,
            target=target,
            reason=reason,
            risk=risk,
            params=params,
            requires_approval=(risk == ActionRisk.HIGH),
        )

        return True, None, action

    def _validate_file_path(self, target_path_str: str) -> Tuple[bool, Optional[str]]:
        """Ensures file paths are contained within workspace and not targeting system roots."""
        # Check system directory patterns
        norm_str = target_path_str.replace("/", "\\")
        for sys_pat in self.SYSTEM_DIR_PATTERNS:
            if re.search(sys_pat, norm_str, re.IGNORECASE) or re.search(sys_pat, target_path_str, re.IGNORECASE):
                return False, f"Path '{target_path_str}' targets a protected system directory."

        # Check path traversal
        try:
            target_p = Path(target_path_str)
            if target_p.is_absolute():
                resolved = target_p.resolve()
            else:
                resolved = (self.workspace_root / target_p).resolve()

            # Must be inside workspace
            try:
                resolved.relative_to(self.workspace_root)
            except ValueError:
                return False, f"Path '{target_path_str}' traverses outside the project workspace."

            return True, None
        except Exception as e:
            return False, f"Invalid path syntax: {e}"

    def _validate_command(self, cmd_str: str) -> Tuple[bool, Optional[str], bool]:
        """Validates command string for destructive or unauthorized operations.
        
        Returns:
            (is_safe, error_message, requires_approval)
        """
        for pat in self.DESTRUCTIVE_COMMAND_PATTERNS:
            if re.search(pat, cmd_str, re.IGNORECASE):
                return False, f"Destructive command pattern matched: '{pat}'", False

        # Commands that alter environment require user approval
        needs_approval = False
        if any(w in cmd_str.lower() for w in ["pip install", "npm install", "cargo add", "migrate", "drop table"]):
            needs_approval = True

        return True, None, needs_approval
