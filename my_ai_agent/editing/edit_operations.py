"""Edit Operations Module for Step 3C.

Defines structured, safe code editing operations rather than crude whole-file rewriting.
Supports INSERT, REPLACE, DELETE, APPEND, PREPEND, and RENAME_SYMBOL.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EditOperationType(str, Enum):
    INSERT = "INSERT"
    REPLACE = "REPLACE"
    DELETE = "DELETE"
    APPEND = "APPEND"
    PREPEND = "PREPEND"
    RENAME_SYMBOL = "RENAME_SYMBOL"


@dataclass
class EditOperation:
    """A precise, atomic modification operation on a single file."""
    operation_type: EditOperationType
    target_file: str                            # Relative path (e.g. "auth.py")
    content: str = ""                           # Code to insert or replace with
    target_symbol: Optional[str] = None         # Symbol name (e.g. "AuthService", "login")
    target_text: Optional[str] = None           # Exact text target to match
    line_number: Optional[int] = None           # 1-indexed target line number
    position: str = "after"                     # "before", "after", "inside", "replace"
    required_imports: List[str] = field(default_factory=list)  # e.g. ["import uuid"]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_type": self.operation_type.value,
            "target_file": self.target_file,
            "target_symbol": self.target_symbol,
            "target_text": self.target_text,
            "line_number": self.line_number,
            "position": self.position,
            "content": self.content,
            "required_imports": self.required_imports,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditOperation":
        op_type = EditOperationType(data["operation_type"])
        return cls(
            operation_type=op_type,
            target_file=data["target_file"],
            content=data.get("content", ""),
            target_symbol=data.get("target_symbol"),
            target_text=data.get("target_text"),
            line_number=data.get("line_number"),
            position=data.get("position", "after"),
            required_imports=data.get("required_imports", []),
            description=data.get("description", ""),
        )


@dataclass
class FileEditProposal:
    """Group of operations planned for a single file."""
    path: str
    operations: List[EditOperation] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "description": self.description,
            "operations": [op.to_dict() for op in self.operations],
        }


@dataclass
class EditProposal:
    """High-level structured proposal produced by synthesizer or planner."""
    intent: str                                 # e.g. "ADD_METHOD", "ADD_FUNCTION", "MODIFY_METHOD"
    description: str
    files: List[FileEditProposal] = field(default_factory=list)
    requires_approval: bool = False
    approval_reason: str = ""

    @property
    def total_operations_count(self) -> int:
        return sum(len(f.operations) for f in self.files)

    @property
    def affected_files(self) -> List[str]:
        return [f.path for f in self.files]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "description": self.description,
            "requires_approval": self.requires_approval,
            "approval_reason": self.approval_reason,
            "total_operations": self.total_operations_count,
            "affected_files": self.affected_files,
            "files": [f.to_dict() for f in self.files],
        }
