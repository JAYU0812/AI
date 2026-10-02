"""Precise Code Editing & Patch Engine for Step 3C.

Provides symbol-aware, transactional, multi-file patching with automatic
import management, syntax validation, diff generation, and rollback.
"""

from my_ai_agent.editing.edit_operations import (
    EditOperation,
    EditOperationType,
    EditProposal,
    FileEditProposal,
)
from my_ai_agent.editing.code_locator import CodeLocator, LocationResult
from my_ai_agent.editing.patch_validator import PatchValidator, ValidationResult
from my_ai_agent.editing.diff_generator import DiffGenerator, DiffResult
from my_ai_agent.editing.patch_engine import PatchEngine, PatchResult, TransactionResult
from my_ai_agent.editing.test_runner import ProjectTestRunner, TestExecutionResult, AutoFixResult

__all__ = [
    "EditOperation",
    "EditOperationType",
    "EditProposal",
    "FileEditProposal",
    "CodeLocator",
    "LocationResult",
    "PatchValidator",
    "ValidationResult",
    "DiffGenerator",
    "DiffResult",
    "PatchEngine",
    "PatchResult",
    "TransactionResult",
    "ProjectTestRunner",
    "TestExecutionResult",
    "AutoFixResult",
]
