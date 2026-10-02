"""Evaluation Categories, Difficulty Levels, Dataset Splits, and Failure Taxonomy.

Defines the centralized taxonomy of evaluation dimensions, splits, and error types
used across all benchmarks and future fine-tuning datasets.
"""

from enum import Enum
from typing import List


class TaskCategory(str, Enum):
    """Benchmark capability categories."""
    GENERAL_UNDERSTANDING = "general_understanding"
    INTENT_UNDERSTANDING = "intent_understanding"
    PROJECT_UNDERSTANDING = "project_understanding"
    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    DEBUGGING = "debugging"
    MULTI_STEP = "multi_step"
    TOOL_SELECTION = "tool_selection"
    STRUCTURED_OUTPUT = "structured_output"
    CONTEXT_USAGE = "context_usage"
    INSTRUCTION_FOLLOWING = "instruction_following"
    VERIFICATION = "verification"
    SAFETY = "safety"
    AUTONOMOUS_COMPLETION = "autonomous_completion"

    @classmethod
    def all_categories(cls) -> List[str]:
        return [c.value for c in cls]


class Difficulty(str, Enum):
    """Task difficulty rating based on objective complexity."""
    EASY = "easy"        # Single file, single operation, direct context, no debugging
    MEDIUM = "medium"    # Multi-file, modification of existing code, tests, moderate reasoning
    HARD = "hard"        # Ambiguous context, debugging, dependency interaction, multi-step
    EXPERT = "expert"    # Full architecture design, complex multi-component repair, strict constraints


class DatasetSplit(str, Enum):
    """Dataset partition to avoid benchmark leakage and overfitting."""
    DEVELOPMENT = "development"  # Iteration & evaluator development
    VALIDATION = "validation"    # Hyperparameter & prompt tuning
    TEST = "test"                # Held-out benchmark (never optimized against)


class EvaluationDimension(str, Enum):
    """Multi-dimensional scoring dimensions."""
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    EXECUTION = "execution"
    CODE_CORRECTNESS = "code_correctness"
    MODIFICATION_CORRECTNESS = "modification_correctness"
    DEBUGGING = "debugging"
    TOOL_SELECTION = "tool_selection"
    CONTEXT_USAGE = "context_usage"
    INSTRUCTION_FOLLOWING = "instruction_following"
    VERIFICATION = "verification"
    SAFETY = "safety"
    TASK_COMPLETION = "task_completion"


class FailureType(str, Enum):
    """Centralized Error Taxonomy for Failure Analysis."""
    NONE = "none"
    MISUNDERSTOOD_REQUEST = "misunderstood_request"
    WRONG_FILE = "wrong_file"
    WRONG_SYMBOL = "wrong_symbol"
    BAD_PLAN = "bad_plan"
    BAD_CODE = "bad_code"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TEST_FAILURE = "test_failure"
    WRONG_TOOL = "wrong_tool"
    CONTEXT_FAILURE = "context_failure"
    STRUCTURED_OUTPUT_FAILURE = "structured_output_failure"
    VERIFICATION_FAILURE = "verification_failure"
    SAFETY_REJECTION = "safety_rejection"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
