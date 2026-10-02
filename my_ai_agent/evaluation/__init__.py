"""Evaluation and Benchmark Package (Step 6A).

Provides rigorous benchmark datasets, deterministic assertions, multi-dimensional
scoring, isolated sandbox execution, and regression tracking.
"""

from my_ai_agent.evaluation.assertions import (
    BaseAssertion,
    CommandFailsAssertion,
    CommandSucceedsAssertion,
    DiffWithinExpectedScopeAssertion,
    FileContainsAssertion,
    FileDoesNotContainAssertion,
    FileExistsAssertion,
    FileNotExistsAssertion,
    NoProtectedFileChangedAssertion,
    NoUnsafeCommandExecutedAssertion,
    OutputContainsAssertion,
    StructuredOutputValidAssertion,
    SymbolCountAssertion,
    SymbolExistsAssertion,
    SyntaxValidAssertion,
    TaskCompletedAssertion,
    TaskFailedSafelyAssertion,
    TestsPassAssertion,
    build_assertion,
)
from my_ai_agent.evaluation.baseline import BaselineManager
from my_ai_agent.evaluation.benchmark import run_benchmark_suite
from my_ai_agent.evaluation.categories import (
    DatasetSplit,
    Difficulty,
    EvaluationDimension,
    FailureType,
    TaskCategory,
)
from my_ai_agent.evaluation.dataset import BenchmarkDataset, DatasetQualityAuditor
from my_ai_agent.evaluation.evaluator import TaskEvaluationResult, TaskEvaluator
from my_ai_agent.evaluation.metrics import TaskMetrics
from my_ai_agent.evaluation.report import BenchmarkReportGenerator
from my_ai_agent.evaluation.runner import BenchmarkRunner
from my_ai_agent.evaluation.safety_eval import SafetyEvaluationResult, SafetyEvaluator
from my_ai_agent.evaluation.scoring import EvaluationScorecard, ScoringEngine
from my_ai_agent.evaluation.task import EvaluationTask

__all__ = [
    "EvaluationTask",
    "TaskCategory",
    "Difficulty",
    "DatasetSplit",
    "EvaluationDimension",
    "FailureType",
    "BaseAssertion",
    "FileExistsAssertion",
    "FileNotExistsAssertion",
    "FileContainsAssertion",
    "FileDoesNotContainAssertion",
    "SymbolExistsAssertion",
    "SymbolCountAssertion",
    "SyntaxValidAssertion",
    "TestsPassAssertion",
    "CommandSucceedsAssertion",
    "CommandFailsAssertion",
    "OutputContainsAssertion",
    "DiffWithinExpectedScopeAssertion",
    "NoProtectedFileChangedAssertion",
    "NoUnsafeCommandExecutedAssertion",
    "StructuredOutputValidAssertion",
    "TaskCompletedAssertion",
    "TaskFailedSafelyAssertion",
    "build_assertion",
    "EvaluationScorecard",
    "ScoringEngine",
    "TaskMetrics",
    "SafetyEvaluator",
    "SafetyEvaluationResult",
    "BenchmarkDataset",
    "DatasetQualityAuditor",
    "TaskEvaluator",
    "TaskEvaluationResult",
    "BenchmarkRunner",
    "BenchmarkReportGenerator",
    "BaselineManager",
    "run_benchmark_suite",
]
