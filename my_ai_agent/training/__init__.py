"""Training Dataset Construction and Quality Engineering Module for Step 6C.

Provides end-to-end tooling to generate, validate, deduplicate, partition,
and export high-quality instruction-tuning datasets from verified Step 6B weaknesses.
"""

from my_ai_agent.training.balance import DatasetBalanceAuditor
from my_ai_agent.training.deduplicator import DatasetDeduplicator
from my_ai_agent.training.exporter import DatasetExporter
from my_ai_agent.training.generator import TrainingDatasetGenerator
from my_ai_agent.training.leakage import BenchmarkLeakageDetector
from my_ai_agent.training.provenance import ExampleProvenance
from my_ai_agent.training.quality import (
    DEFAULT_QUALITY_THRESHOLD,
    scan_for_pii,
    scan_for_secrets,
    score_example_quality,
    validate_constraints,
    validate_python_code,
    validate_structured_json,
)
from my_ai_agent.training.schema import (
    OutputType,
    QualityMetadata,
    TaskType,
    TrainingExample,
)
from my_ai_agent.training.splitter import DatasetSplitter
from my_ai_agent.training.statistics import DatasetStatisticsEngine
from my_ai_agent.training.validator import TrainingExampleValidator

__all__ = [
    "DatasetBalanceAuditor",
    "DatasetDeduplicator",
    "DatasetExporter",
    "DatasetSplitter",
    "DatasetStatisticsEngine",
    "ExampleProvenance",
    "TrainingDatasetGenerator",
    "TrainingExample",
    "TrainingExampleValidator",
    "BenchmarkLeakageDetector",
    "QualityMetadata",
    "OutputType",
    "TaskType",
    "DEFAULT_QUALITY_THRESHOLD",
    "scan_for_secrets",
    "scan_for_pii",
    "score_example_quality",
    "validate_python_code",
    "validate_structured_json",
    "validate_constraints",
]
