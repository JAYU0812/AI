"""Main pipeline entrypoint for Step 6C dataset construction.

Orchestrates generation, quality validation, deduplication, benchmark
leakage detection, source-grouped splitting, statistical aggregation,
export of JSONL datasets, and generation of the dataset quality report.
"""

from typing import Any, Dict, List, Tuple

from my_ai_agent.training.deduplicator import DatasetDeduplicator
from my_ai_agent.training.exporter import DatasetExporter
from my_ai_agent.training.generator import TrainingDatasetGenerator
from my_ai_agent.training.leakage import BenchmarkLeakageDetector
from my_ai_agent.training.schema import QualityMetadata, TrainingExample
from my_ai_agent.training.splitter import DatasetSplitter
from my_ai_agent.training.statistics import DatasetStatisticsEngine
from my_ai_agent.training.validator import TrainingExampleValidator


def build_and_export_dataset() -> Dict[str, Any]:
    """Executes the complete Step 6C dataset construction workflow."""
    print("=== Step 6C: Training Dataset Construction ===")

    # 1. Initialize components
    generator = TrainingDatasetGenerator()
    validator = TrainingExampleValidator()
    deduplicator = DatasetDeduplicator()
    leakage_detector = BenchmarkLeakageDetector()
    splitter = DatasetSplitter()
    exporter = DatasetExporter()

    # 2. Generate raw candidate examples
    print("Generating candidate examples from verified Step 6B weaknesses...")
    raw_candidates = generator.generate_all()
    print(f"Generated {len(raw_candidates)} candidate examples.")

    accepted_examples: List[TrainingExample] = []
    rejected_examples: List[TrainingExample] = []

    # 3. Filter through Quality Gates, Deduplication, and Benchmark Leakage
    print("Executing quality gates, deduplication, and leakage filtering...")
    for ex in raw_candidates:
        # Benchmark Leakage Guard
        has_leakage, leak_reason = leakage_detector.check_example_leakage(ex)
        if has_leakage:
            ex.quality.leakage = True
            ex.quality.rejection_reason = f"Leakage: {leak_reason}"
            rejected_examples.append(ex)
            continue

        # Deduplication Guard
        is_dup, dup_reason = deduplicator.is_duplicate(ex)
        if is_dup:
            ex.quality.duplicate = True
            ex.quality.rejection_reason = f"Duplicate: {dup_reason}"
            rejected_examples.append(ex)
            continue

        # Quality & Syntax/Schema Validation Gate
        is_valid, errors, score = validator.validate(ex)
        if not is_valid:
            rejected_examples.append(ex)
            continue

        # Passed all gates
        deduplicator.register(ex)
        accepted_examples.append(ex)

    print(f"Accepted: {len(accepted_examples)}, Rejected: {len(rejected_examples)}")

    # 4. Partition into Train, Validation, and Test splits
    print("Partitioning into leakage-free source-task grouped splits...")
    train_ex, val_ex, test_ex = splitter.split(accepted_examples)
    print(f"Splits -> Train: {len(train_ex)}, Validation: {len(val_ex)}, Test: {len(test_ex)}")

    # 5. Verify Split Isolation
    is_isolated, violations = BenchmarkLeakageDetector.verify_split_isolation(
        train_ex, val_ex, test_ex
    )
    if not is_isolated:
        raise RuntimeError(f"Split isolation failed! Violations: {violations}")
    print("Cross-split isolation verified: Zero source tasks or instructions cross partitions.")

    # 6. Compute Comprehensive Statistics
    stats = DatasetStatisticsEngine.compute_statistics(
        all_candidates=raw_candidates,
        accepted_examples=accepted_examples,
        train_examples=train_ex,
        val_examples=val_ex,
        test_examples=test_ex,
        rejected_examples=rejected_examples,
    )

    # 7. Export JSONL, Statistics, and Manifest
    print("Exporting JSONL partitions, statistics, and cryptographic manifest...")
    manifest = exporter.export_all(
        train_examples=train_ex,
        val_examples=val_ex,
        test_examples=test_ex,
        rejected_examples=rejected_examples,
        statistics=stats,
    )

    # 8. Generate Quality Report Markdown
    report_file = exporter.generate_quality_report(stats, manifest)
    print(f"Generated quality report at: {report_file}")

    print("=== Step 6C Dataset Construction Completed Successfully ===")
    return {
        "manifest": manifest,
        "statistics": stats,
        "report_path": str(report_file),
    }


if __name__ == "__main__":
    build_and_export_dataset()
