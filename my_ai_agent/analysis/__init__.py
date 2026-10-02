"""Baseline Weakness Analysis and Root-Cause Engineering Package (Step 6B).

Provides automated diagnostic root-cause determination, failure clustering,
ablation experimentation, and priority remediation planning.
"""

from my_ai_agent.analysis.baseline_analysis import BaselineAnalysisPipeline
from my_ai_agent.analysis.category_analyzer import CategoryAnalyzer
from my_ai_agent.analysis.evidence import EvidenceExtractor
from my_ai_agent.analysis.experiments import AblationExperimentRunner
from my_ai_agent.analysis.failure_analyzer import TaskFailureAnalyzer
from my_ai_agent.analysis.pattern_detector import PatternDetector
from my_ai_agent.analysis.recommendations import RecommendationEngine
from my_ai_agent.analysis.report import AnalysisReportGenerator
from my_ai_agent.analysis.root_cause import (
    ConfidenceLevel,
    FailureCluster,
    FixType,
    RootCauseCategory,
    RootCauseRecord,
)

__all__ = [
    "RootCauseCategory",
    "ConfidenceLevel",
    "FixType",
    "RootCauseRecord",
    "FailureCluster",
    "EvidenceExtractor",
    "TaskFailureAnalyzer",
    "CategoryAnalyzer",
    "PatternDetector",
    "RecommendationEngine",
    "AblationExperimentRunner",
    "AnalysisReportGenerator",
    "BaselineAnalysisPipeline",
]
