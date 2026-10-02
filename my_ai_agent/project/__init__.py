"""Project Understanding Engine for Step 3B.

Provides deep, factual understanding of real user projects:
- Recursive file scanning with intelligent ignores
- File classification and sensitivity masking
- Project technology detection (Python, FastAPI, Node, React, Flutter, etc.)
- AST-based symbol and relationship extraction
- Dependency parsing
- Deterministic relevance scoring
- Compact context building for planning
"""

from my_ai_agent.project.file_classifier import FileClassification, FileClassifier
from my_ai_agent.project.scanner import ProjectScanner, ScannedFile
from my_ai_agent.project.dependency_analyzer import DependencyAnalyzer, DependencyInfo
from my_ai_agent.project.symbol_analyzer import SymbolAnalyzer, FileSymbols
from my_ai_agent.project.project_map import ProjectMap, ProjectMapCache
from my_ai_agent.project.analyzer import ProjectAnalyzer, ProjectTypeInfo, EntryPointInfo
from my_ai_agent.project.relevance import RelevanceEngine, RelevantFile
from my_ai_agent.project.context_builder import ContextBuilder, ProjectContext
from my_ai_agent.project.project_understanding import ProjectUnderstandingEngine

__all__ = [
    "FileClassification",
    "FileClassifier",
    "ProjectScanner",
    "ScannedFile",
    "DependencyAnalyzer",
    "DependencyInfo",
    "SymbolAnalyzer",
    "FileSymbols",
    "ProjectMap",
    "ProjectMapCache",
    "ProjectAnalyzer",
    "ProjectTypeInfo",
    "EntryPointInfo",
    "RelevanceEngine",
    "RelevantFile",
    "ContextBuilder",
    "ProjectContext",
    "ProjectUnderstandingEngine",
]
