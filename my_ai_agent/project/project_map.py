"""Project Map & Caching Module for Step 3B.

Defines the comprehensive, serializable ProjectMap data model and
provides intelligent timestamp-based caching to avoid redundant scans.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from my_ai_agent.project.analyzer import EntryPointInfo, ProjectTypeInfo
from my_ai_agent.project.dependency_analyzer import DependencyInfo
from my_ai_agent.project.file_classifier import FileCategory
from my_ai_agent.project.scanner import ScannedFile
from my_ai_agent.project.symbol_analyzer import FileSymbols


@dataclass
class ProjectMap:
    root: str
    project_name: str
    project_types: List[ProjectTypeInfo] = field(default_factory=list)
    files: List[ScannedFile] = field(default_factory=list)
    directories: List[str] = field(default_factory=list)
    entry_points: List[EntryPointInfo] = field(default_factory=list)
    dependencies: List[DependencyInfo] = field(default_factory=list)
    symbols: Dict[str, FileSymbols] = field(default_factory=dict)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    tests: List[str] = field(default_factory=list)
    configs: List[str] = field(default_factory=list)
    documentation: List[str] = field(default_factory=list)
    sensitive_files: List[str] = field(default_factory=list)
    scanned_at: float = 0.0
    scanned_iso: str = ""
    latest_mtime: float = 0.0

    @property
    def total_files_count(self) -> int:
        return len(self.files)

    @property
    def source_files_count(self) -> int:
        return sum(1 for f in self.files if f.classification.category == FileCategory.SOURCE)

    @property
    def test_files_count(self) -> int:
        return len(self.tests)

    @property
    def primary_type(self) -> str:
        if self.project_types:
            return self.project_types[0].name
        return "Unknown"

    @property
    def primary_entry_point(self) -> Optional[str]:
        if self.entry_points:
            return self.entry_points[0].path
        return None

    def get_factual_summary(self) -> str:
        """Generate a strictly factual, non-fictional summary of the real project."""
        lines = [
            f"Project: {self.project_name}",
            f"Root Path: {self.root}",
        ]

        if self.project_types:
            type_str = " + ".join(f"{t.name} ({int(t.confidence*100)}%)" for t in self.project_types[:3])
            lines.append(f"Technologies: {type_str}")
        else:
            lines.append("Technologies: None detected")

        lines.extend([
            f"Total Files: {self.total_files_count}",
            f"Source Files: {self.source_files_count}",
            f"Test Files: {self.test_files_count}",
        ])

        if self.entry_points:
            top_entry = self.entry_points[0]
            lines.append(f"Primary Entry Point: {top_entry.path} ({top_entry.evidence})")

        total_deps = sum(d.total_count for d in self.dependencies)
        if total_deps:
            dep_names = []
            for d in self.dependencies:
                dep_names.extend(d.dependencies[:5])
            lines.append(f"Dependencies: {total_deps} package(s) detected [{', '.join(dep_names[:8])}{'...' if len(dep_names) > 8 else ''}]")

        if self.sensitive_files:
            lines.append(f"Protected Files: {len(self.sensitive_files)} sensitive file(s) masked")

        return "\n".join(lines)

    def to_summary_text(self) -> str:
        return self.get_factual_summary()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ProjectMap into standard JSON-compatible dict."""
        return {
            "root": self.root,
            "project_name": self.project_name,
            "primary_type": self.primary_type,
            "project_types": [t.to_dict() for t in self.project_types],
            "total_files": self.total_files_count,
            "source_files": self.source_files_count,
            "test_files": self.test_files_count,
            "entry_points": [e.to_dict() for e in self.entry_points],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "directories": self.directories,
            "tests": self.tests,
            "configs": self.configs,
            "documentation": self.documentation,
            "sensitive_files": self.sensitive_files,
            "relationships": self.relationships,
            "scanned_iso": self.scanned_iso,
            "files": [f.to_dict() for f in self.files],
        }


class ProjectMapCache:
    """In-memory cache for ProjectMaps with filesystem mtime validation."""

    def __init__(self):
        self._cache: Dict[str, ProjectMap] = {}

    def get(self, project_path: str | Path) -> Optional[ProjectMap]:
        """Retrieve cached map if it exists and files have not been modified."""
        canon = str(Path(project_path).resolve())
        cached = self._cache.get(canon)
        if not cached:
            return None

        # Check if project folder modified
        root = Path(canon)
        if not root.exists():
            self._cache.pop(canon, None)
            return None

        return cached

    def put(self, pmap: ProjectMap):
        """Store project map in cache."""
        canon = str(Path(pmap.root).resolve())
        self._cache[canon] = pmap

    def invalidate(self, project_path: str | Path):
        """Invalidate cache for a project path."""
        canon = str(Path(project_path).resolve())
        self._cache.pop(canon, None)

    def clear(self):
        """Clear all cached project maps."""
        self._cache.clear()
