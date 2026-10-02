"""Project Understanding Engine Orchestrator for Step 3B.

The central facade coordinating scanning, AST symbol analysis, dependency detection,
project map building, relevance scoring, and context generation.
"""

from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from my_ai_agent.project.analyzer import EntryPointInfo, ProjectAnalyzer, ProjectTypeInfo
from my_ai_agent.project.context_builder import ContextBuilder, ProjectContext
from my_ai_agent.project.dependency_analyzer import DependencyAnalyzer, DependencyInfo
from my_ai_agent.project.file_classifier import FileCategory
from my_ai_agent.project.project_map import ProjectMap, ProjectMapCache
from my_ai_agent.project.relevance import RelevanceEngine, RelevantFile
from my_ai_agent.project.scanner import ProjectScanner, ScannedFile
from my_ai_agent.project.symbol_analyzer import FileSymbols, SymbolAnalyzer


class ProjectUnderstandingEngine:
    """High-level engine providing real understanding of projects on disk."""

    def __init__(
        self,
        event_callback: Optional[Callable[[str, Dict], None]] = None,
    ):
        self.event_callback = event_callback
        self.scanner = ProjectScanner(event_callback=self._emit)
        self.cache = ProjectMapCache()
        self.context_builder = ContextBuilder()

    def _emit(self, event_type: str, data: Dict):
        """Emit real backend event."""
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception:
                pass

    def understand_project(
        self,
        project_path: str | Path,
        force_refresh: bool = False,
    ) -> ProjectMap:
        """Scan, analyze, and build a full ProjectMap for the given project path."""
        canon_path = str(Path(project_path).resolve())
        root = Path(canon_path)

        if not force_refresh:
            cached = self.cache.get(canon_path)
            if cached is not None:
                return cached

        # 1. Recursive Scan
        files: List[ScannedFile] = self.scanner.scan(root)

        # 2. Dependency Analysis
        dependencies: List[DependencyInfo] = []
        for f in files:
            if f.classification.category == FileCategory.DEPENDENCY:
                info = DependencyAnalyzer.analyze_file(root / f.path, f.path)
                if info:
                    dependencies.append(info)

        # 3. Source Symbol & AST Extraction
        symbols_map: Dict[str, FileSymbols] = {}
        for f in files:
            if f.classification.is_code and not f.is_sensitive:
                syms = SymbolAnalyzer.analyze_file(root / f.path, f.path)
                if syms:
                    symbols_map[f.path] = syms

        # 4. Project Type & Entry Point Detection
        project_types: List[ProjectTypeInfo] = ProjectAnalyzer.detect_project_types(
            files, dependencies, symbols_map
        )
        self._emit("PROJECT_TYPE_DETECTED", {
            "types": [t.to_dict() for t in project_types],
        })

        entry_points: List[EntryPointInfo] = ProjectAnalyzer.detect_entry_points(
            files, symbols_map
        )

        relationships: Dict[str, List[str]] = ProjectAnalyzer.build_import_relationships(
            files, symbols_map
        )

        # 5. Partition category paths
        directories = sorted(list({str(Path(f.path).parent).replace("\\", "/") for f in files if Path(f.path).parent != Path(".")}))
        tests = [f.path for f in files if f.classification.category == FileCategory.TEST]
        configs = [f.path for f in files if f.classification.category == FileCategory.CONFIG]
        docs = [f.path for f in files if f.classification.category == FileCategory.DOCUMENTATION]
        sensitive = [f.path for f in files if f.is_sensitive]

        latest_mtime = max((f.modified_at for f in files), default=0.0)

        pmap = ProjectMap(
            root=canon_path,
            project_name=root.name,
            project_types=project_types,
            files=files,
            directories=directories,
            entry_points=entry_points,
            dependencies=dependencies,
            symbols=symbols_map,
            relationships=relationships,
            tests=tests,
            configs=configs,
            documentation=docs,
            sensitive_files=sensitive,
            scanned_at=datetime.now().timestamp(),
            scanned_iso=datetime.now().isoformat(),
            latest_mtime=latest_mtime,
        )

        self.cache.put(pmap)
        self._emit("PROJECT_MAP_BUILT", {
            "project_name": pmap.project_name,
            "total_files": pmap.total_files_count,
            "primary_type": pmap.primary_type,
        })

        return pmap

    def get_relevant_context(
        self,
        project_path: str | Path,
        user_request: str,
        max_results: int = 5,
    ) -> ProjectContext:
        """Find relevant files and construct minimal, security-safe context for planning."""
        pmap = self.understand_project(project_path)

        # Rank files
        ranked_files = RelevanceEngine.rank_files(pmap, user_request, max_results=max_results)

        self._emit("RELEVANT_FILES_IDENTIFIED", {
            "query": user_request,
            "relevant_files": [rf.to_dict() for rf in ranked_files],
        })

        # Build context
        context = self.context_builder.build_context(pmap, ranked_files, user_request)

        self._emit("CONTEXT_BUILT", {
            "project_name": context.project_name,
            "relevant_count": len(context.relevant_files),
            "files_inspected": list(context.file_contents.keys()),
        })

        return context

    def get_project_summary(self, project_path: str | Path) -> str:
        """Get factual text summary of the real project."""
        pmap = self.understand_project(project_path)
        return pmap.get_factual_summary()

    def explain_feature(self, project_path: str | Path, query: str) -> str:
        """Factual inquiry / understanding of existing code without making changes."""
        context = self.get_relevant_context(project_path, query, max_results=5)

        lines = [
            f"### Project Analysis: {context.project_name}",
            f"**Technologies:** {context.primary_type}",
            f"**Scanned Location:** `{context.project_root}`\n",
        ]

        if not context.relevant_files:
            lines.append("No specific files directly matched the requested topic in this project.")
            return "\n".join(lines)

        lines.append(f"**Relevant Files Inspected for \"{query}\":**")
        for rf in context.relevant_files[:5]:
            reasons_str = f" ({', '.join(rf.reasons[:2])})" if rf.reasons else ""
            lines.append(f"- `{rf.path}` (Relevance: {int(rf.score*100)}%){reasons_str}")

        lines.append("\n**Code Structure & Implementation Details:**")
        for path, content in context.file_contents.items():
            symbols = context.key_symbols.get(path, [])
            sym_desc = f" [Defines: {', '.join(symbols)}]" if symbols else ""
            lines.append(f"\n#### File `{path}`{sym_desc}")

            # Highlight functions or classes
            content_lines = content.splitlines()
            snippet = "\n".join(content_lines[:25])
            if len(content_lines) > 25:
                snippet += f"\n... [{len(content_lines) - 25} more lines in file]"
            lines.append(f"```python\n{snippet}\n```")

        return "\n".join(lines)

    def refresh(self, project_path: str | Path) -> ProjectMap:
        """Force a fresh scan and analysis of the project."""
        return self.understand_project(project_path, force_refresh=True)
