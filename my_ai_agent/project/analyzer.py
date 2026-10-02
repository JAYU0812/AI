"""Project Analyzer Module for Step 3B.

Analyzes scanned project files to:
1. Detect project technologies with confidence and factual evidence.
2. Identify application entry points.
3. Build internal module import relationships graph.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from my_ai_agent.project.dependency_analyzer import DependencyInfo
from my_ai_agent.project.file_classifier import FileCategory
from my_ai_agent.project.scanner import ScannedFile
from my_ai_agent.project.symbol_analyzer import FileSymbols


@dataclass
class ProjectTypeInfo:
    name: str                           # e.g. "Python", "FastAPI", "React", "Node.js", "Flutter"
    confidence: float                   # 0.0 - 1.0
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
        }


@dataclass
class EntryPointInfo:
    path: str                           # Relative path to entry point file
    confidence: float                   # 0.0 - 1.0
    evidence: str                       # e.g. "Contains if __name__ == '__main__' and main()"

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
        }


class ProjectAnalyzer:
    """Analyzes scanned project files, dependencies, and symbols."""

    @classmethod
    def detect_project_types(
        cls,
        files: List[ScannedFile],
        dependencies: List[DependencyInfo],
        symbols_map: Dict[str, FileSymbols],
    ) -> List[ProjectTypeInfo]:
        """Detect real project technologies with confidence and evidence."""
        types: List[ProjectTypeInfo] = []

        file_names = {f.name.lower() for f in files}
        file_paths = {f.path.lower() for f in files}
        extensions = {f.ext for f in files if f.ext}

        # Collect all dependency names lowercase
        all_deps = set()
        for d in dependencies:
            for dep in d.dependencies + d.dev_dependencies:
                dep_name = dep.split("==")[0].split("@")[0].split(">=")[0].split("<=")[0].strip().lower()
                all_deps.add(dep_name)

        # 1. Python detection
        python_evidence = []
        py_files = [f for f in files if f.ext in (".py", ".pyw")]
        if py_files:
            python_evidence.append(f"{len(py_files)} Python source file{'s' if len(py_files) != 1 else ''}")
        if "requirements.txt" in file_names:
            python_evidence.append("requirements.txt manifest")
        if "pyproject.toml" in file_names:
            python_evidence.append("pyproject.toml manifest")
        if any(d.ecosystem == "python" for d in dependencies):
            python_evidence.append("Python dependencies configured")

        if python_evidence:
            conf = min(0.60 + 0.15 * len(python_evidence), 0.99)
            types.append(ProjectTypeInfo(name="Python", confidence=conf, evidence=python_evidence))

            # Frameworks: FastAPI
            fastapi_ev = []
            if "fastapi" in all_deps:
                fastapi_ev.append("fastapi in project dependencies")
            for p, syms in symbols_map.items():
                if any("fastapi" in imp.lower() for imp in syms.imports):
                    fastapi_ev.append(f"FastAPI imported in '{p}'")
                if any("fastapi" in sig.lower() for sig in syms.entry_point_signals):
                    fastapi_ev.append(f"FastAPI instance initialized in '{p}'")
            if fastapi_ev:
                types.append(ProjectTypeInfo(name="FastAPI", confidence=0.95, evidence=list(set(fastapi_ev))))

            # Frameworks: Flask
            flask_ev = []
            if "flask" in all_deps:
                flask_ev.append("flask in project dependencies")
            for p, syms in symbols_map.items():
                if any("flask" in imp.lower() for imp in syms.imports):
                    flask_ev.append(f"Flask imported in '{p}'")
            if flask_ev:
                types.append(ProjectTypeInfo(name="Flask", confidence=0.95, evidence=list(set(flask_ev))))

        # 2. Node.js / JavaScript / TypeScript detection
        node_evidence = []
        if "package.json" in file_names:
            node_evidence.append("package.json manifest")
        js_files = [f for f in files if f.ext in (".js", ".jsx", ".mjs", ".cjs")]
        if js_files:
            node_evidence.append(f"{len(js_files)} JavaScript file{'s' if len(js_files) != 1 else ''}")
        if node_evidence:
            types.append(ProjectTypeInfo(name="Node.js", confidence=0.90, evidence=node_evidence))

        ts_files = [f for f in files if f.ext in (".ts", ".tsx")]
        if ts_files or "tsconfig.json" in file_names:
            ts_ev = []
            if ts_files:
                ts_ev.append(f"{len(ts_files)} TypeScript file{'s' if len(ts_files) != 1 else ''}")
            if "tsconfig.json" in file_names:
                ts_ev.append("tsconfig.json configuration")
            types.append(ProjectTypeInfo(name="TypeScript", confidence=0.95, evidence=ts_ev))

        # React detection
        react_ev = []
        if "react" in all_deps:
            react_ev.append("react in dependencies")
        jsx_files = [f for f in files if f.ext in (".jsx", ".tsx")]
        if jsx_files:
            react_ev.append(f"{len(jsx_files)} JSX/TSX component file{'s' if len(jsx_files) != 1 else ''}")
        if react_ev:
            types.append(ProjectTypeInfo(name="React", confidence=0.95, evidence=react_ev))

        # 3. Flutter / Dart detection
        if "pubspec.yaml" in file_names or ".dart" in extensions:
            flutter_ev = []
            if "pubspec.yaml" in file_names:
                flutter_ev.append("pubspec.yaml manifest")
            dart_files = [f for f in files if f.ext == ".dart"]
            if dart_files:
                flutter_ev.append(f"{len(dart_files)} Dart file{'s' if len(dart_files) != 1 else ''}")
            types.append(ProjectTypeInfo(name="Flutter", confidence=0.95, evidence=flutter_ev))

        # 4. Rust detection
        if "cargo.toml" in file_names or ".rs" in extensions:
            types.append(ProjectTypeInfo(
                name="Rust",
                confidence=0.95,
                evidence=["Cargo.toml manifest or Rust source files"],
            ))

        # 5. Go detection
        if "go.mod" in file_names or ".go" in extensions:
            types.append(ProjectTypeInfo(
                name="Go",
                confidence=0.95,
                evidence=["go.mod manifest or Go source files"],
            ))

        # 6. Static HTML/CSS fallback if no other framework
        if not types:
            html_files = [f for f in files if f.ext in (".html", ".htm")]
            if html_files:
                types.append(ProjectTypeInfo(
                    name="HTML/CSS/JS",
                    confidence=0.85,
                    evidence=[f"{len(html_files)} HTML document(s) detected"],
                ))
            else:
                types.append(ProjectTypeInfo(
                    name="Generic",
                    confidence=0.50,
                    evidence=["Unidentified project structure"],
                ))

        # Sort by confidence descending
        types.sort(key=lambda t: t.confidence, reverse=True)
        return types

    @classmethod
    def detect_entry_points(
        cls,
        files: List[ScannedFile],
        symbols_map: Dict[str, FileSymbols],
    ) -> List[EntryPointInfo]:
        """Factually identify application entry points."""
        entries: List[EntryPointInfo] = []

        for f in files:
            path_lower = f.path.lower()
            name_lower = f.name.lower()
            syms = symbols_map.get(f.path)

            # Python entry points
            if f.ext == ".py":
                if name_lower in ("main.py", "app.py", "__main__.py", "server.py", "run.py", "manage.py"):
                    ev = f"Standard entry filename '{f.name}'"
                    conf = 0.90
                    if syms and syms.entry_point_signals:
                        ev += f" ({', '.join(syms.entry_point_signals)})"
                        conf = 0.98
                    entries.append(EntryPointInfo(path=f.path, confidence=conf, evidence=ev))
                elif syms and syms.entry_point_signals:
                    entries.append(EntryPointInfo(
                        path=f.path,
                        confidence=0.85,
                        evidence=f"Contains entry signals: {', '.join(syms.entry_point_signals)}",
                    ))

            # Node / JS entry points
            elif f.ext in (".js", ".ts", ".mjs"):
                if name_lower in ("index.js", "index.ts", "server.js", "server.ts", "app.js", "app.ts", "main.js", "main.ts"):
                    entries.append(EntryPointInfo(
                        path=f.path,
                        confidence=0.88,
                        evidence=f"Standard JS entry filename '{f.name}'",
                    ))

            # React / Web entry points
            elif name_lower in ("index.html", "app.jsx", "app.tsx", "main.jsx", "main.tsx"):
                entries.append(EntryPointInfo(
                    path=f.path,
                    confidence=0.85,
                    evidence=f"Frontend web entry point '{f.name}'",
                ))

            # Flutter entry point
            elif f.path == "lib/main.dart" or (f.ext == ".dart" and name_lower == "main.dart"):
                entries.append(EntryPointInfo(
                    path=f.path,
                    confidence=0.98,
                    evidence="Standard Flutter main application entry",
                ))

        # Sort by confidence descending
        entries.sort(key=lambda e: e.confidence, reverse=True)
        return entries

    @classmethod
    def build_import_relationships(
        cls,
        files: List[ScannedFile],
        symbols_map: Dict[str, FileSymbols],
    ) -> Dict[str, List[str]]:
        """Map each source file to internal project files it imports."""
        graph: Dict[str, List[str]] = {}

        # Precompute stem and module mappings
        stem_to_paths: Dict[str, List[str]] = {}
        for f in files:
            stem = Path(f.path).stem.lower()
            stem_to_paths.setdefault(stem, []).append(f.path)

        for f in files:
            syms = symbols_map.get(f.path)
            if not syms or not syms.imports:
                continue

            imported_project_files = set()

            for imp in syms.imports:
                if not imp:
                    continue
                # Normalize import token: e.g. "my_ai_agent.tools" -> "tools"
                parts = imp.split(".")
                last_part = parts[-1].lower()

                # Direct match with another project file
                if last_part in stem_to_paths:
                    for target_path in stem_to_paths[last_part]:
                        if target_path != f.path:
                            imported_project_files.add(target_path)

            if imported_project_files:
                graph[f.path] = sorted(list(imported_project_files))

        return graph
