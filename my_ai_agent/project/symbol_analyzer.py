"""Symbol & AST Analyzer Module for Step 3B.

Extracts classes, functions, imports, decorators, exports, and entry-point indicators
from source files using Python's native AST and regex token analysis for JS/TS.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional, Set


@dataclass
class FunctionSymbol:
    name: str
    line: int
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)
    params: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class ClassSymbol:
    name: str
    line: int
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionSymbol] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class FileSymbols:
    path: str
    classes: List[ClassSymbol] = field(default_factory=list)
    functions: List[FunctionSymbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    imported_names: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    entry_point_signals: List[str] = field(default_factory=list)
    total_lines: int = 0
    code_lines: int = 0

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "classes": [
                {
                    "name": c.name,
                    "line": c.line,
                    "bases": c.bases,
                    "methods": [m.name for m in c.methods],
                    "docstring": c.docstring,
                }
                for c in self.classes
            ],
            "functions": [
                {
                    "name": f.name,
                    "line": f.line,
                    "is_async": f.is_async,
                    "decorators": f.decorators,
                    "params": f.params,
                    "docstring": f.docstring,
                }
                for f in self.functions
            ],
            "imports": self.imports,
            "imported_names": self.imported_names,
            "exports": self.exports,
            "entry_point_signals": self.entry_point_signals,
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
        }


class SymbolAnalyzer:
    """Extracts structural syntax information from supported source languages."""

    @classmethod
    def analyze_file(cls, full_path: Path, rel_path: str) -> Optional[FileSymbols]:
        """Analyze a file and extract syntax symbols."""
        if not full_path.exists() or not full_path.is_file():
            return None

        ext = full_path.suffix.lower()

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        lines = content.splitlines()
        total_lines = len(lines)
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith(("#", "//", "/*", "*")))

        if ext == ".py":
            return cls._analyze_python(content, rel_path, total_lines, code_lines)
        elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs"):
            return cls._analyze_javascript(content, rel_path, total_lines, code_lines)
        else:
            # Fallback for other languages: line metrics only
            return FileSymbols(
                path=rel_path,
                total_lines=total_lines,
                code_lines=code_lines,
            )

    @classmethod
    def _analyze_python(cls, content: str, rel_path: str, total_lines: int, code_lines: int) -> FileSymbols:
        """Parse Python source code using standard library ast module."""
        symbols = FileSymbols(
            path=rel_path,
            total_lines=total_lines,
            code_lines=code_lines,
        )

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # If invalid syntax (e.g. template or work in progress), fallback gracefully
            return symbols

        for node in ast.iter_child_nodes(tree):
            # Imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    symbols.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                symbols.imports.append(mod)
                for alias in node.names:
                    symbols.imported_names.append(f"{mod}.{alias.name}" if mod else alias.name)

            # Classes
            elif isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(f"{ast.unparse(b)}")

                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(
                            FunctionSymbol(
                                name=item.name,
                                line=item.lineno,
                                is_async=isinstance(item, ast.AsyncFunctionDef),
                                decorators=[ast.unparse(d) for d in item.decorator_list],
                                params=[arg.arg for arg in item.args.args],
                                docstring=ast.get_docstring(item),
                            )
                        )

                symbols.classes.append(
                    ClassSymbol(
                        name=node.name,
                        line=node.lineno,
                        bases=bases,
                        methods=methods,
                        docstring=ast.get_docstring(node),
                    )
                )

            # Top-level Functions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.functions.append(
                    FunctionSymbol(
                        name=node.name,
                        line=node.lineno,
                        is_async=isinstance(node, ast.AsyncFunctionDef),
                        decorators=[ast.unparse(d) for d in node.decorator_list],
                        params=[arg.arg for arg in node.args.args],
                        docstring=ast.get_docstring(node),
                    )
                )

            # Entry point indicators (e.g. if __name__ == '__main__')
            elif isinstance(node, ast.If):
                test_str = ast.unparse(node.test)
                if '__name__' in test_str and '__main__' in test_str:
                    symbols.entry_point_signals.append("if __name__ == '__main__'")

            # Framework app instantiation (FastAPI, Flask, etc.)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                        func_name = ast.unparse(node.value.func)
                        if "FastAPI" in func_name:
                            symbols.entry_point_signals.append(f"FastAPI app instance: {target.id}")
                        elif "Flask" in func_name:
                            symbols.entry_point_signals.append(f"Flask app instance: {target.id}")

        return symbols

    @classmethod
    def _analyze_javascript(cls, content: str, rel_path: str, total_lines: int, code_lines: int) -> FileSymbols:
        """Parse JavaScript/TypeScript using reliable regex patterns."""
        symbols = FileSymbols(
            path=rel_path,
            total_lines=total_lines,
            code_lines=code_lines,
        )

        lines = content.splitlines()

        for idx, line in enumerate(lines, start=1):
            line_str = line.strip()

            # Imports: import ... from '...' or import '...'
            m_imp = re.match(r'^import\s+(?:.*?from\s+)?[\'"]([^\'"]+)[\'"]', line_str)
            if m_imp:
                symbols.imports.append(m_imp.group(1))

            # CommonJS requires: require('...')
            for m_req in re.finditer(r'require\([\'"]([^\'"]+)[\'"]\)', line_str):
                symbols.imports.append(m_req.group(1))

            # Classes: class Foo ...
            m_cls = re.match(r'^(?:export\s+)?class\s+([A-Za-z0-9_]+)(?:\s+extends\s+([A-Za-z0-9_]+))?', line_str)
            if m_cls:
                bases = [m_cls.group(2)] if m_cls.group(2) else []
                symbols.classes.append(ClassSymbol(name=m_cls.group(1), line=idx, bases=bases))

            # Functions: function foo(...)
            m_fn = re.match(r'^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)\s*\((.*?)\)', line_str)
            if m_fn:
                params = [p.strip() for p in m_fn.group(2).split(",") if p.strip()]
                is_async = "async" in line_str
                symbols.functions.append(FunctionSymbol(name=m_fn.group(1), line=idx, is_async=is_async, params=params))

            # Arrow / Const functions: const foo = (...) => ...
            m_arrow = re.match(r'^(?:export\s+)?const\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\((.*?)\)\s*=>', line_str)
            if m_arrow:
                params = [p.strip() for p in m_arrow.group(2).split(",") if p.strip()]
                symbols.functions.append(FunctionSymbol(name=m_arrow.group(1), line=idx, params=params))

            # Exports
            if line_str.startswith("export default "):
                symbols.exports.append(line_str[15:].rstrip(";"))
            elif line_str.startswith("module.exports"):
                symbols.exports.append("module.exports")

            # App / Server entry points
            if "app.listen(" in line_str or "server.listen(" in line_str:
                symbols.entry_point_signals.append("HTTP Server listen call")
            if "createRoot(" in line_str or "ReactDOM.render(" in line_str:
                symbols.entry_point_signals.append("React root render")

        return symbols
