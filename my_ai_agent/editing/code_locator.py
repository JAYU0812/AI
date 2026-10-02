"""Code Locator Module for Step 3C.

Locates exact target positions inside code files using multiple strategies:
1. Python AST symbol locator (class, method, function, imports, main block).
2. Exact text / substring matcher.
3. Normalized line matcher.
4. Structural insertion point finder (inside class body, import block, before main).
"""

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Tuple


@dataclass
class LocationResult:
    """Exact location within a file."""
    found: bool
    start_line: int                             # 1-indexed start line
    end_line: int                               # 1-indexed end line (inclusive)
    indentation: str = ""                       # e.g. "    "
    matched_text: str = ""
    strategy: str = "none"                      # "ast_symbol", "exact_text", "normalized", "class_body", "import_block"
    symbol_type: Optional[str] = None           # "class", "function", "method", "import"
    parent_symbol: Optional[str] = None         # Name of enclosing class if method


class CodeLocator:
    """Multi-signal code location engine."""

    @classmethod
    def locate_target(
        cls,
        code: str,
        target_symbol: Optional[str] = None,
        target_text: Optional[str] = None,
        line_number: Optional[int] = None,
        file_ext: str = ".py",
    ) -> LocationResult:
        """Find the exact location using available signals in priority order."""
        lines = code.splitlines(keepends=True)
        total_lines = len(lines)

        # 1. Line number direct lookup
        if line_number is not None and 1 <= line_number <= total_lines:
            target_idx = line_number - 1
            line_content = lines[target_idx]
            indent = cls._get_indentation(line_content)
            return LocationResult(
                found=True,
                start_line=line_number,
                end_line=line_number,
                indentation=indent,
                matched_text=line_content,
                strategy="line_number",
            )

        # 2. AST-based symbol locator for Python files
        if file_ext in (".py", ".pyw") and target_symbol:
            ast_res = cls._locate_python_symbol(code, target_symbol)
            if ast_res.found:
                return ast_res

        # 3. Exact text matching
        if target_text:
            exact_res = cls._locate_exact_text(code, target_text)
            if exact_res.found:
                return exact_res

            # 4. Normalized text matching (ignoring whitespace variations)
            norm_res = cls._locate_normalized_text(code, target_text)
            if norm_res.found:
                return norm_res

        # 5. Regex / keyword matching fallback for symbol
        if target_symbol:
            kw_res = cls._locate_symbol_by_regex(code, target_symbol)
            if kw_res.found:
                return kw_res

        return LocationResult(found=False, start_line=0, end_line=0, strategy="not_found")

    @classmethod
    def find_class_insertion_point(cls, code: str, class_name: str) -> LocationResult:
        """Find the optimal line and indentation to insert a new method inside a class."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return cls._fallback_class_insertion(code, class_name)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_indent = cls._get_node_indentation(code, node)
                method_indent = class_indent + "    "

                # Find last member inside class body
                if node.body:
                    last_member = node.body[-1]
                    insert_line = getattr(last_member, "end_lineno", last_member.lineno)
                    return LocationResult(
                        found=True,
                        start_line=insert_line,
                        end_line=insert_line,
                        indentation=method_indent,
                        strategy="class_body",
                        symbol_type="class",
                        parent_symbol=class_name,
                    )
                else:
                    # Empty class body
                    return LocationResult(
                        found=True,
                        start_line=node.lineno,
                        end_line=node.lineno,
                        indentation=method_indent,
                        strategy="class_body",
                        symbol_type="class",
                        parent_symbol=class_name,
                    )

        # Fallback to regex if AST node not matched
        return cls._fallback_class_insertion(code, class_name)

    @classmethod
    def find_import_insertion_point(cls, code: str) -> LocationResult:
        """Find the ideal insertion line for a new top-level import statement."""
        lines = code.splitlines()
        if not lines:
            return LocationResult(found=True, start_line=0, end_line=0, strategy="import_block")

        try:
            tree = ast.parse(code)
            last_import_line = 0
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    end_ln = getattr(node, "end_lineno", node.lineno)
                    if end_ln > last_import_line:
                        last_import_line = end_ln

            if last_import_line > 0:
                return LocationResult(
                    found=True,
                    start_line=last_import_line,
                    end_line=last_import_line,
                    strategy="import_block",
                    symbol_type="import",
                )

            # Check if there is a top-level module docstring
            if tree.body and isinstance(tree.body[0], ast.Expr):
                val = tree.body[0].value
                if (isinstance(val, ast.Constant) and isinstance(val.value, str)) or isinstance(val, getattr(ast, "Str", type(None))):
                    doc_end = getattr(tree.body[0], "end_lineno", tree.body[0].lineno)
                    return LocationResult(
                        found=True,
                        start_line=doc_end,
                        end_line=doc_end,
                        strategy="import_block",
                        symbol_type="import",
                    )

            # Valid Python without imports or module docstring -> insert at top of file
            return LocationResult(found=True, start_line=0, end_line=0, strategy="import_block")
        except SyntaxError:
            pass

        # Text-based fallback only when AST fails to parse:
        # Check if first non-empty lines are a module docstring
        in_docstring = False
        last_import_idx = -1
        docstring_end_idx = -1

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Stop if we hit class or def before imports
            if not in_docstring and (stripped.startswith("class ") or stripped.startswith("def ")):
                break

            if stripped.startswith('"""') or stripped.startswith("'''"):
                if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    docstring_end_idx = i
                elif not in_docstring:
                    in_docstring = True
                else:
                    in_docstring = False
                    docstring_end_idx = i
                continue

            if in_docstring:
                continue

            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import_idx = i

        if last_import_idx >= 0:
            return LocationResult(
                found=True,
                start_line=last_import_idx + 1,
                end_line=last_import_idx + 1,
                strategy="import_block",
            )
        elif docstring_end_idx >= 0:
            return LocationResult(
                found=True,
                start_line=docstring_end_idx + 1,
                end_line=docstring_end_idx + 1,
                strategy="import_block",
            )

        # Default to top of file (line 0)
        return LocationResult(found=True, start_line=0, end_line=0, strategy="import_block")

    @classmethod
    def find_function_insertion_point(cls, code: str) -> LocationResult:
        """Find line to insert a new top-level function (before main block or at EOF)."""
        lines = code.splitlines()
        for i, line in enumerate(lines):
            if 'if __name__ == "__main__":' in line or "if __name__ == '__main__':" in line:
                return LocationResult(
                    found=True,
                    start_line=max(1, i),
                    end_line=max(1, i),
                    strategy="before_main",
                )
        # End of file
        return LocationResult(
            found=True,
            start_line=max(1, len(lines)),
            end_line=max(1, len(lines)),
            strategy="eof",
        )

    # ── Internal Helpers ──────────────────────────────────────────

    @classmethod
    def _locate_python_symbol(cls, code: str, symbol_name: str) -> LocationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return LocationResult(found=False, start_line=0, end_line=0)

        lines = code.splitlines(keepends=True)

        for node in ast.walk(tree):
            # 1. Match Class
            if isinstance(node, ast.ClassDef) and node.name == symbol_name:
                end_ln = getattr(node, "end_lineno", node.lineno)
                matched = "".join(lines[node.lineno - 1 : end_ln])
                return LocationResult(
                    found=True,
                    start_line=node.lineno,
                    end_line=end_ln,
                    indentation=cls._get_node_indentation(code, node),
                    matched_text=matched,
                    strategy="ast_symbol",
                    symbol_type="class",
                )

            # 2. Match Function or Method
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol_name:
                end_ln = getattr(node, "end_lineno", node.lineno)
                matched = "".join(lines[node.lineno - 1 : end_ln])
                return LocationResult(
                    found=True,
                    start_line=node.lineno,
                    end_line=end_ln,
                    indentation=cls._get_node_indentation(code, node),
                    matched_text=matched,
                    strategy="ast_symbol",
                    symbol_type="function",
                )

        return LocationResult(found=False, start_line=0, end_line=0)

    @classmethod
    def _locate_exact_text(cls, code: str, target: str) -> LocationResult:
        idx = code.find(target)
        if idx == -1:
            return LocationResult(found=False, start_line=0, end_line=0)

        # Compute line numbers
        start_line = code[:idx].count("\n") + 1
        end_line = start_line + target.count("\n")
        lines = code.splitlines(keepends=True)
        indent = cls._get_indentation(lines[start_line - 1]) if start_line <= len(lines) else ""

        return LocationResult(
            found=True,
            start_line=start_line,
            end_line=end_line,
            indentation=indent,
            matched_text=target,
            strategy="exact_text",
        )

    @classmethod
    def _locate_normalized_text(cls, code: str, target: str) -> LocationResult:
        def norm(s: str) -> str:
            return " ".join(s.split())

        code_lines = [norm(line) for line in code.splitlines()]
        target_lines = [norm(line) for line in target.splitlines() if line.strip()]

        if not target_lines:
            return LocationResult(found=False, start_line=0, end_line=0)

        first_target = target_lines[0]
        n_target = len(target_lines)

        for i in range(len(code_lines) - n_target + 1):
            if code_lines[i] == first_target:
                # Check subsequence
                if code_lines[i : i + n_target] == target_lines:
                    raw_lines = code.splitlines(keepends=True)
                    start_line = i + 1
                    end_line = i + n_target
                    indent = cls._get_indentation(raw_lines[i])
                    matched = "".join(raw_lines[i : i + n_target])
                    return LocationResult(
                        found=True,
                        start_line=start_line,
                        end_line=end_line,
                        indentation=indent,
                        matched_text=matched,
                        strategy="normalized",
                    )

        return LocationResult(found=False, start_line=0, end_line=0)

    @classmethod
    def _locate_symbol_by_regex(cls, code: str, symbol_name: str) -> LocationResult:
        lines = code.splitlines()
        pat_func = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(symbol_name)}\b")
        pat_class = re.compile(rf"^\s*class\s+{re.escape(symbol_name)}\b")

        for i, line in enumerate(lines):
            if pat_func.search(line) or pat_class.search(line):
                # Determine end line by indentation
                base_indent = len(line) - len(line.lstrip())
                end_i = i
                for j in range(i + 1, len(lines)):
                    cur_line = lines[j]
                    if not cur_line.strip():
                        continue
                    cur_indent = len(cur_line) - len(cur_line.lstrip())
                    if cur_indent <= base_indent:
                        break
                    end_i = j

                return LocationResult(
                    found=True,
                    start_line=i + 1,
                    end_line=end_i + 1,
                    indentation=line[:base_indent],
                    matched_text="\n".join(lines[i : end_i + 1]),
                    strategy="regex_symbol",
                )

        return LocationResult(found=False, start_line=0, end_line=0)

    @classmethod
    def _fallback_class_insertion(cls, code: str, class_name: str) -> LocationResult:
        lines = code.splitlines()
        pat = re.compile(rf"^\s*class\s+{re.escape(class_name)}\b")

        for i, line in enumerate(lines):
            if pat.search(line):
                base_indent = len(line) - len(line.lstrip())
                method_indent = line[:base_indent] + "    "
                end_i = i
                for j in range(i + 1, len(lines)):
                    cur_line = lines[j]
                    if not cur_line.strip():
                        continue
                    cur_indent = len(cur_line) - len(cur_line.lstrip())
                    if cur_indent <= base_indent:
                        break
                    end_i = j

                return LocationResult(
                    found=True,
                    start_line=end_i + 1,
                    end_line=end_i + 1,
                    indentation=method_indent,
                    strategy="fallback_class_body",
                    symbol_type="class",
                    parent_symbol=class_name,
                )

        return LocationResult(found=False, start_line=0, end_line=0)

    @staticmethod
    def _get_indentation(line: str) -> str:
        return line[: len(line) - len(line.lstrip())]

    @classmethod
    def _get_node_indentation(cls, code: str, node: ast.AST) -> str:
        lines = code.splitlines()
        if hasattr(node, "lineno") and 1 <= node.lineno <= len(lines):
            line = lines[node.lineno - 1]
            return cls._get_indentation(line)
        return ""
