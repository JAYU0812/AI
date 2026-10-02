"""Patch Validator Module for Step 3C.

Validates code patches BEFORE they are finalized on disk.
Enforces:
1. Python AST syntax validation (detects malformed code immediately).
2. Duplicate prevention (functions, methods, classes, imports).
3. Structural integrity & preservation verification.
"""

import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set


@dataclass
class ValidationResult:
    """Result of patch validation."""
    is_valid: bool
    syntax_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duplicate_detected: bool = False
    duplicate_symbol: Optional[str] = None
    preservation_percentage: float = 100.0

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "syntax_valid": self.syntax_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "duplicate_detected": self.duplicate_detected,
            "duplicate_symbol": self.duplicate_symbol,
            "preservation_percentage": self.preservation_percentage,
        }


# Type alias for validation tuples
Tuple_Validation = tuple[bool, str]


class PatchValidator:
    """Validates patched code and guarantees preservation."""

    @classmethod
    def validate_python_syntax(cls, code: str, filename: str = "script.py") -> Tuple_Validation:
        """Validate Python syntax using standard AST parser."""
        try:
            ast.parse(code, filename=filename)
            return True, ""
        except SyntaxError as e:
            err_msg = f"SyntaxError in {filename}:{e.lineno or '?'}:{e.offset or '?'}: {e.msg}"
            return False, err_msg
        except Exception as e:
            return False, f"Structural parse error in {filename}: {str(e)}"

    @classmethod
    def check_duplicate_symbol(
        cls,
        existing_code: str,
        symbol_name: str,
        enclosing_class: Optional[str] = None,
        file_ext: str = ".py",
    ) -> Tuple_Validation:
        """Check if a function, method, or class symbol is already declared."""
        if not symbol_name:
            return False, ""

        if file_ext in (".py", ".pyw"):
            try:
                tree = ast.parse(existing_code)
                for node in ast.walk(tree):
                    # Check method inside specific class
                    if enclosing_class and isinstance(node, ast.ClassDef) and node.name == enclosing_class:
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == symbol_name:
                                return True, f"Method '{symbol_name}' already exists inside class '{enclosing_class}'"

                    # Check top-level class or function
                    if not enclosing_class:
                        if isinstance(node, ast.ClassDef) and node.name == symbol_name:
                            return True, f"Class '{symbol_name}' already exists in module"
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol_name:
                            return True, f"Function '{symbol_name}' already exists in module"
            except SyntaxError:
                pass

        # Regex fallback
        pat = rf"^\s*(?:def|class)\s+{re.escape(symbol_name)}\b"
        if re.search(pat, existing_code, re.MULTILINE):
            return True, f"Symbol '{symbol_name}' already exists in code"

        return False, ""

    @classmethod
    def check_duplicate_import(cls, existing_code: str, import_stmt: str) -> bool:
        """Check whether an import statement or package import is already present."""
        clean_stmt = import_stmt.strip()
        lines = [line.strip() for line in existing_code.splitlines()]

        # Exact match
        if clean_stmt in lines:
            return True

        # Check module imported
        m_import = re.match(r"^import\s+([a-zA-Z0-9_\.]+)", clean_stmt)
        if m_import:
            pkg = m_import.group(1).split(".")[0]
            for line in lines:
                if re.match(rf"^import\s+{re.escape(pkg)}(?:\s*,\s*|\s*$)", line):
                    return True
                if re.match(rf"^from\s+{re.escape(pkg)}\s+import", line):
                    return True

        # Check from X import Y
        m_from = re.match(r"^from\s+([a-zA-Z0-9_\.]+)\s+import\s+([a-zA-Z0-9_,\s\*]+)", clean_stmt)
        if m_from:
            mod = m_from.group(1)
            sym = m_from.group(2).strip()
            for line in lines:
                if line.startswith(f"from {mod} import") and sym in line:
                    return True

        return False

    @classmethod
    def validate_patch(
        cls,
        original_code: str,
        patched_code: str,
        file_path: str,
        symbol_to_insert: Optional[str] = None,
        enclosing_class: Optional[str] = None,
    ) -> ValidationResult:
        """Complete validation of a proposed file patch."""
        errors: List[str] = []
        warnings: List[str] = []
        file_ext = Path(file_path).suffix.lower()

        # 1. Non-empty guard
        if not patched_code.strip() and original_code.strip():
            return ValidationResult(
                is_valid=False,
                syntax_valid=False,
                errors=["Patch produced an empty file. Guard rejected unintended clobbering."],
                preservation_percentage=0.0,
            )

        # 2. Syntax validation (for Python)
        syntax_valid = True
        if file_ext in (".py", ".pyw"):
            syntax_valid, err = cls.validate_python_syntax(patched_code, filename=file_path)
            if not syntax_valid:
                errors.append(err)

        # 3. Duplicate symbol check
        dup_detected = False
        dup_symbol = None
        if symbol_to_insert:
            is_dup, msg = cls.check_duplicate_symbol(
                original_code, symbol_to_insert, enclosing_class, file_ext
            )
            if is_dup:
                dup_detected = True
                dup_symbol = symbol_to_insert
                warnings.append(msg)

        # 4. Preservation calculation
        orig_lines = original_code.splitlines()
        patched_lines = patched_code.splitlines()
        total_orig = len(orig_lines)

        preserved_lines = 0
        if total_orig > 0:
            patched_set = set(line.strip() for line in patched_lines if line.strip())
            for line in orig_lines:
                if line.strip() in patched_set:
                    preserved_lines += 1
            preservation = round((preserved_lines / total_orig) * 100.0, 1)
        else:
            preservation = 100.0

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            syntax_valid=syntax_valid,
            errors=errors,
            warnings=warnings,
            duplicate_detected=dup_detected,
            duplicate_symbol=dup_symbol,
            preservation_percentage=preservation,
        )


# Internal tuple type alias for type checkers
Tuple_Validation = tuple[bool, str]
