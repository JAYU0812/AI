"""Patch Engine Module for Step 3C.

Applies precise, atomic code modifications using structured EditOperations.
Features:
1. Symbol-aware method/function insertion with accurate indentation.
2. Automatic import management (inserts missing imports at top, prevents duplicates).
3. Multi-file transactional editing with atomic commit and automatic rollback.
4. Memory/disk backup checkpoints.
5. Factual diff generation and line preservation statistics.
6. Real event emission and safety boundaries (workspace isolation, sensitive file protection).
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config import config
from my_ai_agent.editing.code_locator import CodeLocator, LocationResult
from my_ai_agent.editing.diff_generator import DiffGenerator, DiffResult
from my_ai_agent.editing.edit_operations import (
    EditOperation,
    EditOperationType,
    EditProposal,
    FileEditProposal,
)
from my_ai_agent.editing.patch_validator import PatchValidator, ValidationResult
from my_ai_agent.project.file_classifier import FileClassifier


@dataclass
class PatchResult:
    """Result of applying an individual file patch."""
    file_path: str
    success: bool
    original_code: str
    patched_code: str
    diff: Optional[DiffResult] = None
    validation: Optional[ValidationResult] = None
    error: str = ""
    operations_applied: int = 0

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "success": self.success,
            "error": self.error,
            "operations_applied": self.operations_applied,
            "diff": self.diff.to_dict() if self.diff else None,
            "validation": self.validation.to_dict() if self.validation else None,
        }


@dataclass
class TransactionResult:
    """Result of a multi-file transactional edit session."""
    success: bool
    proposal_intent: str
    modified_files: List[str] = field(default_factory=list)
    file_results: Dict[str, PatchResult] = field(default_factory=dict)
    diffs: List[DiffResult] = field(default_factory=list)
    rolled_back: bool = False
    errors: List[str] = field(default_factory=list)
    summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "proposal_intent": self.proposal_intent,
            "modified_files": self.modified_files,
            "rolled_back": self.rolled_back,
            "errors": self.errors,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "diffs": [d.to_dict() for d in self.diffs],
            "file_results": {k: v.to_dict() for k, v in self.file_results.items()},
        }


class PatchEngine:
    """Precise code editing and transactional patch execution engine."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = (workspace_root or config.WORKSPACE_DIR).resolve()
        self.backup_dir = self.workspace_root / ".agent_backups"
        self._event_listeners: List[Callable[[str, Dict[str, Any]], None]] = []
        self._checkpoints: Dict[str, Dict[str, str]] = {}

    def subscribe_events(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register listener for real editing events."""
        self._event_listeners.append(callback)

    def _emit(self, event_name: str, payload: Dict[str, Any]):
        for listener in self._event_listeners:
            try:
                listener(event_name, payload)
            except Exception:
                pass

    # ── Workspace Path Safety ──────────────────────────────────────

    def _resolve_safe_path(self, relative_path: str) -> Path:
        """Resolve path and verify it stays strictly inside workspace root."""
        raw = Path(relative_path)
        if raw.is_absolute():
            resolved = raw.resolve()
        else:
            resolved = (self.workspace_root / raw).resolve()

        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            raise PermissionError(
                f"Security boundary: Path traversal blocked outside workspace: '{relative_path}'"
            )

        # Block sensitive files
        classification = FileClassifier.classify(resolved.name)
        if classification.is_sensitive:
            raise PermissionError(
                f"Security boundary: Direct patch on sensitive file '{resolved.name}' is prohibited."
            )

        return resolved

    # ── Checkpoint & Backup Management ────────────────────────────

    def create_checkpoint(self, checkpoint_name: str, files: Optional[List[str]] = None) -> Dict[str, str]:
        """Save a snapshot of workspace files for rollback."""
        snapshot: Dict[str, str] = {}
        files_to_save = files or [
            str(p.relative_to(self.workspace_root)).replace("\\", "/")
            for p in self.workspace_root.rglob("*")
            if p.is_file() and ".agent_backups" not in p.parts
        ]

        for rel in files_to_save:
            try:
                full = self._resolve_safe_path(rel)
                if full.exists() and full.is_file():
                    snapshot[rel] = full.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

        self._checkpoints[checkpoint_name] = snapshot
        return snapshot

    def restore_checkpoint(self, checkpoint_name: str) -> bool:
        """Restore workspace to a saved checkpoint."""
        snapshot = self._checkpoints.get(checkpoint_name)
        if not snapshot:
            return False

        for rel, content in snapshot.items():
            try:
                full = self._resolve_safe_path(rel)
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content, encoding="utf-8")
            except Exception:
                pass
        return True

    # ── Transactional Multi-File Execution ────────────────────────

    def apply_proposal(self, proposal: EditProposal) -> TransactionResult:
        """Execute a multi-file edit proposal transactionally.
        
        If ANY file patch fails or fails syntax validation, ALL modified files
        are immediately rolled back to their original state.
        """
        self._emit("TRANSACTION_STARTED", {
            "intent": proposal.intent,
            "files": proposal.affected_files,
        })

        # 1. Capture snapshot for atomic rollback
        snapshot: Dict[str, str] = {}
        for file_prop in proposal.files:
            try:
                full_path = self._resolve_safe_path(file_prop.path)
                if full_path.exists() and full_path.is_file():
                    snapshot[file_prop.path] = full_path.read_text(encoding="utf-8", errors="replace")
                else:
                    # New file
                    snapshot[file_prop.path] = ""
            except Exception as e:
                return TransactionResult(
                    success=False,
                    proposal_intent=proposal.intent,
                    rolled_back=False,
                    errors=[f"Safety/path resolution error on '{file_prop.path}': {str(e)}"],
                )

        file_results: Dict[str, PatchResult] = {}
        modified_files: List[str] = []
        diffs: List[DiffResult] = []
        errors: List[str] = []

        # 2. Apply patches file-by-file in memory & test validation
        for file_prop in proposal.files:
            original_code = snapshot[file_prop.path]
            patch_res = self.apply_file_operations(
                file_path=file_prop.path,
                original_code=original_code,
                operations=file_prop.operations,
            )
            file_results[file_prop.path] = patch_res

            if not patch_res.success:
                errors.append(f"Failed to patch '{file_prop.path}': {patch_res.error}")
                break

            # Write provisionally to disk
            try:
                full_path = self._resolve_safe_path(file_prop.path)
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(patch_res.patched_code, encoding="utf-8")
                modified_files.append(file_prop.path)
                if patch_res.diff and patch_res.diff.has_changes:
                    diffs.append(patch_res.diff)
            except Exception as e:
                errors.append(f"Failed to write patch to '{file_prop.path}': {str(e)}")
                break

        # 3. If any error occurred, perform atomic ROLLBACK
        if errors:
            self._emit("TRANSACTION_ROLLED_BACK", {
                "intent": proposal.intent,
                "errors": errors,
                "reverting_files": modified_files,
            })
            for rel_path in modified_files:
                orig = snapshot.get(rel_path, "")
                try:
                    full = self._resolve_safe_path(rel_path)
                    if orig:
                        full.write_text(orig, encoding="utf-8")
                    elif full.exists():
                        full.unlink()
                except Exception:
                    pass

            return TransactionResult(
                success=False,
                proposal_intent=proposal.intent,
                modified_files=[],
                file_results=file_results,
                diffs=[],
                rolled_back=True,
                errors=errors,
                summary=f"Transaction rolled back due to error: {'; '.join(errors)}",
            )

        # 4. Commit successfully
        self._emit("TRANSACTION_COMMITTED", {
            "intent": proposal.intent,
            "modified_files": modified_files,
            "diffs_count": len(diffs),
        })

        summary_parts = []
        for d in diffs:
            summary_parts.extend(d.summary_messages)

        return TransactionResult(
            success=True,
            proposal_intent=proposal.intent,
            modified_files=modified_files,
            file_results=file_results,
            diffs=diffs,
            rolled_back=False,
            summary="; ".join(summary_parts) if summary_parts else "Changes applied successfully.",
        )

    # ── Single-File Operations Application ────────────────────────

    def apply_file_operations(
        self,
        file_path: str,
        original_code: str,
        operations: List[EditOperation],
    ) -> PatchResult:
        """Apply a sequence of atomic operations to one file's content."""
        current_code = original_code
        file_ext = Path(file_path).suffix.lower()
        ops_applied = 0

        for op in operations:
            self._emit("LOCATING_CODE", {
                "file": file_path,
                "operation": op.operation_type.value,
                "target_symbol": op.target_symbol,
            })

            # 1. Manage required imports first
            if op.required_imports:
                current_code = self._apply_imports(current_code, op.required_imports, file_ext)

            # 2. Dispatch specific operation
            try:
                if op.operation_type == EditOperationType.INSERT:
                    current_code = self._op_insert(current_code, op, file_ext)
                elif op.operation_type == EditOperationType.REPLACE:
                    current_code = self._op_replace(current_code, op, file_ext)
                elif op.operation_type == EditOperationType.DELETE:
                    current_code = self._op_delete(current_code, op, file_ext)
                elif op.operation_type == EditOperationType.APPEND:
                    current_code = self._op_append(current_code, op, file_ext)
                elif op.operation_type == EditOperationType.PREPEND:
                    current_code = self._op_prepend(current_code, op, file_ext)
                elif op.operation_type == EditOperationType.RENAME_SYMBOL:
                    current_code = self._op_rename(current_code, op, file_ext)
                ops_applied += 1
            except Exception as e:
                return PatchResult(
                    file_path=file_path,
                    success=False,
                    original_code=original_code,
                    patched_code=original_code,
                    error=f"Operation {op.operation_type.value} failed: {str(e)}",
                    operations_applied=ops_applied,
                )

        # 3. Validate patched file
        validation = PatchValidator.validate_patch(
            original_code=original_code,
            patched_code=current_code,
            file_path=file_path,
        )

        if not validation.is_valid:
            return PatchResult(
                file_path=file_path,
                success=False,
                original_code=original_code,
                patched_code=current_code,
                validation=validation,
                error=f"Patch validation failed: {'; '.join(validation.errors)}",
                operations_applied=ops_applied,
            )

        # 4. Generate diff
        diff = DiffGenerator.generate_diff(
            original_code=original_code,
            modified_code=current_code,
            file_path=file_path,
        )

        self._emit("PATCH_VALIDATED", {
            "file": file_path,
            "lines_added": diff.lines_added,
            "lines_removed": diff.lines_removed,
            "preservation": diff.preservation_percentage,
        })

        return PatchResult(
            file_path=file_path,
            success=True,
            original_code=original_code,
            patched_code=current_code,
            diff=diff,
            validation=validation,
            operations_applied=ops_applied,
        )

    # ── Operation Handlers ────────────────────────────────────────

    def _op_insert(self, code: str, op: EditOperation, file_ext: str) -> str:
        """Insert content symbol-aware inside a class, before/after a symbol, or at a line."""
        # 1. If target is inside a class (e.g. adding method to AuthService)
        if op.target_symbol:
            loc = CodeLocator.find_class_insertion_point(code, op.target_symbol)
            if loc.found:
                indent = loc.indentation or "    "
                indented_content = self._indent_content(op.content, indent)
                lines = code.splitlines(keepends=True)
                insert_idx = loc.start_line  # 1-indexed

                # Insert with empty line separator
                chunk = "\n" + indented_content.rstrip() + "\n"
                lines.insert(insert_idx, chunk)
                return "".join(lines)

            # Target symbol is a function or exact symbol
            loc_sym = CodeLocator.locate_target(code, target_symbol=op.target_symbol, file_ext=file_ext)
            if loc_sym.found:
                lines = code.splitlines(keepends=True)
                indent = loc_sym.indentation
                indented_content = self._indent_content(op.content, indent)
                target_idx = loc_sym.end_line if op.position == "after" else (loc_sym.start_line - 1)
                chunk = "\n" + indented_content.rstrip() + "\n"
                lines.insert(target_idx, chunk)
                return "".join(lines)

        # 2. Exact text target
        if op.target_text:
            loc_text = CodeLocator.locate_target(code, target_text=op.target_text, file_ext=file_ext)
            if loc_text.found:
                lines = code.splitlines(keepends=True)
                indent = loc_text.indentation
                indented_content = self._indent_content(op.content, indent)
                target_idx = loc_text.end_line if op.position == "after" else (loc_text.start_line - 1)
                lines.insert(target_idx, indented_content.rstrip() + "\n")
                return "".join(lines)

        # 3. Line number target
        if op.line_number is not None:
            lines = code.splitlines(keepends=True)
            target_idx = min(len(lines), max(0, op.line_number - 1 if op.position == "before" else op.line_number))
            lines.insert(target_idx, op.content.rstrip() + "\n")
            return "".join(lines)

        # 4. Fallback: insert before main block or append
        loc_fn = CodeLocator.find_function_insertion_point(code)
        lines = code.splitlines(keepends=True)
        chunk = "\n" + op.content.rstrip() + "\n\n"
        lines.insert(loc_fn.start_line, chunk)
        return "".join(lines)

    def _op_replace(self, code: str, op: EditOperation, file_ext: str) -> str:
        """Replace target symbol, exact text, or line range."""
        loc = CodeLocator.locate_target(
            code=code,
            target_symbol=op.target_symbol,
            target_text=op.target_text,
            line_number=op.line_number,
            file_ext=file_ext,
        )
        if not loc.found:
            raise ValueError(f"Target for REPLACE not found: symbol='{op.target_symbol}', text='{op.target_text}'")

        lines = code.splitlines(keepends=True)
        start_idx = loc.start_line - 1
        end_idx = loc.end_line  # slice end

        indent = loc.indentation
        indented_content = self._indent_content(op.content, indent)
        replacement_lines = [indented_content.rstrip() + "\n"]

        new_lines = lines[:start_idx] + replacement_lines + lines[end_idx:]
        return "".join(new_lines)

    def _op_delete(self, code: str, op: EditOperation, file_ext: str) -> str:
        """Delete target symbol, text, or line range."""
        loc = CodeLocator.locate_target(
            code=code,
            target_symbol=op.target_symbol,
            target_text=op.target_text,
            line_number=op.line_number,
            file_ext=file_ext,
        )
        if not loc.found:
            raise ValueError(f"Target for DELETE not found: symbol='{op.target_symbol}'")

        lines = code.splitlines(keepends=True)
        start_idx = loc.start_line - 1
        end_idx = loc.end_line
        new_lines = lines[:start_idx] + lines[end_idx:]
        return "".join(new_lines)

    def _op_append(self, code: str, op: EditOperation, file_ext: str) -> str:
        """Append content to file or class body."""
        if op.target_symbol:
            return self._op_insert(code, op, file_ext)
        return code.rstrip() + "\n\n" + op.content.rstrip() + "\n"

    def _op_prepend(self, code: str, op: EditOperation, file_ext: str) -> str:
        """Prepend content to file."""
        return op.content.rstrip() + "\n\n" + code.lstrip()

    def _op_rename(self, code: str, op: EditOperation, file_ext: str) -> str:
        """Safely rename a symbol across the file."""
        if not op.target_symbol or not op.content:
            return code
        pat = rf"\b{re.escape(op.target_symbol)}\b"
        return re.sub(pat, op.content.strip(), code)

    # ── Import Management ─────────────────────────────────────────

    def _apply_imports(self, code: str, required_imports: List[str], file_ext: str) -> str:
        """Ensure required imports are present without creating duplicates."""
        missing = [imp for imp in required_imports if not PatchValidator.check_duplicate_import(code, imp)]
        if not missing:
            return code

        loc = CodeLocator.find_import_insertion_point(code)
        lines = code.splitlines(keepends=True)

        insert_idx = loc.start_line  # 1-indexed insertion point
        import_block = "".join(f"{imp.strip()}\n" for imp in missing)

        if insert_idx == 0 or not lines:
            return import_block + "\n" + code

        target_idx = min(len(lines), insert_idx)
        lines.insert(target_idx, import_block)
        return "".join(lines)

    @staticmethod
    def _indent_content(content: str, base_indent: str) -> str:
        """Apply uniform base indentation to a multi-line code block."""
        lines = content.splitlines()
        if not lines:
            return ""

        # Determine relative indentation from first non-empty line
        rel_indent = 0
        for line in lines:
            if line.strip():
                rel_indent = len(line) - len(line.lstrip())
                break

        indented = []
        for line in lines:
            if not line.strip():
                indented.append("")
            else:
                curr_indent = len(line) - len(line.lstrip())
                adjusted = max(0, curr_indent - rel_indent)
                indented.append(base_indent + (" " * adjusted) + line.lstrip())

        return "\n".join(indented)
