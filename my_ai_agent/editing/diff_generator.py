"""Diff Generator Module for Step 3C.

Generates factual, human-readable before/after unified diffs using standard difflib.
Computes precise code preservation statistics (lines added, removed, preserved).
Never generates fictional diffs.
"""

from dataclasses import dataclass, field
import difflib
from typing import Dict, List, Optional


@dataclass
class DiffResult:
    """Factual diff output and line preservation statistics."""
    file_path: str
    unified_diff: str                           # Unified diff text with --- and +++ headers
    lines_added: int
    lines_removed: int
    lines_preserved: int
    total_original_lines: int
    total_modified_lines: int
    summary_messages: List[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return self.lines_added > 0 or self.lines_removed > 0

    @property
    def preservation_percentage(self) -> float:
        if self.total_original_lines == 0:
            return 100.0
        return round((self.lines_preserved / self.total_original_lines) * 100.0, 1)

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "has_changes": self.has_changes,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "lines_preserved": self.lines_preserved,
            "total_original_lines": self.total_original_lines,
            "total_modified_lines": self.total_modified_lines,
            "preservation_percentage": self.preservation_percentage,
            "unified_diff": self.unified_diff,
            "summary_messages": self.summary_messages,
        }


class DiffGenerator:
    """Generates standard diffs and preservation metrics."""

    @classmethod
    def generate_diff(
        cls,
        original_code: str,
        modified_code: str,
        file_path: str = "file",
        context_lines: int = 3,
    ) -> DiffResult:
        """Compute unified diff and accurate line delta metrics."""
        orig_lines = original_code.splitlines(keepends=True)
        mod_lines = modified_code.splitlines(keepends=True)

        diff = difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=context_lines,
        )
        diff_text = "".join(diff)

        # Count added and removed lines
        added_count = 0
        removed_count = 0
        summary_messages = []

        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added_count += 1
                stripped = line[1:].strip()
                if stripped.startswith("def ") or stripped.startswith("class "):
                    summary_messages.append(f"Added: {stripped.split('(')[0]}")
                elif stripped.startswith("import ") or stripped.startswith("from "):
                    summary_messages.append(f"Added import: {stripped}")
            elif line.startswith("-") and not line.startswith("---"):
                removed_count += 1
                stripped = line[1:].strip()
                if stripped.startswith("def ") or stripped.startswith("class "):
                    summary_messages.append(f"Removed: {stripped.split('(')[0]}")

        # Compute preserved lines
        total_orig = len(orig_lines)
        total_mod = len(mod_lines)
        preserved_count = max(0, total_orig - removed_count)

        if not summary_messages:
            if added_count > 0 and removed_count > 0:
                summary_messages.append(f"Modified {file_path} (+{added_count} / -{removed_count} lines)")
            elif added_count > 0:
                summary_messages.append(f"Added {added_count} line{'s' if added_count != 1 else ''} to {file_path}")
            elif removed_count > 0:
                summary_messages.append(f"Removed {removed_count} line{'s' if removed_count != 1 else ''} from {file_path}")
            else:
                summary_messages.append(f"No changes in {file_path}")

        return DiffResult(
            file_path=file_path,
            unified_diff=diff_text,
            lines_added=added_count,
            lines_removed=removed_count,
            lines_preserved=preserved_count,
            total_original_lines=total_orig,
            total_modified_lines=total_mod,
            summary_messages=summary_messages,
        )
