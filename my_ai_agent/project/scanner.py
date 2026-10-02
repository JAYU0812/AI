"""Project Scanner Module for Step 3B.

Recursively scans a real project directory on disk, applies configurable ignore rules,
respects .gitignore, gathers metadata, and classifies files while safeguarding sensitive files.
"""

from dataclasses import dataclass, field
from datetime import datetime
import fnmatch
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from my_ai_agent.project.file_classifier import FileClassification, FileClassifier


@dataclass
class ScannedFile:
    path: str               # Relative path with forward slashes (e.g. "src/main.py")
    name: str               # File name (e.g. "main.py")
    ext: str                # Lowercase extension with dot (e.g. ".py")
    size_bytes: int
    modified_at: float
    modified_iso: str
    classification: FileClassification
    is_sensitive: bool = False

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "name": self.name,
            "ext": self.ext,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "modified_iso": self.modified_iso,
            "category": self.classification.category.value,
            "is_sensitive": self.is_sensitive,
            "is_code": self.classification.is_code,
        }


class ProjectScanner:
    """Configurable scanner for discovering real project structure."""

    DEFAULT_IGNORED_DIRS: Set[str] = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "dist",
        "build",
        ".dart_tool",
        ".idea",
        ".vscode",
        "coverage",
        ".cache",
        "target",
        ".next",
        ".nuxt",
        ".gemini",
        ".artifacts",
        "artifacts",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }

    def __init__(
        self,
        ignored_dirs: Optional[Set[str]] = None,
        max_file_size_bytes: int = 5 * 1024 * 1024,  # 5 MB cap per file
        event_callback: Optional[Callable[[str, Dict], None]] = None,
    ):
        self.ignored_dirs = set(ignored_dirs or self.DEFAULT_IGNORED_DIRS)
        self.max_file_size_bytes = max_file_size_bytes
        self.event_callback = event_callback

    def _emit(self, event_type: str, data: Dict):
        """Emit real backend event."""
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception:
                pass

    def _load_gitignore(self, root_dir: Path) -> List[str]:
        """Read and parse root .gitignore rules if present."""
        gi_file = root_dir / ".gitignore"
        rules = []
        if gi_file.exists() and gi_file.is_file():
            try:
                content = gi_file.read_text(encoding="utf-8", errors="replace")
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        rules.append(stripped)
            except Exception:
                pass
        return rules

    def _is_git_ignored(self, rel_path: str, rules: List[str]) -> bool:
        """Check if path matches any gitignore rule."""
        rel_norm = rel_path.replace("\\", "/")
        for rule in rules:
            rule_clean = rule.rstrip("/")
            if fnmatch.fnmatch(rel_norm, rule_clean) or fnmatch.fnmatch(rel_norm, f"*/{rule_clean}"):
                return True
            if rule.endswith("/") and (rel_norm.startswith(rule_clean + "/") or f"/{rule_clean}/" in rel_norm):
                return True
        return False

    def scan(self, project_path: str | Path) -> List[ScannedFile]:
        """Recursively scan the project directory."""
        root = Path(project_path).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Project directory does not exist: {root}")

        self._emit("PROJECT_SCAN_STARTED", {"project_path": str(root)})
        gitignore_rules = self._load_gitignore(root)

        scanned_files: List[ScannedFile] = []

        for current_root, dirs, files in os.walk(root):
            curr_path = Path(current_root)
            rel_curr = curr_path.relative_to(root)
            parts = rel_curr.parts

            # Filter out ignored directories in-place to prevent traversing them
            dirs[:] = [
                d for d in dirs
                if d not in self.ignored_dirs
                and not any(fnmatch.fnmatch(d, ign) for ign in self.ignored_dirs)
                and not self._is_git_ignored(f"{str(rel_curr / d).replace('\\', '/')}/", gitignore_rules)
            ]

            for file_name in files:
                full_path = curr_path / file_name
                rel_file = full_path.relative_to(root)
                rel_file_str = str(rel_file).replace("\\", "/")

                # Check gitignore
                if self._is_git_ignored(rel_file_str, gitignore_rules):
                    continue

                try:
                    stat = full_path.stat()
                except OSError:
                    continue

                classification = FileClassifier.classify(rel_file_str)

                # Flag sensitive files
                is_sensitive = classification.is_sensitive or (
                    file_name.startswith(".env") or ".key" in file_name or ".pem" in file_name
                )

                item = ScannedFile(
                    path=rel_file_str,
                    name=file_name,
                    ext=full_path.suffix.lower(),
                    size_bytes=stat.st_size,
                    modified_at=stat.st_mtime,
                    modified_iso=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    classification=classification,
                    is_sensitive=is_sensitive,
                )
                scanned_files.append(item)

        scanned_files.sort(key=lambda f: f.path)

        self._emit("PROJECT_SCAN_COMPLETED", {
            "project_path": str(root),
            "files_count": len(scanned_files),
        })

        return scanned_files
