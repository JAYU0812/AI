"""Context Builder Module for Step 3B.

Constructs minimal, factual, and security-safe project context for the Planner and Agent.
Never dumps the entire project into context; reads only top relevant files,
protects sensitive files, and summarizes relevant symbols and dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from my_ai_agent.project.project_map import ProjectMap
from my_ai_agent.project.relevance import RelevantFile


@dataclass
class ProjectContext:
    project_name: str
    primary_type: str
    project_root: str
    summary_text: str
    relevant_files: List[RelevantFile] = field(default_factory=list)
    key_symbols: Dict[str, List[str]] = field(default_factory=dict)
    file_contents: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

    def to_formatted_prompt_block(self) -> str:
        """Format into a concise, structured markdown block for agent decision-making."""
        lines = [
            "### Real Project Context (Step 3B)",
            f"- **Project:** {self.project_name} ({self.primary_type})",
            f"- **Root:** `{self.project_root}`",
        ]

        if self.dependencies:
            lines.append(f"- **Key Dependencies:** {', '.join(self.dependencies[:10])}")

        if self.relevant_files:
            lines.append("\n#### Relevant Project Files Identified:")
            for rf in self.relevant_files[:5]:
                sym_str = f" [Symbols: {', '.join(rf.matched_symbols)}]" if rf.matched_symbols else ""
                reason_str = f" — {rf.reasons[0]}" if rf.reasons else ""
                lines.append(f"  • `{rf.path}` (Relevance: {int(rf.score * 100)}%){sym_str}{reason_str}")

        if self.key_symbols:
            lines.append("\n#### Relevant Code Symbols:")
            for path, sym_list in list(self.key_symbols.items())[:5]:
                lines.append(f"  • **{path}**: {', '.join(sym_list)}")

        if self.file_contents:
            lines.append("\n#### Inspected File Contents:")
            for path, content in self.file_contents.items():
                ext = Path(path).suffix.lstrip(".") or "text"
                lines.append(f"``` {path} ({ext})\n{content}\n```")

        return "\n".join(lines)


class ContextBuilder:
    """Builds targeted, compact context from ProjectMap and Relevance ranking."""

    def __init__(self, max_files_to_read: int = 3, max_file_chars: int = 4000):
        self.max_files_to_read = max_files_to_read
        self.max_file_chars = max_file_chars

    def build_context(
        self,
        pmap: ProjectMap,
        ranked_files: List[RelevantFile],
        user_request: str,
    ) -> ProjectContext:
        """Construct compact context respecting safety boundaries."""
        root_path = Path(pmap.root)

        key_symbols: Dict[str, List[str]] = {}
        file_contents: Dict[str, str] = {}
        files_read_count = 0

        # Collect dependencies
        deps_summary = []
        for d in pmap.dependencies:
            deps_summary.extend(d.dependencies[:6])

        # Inspect top relevant files
        for rf in ranked_files:
            syms = pmap.symbols.get(rf.path)
            if syms:
                items = []
                for c in syms.classes:
                    methods_str = f"({', '.join(m.name for m in c.methods[:4])})" if c.methods else ""
                    items.append(f"class {c.name}{methods_str}")
                for fn in syms.functions:
                    items.append(f"def {fn.name}()")
                if items:
                    key_symbols[rf.path] = items

            # Safety check: NEVER read sensitive files!
            if rf.is_sensitive:
                file_contents[rf.path] = "[PROTECTED SENSITIVE FILE: Contents masked for security]"
                continue

            # Read content for top non-sensitive files up to budget
            if files_read_count < self.max_files_to_read:
                full_path = root_path / rf.path
                if full_path.exists() and full_path.is_file():
                    try:
                        raw = full_path.read_text(encoding="utf-8", errors="replace")
                        if len(raw) > self.max_file_chars:
                            content = raw[:self.max_file_chars] + f"\n... [Truncated remaining {len(raw) - self.max_file_chars} characters]"
                        else:
                            content = raw
                        file_contents[rf.path] = content
                        files_read_count += 1
                    except Exception as e:
                        file_contents[rf.path] = f"[Could not read file: {e}]"

        return ProjectContext(
            project_name=pmap.project_name,
            primary_type=pmap.primary_type,
            project_root=pmap.root,
            summary_text=pmap.get_factual_summary(),
            relevant_files=ranked_files,
            key_symbols=key_symbols,
            file_contents=file_contents,
            dependencies=deps_summary,
        )
