"""Context Manager and Budgeting System for Step 4.

Assembles and prioritizes relevant context for models using the Step 3B
relevance engine. Enforces strict context byte and file budgets so models
receive only high-value, non-sensitive context.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from my_ai_agent.project.context_builder import ContextBuilder
from my_ai_agent.project.project_map import ProjectMap
from my_ai_agent.project.project_understanding import ProjectUnderstandingEngine
from my_ai_agent.project.relevance import RelevanceEngine


@dataclass
class ContextBudget:
    """Configurable budget limits for model context assembly."""
    max_files: int = 10
    max_file_bytes: int = 200_000
    max_context_bytes: int = 500_000
    max_symbols: int = 50
    max_history_items: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "max_context_bytes": self.max_context_bytes,
            "max_symbols": self.max_symbols,
            "max_history_items": self.max_history_items,
        }


@dataclass
class AssembledContext:
    """Factual, prioritized context package for model reasoning."""
    system_role: str
    user_request: str
    project_summary: str
    relevant_files: List[Dict[str, Any]]
    relevant_symbols: List[Dict[str, Any]]
    recent_observations: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    total_bytes: int
    truncated: bool

    def to_markdown(self) -> str:
        """Render context as clean, structured markdown."""
        sections = [
            f"# SYSTEM ROLE\n{self.system_role}\n",
            f"# USER REQUEST\n{self.user_request}\n",
            f"# PROJECT SUMMARY\n{self.project_summary}\n",
        ]

        if self.relevant_files:
            sections.append("# RELEVANT FILES")
            for f in self.relevant_files:
                sections.append(f"## File: `{f['path']}` (Relevance: {f.get('relevance_score', 0.0):.2f})")
                sections.append(f"```\n{f.get('content', '')}\n```\n")

        if self.relevant_symbols:
            sections.append("# RELEVANT SYMBOLS")
            for sym in self.relevant_symbols:
                sections.append(f"- **{sym.get('name')}** ({sym.get('type')}) in `{sym.get('file')}`: {sym.get('signature', '')}")
            sections.append("")

        if self.recent_observations:
            sections.append("# RECENT OBSERVATIONS & TEST RESULTS")
            for obs in self.recent_observations:
                status = "PASSED" if obs.get("success") else "FAILED"
                sections.append(f"- [{obs.get('action')}] {status}: {obs.get('summary', '')}")
            sections.append("")

        if self.conversation_history:
            sections.append("# RECENT CONVERSATION")
            for msg in self.conversation_history:
                sections.append(f"**{msg.get('role', 'user').title()}**: {msg.get('content', '')}")
            sections.append("")

        return "\n".join(sections)


class ContextManager:
    """Orchestrates context gathering, Step 3B relevance prioritization, and budget enforcement."""

    DEFAULT_SYSTEM_ROLE = (
        "You are an AI software engineering worker. You understand user goals, "
        "inspect project architecture, plan atomic code modifications, observe tool outputs, "
        "diagnose errors factually, and provide structured action proposals."
    )

    def __init__(
        self,
        workspace_root: Path | str,
        project_engine: Optional[ProjectUnderstandingEngine] = None,
        budget: Optional[ContextBudget] = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.project_engine = project_engine or ProjectUnderstandingEngine()
        self.budget = budget or ContextBudget()

    def assemble_context(
        self,
        user_request: str,
        recent_observations: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        system_role: Optional[str] = None,
    ) -> AssembledContext:
        """Assembles factual prioritized context adhering strictly to ContextBudget."""
        sys_role = system_role or self.DEFAULT_SYSTEM_ROLE
        pmap: ProjectMap = self.project_engine.understand_project(self.workspace_root)

        # 1. Project Summary
        types_str = ", ".join(pt.name for pt in pmap.project_types) or "N/A"
        summary = (
            f"Type: {pmap.primary_type} | Files: {pmap.total_files_count} | "
            f"Entry: {pmap.primary_entry_point or 'N/A'} | Stacks: {types_str}"
        )

        # 2. Prioritize Relevant Files using Step 3B RelevanceEngine
        scored_files = RelevanceEngine.rank_files(pmap, user_request)
        selected_files: List[Dict[str, Any]] = []
        file_bytes_accumulated = 0
        truncated = False

        for scored in scored_files[:self.budget.max_files]:
            rel_path = scored.path
            # Never include sensitive files
            if scored.is_sensitive or scored.category == "SENSITIVE":
                continue

            full_path = self.workspace_root / rel_path
            if not full_path.exists() or not full_path.is_file():
                continue

            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
                content_len = len(content.encode("utf-8"))

                # Check budget
                if file_bytes_accumulated + content_len > self.budget.max_file_bytes:
                    # Truncate content
                    allowed_bytes = max(0, self.budget.max_file_bytes - file_bytes_accumulated)
                    if allowed_bytes > 200:
                        content = content[:allowed_bytes] + "\n... [Truncated by Context Budget] ..."
                        file_bytes_accumulated += allowed_bytes
                        selected_files.append({
                            "path": rel_path,
                            "relevance_score": scored.score,
                            "content": content,
                        })
                        truncated = True
                    break
                else:
                    file_bytes_accumulated += content_len
                    selected_files.append({
                        "path": rel_path,
                        "relevance_score": scored.score,
                        "content": content,
                    })
            except Exception:
                continue

        # 3. Relevant Symbols
        selected_symbols: List[Dict[str, Any]] = []
        if isinstance(pmap.symbols, dict):
            for file_path, file_sym in pmap.symbols.items():
                if any(f["path"] == file_path for f in selected_files):
                    for fn in getattr(file_sym, "functions", []):
                        selected_symbols.append({
                            "name": fn.name,
                            "type": "function",
                            "file": file_path,
                            "signature": getattr(fn, "signature", ""),
                        })
                    for cls_sym in getattr(file_sym, "classes", []):
                        selected_symbols.append({
                            "name": cls_sym.name,
                            "type": "class",
                            "file": file_path,
                            "signature": f"class {cls_sym.name}",
                        })

        # 4. Recent Observations
        obs_list = (recent_observations or [])[-self.budget.max_history_items:]

        # 5. Conversation History
        conv_list = (conversation_history or [])[-self.budget.max_history_items:]

        # Calculate total bytes
        assembled = AssembledContext(
            system_role=sys_role,
            user_request=user_request,
            project_summary=summary,
            relevant_files=selected_files,
            relevant_symbols=selected_symbols,
            recent_observations=obs_list,
            conversation_history=conv_list,
            total_bytes=0,
            truncated=truncated,
        )

        md = assembled.to_markdown()
        total_len = len(md.encode("utf-8"))
        assembled.total_bytes = total_len
        if total_len > self.budget.max_context_bytes:
            assembled.truncated = True

        return assembled
