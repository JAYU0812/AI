"""Relevance Engine Module for Step 3B.

Scores and ranks project files against a natural language user request
using deterministic signals (filenames, symbols, imports, categories).
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional, Set

from my_ai_agent.project.file_classifier import FileCategory
from my_ai_agent.project.project_map import ProjectMap
from my_ai_agent.project.scanner import ScannedFile


@dataclass
class RelevantFile:
    path: str
    score: float                        # 0.0 - 1.0
    reasons: List[str] = field(default_factory=list)
    matched_symbols: List[str] = field(default_factory=list)
    category: str = "SOURCE"
    is_sensitive: bool = False

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "score": round(self.score, 2),
            "reasons": self.reasons,
            "matched_symbols": self.matched_symbols,
            "category": self.category,
            "is_sensitive": self.is_sensitive,
        }


class RelevanceEngine:
    """Deterministic file relevance ranking engine."""

    STOPWORDS: Set[str] = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "can", "could", "will", "would", "should", "and", "or",
        "but", "if", "then", "else", "when", "up", "out", "so", "than", "too", "very",
        "s", "t", "just", "now", "it", "its", "this", "that", "these", "those",
        "please", "make", "create", "build", "write", "add", "implement", "update",
        "modify", "change", "fix", "run", "file", "code", "project", "app", "me", "my",
    }

    CONCEPT_EXPANSIONS: Dict[str, Set[str]] = {
        "auth": {"login", "logout", "authenticate", "authentication", "token", "jwt", "password", "session", "user", "credential"},
        "authentication": {"auth", "login", "logout", "authenticate", "token", "password", "session", "user"},
        "login": {"auth", "logout", "authenticate", "authentication", "token", "password", "session", "user"},
        "logout": {"auth", "login", "authenticate", "authentication", "session", "token", "user"},
        "db": {"database", "models", "schema", "store", "repository", "sql", "migration"},
        "database": {"db", "models", "schema", "store", "repository", "sql", "migration"},
        "api": {"route", "routes", "endpoint", "controller", "server", "fastapi", "flask"},
        "test": {"tests", "spec", "assert", "verify", "unittest", "pytest"},
    }

    @classmethod
    def _extract_query_tokens(cls, query: str) -> List[str]:
        """Extract meaningful keywords and domain concepts from user request."""
        clean = re.sub(r"[^a-zA-Z0-9_\-\.]", " ", query.lower())
        tokens = [t.strip(".-_") for t in clean.split() if len(t.strip(".-_")) >= 2]
        meaningful = [t for t in tokens if t not in cls.STOPWORDS]

        # Expand domain concepts
        expanded = set(meaningful)
        for t in meaningful:
            for concept, synonyms in cls.CONCEPT_EXPANSIONS.items():
                if t == concept or t in synonyms:
                    expanded.add(concept)
                    expanded.update(synonyms)

        return list(expanded) if expanded else tokens

    @classmethod
    def rank_files(
        cls,
        pmap: ProjectMap,
        user_request: str,
        max_results: int = 10,
    ) -> List[RelevantFile]:
        """Score and rank files in ProjectMap according to relevance to user request."""
        tokens = cls._extract_query_tokens(user_request)
        if not tokens:
            # If no specific query tokens, prioritize entry points and primary source files
            ranked = []
            for e in pmap.entry_points[:max_results]:
                ranked.append(RelevantFile(
                    path=e.path,
                    score=0.75,
                    reasons=[f"Default entry point: {e.evidence}"],
                ))
            return ranked

        scored_results: List[RelevantFile] = []

        for f in pmap.files:
            # Skip build or generated files by default
            if f.classification.category in (FileCategory.BUILD, FileCategory.GENERATED):
                continue

            score = 0.0
            reasons = []
            matched_symbols = []

            path_lower = f.path.lower()
            name_lower = f.name.lower()
            stem_lower = Path(f.name).stem.lower()

            # 1. Exact or partial filename token match
            for t in tokens:
                if t == stem_lower:
                    score += 0.50
                    reasons.append(f"Filename stem matches '{t}' exactly")
                elif t in stem_lower or (len(t) >= 4 and stem_lower in t) or (len(stem_lower) >= 3 and t.startswith(stem_lower)):
                    score += 0.40
                    reasons.append(f"Filename related to keyword '{t}'")
                elif t in path_lower:
                    score += 0.20
                    reasons.append(f"Path contains keyword '{t}'")

            # 2. Symbol matches (classes, functions)
            syms = pmap.symbols.get(f.path)
            if syms:
                for c in syms.classes:
                    c_name_lower = c.name.lower()
                    for t in tokens:
                        if t in c_name_lower or (len(t) >= 4 and (c_name_lower in t or c_name_lower.startswith(t[:4]))):
                            score += 0.35
                            reasons.append(f"Class '{c.name}' matches keyword '{t}'")
                            matched_symbols.append(c.name)

                for fn in syms.functions:
                    fn_name_lower = fn.name.lower()
                    for t in tokens:
                        if t in fn_name_lower or (len(t) >= 4 and (fn_name_lower in t or fn_name_lower.startswith(t[:4]))):
                            score += 0.30
                            reasons.append(f"Function '{fn.name}()' matches keyword '{t}'")
                            matched_symbols.append(f"{fn.name}()")

                # Match imported names or exports
                for imp in syms.imports:
                    imp_lower = imp.lower()
                    for t in tokens:
                        if t == imp_lower or t in imp_lower.split("."):
                            score += 0.15
                            reasons.append(f"Imports module '{imp}' related to '{t}'")

            # 3. Category weighting
            if f.classification.category == FileCategory.SOURCE:
                score += 0.05
            elif f.classification.category == FileCategory.TEST:
                # If query contains test/spec, boost tests
                if any(k in user_request.lower() for k in ("test", "spec", "verify", "assert", "coverage")):
                    score += 0.40
                    reasons.append("Test file relevant to testing task")
            elif f.classification.category == FileCategory.CONFIG:
                if any(k in user_request.lower() for k in ("config", "setting", "setup", "env", "port", "build")):
                    score += 0.35
                    reasons.append("Config file relevant to setup/configuration task")

            # 4. Entry Point boost
            if any(e.path == f.path for e in pmap.entry_points):
                score += 0.10
                reasons.append("Application entry point")

            # Sensitive files safeguard: flag them
            if f.is_sensitive:
                reasons.append("PROTECTED SENSITIVE FILE — contents will NOT be read")

            if score > 0.0:
                # Clamp score between 0.05 and 0.99
                final_score = min(max(score, 0.05), 0.99)
                scored_results.append(
                    RelevantFile(
                        path=f.path,
                        score=final_score,
                        reasons=reasons,
                        matched_symbols=list(set(matched_symbols)),
                        category=f.classification.category.value,
                        is_sensitive=f.is_sensitive,
                    )
                )

        # Sort by score descending
        scored_results.sort(key=lambda r: r.score, reverse=True)

        # 5. Relationship boost: if file A is high scoring, boost files it imports or imported by it
        if scored_results:
            top_paths = {r.path for r in scored_results[:3]}
            for item in scored_results:
                related = pmap.relationships.get(item.path, [])
                if any(rel in top_paths for rel in related):
                    item.score = min(item.score + 0.10, 0.99)
                    item.reasons.append("Directly imported by primary relevant file")

            scored_results.sort(key=lambda r: r.score, reverse=True)

        return scored_results[:max_results]
