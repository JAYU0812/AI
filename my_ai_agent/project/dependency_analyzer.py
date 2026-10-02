"""Dependency Analyzer Module for Step 3B.

Factually inspects real project dependency files (requirements.txt, package.json,
pyproject.toml, pubspec.yaml, Cargo.toml, go.mod, etc.) without running package managers.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set


@dataclass
class DependencyInfo:
    ecosystem: str                          # "python", "node", "flutter", "rust", "go", "php"
    manifest_path: str                      # Relative path (e.g. "requirements.txt")
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.dependencies) + len(self.dev_dependencies)

    def to_dict(self) -> Dict:
        return {
            "ecosystem": self.ecosystem,
            "manifest_path": self.manifest_path,
            "dependencies": self.dependencies,
            "dev_dependencies": self.dev_dependencies,
            "total_count": self.total_count,
        }


class DependencyAnalyzer:
    """Parses real project manifests into structured dependency lists."""

    @classmethod
    def analyze_file(cls, full_path: Path, rel_path: str) -> Optional[DependencyInfo]:
        """Inspect a single dependency manifest."""
        name_lower = full_path.name.lower()

        if not full_path.exists() or not full_path.is_file():
            return None

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        # 1. Python requirements.txt
        if name_lower.startswith("requirements") and name_lower.endswith(".txt"):
            deps = cls._parse_requirements_txt(content)
            is_dev = "dev" in name_lower or "test" in name_lower
            return DependencyInfo(
                ecosystem="python",
                manifest_path=rel_path,
                dependencies=[] if is_dev else deps,
                dev_dependencies=deps if is_dev else [],
            )

        # 2. Python pyproject.toml
        if name_lower == "pyproject.toml":
            return cls._parse_pyproject_toml(content, rel_path)

        # 3. Node package.json
        if name_lower == "package.json":
            return cls._parse_package_json(content, rel_path)

        # 4. Flutter/Dart pubspec.yaml
        if name_lower == "pubspec.yaml":
            return cls._parse_pubspec_yaml(content, rel_path)

        # 5. Rust Cargo.toml
        if name_lower == "cargo.toml":
            return cls._parse_cargo_toml(content, rel_path)

        # 6. Go go.mod
        if name_lower == "go.mod":
            return cls._parse_go_mod(content, rel_path)

        # 7. PHP composer.json
        if name_lower == "composer.json":
            return cls._parse_composer_json(content, rel_path)

        return None

    @classmethod
    def _parse_requirements_txt(cls, content: str) -> List[str]:
        deps = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Strip inline comments
            line = line.split("#")[0].strip()
            # Extract package name (before ==, >=, <=, ~=, <, >, ;)
            m = re.match(r"^([a-zA-Z0-9_\-\.]+)", line)
            if m:
                pkg_name = m.group(1).strip()
                if pkg_name:
                    deps.append(pkg_name.lower())
        return deps

    @classmethod
    def _parse_package_json(cls, content: str, rel_path: str) -> Optional[DependencyInfo]:
        try:
            data = json.loads(content)
        except Exception:
            return None

        deps = []
        dev_deps = []

        if isinstance(data.get("dependencies"), dict):
            for k in data["dependencies"].keys():
                deps.append(k)

        if isinstance(data.get("devDependencies"), dict):
            for k in data["devDependencies"].keys():
                dev_deps.append(k)

        return DependencyInfo(
            ecosystem="node",
            manifest_path=rel_path,
            dependencies=deps,
            dev_dependencies=dev_deps,
        )

    @classmethod
    def _parse_pyproject_toml(cls, content: str, rel_path: str) -> DependencyInfo:
        deps = []
        dev_deps = []

        in_deps = False
        in_dev_deps = False

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                header = stripped.lower()
                in_deps = (header == "[project.dependencies]" or header == "[tool.poetry.dependencies]")
                in_dev_deps = ("dev-dependencies" in header or "test" in header or "group.dev.dependencies" in header)
                continue

            if in_deps:
                m = re.match(r'^["\']?([a-zA-Z0-9_\-\.]+)', stripped)
                if m and not stripped.startswith("#"):
                    deps.append(stripped.strip('",\' '))
            elif in_dev_deps:
                m = re.match(r'^["\']?([a-zA-Z0-9_\-\.]+)', stripped)
                if m and not stripped.startswith("#"):
                    dev_deps.append(stripped.strip('",\' '))

        return DependencyInfo(
            ecosystem="python",
            manifest_path=rel_path,
            dependencies=deps,
            dev_dependencies=dev_deps,
        )

    @classmethod
    def _parse_pubspec_yaml(cls, content: str, rel_path: str) -> DependencyInfo:
        deps = []
        dev_deps = []
        section = None

        for line in content.splitlines():
            line_str = line.rstrip()
            if not line_str or line_str.startswith("#"):
                continue

            if not line_str.startswith(" ") and ":" in line_str:
                sec_name = line_str.split(":")[0].strip()
                if sec_name == "dependencies":
                    section = "deps"
                elif sec_name == "dev_dependencies":
                    section = "dev_deps"
                else:
                    section = None
                continue

            if section and line_str.startswith("  ") and not line_str.startswith("    "):
                pkg_part = line_str.strip().split(":")[0].strip()
                if pkg_part and pkg_part != "sdk" and pkg_part != "flutter":
                    if section == "deps":
                        deps.append(pkg_part)
                    else:
                        dev_deps.append(pkg_part)

        return DependencyInfo(
            ecosystem="flutter",
            manifest_path=rel_path,
            dependencies=deps,
            dev_dependencies=dev_deps,
        )

    @classmethod
    def _parse_cargo_toml(cls, content: str, rel_path: str) -> DependencyInfo:
        deps = []
        dev_deps = []
        section = None

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                h = stripped.lower()
                if h == "[dependencies]":
                    section = "deps"
                elif h == "[dev-dependencies]":
                    section = "dev"
                else:
                    section = None
                continue

            if section and "=" in stripped and not stripped.startswith("#"):
                pkg = stripped.split("=")[0].strip()
                if pkg:
                    (deps if section == "deps" else dev_deps).append(pkg)

        return DependencyInfo(
            ecosystem="rust",
            manifest_path=rel_path,
            dependencies=deps,
            dev_dependencies=dev_deps,
        )

    @classmethod
    def _parse_go_mod(cls, content: str, rel_path: str) -> DependencyInfo:
        deps = []
        in_require_block = False

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("require ("):
                in_require_block = True
                continue
            if in_require_block:
                if stripped == ")":
                    in_require_block = False
                    continue
                parts = stripped.split()
                if parts and not parts[0].startswith("//"):
                    deps.append(parts[0])
            elif stripped.startswith("require ") and not stripped.startswith("require ("):
                parts = stripped[8:].strip().split()
                if parts:
                    deps.append(parts[0])

        return DependencyInfo(
            ecosystem="go",
            manifest_path=rel_path,
            dependencies=deps,
            dev_dependencies=[],
        )

    @classmethod
    def _parse_composer_json(cls, content: str, rel_path: str) -> Optional[DependencyInfo]:
        try:
            data = json.loads(content)
        except Exception:
            return None

        deps = list(data.get("require", {}).keys())
        dev_deps = list(data.get("require-dev", {}).keys())

        return DependencyInfo(
            ecosystem="php",
            manifest_path=rel_path,
            dependencies=deps,
            dev_dependencies=dev_deps,
        )
