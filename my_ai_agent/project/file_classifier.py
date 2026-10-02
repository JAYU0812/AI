"""File Classification Module for Step 3B.

Categorizes project files into deterministic semantic categories:
SOURCE, TEST, CONFIG, DEPENDENCY, DOCUMENTATION, ASSET, DATABASE, SCRIPT, BUILD, SENSITIVE, UNKNOWN.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re


class FileCategory(str, Enum):
    SOURCE = "SOURCE"
    TEST = "TEST"
    CONFIG = "CONFIG"
    DEPENDENCY = "DEPENDENCY"
    DOCUMENTATION = "DOCUMENTATION"
    ASSET = "ASSET"
    DATABASE = "DATABASE"
    SCRIPT = "SCRIPT"
    BUILD = "BUILD"
    GENERATED = "GENERATED"
    SENSITIVE = "SENSITIVE"
    UNKNOWN = "UNKNOWN"


@dataclass
class FileClassification:
    category: FileCategory
    confidence: float
    reason: str
    is_sensitive: bool = False
    is_code: bool = False


class FileClassifier:
    """Deterministic, rule-based file classifier."""

    # Sensitive filename patterns (never read or expose raw contents)
    SENSITIVE_PATTERNS = [
        re.compile(r"^\.env(\..+)?$", re.IGNORECASE),
        re.compile(r".*(secret|credential|password|token).*", re.IGNORECASE),
        re.compile(r".*\.(pem|key|pkcs12|p12|id_rsa|id_dsa|id_ed25519)$", re.IGNORECASE),
        re.compile(r"^id_(rsa|dsa|ecdsa|ed25519)$", re.IGNORECASE),
    ]

    # Dependency manifests
    DEPENDENCY_FILENAMES = {
        "requirements.txt", "requirements-dev.txt", "requirements.in",
        "pipfile", "pipfile.lock", "pyproject.toml", "setup.py", "setup.cfg",
        "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "pubspec.yaml", "pubspec.lock",
        "cargo.toml", "cargo.lock",
        "go.mod", "go.sum",
        "composer.json", "composer.lock",
        "pom.xml", "build.gradle", "build.gradle.kts", "gemfile", "gemfile.lock",
    }

    # Configuration files
    CONFIG_PATTERNS = [
        re.compile(r"^tsconfig.*\.json$", re.IGNORECASE),
        re.compile(r"^vite\.config\.(js|ts|mjs|cjs)$", re.IGNORECASE),
        re.compile(r"^webpack\.config\.(js|ts|mjs|cjs)$", re.IGNORECASE),
        re.compile(r"^\.eslintrc(\..+)?$", re.IGNORECASE),
        re.compile(r"^\.prettierrc(\..+)?$", re.IGNORECASE),
        re.compile(r"^babel\.config\.(js|json)$", re.IGNORECASE),
        re.compile(r"^pytest\.ini$", re.IGNORECASE),
        re.compile(r"^tox\.ini$", re.IGNORECASE),
        re.compile(r"^dockerfile.*$", re.IGNORECASE),
        re.compile(r"^docker-compose.*\.ya?ml$", re.IGNORECASE),
        re.compile(r"^makefile$", re.IGNORECASE),
        re.compile(r"^\.gitignore$", re.IGNORECASE),
        re.compile(r"^\.dockerignore$", re.IGNORECASE),
        re.compile(r"^l10n\.yaml$", re.IGNORECASE),
        re.compile(r"^analysis_options\.yaml$", re.IGNORECASE),
    ]

    # Documentation files
    DOC_EXTENSIONS = {".md", ".markdown", ".rst", ".adoc"}
    DOC_FILENAMES = {"readme", "license", "changelog", "contributing", "authors", "notice"}

    # Asset extensions
    ASSET_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp", ".tiff",
        ".woff", ".woff2", ".ttf", ".eot", ".otf",
        ".mp3", ".wav", ".ogg", ".mp4", ".webm", ".avi", ".mov",
        ".pdf",
    }

    # Database extensions
    DATABASE_EXTENSIONS = {".sqlite", ".sqlite3", ".db", ".sql", ".prisma"}

    # Script extensions
    SCRIPT_EXTENSIONS = {".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1"}

    # Source code extensions
    SOURCE_EXTENSIONS = {
        ".py", ".pyw",
        ".js", ".jsx", ".mjs", ".cjs",
        ".ts", ".tsx",
        ".dart",
        ".go",
        ".rs",
        ".java", ".kt", ".kts",
        ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
        ".cs",
        ".php",
        ".rb",
        ".swift",
        ".html", ".htm",
        ".css", ".scss", ".sass", ".less",
        ".vue", ".svelte",
    }

    # Build / Generated patterns
    BUILD_EXTENSIONS = {".min.js", ".min.css", ".map", ".pyc", ".pyo", ".pyd", ".class", ".o", ".obj", ".exe", ".dll", ".so", ".dylib"}

    @classmethod
    def classify(cls, relative_path: str | Path) -> FileClassification:
        """Classify a file based on its relative path and name."""
        rel = Path(relative_path)
        name = rel.name
        name_lower = name.lower()
        ext_lower = rel.suffix.lower()
        parts_lower = [p.lower() for p in rel.parts]

        # 1. SENSITIVE Check (highest priority)
        for pat in cls.SENSITIVE_PATTERNS:
            if pat.match(name) or any(pat.match(p) for p in parts_lower):
                return FileClassification(
                    category=FileCategory.SENSITIVE,
                    confidence=1.0,
                    reason=f"Matched sensitive pattern '{pat.pattern}'",
                    is_sensitive=True,
                    is_code=False,
                )

        # 2. TEST Check
        is_test_dir = any(p in ("tests", "test", "__tests__", "spec", "specs") for p in parts_lower[:-1])
        is_test_file = (
            name_lower.startswith("test_")
            or name_lower.endswith("_test.py")
            or name_lower.endswith(".test.js")
            or name_lower.endswith(".spec.js")
            or name_lower.endswith(".test.ts")
            or name_lower.endswith(".spec.ts")
            or name_lower.endswith(".test.jsx")
            or name_lower.endswith(".spec.jsx")
            or name_lower.endswith(".test.tsx")
            or name_lower.endswith(".spec.tsx")
            or name_lower.endswith("_test.dart")
        )
        if is_test_file or (is_test_dir and ext_lower in cls.SOURCE_EXTENSIONS):
            return FileClassification(
                category=FileCategory.TEST,
                confidence=0.95 if is_test_file else 0.85,
                reason="Test naming convention or located in test directory",
                is_sensitive=False,
                is_code=True,
            )

        # 3. DEPENDENCY Check
        if name_lower in cls.DEPENDENCY_FILENAMES:
            return FileClassification(
                category=FileCategory.DEPENDENCY,
                confidence=1.0,
                reason=f"Standard dependency manifest: '{name}'",
                is_sensitive=False,
                is_code=False,
            )

        # 4. CONFIG Check
        for pat in cls.CONFIG_PATTERNS:
            if pat.match(name_lower):
                return FileClassification(
                    category=FileCategory.CONFIG,
                    confidence=0.95,
                    reason=f"Configuration file pattern: '{pat.pattern}'",
                    is_sensitive=False,
                    is_code=False,
                )

        # 5. DOCUMENTATION Check
        stem_lower = rel.stem.lower()
        if (
            ext_lower in cls.DOC_EXTENSIONS
            or stem_lower in cls.DOC_FILENAMES
            or "docs" in parts_lower[:-1]
            or "doc" in parts_lower[:-1]
        ):
            return FileClassification(
                category=FileCategory.DOCUMENTATION,
                confidence=0.95 if stem_lower in cls.DOC_FILENAMES else 0.85,
                reason="Documentation extension or file naming",
                is_sensitive=False,
                is_code=False,
            )

        # 6. ASSET Check
        if ext_lower in cls.ASSET_EXTENSIONS:
            return FileClassification(
                category=FileCategory.ASSET,
                confidence=0.98,
                reason=f"Static asset extension: '{ext_lower}'",
                is_sensitive=False,
                is_code=False,
            )

        # 7. DATABASE Check
        if ext_lower in cls.DATABASE_EXTENSIONS:
            return FileClassification(
                category=FileCategory.DATABASE,
                confidence=0.95,
                reason=f"Database artifact extension: '{ext_lower}'",
                is_sensitive=False,
                is_code=False,
            )

        # 8. SCRIPT Check
        if ext_lower in cls.SCRIPT_EXTENSIONS:
            return FileClassification(
                category=FileCategory.SCRIPT,
                confidence=0.90,
                reason=f"Executable shell script: '{ext_lower}'",
                is_sensitive=False,
                is_code=True,
            )

        # 9. BUILD / GENERATED Check
        is_build_dir = any(p in ("dist", "build", "out", "target", ".next", ".cache") for p in parts_lower[:-1])
        if is_build_dir or any(name_lower.endswith(b) for b in cls.BUILD_EXTENSIONS):
            return FileClassification(
                category=FileCategory.BUILD,
                confidence=0.90,
                reason="Build artifact or generated bundle",
                is_sensitive=False,
                is_code=False,
            )

        # 10. SOURCE Code Check
        if ext_lower in cls.SOURCE_EXTENSIONS:
            return FileClassification(
                category=FileCategory.SOURCE,
                confidence=0.95,
                reason=f"Primary source code file for extension '{ext_lower}'",
                is_sensitive=False,
                is_code=True,
            )

        # Fallback: UNKNOWN
        return FileClassification(
            category=FileCategory.UNKNOWN,
            confidence=0.50,
            reason=f"Unrecognized file format or extension: '{ext_lower or name}'",
            is_sensitive=False,
            is_code=False,
        )
