"""Configuration module for the AI Agent Engine.

Defines workspace paths, limits, execution options, and visual styling.
"""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class AgentConfig:
    # Project root and workspace
    BASE_DIR: Path = Path(__file__).resolve().parent
    WORKSPACE_DIR: Path = Path(__file__).resolve().parent.parent / "workspace"
    DB_PATH: Path = Path(__file__).resolve().parent / "agent_memory.db"

    # Execution limits & safety
    COMMAND_TIMEOUT_SECONDS: int = 30
    MAX_FILE_READ_BYTES: int = 500_000
    MAX_PLANNING_STEPS: int = 5

    # Intelligence & Model Provider Configuration
    AI_MODEL_PROVIDER: str = os.getenv("AI_MODEL_PROVIDER", "deterministic")
    AI_MODEL_NAME: str = os.getenv("AI_MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    AI_MODEL_BACKEND: str = os.getenv("AI_MODEL_BACKEND", "transformers")
    AI_MODEL_PATH: str = os.getenv("AI_MODEL_PATH", "")
    AI_MODEL_DEVICE: str = os.getenv("AI_MODEL_DEVICE", "cpu")
    AI_MODEL_CONTEXT_LENGTH: int = int(os.getenv("AI_MODEL_CONTEXT_LENGTH", "2048"))
    AI_MODEL_TEMPERATURE: float = float(os.getenv("AI_MODEL_TEMPERATURE", "0.2"))
    AI_MODEL_MAX_TOKENS: int = int(os.getenv("AI_MODEL_MAX_TOKENS", "512"))
    AI_MODEL_TIMEOUT: int = int(os.getenv("AI_MODEL_TIMEOUT", "60"))
    AI_MODEL_STREAMING: bool = os.getenv("AI_MODEL_STREAMING", "true").lower() in ("true", "1", "yes")
    AI_MODEL_FALLBACK: str = os.getenv("AI_MODEL_FALLBACK", "deterministic")  # 'deterministic' or 'none'

    # Console visual styling
    ENABLE_COLOR: bool = True

    def __post_init__(self):
        # Ensure directories exist
        self.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    def set_workspace(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        target.mkdir(parents=True, exist_ok=True)
        self.WORKSPACE_DIR = target
        return self.WORKSPACE_DIR


# Default global configuration instance
config = AgentConfig()


# Terminal styling utilities
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    @classmethod
    def paint(cls, text: str, *styles: str) -> str:
        """Wrap text in ANSI styles if color is enabled."""
        if not config.ENABLE_COLOR:
            return text
        prefix = "".join(styles)
        return f"{prefix}{text}{cls.RESET}"
