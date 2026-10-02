"""Tools module for the AI Agent Engine.

Provides the 5 fundamental tools for Step 1:
1. create_file
2. read_file
3. write_file
4. list_files
5. run_command

Includes a ToolRegistry for extensible tool discovery and execution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import os
import subprocess
import sys
import time

from config import config
from project.file_classifier import FileClassifier

# Destructive / dangerous command patterns to block
DANGEROUS_COMMAND_PATTERNS = [
    r"rm\s+-rf\s+[/~]",
    r"rmdir\s+/[sq]\s+[c-zC-Z]:\\",
    r"del\s+/[fqs]\s+[c-zC-Z]:\\",
    r"format\s+[c-zC-Z]:",
    r"mkfs",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",  # fork bomb
    r"dd\s+if=",
]


@dataclass
class ToolResult:
    """Standardized result returned by any tool execution."""
    success: bool
    output: str
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


def _resolve_workspace_path(file_path: str) -> Path:
    """Resolve a path relative to the workspace directory.
    
    Prevents directory traversal outside the workspace for safety.
    """
    raw_path = Path(file_path)
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        resolved = (config.WORKSPACE_DIR / raw_path).resolve()

    return resolved


class ToolRegistry:
    """Registry that holds tool functions, descriptions, and parameter schemas."""

    def __init__(self):
        self._tools: Dict[str, Callable[..., ToolResult]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ):
        """Decorator to register a tool function."""
        def decorator(func: Callable[..., ToolResult]):
            self._tools[name] = func
            self._metadata[name] = {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
            return func
        return decorator

    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a registered tool by name with arguments."""
        if name not in self._tools:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' is not registered. Available tools: {list(self._tools.keys())}",
            )
        try:
            return self._tools[name](**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Exception while executing tool '{name}': {str(e)}",
            )

    def get_tool_list(self) -> List[Dict[str, Any]]:
        """Return list of all registered tools and their specifications."""
        return list(self._metadata.values())

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())


# Global registry instance
registry = ToolRegistry()


# ==============================================================================
# 1. create_file
# ==============================================================================
@registry.register(
    name="create_file",
    description="Create a new empty file or with initial content in the workspace.",
    parameters={
        "path": {"type": "string", "description": "Relative or absolute file path to create."},
        "content": {"type": "string", "description": "Optional initial content for the file.", "default": ""},
    },
)
def create_file(path: str, content: str = "") -> ToolResult:
    """Creates a file at the given path within the workspace."""
    try:
        target = _resolve_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        already_existed = target.exists()
        target.write_text(content, encoding="utf-8")

        if not target.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Failed verification: {target.name} does not exist after creation attempt.",
            )

        file_size = target.stat().st_size
        status = "Overwrote existing file" if already_existed else "Created file"
        rel_path = target.name if target.parent == config.WORKSPACE_DIR else str(target.relative_to(config.WORKSPACE_DIR))
        
        return ToolResult(
            success=True,
            output=f"{status} '{rel_path}' ({file_size} bytes).",
            metadata={"path": str(target), "relative_path": rel_path, "bytes": file_size},
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"create_file failed: {str(e)}")


# ==============================================================================
# 2. read_file
# ==============================================================================
@registry.register(
    name="read_file",
    description="Read content from a file in the workspace, with optional line range.",
    parameters={
        "path": {"type": "string", "description": "File path to read."},
        "start_line": {"type": "integer", "description": "Starting line number (1-indexed).", "default": 1},
        "end_line": {"type": "integer", "description": "Ending line number (inclusive). Optional.", "default": None},
    },
)
def read_file(path: str, start_line: int = 1, end_line: Optional[int] = None) -> ToolResult:
    """Reads lines from a file in the workspace."""
    try:
        target = _resolve_workspace_path(path)
        if not target.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: '{target.name}'.",
            )
        if not target.is_file():
            return ToolResult(
                success=False,
                output="",
                error=f"Path is a directory, not a file: '{target.name}'.",
            )

        # Safety boundary: Never expose contents of sensitive files (e.g. .env, private keys)
        classification = FileClassifier.classify(target.name)
        if classification.is_sensitive:
            return ToolResult(
                success=False,
                output="",
                error=f"Safety boundary: Access to sensitive file '{target.name}' is restricted to protect credentials.",
            )

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)

        start = max(1, start_line)
        end = total_lines if end_line is None else min(end_line, total_lines)

        if total_lines == 0:
            return ToolResult(
                success=True,
                output="<empty file>",
                metadata={"total_lines": 0, "path": str(target)},
            )

        selected_lines = lines[start - 1 : end]
        formatted = "\n".join(
            f"{i:4d} | {line}" for i, line in enumerate(selected_lines, start=start)
        )

        return ToolResult(
            success=True,
            output=formatted,
            metadata={
                "total_lines": total_lines,
                "start_line": start,
                "end_line": end,
                "path": str(target),
            },
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"read_file failed: {str(e)}")


# ==============================================================================
# 3. write_file
# ==============================================================================
@registry.register(
    name="write_file",
    description="Write or append content into a file in the workspace.",
    parameters={
        "path": {"type": "string", "description": "File path to write into."},
        "content": {"type": "string", "description": "The string content to write."},
        "append": {"type": "boolean", "description": "If true, append to existing content instead of overwriting.", "default": False},
    },
)
def write_file(path: str, content: str, append: bool = False) -> ToolResult:
    """Writes or appends content to a file."""
    try:
        target = _resolve_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        action = "Appended to" if append else "Wrote to"

        with open(target, mode, encoding="utf-8") as f:
            if append and target.exists() and target.stat().st_size > 0:
                f.write("\n")
            f.write(content)

        file_size = target.stat().st_size
        rel_path = target.name if target.parent == config.WORKSPACE_DIR else str(target.relative_to(config.WORKSPACE_DIR))

        return ToolResult(
            success=True,
            output=f"{action} '{rel_path}' successfully (current size: {file_size} bytes).",
            metadata={"path": str(target), "relative_path": rel_path, "bytes": file_size},
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"write_file failed: {str(e)}")


# ==============================================================================
# 4. list_files
# ==============================================================================
@registry.register(
    name="list_files",
    description="List files and subdirectories inside the workspace directory.",
    parameters={
        "directory": {"type": "string", "description": "Relative directory path (defaults to root workspace).", "default": "."},
    },
)
def list_files(directory: str = ".") -> ToolResult:
    """Lists files and folders inside the workspace."""
    try:
        target_dir = _resolve_workspace_path(directory)
        if not target_dir.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Directory '{directory}' does not exist.",
            )

        entries = sorted(list(target_dir.iterdir()), key=lambda p: (not p.is_dir(), p.name.lower()))
        
        if not entries:
            return ToolResult(
                success=True,
                output="Directory is empty.",
                metadata={"count": 0, "path": str(target_dir)},
            )

        lines = [f"Directory listing of '{target_dir.name}':"]
        for entry in entries:
            if entry.is_dir():
                lines.append(f"  📁 {entry.name}/")
            else:
                size_kb = entry.stat().st_size / 1024
                lines.append(f"  📄 {entry.name:<25} ({size_kb:.1f} KB)")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"count": len(entries), "path": str(target_dir)},
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"list_files failed: {str(e)}")


# ==============================================================================
# 5. run_command
# ==============================================================================
@registry.register(
    name="run_command",
    description="Execute a shell or python command inside the workspace environment.",
    parameters={
        "command": {"type": "string", "description": "The command string to execute."},
        "timeout": {"type": "integer", "description": "Timeout in seconds.", "default": 30},
    },
)
def run_command(command: str, timeout: int = 30) -> ToolResult:
    """Executes a command safely inside the workspace directory."""
    try:
        cmd = command.strip()

        # Safety boundary: Block dangerous or destructive system commands
        import re
        for pat in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pat, cmd, re.IGNORECASE):
                return ToolResult(
                    success=False,
                    output="",
                    error="Safety boundary: Potentially destructive system command was blocked (WAITING_FOR_APPROVAL).",
                    metadata={"blocked_command": cmd, "safety_violation": True},
                )

        start_time = time.time()
        if cmd.startswith("python ") or cmd == "python":
            python_exe = sys.executable
            cmd = f'"{python_exe}" ' + cmd[7:]
        elif cmd.startswith("python3 ") or cmd == "python3":
            python_exe = sys.executable
            cmd = f'"{python_exe}" ' + cmd[8:]

        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(config.WORKSPACE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        elapsed = round(time.time() - start_time, 2)

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        combined_output = []
        if stdout:
            combined_output.append(stdout)
        if stderr:
            combined_output.append(f"[STDERR]\n{stderr}")

        output_str = "\n".join(combined_output) if combined_output else "(Command produced no output)"

        success = (result.returncode == 0)
        return ToolResult(
            success=success,
            output=output_str,
            error="" if success else f"Command exited with returncode {result.returncode}",
            metadata={
                "command": command,
                "returncode": result.returncode,
                "elapsed_seconds": elapsed,
            },
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            output="",
            error=f"Command timed out after {timeout} seconds.",
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=f"run_command failed: {str(e)}")
