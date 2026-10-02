"""my_ai_agent package initialization.

Ensures the agent directory is automatically on sys.path for internal module imports.
"""

from pathlib import Path
import sys

_agent_dir = Path(__file__).resolve().parent
if str(_agent_dir) not in sys.path:
    sys.path.insert(0, str(_agent_dir))
