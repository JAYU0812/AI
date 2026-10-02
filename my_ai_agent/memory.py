"""Memory module for the AI Agent Engine.

Provides dual-layer memory:
1. Short-term Working Memory: Active session turns, last touched files, recent context.
2. Long-term Persistent Memory: SQLite database recording all interactions and observations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import sqlite3
import uuid

from config import config


@dataclass
class MemoryRecord:
    id: Optional[int]
    session_id: str
    step: int
    user_input: str
    intent: str
    thought: str
    tool_name: str
    tool_args: Dict[str, Any]
    observation: str
    response: str
    created_at: str


class AgentMemory:
    """Manages session history, working context, and SQLite persistent storage."""

    def __init__(self, db_path: Optional[Path] = None, session_id: Optional[str] = None):
        self.db_path = db_path or config.DB_PATH
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.step_counter = 0

        # Short-term working context
        self.short_term_history: List[MemoryRecord] = []
        self.working_context: Dict[str, Any] = {
            "last_file": None,
            "last_command": None,
            "last_tool": None,
            "last_success": None,
        }

        self._init_db()

    def _init_db(self):
        """Initialize SQLite tables for persistent logging."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    step INTEGER,
                    user_input TEXT,
                    intent TEXT,
                    thought TEXT,
                    tool_name TEXT,
                    tool_args TEXT,
                    observation TEXT,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    goal TEXT,
                    intent TEXT,
                    total_steps INTEGER,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    path TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT,
                    role TEXT,
                    text TEXT,
                    result_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS autonomous_tasks (
                    task_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    request TEXT,
                    status TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    attempt INTEGER,
                    max_attempts INTEGER,
                    files_changed TEXT,
                    commands_run INTEGER,
                    tests_run INTEGER,
                    repair_attempts INTEGER,
                    final_summary TEXT,
                    task_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_proj ON chat_messages(project_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_auto_tasks_proj ON autonomous_tasks(project_id)"
            )
            cursor.execute(
                "INSERT OR IGNORE INTO sessions (session_id, description) VALUES (?, ?)",
                (self.session_id, f"Session started at {datetime.now().isoformat()}"),
            )
            conn.commit()

    def record_interaction(
        self,
        user_input: str,
        intent: str,
        thought: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        observation: str,
        response: str,
    ) -> MemoryRecord:
        """Save an interaction step into working memory and persistent SQLite."""
        self.step_counter += 1
        now_iso = datetime.now().isoformat()

        # Update working context
        if "path" in tool_args:
            self.working_context["last_file"] = tool_args["path"]
        if "command" in tool_args:
            self.working_context["last_command"] = tool_args["command"]
        self.working_context["last_tool"] = tool_name

        record = MemoryRecord(
            id=None,
            session_id=self.session_id,
            step=self.step_counter,
            user_input=user_input,
            intent=intent,
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args,
            observation=observation,
            response=response,
            created_at=now_iso,
        )

        # Store in short-term buffer
        self.short_term_history.append(record)

        # Persist to SQLite
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO interactions (
                        session_id, step, user_input, intent, thought, tool_name, tool_args, observation, response
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.session_id,
                        self.step_counter,
                        user_input,
                        intent,
                        thought,
                        tool_name,
                        json.dumps(tool_args),
                        observation,
                        response,
                    ),
                )
                record.id = cursor.lastrowid
                conn.commit()
        except Exception as e:
            print(f"[Memory Warning] Failed to persist interaction: {e}")

        return record

    def get_recent_history(self, limit: int = 5) -> List[MemoryRecord]:
        """Fetch the most recent N interactions from this session or SQLite store."""
        if self.short_term_history:
            return self.short_term_history[-limit:]

        # Fallback to persistent SQLite storage
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, session_id, step, user_input, intent, thought, tool_name, tool_args, observation, response, created_at
                    FROM interactions
                    ORDER BY id DESC LIMIT ?
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                records = []
                for row in reversed(rows):
                    records.append(
                        MemoryRecord(
                            id=row[0],
                            session_id=row[1],
                            step=row[2],
                            user_input=row[3],
                            intent=row[4],
                            thought=row[5],
                            tool_name=row[6],
                            tool_args=json.loads(row[7]),
                            observation=row[8],
                            response=row[9],
                            created_at=row[10],
                        )
                    )
                return records
        except Exception:
            return []

    def get_context(self) -> Dict[str, Any]:
        """Return the current working context for decision-making."""
        return {
            "session_id": self.session_id,
            "current_step": self.step_counter,
            **self.working_context,
        }

    def record_plan(self, goal: str, intent: str, total_steps: int, status: str = "COMPLETED") -> Optional[int]:
        """Persist an execution plan summary to SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO plans (session_id, goal, intent, total_steps, status)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (self.session_id, goal, intent, total_steps, status),
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            print(f"[Memory Warning] Failed to persist plan: {e}")
            return None

    def get_recent_plans(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recently executed plans."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, session_id, goal, intent, total_steps, status, created_at
                    FROM plans ORDER BY id DESC LIMIT ?
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "id": r[0],
                        "session_id": r[1],
                        "goal": r[2],
                        "intent": r[3],
                        "total_steps": r[4],
                        "status": r[5],
                        "created_at": r[6],
                    }
                    for r in rows
                ]
        except Exception:
            return []

    def register_project(self, project_id: str, name: str, path: str, description: str = "") -> Dict[str, Any]:
        """Register or update a user project in SQLite."""
        norm_path = str(Path(path).resolve())
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO projects (id, name, path, description, created_at, last_opened_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name=excluded.name,
                    last_opened_at=excluded.last_opened_at,
                    description=excluded.description
                """,
                (project_id, name, norm_path, description, now, now),
            )
            conn.commit()
        return {"id": project_id, "name": name, "path": norm_path, "description": description}

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all tracked projects ordered by last opened."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, path, created_at, last_opened_at, description FROM projects ORDER BY last_opened_at DESC"
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "name": r[1],
                    "path": r[2],
                    "created_at": r[3],
                    "last_opened_at": r[4],
                    "description": r[5] or "",
                }
                for r in rows
            ]

    def get_project(self, project_id_or_path: str) -> Optional[Dict[str, Any]]:
        """Get project by ID or Path."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, path, created_at, last_opened_at, description FROM projects WHERE id=? OR path=?",
                (project_id_or_path, str(Path(project_id_or_path).resolve())),
            )
            r = cursor.fetchone()
            if not r:
                return None
            return {
                "id": r[0],
                "name": r[1],
                "path": r[2],
                "created_at": r[3],
                "last_opened_at": r[4],
                "description": r[5] or "",
            }

    def delete_project(self, project_id: str):
        """Remove a project from tracked projects."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM projects WHERE id=?", (project_id,))
            conn.commit()

    def touch_project(self, project_id: str):
        """Update last opened timestamp for project."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE projects SET last_opened_at=? WHERE id=?", (now, project_id))
            conn.commit()

    def save_chat_message(
        self,
        project_id: str,
        role: str,
        text: str,
        result_dict: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist a chat message turn to SQLite."""
        res_json = json.dumps(result_dict) if result_dict else None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_messages (project_id, role, text, result_json)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, role, text, res_json),
            )
            conn.commit()
            return cursor.lastrowid

    def get_chat_messages(self, project_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Retrieve stored chat messages for a given project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, project_id, role, text, result_json, created_at
                FROM chat_messages
                WHERE project_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (project_id, limit),
            )
            rows = cursor.fetchall()
            messages = []
            for r in rows:
                parsed_res = None
                if r[4]:
                    try:
                        parsed_res = json.loads(r[4])
                    except Exception:
                        parsed_res = None
                messages.append({
                    "id": r[0],
                    "project_id": r[1],
                    "role": r[2],
                    "text": r[3],
                    "result": parsed_res,
                    "created_at": r[5],
                })
            return messages

    def clear_chat_messages(self, project_id: str):
        """Clear all chat messages for a specific project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_messages WHERE project_id = ?", (project_id,))
            conn.commit()

    def get_chat_message_count(self, project_id: str) -> int:
        """Get total number of messages stored for a project."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chat_messages WHERE project_id = ?", (project_id,))
                row = cursor.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def save_autonomous_task(self, task_dict: Dict[str, Any]):
        """Persist or update an autonomous task record in SQLite."""
        task_id = task_dict.get("task_id")
        project_id = task_dict.get("project_id", "default")
        req = task_dict.get("user_request", "")
        status = task_dict.get("status", "QUEUED")
        started_at = task_dict.get("started_at", "")
        ended_at = task_dict.get("ended_at")
        attempt = task_dict.get("attempt", 1)
        max_attempts = task_dict.get("max_attempts", 5)
        files_changed = json.dumps(task_dict.get("modified_files", []))
        budget = task_dict.get("budget", {})
        cmds_run = budget.get("commands_run", 0)
        repair_att = budget.get("repair_attempts", 0)
        final_summary = task_dict.get("final_summary", "")
        task_json = json.dumps(task_dict)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO autonomous_tasks (
                    task_id, project_id, request, status, started_at, ended_at,
                    attempt, max_attempts, files_changed, commands_run, tests_run,
                    repair_attempts, final_summary, task_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status=excluded.status,
                    ended_at=excluded.ended_at,
                    attempt=excluded.attempt,
                    files_changed=excluded.files_changed,
                    commands_run=excluded.commands_run,
                    repair_attempts=excluded.repair_attempts,
                    final_summary=excluded.final_summary,
                    task_json=excluded.task_json
                """,
                (
                    task_id, project_id, req, status, started_at, ended_at,
                    attempt, max_attempts, files_changed, cmds_run, 0,
                    repair_att, final_summary, task_json
                ),
            )
            conn.commit()

    def get_autonomous_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored autonomous task by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT task_json FROM autonomous_tasks WHERE task_id = ?",
                (task_id,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except Exception:
                    pass
        return None

    def get_autonomous_tasks_for_project(self, project_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve all stored autonomous tasks for a project ordered by time."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT task_id, project_id, request, status, started_at, ended_at,
                       attempt, max_attempts, files_changed, commands_run, tests_run,
                       repair_attempts, final_summary, task_json
                FROM autonomous_tasks
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            )
            rows = cursor.fetchall()
            tasks = []
            for r in rows:
                parsed = None
                if r[13]:
                    try:
                        parsed = json.loads(r[13])
                    except Exception:
                        pass
                if not parsed:
                    parsed = {
                        "task_id": r[0],
                        "project_id": r[1],
                        "user_request": r[2],
                        "status": r[3],
                        "started_at": r[4],
                        "ended_at": r[5],
                        "attempt": r[6],
                        "max_attempts": r[7],
                        "modified_files": json.loads(r[8]) if r[8] else [],
                        "commands_run": r[9],
                        "repair_attempts": r[11],
                        "final_summary": r[12],
                    }
                tasks.append(parsed)
            return tasks

    def clear(self):
        """Clear active working memory and start a new session ID."""
        self.session_id = str(uuid.uuid4())[:8]
        self.step_counter = 0
        self.short_term_history.clear()
        self.working_context = {
            "last_file": None,
            "last_command": None,
            "last_tool": None,
            "last_success": None,
        }
        self._init_db()

