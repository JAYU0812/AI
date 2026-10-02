"""Benchmark Dataset Manager, Loader, and Quality Auditor.

Loads versioned tasks, enforces splits (development, validation, test),
and validates against duplicate IDs, missing assertions, and accidental secrets.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from my_ai_agent.evaluation.categories import DatasetSplit, Difficulty, TaskCategory
from my_ai_agent.evaluation.task import EvaluationTask


DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
PROJECTS_DIR = Path(__file__).resolve().parent / "projects"


class DatasetQualityAuditor:
    """Audits evaluation tasks for integrity, secret exposure, duplicates, and valid templates."""

    FORBIDDEN_PROMPT_SECRETS = ["sk-", "ghp_", "password123", "PRIVATE KEY", "-----BEGIN"]

    @classmethod
    def audit_task(cls, task: EvaluationTask, existing_ids: Set[str]) -> List[str]:
        """Audits a single task returning any quality or security discrepancies."""
        issues = []
        if task.id in existing_ids:
            issues.append(f"Duplicate task id: '{task.id}'")
        
        # Schema checks
        issues.extend(task.validate())

        # Secret audit
        for secret in cls.FORBIDDEN_PROMPT_SECRETS:
            if secret in task.prompt:
                issues.append(f"Task '{task.id}' prompt contains potential secret or credential: '{secret}'")

        # Project template check
        if task.project_template:
            tpl_dir = PROJECTS_DIR / task.project_template
            if not tpl_dir.exists() or not tpl_dir.is_dir():
                issues.append(f"Task '{task.id}': Project template directory '{task.project_template}' does not exist")

        return issues

    @classmethod
    def audit_dataset(cls, tasks: List[EvaluationTask]) -> Dict[str, Any]:
        """Audits an entire collection of tasks."""
        existing_ids: Set[str] = set()
        all_issues: Dict[str, List[str]] = {}

        for task in tasks:
            task_issues = cls.audit_task(task, existing_ids)
            if task_issues:
                all_issues[task.id] = task_issues
            existing_ids.add(task.id)

        is_valid = len(all_issues) == 0
        return {
            "valid": is_valid,
            "total_tasks": len(tasks),
            "issues": all_issues,
        }


class BenchmarkDataset:
    """Manages versioned benchmark tasks and querying."""

    def __init__(self, version: str = "v1", dataset_dir: Optional[Path] = None):
        self.version = version
        self.dataset_dir = (dataset_dir or (DATASETS_DIR / version)).resolve()
        self.tasks: List[EvaluationTask] = []
        self._load_all_tasks()

    def _load_all_tasks(self) -> None:
        """Loads tasks from all JSON files in the dataset directory."""
        self.tasks.clear()
        if not self.dataset_dir.exists():
            return

        for p in self.dataset_dir.rglob("*.json"):
            try:
                content = p.read_text(encoding="utf-8")
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        self.tasks.append(EvaluationTask.from_dict(item))
                elif isinstance(data, dict):
                    if "tasks" in data and isinstance(data["tasks"], list):
                        for item in data["tasks"]:
                            self.tasks.append(EvaluationTask.from_dict(item))
                    else:
                        self.tasks.append(EvaluationTask.from_dict(data))
            except Exception as e:
                print(f"[Dataset] Failed to load {p.name}: {e}")

    def filter(
        self,
        category: Optional[str | TaskCategory] = None,
        difficulty: Optional[str | Difficulty] = None,
        split: Optional[str | DatasetSplit] = None,
        task_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[EvaluationTask]:
        """Filters dataset tasks according to criteria."""
        results = self.tasks

        if task_id:
            return [t for t in results if t.id == task_id]

        if category:
            cat_val = category.value if isinstance(category, TaskCategory) else str(category)
            results = [t for t in results if (t.category.value if isinstance(t.category, TaskCategory) else str(t.category)) == cat_val]

        if difficulty:
            diff_val = difficulty.value if isinstance(difficulty, Difficulty) else str(difficulty)
            results = [t for t in results if (t.difficulty.value if isinstance(t.difficulty, Difficulty) else str(t.difficulty)) == diff_val]

        if split:
            split_val = split.value if isinstance(split, DatasetSplit) else str(split)
            results = [t for t in results if (t.split.value if isinstance(t.split, DatasetSplit) else str(t.split)) == split_val]

        if tag:
            results = [t for t in results if tag in t.tags]

        return results

    def save_task(self, task: EvaluationTask, subcategory: Optional[str] = None) -> Path:
        """Saves a task to disk in the appropriate category directory."""
        cat_dir = self.dataset_dir / (subcategory or (task.category.value if isinstance(task.category, TaskCategory) else str(task.category)))
        cat_dir.mkdir(parents=True, exist_ok=True)
        file_path = cat_dir / f"{task.id}.json"
        file_path.write_text(json.dumps(task.to_dict(), indent=2), encoding="utf-8")
        # Update in-memory tasks list
        self.tasks = [t for t in self.tasks if t.id != task.id]
        self.tasks.append(task)
        return file_path

    def audit(self) -> Dict[str, Any]:
        """Performs a comprehensive quality audit of all tasks."""
        return DatasetQualityAuditor.audit_dataset(self.tasks)
