"""Evaluation Task Model.

Represents a benchmark evaluation task with metadata, input prompt, project template,
expected behavior, constraints, and machine-checkable assertions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from my_ai_agent.evaluation.assertions import BaseAssertion, build_assertion
from my_ai_agent.evaluation.categories import DatasetSplit, Difficulty, TaskCategory


@dataclass
class EvaluationTask:
    """Structured evaluation task for agent and model benchmark."""
    id: str
    category: TaskCategory
    difficulty: Difficulty
    prompt: str
    project_template: str = "python_basic"
    split: DatasetSplit = DatasetSplit.TEST
    version: str = "1.0"
    expected_behavior: Dict[str, Any] = field(default_factory=dict)
    required_files: List[str] = field(default_factory=list)
    required_symbols: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)
    assertions: List[BaseAssertion] = field(default_factory=list)
    timeout_seconds: int = 60
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes task into a dictionary representation."""
        return {
            "id": self.id,
            "version": self.version,
            "category": self.category.value if isinstance(self.category, TaskCategory) else str(self.category),
            "difficulty": self.difficulty.value if isinstance(self.difficulty, Difficulty) else str(self.difficulty),
            "split": self.split.value if isinstance(self.split, DatasetSplit) else str(self.split),
            "prompt": self.prompt,
            "project_template": self.project_template,
            "expected_behavior": self.expected_behavior,
            "required_files": self.required_files,
            "required_symbols": self.required_symbols,
            "forbidden_actions": self.forbidden_actions,
            "assertions": [a.to_dict() for a in self.assertions],
            "timeout_seconds": self.timeout_seconds,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationTask":
        """Instantiates an EvaluationTask from dictionary configuration."""
        raw_assertions = data.get("assertions", [])
        parsed_assertions: List[BaseAssertion] = []
        for a in raw_assertions:
            if isinstance(a, BaseAssertion):
                parsed_assertions.append(a)
            elif isinstance(a, dict):
                parsed_assertions.append(build_assertion(a))

        return cls(
            id=data["id"],
            category=TaskCategory(data["category"]),
            difficulty=Difficulty(data.get("difficulty", "easy")),
            split=DatasetSplit(data.get("split", "test")),
            version=data.get("version", "1.0"),
            prompt=data["prompt"],
            project_template=data.get("project_template", "python_basic"),
            expected_behavior=data.get("expected_behavior", {}),
            required_files=data.get("required_files", []),
            required_symbols=data.get("required_symbols", []),
            forbidden_actions=data.get("forbidden_actions", []),
            assertions=parsed_assertions,
            timeout_seconds=data.get("timeout_seconds", 60),
            tags=data.get("tags", []),
        )

    def validate(self) -> List[str]:
        """Validates task integrity, ensuring no missing fields or invalid configurations."""
        errors = []
        if not self.id or not isinstance(self.id, str):
            errors.append("Task id must be a non-empty string")
        if not self.prompt or not isinstance(self.prompt, str):
            errors.append(f"Task '{self.id}': Prompt must be a non-empty string")
        if not self.category or not isinstance(self.category, TaskCategory):
            errors.append(f"Task '{self.id}': Invalid category: {self.category}")
        if not self.difficulty or not isinstance(self.difficulty, Difficulty):
            errors.append(f"Task '{self.id}': Invalid difficulty: {self.difficulty}")
        if not self.assertions:
            errors.append(f"Task '{self.id}': Must specify at least one assertion")
        return errors
