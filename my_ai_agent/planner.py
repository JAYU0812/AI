"""Multi-Step Planning & Task State Engine for Step 2.

Decomposes high-level compound user goals into sequential tool steps,
manages task execution states, and coordinates context passing between tools.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from nlp import ExtractedEntities, IntentType, NLUResult
from synthesizer import CodeSynthesizer
from tools import ToolResult


class PlanState(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class PlanStep:
    step_num: int
    action_type: str
    tool_name: str
    tool_args: Dict[str, Any]
    description: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[ToolResult] = None
    observation: str = ""


@dataclass
class ExecutionPlan:
    goal: str
    intent: IntentType
    entities: ExtractedEntities
    steps: List[PlanStep] = field(default_factory=list)
    state: PlanState = PlanState.PENDING
    current_index: int = 0
    final_output: str = ""

    def add_step(self, action_type: str, tool_name: str, tool_args: Dict[str, Any], description: str):
        step_num = len(self.steps) + 1
        step = PlanStep(
            step_num=step_num,
            action_type=action_type,
            tool_name=tool_name,
            tool_args=tool_args,
            description=description,
        )
        self.steps.append(step)


class MultiStepPlanner:
    """Decomposes compound goals into an executable sequence of tool invocations."""

    def __init__(self, synthesizer: Optional[CodeSynthesizer] = None):
        self.synthesizer = synthesizer or CodeSynthesizer()

    def create_plan(self, nlu: NLUResult, working_context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Constructs an ExecutionPlan based on classified intent and extracted entities."""
        intent = nlu.intent
        entities = nlu.entities
        plan = ExecutionPlan(
            goal=nlu.raw_input,
            intent=intent,
            entities=entities,
        )

        # -------------------------------------------------------------
        # 1. Compound: BUILD_AND_RUN
        # (e.g. "Make me a Python file that calculates average of 3 numbers and run it")
        # -------------------------------------------------------------
        if intent == IntentType.BUILD_AND_RUN:
            fn = entities.filename or "task_script.py"
            code = self.synthesizer.synthesize(entities)

            # Step 1: Create file
            plan.add_step(
                action_type="CREATE_FILE",
                tool_name="create_file",
                tool_args={"path": fn, "content": ""},
                description=f"Create file '{fn}' in workspace",
            )
            # Step 2: Write synthesized code
            plan.add_step(
                action_type="WRITE_FILE",
                tool_name="write_file",
                tool_args={"path": fn, "content": code, "append": False},
                description=f"Synthesize and write {entities.language} code for '{entities.purpose}' into '{fn}'",
            )
            # Step 3: Run command
            cmd = f"python {fn}"
            plan.add_step(
                action_type="RUN_COMMAND",
                tool_name="run_command",
                tool_args={"command": cmd},
                description=f"Execute generated script '{fn}' in workspace",
            )
            return plan

        # -------------------------------------------------------------
        # 2. Compound: BUILD_FILE (without auto-run)
        # -------------------------------------------------------------
        if intent == IntentType.BUILD_FILE:
            fn = entities.filename or "task_script.py"
            code = self.synthesizer.synthesize(entities)

            plan.add_step(
                action_type="CREATE_FILE",
                tool_name="create_file",
                tool_args={"path": fn, "content": ""},
                description=f"Create file '{fn}' in workspace",
            )
            plan.add_step(
                action_type="WRITE_FILE",
                tool_name="write_file",
                tool_args={"path": fn, "content": code, "append": False},
                description=f"Write code for '{entities.purpose}' into '{fn}'",
            )
            return plan

        # -------------------------------------------------------------
        # 3. Compound: REFACTOR_FILE (Step 3B — Read before modify)
        # -------------------------------------------------------------
        if intent == IntentType.REFACTOR_FILE:
            fn = entities.filename or (working_context.get("last_file") if working_context else "main.py")
            # Step 1: Read existing code first
            plan.add_step(
                action_type="READ_FILE",
                tool_name="read_file",
                tool_args={"path": fn},
                description=f"Read existing source code from '{fn}'",
            )
            # Step 2: Write extended code
            plan.add_step(
                action_type="WRITE_FILE",
                tool_name="write_file",
                tool_args={"path": fn, "content": "", "append": False},
                description=f"Refactor and update '{fn}' with '{entities.purpose}'",
            )
            # Step 3: Run / verify if Python
            if fn.endswith(".py"):
                cmd = f"python {fn}"
                plan.add_step(
                    action_type="RUN_COMMAND",
                    tool_name="run_command",
                    tool_args={"command": cmd},
                    description=f"Execute refactored script '{fn}' to verify changes",
                )
            return plan

        # -------------------------------------------------------------
        # 4. UNDERSTAND_PROJECT (Step 3B — Factual inquiry)
        # -------------------------------------------------------------
        if intent == IntentType.UNDERSTAND_PROJECT:
            plan.add_step(
                action_type="UNDERSTAND_PROJECT",
                tool_name="list_files",
                tool_args={"directory": "."},
                description=f"Inspect project map and examine '{entities.purpose}'",
            )
            return plan

        # -------------------------------------------------------------
        # 3. Atomic Intents
        # -------------------------------------------------------------
        if intent == IntentType.CREATE_FILE:
            fn = entities.filename or "unnamed.txt"
            plan.add_step(
                action_type="CREATE_FILE",
                tool_name="create_file",
                tool_args={"path": fn, "content": ""},
                description=f"Create file '{fn}'",
            )
            return plan

        if intent == IntentType.WRITE_FILE:
            fn = entities.filename or (working_context.get("last_file") if working_context else "unnamed.txt")
            content = entities.raw_slots.get("content", "")
            plan.add_step(
                action_type="WRITE_FILE",
                tool_name="write_file",
                tool_args={"path": fn, "content": content, "append": False},
                description=f"Write content to '{fn}'",
            )
            return plan

        if intent == IntentType.READ_FILE:
            fn = entities.filename or (working_context.get("last_file") if working_context else "unnamed.txt")
            plan.add_step(
                action_type="READ_FILE",
                tool_name="read_file",
                tool_args={"path": fn},
                description=f"Read contents of '{fn}'",
            )
            return plan

        if intent == IntentType.LIST_FILES:
            plan.add_step(
                action_type="LIST_FILES",
                tool_name="list_files",
                tool_args={"directory": "."},
                description="List files in workspace directory",
            )
            return plan

        if intent == IntentType.RUN_COMMAND:
            cmd = entities.raw_slots.get("command", f"python {entities.filename or ''}")
            plan.add_step(
                action_type="RUN_COMMAND",
                tool_name="run_command",
                tool_args={"command": cmd},
                description=f"Run command '{cmd}'",
            )
            return plan

        # Non-tool intents return empty steps
        return plan
