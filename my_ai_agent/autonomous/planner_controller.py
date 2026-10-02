"""Planner Controller for Step 3D.

Orchestrates:
1. Initial structured plan generation from Step 3B Project Understanding.
2. Dynamic plan adaptation based on real observations (skipping redundant steps, injecting repairs).
3. Risk classification (LOW, MEDIUM, HIGH) on each planned action.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from my_ai_agent.autonomous.action import ActionRisk, ActionType, AutonomousAction
from my_ai_agent.autonomous.observation import ActionObservation
from my_ai_agent.project.project_map import ProjectMap
from my_ai_agent.project.project_understanding import ProjectUnderstandingEngine


class PlannerController:
    """Creates initial plans and dynamically adapts them to project realities."""

    def __init__(self, project_engine: Optional[ProjectUnderstandingEngine] = None):
        self.project_engine = project_engine or ProjectUnderstandingEngine()

    def create_initial_plan(
        self,
        user_request: str,
        workspace_root: Path,
        pmap: Optional[ProjectMap] = None,
    ) -> List[AutonomousAction]:
        """Formulate a structured list of actions using project understanding."""
        root = Path(workspace_root).resolve()
        if pmap is None:
            pmap = self.project_engine.understand_project(root)

        rel_ctx = self.project_engine.get_relevant_context(root, user_request)
        req_lower = user_request.lower()
        actions: List[AutonomousAction] = []

        # 1. First step: Inspect project & understand relevant files
        actions.append(AutonomousAction(
            type=ActionType.INSPECT_PROJECT,
            target=str(root.name),
            reason=f"Understand architecture and relevant context for '{user_request}'",
            risk=ActionRisk.LOW,
        ))

        # Scenario A: "Fix the failing tests" or bug-fixing request
        if any(k in req_lower for k in ("fix", "failing test", "repair", "debug", "broken")):
            actions.append(AutonomousAction(
                type=ActionType.RUN_TESTS,
                target="tests",
                reason="Execute test suite to observe and pinpoint current test failures",
                risk=ActionRisk.LOW,
            ))
            actions.append(AutonomousAction(
                type=ActionType.VERIFY,
                target="project",
                reason="Verify that bug is resolved and all tests pass",
                risk=ActionRisk.LOW,
            ))
            return actions

        # Scenario B: Feature Addition / Multi-function implementation (e.g. calculator or API)
        target_src_file = None
        target_test_file = None

        if rel_ctx.relevant_files:
            for rf in rel_ctx.relevant_files:
                if "test" in rf.path.lower():
                    if not target_test_file:
                        target_test_file = rf.path
                else:
                    if not target_src_file:
                        target_src_file = rf.path

        # If not found from context, infer reasonable filenames
        if not target_src_file:
            if "calculator" in req_lower:
                target_src_file = "calculator.py"
            elif "cli" in req_lower or "task" in req_lower:
                target_src_file = "tasks.py"
            elif "api" in req_lower or "app" in req_lower:
                target_src_file = "app.py"
            else:
                target_src_file = pmap.primary_entry_point or "main.py"

        if not target_test_file:
            base_stem = Path(target_src_file).stem
            target_test_file = f"tests/test_{base_stem}.py"

        # Action: Modify or Create Source File
        src_exists = (root / target_src_file).exists()
        actions.append(AutonomousAction(
            type=ActionType.PATCH_FILE if src_exists else ActionType.CREATE_FILE,
            target=target_src_file,
            reason=f"Implement requested functionality in {target_src_file}",
            risk=ActionRisk.LOW,
            params={"feature_request": user_request},
        ))

        # Action: Add / Update Unit Tests
        test_exists = (root / target_test_file).exists()
        actions.append(AutonomousAction(
            type=ActionType.PATCH_FILE if test_exists else ActionType.CREATE_FILE,
            target=target_test_file,
            reason=f"Create/update automated unit tests in {target_test_file}",
            risk=ActionRisk.LOW,
            params={"feature_request": user_request, "is_test_file": True},
        ))

        # Action: Execute Tests
        actions.append(AutonomousAction(
            type=ActionType.RUN_TESTS,
            target=target_test_file,
            reason="Execute tests to verify new implementation",
            risk=ActionRisk.LOW,
        ))

        # Action: Completion Verification
        actions.append(AutonomousAction(
            type=ActionType.VERIFY,
            target="project",
            reason="Perform final evidence-based completion verification",
            risk=ActionRisk.LOW,
        ))

        return actions

    def adapt_plan(
        self,
        remaining_actions: List[AutonomousAction],
        observation: ActionObservation,
        workspace_root: Path,
    ) -> List[AutonomousAction]:
        """Adapt upcoming actions dynamically based on latest action observation."""
        adapted = list(remaining_actions)

        # 1. If RUN_TESTS failed: inject a DIAGNOSE and REPAIR action before re-testing
        if observation.action_type == ActionType.RUN_TESTS and not observation.success:
            # Check if there is already an upcoming repair action
            has_repair = any(a.type == ActionType.PATCH_FILE and "repair" in a.reason.lower() for a in adapted)
            if not has_repair:
                # Prepend repair action and follow-up re-test action
                repair_action = AutonomousAction(
                    type=ActionType.PATCH_FILE,
                    target="auto_repair",
                    reason="Repair code based on test failure diagnosis",
                    risk=ActionRisk.LOW,
                    params={"is_repair": True, "error_output": observation.output or observation.error},
                )
                retest_action = AutonomousAction(
                    type=ActionType.RUN_TESTS,
                    target="tests",
                    reason="Re-test project to verify corrective repair",
                    risk=ActionRisk.LOW,
                )
                adapted.insert(0, retest_action)
                adapted.insert(0, repair_action)

        # 2. If an action discovered dependency already present, skip redundant install
        if observation.action_type == ActionType.INSPECT_PROJECT and observation.success:
            # E.g. If dependency step is in plan but already exists in manifest
            pass

        return adapted
