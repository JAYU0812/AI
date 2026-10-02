"""Central Autonomous Execution Loop for Step 3D.

Orchestrates the complete autonomous cycle:
Understand ➔ Plan ➔ Act ➔ Observe ➔ Reason ➔ Test ➔ Diagnose ➔ Repair ➔ Verify ➔ Complete

Every step is observable and emits real AgentEvents.
Enforces cancellation, pause/resume on user approval, and task budgets.
"""

from pathlib import Path
import threading
import time
from typing import Dict, Optional

from my_ai_agent.autonomous.action import ActionRisk, ActionType, AutonomousAction
from my_ai_agent.autonomous.action_executor import ActionExecutor
from my_ai_agent.autonomous.completion import CompletionReport, CompletionVerifier
from my_ai_agent.autonomous.diagnosis import DiagnosisEngine, DiagnosticReport
from my_ai_agent.autonomous.events import AgentEvent, AgentEventType, broadcaster
from my_ai_agent.autonomous.observation import ActionObservation
from my_ai_agent.autonomous.planner_controller import PlannerController
from my_ai_agent.autonomous.repair import RepairEngine, RepairResult
from my_ai_agent.autonomous.task import AutonomousTask
from my_ai_agent.autonomous.task_state import TaskStatus
from my_ai_agent.editing.patch_engine import PatchEngine
from my_ai_agent.project.project_understanding import ProjectUnderstandingEngine


class AutonomousExecutionEngine:
    """Coordinates autonomous end-to-end task execution."""

    def __init__(
        self,
        workspace_root: Path,
        project_engine: Optional[ProjectUnderstandingEngine] = None,
        patch_engine: Optional[PatchEngine] = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.project_engine = project_engine or ProjectUnderstandingEngine()
        self.patch_engine = patch_engine or PatchEngine(workspace_root=self.workspace_root)
        self.planner = PlannerController(project_engine=self.project_engine)
        self.executor = ActionExecutor(
            workspace_root=self.workspace_root,
            patch_engine=self.patch_engine,
            project_engine=self.project_engine,
        )

        # Active tasks registry
        self._active_tasks: Dict[str, AutonomousTask] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._lock = threading.Lock()

    def get_task(self, task_id: str) -> Optional[AutonomousTask]:
        with self._lock:
            return self._active_tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Signals task cancellation."""
        with self._lock:
            task = self._active_tasks.get(task_id)
            if not task or task.is_terminal():
                return False
            self._cancel_flags[task_id] = True

        if task.status != TaskStatus.CANCELLED:
            task.transition_to(TaskStatus.CANCELLED, reason="Cancelled by user instruction")
            broadcaster.emit(AgentEvent(
                task_id=task.task_id,
                event_type=AgentEventType.TASK_CANCELLED,
                message="Task cancelled by user. Work halted safely.",
            ))
        return True

    def approve_action(self, task_id: str) -> bool:
        """Approves a paused action in WAITING_FOR_APPROVAL and resumes execution."""
        task = self.get_task(task_id)
        if not task or task.status != TaskStatus.WAITING_FOR_APPROVAL or not task.pending_approval_action:
            return False

        action = task.pending_approval_action
        action.approved = True
        action.status = "APPROVED"
        task.pending_approval_action = None

        broadcaster.emit(AgentEvent(
            task_id=task.task_id,
            event_type=AgentEventType.APPROVAL_GRANTED,
            message=f"Action '{action.type.value}' on '{action.target}' approved by user.",
            metadata={"action_id": action.id},
        ))

        task.transition_to(TaskStatus.EXECUTING, reason="User approved pending action")
        return True

    def reject_action(self, task_id: str, reason: str = "") -> bool:
        """Rejects a paused action in WAITING_FOR_APPROVAL."""
        task = self.get_task(task_id)
        if not task or task.status != TaskStatus.WAITING_FOR_APPROVAL or not task.pending_approval_action:
            return False

        action = task.pending_approval_action
        action.status = "REJECTED"
        task.pending_approval_action = None

        broadcaster.emit(AgentEvent(
            task_id=task.task_id,
            event_type=AgentEventType.APPROVAL_REJECTED,
            message=f"Action '{action.type.value}' rejected by user. {reason}".strip(),
            metadata={"action_id": action.id},
        ))

        task.transition_to(TaskStatus.FAILED, reason=f"High-risk action rejected by user: {reason or 'Denied'}")
        return True

    def retry_task(self, task_id: str) -> Optional[AutonomousTask]:
        """Restart or continue a failed/cancelled task."""
        task = self.get_task(task_id)
        if not task:
            return None

        if task.attempt >= task.max_attempts:
            return task

        task.attempt += 1
        with self._lock:
            self._cancel_flags[task_id] = False

        broadcaster.emit(AgentEvent(
            task_id=task.task_id,
            event_type=AgentEventType.RETRYING,
            message=f"Retrying task (attempt {task.attempt}/{task.max_attempts})...",
            metadata={"attempt": task.attempt},
        ))

        task.transition_to(TaskStatus.PLANNING, reason=f"Retry attempt {task.attempt}")
        return self.execute_task(task)

    # ── Main Autonomous Execution Driver ──────────────────────────

    def run_task(self, user_request: str, project_id: str = "default") -> AutonomousTask:
        """Initialize and run an autonomous task end-to-end."""
        task = AutonomousTask(user_request=user_request, project_id=project_id)
        with self._lock:
            self._active_tasks[task.task_id] = task
            self._cancel_flags[task.task_id] = False

        broadcaster.emit(AgentEvent(
            task_id=task.task_id,
            event_type=AgentEventType.TASK_STARTED,
            message=f"Autonomous task initialized: '{user_request}'",
            metadata={"project_id": project_id},
        ))

        return self.execute_task(task)

    def execute_task(self, task: AutonomousTask) -> AutonomousTask:
        """The core step-by-step reasoning and action loop."""
        task_id = task.task_id

        # Phase 1: Understand Project (Step 3B)
        if task.status == TaskStatus.QUEUED:
            task.transition_to(TaskStatus.UNDERSTANDING, reason="Analyzing project architecture")
            broadcaster.emit(AgentEvent(
                task_id=task_id,
                event_type=AgentEventType.UNDERSTANDING_PROJECT,
                message="Inspecting project structure, dependencies, and symbols...",
            ))
            pmap = self.project_engine.understand_project(self.workspace_root)
            broadcaster.emit(AgentEvent(
                task_id=task_id,
                event_type=AgentEventType.PROJECT_ANALYZED,
                message=f"Identified {pmap.primary_type} project with {pmap.total_files_count} files.",
                metadata={"primary_type": pmap.primary_type, "entry_point": pmap.primary_entry_point},
            ))
            task.transition_to(TaskStatus.PLANNING, reason="Project analyzed, generating plan")

        # Phase 2: Formulate Initial Plan
        if task.status == TaskStatus.PLANNING:
            plan = self.planner.create_initial_plan(task.user_request, self.workspace_root)
            task.plan = plan
            broadcaster.emit(AgentEvent(
                task_id=task_id,
                event_type=AgentEventType.PLAN_CREATED,
                message=f"Plan formulated with {len(plan)} actions.",
                metadata={"plan": [a.to_dict() for a in plan]},
            ))
            task.transition_to(TaskStatus.EXECUTING, reason="Executing action plan")

        # Phase 3: Action ➔ Observe ➔ Reason ➔ Adapt Loop
        while not task.is_terminal():
            # 1. Check cancellation
            if self._cancel_flags.get(task_id, False):
                if task.status != TaskStatus.CANCELLED:
                    task.transition_to(TaskStatus.CANCELLED, reason="Task cancelled")
                break

            # 2. Check operational budget
            budget_err = task.budget.check_budget()
            if budget_err:
                broadcaster.emit(AgentEvent(
                    task_id=task_id,
                    event_type=AgentEventType.TASK_FAILED,
                    message=f"Task halted: {budget_err}",
                    metadata={"budget": task.budget.to_dict()},
                ))
                task.final_summary = f"Task stopped by operational ceiling: {budget_err}"
                task.transition_to(TaskStatus.FAILED, reason=budget_err)
                break

            # 3. Check if any actions remain
            if not task.plan:
                # If plan empty, perform final verification
                self._run_final_verification(task)
                break

            current_action: AutonomousAction = task.plan.pop(0)

            # 4. Check approval boundary
            if current_action.requires_approval and not current_action.approved:
                task.pending_approval_action = current_action
                task.plan.insert(0, current_action)  # Put back for resumption
                task.transition_to(TaskStatus.WAITING_FOR_APPROVAL, reason=f"Action '{current_action.type.value}' requires user approval")
                broadcaster.emit(AgentEvent(
                    task_id=task_id,
                    event_type=AgentEventType.APPROVAL_REQUESTED,
                    message=f"Approval requested for high-risk action: {current_action.reason} ({current_action.target})",
                    metadata={"action": current_action.to_dict()},
                ))
                # Pause execution loop until approval endpoint called
                return task

            # 5. Emit Action Started
            broadcaster.emit(AgentEvent(
                task_id=task_id,
                event_type=AgentEventType.ACTION_STARTED,
                message=f"Executing: {current_action.reason} [{current_action.type.value}]",
                metadata={"action": current_action.to_dict()},
            ))

            # 6. Record step in budget
            task.budget.record_step()
            if current_action.type == ActionType.RUN_COMMAND:
                task.budget.record_command()

            # Pass modified files context into action if needed
            current_action.params["modified_files"] = list(task.modified_files)

            # 7. Execute Action
            obs: ActionObservation = self.executor.execute_action(current_action)
            task.observations.append(obs)
            current_action.status = "COMPLETED" if obs.success else "FAILED"
            task.completed_actions.append(current_action)

            # Track modified files
            for f in obs.modified_files:
                task.modified_files.add(f)
                task.budget.record_file_changed(f)

            # 8. Emit Action Completed / Test results
            if current_action.type == ActionType.RUN_TESTS:
                ev_type = AgentEventType.TEST_PASSED if obs.success else AgentEventType.TEST_FAILED
                broadcaster.emit(AgentEvent(
                    task_id=task_id,
                    event_type=ev_type,
                    message=f"Tests {'passed' if obs.success else 'failed'} (exit code {obs.exit_code})",
                    metadata={"output": obs.output[:300]},
                ))
            else:
                broadcaster.emit(AgentEvent(
                    task_id=task_id,
                    event_type=AgentEventType.ACTION_COMPLETED,
                    message=obs.output or f"Action {current_action.type.value} completed",
                    metadata={"diffs_count": len(obs.diffs)},
                ))

            # 9. Handle Failure ➔ Error ➔ Diagnosis ➔ Repair loop
            if not obs.success:
                # If tests or command failed, transition to DIAGNOSING & REPAIRING
                if current_action.type in (ActionType.RUN_TESTS, ActionType.RUN_COMMAND):
                    task.transition_to(TaskStatus.DIAGNOSING, reason="Action encountered failure, analyzing diagnosis")
                    diagnosis: DiagnosticReport = DiagnosisEngine.diagnose(
                        error_output=obs.output or obs.error,
                        workspace_root=self.workspace_root,
                    )
                    broadcaster.emit(AgentEvent(
                        task_id=task_id,
                        event_type=AgentEventType.ERROR_DIAGNOSED,
                        message=f"Diagnosed {diagnosis.category.value}: {diagnosis.error_message}",
                        metadata=diagnosis.to_dict(),
                    ))

                    # Attempt Repair if under budget
                    if task.budget.repair_attempts < task.budget.max_repair_attempts:
                        task.budget.record_repair()
                        task.transition_to(TaskStatus.REPAIRING, reason=f"Applying repair {task.budget.repair_attempts}/{task.budget.max_repair_attempts}")
                        broadcaster.emit(AgentEvent(
                            task_id=task_id,
                            event_type=AgentEventType.REPAIR_STARTED,
                            message=f"Synthesizing corrective patch for {diagnosis.affected_file or 'project'}...",
                            metadata={"attempt": task.budget.repair_attempts},
                        ))

                        repair_res: RepairResult = RepairEngine.attempt_repair(
                            report=diagnosis,
                            workspace_root=self.workspace_root,
                            patch_engine=self.patch_engine,
                        )

                        if repair_res.success:
                            for mf in repair_res.modified_files:
                                task.modified_files.add(mf)
                                task.budget.record_file_changed(mf)

                            broadcaster.emit(AgentEvent(
                                task_id=task_id,
                                event_type=AgentEventType.PATCH_APPLIED,
                                message=f"Corrective patch applied to {', '.join(repair_res.modified_files) or 'code'}",
                                metadata={"diffs": repair_res.diffs},
                            ))
                            # Re-run tests next
                            task.plan.insert(0, AutonomousAction(
                                type=ActionType.RUN_TESTS,
                                target="tests",
                                reason="Re-run tests to verify corrective patch",
                                risk=ActionRisk.LOW,
                            ))
                            task.transition_to(TaskStatus.TESTING, reason="Verifying patch with tests")
                            continue
                        else:
                            # Repair formulation failed
                            broadcaster.emit(AgentEvent(
                                task_id=task_id,
                                event_type=AgentEventType.TASK_FAILED,
                                message=f"Autonomous repair failed: {repair_res.error}",
                            ))
                            task.final_summary = f"I attempted automatic repair but encountered: {repair_res.error}"
                            task.transition_to(TaskStatus.FAILED, reason=repair_res.error)
                            break
                    else:
                        limit_msg = f"I attempted {task.budget.max_repair_attempts} automatic repairs. The remaining failure is: {diagnosis.error_message}"
                        broadcaster.emit(AgentEvent(
                            task_id=task_id,
                            event_type=AgentEventType.TASK_FAILED,
                            message=limit_msg,
                        ))
                        task.final_summary = limit_msg
                        task.transition_to(TaskStatus.FAILED, reason=limit_msg)
                        break
                else:
                    # Non-test action failed
                    task.final_summary = f"Action {current_action.type.value} failed: {obs.error}"
                    task.transition_to(TaskStatus.FAILED, reason=obs.error)
                    break

            # 10. Dynamic Plan Adaptation
            task.plan = self.planner.adapt_plan(task.plan, obs, self.workspace_root)

        return task

    def _run_final_verification(self, task: AutonomousTask):
        """Perform evidence-based completion check."""
        task.transition_to(TaskStatus.VERIFYING, reason="Executing final evidence verification")
        broadcaster.emit(AgentEvent(
            task_id=task.task_id,
            event_type=AgentEventType.VERIFICATION_STARTED,
            message="Verifying AST syntax, symbols, and test outcomes...",
        ))

        report: CompletionReport = CompletionVerifier.verify(
            workspace_root=self.workspace_root,
            modified_files=task.modified_files,
            user_request=task.user_request,
        )

        task.final_summary = report.summary

        if report.status == TaskStatus.COMPLETED:
            task.transition_to(TaskStatus.COMPLETED, reason="All verification criteria factually satisfied")
            broadcaster.emit(AgentEvent(
                task_id=task.task_id,
                event_type=AgentEventType.TASK_COMPLETED,
                message=report.summary,
                metadata={"evidence": report.evidence, "verified_files": report.verified_files},
            ))
        elif report.status == TaskStatus.PARTIALLY_COMPLETED:
            task.transition_to(TaskStatus.PARTIALLY_COMPLETED, reason=report.summary)
            broadcaster.emit(AgentEvent(
                task_id=task.task_id,
                event_type=AgentEventType.TASK_PARTIALLY_COMPLETED,
                message=report.summary,
                metadata={"evidence": report.evidence, "unmet": report.unmet_criteria},
            ))
        else:
            task.transition_to(TaskStatus.FAILED, reason=report.summary)
            broadcaster.emit(AgentEvent(
                task_id=task.task_id,
                event_type=AgentEventType.TASK_FAILED,
                message=report.summary,
                metadata={"unmet": report.unmet_criteria},
            ))
