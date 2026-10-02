"""Core Agent Engine module for Step 2.

Implements the advanced cognitive loop:
User Input ──▶ NLP (Intent + Entity Extraction)
           ──▶ Multi-Step Planner (Goal Decomposition)
           ──▶ Tool Chaining Execution (Sequential Tool Loop)
           ──▶ Self-Verification & Error Recovery
           ──▶ Dual Memory Storage & Response Synthesis
"""

from typing import Any, Dict, List, Optional
import sys

from config import Style, config
from memory import AgentMemory
from nlp import ExtractedEntities, IntentType, NLUResult, NaturalLanguageUnderstanding
from planner import ExecutionPlan, MultiStepPlanner, PlanState, PlanStep, StepStatus
from synthesizer import CodeSynthesizer
from tools import ToolRegistry, ToolResult, registry
from verifier import SelfVerifier, VerificationResult


class AgentEngine:
    """The central agent orchestrator coordinating NLP, Planning, Tools, Verification, and Memory."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[AgentMemory] = None,
        verbose: bool = True,
    ):
        self.registry = tool_registry or registry
        self.memory = memory or AgentMemory()
        self.nlu = NaturalLanguageUnderstanding()
        self.synthesizer = CodeSynthesizer()
        self.planner = MultiStepPlanner(synthesizer=self.synthesizer)
        self.verifier = SelfVerifier()
        self.verbose = verbose

    def _render_badge(self, category: str, icon: str, detail: str, color: str):
        """Displays a clean, styled badge in console output."""
        if not self.verbose:
            return
        tag = Style.paint(f"[{icon} {category:<13}]", color, Style.BOLD)
        print(f"{tag} {detail}")

    def run_cycle(self, user_input: str) -> str:
        """Executes a full agent turn."""
        clean_input = user_input.strip()
        if not clean_input:
            return "Please provide a command, question, or task."

        last_file = self.memory.working_context.get("last_file")

        # -------------------------------------------------------------
        # 1. Intent Classification & Entity Extraction
        # -------------------------------------------------------------
        nlu_result = self.nlu.parse(clean_input, last_file=last_file)

        # Handle system commands
        if nlu_result.intent == IntentType.EXIT:
            return "Goodbye! Agent shutting down."

        if nlu_result.intent == IntentType.HELP:
            return self._generate_help_response()

        if nlu_result.intent == IntentType.SHOW_MEMORY:
            return self._generate_memory_summary()

        if nlu_result.intent == IntentType.CLEAR_MEMORY:
            self.memory.clear()
            return "Agent memory reset for fresh session."

        if nlu_result.intent == IntentType.UNKNOWN:
            return (
                f"I could not determine the intent for: '{clean_input}'.\n"
                f"Try commands like:\n"
                f"  - Can you make me a Python file that calculates the average of three numbers and then run it?\n"
                f"  - Create test.py\n"
                f"  - Write print(\"Hello\") into test.py\n"
                f"  - Run test.py\n"
                f"  - List files"
            )

        # Render extracted intent & entities
        self._render_badge(
            "Intent",
            "🧠",
            f"{nlu_result.intent.value} (Confidence: {nlu_result.confidence * 100:.0f}%)",
            Style.CYAN,
        )
        if nlu_result.entities.purpose or nlu_result.entities.filename:
            ent_summary = (
                f"Target='{nlu_result.entities.filename or 'None'}', "
                f"Purpose='{nlu_result.entities.purpose or 'N/A'}', "
                f"Language='{nlu_result.entities.language}'"
            )
            self._render_badge("Entities", "🏷️ ", ent_summary, Style.BRIGHT_CYAN)

        # -------------------------------------------------------------
        # 2. Multi-Step Planning & Goal Decomposition
        # -------------------------------------------------------------
        plan = self.planner.create_plan(nlu_result, self.memory.working_context)
        total_steps = len(plan.steps)

        if total_steps == 0:
            return "Plan created with 0 steps. No action taken."

        plan_desc = f"Goal decomposed into {total_steps} step(s):"
        self._render_badge("Planner", "📋", plan_desc, Style.YELLOW)
        for s in plan.steps:
            if self.verbose:
                print(f"      {Style.paint(f'{s.step_num}.', Style.BOLD)} {s.description}")

        # -------------------------------------------------------------
        # 3. Tool Chaining & Execution Loop
        # -------------------------------------------------------------
        plan.state = PlanState.IN_PROGRESS
        last_tool_output = ""
        verification_passed = True
        verification_msg = ""

        for step in plan.steps:
            step.status = StepStatus.RUNNING
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in step.tool_args.items())
            self._render_badge(
                f"Step {step.step_num}/{total_steps}",
                "⚙️ ",
                f"{step.tool_name}({args_str})",
                Style.MAGENTA,
            )

            # Tool execution
            result: ToolResult = self.registry.execute(step.tool_name, **step.tool_args)
            step.result = result

            obs_text = result.output if result.success else f"Error: {result.error}"
            step.observation = obs_text
            last_tool_output = obs_text

            obs_preview = obs_text if len(obs_text) < 140 else obs_text[:137] + "..."
            obs_color = Style.GREEN if result.success else Style.RED
            self._render_badge("Observation", "👀", obs_preview, obs_color)

            # Record step interaction in memory
            self.memory.record_interaction(
                user_input=clean_input if step.step_num == 1 else f"[Substep {step.step_num}] {step.description}",
                intent=step.action_type,
                thought=step.description,
                tool_name=step.tool_name,
                tool_args=step.tool_args,
                observation=obs_text,
                response=obs_preview,
            )

            # ---------------------------------------------------------
            # 4. Self-Verification & Error Handling
            # ---------------------------------------------------------
            if step.action_type == "RUN_COMMAND":
                ret_code = result.metadata.get("returncode", 0 if result.success else 1)
                v_res: VerificationResult = self.verifier.verify_execution(
                    obs_text,
                    ret_code,
                    purpose=nlu_result.entities.purpose,
                )
                v_color = Style.BRIGHT_GREEN if v_res.passed else Style.BRIGHT_RED
                self._render_badge("Verification", "🔍", f"{v_res.status}: {v_res.diagnostics}", v_color)
                if not v_res.passed:
                    verification_passed = False
                    verification_msg = v_res.diagnostics
                    if v_res.suggested_fix and self.verbose:
                        print(f"      💡 {Style.paint('Suggested Fix:', Style.BOLD, Style.YELLOW)} {v_res.suggested_fix}")

            if not result.success:
                step.status = StepStatus.FAILED
                plan.state = PlanState.FAILED
                self.memory.record_plan(plan.goal, plan.intent.value, total_steps, status="FAILED")
                return f"❌ Step {step.step_num} failed: {result.error}"

            step.status = StepStatus.SUCCESS

        plan.state = PlanState.COMPLETED
        self.memory.record_plan(plan.goal, plan.intent.value, total_steps, status="COMPLETED")

        # -------------------------------------------------------------
        # 5. Final Response Synthesis
        # -------------------------------------------------------------
        return self._synthesize_plan_response(plan, nlu_result, last_tool_output, verification_passed)

    def _synthesize_plan_response(
        self,
        plan: ExecutionPlan,
        nlu: NLUResult,
        last_output: str,
        verified: bool,
    ) -> str:
        intent = plan.intent
        ent = nlu.entities

        if intent == IntentType.BUILD_AND_RUN:
            ver_badge = " [Self-Verification: ✅ PASSED]" if verified else " [Self-Verification: ⚠️ Issues detected]"
            return (
                f"Successfully built and executed '{ent.filename}' for '{ent.purpose}'{ver_badge}.\n\n"
                f"Program Execution Output:\n"
                f"-----------------------------------------\n"
                f"{last_output}\n"
                f"-----------------------------------------"
            )

        if intent == IntentType.BUILD_FILE:
            return f"Successfully created and synthesized '{ent.filename}' for '{ent.purpose}'."

        if intent == IntentType.CREATE_FILE:
            return f"Created file '{ent.filename}' in workspace."

        if intent == IntentType.WRITE_FILE:
            return f"Updated file '{ent.filename}' with provided content."

        if intent == IntentType.READ_FILE:
            return f"Content of '{ent.filename}':\n{last_output}"

        if intent == IntentType.LIST_FILES:
            return f"{last_output}"

        if intent == IntentType.RUN_COMMAND:
            return f"Execution Result:\n{last_output}"

        return last_output

    def _generate_help_response(self) -> str:
        lines = [
            Style.paint("🤖 AI Agent Engine - Step 2 Capabilities", Style.BOLD, Style.CYAN),
            "Your agent now features Intent Classification, Entity Extraction, Multi-Step Planning,",
            "Zero-LLM Code Synthesis, and Self-Verification!",
            "",
            "Try complex natural-language requests:",
            "  • " + Style.paint('Can you make me a Python file that calculates the average of three numbers and then run it?', Style.YELLOW),
            "  • " + Style.paint('Please build a script to compute fibonacci numbers and run it', Style.YELLOW),
            "  • " + Style.paint('Create a prime number checker script and run it', Style.YELLOW),
            "",
            "Step 1 basic commands also work:",
            "  • Create test.py",
            "  • Write print(\"Hello\") into test.py",
            "  • Run test.py",
            "  • List files",
            "  • History",
        ]
        return "\n".join(lines)

    def _generate_memory_summary(self) -> str:
        recent = self.memory.get_recent_history(5)
        plans = self.memory.get_recent_plans(3)

        lines = [
            Style.paint(f"🧠 Agent Memory (Session: {self.memory.session_id})", Style.BOLD, Style.CYAN)
        ]

        if plans:
            lines.append("\n" + Style.paint("Recent Multi-Step Plans:", Style.BOLD))
            for p in plans:
                lines.append(f"  • [Plan #{p['id']}] \"{p['goal']}\" → {p['total_steps']} steps → Status: {p['status']}")

        lines.append("\n" + Style.paint("Recent Step Interactions:", Style.BOLD))
        if recent:
            for rec in recent:
                lines.append(f"  • [Step {rec.step}] {rec.tool_name} → {rec.intent}")
        else:
            lines.append("  (No recent steps)")

        ctx = self.memory.get_context()
        lines.append(f"\nContext: last_file={ctx.get('last_file')}, last_command={ctx.get('last_command')}")
        return "\n".join(lines)
