"""Prompt Construction System for Step 4 Model Integration.

Formats standardized prompts containing system role, project context,
file contents, recent observations, and strict structured output instructions.
"""

from typing import Any, Dict, Optional

from my_ai_agent.intelligence.context_manager import AssembledContext


class PromptBuilder:
    """Builds standardized prompts for model inference."""

    SYSTEM_ROLE = (
        "You are an AI software engineering worker. You understand user goals, "
        "inspect project architecture, plan atomic code modifications, observe tool outputs, "
        "diagnose errors factually, and provide structured action proposals."
    )

    @classmethod
    def build_action_prompt(
        cls,
        user_request: str,
        context: AssembledContext,
        instruction: Optional[str] = None,
    ) -> str:
        """Constructs prompt asking model to decide the next action."""
        prompt_parts = [
            f"=== SYSTEM ROLE ===\n{cls.SYSTEM_ROLE}\n",
            f"=== PROJECT CONTEXT ===\n{context.project_summary}\n",
        ]

        if context.relevant_files:
            prompt_parts.append("=== RELEVANT CODE FILES ===")
            for f in context.relevant_files:
                prompt_parts.append(f"File `{f['path']}`:\n```\n{f.get('content', '')}\n```\n")

        if context.recent_observations:
            prompt_parts.append("=== RECENT OBSERVATIONS & ERRORS ===")
            for obs in context.recent_observations:
                prompt_parts.append(f"- [{obs.get('action')}] Success={obs.get('success')}: {obs.get('summary', '')}")
            prompt_parts.append("")

        prompt_parts.append(f"=== CURRENT GOAL ===\n{user_request}\n")

        if instruction:
            prompt_parts.append(f"=== SPECIFIC INSTRUCTION ===\n{instruction}\n")

        prompt_parts.append(
            "=== OUTPUT FORMAT ===\n"
            "Return a valid JSON object matching the action_decision schema:\n"
            "{\n"
            '  "action": "PATCH_FILE | READ_FILE | RUN_TESTS | RUN_COMMAND | CREATE_FILE | FINISH",\n'
            '  "target": "target file or command",\n'
            '  "reason": "factual rationale for this action",\n'
            '  "risk": "LOW | MEDIUM | HIGH"\n'
            "}"
        )

        return "\n".join(prompt_parts)

    @classmethod
    def build_planning_prompt(
        cls,
        user_request: str,
        context: AssembledContext,
    ) -> str:
        """Constructs prompt asking model to generate an initial plan."""
        return (
            f"=== SYSTEM ROLE ===\n{cls.SYSTEM_ROLE}\n\n"
            f"=== PROJECT CONTEXT ===\n{context.project_summary}\n\n"
            f"=== REQUEST ===\n{user_request}\n\n"
            "Generate an atomic, step-by-step implementation plan. Return a valid JSON matching the plan schema:\n"
            "{\n"
            f'  "goal": "{user_request}",\n'
            '  "steps": [\n'
            '    {"description": "...", "action": "...", "target": "...", "reason": "...", "risk": "LOW"}\n'
            "  ]\n"
            "}"
        )
