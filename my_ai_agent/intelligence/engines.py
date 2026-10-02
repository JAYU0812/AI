"""Explicit Intelligence Capability Interfaces for Step 4.

Separates model-powered intelligence into dedicated capability interfaces:
- IntentEngine: Understood goals and entity extraction
- PlanningEngine: Multi-step atomic task decomposition
- CodingEngine: Code synthesis and EditProposal generation
- DiagnosisIntelligence: Error classification and root cause analysis
- RepairIntelligence: Corrective patch proposals
- ExplanationEngine: User-facing factual project and task summaries

Bridges the AIModel interface with Step 3B, Step 3C, and Step 3D components.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from my_ai_agent.autonomous.action import ActionRisk, ActionType, AutonomousAction
from my_ai_agent.autonomous.diagnosis import DiagnosisEngine, DiagnosticReport
from my_ai_agent.autonomous.repair import RepairEngine, RepairResult
from my_ai_agent.editing.edit_operations import EditProposal
from my_ai_agent.editing.patch_engine import PatchEngine
from my_ai_agent.intelligence.context_manager import AssembledContext, ContextManager
from my_ai_agent.intelligence.model import AIModel
from my_ai_agent.intelligence.model_router import ModelRouter, router
from my_ai_agent.intelligence.reasoning import ReasoningSummary
from my_ai_agent.nlp import IntentType, NaturalLanguageUnderstanding


class BaseIntelligenceEngine(ABC):
    """Base class for capability-specific intelligence engines."""

    def __init__(self, model: Optional[AIModel] = None, model_router: Optional[ModelRouter] = None):
        self._router = model_router or router
        self._model = model

    @property
    def model(self) -> AIModel:
        return self._model or self._router.route()


class IntentEngine(BaseIntelligenceEngine):
    """Understands user intent and extracts targeted parameters."""

    def understand(self, request: str) -> Dict[str, Any]:
        """Classify user intent and extract target files and parameters."""
        nlu = NaturalLanguageUnderstanding()
        parse_res = nlu.parse(request)
        entities = parse_res.entities
        return {
            "intent": parse_res.intent.value,
            "confidence": parse_res.confidence,
            "filename": entities.filename,
            "language": entities.language,
            "raw_request": request,
        }


class PlanningEngine(BaseIntelligenceEngine):
    """Decomposes user requests into atomic autonomous actions."""

    def create_plan(
        self,
        request: str,
        workspace_root: Path,
        context: Optional[AssembledContext] = None,
    ) -> List[AutonomousAction]:
        """Creates an actionable plan."""
        from my_ai_agent.autonomous.planner_controller import PlannerController
        controller = PlannerController()
        return controller.create_initial_plan(request, workspace_root)


class CodingEngine(BaseIntelligenceEngine):
    """Proposes code modifications as validated EditProposals.
    
    NEVER bypasses PatchEngine or writes raw unvalidated code.
    """

    def generate_edit_proposal(
        self,
        request: str,
        target_file: str,
        workspace_root: Path,
        context: Optional[AssembledContext] = None,
    ) -> Optional[EditProposal]:
        """Generates a structured EditProposal for target_file."""
        from my_ai_agent.editing.code_synthesizer import CodeSynthesizer
        from my_ai_agent.project.project_understanding import ProjectUnderstandingEngine

        root = Path(workspace_root).resolve()
        file_path = root / target_file
        existing_code = ""
        if file_path.exists():
            existing_code = file_path.read_text(encoding="utf-8", errors="ignore")

        p_engine = ProjectUnderstandingEngine()
        pmap = p_engine.understand_project(root)

        return CodeSynthesizer.synthesize_feature_proposal(
            feature_request=request,
            target_file=target_file,
            existing_code=existing_code,
            project_map=pmap,
        )


class DiagnosisIntelligence(BaseIntelligenceEngine):
    """Diagnoses test or command execution failures."""

    def diagnose_failure(
        self,
        error_output: str,
        workspace_root: Path,
    ) -> DiagnosticReport:
        """Categorizes error and identifies affected files, lines, and root cause."""
        return DiagnosisEngine.diagnose(error_output, workspace_root=workspace_root)


class RepairIntelligence(BaseIntelligenceEngine):
    """Synthesizes and applies corrective patches for diagnosed errors."""

    def propose_and_apply_repair(
        self,
        report: DiagnosticReport,
        workspace_root: Path,
        patch_engine: PatchEngine,
    ) -> RepairResult:
        """Applies corrective patch using Step 3C PatchEngine."""
        return RepairEngine.attempt_repair(
            report=report,
            workspace_root=workspace_root,
            patch_engine=patch_engine,
        )


class ExplanationEngine(BaseIntelligenceEngine):
    """Produces safe, transparent, user-facing explanations."""

    def explain_task_and_strategy(
        self,
        request: str,
        context: AssembledContext,
    ) -> ReasoningSummary:
        """Generates a clear summary of what was found and what the agent will do."""
        files = [f["path"] for f in context.relevant_files]
        symbols = [s["name"] for s in context.relevant_symbols]
        return ReasoningSummary.synthesize_from_context(
            objective=request,
            project_type=context.project_summary.split("|")[0].replace("Type:", "").strip(),
            identified_files=files,
            identified_symbols=symbols,
            has_tests=any("test" in f.lower() for f in files),
        )
