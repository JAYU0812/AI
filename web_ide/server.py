"""Visual AI IDE Server (Step 3A.1).

Provides multi-project management, real-time filesystem synchronization,
and Python agent orchestration for real user-created projects.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import shutil
import sys
import time
import uuid

# Configure UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure parent and my_ai_agent package are on python path
BASE_DIR = Path(__file__).resolve().parent.parent
AGENT_DIR = BASE_DIR / "my_ai_agent"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
STATIC_FALLBACK = BASE_DIR / "web_ide" / "static"
# Always serve our rebuilt static UI (Step 3A.1).
# Comment the next line and uncomment the line below to use the React dist build instead.
STATIC_DIR = STATIC_FALLBACK
# STATIC_DIR = FRONTEND_DIST if (FRONTEND_DIST / "index.html").exists() else STATIC_FALLBACK
DEFAULT_PROJECTS_DIR = Path("C:/jay/Projects")

for p in (str(BASE_DIR), str(AGENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import config
from memory import AgentMemory
from tools import create_file, read_file, write_file, list_files, run_command, registry
from nlp import NaturalLanguageUnderstanding, IntentType
from synthesizer import CodeSynthesizer
from planner import MultiStepPlanner, PlanState, StepStatus
from verifier import SelfVerifier
from agent import AgentEngine
from my_ai_agent.project import ProjectUnderstandingEngine
from my_ai_agent.autonomous import AutonomousExecutionEngine, broadcaster, AgentEvent, AgentEventType

# Create dedicated agent and project understanding instances for Web IDE
ide_memory = AgentMemory()
ide_agent = AgentEngine(tool_registry=registry, memory=ide_memory, verbose=False)
project_engine = ProjectUnderstandingEngine()
auto_engine = AutonomousExecutionEngine(workspace_root=config.WORKSPACE_DIR, project_engine=project_engine)

# Active Project State
current_project: Dict[str, Any] = {
    "id": "default",
    "name": "Default Workspace",
    "path": str(config.WORKSPACE_DIR),
}


def sync_projects_from_disk():
    """Discover existing folders in C:\\jay\\Projects and register in SQLite."""
    if not DEFAULT_PROJECTS_DIR.exists():
        DEFAULT_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    for entry in DEFAULT_PROJECTS_DIR.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            existing = ide_memory.get_project(str(entry.resolve()))
            if not existing:
                proj_id = f"proj_{entry.name.lower().replace(' ', '_')}"
                ide_memory.register_project(
                    project_id=proj_id,
                    name=entry.name,
                    path=str(entry.resolve()),
                    description="Local project folder",
                )


def get_project_file_count(project_path: Path) -> int:
    """Count non-hidden files in a project folder."""
    try:
        if not project_path.exists():
            return 0
        return sum(1 for p in project_path.rglob("*") if p.is_file() and not p.name.startswith("."))
    except Exception:
        return 0


def get_workspace_files():
    """List all real files in the active project directory."""
    workspace = config.WORKSPACE_DIR
    files = []
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)

    for entry in sorted(workspace.rglob("*")):
        if entry.is_file() and not any(part.startswith(".") for part in entry.parts):
            rel = entry.relative_to(workspace)
            stat = entry.stat()
            files.append({
                "name": entry.name,
                "path": str(rel).replace("\\", "/"),
                "size_bytes": stat.st_size,
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "ext": entry.suffix.lower(),
            })
    return files


CONVERSATIONAL_PATTERNS = (
    "hello", "hi", "hey", "howdy", "greetings",
    "how are you", "what can you do", "what do you do",
    "help", "what are you", "who are you", "capabilities",
    "thanks", "thank you", "great", "awesome", "good job",
    "nice", "cool", "ok", "okay",
)


def handle_agent_turn(user_message: str, project_id: Optional[str] = None):
    """Executes agent turn inside the active project directory with persistence."""
    clean_msg = user_message.strip()
    if not clean_msg:
        return {"error": "Empty message"}

    proj_id = project_id or current_project.get("id") or "default"

    # Persist user message to SQLite
    try:
        ide_memory.save_chat_message(proj_id, "user", clean_msg)
    except Exception as e:
        print(f"[Memory Warning] Failed to save user chat turn: {e}")

    # Handle conversational / greeting messages directly
    lower = clean_msg.lower()
    if any(lower == p or lower.startswith(p + " ") or lower.startswith(p + "!") or lower.startswith(p + ",")
           for p in CONVERSATIONAL_PATTERNS):
        project_path = str(config.WORKSPACE_DIR)
        existing_files = get_workspace_files()
        file_count = len(existing_files)
        file_summary = (
            f"Your project has {file_count} file{'s' if file_count != 1 else ''}." if file_count
            else "Your project folder is empty — ready to start."
        )
        replies = {
            "help": (
                f"I'm your AI coding agent. Tell me what to build and I'll handle it autonomously.\n\n"
                f"Try things like:\n"
                f"• \"Create a Python program that calculates Fibonacci numbers and run it\"\n"
                f"• \"Build a calculator script and run it\"\n"
                f"• \"Create a file called main.py and write Hello World in it\"\n\n"
                f"Working directory: {project_path}"
            ),
            "thanks": "You're welcome! Let me know if there's anything else to build or fix.",
            "great": "Glad it worked! What would you like to build next?",
            "good job": "Thanks! What's next?",
        }
        reply_text = None
        for key, reply in replies.items():
            if key in lower:
                reply_text = reply
                break
        if not reply_text:
            reply_text = (
                f"Hello! I'm your AI agent, ready to work on this project.\n\n"
                f"{file_summary}\n\n"
                f"Tell me what you'd like to build, create, or fix — I'll handle it autonomously."
            )

        conv_result = {
            "user_message": clean_msg,
            "response": reply_text,
            "intent": {"name": "GREET", "confidence": 1.0, "explanation": "Conversational"},
            "entities": {},
            "plan": {"goal": "", "state": "completed", "steps": []},
            "executed_steps": [],
            "verification": None,
            "active_file": None,
            "terminal_output": "",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        try:
            ide_memory.save_chat_message(proj_id, "agent", reply_text, None)
        except Exception as e:
            print(f"[Memory Warning] Failed to save greeting turn: {e}")
        return conv_result

    last_file = ide_memory.working_context.get("last_file")

    # Step 3B: Project Understanding Engine
    pmap = project_engine.understand_project(config.WORKSPACE_DIR)

    # 1. NLP Step
    nlu_res = ide_agent.nlu.parse(clean_msg, last_file=last_file)

    # 2. Check if user is asking to understand or explain something in the project
    if nlu_res.intent == IntentType.UNDERSTAND_PROJECT:
        explanation = project_engine.explain_feature(config.WORKSPACE_DIR, clean_msg)
        understand_res = {
            "user_message": clean_msg,
            "response": explanation,
            "intent": {
                "name": "UNDERSTAND_PROJECT",
                "confidence": nlu_res.confidence,
                "explanation": nlu_res.explanation,
            },
            "entities": {
                "purpose": nlu_res.entities.purpose,
            },
            "plan": {
                "goal": f"Analyze project for '{nlu_res.entities.purpose}'",
                "state": "completed",
                "steps": [],
            },
            "project_understanding": {
                "project_name": pmap.project_name,
                "primary_type": pmap.primary_type,
                "total_files": pmap.total_files_count,
                "source_files": pmap.source_files_count,
                "entry_point": pmap.primary_entry_point,
                "dependencies_count": sum(d.total_count for d in pmap.dependencies),
            },
            "executed_steps": [],
            "verification": None,
            "active_file": pmap.primary_entry_point,
            "terminal_output": "",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        try:
            ide_memory.save_chat_message(proj_id, "agent", explanation, understand_res)
        except Exception:
            pass
        return understand_res

    # 3. Read Before Modify: for REFACTOR_FILE or code modification requests, bind to relevant file
    code_mod_keywords = (
        "function", "method", "feature", "logout", "login", "uuid", "session",
        "percentage", "support", "endpoint", "class", "route", "service", "handler"
    )
    if nlu_res.intent == IntentType.REFACTOR_FILE or any(k in clean_msg.lower() for k in ("add", "modify", "update", "refactor", "patch")) and any(k in clean_msg.lower() for k in code_mod_keywords):
        rel_ctx = project_engine.get_relevant_context(config.WORKSPACE_DIR, clean_msg)
        if rel_ctx.relevant_files:
            top_file = rel_ctx.relevant_files[0].path
            filename_explicit = bool(
                nlu_res.entities.filename
                and nlu_res.entities.filename.lower() in clean_msg.lower()
                and any(nlu_res.entities.filename.endswith(ext) for ext in (".py", ".js", ".ts", ".html", ".css", ".json"))
            )
            if not filename_explicit or not (config.WORKSPACE_DIR / (nlu_res.entities.filename or "")).exists():
                nlu_res.entities.filename = top_file
                nlu_res.intent = IntentType.REFACTOR_FILE

    # Step 3D: Autonomous Coding & Debugging Loop
    lower_msg = clean_msg.lower()
    is_autonomous_task = any(p in lower_msg for p in (
        "add subtract, multiply and divide",
        "fix the failing tests",
        "fix failing tests",
        "build a small python cli",
        "build a python cli",
        "build a python rest api",
    ))
    if is_autonomous_task:
        auto_eng = AutonomousExecutionEngine(
            workspace_root=config.WORKSPACE_DIR,
            project_engine=project_engine,
        )
        task = auto_eng.run_task(user_request=clean_msg, project_id=proj_id)
        try:
            ide_memory.save_autonomous_task(task.to_dict())
        except Exception:
            pass

        plan_steps = []
        for i, a in enumerate(task.completed_actions):
            plan_steps.append({
                "step_num": i + 1,
                "action_type": a.type.value,
                "description": a.reason,
                "tool_name": a.type.value.lower(),
                "tool_args": {"target": a.target},
                "observation": task.observations[i].output if i < len(task.observations) else "",
                "success": a.status == "COMPLETED",
            })

        diffs_collected = []
        for obs in task.observations:
            for d in obs.diffs:
                diffs_collected.append(d)

        test_run_res = None
        for obs in reversed(task.observations):
            if obs.action_type == ActionType.RUN_TESTS:
                test_run_res = {
                    "passed": obs.success,
                    "returncode": obs.exit_code or (0 if obs.success else 1),
                    "output": obs.output,
                    "error": obs.error,
                }
                break

        auto_result = {
            "user_message": clean_msg,
            "response": task.final_summary or f"Autonomous task {task.status.value}: {task.user_request}",
            "intent": {"name": "AUTONOMOUS_TASK", "confidence": 1.0, "explanation": "Step 3D Autonomous Execution"},
            "entities": {"task_id": task.task_id, "status": task.status.value},
            "plan": {
                "goal": task.user_request,
                "state": "completed" if task.status == TaskStatus.COMPLETED else "failed",
                "steps": [a.to_dict() for a in task.completed_actions],
            },
            "executed_steps": plan_steps,
            "diffs": diffs_collected,
            "test_results": test_run_res,
            "verification": {"passed": task.status == TaskStatus.COMPLETED, "evidence": task.final_summary},
            "active_file": list(task.modified_files)[0] if task.modified_files else None,
            "terminal_output": test_run_res.get("output", "") if test_run_res else "",
            "timestamp": time.strftime("%H:%M:%S"),
            "task": task.to_dict(),
        }
        try:
            ide_memory.save_chat_message(proj_id, "agent", auto_result["response"], auto_result)
        except Exception:
            pass
        return auto_result

    # Contextual override for web/real-estate landing pages
    if any(k in lower_msg for k in ("landing page", "website", "real estate", "real-estate", "login page")):
        if "react" in lower_msg or "tailwind" in lower_msg or "html" in lower_msg or "landing" in lower_msg:
            nlu_res.intent = IntentType.BUILD_AND_RUN if "run" in lower_msg else IntentType.BUILD_FILE
            nlu_res.entities.purpose = "real estate landing page"
            nlu_res.entities.language = "html"
            nlu_res.entities.filename = "index.html"

    # 4. Planning step
    plan = ide_agent.planner.create_plan(nlu_res, ide_memory.working_context)

    # If it's a real-estate landing page task, synthesize multi-file modern landing page
    if "real estate" in lower_msg or "landing page" in lower_msg:
        plan.steps.clear()
        plan.add_step(
            action_type="CREATE_FILE",
            tool_name="create_file",
            tool_args={"path": "index.html", "content": ""},
            description="Create index.html for real estate landing page",
        )
        plan.add_step(
            action_type="WRITE_FILE",
            tool_name="write_file",
            tool_args={
                "path": "index.html",
                "content": """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Aura Living - Modern Real Estate</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
  </style>
</head>
<body class="bg-slate-50 text-slate-900">
  <!-- Navigation -->
  <nav class="bg-white border-b border-slate-200 sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <span class="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">AURA</span>
        <span class="text-xs font-semibold uppercase tracking-wider text-slate-500">Real Estate</span>
      </div>
      <div class="hidden md:flex items-center gap-8 text-sm font-medium text-slate-600">
        <a href="#properties" class="hover:text-blue-600 transition">Properties</a>
        <a href="#about" class="hover:text-blue-600 transition">About</a>
        <a href="#contact" class="hover:text-blue-600 transition">Contact</a>
      </div>
      <button class="bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold px-5 py-2.5 rounded-lg transition shadow-sm">
        Schedule Tour
      </button>
    </div>
  </nav>

  <!-- Hero Section -->
  <header class="max-w-7xl mx-auto px-6 py-20 text-center">
    <span class="inline-block bg-blue-50 text-blue-700 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider mb-4 border border-blue-100">
      Premium Architecture & Living
    </span>
    <h1 class="text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900 mb-6">
      Find Your Next Architectural Home
    </h1>
    <p class="text-lg text-slate-600 max-w-2xl mx-auto mb-10 leading-relaxed">
      Curated modern residences, luxury waterfront penthouses, and bespoke sustainable retreats built for refined living.
    </p>
    <div class="max-w-xl mx-auto bg-white p-2 rounded-xl shadow-lg border border-slate-200 flex flex-col sm:flex-row gap-2">
      <input type="text" placeholder="Location, Neighborhood or City..." class="flex-1 px-4 py-3 rounded-lg text-sm text-slate-800 outline-none">
      <button class="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-3 rounded-lg text-sm transition">
        Search Homes
      </button>
    </div>
  </header>

  <!-- Featured Listings -->
  <section id="properties" class="max-w-7xl mx-auto px-6 py-12">
    <div class="flex items-center justify-between mb-8">
      <div>
        <h2 class="text-2xl font-bold text-slate-900">Featured Modern Estates</h2>
        <p class="text-sm text-slate-500">Hand-selected residential architectural masterpieces</p>
      </div>
      <a href="#" class="text-sm font-semibold text-blue-600 hover:underline">View All (24) →</a>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
      <!-- Card 1 -->
      <div class="bg-white rounded-2xl overflow-hidden border border-slate-200 shadow-sm hover:shadow-md transition">
        <div class="h-56 bg-slate-200 flex items-center justify-center text-slate-400">
          <span class="text-4xl">🏡</span>
        </div>
        <div class="p-6">
          <div class="text-xl font-bold text-slate-900">$2,450,000</div>
          <div class="text-sm font-semibold text-slate-700 mt-1">The Glass Pavilion</div>
          <div class="text-xs text-slate-500 mb-4">442 Meadow Vista, Aspen CO</div>
          <div class="flex items-center gap-4 text-xs text-slate-600 border-t border-slate-100 pt-4">
            <span>🛏 4 Beds</span>
            <span>🛁 4.5 Baths</span>
            <span>📐 4,800 sqft</span>
          </div>
        </div>
      </div>

      <!-- Card 2 -->
      <div class="bg-white rounded-2xl overflow-hidden border border-slate-200 shadow-sm hover:shadow-md transition">
        <div class="h-56 bg-slate-200 flex items-center justify-center text-slate-400">
          <span class="text-4xl">🏙️</span>
        </div>
        <div class="p-6">
          <div class="text-xl font-bold text-slate-900">$3,890,000</div>
          <div class="text-sm font-semibold text-slate-700 mt-1">Skyline Penthouse 18</div>
          <div class="text-xs text-slate-500 mb-4">700 Ocean Boulevard, Miami FL</div>
          <div class="flex items-center gap-4 text-xs text-slate-600 border-t border-slate-100 pt-4">
            <span>🛏 3 Beds</span>
            <span>🛁 3.5 Baths</span>
            <span>📐 3,200 sqft</span>
          </div>
        </div>
      </div>

      <!-- Card 3 -->
      <div class="bg-white rounded-2xl overflow-hidden border border-slate-200 shadow-sm hover:shadow-md transition">
        <div class="h-56 bg-slate-200 flex items-center justify-center text-slate-400">
          <span class="text-4xl">🌲</span>
        </div>
        <div class="p-6">
          <div class="text-xl font-bold text-slate-900">$1,920,000</div>
          <div class="text-sm font-semibold text-slate-700 mt-1">Cedar Creek Retreat</div>
          <div class="text-xs text-slate-500 mb-4">12 Ridgepoint Trail, Austin TX</div>
          <div class="flex items-center gap-4 text-xs text-slate-600 border-t border-slate-100 pt-4">
            <span>🛏 5 Beds</span>
            <span>🛁 5 Baths</span>
            <span>📐 5,100 sqft</span>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Footer -->
  <footer class="bg-white border-t border-slate-200 mt-20 py-8 text-center text-xs text-slate-500">
    © 2026 Aura Real Estate. Built with React and Tailwind concepts by Custom AI Agent.
  </footer>
</body>
</html>""",
                "append": False,
            },
            description="Write modern Tailwind real-estate landing page into index.html",
        )

    # 4. Tool Execution Loop
    executed_steps = []
    terminal_output = ""
    verification_info = None
    active_file = nlu_res.entities.filename or ("index.html" if "landing page" in lower_msg else None)
    read_cache = {}
    files_modified = False
    applied_diffs = []
    test_run_result = None
    autofix_result = None

    plan.state = PlanState.IN_PROGRESS
    for step in plan.steps:
        step.status = StepStatus.RUNNING

        # Step 3C: Precise Patch Engine integration
        if step.action_type == "WRITE_FILE" and (not step.tool_args.get("content") or nlu_res.intent == IntentType.REFACTOR_FILE):
            p_arg = step.tool_args.get("path", "")
            existing_code = read_cache.get(p_arg, "")
            if not existing_code:
                f_target = config.WORKSPACE_DIR / p_arg
                if f_target.exists() and f_target.is_file():
                    existing_code = f_target.read_text(encoding="utf-8", errors="replace")
            if existing_code:
                from my_ai_agent.editing.patch_engine import PatchEngine
                engine = PatchEngine(workspace_root=config.WORKSPACE_DIR)
                proposal = ide_agent.synthesizer.propose_edit(
                    feature_request=clean_msg,
                    target_file=p_arg,
                    existing_code=existing_code,
                )
                tx_res = engine.apply_proposal(proposal)
                if tx_res.success:
                    f_res = tx_res.file_results.get(p_arg)
                    step.tool_args["content"] = f_res.patched_code if f_res else existing_code
                    if tx_res.diffs:
                        for d in tx_res.diffs:
                            applied_diffs.append(d.to_dict())
                else:
                    refactored_code = ide_agent.synthesizer.refactor_code(existing_code, clean_msg)
                    step.tool_args["content"] = refactored_code

        res = registry.execute(step.tool_name, **step.tool_args)
        step.result = res
        obs = res.output if res.success else f"Error: {res.error}"
        step.observation = obs

        if "path" in step.tool_args:
            active_file = step.tool_args["path"]

        if step.action_type == "READ_FILE" and res.success:
            f_target = config.WORKSPACE_DIR / step.tool_args.get("path", "")
            if f_target.exists() and f_target.is_file():
                read_cache[step.tool_args.get("path")] = f_target.read_text(encoding="utf-8", errors="replace")

        if step.action_type in ("CREATE_FILE", "WRITE_FILE") and res.success:
            files_modified = True

        if step.action_type == "RUN_COMMAND":
            terminal_output = obs
            ret_code = res.metadata.get("returncode", 0 if res.success else 1)
            v_res = ide_agent.verifier.verify_execution(obs, ret_code, purpose=nlu_res.entities.purpose)
            verification_info = {
                "passed": v_res.passed,
                "status": v_res.status,
                "diagnostics": v_res.diagnostics,
                "suggested_fix": v_res.suggested_fix,
            }

        executed_steps.append({
            "step_num": step.step_num,
            "action_type": step.action_type,
            "description": step.description,
            "tool_name": step.tool_name,
            "tool_args": step.tool_args,
            "observation": obs,
            "success": res.success,
        })

        ide_memory.record_interaction(
            user_input=clean_msg if step.step_num == 1 else f"[{step.action_type}] {step.description}",
            intent=step.action_type,
            thought=step.description,
            tool_name=step.tool_name,
            tool_args=step.tool_args,
            observation=obs,
            response=obs[:100],
        )

        if not res.success:
            step.status = StepStatus.FAILED
            plan.state = PlanState.FAILED
            break
        step.status = StepStatus.SUCCESS

    if plan.state != PlanState.FAILED:
        plan.state = PlanState.COMPLETED

    # Step 3B: Refresh project map after agent modifications
    if files_modified:
        try:
            pmap = project_engine.refresh(config.WORKSPACE_DIR)
        except Exception as e:
            print(f"[ProjectEngine Refresh Warning]: {e}")

        # Step 3C: Autonomous Test Execution & Auto-Fix loop
        try:
            from my_ai_agent.editing.test_runner import ProjectTestRunner
            from my_ai_agent.editing.patch_engine import PatchEngine
            initial_test = ProjectTestRunner.run_tests(config.WORKSPACE_DIR)
            if initial_test:
                test_run_result = initial_test.to_dict()
                if not initial_test.passed:
                    engine = PatchEngine(workspace_root=config.WORKSPACE_DIR)
                    fix_res = ProjectTestRunner.run_autofix_loop(
                        workspace_root=config.WORKSPACE_DIR,
                        patch_engine=engine,
                        initial_test_res=initial_test,
                    )
                    autofix_result = fix_res.to_dict()
                    if fix_res.success and fix_res.final_test_result:
                        test_run_result = fix_res.final_test_result.to_dict()
        except Exception as e:
            print(f"[TestRunner Warning]: {e}")

    ide_memory.record_plan(plan.goal, plan.intent.value, len(plan.steps), status=plan.state.value)

    last_obs = executed_steps[-1]["observation"] if executed_steps else ""
    is_verified = verification_info["passed"] if verification_info else True
    final_reply = ide_agent._synthesize_plan_response(plan, nlu_res, last_obs, is_verified)

    # Ensure response is never empty — always give the user feedback
    if not final_reply or not final_reply.strip():
        if plan.state == PlanState.COMPLETED and executed_steps:
            files_created = [s["tool_args"].get("path", "") for s in executed_steps
                             if s.get("action_type") in ("CREATE_FILE", "WRITE_FILE") and s.get("success")]
            if files_created:
                final_reply = f"Done. Created/updated: {', '.join(files_created)}"
            else:
                final_reply = "Done. Task completed successfully."
        elif plan.state == PlanState.FAILED:
            final_reply = "I ran into a problem completing that task. Check the details below."
        else:
            # Low confidence / unknown intent — be helpful
            final_reply = (
                f"I'm not sure how to handle that yet. Try being more specific, for example:\n"
                f"\"Create a Python script that prints Hello World and run it\"\n"
                f"\"Build a calculator app\""
            )

    result_payload = {
        "user_message": clean_msg,
        "response": final_reply,
        "intent": {
            "name": nlu_res.intent.value,
            "confidence": nlu_res.confidence,
            "explanation": nlu_res.explanation,
        },
        "entities": {
            "language": nlu_res.entities.language,
            "filename": active_file,
            "purpose": nlu_res.entities.purpose,
            "inputs": nlu_res.entities.inputs,
            "run_after": nlu_res.entities.run_after,
        },
        "plan": {
            "goal": plan.goal,
            "state": plan.state.value,
            "steps": [
                {
                    "step_num": s.step_num,
                    "action_type": s.action_type,
                    "description": s.description,
                    "status": s.status.value,
                }
                for s in plan.steps
            ],
        },
        "executed_steps": executed_steps,
        "verification": verification_info,
        "diffs": applied_diffs,
        "test_results": test_run_result,
        "autofix": autofix_result,
        "project_understanding": {
            "project_name": pmap.project_name,
            "primary_type": pmap.primary_type,
            "total_files": pmap.total_files_count,
            "source_files": pmap.source_files_count,
            "test_files": pmap.test_files_count,
            "entry_point": pmap.primary_entry_point,
            "dependencies_count": sum(d.total_count for d in pmap.dependencies),
            "relevant_files": [f.path for f in (rel_ctx.relevant_files if 'rel_ctx' in locals() else [])[:5]],
        },
        "active_file": active_file,
        "terminal_output": terminal_output,
        "timestamp": time.strftime("%H:%M:%S"),
    }

    try:
        ide_memory.save_chat_message(proj_id, "agent", final_reply, result_payload)
    except Exception as e:
        print(f"[Memory Warning] Failed to save agent chat turn: {e}")

    return result_payload


# ==============================================================================
# FastAPI Application Implementation
# ==============================================================================
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Visual AI IDE Backend")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ProjectCreateRequest(BaseModel):
        name: str
        location: Optional[str] = "C:/jay/Projects"
        template: Optional[str] = "empty"
        description: Optional[str] = ""

    class ProjectOpenRequest(BaseModel):
        id: Optional[str] = None
        path: Optional[str] = None

    class ChatRequest(BaseModel):
        message: str
        project_id: Optional[str] = None

    class FileWriteRequest(BaseModel):
        content: str

    class CommandRequest(BaseModel):
        command: str

    @app.get("/api/health")
    def api_health():
        return {
            "status": "online",
            "active_project": current_project,
            "workspace": str(config.WORKSPACE_DIR),
            "python_version": sys.version.split()[0],
            "agent_state": "ready",
            "session_id": ide_memory.session_id,
        }

    # -------------------------------------------------------------
    # Project Management Endpoints
    # -------------------------------------------------------------
    @app.get("/api/projects")
    def api_list_projects():
        sync_projects_from_disk()
        projects = ide_memory.list_projects()
        enriched = []
        for p in projects:
            p_dir = Path(p["path"])
            file_count = get_project_file_count(p_dir)
            msg_count = ide_memory.get_chat_message_count(p["id"])
            enriched.append({
                **p,
                "file_count": file_count,
                "message_count": msg_count,
                "exists": p_dir.exists(),
            })
        return {"projects": enriched}

    @app.post("/api/projects")
    def api_create_project(req: ProjectCreateRequest):
        location_dir = Path(req.location or "C:/jay/Projects").resolve()
        target_dir = location_dir / req.name.strip().replace(" ", "_")
        target_dir.mkdir(parents=True, exist_ok=True)

        proj_id = f"proj_{req.name.strip().lower().replace(' ', '_')}_{str(uuid.uuid4())[:4]}"

        # Initialize starter files based on template
        if req.template == "python":
            (target_dir / "main.py").write_text(
                f'"""{req.name} Application."""\n\ndef main():\n    print("Hello from {req.name}")\n\nif __name__ == "__main__":\n    main()\n',
                encoding="utf-8",
            )
        elif req.template == "web":
            (target_dir / "index.html").write_text(
                f'<!DOCTYPE html>\n<html>\n<head>\n  <title>{req.name}</title>\n</head>\n<body>\n  <h1>{req.name}</h1>\n</body>\n</html>',
                encoding="utf-8",
            )

        saved = ide_memory.register_project(
            project_id=proj_id,
            name=req.name,
            path=str(target_dir),
            description=req.description or f"Project created with template '{req.template}'",
        )

        # Automatically open the created project
        config.set_workspace(target_dir)
        global current_project
        current_project = {
            "id": proj_id,
            "name": req.name,
            "path": str(target_dir),
        }

        return {"status": "created", "project": {**saved, "file_count": get_project_file_count(target_dir)}}

    @app.post("/api/projects/open")
    def api_open_project(req: ProjectOpenRequest):
        global current_project
        target = None

        if req.id:
            target = ide_memory.get_project(req.id)
        elif req.path:
            target = ide_memory.get_project(req.path)

        if not target and req.path:
            p_dir = Path(req.path).resolve()
            if p_dir.exists() and p_dir.is_dir():
                target = ide_memory.register_project(
                    project_id=f"proj_{p_dir.name.lower()}",
                    name=p_dir.name,
                    path=str(p_dir),
                    description="Opened from directory",
                )

        if not target:
            raise HTTPException(status_code=404, detail="Project not found")

        project_dir = Path(target["path"]).resolve()
        project_dir.mkdir(parents=True, exist_ok=True)
        config.set_workspace(project_dir)
        ide_memory.touch_project(target["id"])

        current_project = {
            "id": target["id"],
            "name": target["name"],
            "path": str(project_dir),
        }

        return {"status": "opened", "project": current_project}

    @app.get("/api/project/active")
    def api_get_active_project():
        return {"project": current_project}

    # -------------------------------------------------------------
    # Step 3B: Project Understanding Endpoints
    # -------------------------------------------------------------
    class ProjectAnalyzeRequest(BaseModel):
        query: Optional[str] = None

    @app.get("/api/projects/{project_id}/map")
    def api_project_map(project_id: str):
        proj = ide_memory.get_project(project_id)
        if not proj:
            if current_project and current_project.get("id") == project_id:
                proj = current_project
            else:
                raise HTTPException(status_code=404, detail="Project not found")
        p_path = Path(proj["path"])
        if not p_path.exists():
            raise HTTPException(status_code=404, detail="Project directory does not exist on disk")
        pmap = project_engine.understand_project(p_path)
        return pmap.to_dict()

    @app.get("/api/projects/{project_id}/summary")
    def api_project_summary(project_id: str):
        proj = ide_memory.get_project(project_id)
        if not proj:
            if current_project and current_project.get("id") == project_id:
                proj = current_project
            else:
                raise HTTPException(status_code=404, detail="Project not found")
        p_path = Path(proj["path"])
        if not p_path.exists():
            raise HTTPException(status_code=404, detail="Project directory does not exist on disk")
        pmap = project_engine.understand_project(p_path)
        return {
            "project_id": project_id,
            "project_name": pmap.project_name,
            "summary_text": pmap.to_summary_text(),
            "primary_type": pmap.primary_type,
            "total_files": pmap.total_files_count,
            "source_files": pmap.source_files_count,
            "test_files": pmap.test_files_count,
            "entry_points": [ep.path for ep in pmap.entry_points],
            "dependencies_count": sum(d.total_count for d in pmap.dependencies),
            "modules": list(pmap.relationships.keys())[:20],
        }

    @app.post("/api/projects/{project_id}/analyze")
    def api_project_analyze(project_id: str, req: Optional[ProjectAnalyzeRequest] = None):
        proj = ide_memory.get_project(project_id)
        if not proj:
            if current_project and current_project.get("id") == project_id:
                proj = current_project
            else:
                raise HTTPException(status_code=404, detail="Project not found")
        p_path = Path(proj["path"])
        if not p_path.exists():
            raise HTTPException(status_code=404, detail="Project directory does not exist on disk")
        query = req.query if req and req.query else ""
        if query:
            ctx = project_engine.get_relevant_context(p_path, query)
            return {
                "project_id": project_id,
                "query": query,
                "project_type": ctx.primary_type,
                "relevant_files": [
                    {
                        "path": f.path,
                        "relevance_score": round(f.score, 2),
                        "reasons": f.reasons,
                        "category": f.category,
                        "matched_symbols": f.matched_symbols,
                    }
                    for f in ctx.relevant_files
                ],
                "relevant_symbols": ctx.key_symbols,
                "dependencies": ctx.dependencies,
                "context_prompt": ctx.to_formatted_prompt_block(),
            }
        else:
            pmap = project_engine.refresh(p_path)
            return {
                "project_id": project_id,
                "status": "refreshed",
                "project_map": pmap.to_dict(),
            }

    # -------------------------------------------------------------
    # Workspace & Files Endpoints
    # -------------------------------------------------------------
    @app.get("/api/files")
    def api_list_files():
        return {"files": get_workspace_files(), "project": current_project}

    @app.get("/api/files/{file_path:path}")
    def api_get_file(file_path: str):
        target = config.WORKSPACE_DIR / file_path
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        content = target.read_text(encoding="utf-8", errors="replace")
        return {
            "name": target.name,
            "path": file_path,
            "content": content,
            "lines": len(content.splitlines()),
            "size_bytes": target.stat().st_size,
        }

    @app.post("/api/files/{file_path:path}")
    def api_save_file(file_path: str, req: FileWriteRequest):
        res = write_file(file_path, req.content, append=False)
        if not res.success:
            raise HTTPException(status_code=500, detail=res.error)
        return {"status": "saved", "output": res.output}

    @app.delete("/api/files/{file_path:path}")
    def api_delete_file(file_path: str):
        target = config.WORKSPACE_DIR / file_path
        if target.exists() and target.is_file():
            target.unlink()
            return {"status": "deleted", "path": file_path}
        raise HTTPException(status_code=404, detail="File not found")

    @app.post("/api/chat")
    def api_chat(req: ChatRequest):
        result = handle_agent_turn(req.message, project_id=req.project_id)
        return result

    @app.get("/api/projects/{project_id}/messages")
    def api_get_project_messages(project_id: str, limit: int = 200):
        messages = ide_memory.get_chat_messages(project_id, limit=limit)
        return {"messages": messages, "project_id": project_id}

    @app.delete("/api/projects/{project_id}/messages")
    def api_clear_project_messages(project_id: str):
        ide_memory.clear_chat_messages(project_id)
        return {"status": "cleared", "project_id": project_id}

    @app.get("/api/memory")
    def api_memory(limit: int = 30):
        """Return recent interactions from agent SQLite memory."""
        records = ide_memory.get_recent_history(limit=limit)
        history = []
        for r in reversed(records):
            history.append({
                "id": r.id,
                "session_id": r.session_id,
                "step": r.step,
                "user_input": r.user_input,
                "intent": r.intent,
                "tool_name": r.tool_name,
                "observation": r.observation[:200] if r.observation else "",
                "response": r.response[:200] if r.response else "",
                "timestamp": r.created_at,
            })
        return {"history": history, "session_id": ide_memory.session_id}

    @app.post("/api/run")
    def api_run_command(req: CommandRequest):
        res = run_command(req.command)
        return {
            "success": res.success,
            "output": res.output,
            "error": res.error,
            "metadata": res.metadata,
        }

    # -------------------------------------------------------------
    # Step 3C: Precise Patch Engine Endpoint
    # -------------------------------------------------------------
    class PatchApplyRequest(BaseModel):
        proposal: Optional[Dict[str, Any]] = None
        intent: Optional[str] = "MANUAL_PATCH"
        description: Optional[str] = ""
        files: Optional[List[Dict[str, Any]]] = None
        requires_approval: Optional[bool] = False

    @app.post("/api/projects/{project_id}/patch")
    def api_project_patch(project_id: str, req: PatchApplyRequest):
        proj = ide_memory.get_project(project_id)
        if proj and Path(proj.get("path", "")).exists():
            p_path = Path(proj["path"])
        elif Path(config.WORKSPACE_DIR).exists():
            p_path = Path(config.WORKSPACE_DIR)
        else:
            raise HTTPException(status_code=404, detail="Project directory does not exist on disk")

        from my_ai_agent.editing.patch_engine import PatchEngine
        from my_ai_agent.editing.edit_operations import EditProposal, FileEditProposal, EditOperation

        prop_data = req.proposal if req.proposal is not None else req.model_dump()
        files = []
        for fp in prop_data.get("files", []):
            ops = [EditOperation.from_dict(o) for o in fp.get("operations", [])]
            files.append(FileEditProposal(path=fp["path"], operations=ops, description=fp.get("description", "")))

        proposal = EditProposal(
            intent=prop_data.get("intent", "MANUAL_PATCH"),
            description=prop_data.get("description", ""),
            files=files,
            requires_approval=prop_data.get("requires_approval", False),
        )

        engine = PatchEngine(workspace_root=p_path)
        tx_res = engine.apply_proposal(proposal)
        return tx_res.to_dict()

    # -------------------------------------------------------------
    # Step 3D: Autonomous Task Engine Endpoints
    # -------------------------------------------------------------
    class AutonomousTaskRunRequest(BaseModel):
        message: str
        project_id: Optional[str] = None

    class ApprovalRequest(BaseModel):
        reason: Optional[str] = ""

    @app.post("/api/tasks/run")
    def api_run_autonomous_task(req: AutonomousTaskRunRequest):
        proj_id = req.project_id or current_project.get("id") or "default"
        proj = ide_memory.get_project(proj_id)
        if proj and Path(proj.get("path", "")).exists():
            w_root = Path(proj["path"])
        else:
            w_root = Path(config.WORKSPACE_DIR)

        auto_eng = AutonomousExecutionEngine(
            workspace_root=w_root,
            project_engine=project_engine,
        )
        task = auto_eng.run_task(user_request=req.message, project_id=proj_id)
        try:
            ide_memory.save_autonomous_task(task.to_dict())
        except Exception as e:
            print(f"[Memory Warning] Failed to save autonomous task: {e}")
        return task.to_dict()

    @app.get("/api/tasks/{task_id}")
    def api_get_task(task_id: str):
        t = auto_engine.get_task(task_id)
        if t:
            return t.to_dict()
        stored = ide_memory.get_autonomous_task(task_id)
        if stored:
            return stored
        raise HTTPException(status_code=404, detail="Task not found")

    @app.get("/api/tasks/{task_id}/events")
    def api_task_events(task_id: str):
        import queue

        def event_generator():
            q = broadcaster.subscribe(task_id)
            try:
                while True:
                    try:
                        event = q.get(timeout=20.0)
                        yield event.to_sse()
                        if event.event_type.value in (
                            "TASK_COMPLETED", "TASK_PARTIALLY_COMPLETED", "TASK_FAILED", "TASK_CANCELLED"
                        ):
                            break
                    except queue.Empty:
                        yield ": keep-alive\n\n"
            finally:
                broadcaster.unsubscribe(task_id, q)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/api/tasks/{task_id}/approve")
    def api_approve_task(task_id: str):
        ok = auto_engine.approve_action(task_id)
        t = auto_engine.get_task(task_id)
        if not ok or not t:
            raise HTTPException(status_code=400, detail="No pending approval found for this task")
        res_task = auto_engine.execute_task(t)
        try:
            ide_memory.save_autonomous_task(res_task.to_dict())
        except Exception:
            pass
        return res_task.to_dict()

    @app.post("/api/tasks/{task_id}/reject")
    def api_reject_task(task_id: str, req: ApprovalRequest):
        ok = auto_engine.reject_action(task_id, reason=req.reason or "Rejected by user")
        t = auto_engine.get_task(task_id)
        if not ok or not t:
            raise HTTPException(status_code=400, detail="Cannot reject task action")
        try:
            ide_memory.save_autonomous_task(t.to_dict())
        except Exception:
            pass
        return t.to_dict()

    @app.post("/api/tasks/{task_id}/cancel")
    def api_cancel_task(task_id: str):
        ok = auto_engine.cancel_task(task_id)
        t = auto_engine.get_task(task_id)
        if t:
            try:
                ide_memory.save_autonomous_task(t.to_dict())
            except Exception:
                pass
            return t.to_dict()
        return {"cancelled": ok}

    @app.post("/api/tasks/{task_id}/retry")
    def api_retry_task(task_id: str):
        res = auto_engine.retry_task(task_id)
        if not res:
            raise HTTPException(status_code=404, detail="Task not found or maximum attempts exceeded")
        try:
            ide_memory.save_autonomous_task(res.to_dict())
        except Exception:
            pass
        return res.to_dict()

    @app.get("/api/projects/{project_id}/tasks")
    def api_get_project_tasks(project_id: str, limit: int = 50):
        tasks = ide_memory.get_autonomous_tasks_for_project(project_id, limit=limit)
        return {"project_id": project_id, "tasks": tasks, "count": len(tasks)}

    # Model Intelligence and Health Endpoints (Step 5R, 5U)
    class ModelSelectRequest(BaseModel):
        provider: Optional[str] = None
        model_name: Optional[str] = None
        fallback: Optional[str] = None
        load_weights: Optional[bool] = False

    @app.get("/api/model/info")
    def api_model_info():
        from my_ai_agent.config import config
        from my_ai_agent.intelligence import router, registry
        active = router.route()
        health = active.health.to_dict()
        caps = active.capabilities.to_dict()
        return {
            "provider": getattr(config, "AI_MODEL_PROVIDER", "deterministic"),
            "configured_name": getattr(config, "AI_MODEL_NAME", "deterministic"),
            "active_model_id": active.model_id,
            "backend": getattr(config, "AI_MODEL_BACKEND", "transformers"),
            "device": getattr(config, "AI_MODEL_DEVICE", "cpu"),
            "fallback": getattr(config, "AI_MODEL_FALLBACK", "deterministic"),
            "health": health,
            "capabilities": caps,
            "registered_models": registry.list_models(),
        }

    @app.get("/api/model/health")
    def api_model_health():
        from my_ai_agent.intelligence import router
        active = router.route()
        return active.health.to_dict()

    @app.post("/api/model/select")
    def api_model_select(req: ModelSelectRequest):
        from my_ai_agent.config import config
        from my_ai_agent.intelligence import router, registry
        if req.provider:
            config.AI_MODEL_PROVIDER = req.provider.lower()
        if req.model_name:
            config.AI_MODEL_NAME = req.model_name
        if req.fallback:
            config.AI_MODEL_FALLBACK = req.fallback.lower()
        
        if req.load_weights and (req.provider == "local" or config.AI_MODEL_PROVIDER == "local"):
            local_model = registry.get("local")
            if local_model and hasattr(local_model, "load"):
                local_model.load()

        active = router.route()
        return {
            "status": "updated",
            "provider": config.AI_MODEL_PROVIDER,
            "active_model_id": active.model_id,
            "health": active.health.to_dict(),
        }


    # Serve static assets
    if (STATIC_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")
    if STATIC_FALLBACK.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_FALLBACK)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    def start_server(host: str = "127.0.0.1", port: int = 8000):
        print(f"\n[IDE Server] Launching Visual AI IDE Server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="warning")

except ImportError:
    pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    start_server(port=port)
