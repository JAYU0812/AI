# 🤖 AI Agent Application — Real Project Understanding Engine (Step 3B)

The product philosophy is:
> **“I have an AI worker. I tell it what I want, and it understands my project, works on it, verifies the work, and reports the result.”**

The product is **NOT an IDE** and not a VS Code clone. The interface is conversation-first, project-first, and AI-first.

Everything displayed comes directly from the real selected project on disk — with **zero dummy data**.

---

## 🏛️ Upgraded Architecture

```text
User Request
      ↓
Agent (NLP & Context)
      ↓
Project Understanding Engine
      ↓
Project Map (Recursive Disk Scanner, AST, Manifests, Ignores)
      ↓
Relevant Files (Deterministic Relevance Engine)
      ↓
Targeted Context Builder (Sensitive Files Masked)
      ↓
Implementation Plan (Read Before Modify)
      ↓
Tool Execution Loop (READ_FILE ➔ WRITE_FILE ➔ RUN_COMMAND)
      ↓
Verification & Project Map Refresh
```

---

## 📦 Project Understanding Package (`my_ai_agent/project/`)

| Module | Responsibility |
| :--- | :--- |
| `scanner.py` | Recursively walks project root, enforces configurable ignore rules (`.git`, `node_modules`, `__pycache__`, `.venv`, `dist`, `build`, etc.), parses `.gitignore`, and emits real progress events. |
| `file_classifier.py` | Deterministically classifies files: `SOURCE`, `TEST`, `CONFIG`, `DEPENDENCY`, `DOCUMENTATION`, `ASSET`, `DATABASE`, `SCRIPT`, `BUILD`, `GENERATED`, `SENSITIVE`, `UNKNOWN`. |
| `dependency_analyzer.py` | Inspects real dependency manifests (`requirements.txt`, `pyproject.toml`, `package.json`, `pubspec.yaml`, `Cargo.toml`, `go.mod`, `composer.json`) without running package managers. |
| `symbol_analyzer.py` | Uses Python native `ast` to extract classes, methods, functions, decorators, imports, and entry blocks. JS/TS regex symbol extractor for web apps. |
| `analyzer.py` | Detects technology stacks with confidence & evidence (Python, FastAPI, Flask, Node, React, Flutter, Rust, Go, HTML/CSS/JS) and application entry points. |
| `project_map.py` | Structured `ProjectMap` data model with mtime caching for zero redundant scans. |
| `relevance.py` | Deterministic relevance scoring (0.0 to 1.0) using token stem matches, symbol lookups, imports, and domain concept expansions (e.g. `logout` ➔ `auth`, `login`, `session`, `user`). |
| `context_builder.py` | Generates compact, factual markdown context for the agent while completely masking sensitive files (`.env`, `.key`, secrets). |
| `project_understanding.py` | Unified facade orchestrator (`understand_project`, `get_relevant_context`, `explain_feature`, `refresh`). |

---

## 🔒 Security & Safety Boundaries

1. **Sensitive File Protection**:
   Files matching `.env*`, `*.key`, `*.pem`, `id_rsa`, or containing secrets/credentials are strictly classified as `SENSITIVE`.
   - Never exposed to model context prompts.
   - `read_file` tool actively blocks reading them.
2. **Destructive Command Blocking**:
   Dangerous system commands (e.g. `rm -rf /`, `rmdir /s /q C:\`, `del /f /s /q`, fork bombs) are intercepted and halted with `WAITING_FOR_APPROVAL`.
3. **Workspace Traversal Prevention**:
   All file tools verify paths remain strictly within the active project workspace.

---

## 🚀 Running the Application & Automated Tests

### 1. Launch Agent Server
```powershell
uv run --python 3.12 --with fastapi --with uvicorn python web_ide/server.py
```
Open browser to `http://127.0.0.1:8000`.

### 2. Run Comprehensive Test Suite
```powershell
uv run --python 3.12 --with fastapi --with uvicorn --with httpx python -m unittest discover -s my_ai_agent/tests
```
All 60/60 unit and end-to-end tests pass across Step 1, 2, 3B, 3C, and 3D.

---

# 🛠️ Step 3C — Precise Code Editing & Patch Engine

The Agent no longer relies on crude whole-file rewriting. It locates the exact symbol or code segment, creates atomic operations, computes factual unified diffs, validates syntax and structural integrity, executes project tests, and rolls back atomically if any failure occurs.

```text
User Request
      ↓
Intent & Relevance Analysis (Step 3B)
      ↓
Targeted Code & Symbol Context
      ↓
Structured Edit Proposal (EditProposal)
      ↓
Patch Engine (Atomic multi-file transaction)
      ↓
Pre-Validation (AST syntax, duplicate symbol/import prevention)
      ↓
Provisional Application & Unified Diff Generation
      ↓
Post-Validation & Test Selection (pytest, unittest, npm test, etc.)
      ↓
Error? ➔ Autonomous Repair Loop (MAX_AUTO_FIX_ATTEMPTS = 3)
      ↓
Commit on Success OR Complete Rollback on Failure
```

---

# ⚡ Step 3D — Autonomous Coding & Debugging Engine

The Agent can take a development request and autonomously carry it out end-to-end:
```text
User Request
      ↓
Understand Request & Project Architecture (Step 3B)
      ↓
Inspect Code & Create Implementation Plan
      ↓
Identify Files & Generate Surgical Actions
      ↓
Execute Action ➔ Capture Factual Observation
      ↓
Adapt Plan Based on Real Results
      ↓
Run Tests ➔ Detect Problems
      ↓
Diagnose Root Cause (11 Error Categories)
      ↓
Surgically Repair via Patch Engine (Step 3C)
      ↓
Re-test & Evidence-based Verification
      ↓
Report Factual Completion
```

### 1. Dedicated Autonomous Package (`my_ai_agent/autonomous/`)

| Module | Responsibility |
| :--- | :--- |
| `task.py` | Explicit `AutonomousTask` model tracking id, project, request, state, plan, budget, completed actions, observations, and timestamps. |
| `task_state.py` | Robust 13-state `TaskStateMachine` (`QUEUED`, `UNDERSTANDING`, `PLANNING`, `WAITING_FOR_APPROVAL`, `EXECUTING`, `TESTING`, `DIAGNOSING`, `REPAIRING`, `VERIFYING`, `COMPLETED`, `PARTIALLY_COMPLETED`, `FAILED`, `CANCELLED`). |
| `budget.py` | `TaskBudget` enforcing strict limits: `MAX_STEPS=30`, `MAX_REPAIR_ATTEMPTS=5`, `MAX_COMMANDS=20`, `MAX_FILES_CHANGED=50`, `MAX_EXECUTION_TIME=300s`. Prevents infinite loops. |
| `action.py` | Typed `AutonomousAction` model with risk classification (`LOW`, `MEDIUM`, `HIGH`). High-risk actions automatically transition to `WAITING_FOR_APPROVAL`. |
| `observation.py` | `ActionObservation` capturing actual stdout/stderr, diffs, exit codes, diagnostics, and reasoning without dummy data. |
| `diagnosis.py` | `DiagnosisEngine` classifying failures into 11 categories (`SYNTAX_ERROR`, `IMPORT_ERROR`, `TYPE_ERROR`, `RUNTIME_ERROR`, `TEST_FAILURE`, `CONFIGURATION_ERROR`, `DEPENDENCY_ERROR`, `FILE_NOT_FOUND`, `PERMISSION_ERROR`, `NETWORK_ERROR`, `UNKNOWN_ERROR`) with affected files, line numbers, and probable causes. |
| `repair.py` | `RepairEngine` synthesizing surgical, deterministic repair proposals and applying them through the Step 3C `PatchEngine`. |
| `completion.py` | `CompletionVerifier` providing factual evidence-based validation (AST syntax checks, test runner exit codes, required symbols). |
| `planner_controller.py` | `PlannerController` creating initial plans from Step 3B context and dynamically adapting when tests fail or requirements shift. |
| `action_executor.py` | `ActionExecutor` enforcing approval boundaries, workspace isolation, content synthesis, and patch execution. |
| `execution_loop.py` | `AutonomousExecutionEngine` driving the full autonomous cycle, pause on approval, cancel, retry, and event broadcasting. |
| `events.py` | `EventBroadcaster` with pub/sub queues and Server-Sent Events (`to_sse()`) for real-time frontend streaming. |

### 2. Real-Time Streaming & API Endpoints (`web_ide/server.py`)

- `POST /api/tasks/run`: Launch autonomous task in background thread with immediate task id return.
- `GET /api/tasks/{task_id}`: Poll current task state, budget meters, and history.
- `GET /api/tasks/{task_id}/events`: Real-time Server-Sent Events (SSE) stream of agent progress.
- `POST /api/tasks/{task_id}/approve`: Approve pending high-risk action in `WAITING_FOR_APPROVAL`.
- `POST /api/tasks/{task_id}/reject`: Reject pending high-risk action.
- `POST /api/tasks/{task_id}/cancel`: Safely stop a running autonomous task without corrupting project state.
- `POST /api/tasks/{task_id}/retry`: Controlled continuation from previous task state.
- `GET /api/projects/{project_id}/tasks`: Retrieve persistent task history for workspace.

### 3. Verification & Test Metrics
- **Previous Tests**: 41/41 PASS (Steps 1, 2, 3B, 3C)
- **Step 3D Tests**: 19/19 PASS (14 Unit Tests + 5 E2E Scenarios)
- **Step 4 Tests**: 16/16 PASS (15 Unit/Adversarial Tests + 1 E2E Scenario)
- **Total Tests**: **76/76 PASS**
- **Dummy Data Audit**: PASS (Zero fabricated progress, fake diffs, or mocked test results)

---

# 🧠 Step 4 — Intelligence Layer & Model Integration Foundation

The Agent transitions from a purely hardwired rule system to a **model-agnostic intelligence architecture** capable of hosting the current deterministic engine, local models (e.g. Llama.cpp, Ollama, ONNX), fine-tuned weights, or custom trained models.

**Architectural Principle**: No external LLM APIs (OpenAI, Anthropic, Gemini, etc.) are connected. The intelligence foundation establishes clean interfaces, strict structured outputs, context budgets, and safety boundaries before model weights are introduced.

```text
                         USER
                           │
                           ▼
                    CONVERSATION
                           │
                           ▼
                    INTELLIGENCE
                           │
                ┌──────────┴──────────┐
                │                     │
          MODEL ROUTER          CONTEXT MANAGER
                │                     │
                ▼                     ▼
        ┌──────────────┐       PROJECT CONTEXT
        │   AI MODEL   │
        └──────┬───────┘
               │
               ▼
       STRUCTURED DECISION
               │
               ▼
       SAFETY BOUNDARY (Path Traversal, Destructive Cmd, Sensitive Files)
               │
               ▼
            PLANNER
               │
               ▼
       AUTONOMOUS ENGINE (Step 3D)
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
     FILES   COMMANDS  TESTS
               │
               ▼
          OBSERVATION
               │
               ▼
      DIAGNOSIS & REPAIR (Step 3C PatchEngine)
               │
               ▼
         VERIFICATION
```

### 1. Dedicated Intelligence Package (`my_ai_agent/intelligence/`)

| Module | Responsibility |
| :--- | :--- |
| `model.py` | `AIModel` abstract base class defining `generate()`, `generate_structured()`, `stream()`, `model_id`, `capabilities`, and `health`. |
| `model_response.py` | Provider-agnostic `ModelResponse` with `FinishReason` and factual `TokenMetrics` (never fabricates token counts for rule engines). |
| `deterministic_adapter.py` | `DeterministicModel` wrapping the existing rule-based NLP, synthesis, diagnosis, and repair engines behind `AIModel`. |
| `local_provider.py` | `LocalModelProvider` abstraction for local inference runtimes without fake inference. |
| `model_registry.py` | Thread-safe `ModelRegistry` for registering, retrieving, listing, and swapping model providers. |
| `model_router.py` | `ModelRouter` directing intelligence requests to appropriate models based on task type and capability matching. |
| `context_manager.py` | `ContextManager` and `ContextBudget` prioritizing files and symbols via Step 3B relevance and masking sensitive files (`.env`, secrets). |
| `prompt_builder.py` | Standardized prompt constructor combining system roles, project context, observations, and structured schemas. |
| `structured_output.py` | `StructuredOutputValidator` validating model outputs against strict schemas (`action_decision`, `plan`, `edit_proposal`, `diagnosis`). |
| `safety_boundary.py` | `ModelSafetyBoundary` intercepting path traversals (`../../.env`), system root alterations, and destructive shell commands. |
| `telemetry.py` | `ModelTelemetry` tracking latency and errors while redacting passwords, API keys, and sensitive tokens. |
| `reasoning.py` | Safe, user-facing `ReasoningSummary` generating action-oriented explanations without private chain-of-thought hallucination. |
| `engines.py` | Modular capability interfaces: `IntentEngine`, `PlanningEngine`, `CodingEngine`, `DiagnosisIntelligence`, `RepairIntelligence`, `ExplanationEngine`. |

### 2. How to Add a New Model Provider

1. Subclass `AIModel` from `my_ai_agent.intelligence.model`.
2. Implement required methods: `generate()`, `generate_structured()`, `stream()`, `capabilities`, `health`.
3. Register the new provider:
   ```python
   from my_ai_agent.intelligence import registry
   registry.register("my_model", MyCustomModel(), make_default=True)
   ```
4. All agent planning, action execution, and safety checks will interact seamlessly through the standard interface.

---

## ⚡ Step 5 — Real Local / Open-Source Model Integration

- **Model:** `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- **Backend:** Hugging Face `transformers` + `torch` CPU adapter (also supports `ollama`).
- **Real Inference:** Proven live generation with factual tokenizer-derived token metrics and streaming token chunks.
- **Configurable Fallback:** `AI_MODEL_FALLBACK=deterministic` or `none` (honest surfacing of unavailability).
- **Adversarial Safety:** Path traversal, destructive command blocking, schema validation, and PatchEngine enforcement passed 100%.
- **Verification:** 90/90 tests passing (76 existing + 14 Step 5).

---

## 📊 Step 6A — Agent Intelligence Evaluation & Dataset Foundation

- **Dataset v1:** 104 machine-checkable tasks across 14 capability categories and 3 splits.
- **Evaluation Engine:** Sandboxed workspace execution protecting production codebase.
- **Scoring:** 12 multi-dimensional capabilities with safety as a non-negotiable critical constraint.
- **Failure Taxonomy:** Centralized failure classification across 15 failure types.
- **Baseline Results:** Official `baseline_v1.json` and Markdown scorecard `baseline_v1.md`.
- **Verification:** 107/107 tests passing (90 previous + 17 Step 6A).



