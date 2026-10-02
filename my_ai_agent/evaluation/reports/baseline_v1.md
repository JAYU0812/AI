# Agent Intelligence Benchmark Report — deterministic

**Generated on:** 2026-10-01 18:24:10  
**Dataset Version:** V1 | **Platform:** Windows AMD64  
**Overall Score:** `62.5%` (50/104 Passed, 0 Safety Violations)

## 1. Executive Summary

The Agent Intelligence Evaluation Suite executed `104` machine-checkable benchmark tasks against `deterministic`. The model achieved a weighted overall performance of **62.5%** with **50** tasks fully passing, **23** tasks partially passing, and **31** task failures. Critical safety enforcement recorded **0** violations.

| Metric | Result |
| :--- | :--- |
| **Overall Score** | **`62.5%`** |
| **Tasks Passed** | `50` / `104` (48.1%) |
| **Tasks Partial** | `23` / `104` (22.1%) |
| **Tasks Failed** | `31` / `104` (29.8%) |
| **Safety Violations** | **`0`** |
| **Total Execution Time** | `86.7s` |

## 2. Hardware & Runtime Profile

| Specification | Value |
| :--- | :--- |
| **OS** | Windows 11 (10.0.26200) |
| **Architecture** | AMD64 |
| **Python Version** | 3.12.14 |
| **Model ID** | `deterministic` |

## 3. Capability Category Scores

| Category | Tasks | Average Score | Status |
| :--- | :---: | :---: | :--- |
| **Debugging** | 10 | `93.8%` | 🟢 Excellent |
| **Project Understanding** | 8 | `91.7%` | 🟢 Excellent |
| **Verification** | 6 | `91.7%` | 🟢 Excellent |
| **General Understanding** | 6 | `86.1%` | 🟢 Excellent |
| **Context Usage** | 6 | `85.8%` | 🟢 Excellent |
| **Autonomous Completion** | 6 | `74.0%` | 🟡 Good |
| **Code Modification** | 10 | `66.9%` | 🟡 Good |
| **Safety** | 10 | `65.8%` | 🟡 Good |
| **Multi Step** | 8 | `64.6%` | 🟡 Good |
| **Instruction Following** | 6 | `49.0%` | 🔴 Needs Improvement |
| **Code Generation** | 10 | `36.5%` | 🔴 Needs Improvement |
| **Intent Understanding** | 6 | `16.7%` | 🔴 Needs Improvement |
| **Structured Output** | 6 | `16.7%` | 🔴 Needs Improvement |
| **Tool Selection** | 6 | `16.7%` | 🔴 Needs Improvement |

## 4. Safety Evaluation & Critical Boundaries

> [!NOTE]
> **Zero safety violations detected.** All adversarial attempts (path traversal, destructive commands, sensitive file access, and engine bypass) were safely intercepted and blocked by the ModelSafetyBoundary.

## 5. Failure Taxonomy Analysis

| Failure Type | Occurrences | Description |
| :--- | :---: | :--- |
| `runtime_error` | 20 | Runtime Error |
| `structured_output_failure` | 6 | Structured Output Failure |
| `test_failure` | 5 | Test Failure |

## 7. Reproducibility Information

To reproduce this exact benchmark run locally:
```powershell
uv run --python 3.12 python -m my_ai_agent.evaluation.runner --dataset v1 --model deterministic
```
