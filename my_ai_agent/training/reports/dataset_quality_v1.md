# Step 6C — High-Quality Training Dataset Construction Report

## Executive Summary

- **Dataset Version:** 1.0
- **Creation Date:** 2026-10-01T13:18:46.487898+00:00
- **Total Candidates Generated:** 42
- **Total Accepted Examples:** 42
- **Total Rejected Examples:** 0
- **Acceptance Rate:** 100.0%
- **Average Quality Score:** 1.0 / 1.0000
- **Source Candidate Coverage:** 100.0% (40/40 verified Step 6B candidate tasks)

---

## Dataset Objective

The objective of **Step 6C** is to transform the verified weaknesses and diagnostic root causes identified in Step 6B into a high-quality, diverse, reproducible, and leakage-resistant instruction-tuning dataset.

The dataset instills seven core agent capabilities:
1. **Understand Request:** Accurately parse user instructions and natural language intents into canonical actions.
2. **Respect Constraints:** Adhere strictly to file scope limits, negative constraints, and docstring requirements.
3. **Select Correct Action:** Emit deterministic, valid tool actions (`READ_FILE`, `PATCH_FILE`, `SEARCH_CODE`, `INSPECT_PROJECT`, `RUN_TESTS`, `FINISH`).
4. **Produce Valid Structured Output:** Output pure, schema-compliant JSON conforming to `SchemaType.ACTION_DECISION` without conversational prose or markdown wrappers.
5. **Generate Correct Code:** Synthesize modular, PEP 8 / Python 3.12 compliant code with verified unit test assertions.
6. **Modify Code Surgically:** Execute AST-safe code modifications that preserve existing functions and avoid duplicate symbol definitions.
7. **Preserve Existing Behavior:** Safeguard non-targeted logic and existing regression test suites.

---

## Source Candidates

The dataset directly operationalizes the **40 fine-tuning candidates** isolated in `my_ai_agent/analysis/results/training_candidates_v1.json`:
- **Domain A (Novel Code Generation):** `code_gen_001` through `code_gen_010` (10 tasks)
- **Domain B (Surgical Code Modification):** `code_mod_002`, `003`, `005`, `006`, `007`, `009`, `010` (7 tasks)
- **Domain C (Strict Structured Output):** `struct_001` - `struct_006`, `intent_001` - `intent_006`, `tool_sel_001` - `tool_sel_006` (18 tasks)
- **Domain D (Instruction Following):** `instr_001` through `instr_005` (5 tasks)

*Total Candidates:* 40 / 40 (100% accounted for and traced).

---

## Dataset Schema

Every training example adheres strictly to the canonical schema:

```json
{
  "id": "train_000001",
  "version": "1.0",
  "task_type": "code_generation",
  "instruction": "...",
  "context": {
    "project_type": "python",
    "files": []
  },
  "constraints": [],
  "expected_behavior": "...",
  "target_output": "...",
  "output_type": "code",
  "tools_allowed": [],
  "verification": [],
  "provenance": {
    "source_task_id": "code_gen_001",
    "source_benchmark_version": "v1.0",
    "source_step": "6B",
    "source_root_cause": "MODEL_CAPABILITY",
    "source_category": "code_generation",
    "source_failure_type": "runtime_error"
  },
  "difficulty": "medium",
  "quality": {
    "validated": true,
    "duplicate": false,
    "leakage": false,
    "score": 0.95
  }
}
```

---

## Domain Distribution

| Domain / Task Type | Example Count | Percentage |
| :--- | :---: | :---: |
| **Novel Code Generation** (`code_generation`) | 12 | 28.57% |
| **Surgical Code Modification** (`code_modification`) | 7 | 16.67% |
| **Strict Structured Output** (`structured_output`) | 18 | 42.86% |
| **Instruction & Constraint Following** (`instruction_following`) | 5 | 11.9% |
| **Total** | **42** | **100.0%** |

---

## Difficulty Distribution

| Difficulty Level | Example Count | Percentage |
| :--- | :---: | :---: |
| **Easy** | 18 | 42.86% |
| **Medium** | 18 | 42.86% |
| **Hard** | 6 | 14.29% |

---

## Augmentation Strategy

To foster true model generalization rather than rote memorization:
- **No Paraphrase Spam:** Uncontrolled superficial rephrasing was strictly forbidden.
- **Controlled Diversity Dimensions:**
  1. *Boundary Variations:* Empty inputs, zero denominators, negative numbers, single-element collections.
  2. *Input Variations:* Scaling vectors, matrix products, parsing formats, type conversions.
  3. *Constraint Variations:* Single-file locks, negative constraints, docstring enforcement, immutable public API signatures.
- **Traceability:** Every augmented example records its `derived_from` parent ID and its `augmentation_type`.

---

## Validation Strategy

Each candidate example was subjected to multi-stage programmatic gate checks:
1. **Python AST Validation:** Checked via `ast.parse` to ensure zero syntax errors in code targets.
2. **JSON Schema Conformance:** Verified via `StructuredOutputValidator` against canonical `ACTION_DECISION_SCHEMA`. Preamble, conversational text, and markdown blocks were strictly rejected.
3. **Constraint Adherence:** Evaluated via rule checking on negative constraints, docstring presence, and file scopes.
4. **Execution Unit Tests:** Verification calls defined for every code example.

---

## Deduplication

- **Exact & Normalized Deduplication:** Implemented via SHA-256 digests over normalized instruction and target pairs.
- **Code AST Fingerprinting:** Identified identical AST structure sequences to eliminate redundant code samples.
- **Total Duplicates Isolated:** 0

---

## Leakage Prevention

- **Step 6A Benchmark Test Protection:** The 74 held-out evaluation tasks from Step 6A were indexed. Zero benchmark prompts or solutions were admitted into the training dataset.
- **Cross-Split Isolation:** Source candidates were partitioned using source-task grouped splitting. All augmentations and base examples sharing the same `source_task_id` were co-located in the same partition, preventing cross-split leakage.

---

## Security / Secret Scanning

- **Scan Coverage:** Regular expressions inspected for private cryptographic keys, API keys, bearer tokens, JWTs, database connection URIs, AWS keys, hardcoded passwords, and `.env` variable assignments.
- **PII Scanning:** Inspected for unmasked credit cards, social security numbers, and personal email addresses.
- **Detected Violations in Accepted Dataset:** 0 (100% clean).

---

## Training / Validation / Test Split

| Split Partition | Example Count | Percentage |
| :--- | :---: | :---: |
| **Training** (`training_dataset_v1.jsonl`) | 32 | 76.19% |
| **Validation** (`validation_dataset_v1.jsonl`) | 5 | 11.9% |
| **Test** (`test_dataset_v1.jsonl`) | 5 | 11.9% |

---

## Quality Statistics

- **Average Quality Score:** 1.0
- **Quality Distribution:**
  - 0.95 – 1.00: 42 examples
  - 0.90 – 0.94: 0 examples
  - 0.85 – 0.89: 0 examples

---

## Rejected Examples

A dedicated ledger of rejected samples is maintained in `rejected_examples_v1.jsonl` with explicit rejection reasons:
{}

---

## Dataset Hashes (SHA-256)

```text
training_dataset_v1.jsonl:   sha256:5593b4a17f34691cdb634576a8d0a52cd1c09a050be3695926ce3036f7ee5b07
validation_dataset_v1.jsonl: sha256:19f93efea0330679fd8806749b2eff953feef8dd5396f4f03f5165cdaa196327
test_dataset_v1.jsonl:       sha256:cbb06d3b13f82ed074229d24491b25a5f2ec12263bdc74c9f87dae4d9707acd9
rejected_examples_v1.jsonl:  sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
dataset_statistics_v1.json:  sha256:31e3b208b18f46c91276b7ae8da0c732280903f39137e25262779549ac712f7d
dataset_manifest_v1.json:    sha256:9f89e14b2e81567c3221be63e868e1467a5e5ee17632c2f98f6633e02eb3132a
```

---

## Known Limitations

1. **Format Scope:** Primary language target is Python 3.12. Multilingual datasets (TypeScript/Go/Rust) can be expanded in subsequent dataset versions.
2. **Context Size:** Examples are optimized for small-to-medium project contexts (1-10 files), keeping token overhead compact for local fine-tuning.

---

## Recommended Step 6D Experiment

- **Model Target:** SmolLM2-135M-Instruct or equivalent local foundation model.
- **Technique:** Parameter-Efficient Fine-Tuning (PEFT) with LoRA/QLoRA on rank $r=8$ or $r=16$.
- **Objective:** Compare pre-fine-tuned baseline (62.5% benchmark score) vs post-fine-tuned model on the untouched Step 6A held-out test suite.
