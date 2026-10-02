"""Natural Language Processing (NLP) & Intent Classification Module for Step 2.

Performs intent classification, entity extraction, and slot-filling WITHOUT external LLMs,
using linguistic patterns, token scoring, and semantic feature analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import re


class IntentType(str, Enum):
    # Compound / Multi-Step Intents (Step 2 & 3B)
    BUILD_AND_RUN = "BUILD_AND_RUN"       # Create + Synthesize + Write + Run + Verify
    BUILD_FILE = "BUILD_FILE"             # Create + Synthesize + Write
    REFACTOR_FILE = "REFACTOR_FILE"       # Read + Modify + Write + Run
    UNDERSTAND_PROJECT = "UNDERSTAND_PROJECT"  # Inspect + Map + Explain / Factual analysis

    # Atomic Intents (Step 1 baseline)
    CREATE_FILE = "CREATE_FILE"
    WRITE_FILE = "WRITE_FILE"
    READ_FILE = "READ_FILE"
    LIST_FILES = "LIST_FILES"
    RUN_COMMAND = "RUN_COMMAND"

    # System Intents
    SHOW_MEMORY = "SHOW_MEMORY"
    CLEAR_MEMORY = "CLEAR_MEMORY"
    HELP = "HELP"
    EXIT = "EXIT"
    UNKNOWN = "UNKNOWN"


@dataclass
class ExtractedEntities:
    language: str = "python"
    filename: Optional[str] = None
    purpose: str = ""
    inputs: List[Any] = field(default_factory=list)
    run_after: bool = False
    verify_output: bool = True
    raw_slots: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NLUResult:
    intent: IntentType
    entities: ExtractedEntities
    confidence: float
    raw_input: str
    tokens: List[str]
    explanation: str


class NaturalLanguageUnderstanding:
    """Zero-LLM Intent Classifier and Entity Extractor."""

    # Stopwords that don't change core intent
    POLITE_PREFIXES = {
        "can you", "could you", "would you", "please", "can you please",
        "could you please", "i want to", "i need to", "help me",
        "make me", "create me", "build me", "write me"
    }

    # Common programming task keywords for purpose extraction
    TASK_KEYWORDS = {
        # Specific task nouns first (most specific → least specific)
        "calculator": "calculator.py",
        "fibonacci": "fibonacci.py",
        "factorial": "factorial.py",
        "prime":     "prime_checker.py",
        "average":   "average_calculator.py",
        "mean":      "average_calculator.py",
        "reverse":   "string_reverser.py",
        "sort":      "sorter.py",
        "count":     "word_counter.py",
        "hello":     "hello.py",
        # Generic last
        "sum":       "sum_calculator.py",
        "add":       "addition.py",
    }

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """Strip redundant whitespace, normalize quotes."""
        t = text.strip()
        t = t.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        return t

    def extract_filename(self, text: str) -> Optional[str]:
        """Extract explicit filename from text or return None."""
        # Matches foo.py, test.txt, script.py, etc.
        match = re.search(r"([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)", text)
        if match:
            return match.group(1)
        return None

    def extract_entities(self, text: str) -> ExtractedEntities:
        """Extracts structured entities: language, filename, purpose, inputs, run_after."""
        entities = ExtractedEntities()
        lower = text.lower()

        # 1. Language detection
        if "python" in lower or ".py" in lower:
            entities.language = "python"
        elif "javascript" in lower or "js" in lower or ".js" in lower:
            entities.language = "javascript"
        elif "bash" in lower or "shell" in lower or ".sh" in lower:
            entities.language = "bash"

        # 2. Run-after detection
        # Catches: "and run it", "then run it", "run it", "execute it",
        #          "Run it and fix", "run and verify", "and test it"
        if re.search(
            r"(?:and\s+)?(?:then\s+)?(?:run|execute|test)\s+(?:it|the\s+file|this|code|and)"
            r"|\brun\s+it\b|\bexecute\s+it\b|\btest\s+it\b",
            lower
        ):
            entities.run_after = True

        # Also detect: "Run it and fix any errors", "run and fix errors"
        if re.search(r"\brun\b.*\bfix\b|\bfix\b.*\berror\b", lower):
            entities.run_after = True

        # 3. Explicit filename
        explicit_fn = self.extract_filename(text)
        if explicit_fn:
            entities.filename = explicit_fn

        # 4. Purpose extraction
        # e.g., "that calculates the average of three numbers" -> "calculates the average of three numbers"
        purpose_match = re.search(
            r"\b(?:that|which|to)\b\s+([a-zA-Z0-9_\s\+\-\*\/\(\)\,\.]+?)(?:\s+and\s+(?:then\s+)?(?:run|execute)|[.!?]|$)",
            text,
            re.IGNORECASE,
        )
        if purpose_match:
            entities.purpose = purpose_match.group(1).strip()
        else:
            # Fallback: extract purpose from keywords in the full message
            for kw in self.TASK_KEYWORDS:
                if kw in lower:
                    entities.purpose = f"{kw} program"
                    break

        # Additional: if purpose is still empty, derive from the full message (first sentence)
        if not entities.purpose:
            first_sentence = re.split(r"[.!?]|\band\s+run\b|\band\s+execute\b", text)[0].strip()
            # Remove polite prefixes
            for prefix in sorted(self.POLITE_PREFIXES, key=len, reverse=True):
                if first_sentence.lower().startswith(prefix):
                    first_sentence = first_sentence[len(prefix):].strip()
                    break
            if len(first_sentence) > 4:
                entities.purpose = first_sentence[:120]

        # If no explicit filename, infer appropriate filename from purpose
        if not entities.filename:
            for kw, default_fn in self.TASK_KEYWORDS.items():
                if kw in lower or kw in entities.purpose.lower():
                    entities.filename = default_fn
                    break
            if not entities.filename:
                ext = ".py" if entities.language == "python" else ".txt"
                entities.filename = f"generated_script{ext}"

        # 5. Extract input values (e.g., "three numbers", "numbers 10, 20, 30", "[1, 2, 3]")
        if "three" in lower or "3" in lower:
            entities.inputs = [10, 20, 30]
        elif "two" in lower or "2" in lower:
            entities.inputs = [15, 25]

        # Extract explicit numbers if present: "numbers 5, 10, 15"
        num_matches = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        if num_matches and len(num_matches) >= 2:
            entities.inputs = [float(n) if "." in n else int(n) for n in num_matches]

        return entities

    def parse(self, user_input: str, last_file: Optional[str] = None) -> NLUResult:
        """Parse natural language into NLUResult with intent, entities, and confidence."""
        raw = self.clean_text(user_input)
        lower = raw.lower()
        tokens = [t.strip(",.?!") for t in lower.split()]

        # 1. System Intents
        if lower in ("exit", "quit", "q", "bye"):
            return NLUResult(IntentType.EXIT, ExtractedEntities(), 1.0, raw, tokens, "User requested exit")
        if lower in ("help", "?", "commands"):
            return NLUResult(IntentType.HELP, ExtractedEntities(), 1.0, raw, tokens, "User requested help")
        if lower in ("history", "memory", "show history", "show memory"):
            return NLUResult(IntentType.SHOW_MEMORY, ExtractedEntities(), 1.0, raw, tokens, "User requested memory")
        if lower in ("clear", "reset", "clear memory"):
            return NLUResult(IntentType.CLEAR_MEMORY, ExtractedEntities(), 1.0, raw, tokens, "User requested clear")

        # 2. Extract entities
        entities = self.extract_entities(raw)

        # 3. Intent Detection
        # 3. Intent Detection
        # Check for explicit WRITE_FILE: "Write <content> into <file>"
        write_match = re.search(r"^(?:write|add|put|insert)\s+(.*?)\s+(?:into|to|in)\s+([^\s]+)$", raw, re.I)
        if write_match:
            content = write_match.group(1).strip()
            target_f = write_match.group(2).strip().strip("'\"")
            if target_f in ("it", "the file") and last_file:
                target_f = last_file
            entities.filename = target_f
            entities.raw_slots["content"] = content
            return NLUResult(
                intent=IntentType.WRITE_FILE,
                entities=entities,
                confidence=0.95,
                raw_input=raw,
                tokens=tokens,
                explanation=f"Write content to '{target_f}'.",
            )

        # Check for LIST_FILES
        if lower in ("ls", "dir", "list files", "show files", "list directory", "files") or (
            "list" in tokens and "files" in tokens
        ):
            return NLUResult(
                intent=IntentType.LIST_FILES,
                entities=entities,
                confidence=0.98,
                raw_input=raw,
                tokens=tokens,
                explanation="List files in workspace.",
            )

        # Check for UNDERSTAND_PROJECT (Step 3B)
        understand_signals = any(k in lower for k in (
            "understand this project", "understand the project", "analyze this project",
            "analyze the project", "explain this project", "explain the project",
            "how does", "how does the", "how does authentication", "how does auth",
            "tell me how", "what does this project do", "project summary",
            "project map", "inspect project", "architecture of this project",
            "explore project", "scan project",
        ))
        if understand_signals:
            query_subject = raw
            for prefix in ("understand this project and tell me", "tell me", "understand this project and explain", "explain", "understand this project"):
                if lower.startswith(prefix):
                    query_subject = raw[len(prefix):].strip(" :,.-")
                    break
            entities.purpose = query_subject or "project overview"
            return NLUResult(
                intent=IntentType.UNDERSTAND_PROJECT,
                entities=entities,
                confidence=0.96,
                raw_input=raw,
                tokens=tokens,
                explanation=f"Project Understanding inquiry: '{entities.purpose[:60]}'.",
            )

        # Check for explicit atomic CREATE_FILE: "create test.py", "create empty file test.py", "touch main.py"
        atomic_create = re.search(r"^(?:create|touch|make)\s+(?:an?\s+)?(?:empty\s+)?(?:file\s+)?([^\s]+\.[a-zA-Z0-9_]+)$", lower)
        if atomic_create:
            target_f = atomic_create.group(1).strip("'\"")
            entities.filename = target_f
            return NLUResult(
                intent=IntentType.CREATE_FILE,
                entities=entities,
                confidence=0.98,
                raw_input=raw,
                tokens=tokens,
                explanation=f"Create file '{target_f}'.",
            )

        # Check for REFACTOR_FILE (Step 3B: Read existing code before modifying)
        refactor_match = re.search(r"^(?:add|update|modify|change|refactor|fix|extend)\s+(.*?)(?:\s+(?:to|in)\s+([^\s]+\.[a-zA-Z0-9_]+))?$", raw, re.I)
        if refactor_match and any(k in lower for k in ("function", "method", "feature", "endpoint", "route", "logout", "percentage", "bug", "support")):
            purpose = refactor_match.group(1).strip()
            explicit_file = refactor_match.group(2)
            target_f = explicit_file.strip("'\"") if explicit_file else (last_file or entities.filename)
            entities.filename = target_f
            entities.purpose = purpose
            return NLUResult(
                intent=IntentType.REFACTOR_FILE,
                entities=entities,
                confidence=0.94,
                raw_input=raw,
                tokens=tokens,
                explanation=f"Refactor/modify existing code: '{purpose[:60]}'.",
            )

        # Check for compound BUILD_AND_RUN or BUILD_FILE
        # Pattern signals: any creation verb + any "thing being built" + optional run
        is_creation = any(k in lower for k in (
            "make", "create", "build", "write", "generate", "implement",
            "develop", "code", "add"
        ))

        # Broad file/product concept — includes task nouns like "calculator", "app", "tool"
        has_file_concept = any(k in lower for k in (
            "file", "script", "program", "code", ".py", ".js", ".html",
            "app", "application", "tool", "function", "module", "class",
            "calculator", "checker", "generator", "converter", "manager",
            "tracker", "analyzer", "parser", "server", "api", "website",
        ))

        # If no explicit file concept found but we have a purpose and creation verb,
        # treat purpose itself as the file concept ("create a calculator" → file concept = calculator)
        if is_creation and not has_file_concept and entities.purpose:
            has_file_concept = True

        has_run_request = entities.run_after or any(k in lower for k in (
            "and run it", "then run it", "run it", "execute it", "run and",
            "and execute", "fix any errors", "fix errors",
        ))

        if is_creation and has_file_concept and has_run_request:
            return NLUResult(
                intent=IntentType.BUILD_AND_RUN,
                entities=entities,
                confidence=0.96,
                raw_input=raw,
                tokens=tokens,
                explanation=f"Compound: Build '{entities.filename}' for '{entities.purpose[:60]}' and run.",
            )

        if is_creation and has_file_concept and entities.purpose:
            return NLUResult(
                intent=IntentType.BUILD_FILE,
                entities=entities,
                confidence=0.92,
                raw_input=raw,
                tokens=tokens,
                explanation=f"Build file '{entities.filename}' for '{entities.purpose[:60]}'.",
            )

        # Final fallback: if we have a purpose and a run request but no creation verb,
        # treat it as BUILD_AND_RUN (e.g., "Python calculator that runs")
        if has_file_concept and has_run_request and entities.purpose:
            return NLUResult(
                intent=IntentType.BUILD_AND_RUN,
                entities=entities,
                confidence=0.80,
                raw_input=raw,
                tokens=tokens,
                explanation=f"Inferred build+run for '{entities.purpose[:60]}'.",
            )

        # Check for READ_FILE: "Read test.py", "Show me test.py", "Cat it"
        read_match = re.search(r"^(?:read|cat|show|view|display)\s+(?:file\s+)?([^\s]+)$", lower)
        if read_match:
            target_f = read_match.group(1).strip("'\"")
            if target_f in ("it", "the file") and last_file:
                target_f = last_file
            entities.filename = target_f
            return NLUResult(
                intent=IntentType.READ_FILE,
                entities=entities,
                confidence=0.95,
                raw_input=raw,
                tokens=tokens,
                explanation=f"Read file '{target_f}'.",
            )

        # Check for RUN_COMMAND: "Run test.py", "Run python test.py", "Run it"
        run_match = re.search(r"^(?:run|execute|exec)\s+(.+)$", raw, re.I)
        if run_match:
            cmd = run_match.group(1).strip()
            if cmd.lower() in ("it", "the file") and last_file:
                cmd = f"python {last_file}"
            elif cmd.endswith(".py") and not cmd.startswith("python"):
                cmd = f"python {cmd}"
            entities.raw_slots["command"] = cmd
            return NLUResult(
                intent=IntentType.RUN_COMMAND,
                entities=entities,
                confidence=0.95,
                raw_input=raw,
                tokens=tokens,
                explanation=f"Run command '{cmd}'.",
            )

        # Check for simple CREATE_FILE: "Create test.py", "Touch main.py"
        create_match = re.search(r"^(?:create|touch|make)\s+(?:a\s+)?(?:file\s+)?([^\s]+)$", lower)
        if create_match and not entities.purpose:
            entities.filename = create_match.group(1).strip("'\"")
            return NLUResult(
                intent=IntentType.CREATE_FILE,
                entities=entities,
                confidence=0.90,
                raw_input=raw,
                tokens=tokens,
                explanation=f"Create file '{entities.filename}'.",
            )

        # Fallback unknown
        return NLUResult(
            intent=IntentType.UNKNOWN,
            entities=entities,
            confidence=0.1,
            raw_input=raw,
            tokens=tokens,
            explanation="Unrecognized input pattern.",
        )
