"""Dataset Generator for Step 6C.

Synthesizes high-quality, diverse, verified training examples derived from
the 40 verified Step 6B candidate weaknesses across the 4 core domains:
- Domain A: Novel Code Generation (code_gen_001 to 010)
- Domain B: Surgical Code Modification (code_mod_002 to 010)
- Domain C: Strict Structured Output (struct_001 to 006, intent_001 to 006, tool_sel_001 to 006)
- Domain D: Instruction & Constraint Following (instr_001 to 005)

Every target output is independently verified (AST syntax, JSON schema, constraint compliance).
No benchmark evaluation prompts or answers are memorized or leaked.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from my_ai_agent.training.provenance import ExampleProvenance
from my_ai_agent.training.schema import OutputType, QualityMetadata, TaskType, TrainingExample

CANDIDATES_FILE = (
    Path(__file__).resolve().parent.parent / "analysis" / "results" / "training_candidates_v1.json"
)


class TrainingDatasetGenerator:
    """Constructs verified training examples from Step 6B fine-tuning candidates."""

    def __init__(self, candidates_file: Optional[Path] = None):
        self.candidates_file = candidates_file or CANDIDATES_FILE
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self._load_candidates()

    def _load_candidates(self) -> None:
        """Loads the verified 40 Step 6B candidate metadata."""
        if self.candidates_file.exists():
            with open(self.candidates_file, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
                for item in raw_list:
                    self.candidates[item["task_id"]] = item

    def get_candidate_meta(self, task_id: str) -> Dict[str, Any]:
        """Returns metadata for a candidate, or defaults if not loaded."""
        return self.candidates.get(
            task_id,
            {
                "task_id": task_id,
                "category": "unknown",
                "difficulty": "medium",
                "primary_root_cause": "MODEL_CAPABILITY",
                "evidence": ["Reported failure taxonomy: runtime_error"],
            },
        )

    def _create_provenance(
        self,
        task_id: str,
        derived_from: Optional[str] = None,
        augmentation_type: Optional[str] = None,
    ) -> ExampleProvenance:
        meta = self.get_candidate_meta(task_id)
        evidence = " ".join(meta.get("evidence", []))
        fail_type = "runtime_error"
        if "StructuredOutput" in evidence or "structured" in evidence.lower():
            fail_type = "structured_output_invalid"
        elif "FileExists" in evidence or "SyntaxValid" in evidence:
            fail_type = "missing_file_or_syntax"

        return ExampleProvenance(
            source_task_id=task_id,
            source_benchmark_version="v1.0",
            source_step="6B",
            source_root_cause=meta.get("primary_root_cause", "MODEL_CAPABILITY"),
            source_category=meta.get("category", "code_generation"),
            source_failure_type=fail_type,
            derived_from=derived_from,
            augmentation_type=augmentation_type,
        )

    def generate_all(self) -> List[TrainingExample]:
        """Generates all training examples across all 4 domains."""
        examples: List[TrainingExample] = []

        # Domain A: Novel Code Generation (code_gen_001 to 010)
        examples.extend(self._generate_domain_a_code_gen())

        # Domain B: Surgical Code Modification (code_mod_002, 003, 005, 006, 007, 009, 010)
        examples.extend(self._generate_domain_b_code_mod())

        # Domain C: Strict Structured Output (struct_001-006, intent_001-006, tool_sel_001-006)
        examples.extend(self._generate_domain_c_structured_output())

        # Domain D: Instruction & Constraint Following (instr_001 to 005)
        examples.extend(self._generate_domain_d_instruction_following())

        return examples

    # ── DOMAIN A: NOVEL CODE GENERATION ───────────────────────────────

    def _generate_domain_a_code_gen(self) -> List[TrainingExample]:
        exs: List[TrainingExample] = []
        counter = 1

        # Candidate: code_gen_001 (Multiplication / Scaling logic)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Implement a function `scale_vector(vector: list[float], factor: float) -> list[float]` in `math_ops.py` that multiplies every element of a vector by a given scalar factor.",
                context={"project_type": "python", "files": [{"path": "math_ops.py", "content": ""}]},
                constraints=["Pure Python standard library only", "Must include type annotations and docstring", "Return a new list without mutating input"],
                expected_behavior="Multiplies each element of vector by factor",
                target_output='''def scale_vector(vector: list[float], factor: float) -> list[float]:
    """Scales all components in a numeric vector by a scalar factor.

    Args:
        vector: List of numeric float or integer values.
        factor: Scalar multiplier.

    Returns:
        A new list with scaled components.
    """
    return [x * factor for x in vector]
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE", "PATCH_FILE"],
                verification=[{"call": "scale_vector([1.0, 2.0, 3.0], 2.5)", "expected": [2.5, 5.0, 7.5]}],
                provenance=self._create_provenance("code_gen_001", augmentation_type="canonical"),
                difficulty="easy",
            )
        )
        counter += 1

        # code_gen_001 Augmentation: product of iterable
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Implement a function `compute_product(numbers: list[int | float]) -> float` in `math_ops.py` that computes the product of all elements, returning 1.0 for an empty list.",
                context={"project_type": "python", "files": [{"path": "math_ops.py", "content": ""}]},
                constraints=["Return 1.0 if numbers is empty", "Include docstring and type hints"],
                expected_behavior="Computes cumulative product of numbers",
                target_output='''def compute_product(numbers: list[int | float]) -> float:
    """Computes the multiplicative product of a list of numbers.

    Args:
        numbers: List of numeric values.

    Returns:
        The total product as a float, or 1.0 if empty.
    """
    if not numbers:
        return 1.0
    product = 1.0
    for num in numbers:
        product *= float(num)
    return product
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE", "PATCH_FILE"],
                verification=[{"call": "compute_product([2, 3, 4])", "expected": 24.0}],
                provenance=self._create_provenance("code_gen_001", derived_from=f"train_{counter-1:06d}", augmentation_type="input_variation"),
                difficulty="easy",
            )
        )
        counter += 1

        # Candidate: code_gen_002 (Safe division / ratio calculations)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Write a function `safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float` in `calculators.py` that computes numerator / denominator safely, returning default when denominator is zero.",
                context={"project_type": "python", "files": [{"path": "calculators.py", "content": ""}]},
                constraints=["Handle ZeroDivisionError by returning default value", "Provide complete type annotations"],
                expected_behavior="Computes division ratio with zero protection",
                target_output='''def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Calculates division safely, preventing ZeroDivisionError.

    Args:
        numerator: Dividend value.
        denominator: Divisor value.
        default: Fallback return value if divisor is zero.

    Returns:
        Division result as float, or default.
    """
    if denominator == 0.0:
        return default
    return float(numerator) / float(denominator)
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE", "PATCH_FILE"],
                verification=[{"call": "safe_ratio(10.0, 0.0, -1.0)", "expected": -1.0}],
                provenance=self._create_provenance("code_gen_002", augmentation_type="canonical"),
                difficulty="medium",
            )
        )
        counter += 1

        # code_gen_002 Augmentation: batch division / percentage
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Implement `compute_percentages(parts: list[float], total: float) -> list[float]` in `calculators.py` returning each part's percentage (0-100), returning all zeros if total <= 0.",
                context={"project_type": "python", "files": [{"path": "calculators.py", "content": ""}]},
                constraints=["Return empty list if parts is empty", "Return zeros if total <= 0"],
                expected_behavior="Computes percentage proportions",
                target_output='''def compute_percentages(parts: list[float], total: float) -> list[float]:
    """Calculates percentage share of each part relative to total.

    Args:
        parts: List of values.
        total: Denominator base for total percentage.

    Returns:
        List of percentages in 0-100 range.
    """
    if total <= 0.0 or not parts:
        return [0.0 for _ in parts]
    return [(part / total) * 100.0 for part in parts]
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE", "PATCH_FILE"],
                verification=[{"call": "compute_percentages([25, 75], 100)", "expected": [25.0, 75.0]}],
                provenance=self._create_provenance("code_gen_002", derived_from=f"train_{counter-1:06d}", augmentation_type="boundary_variation"),
                difficulty="medium",
            )
        )
        counter += 1

        # Candidate: code_gen_003 (Arbitrary module synthesis / math tools)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Create a module `geometry_tools.py` containing a function `euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float` that computes the geometric distance between two points.",
                context={"project_type": "python", "files": [{"path": "geometry_tools.py", "content": ""}]},
                constraints=["Use math module", "Ensure inputs are unpacked correctly", "Include docstrings"],
                expected_behavior="Computes sqrt((x2-x1)^2 + (y2-y1)^2)",
                target_output='''import math


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculates the Euclidean distance between two 2D coordinates.

    Args:
        p1: (x1, y1) coordinate tuple.
        p2: (x2, y2) coordinate tuple.

    Returns:
        Straight-line distance as float.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.hypot(dx, dy)
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE"],
                verification=[{"call": "euclidean_distance((0, 0), (3, 4))", "expected": 5.0}],
                provenance=self._create_provenance("code_gen_003", augmentation_type="canonical"),
                difficulty="medium",
            )
        )
        counter += 1

        # Candidate: code_gen_004 (Exponentiation / Power calculation)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Write a function `integer_power(base: int, exponent: int) -> int` in `exponent_tools.py` that raises base to a non-negative exponent, raising ValueError for negative exponents.",
                context={"project_type": "python", "files": [{"path": "exponent_tools.py", "content": ""}]},
                constraints=["Raise ValueError if exponent < 0", "Include docstrings and type annotations"],
                expected_behavior="Computes base ** exponent with non-negative validation",
                target_output='''def integer_power(base: int, exponent: int) -> int:
    """Computes integer exponentiation for non-negative powers.

    Args:
        base: Base integer.
        exponent: Non-negative integer power.

    Returns:
        Resulting integer power.

    Raises:
        ValueError: If exponent is negative.
    """
    if exponent < 0:
        raise ValueError("Exponent must be non-negative")
    return base ** exponent
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE", "PATCH_FILE"],
                verification=[{"call": "integer_power(2, 5)", "expected": 32}],
                provenance=self._create_provenance("code_gen_004", augmentation_type="canonical"),
                difficulty="easy",
            )
        )
        counter += 1

        # Candidate: code_gen_005 (String processing / Word formatting)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Create a module `string_transformers.py` with a function `slugify(text: str) -> str` that converts any string into a lowercase URL slug separated by hyphens.",
                context={"project_type": "python", "files": [{"path": "string_transformers.py", "content": ""}]},
                constraints=["Remove special characters", "Lowercase all characters", "Join words with single hyphen"],
                expected_behavior="Transforms 'Hello World! 2026' into 'hello-world-2026'",
                target_output='''import re


def slugify(text: str) -> str:
    """Transforms raw text into a normalized, hyphen-separated URL slug.

    Args:
        text: Raw input string.

    Returns:
        Normalized lowercase alphanumeric slug.
    """
    clean = re.sub(r"[^\\w\\s-]", "", text.strip().lower())
    return re.sub(r"[-\\s]+", "-", clean)
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE"],
                verification=[{"call": "slugify('Hello World 2026')", "expected": "hello-world-2026"}],
                provenance=self._create_provenance("code_gen_005", augmentation_type="canonical"),
                difficulty="easy",
            )
        )
        counter += 1

        # Candidate: code_gen_006 (String parsing / Tokenization)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Implement `extract_hashtags(post_content: str) -> list[str]` in `text_parser.py` that extracts all unique hashtag words without the '#' prefix in order of appearance.",
                context={"project_type": "python", "files": [{"path": "text_parser.py", "content": ""}]},
                constraints=["Preserve case-insensitive uniqueness while keeping first seen case", "Ignore lone '#'"],
                expected_behavior="Extracts list of hashtags",
                target_output='''import re


def extract_hashtags(post_content: str) -> list[str]:
    """Extracts unique hashtag identifiers from text in appearance order.

    Args:
        post_content: Raw message text containing hashtag tokens.

    Returns:
        List of hashtag strings excluding the '#' character.
    """
    tags = re.findall(r"#(\\w+)", post_content)
    seen = set()
    result = []
    for tag in tags:
        lower_tag = tag.lower()
        if lower_tag not in seen:
            seen.add(lower_tag)
            result.append(tag)
    return result
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE", "PATCH_FILE"],
                verification=[{"call": "extract_hashtags('Learning #Python and #AI! #python rocks')", "expected": ["Python", "AI"]}],
                provenance=self._create_provenance("code_gen_006", augmentation_type="canonical"),
                difficulty="medium",
            )
        )
        counter += 1

        # Candidate: code_gen_007 (List deduplication / Preserving order)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Write a function `deduplicate_preserve_order(items: list[Any]) -> list[Any]` in `collection_utils.py` that removes duplicates while preserving original sequence order.",
                context={"project_type": "python", "files": [{"path": "collection_utils.py", "content": ""}]},
                constraints=["Preserve initial element ordering", "Support unhashable items via representation fallback"],
                expected_behavior="Deduplicates elements keeping first occurrences",
                target_output='''from typing import Any


def deduplicate_preserve_order(items: list[Any]) -> list[Any]:
    """Removes duplicate items from list while strictly preserving first-seen order.

    Args:
        items: List of elements.

    Returns:
        New list containing only unique items in original order.
    """
    seen = set()
    result = []
    for item in items:
        try:
            key = item
            if key not in seen:
                seen.add(key)
                result.append(item)
        except TypeError:
            key = repr(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE", "PATCH_FILE"],
                verification=[{"call": "deduplicate_preserve_order([3, 1, 3, 2, 1, 4])", "expected": [3, 1, 2, 4]}],
                provenance=self._create_provenance("code_gen_007", augmentation_type="canonical"),
                difficulty="medium",
            )
        )
        counter += 1

        # Candidate: code_gen_008 (I/O and CSV utilities)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Create a module `csv_reader.py` with `read_csv_as_dicts(file_path: str) -> list[dict[str, str]]` that reads a CSV file and yields a list of row dictionaries.",
                context={"project_type": "python", "files": [{"path": "csv_reader.py", "content": ""}]},
                constraints=["Use built-in csv module", "Return empty list if file is empty or does not exist"],
                expected_behavior="Parses CSV file into dictionary records",
                target_output='''import csv
import os


def read_csv_as_dicts(file_path: str) -> list[dict[str, str]]:
    """Reads a CSV file into a list of dictionaries keyed by header row.

    Args:
        file_path: Absolute or relative filesystem path to the CSV file.

    Returns:
        List of column-to-value dictionaries. Returns empty list if missing.
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE"],
                verification=[{"call": "read_csv_as_dicts('non_existent.csv')", "expected": []}],
                provenance=self._create_provenance("code_gen_008", augmentation_type="canonical"),
                difficulty="hard",
            )
        )
        counter += 1

        # Candidate: code_gen_009 (Type converters)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Write a function `parse_boolean(value: Any, default: bool = False) -> bool` in `type_converters.py` that converts string ('true', 'yes', '1', 't') or numeric inputs to boolean.",
                context={"project_type": "python", "files": [{"path": "type_converters.py", "content": ""}]},
                constraints=["Case-insensitive string parsing", "Return default for unparseable values"],
                expected_behavior="Flexible boolean parsing",
                target_output='''from typing import Any


def parse_boolean(value: Any, default: bool = False) -> bool:
    """Parses various string, numeric, and primitive representations into boolean.

    Args:
        value: Input object to parse.
        default: Fallback boolean if value cannot be interpreted.

    Returns:
        Parsed boolean value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "t", "enable", "enabled", "on"):
            return True
        if v in ("false", "0", "no", "n", "f", "disable", "disabled", "off"):
            return False
    return default
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE", "PATCH_FILE"],
                verification=[{"call": "parse_boolean('YES')", "expected": True}, {"call": "parse_boolean('0')", "expected": False}],
                provenance=self._create_provenance("code_gen_009", augmentation_type="canonical"),
                difficulty="easy",
            )
        )
        counter += 1

        # Candidate: code_gen_010 (Bounds checking / Clamping values)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_GENERATION.value,
                instruction="Implement `clamp_value(val: float, min_val: float, max_val: float) -> float` in `numeric_bounds.py` that clamps val between min_val and max_val.",
                context={"project_type": "python", "files": [{"path": "numeric_bounds.py", "content": ""}]},
                constraints=["Raise ValueError if min_val > max_val", "Return float"],
                expected_behavior="Restricts value to range [min_val, max_val]",
                target_output='''def clamp_value(val: float, min_val: float, max_val: float) -> float:
    """Restricts a numeric value within inclusive minimum and maximum bounds.

    Args:
        val: Input number.
        min_val: Lower boundary.
        max_val: Upper boundary.

    Returns:
        Clamped value.

    Raises:
        ValueError: If min_val exceeds max_val.
    """
    if min_val > max_val:
        raise ValueError("min_val cannot be greater than max_val")
    return max(float(min_val), min(float(val), float(max_val)))
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE", "PATCH_FILE"],
                verification=[{"call": "clamp_value(15.0, 0.0, 10.0)", "expected": 10.0}],
                provenance=self._create_provenance("code_gen_010", augmentation_type="canonical"),
                difficulty="medium",
            )
        )
        counter += 1

        return exs

    # ── DOMAIN B: SURGICAL CODE MODIFICATION ──────────────────────────

    def _generate_domain_b_code_mod(self) -> List[TrainingExample]:
        exs: List[TrainingExample] = []
        counter = 101

        # Candidate: code_mod_002 (Insert method into existing math utility without duplicate symbols)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_MODIFICATION.value,
                instruction="Add a new function `multiply(a: int, b: int) -> int` to `utils.py` without modifying or removing existing `add` or `subtract` functions.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "utils.py",
                            "content": (
                                "def add(a: int, b: int) -> int:\n"
                                "    return a + b\n\n"
                                "def subtract(a: int, b: int) -> int:\n"
                                "    return a - b\n"
                            ),
                        }
                    ],
                },
                constraints=["Preserve existing add and subtract functions untouched", "Do not duplicate existing symbols", "Append new multiply function"],
                expected_behavior="Adds multiply function preserving existing code",
                target_output='''def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiplies two integers.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        Product of a and b.
    """
    return a * b
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"symbol_check": "multiply in utils.py", "existing_check": "add and subtract preserved"}],
                provenance=self._create_provenance("code_mod_002", augmentation_type="canonical"),
                difficulty="easy",
            )
        )
        counter += 1

        # Candidate: code_mod_003 (Add power method to Calculator class)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_MODIFICATION.value,
                instruction="Add a method `power(self, exponent: float) -> float` to the `Calculator` class in `calculator.py` that computes current value raised to exponent.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "calculator.py",
                            "content": (
                                "class Calculator:\n"
                                "    def __init__(self, value: float = 0.0):\n"
                                "        self.value = value\n\n"
                                "    def add(self, n: float) -> float:\n"
                                "        self.value += n\n"
                                "        return self.value\n"
                            ),
                        }
                    ],
                },
                constraints=["Preserve existing class methods", "Update internal self.value and return it", "Maintain class indentation"],
                expected_behavior="Adds power method to Calculator",
                target_output='''class Calculator:
    def __init__(self, value: float = 0.0):
        self.value = value

    def add(self, n: float) -> float:
        self.value += n
        return self.value

    def power(self, exponent: float) -> float:
        """Raises internal calculator value to the given exponent.

        Args:
            exponent: Exponent power.

        Returns:
            Updated internal value.
        """
        self.value = self.value ** exponent
        return self.value
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"symbol_check": "Calculator.power exists"}],
                provenance=self._create_provenance("code_mod_003", augmentation_type="canonical"),
                difficulty="medium",
            )
        )
        counter += 1

        # Candidate: code_mod_005 (Add active count query to AuthService)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_MODIFICATION.value,
                instruction="Add a method `get_active_count(self) -> int` to `AuthService` in `auth_service.py` that returns the number of active sessions.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "auth_service.py",
                            "content": (
                                "class AuthService:\n"
                                "    def __init__(self):\n"
                                "        self.active_sessions = set()\n\n"
                                "    def login(self, username: str) -> str:\n"
                                "        token = f'token_{username}'\n"
                                "        self.active_sessions.add(token)\n"
                                "        return token\n"
                            ),
                        }
                    ],
                },
                constraints=["Do not modify login logic", "Return integer count of active_sessions set"],
                expected_behavior="Adds get_active_count returning len(self.active_sessions)",
                target_output='''class AuthService:
    def __init__(self):
        self.active_sessions = set()

    def login(self, username: str) -> str:
        token = f"token_{username}"
        self.active_sessions.add(token)
        return token

    def get_active_count(self) -> int:
        """Returns the total number of currently active authenticated sessions.

        Returns:
            Integer count of active sessions.
        """
        return len(self.active_sessions)
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"symbol_check": "AuthService.get_active_count exists"}],
                provenance=self._create_provenance("code_mod_005", augmentation_type="canonical"),
                difficulty="hard",
            )
        )
        counter += 1

        # Candidate: code_mod_006 (Add modulo method to Calculator class)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_MODIFICATION.value,
                instruction="Add a method `modulo(self, divisor: float) -> float` to `Calculator` in `calculator.py`, raising ValueError if divisor is zero.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "calculator.py",
                            "content": (
                                "class Calculator:\n"
                                "    def __init__(self, value: float = 0.0):\n"
                                "        self.value = value\n"
                            ),
                        }
                    ],
                },
                constraints=["Check for zero divisor", "Raise ValueError on zero divisor", "Update and return self.value"],
                expected_behavior="Adds modulo method with ZeroDivision protection",
                target_output='''class Calculator:
    def __init__(self, value: float = 0.0):
        self.value = value

    def modulo(self, divisor: float) -> float:
        """Computes remainder of current value divided by divisor.

        Args:
            divisor: Non-zero divisor.

        Returns:
            Updated internal value.

        Raises:
            ValueError: If divisor is 0.
        """
        if divisor == 0.0:
            raise ValueError("Modulo by zero is undefined")
        self.value = self.value % divisor
        return self.value
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"symbol_check": "Calculator.modulo exists"}],
                provenance=self._create_provenance("code_mod_006", augmentation_type="canonical"),
                difficulty="easy",
            )
        )
        counter += 1

        # Candidate: code_mod_007 (Add clear_sessions method to AuthService)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_MODIFICATION.value,
                instruction="Add a method `clear_sessions(self) -> int` to `AuthService` in `auth_service.py` that clears all active sessions and returns how many were removed.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "auth_service.py",
                            "content": (
                                "class AuthService:\n"
                                "    def __init__(self):\n"
                                "        self.active_sessions = set()\n"
                            ),
                        }
                    ],
                },
                constraints=["Clears active_sessions set", "Returns integer number of sessions cleared"],
                expected_behavior="Adds clear_sessions method",
                target_output='''class AuthService:
    def __init__(self):
        self.active_sessions = set()

    def clear_sessions(self) -> int:
        """Clears all active authenticated user sessions.

        Returns:
            Count of sessions that were removed.
        """
        count = len(self.active_sessions)
        self.active_sessions.clear()
        return count
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"symbol_check": "AuthService.clear_sessions exists"}],
                provenance=self._create_provenance("code_mod_007", augmentation_type="canonical"),
                difficulty="medium",
            )
        )
        counter += 1

        # Candidate: code_mod_009 (Add square function to utils)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_MODIFICATION.value,
                instruction="Add `square(n: int | float) -> int | float` to `utils.py` without modifying existing functions.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "utils.py",
                            "content": (
                                "def add(a: int, b: int) -> int:\n"
                                "    return a + b\n"
                            ),
                        }
                    ],
                },
                constraints=["Preserve add function", "Append square function with docstring"],
                expected_behavior="Appends square function to utils.py",
                target_output='''def add(a: int, b: int) -> int:
    return a + b


def square(n: int | float) -> int | float:
    """Computes the square of a numeric value.

    Args:
        n: Number to square.

    Returns:
        n squared.
    """
    return n * n
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"symbol_check": "square in utils.py"}],
                provenance=self._create_provenance("code_mod_009", augmentation_type="canonical"),
                difficulty="easy",
            )
        )
        counter += 1

        # Candidate: code_mod_010 (Add revoke_all to AuthService)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.CODE_MODIFICATION.value,
                instruction="Add `revoke_all(self, prefix: str = '') -> int` to `AuthService` in `auth_service.py` that revokes sessions matching a prefix (or all if empty) and returns the count revoked.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "auth_service.py",
                            "content": (
                                "class AuthService:\n"
                                "    def __init__(self):\n"
                                "        self.active_sessions = set()\n"
                            ),
                        }
                    ],
                },
                constraints=["Revoke matching prefix tokens", "Return count revoked", "Handle empty prefix as all"],
                expected_behavior="Adds revoke_all method with prefix filtering",
                target_output='''class AuthService:
    def __init__(self):
        self.active_sessions = set()

    def revoke_all(self, prefix: str = "") -> int:
        """Revokes all active sessions matching an optional prefix token.

        Args:
            prefix: Token prefix to filter. Defaults to empty (matches all).

        Returns:
            Number of sessions revoked.
        """
        to_revoke = [s for s in self.active_sessions if s.startswith(prefix)]
        for token in to_revoke:
            self.active_sessions.remove(token)
        return len(to_revoke)
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"symbol_check": "AuthService.revoke_all exists"}],
                provenance=self._create_provenance("code_mod_010", augmentation_type="canonical"),
                difficulty="hard",
            )
        )
        counter += 1

        return exs

    # ── DOMAIN C: STRICT STRUCTURED OUTPUT ────────────────────────────

    def _generate_domain_c_structured_output(self) -> List[TrainingExample]:
        exs: List[TrainingExample] = []
        counter = 201

        # Candidates struct_001 to struct_006 (Canonical Action Decisions and Plans)
        struct_defs = [
            (
                "struct_001",
                "Generate a structured action decision to inspect the current project structure.",
                {
                    "action": "INSPECT_PROJECT",
                    "target": ".",
                    "reason": "Map out project file layout and dependencies before planning changes.",
                    "risk": "LOW",
                },
                "easy",
            ),
            (
                "struct_002",
                "Generate a structured action decision to run the pytest test suite in the project directory.",
                {
                    "action": "RUN_TESTS",
                    "target": "tests/",
                    "reason": "Verify that all unit tests currently pass before initiating edits.",
                    "risk": "LOW",
                },
                "medium",
            ),
            (
                "struct_003",
                "Generate a structured action decision to read the contents of config/settings.py.",
                {
                    "action": "READ_FILE",
                    "target": "config/settings.py",
                    "reason": "Inspect configuration parameters to locate environment definitions.",
                    "risk": "LOW",
                },
                "medium",
            ),
            (
                "struct_004",
                "Generate a structured action decision to apply a unified patch to calculator.py.",
                {
                    "action": "PATCH_FILE",
                    "target": "calculator.py",
                    "reason": "Add missing power method to Calculator class.",
                    "risk": "MEDIUM",
                },
                "hard",
            ),
            (
                "struct_005",
                "Generate a structured action decision to search the codebase for usages of `get_auth_token`.",
                {
                    "action": "SEARCH_CODE",
                    "target": "get_auth_token",
                    "reason": "Find all callers of authentication token method across the repository.",
                    "risk": "LOW",
                },
                "medium",
            ),
            (
                "struct_006",
                "Generate a structured action decision indicating task completion after successful verification.",
                {
                    "action": "FINISH",
                    "target": "",
                    "reason": "All requested functions were implemented, verified, and all unit tests passed.",
                    "risk": "LOW",
                },
                "easy",
            ),
        ]

        for task_id, instruction, decision_dict, diff in struct_defs:
            json_target = json.dumps(decision_dict, indent=2)
            exs.append(
                TrainingExample(
                    id=f"train_{counter:06d}",
                    version="1.0",
                    task_type=TaskType.STRUCTURED_OUTPUT.value,
                    instruction=f"{instruction} Output pure JSON conforming to SchemaType.ACTION_DECISION with no markdown block or conversational preamble.",
                    context={"project_type": "python", "files": []},
                    constraints=[
                        "Return valid JSON only",
                        "Must contain required fields: action, target, reason, risk",
                        "action must be a valid canonical action enum",
                        "No markdown code blocks (no ```json)",
                        "No conversational preamble",
                    ],
                    expected_behavior="Emits valid action_decision JSON payload",
                    target_output=json_target,
                    output_type=OutputType.JSON.value,
                    tools_allowed=[],
                    verification=[{"schema_validation": "SchemaType.ACTION_DECISION"}],
                    provenance=self._create_provenance(task_id, augmentation_type="canonical"),
                    difficulty=diff,
                )
            )
            counter += 1

        # Candidates intent_001 to intent_006 (Structured Intent & Plan Generation)
        intent_defs = [
            (
                "intent_001",
                "User request: 'Can you look through my repository and check what files we have?' Produce a structured action decision.",
                {
                    "action": "INSPECT_PROJECT",
                    "target": ".",
                    "reason": "User requested inspecting repository file contents.",
                    "risk": "LOW",
                },
                "easy",
            ),
            (
                "intent_002",
                "User request: 'Read main.py so we know how the application starts.' Produce a structured action decision.",
                {
                    "action": "READ_FILE",
                    "target": "main.py",
                    "reason": "Inspect main entry point to understand application bootstrapping.",
                    "risk": "LOW",
                },
                "medium",
            ),
            (
                "intent_003",
                "User request: 'Search for all occurrences of DatabaseConnection in the project.' Produce a structured action decision.",
                {
                    "action": "SEARCH_CODE",
                    "target": "DatabaseConnection",
                    "reason": "Search codebase for occurrences of DatabaseConnection class.",
                    "risk": "LOW",
                },
                "easy",
            ),
            (
                "intent_004",
                "User request: 'Run the pytest test suite to see if anything is currently broken.' Produce a structured action decision.",
                {
                    "action": "RUN_TESTS",
                    "target": "tests/",
                    "reason": "Execute project unit tests to detect any existing failures.",
                    "risk": "LOW",
                },
                "medium",
            ),
            (
                "intent_005",
                "User request: 'Create a new helper file string_utils.py with text manipulation functions.' Produce a structured action decision.",
                {
                    "action": "CREATE_FILE",
                    "target": "string_utils.py",
                    "reason": "Create the requested helper file string_utils.py for text utilities.",
                    "risk": "LOW",
                },
                "easy",
            ),
            (
                "intent_006",
                "User request: 'We have finished adding the feature and all tests passed. We are done.' Produce a structured action decision.",
                {
                    "action": "FINISH",
                    "target": "",
                    "reason": "Feature development and testing are confirmed complete.",
                    "risk": "LOW",
                },
                "hard",
            ),
        ]

        for task_id, instruction, decision_dict, diff in intent_defs:
            json_target = json.dumps(decision_dict, indent=2)
            exs.append(
                TrainingExample(
                    id=f"train_{counter:06d}",
                    version="1.0",
                    task_type=TaskType.STRUCTURED_OUTPUT.value,
                    instruction=f"{instruction} Output pure JSON conforming to SchemaType.ACTION_DECISION with no surrounding text.",
                    context={"project_type": "python", "files": []},
                    constraints=[
                        "Valid JSON object only",
                        "Must contain required fields: action, target, reason, risk",
                        "No markdown or conversational preamble",
                    ],
                    expected_behavior="Emits action decision corresponding to natural language intent",
                    target_output=json_target,
                    output_type=OutputType.JSON.value,
                    tools_allowed=[],
                    verification=[{"schema_validation": "SchemaType.ACTION_DECISION"}],
                    provenance=self._create_provenance(task_id, augmentation_type="canonical"),
                    difficulty=diff,
                )
            )
            counter += 1

        # Candidates tool_sel_001 to tool_sel_006 (Precise Tool & Parameter Selection)
        tool_defs = [
            (
                "tool_sel_001",
                "Select the appropriate tool action and parameters to inspect directory structure of '.'",
                {
                    "action": "INSPECT_PROJECT",
                    "target": ".",
                    "reason": "Inspect workspace file tree.",
                    "risk": "LOW",
                },
                "easy",
            ),
            (
                "tool_sel_002",
                "Select the appropriate tool action and parameters to read the file `auth/jwt.py`.",
                {
                    "action": "READ_FILE",
                    "target": "auth/jwt.py",
                    "reason": "Read contents of auth/jwt.py to verify token decoding.",
                    "risk": "LOW",
                },
                "easy",
            ),
            (
                "tool_sel_003",
                "Select the appropriate tool action and parameters to search for symbol `hash_password` in the codebase.",
                {
                    "action": "SEARCH_CODE",
                    "target": "hash_password",
                    "reason": "Locate definition and usages of hash_password function.",
                    "risk": "LOW",
                },
                "medium",
            ),
            (
                "tool_sel_004",
                "Select the appropriate tool action and parameters to create a new module `validators/email.py`.",
                {
                    "action": "CREATE_FILE",
                    "target": "validators/email.py",
                    "reason": "Create new validator file for email address format checking.",
                    "risk": "LOW",
                },
                "easy",
            ),
            (
                "tool_sel_005",
                "Select the appropriate tool action and parameters to apply patch edits to `utils/math.py`.",
                {
                    "action": "PATCH_FILE",
                    "target": "utils/math.py",
                    "reason": "Apply patch inserting factorial function.",
                    "risk": "MEDIUM",
                },
                "medium",
            ),
            (
                "tool_sel_006",
                "Select the appropriate tool action and parameters to run unit tests in `tests/test_math.py`.",
                {
                    "action": "RUN_TESTS",
                    "target": "tests/test_math.py",
                    "reason": "Execute test suite to verify math operations.",
                    "risk": "LOW",
                },
                "medium",
            ),
        ]

        for task_id, instruction, decision_dict, diff in tool_defs:
            json_target = json.dumps(decision_dict, indent=2)
            exs.append(
                TrainingExample(
                    id=f"train_{counter:06d}",
                    version="1.0",
                    task_type=TaskType.STRUCTURED_OUTPUT.value,
                    instruction=f"{instruction} Output pure JSON matching ACTION_DECISION schema.",
                    context={"project_type": "python", "files": []},
                    constraints=[
                        "Valid JSON object only",
                        "Required fields: action, target, reason, risk",
                        "Risk must be LOW, MEDIUM, or HIGH",
                    ],
                    expected_behavior="Selects appropriate tool action with parameters",
                    target_output=json_target,
                    output_type=OutputType.JSON.value,
                    tools_allowed=[],
                    verification=[{"schema_validation": "SchemaType.ACTION_DECISION"}],
                    provenance=self._create_provenance(task_id, augmentation_type="canonical"),
                    difficulty=diff,
                )
            )
            counter += 1

        return exs

    # ── DOMAIN D: INSTRUCTION & CONSTRAINT FOLLOWING ─────────────────

    def _generate_domain_d_instruction_following(self) -> List[TrainingExample]:
        exs: List[TrainingExample] = []
        counter = 301

        # Candidate: instr_001 (Single-file creation constraint / no external dependencies)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.INSTRUCTION_FOLLOWING.value,
                instruction="Create a module `math_helpers.py` containing `is_prime(n: int) -> bool`. STRICT CONSTRAINT: Modify only `math_helpers.py`. Do NOT create any other files. Do NOT install any external dependencies.",
                context={"project_type": "python", "files": [{"path": "math_helpers.py", "content": ""}]},
                constraints=[
                    "Modify only math_helpers.py",
                    "Do not create new files",
                    "Do not install dependencies",
                    "Must include docstring and type hints",
                ],
                expected_behavior="Creates is_prime in math_helpers.py respecting single-file boundary",
                target_output='''def is_prime(n: int) -> bool:
    """Determines whether an integer is a prime number.

    Args:
        n: Integer number to test.

    Returns:
        True if n is prime, False otherwise.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["CREATE_FILE"],
                verification=[{"call": "is_prime(17)", "expected": True}, {"call": "is_prime(18)", "expected": False}],
                provenance=self._create_provenance("instr_001", augmentation_type="canonical"),
                difficulty="easy",
            )
        )
        counter += 1

        # Candidate: instr_002 (Preserve public function signature / Docstring requirement)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.INSTRUCTION_FOLLOWING.value,
                instruction="In `utils.py`, add `multiply(a: int, b: int) -> int`. STRICT CONSTRAINT: Do not alter the signatures or behavior of existing functions. Add a complete Google-style docstring.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "utils.py",
                            "content": (
                                "def add(a: int, b: int) -> int:\n"
                                "    return a + b\n\n"
                                "def subtract(a: int, b: int) -> int:\n"
                                "    return a - b\n"
                            ),
                        }
                    ],
                },
                constraints=[
                    "Preserve public signatures of add and subtract",
                    "Add Google-style docstring with Args and Returns",
                    "Do not modify other files",
                ],
                expected_behavior="Appends multiply with docstring while preserving add and subtract",
                target_output='''def add(a: int, b: int) -> int:
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiplies two integers.

    Args:
        a: First integer multiplicand.
        b: Second integer multiplier.

    Returns:
        Product of a and b.
    """
    return a * b
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"symbol_check": "multiply exists", "docstring_check": "Args in docstring"}],
                provenance=self._create_provenance("instr_002", augmentation_type="canonical"),
                difficulty="medium",
            )
        )
        counter += 1

        # Candidate: instr_003 (Strict error handling constraint / Zero division protection)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.INSTRUCTION_FOLLOWING.value,
                instruction="In `utils.py`, add `divide(a: float, b: float) -> float`. STRICT CONSTRAINT: When b == 0, raise ZeroDivisionError with message 'Cannot divide by zero'. Do NOT return None or 0.0.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "utils.py",
                            "content": (
                                "def add(a: int, b: int) -> int:\n"
                                "    return a + b\n"
                            ),
                        }
                    ],
                },
                constraints=[
                    "Raise ZeroDivisionError('Cannot divide by zero') on b == 0",
                    "Do not return None or 0.0 on division by zero",
                    "Preserve existing add function",
                ],
                expected_behavior="Adds divide with explicit ZeroDivisionError exception message",
                target_output='''def add(a: int, b: int) -> int:
    return a + b


def divide(a: float, b: float) -> float:
    """Divides numerator a by denominator b.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        Quotient of a / b.

    Raises:
        ZeroDivisionError: If b is zero.
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return float(a) / float(b)
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"call": "divide(10.0, 2.0)", "expected": 5.0}],
                provenance=self._create_provenance("instr_003", augmentation_type="canonical"),
                difficulty="medium",
            )
        )
        counter += 1

        # Candidate: instr_004 (Class modification constraint / single target file)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.INSTRUCTION_FOLLOWING.value,
                instruction="In `calculator.py`, add method `power(self, exp: float) -> float` to class `Calculator`. STRICT CONSTRAINT: Modify only `calculator.py`. Do NOT touch any test files. Do NOT add standalone functions outside Calculator.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "calculator.py",
                            "content": (
                                "class Calculator:\n"
                                "    def __init__(self, value: float = 0.0):\n"
                                "        self.value = value\n"
                            ),
                        }
                    ],
                },
                constraints=[
                    "Modify only calculator.py",
                    "Do not touch test files",
                    "Do not add functions outside Calculator class",
                ],
                expected_behavior="Adds power method to Calculator class within calculator.py",
                target_output='''class Calculator:
    def __init__(self, value: float = 0.0):
        self.value = value

    def power(self, exp: float) -> float:
        """Raises internal calculator value to exp power.

        Args:
            exp: Exponent.

        Returns:
            Updated internal value.
        """
        self.value = self.value ** exp
        return self.value
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"symbol_check": "Calculator.power exists"}],
                provenance=self._create_provenance("instr_004", augmentation_type="canonical"),
                difficulty="hard",
            )
        )
        counter += 1

        # Candidate: instr_005 (Docstring & boolean return constraint on auth method)
        exs.append(
            TrainingExample(
                id=f"train_{counter:06d}",
                version="1.0",
                task_type=TaskType.INSTRUCTION_FOLLOWING.value,
                instruction="In `auth_service.py`, add method `is_valid_token(self, token: str) -> bool` to `AuthService`. STRICT CONSTRAINT: Return strictly boolean (True/False). Token must start with 'token_' and be non-empty. Include docstrings.",
                context={
                    "project_type": "python",
                    "files": [
                        {
                            "path": "auth_service.py",
                            "content": (
                                "class AuthService:\n"
                                "    def __init__(self):\n"
                                "        self.active_sessions = set()\n"
                            ),
                        }
                    ],
                },
                constraints=[
                    "Return boolean type only",
                    "Token must start with 'token_' and have length > 6",
                    "Include descriptive docstring",
                ],
                expected_behavior="Adds is_valid_token with strict format verification",
                target_output='''class AuthService:
    def __init__(self):
        self.active_sessions = set()

    def is_valid_token(self, token: str) -> bool:
        """Validates token structure according to session conventions.

        Args:
            token: Authentication token string.

        Returns:
            True if token format is valid, False otherwise.
        """
        if not isinstance(token, str):
            return False
        return token.startswith("token_") and len(token) > 6
''',
                output_type=OutputType.CODE.value,
                tools_allowed=["PATCH_FILE"],
                verification=[{"symbol_check": "AuthService.is_valid_token exists"}],
                provenance=self._create_provenance("instr_005", augmentation_type="canonical"),
                difficulty="easy",
            )
        )
        counter += 1

        return exs
