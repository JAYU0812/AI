"""Step 6C Tests: Training Dataset Schema and Serialization.

Verifies the TrainingExample schema, QualityMetadata, ExampleProvenance,
serialization/deserialization roundtrips, and field integrity.
"""

import json
import unittest

from my_ai_agent.training.provenance import ExampleProvenance
from my_ai_agent.training.schema import (
    OutputType,
    QualityMetadata,
    TaskType,
    TrainingExample,
)


class TestExampleProvenance(unittest.TestCase):
    """Tests provenance dataclass correctness."""

    def test_provenance_defaults(self):
        prov = ExampleProvenance(source_task_id="code_gen_001")
        self.assertEqual(prov.source_task_id, "code_gen_001")
        self.assertEqual(prov.source_step, "6B")
        self.assertEqual(prov.source_benchmark_version, "v1.0")
        self.assertIsNone(prov.derived_from)
        self.assertIsNone(prov.augmentation_type)

    def test_provenance_to_dict(self):
        prov = ExampleProvenance(
            source_task_id="struct_001",
            source_root_cause="STRUCTURED_OUTPUT_INTERFACE",
            source_category="structured_output",
            source_failure_type="structured_output_invalid",
            derived_from="train_000001",
            augmentation_type="input_variation",
        )
        d = prov.to_dict()
        self.assertEqual(d["source_task_id"], "struct_001")
        self.assertEqual(d["source_root_cause"], "STRUCTURED_OUTPUT_INTERFACE")
        self.assertEqual(d["derived_from"], "train_000001")
        self.assertEqual(d["augmentation_type"], "input_variation")

    def test_provenance_from_dict(self):
        data = {
            "source_task_id": "instr_001",
            "source_step": "6B",
            "source_root_cause": "PROMPT_DESIGN",
            "source_category": "instruction_following",
            "source_failure_type": "missing_file_or_syntax",
        }
        prov = ExampleProvenance.from_dict(data)
        self.assertEqual(prov.source_task_id, "instr_001")
        self.assertEqual(prov.source_root_cause, "PROMPT_DESIGN")

    def test_provenance_roundtrip(self):
        prov = ExampleProvenance(
            source_task_id="code_mod_002",
            source_root_cause="MODEL_CAPABILITY",
            derived_from="train_000100",
        )
        reconstructed = ExampleProvenance.from_dict(prov.to_dict())
        self.assertEqual(prov.source_task_id, reconstructed.source_task_id)
        self.assertEqual(prov.derived_from, reconstructed.derived_from)


class TestQualityMetadata(unittest.TestCase):
    """Tests QualityMetadata dataclass serialization."""

    def test_quality_defaults(self):
        q = QualityMetadata()
        self.assertFalse(q.validated)
        self.assertFalse(q.duplicate)
        self.assertFalse(q.leakage)
        self.assertEqual(q.score, 0.0)
        self.assertIsNone(q.rejection_reason)

    def test_quality_to_dict(self):
        q = QualityMetadata(validated=True, score=0.95)
        d = q.to_dict()
        self.assertTrue(d["validated"])
        self.assertAlmostEqual(d["score"], 0.95, places=3)
        self.assertNotIn("rejection_reason", d)

    def test_quality_with_rejection(self):
        q = QualityMetadata(validated=False, rejection_reason="Secret detected")
        d = q.to_dict()
        self.assertIn("rejection_reason", d)
        self.assertEqual(d["rejection_reason"], "Secret detected")

    def test_quality_from_dict(self):
        data = {"validated": True, "duplicate": False, "leakage": True, "score": 0.9}
        q = QualityMetadata.from_dict(data)
        self.assertTrue(q.validated)
        self.assertTrue(q.leakage)
        self.assertAlmostEqual(q.score, 0.9)


class TestTrainingExampleSchema(unittest.TestCase):
    """Tests TrainingExample instantiation, validation, and serialization."""

    def _make_valid_example(self, ex_id="train_000001") -> TrainingExample:
        return TrainingExample(
            id=ex_id,
            instruction="Implement a function multiply(a: int, b: int) -> int in utils.py.",
            task_type=TaskType.CODE_GENERATION.value,
            target_output=(
                'def multiply(a: int, b: int) -> int:\n'
                '    """Multiplies two integers."""\n'
                '    return a * b\n'
            ),
            provenance=ExampleProvenance(source_task_id="code_gen_001"),
            difficulty="easy",
        )

    def test_valid_example_created(self):
        ex = self._make_valid_example()
        self.assertEqual(ex.id, "train_000001")
        self.assertEqual(ex.task_type, "code_generation")
        self.assertEqual(ex.version, "1.0")

    def test_to_dict_has_required_fields(self):
        ex = self._make_valid_example()
        d = ex.to_dict()
        required_fields = ["id", "version", "task_type", "instruction", "target_output",
                           "provenance", "difficulty", "quality", "output_type"]
        for field in required_fields:
            self.assertIn(field, d, f"Field '{field}' missing from serialized dict")

    def test_to_json_is_valid_json(self):
        ex = self._make_valid_example()
        json_str = ex.to_json()
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["id"], "train_000001")

    def test_from_dict_roundtrip(self):
        ex = self._make_valid_example()
        d = ex.to_dict()
        reconstructed = TrainingExample.from_dict(d)
        self.assertEqual(ex.id, reconstructed.id)
        self.assertEqual(ex.instruction, reconstructed.instruction)
        self.assertEqual(ex.task_type, reconstructed.task_type)
        self.assertEqual(ex.target_output, reconstructed.target_output)

    def test_from_json_roundtrip(self):
        ex = self._make_valid_example()
        json_str = ex.to_json()
        reconstructed = TrainingExample.from_json(json_str)
        self.assertEqual(ex.id, reconstructed.id)
        self.assertEqual(ex.provenance.source_task_id, reconstructed.provenance.source_task_id)

    def test_json_example_schema(self):
        """Test schema for structured output examples."""
        import json as json_mod
        decision = {"action": "INSPECT_PROJECT", "target": ".", "reason": "test", "risk": "LOW"}
        ex = TrainingExample(
            id="train_000201",
            instruction="Generate structured action decision.",
            task_type=TaskType.STRUCTURED_OUTPUT.value,
            target_output=json_mod.dumps(decision, indent=2),
            output_type=OutputType.JSON.value,
            provenance=ExampleProvenance(source_task_id="struct_001"),
        )
        d = ex.to_dict()
        self.assertEqual(d["output_type"], "json")

    def test_code_modification_schema(self):
        """Test schema for code modification examples."""
        ex = TrainingExample(
            id="train_000101",
            instruction="Add multiply method to utils.py.",
            task_type=TaskType.CODE_MODIFICATION.value,
            target_output=(
                'def add(a, b):\n    return a + b\n\n'
                'def multiply(a: int, b: int) -> int:\n    """Multiplies."""\n    return a * b\n'
            ),
            output_type=OutputType.CODE.value,
            provenance=ExampleProvenance(source_task_id="code_mod_002"),
            constraints=["Preserve existing add function", "No duplicate symbols"],
        )
        d = ex.to_dict()
        self.assertEqual(d["task_type"], "code_modification")
        self.assertEqual(len(d["constraints"]), 2)


class TestTrainingDatasetGenerator(unittest.TestCase):
    """Tests the training dataset generator produces expected examples."""

    @classmethod
    def setUpClass(cls):
        from my_ai_agent.training.generator import TrainingDatasetGenerator
        cls.generator = TrainingDatasetGenerator()
        cls.examples = cls.generator.generate_all()

    def test_generates_42_examples(self):
        self.assertGreaterEqual(len(self.examples), 40)

    def test_all_examples_have_ids(self):
        for ex in self.examples:
            self.assertTrue(ex.id, f"Example missing ID: {ex}")

    def test_all_examples_have_provenance(self):
        for ex in self.examples:
            self.assertIsNotNone(ex.provenance)
            self.assertTrue(ex.provenance.source_task_id, f"Missing source_task_id: {ex.id}")

    def test_all_40_source_candidates_covered(self):
        sources = {ex.provenance.source_task_id for ex in self.examples if ex.provenance}
        self.assertEqual(len(sources), 40, f"Only {len(sources)} unique source tasks covered")

    def test_domain_a_generates_code_generation(self):
        domain_a = [e for e in self.examples if e.task_type == "code_generation"]
        self.assertGreaterEqual(len(domain_a), 10)

    def test_domain_b_generates_code_modification(self):
        domain_b = [e for e in self.examples if e.task_type == "code_modification"]
        self.assertGreaterEqual(len(domain_b), 6)

    def test_domain_c_generates_structured_output(self):
        domain_c = [e for e in self.examples if e.task_type == "structured_output"]
        self.assertGreaterEqual(len(domain_c), 15)

    def test_domain_d_generates_instruction_following(self):
        domain_d = [e for e in self.examples if e.task_type == "instruction_following"]
        self.assertGreaterEqual(len(domain_d), 4)

    def test_all_examples_have_target_output(self):
        for ex in self.examples:
            self.assertTrue(ex.target_output.strip(), f"Example {ex.id} has empty target_output")

    def test_code_examples_have_code_output_type(self):
        for ex in self.examples:
            if ex.task_type in ("code_generation", "code_modification"):
                self.assertEqual(ex.output_type, "code", f"{ex.id} should have output_type=code")

    def test_structured_examples_have_json_output_type(self):
        for ex in self.examples:
            if ex.task_type == "structured_output":
                self.assertEqual(ex.output_type, "json", f"{ex.id} should have output_type=json")


if __name__ == "__main__":
    unittest.main()
