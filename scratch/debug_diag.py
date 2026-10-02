import sys
sys.path.insert(0, ".")
from my_ai_agent.autonomous.diagnosis import DiagnosisEngine
from my_ai_agent.autonomous.repair import RepairEngine
from my_ai_agent.editing.patch_engine import PatchEngine
from pathlib import Path
import tempfile

text = """F
======================================================================
FAIL: test_divide (test_calculator.TestCalculator.test_divide)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "C:\\Temp\\tests\\test_calculator.py", line 9, in test_divide
    self.assertEqual(divide(10, 2), 5)
AssertionError: 20 != 5
"""

rep = DiagnosisEngine.diagnose(text)
print("CATEGORY:", rep.category)
print("FAILING_SYM:", rep.failing_symbol)
print("AFFECTED_FILE:", rep.affected_file)

d = Path(tempfile.mkdtemp())
(d / "calculator.py").write_text("def divide(a, b):\n    return a * b\n")
engine = PatchEngine(workspace_root=d)
repair_res = RepairEngine.attempt_repair(rep, d, engine)
print("REPAIR SUCCESS:", repair_res.success)
print("ERROR:", repair_res.error)
