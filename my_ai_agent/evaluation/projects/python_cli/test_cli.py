import unittest
from cli import parse_args, run_cli

class TestCLI(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_args(["greet"]), "greet")

    def test_run(self):
        self.assertEqual(run_cli(["greet"]), "Hello World")

if __name__ == "__main__":
    unittest.main()
