import unittest
from calculator import add, subtract, divide

class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_divide(self):
        self.assertEqual(divide(6, 2), 3)

if __name__ == "__main__":
    unittest.main()
