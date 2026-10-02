import unittest
from auth_service import AuthService

class TestAuthService(unittest.TestCase):
    def setUp(self):
        self.auth = AuthService()

    def test_login(self):
        token = self.auth.login("alice")
        self.assertTrue(self.auth.is_authenticated(token))

if __name__ == "__main__":
    unittest.main()
