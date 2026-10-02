class AuthService:
    def __init__(self):
        self.active_sessions = set()

    def login(self, username: str) -> str:
        token = f"token_{username}"
        self.active_sessions.add(token)
        return token

    def is_authenticated(self, token: str) -> bool:
        return token in self.active_sessions
