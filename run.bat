@echo off
REM Launcher script for Custom AI Agent & Visual IDE
cd /d "%~dp0"

if "%1"=="--demo" (
    uv run --python 3.12 python my_ai_agent/main.py --demo
) else if "%1"=="--test" (
    uv run --python 3.12 python -m unittest my_ai_agent/tests/test_agent.py
) else if "%1"=="--ui" (
    uv run --python 3.12 --with fastapi --with uvicorn python web_ide/server.py
) else (
    uv run --python 3.12 python my_ai_agent/main.py %*
)
