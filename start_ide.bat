@echo off
REM MY AI — Launcher
REM Automatically kills any process on port 8000, then starts fresh.
cd /d "%~dp0"

echo.
echo  MY AI — Starting...
echo.

REM Kill any process already on port 8000
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo  Stopping old server (PID %%a)...
    taskkill /PID %%a /F >nul 2>&1
    timeout /t 1 /nobreak >nul
)

echo  Launching server at http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"
uv run --python 3.12 --with fastapi --with uvicorn python web_ide/server.py
