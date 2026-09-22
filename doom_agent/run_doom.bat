@echo off
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "PYTHONPATH=%PROJECT_ROOT%"
echo Launching OpenSourceJev DOOM Slayer Agent...
"%PROJECT_ROOT%\.venv\Scripts\python.exe" "%SCRIPT_DIR%doom_agent.py" %*
pause
