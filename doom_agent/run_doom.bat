@echo off
set PYTHONPATH=..
echo Launching Jev DOOM Slayer Agent...
..\.venv\Scripts\python.exe doom_agent.py %*
pause
