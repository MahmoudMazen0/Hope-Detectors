@echo off
echo Starting Medical Diagnostics Dashboard...
cd /d "%~dp0"
call .\venv311\Scripts\python.exe gui_app.py
pause
