@echo off
cd /d "%~dp0"
call venv311\Scripts\activate
python src\gui_app.py
pause
