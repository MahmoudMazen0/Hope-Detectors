@echo off
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Please check your internet connection or pip installation.
    pause
    exit /b
)

echo Starting Medical Predictor App...
python gui_app.py
pause
