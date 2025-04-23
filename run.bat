@echo off
REM Check if virtual environment exists
IF NOT EXIST .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate the virtual environment using PowerShell
echo Activating virtual environment...
powershell -ExecutionPolicy Bypass -Command ".venv\Scripts\Activate.ps1; pip install -r requirements.txt; python main.py"

REM Keep the window open
pause