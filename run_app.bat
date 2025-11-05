@echo off
REM FinLlama Sentiment Analysis App Launcher
REM Ensures PyTorch loads before Streamlit to avoid DLL conflicts

echo.
echo ================================================================================
echo FinLlama: Advanced Financial Sentiment Analysis
echo ================================================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Run the Python launcher script
python run_app.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: Failed to start the application
    echo ================================================================================
    echo.
    echo Please check that:
    echo   1. Python is installed and in your PATH
    echo   2. All dependencies are installed: pip install -r requirements_modern.txt
    echo   3. PyTorch and transformers are properly installed
    echo.
    pause
)
