@echo off
REM AI Narrative Nexus - Run Script for Windows

echo.
echo ========================================
echo AI Narrative Nexus
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python is installed

REM Check if requirements are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] Installing dependencies...
    echo Please wait, this may take a few minutes on first run...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo [OK] All dependencies are ready

echo.
echo Starting FastAPI server...
echo.
echo [INFO] The application will be available at: http://localhost:8000
echo [INFO] Press Ctrl+C to stop the server
echo.

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
