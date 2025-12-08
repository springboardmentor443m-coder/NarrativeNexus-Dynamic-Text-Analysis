@echo off
REM ============================================================
REM NarrativeNexus - Start All Servers
REM ============================================================
echo.
echo ============================================================
echo Starting NarrativeNexus Platform
echo ============================================================
echo.

cd /d D:\InfosysNarrativeNexus

REM Kill any existing Python processes first
echo Stopping any existing Python processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

REM Start FastAPI Backend
echo [1/2] Starting FastAPI Backend (Port 8000)...
start "NarrativeNexus Backend" /MIN cmd /k "cd /d D:\InfosysNarrativeNexus && venv\Scripts\activate.bat && uvicorn src.main:app --host 127.0.0.1 --port 8000"

REM Wait for backend to fully load
echo Waiting for backend to load models...
timeout /t 12 /nobreak >nul

REM Start Flask Frontend
echo [2/2] Starting Flask UI (Port 5000)...
start "NarrativeNexus Frontend" /MIN cmd /k "cd /d D:\InfosysNarrativeNexus && venv\Scripts\activate.bat && python ui\app.py"

REM Wait for Flask to start
timeout /t 5 /nobreak >nul

echo.
echo ============================================================
echo SERVERS STARTED SUCCESSFULLY!
echo ============================================================
echo Frontend UI:  http://127.0.0.1:5000
echo Backend API:  http://127.0.0.1:8000
echo API Docs:     http://127.0.0.1:8000/docs
echo ============================================================
echo.
echo Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:5000

echo.
echo ============================================================
echo Servers are running in minimized windows
echo To stop: Close the "NarrativeNexus Backend" and 
echo          "NarrativeNexus Frontend" windows
echo ============================================================
echo.
pause
