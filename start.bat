@echo off
title Memory Pipeline
cd /d "%~dp0"

echo ============================================
echo   Memory Optimization Pipeline
echo ============================================
echo.

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8520 ^| findstr LISTENING 2^>nul') do (
    echo Killing old server...
    taskkill /F /PID %%a 2>nul
)

set PYTHON=C:\Users\z2cc_\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PYTHON%" set PYTHON=python

"%PYTHON%" -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Flask...
    "%PYTHON%" -m pip install flask -q
)

echo.
echo Starting server — browser will open automatically
echo Close this window to stop.
echo ============================================
echo.

"%PYTHON%" server.py
pause
