@echo off
title Memory Optimization Pipeline
cd /d "%~dp0"

echo ============================================
echo   Memory Optimization Pipeline
echo ============================================
echo.

:: Use the correct Python
set PYTHON=C:\Users\z2cc_\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PYTHON%" set PYTHON=python

echo Checking dependencies...
"%PYTHON%" -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing Flask...
    "%PYTHON%" -m pip install flask -q
)

echo.
echo Starting server at http://localhost:8520
echo Close this window to stop.
echo ============================================
echo.

start "" http://localhost:8520
"%PYTHON%" server.py
pause
