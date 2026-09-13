@echo off
REM getdata one-click import for Windows (no AI, CSV descriptions + summary)
REM Location: D:\paper-factor-data\scripts\getdata.bat  (everything on D drive)
REM Usage: double-click or run in cmd: getdata.bat
setlocal EnableExtensions

REM ---- data root = parent of this script dir (scripts\.. = D:\paper-factor-data) ----
pushd "%~dp0.." >nul 2>nul
set "DATA_ROOT=%CD%"
popd >nul 2>nul

REM ---- project repo (for prompts.yaml): read from D:\...\repo_path.txt ----
set "PAPER_FACTOR_REPO="
if exist "%DATA_ROOT%\repo_path.txt" set /p PAPER_FACTOR_REPO=<"%DATA_ROOT%\repo_path.txt"

REM ---- import script is in this same directory ----
set "IMPORT_SCRIPT=%~dp0import_new_data.py"

echo ==========================================
echo  getdata import (Windows)
echo  DATA_ROOT: %DATA_ROOT%
echo  SCRIPT:    %IMPORT_SCRIPT%
if defined PAPER_FACTOR_REPO echo  REPO:      %PAPER_FACTOR_REPO%
echo ==========================================

REM ---- find python: conda rdagent env first, then python, then py ----
set "PY="
if exist "%USERPROFILE%\miniconda3\envs\rdagent\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\rdagent\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\miniconda3\envs\rdagent\python.exe" set "PY=%LOCALAPPDATA%\miniconda3\envs\rdagent\python.exe"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY where py >nul 2>nul && set "PY=py"

if not defined PY (
    echo [ERROR] Python not found. Install Python or configure conda env rdagent.
    pause
    exit /b 1
)
echo Using Python: %PY%
echo.

REM ---- 1. scan raw data ----
echo [1/4] scan raw data ...
"%PY%" "%IMPORT_SCRIPT%" --check
if errorlevel 1 goto :fail

REM ---- 2. import data ----
echo.
echo [2/4] import data ...
"%PY%" "%IMPORT_SCRIPT%"
if errorlevel 1 goto :fail

REM ---- 3. update prompt files ----
echo.
echo [3/4] update prompt files ...
"%PY%" "%IMPORT_SCRIPT%" --update-prompts-only
if errorlevel 1 goto :fail

REM ---- 4. summary ----
echo.
echo [4/4] summary ...
"%PY%" "%IMPORT_SCRIPT%" --summary

echo.
echo ==========================================
echo  OK - getdata done
echo ==========================================
pause
exit /b 0

:fail
echo.
echo  ERROR - getdata failed, check messages above.
pause
exit /b 1
