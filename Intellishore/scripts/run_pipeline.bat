@echo off
REM dengue forecasting pipeline execution script (Windows Batch)
REM runs the complete forecasting pipeline from data loading to visualization

echo ==================================
echo Dengue Forecasting Pipeline
echo ==================================
echo.

REM check if python is available
where python >nul 2>nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
) else (
    where python3 >nul 2>nul
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python3
    ) else (
        echo Error: Python is not installed or not in PATH
        exit /b 1
    )
)

echo Using: %PYTHON_CMD%
echo.

REM check if virtual environment exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No virtual environment found (optional)
)

echo.

REM check if required packages are installed
echo Checking dependencies...
%PYTHON_CMD% -c "import pandas, numpy, sklearn, matplotlib" >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing required packages...
    %PYTHON_CMD% -m pip install -r requirements.txt
) else (
    echo All dependencies satisfied ✓
)

echo.
echo Starting pipeline execution...
echo.

REM run the pipeline
%PYTHON_CMD% -m src.main

if %errorlevel% equ 0 (
    echo.
    echo ==================================
    echo Pipeline execution completed ✓
    echo ==================================
) else (
    echo.
    echo ==================================
    echo Pipeline execution failed ✗
    echo ==================================
    exit /b 1
)
