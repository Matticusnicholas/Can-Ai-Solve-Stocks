@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo    Stock Filter Optimizer - Setup Script
echo ============================================================
echo.

:: Check for Python
echo [1/5] Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

:: Check Python version (need 3.9+)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if %MAJOR% lss 3 (
    echo [ERROR] Python 3.9+ required, found %PYTHON_VERSION%
    pause
    exit /b 1
)
if %MAJOR% equ 3 if %MINOR% lss 9 (
    echo [ERROR] Python 3.9+ required, found %PYTHON_VERSION%
    pause
    exit /b 1
)

:: Check for pip
echo.
echo [2/5] Checking for pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip is not installed
    echo Installing pip...
    python -m ensurepip --upgrade
)
echo [OK] pip is available

:: Create virtual environment
echo.
echo [3/5] Creating virtual environment...
if exist "venv" (
    echo [INFO] Virtual environment already exists
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

:: Activate virtual environment
echo.
echo [4/5] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

:: Install dependencies
echo.
echo [5/5] Installing dependencies...
echo This may take a few minutes...
echo.

pip install --upgrade pip >nul 2>&1

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Some packages may have failed to install
    echo Attempting to install core packages individually...

    pip install pandas numpy
    pip install yfinance
    pip install scikit-learn
    pip install xgboost lightgbm
    pip install fastapi uvicorn
    pip install ta pandas-ta
    pip install matplotlib plotly seaborn
    pip install tqdm joblib
)

echo.
echo ============================================================
echo    Setup Complete!
echo ============================================================
echo.
echo To run the application:
echo   1. Double-click 'run.bat' to start the web server
echo   2. Open http://localhost:8000 in your browser
echo.
echo Or use the command line:
echo   run.bat              - Start web server
echo   run.bat collect      - Collect stock data
echo   run.bat train        - Train ML models
echo   run.bat optimize     - Optimize filters
echo   run.bat all          - Run complete pipeline
echo.
pause
