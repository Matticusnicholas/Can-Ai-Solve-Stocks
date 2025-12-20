@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo    Stock Filter Optimizer
echo ============================================================
echo.

:: Check if virtual environment exists
if not exist "venv" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

:: Check command argument
set COMMAND=%1

if "%COMMAND%"=="" (
    set COMMAND=serve
)

if "%COMMAND%"=="serve" (
    echo Starting web server...
    echo.
    echo ============================================================
    echo    Web Dashboard Starting
    echo ============================================================
    echo.
    echo    Open your browser to: http://localhost:8000
    echo.
    echo    Press Ctrl+C to stop the server
    echo ============================================================
    echo.
    python main.py serve
    goto :end
)

if "%COMMAND%"=="collect" (
    echo Collecting S&P 500 data...
    echo This may take 10-20 minutes depending on your internet connection.
    echo.
    python main.py collect
    goto :end
)

if "%COMMAND%"=="train" (
    echo Training ML models [CPU]...
    echo This may take 5-15 minutes depending on your hardware.
    echo.
    python main.py train
    goto :end
)

if "%COMMAND%"=="train-gpu" (
    echo Training ML models with GPU/CUDA acceleration...
    echo Using XGBoost GPU, LightGBM GPU, and PyTorch LSTM
    echo.
    python main.py train-gpu %2 %3 %4 %5 %6 %7 %8 %9
    goto :end
)

if "%COMMAND%"=="optimize" (
    echo Optimizing filter criteria...
    echo.
    python main.py optimize
    goto :end
)

if "%COMMAND%"=="backtest" (
    echo Running backtest...
    echo.
    python main.py backtest
    goto :end
)

if "%COMMAND%"=="all" (
    echo Running complete pipeline...
    echo This will take 30-60 minutes.
    echo.
    python main.py all
    goto :end
)

if "%COMMAND%"=="help" (
    goto :showhelp
)

echo Unknown command: %COMMAND%
:showhelp
echo.
echo Usage: run.bat [command]
echo.
echo Commands:
echo   serve      Start web dashboard (default)
echo   collect    Download S&P 500 historical data
echo   train      Train ML models (CPU)
echo   train-gpu  Train ML models with GPU/CUDA acceleration
echo   optimize   Find optimal filter criteria
echo   backtest   Validate filters on historical data
echo   all        Run complete pipeline
echo   help       Show this help message
echo.
echo GPU Options (for train-gpu):
echo   --xgb-trees N     XGBoost trees (default: 1000)
echo   --lgb-trees N     LightGBM trees (default: 1000)
echo   --lstm-epochs N   LSTM epochs (default: 50)
echo   --no-lstm         Disable LSTM model
echo.

:end
pause
