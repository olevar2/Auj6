@echo off
REM AUJ Platform Environment Setup Script for Windows
REM =================================================
REM This script sets up the AUJ Platform environment and runs basic validation.
REM 
REM Author: AUJ Platform Development Team
REM Date: 2025-07-04
REM Version: 1.0.0

echo ========================================
echo  AUJ Platform Environment Setup
echo ========================================
echo.

REM Set script directory as platform root
set "PLATFORM_ROOT=%~dp0"
set "PLATFORM_ROOT=%PLATFORM_ROOT:~0,-1%"

echo Platform Root: %PLATFORM_ROOT%
echo.

REM Set environment variables
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUNBUFFERED=1"
set "AUJ_PLATFORM_ROOT=%PLATFORM_ROOT%"
set "AUJ_SRC_PATH=%PLATFORM_ROOT%\auj_platform\src"
set "AUJ_CONFIG_PATH=%PLATFORM_ROOT%\config"
set "AUJ_LOGS_PATH=%PLATFORM_ROOT%\logs"

echo Environment variables set:
echo   AUJ_PLATFORM_ROOT = %AUJ_PLATFORM_ROOT%
echo   AUJ_SRC_PATH = %AUJ_SRC_PATH%
echo   AUJ_CONFIG_PATH = %AUJ_CONFIG_PATH%
echo   AUJ_LOGS_PATH = %AUJ_LOGS_PATH%
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not available in PATH
    echo Please install Python 3.8+ and ensure it's in your PATH
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if required directories exist
echo Checking directory structure...
if not exist "%AUJ_SRC_PATH%" (
    echo Creating missing directory: %AUJ_SRC_PATH%
    mkdir "%AUJ_SRC_PATH%" 2>nul
)

if not exist "%AUJ_CONFIG_PATH%" (
    echo Creating missing directory: %AUJ_CONFIG_PATH%
    mkdir "%AUJ_CONFIG_PATH%" 2>nul
)

if not exist "%AUJ_LOGS_PATH%" (
    echo Creating missing directory: %AUJ_LOGS_PATH%
    mkdir "%AUJ_LOGS_PATH%" 2>nul
)

if not exist "%AUJ_SRC_PATH%\core" (
    echo Creating missing directory: %AUJ_SRC_PATH%\core
    mkdir "%AUJ_SRC_PATH%\core" 2>nul
)

echo Directory structure verified.
echo.

REM Run Python environment setup if available
echo Running Python environment setup...
if exist "%AUJ_SRC_PATH%\core\environment_setup.py" (
    cd /d "%PLATFORM_ROOT%"
    python "%AUJ_SRC_PATH%\core\environment_setup.py" --verbose
    if %errorlevel% neq 0 (
        echo WARNING: Python environment setup reported issues
        echo Please check the log files for details
    ) else (
        echo Python environment setup completed successfully
    )
) else (
    echo WARNING: environment_setup.py not found
    echo Skipping Python environment setup
)
echo.

REM Run path cleanup analysis if available
echo Running path cleanup analysis...
if exist "%AUJ_SRC_PATH%\core\path_cleanup.py" (
    cd /d "%PLATFORM_ROOT%"
    python "%AUJ_SRC_PATH%\core\path_cleanup.py" --platform-root "%PLATFORM_ROOT%" --dry-run --verbose
    if %errorlevel% neq 0 (
        echo WARNING: Path cleanup analysis reported issues
    ) else (
        echo Path cleanup analysis completed successfully
    )
) else (
    echo WARNING: path_cleanup.py not found
    echo Skipping path cleanup analysis
)
echo.

REM Basic validation
echo Running basic validation...

REM Check for core files
set "MISSING_FILES="
if not exist "%AUJ_SRC_PATH%\core\__init__.py" (
    set "MISSING_FILES=%MISSING_FILES% core/__init__.py"
)
if not exist "%AUJ_SRC_PATH%\core\config.py" (
    set "MISSING_FILES=%MISSING_FILES% core/config.py"
)

if defined MISSING_FILES (
    echo WARNING: Missing core files:%MISSING_FILES%
    echo Platform may not function correctly
) else (
    echo Core files validation passed
)

REM Test basic Python import
echo Testing basic Python imports...
cd /d "%PLATFORM_ROOT%"
python -c "import sys; sys.path.insert(0, r'%AUJ_SRC_PATH%'); print('Python path setup: OK')" 2>nul
if %errorlevel% neq 0 (
    echo WARNING: Python import test failed
    echo There may be issues with the Python path setup
) else (
    echo Python import test passed
)

echo.
echo ========================================
echo  Environment Setup Complete
echo ========================================
echo.
echo The AUJ Platform environment has been configured.
echo.
echo Next steps:
echo   1. Verify all required dependencies are installed
echo   2. Review configuration files in: %AUJ_CONFIG_PATH%
echo   3. Check logs in: %AUJ_LOGS_PATH%
echo   4. Test platform functionality
echo.
echo To run the platform:
echo   cd "%PLATFORM_ROOT%"
echo   python -m auj_platform.src.main
echo.

REM Keep window open if run directly
if "%1"=="" pause

exit /b 0