@echo off
REM install_chart_dependencies.bat
REM Windows installation script for AUJ Platform Chart Analysis Dependencies

echo 🚀 Installing AUJ Platform Chart Analysis Dependencies...

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install Python and pip first.
    pause
    exit /b 1
)

REM Install basic requirements
echo 📦 Installing basic requirements...
pip install plotly>=5.17.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0

REM Try to install TA-Lib
echo 🔧 Installing TA-Lib...
pip install TA-Lib==0.4.29
if errorlevel 1 (
    echo ⚠️ TA-Lib installation failed. The chart analysis will use fallback implementations.
    echo 💡 To install TA-Lib manually on Windows:
    echo    1. Download the wheel file from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
    echo    2. Install with: pip install downloaded_file.whl
) else (
    echo ✅ TA-Lib installed successfully!
)

REM Install optional dependencies
echo 📊 Installing optional dependencies...
pip install scikit-learn>=1.3.0
pip install websocket-client>=1.6.0
pip install requests>=2.31.0

echo 🎉 Installation complete!
echo 🚀 You can now use the advanced chart analysis features in AUJ Platform!
pause
