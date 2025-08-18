#!/bin/bash

# AUJ Platform Environment Setup Script for Unix/Linux/macOS
# ==========================================================
# This script sets up the AUJ Platform environment and runs basic validation.
# 
# Author: AUJ Platform Development Team
# Date: 2025-07-04
# Version: 1.0.0

set -e  # Exit on any error

echo "========================================"
echo "  AUJ Platform Environment Setup"
echo "========================================"
echo

# Determine platform root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_ROOT="$SCRIPT_DIR"

echo "Platform Root: $PLATFORM_ROOT"
echo

# Set environment variables
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export AUJ_PLATFORM_ROOT="$PLATFORM_ROOT"
export AUJ_SRC_PATH="$PLATFORM_ROOT/auj_platform/src"
export AUJ_CONFIG_PATH="$PLATFORM_ROOT/config"
export AUJ_LOGS_PATH="$PLATFORM_ROOT/logs"

echo "Environment variables set:"
echo "  AUJ_PLATFORM_ROOT = $AUJ_PLATFORM_ROOT"
echo "  AUJ_SRC_PATH = $AUJ_SRC_PATH"
echo "  AUJ_CONFIG_PATH = $AUJ_CONFIG_PATH"
echo "  AUJ_LOGS_PATH = $AUJ_LOGS_PATH"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not available in PATH"
    echo "Please install Python 3.8+ and ensure it's in your PATH"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "Python version:"
$PYTHON_CMD --version
echo

# Check Python version (must be 3.8+)
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "ERROR: Python 3.8+ is required, but found Python $PYTHON_VERSION"
    exit 1
fi

# Check if required directories exist and create them if needed
echo "Checking directory structure..."

for dir in "$AUJ_SRC_PATH" "$AUJ_CONFIG_PATH" "$AUJ_LOGS_PATH" "$AUJ_SRC_PATH/core"; do
    if [ ! -d "$dir" ]; then
        echo "Creating missing directory: $dir"
        mkdir -p "$dir"
    fi
done

echo "Directory structure verified."
echo

# Run Python environment setup if available
echo "Running Python environment setup..."
if [ -f "$AUJ_SRC_PATH/core/environment_setup.py" ]; then
    cd "$PLATFORM_ROOT"
    if $PYTHON_CMD "$AUJ_SRC_PATH/core/environment_setup.py" --verbose; then
        echo "Python environment setup completed successfully"
    else
        echo "WARNING: Python environment setup reported issues"
        echo "Please check the log files for details"
    fi
else
    echo "WARNING: environment_setup.py not found"
    echo "Skipping Python environment setup"
fi
echo

# Run path cleanup analysis if available
echo "Running path cleanup analysis..."
if [ -f "$AUJ_SRC_PATH/core/path_cleanup.py" ]; then
    cd "$PLATFORM_ROOT"
    if $PYTHON_CMD "$AUJ_SRC_PATH/core/path_cleanup.py" --platform-root "$PLATFORM_ROOT" --dry-run --verbose; then
        echo "Path cleanup analysis completed successfully"
    else
        echo "WARNING: Path cleanup analysis reported issues"
    fi
else
    echo "WARNING: path_cleanup.py not found"
    echo "Skipping path cleanup analysis"
fi
echo

# Basic validation
echo "Running basic validation..."

# Check for core files
MISSING_FILES=""
for file in "$AUJ_SRC_PATH/core/__init__.py" "$AUJ_SRC_PATH/core/config.py"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES="$MISSING_FILES $(basename $file)"
    fi
done

if [ -n "$MISSING_FILES" ]; then
    echo "WARNING: Missing core files:$MISSING_FILES"
    echo "Platform may not function correctly"
else
    echo "Core files validation passed"
fi

# Test basic Python import
echo "Testing basic Python imports..."
cd "$PLATFORM_ROOT"
if $PYTHON_CMD -c "import sys; sys.path.insert(0, '$AUJ_SRC_PATH'); print('Python path setup: OK')" 2>/dev/null; then
    echo "Python import test passed"
else
    echo "WARNING: Python import test failed"
    echo "There may be issues with the Python path setup"
fi

# Make scripts executable
echo "Setting script permissions..."
chmod +x "$0" 2>/dev/null || true

echo
echo "========================================"
echo "  Environment Setup Complete"
echo "========================================"
echo
echo "The AUJ Platform environment has been configured."
echo
echo "Next steps:"
echo "  1. Verify all required dependencies are installed"
echo "  2. Review configuration files in: $AUJ_CONFIG_PATH"
echo "  3. Check logs in: $AUJ_LOGS_PATH"
echo "  4. Test platform functionality"
echo
echo "To run the platform:"
echo "  cd \"$PLATFORM_ROOT\""
echo "  $PYTHON_CMD -m auj_platform.src.main"
echo

# Function to check if script is being sourced or executed
is_sourced() {
    if [ -n "$ZSH_VERSION" ]; then
        case $ZSH_EVAL_CONTEXT in *:file:*) return 0;; esac
    else
        case ${0##*/} in sh|-sh|bash|-bash) return 0;; esac
    fi
    return 1
}

# Export environment variables if sourced
if is_sourced; then
    echo "Environment variables have been exported to your shell session."
    echo "You can now run AUJ Platform commands from this terminal."
fi

exit 0