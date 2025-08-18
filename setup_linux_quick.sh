#!/bin/bash
# AUJ Platform Linux Quick Setup Script
# Simple installation for MetaApi-based trading platform

set -e  # Exit on any error

echo "🚀 AUJ Platform Linux Quick Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux only!"
    exit 1
fi

print_status "Detected Linux system: $(uname -a)"

# Check Python version
print_header "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python version: $PYTHON_VERSION"

    # Check if version is >= 3.9
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
        print_status "Python version is compatible"
    else
        print_error "Python 3.9+ required. Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python3 not found! Please install Python 3.9+"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    print_status "pip3 is available"
else
    print_error "pip3 not found! Please install pip3"
    exit 1
fi

# Create virtual environment
print_header "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
print_header "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_header "Installing AUJ Platform requirements..."
if [ -f "requirements-linux.txt" ]; then
    pip install -r requirements-linux.txt
    print_status "Linux requirements installed"
else
    print_warning "requirements-linux.txt not found, installing from pyproject.toml"
    pip install -e ".[linux]"
fi

# Create necessary directories
print_header "Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p backups
mkdir -p config
print_status "Directories created"

# Check for .env file
print_header "Checking configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_warning "Created .env from .env.example"
        print_warning "Please edit .env file with your MetaApi credentials!"
    else
        print_error ".env.example not found!"
        exit 1
    fi
else
    print_status ".env file already exists"
fi

# Check MetaApi credentials
print_header "Checking MetaApi configuration..."
if grep -q "your_metaapi_token_here" .env 2>/dev/null; then
    print_warning "MetaApi token not configured in .env file"
    print_warning "Please set AUJ_METAAPI_TOKEN and AUJ_METAAPI_ACCOUNT_ID"
else
    print_status "MetaApi configuration appears to be set"
fi

# Test installation
print_header "Testing installation..."
export PYTHONPATH=$PWD:$PYTHONPATH

# Basic import test
python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from auj_platform.core.platform_detection import PlatformDetector
    print('✅ Platform detection module imported successfully')

    detector = PlatformDetector()
    platform_info = detector.detect_platform()
    print(f'✅ Platform detected: {platform_info[\"os\"]}')

    if platform_info['os'] == 'linux':
        print('✅ Linux platform confirmed')
    else:
        print(f'⚠️  Expected Linux but got: {platform_info[\"os\"]}')

except ImportError as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Test failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_status "Installation test passed!"
else
    print_error "Installation test failed!"
    exit 1
fi

# Final instructions
echo ""
echo "🎉 AUJ Platform setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your MetaApi credentials:"
echo "   - AUJ_METAAPI_TOKEN=your_token_here"
echo "   - AUJ_METAAPI_ACCOUNT_ID=your_account_id_here"
echo ""
echo "2. Test the platform:"
echo "   source venv/bin/activate"
echo "   python3 run_linux.py"
echo ""
echo "3. Run the dashboard:"
echo "   python3 -m auj_platform.dashboard.app"
echo ""
echo "For more help, check: config/README.md"
