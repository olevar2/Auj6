#!/bin/bash
# install_chart_dependencies.sh
# Installation script for AUJ Platform Chart Analysis Dependencies

echo "🚀 Installing AUJ Platform Chart Analysis Dependencies..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install Python and pip first."
    exit 1
fi

# Install basic requirements
echo "📦 Installing basic requirements..."
pip install plotly>=5.17.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0

# Try to install TA-Lib (this might require additional system dependencies)
echo "🔧 Installing TA-Lib..."
if pip install TA-Lib==0.4.29; then
    echo "✅ TA-Lib installed successfully!"
else
    echo "⚠️ TA-Lib installation failed. The chart analysis will use fallback implementations."
    echo "💡 To install TA-Lib manually:"
    echo "   - Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib"
    echo "   - macOS: brew install ta-lib"
    echo "   - Linux: sudo apt-get install libta-lib-dev (Ubuntu/Debian)"
fi

# Install optional dependencies
echo "📊 Installing optional dependencies..."
pip install scikit-learn>=1.3.0
pip install websocket-client>=1.6.0
pip install requests>=2.31.0

echo "🎉 Installation complete!"
echo "🚀 You can now use the advanced chart analysis features in AUJ Platform!"
