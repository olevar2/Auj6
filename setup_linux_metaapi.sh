#!/bin/bash
# AUJ Platform Linux Setup Script
# Quick setup for MetaApi integration

set -e

echo "🚀 Setting up AUJ Platform for Linux with MetaApi..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Linux requirements
echo "📥 Installing MetaApi dependencies..."
pip install -r requirements-linux.txt

# Create basic directories
echo "📁 Creating directories..."
mkdir -p data logs config backups

# Copy environment template
echo "⚙️ Setting up environment..."
if [ ! -f .env ]; then
    cp config/.env.template .env
    echo "✅ Created .env file from template"
    echo "⚠️  Please edit .env and add your MetaApi credentials:"
    echo "   - AUJ_METAAPI_TOKEN=your_token_here"
    echo "   - AUJ_METAAPI_ACCOUNT_ID=your_account_id_here"
fi

# Set permissions
chmod +x scripts/*.sh 2>/dev/null || true

echo ""
echo "✅ AUJ Platform setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your MetaApi credentials"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python -m auj_platform.dashboard.app"
echo ""
echo "🔗 Get MetaApi credentials at: https://app.metaapi.cloud/"