#!/usr/bin/env python3
"""
AUJ Platform Dashboard Launcher

This script starts the Streamlit dashboard for AUJ Platform trading system.
It ensures all dependencies are installed and provides helpful startup information.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_and_install_dependencies():
    """Check if required packages are installed, install if missing"""
    print("🔍 Checking dashboard dependencies...")
    
    try:
        # Check if streamlit is available
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("📦 Installing dashboard dependencies...")
        requirements_path = Path(__file__).parent / "requirements.txt"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        print("✅ Dependencies installed successfully")

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("\n🚀 Starting AUJ Platform Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🔧 Make sure the API server is running at: http://127.0.0.1:8000")
    print("\n" + "="*60)
    
    # Get the app.py path
    app_path = Path(__file__).parent / "app.py"
    
    # Start streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    print("🌟 AUJ Platform Dashboard Launcher 🌟")
    print("Advanced AI Trading System Dashboard")
    print("="*50)
    
    # Check dependencies
    check_and_install_dependencies()
    
    # Start the dashboard
    start_dashboard()
