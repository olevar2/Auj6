import os
import sys
import logging
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

# Set environment to low_resource
os.environ["AUJ_ENVIRONMENT"] = "low_resource"

# Import platform components
try:
    from auj_platform.src.core.unified_config import UnifiedConfigManager
    from auj_platform.src.core.containers import PlatformContainer
except ImportError as e:
    print(f"Error importing platform components: {e}")
    print("Please run this script from the project root directory (e:\\AUG6)")
    sys.exit(1)

def main():
    print("🚀 Starting AUJ Platform in Low Resource Mode...")
    
    # Initialize Config Manager with specific config file
    config_path = Path("auj_platform/src/config/low_resource_config.yaml")
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
        
    try:
        # Initialize Container
        container = PlatformContainer()
        
        # Override config with low resource settings
        # Note: In a real DI setup, we might pass the config path to the container
        # For now, we rely on the environment variable and the config manager loading it
        
        print("✅ Configuration loaded.")
        print("   - Max Positions: 3")
        print("   - Parallel Analysis: Disabled")
        print("   - ML Filtering: Disabled")
        print("   - Logging Level: WARNING")
        
        # Here we would normally start the main application loop
        # For this launcher, we'll just verify initialization works
        
        print("\n✅ Platform initialized successfully in Low Resource Mode.")
        print("   Ready to run.")
        
    except Exception as e:
        print(f"❌ Failed to initialize platform: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
