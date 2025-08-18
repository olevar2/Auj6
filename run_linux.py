#!/usr/bin/env python3
"""
AUJ Platform Linux Launcher
Simple launcher for MetaApi-based trading platform on Linux
"""

import logging
import os
import platform
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/auj_platform.log', mode='a')
        ]
    )

def check_environment():
    """Check if environment is properly configured for MetaApi"""
    logger = logging.getLogger(__name__)

    # Check platform
    current_platform = platform.system().lower()
    logger.info(f"Detected platform: {current_platform}")

    # Check MetaApi credentials
    metaapi_token = os.getenv('AUJ_METAAPI_TOKEN')
    metaapi_account = os.getenv('AUJ_METAAPI_ACCOUNT_ID')

    if not metaapi_token:
        logger.error("AUJ_METAAPI_TOKEN environment variable not set!")
        logger.info("Please set your MetaApi token: export AUJ_METAAPI_TOKEN=your_token_here")
        return False

    if not metaapi_account:
        logger.error("AUJ_METAAPI_ACCOUNT_ID environment variable not set!")
        logger.info("Please set your MetaApi account ID: export AUJ_METAAPI_ACCOUNT_ID=your_account_id_here")
        return False

    logger.info("✅ MetaApi credentials found")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    logger = logging.getLogger(__name__)

    required_packages = [
        'metaapi_cloud_sdk',
        'websockets',
        'aiohttp',
        'pandas',
        'numpy',
        'yaml',
        'fastapi',
        'uvicorn'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} is missing")

    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install missing packages with: pip install -e .[linux]")
        return False

    return True

def initialize_platform():
    """Initialize AUJ Platform with MetaApi"""
    logger = logging.getLogger(__name__)

    try:
        # Import platform detection
        from auj_platform.core.platform_detection import PlatformDetector

        detector = PlatformDetector()
        platform_info = detector.detect_platform()

        logger.info(f"Platform detection: {platform_info}")

        # Check recommended broker
        recommended_broker = detector.get_recommended_broker_config()
        logger.info(f"Recommended broker: {recommended_broker['broker_type']}")

        if recommended_broker['broker_type'] != 'metaapi':
            logger.warning(f"Expected MetaApi but got {recommended_broker['broker_type']}")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize platform: {e}")
        return False

def test_metaapi_connection():
    """Test MetaApi connection"""
    logger = logging.getLogger(__name__)

    try:
        from auj_platform.src.data_providers.metaapi_provider import MetaApiProvider

        logger.info("Testing MetaApi connection...")

        # Initialize provider (basic test)
        try:
            provider = MetaApiProvider()
            logger.info("✅ MetaApi provider initialized successfully")
            del provider  # Clean up
        except Exception as init_error:
            logger.error(f"Provider initialization failed: {init_error}")
            raise

        return True

    except Exception as e:
        logger.error(f"❌ MetaApi connection test failed: {e}")
        return False

def main():
    """Main launcher function"""
    print("🚀 AUJ Platform Linux Launcher")
    print("=" * 50)

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting AUJ Platform on Linux...")

    # Check environment
    if not check_environment():
        logger.error("❌ Environment check failed")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependencies check failed")
        sys.exit(1)

    # Initialize platform
    if not initialize_platform():
        logger.error("❌ Platform initialization failed")
        sys.exit(1)

    # Test MetaApi connection
    if not test_metaapi_connection():
        logger.error("❌ MetaApi connection test failed")
        sys.exit(1)

    logger.info("✅ All checks passed! Platform is ready.")
    print("\n🎉 AUJ Platform is ready to use MetaApi on Linux!")
    print("\nNext steps:")
    print("1. Run the dashboard: python -m auj_platform.dashboard.app")
    print("2. Start the API: python -m auj_platform.src.api.main")
    print("3. Run trading agents: python -m auj_platform.src.agents.main")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
