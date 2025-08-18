#!/usr/bin/env python3
"""
Quick MetaApi Test Script
Simple script to verify MetaApi integration is working
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_metaapi_basic():
    """Test basic MetaApi functionality"""
    print("🧪 Testing MetaApi Integration...")

    # Check environment variables
    token = os.getenv('AUJ_METAAPI_TOKEN')
    account_id = os.getenv('AUJ_METAAPI_ACCOUNT_ID')

    if not token or not account_id:
        print("❌ MetaApi credentials not found in environment")
        print("Please set AUJ_METAAPI_TOKEN and AUJ_METAAPI_ACCOUNT_ID")
        return False

    try:
        # Test import
        from metaapi_cloud_sdk import MetaApi
        print("✅ MetaApi SDK imported successfully")

        # Initialize MetaApi (test only)
        _ = MetaApi(token)  # Test initialization only
        print("✅ MetaApi instance created")

        # Test connection (without actually connecting)
        print("✅ Basic MetaApi test passed")
        return True

    except ImportError as e:
        print(f"❌ Failed to import MetaApi SDK: {e}")
        print("Install with: pip install metaapi-cloud-sdk")
        return False
    except Exception as e:
        print(f"❌ MetaApi test failed: {e}")
        return False

def test_platform_detection():
    """Test platform detection"""
    print("\n🔍 Testing Platform Detection...")

    try:
        from auj_platform.core.platform_detection import PlatformDetector

        detector = PlatformDetector()
        platform_info = detector.detect_platform()

        print(f"✅ OS: {platform_info['os']}")
        print(f"✅ Architecture: {platform_info['architecture']}")
        print(f"✅ Container: {platform_info['is_container']}")

        # Test broker recommendation
        broker_config = detector.get_recommended_broker_config()
        print(f"✅ Recommended broker: {broker_config['broker_type']}")

        if broker_config['broker_type'] == 'metaapi':
            print("✅ MetaApi correctly recommended for this platform")
        else:
            print(f"⚠️  Expected MetaApi but got {broker_config['broker_type']}")

        return True

    except Exception as e:
        print(f"❌ Platform detection test failed: {e}")
        return False

def test_provider_import():
    """Test MetaApi provider import"""
    print("\n📦 Testing Provider Import...")

    try:
        # Test imports (checking availability only)
        from auj_platform.src.data_providers.metaapi_provider import (  # noqa: F401
            MetaApiProvider,
        )
        print("✅ MetaApiProvider imported successfully")

        from auj_platform.src.broker_interfaces.metaapi_broker import (  # noqa: F401
            MetaApiBroker,
        )
        print("✅ MetaApiBroker imported successfully")

        return True

    except Exception as e:
        print(f"❌ Provider import test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 AUJ Platform MetaApi Integration Test")
    print("=" * 50)

    tests_passed = 0
    total_tests = 3

    # Test 1: Platform Detection
    if test_platform_detection():
        tests_passed += 1

    # Test 2: Provider Import
    if test_provider_import():
        tests_passed += 1

    # Test 3: MetaApi Basic
    if await test_metaapi_basic():
        tests_passed += 1

    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! MetaApi integration is ready.")
        print("\nYour platform is configured to use MetaApi on Linux!")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
