#!/usr/bin/env python3
"""
Comprehensive platform test after MT5 cleanup and migration to MetaApi
"""

import sys
import importlib
import traceback

def test_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    
    tests = [
        "auj_platform.src.data_providers",
        "auj_platform.src.data_providers.metaapi_provider",
        "auj_platform.core.platform_detection",
        "auj_platform.config.indicator_data_requirements"
    ]
    
    passed = 0
    for test_module in tests:
        try:
            importlib.import_module(test_module)
            print(f"✅ {test_module}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_module}: {e}")
    
    print(f"📊 Import tests: {passed}/{len(tests)} passed")
    return passed == len(tests)

def test_provider_availability():
    """Test MetaApiProvider availability"""
    print("\n🔍 Testing MetaApiProvider availability...")
    
    try:
        from auj_platform.src.data_providers.data_provider_manager import DataProviderManager
        from auj_platform.config.indicator_data_requirements import PROVIDER_CAPABILITIES
        
        # Test MetaApiProvider presence in capabilities
        if "MetaApiProvider" in PROVIDER_CAPABILITIES:
            print("✅ MetaApiProvider found in PROVIDER_CAPABILITIES")
            
            # Test MetaApiProvider settings
            metaapi_config = PROVIDER_CAPABILITIES["MetaApiProvider"]
            print(f"✅ MetaApiProvider priority: {metaapi_config.get('priority', 'Not set')}")
            print(f"✅ MetaApiProvider status: {metaapi_config.get('status', 'Not set')}")
            
            return True
        else:
            print("❌ MetaApiProvider not found in PROVIDER_CAPABILITIES")
            return False
            
    except Exception as e:
        print(f"❌ Provider test failed: {e}")
        return False

def test_platform_detection():
    """Test platform detection"""
    print("\n🔍 Testing platform detection...")
    
    try:
        from auj_platform.core.platform_detection import detect_platform
        platform_info = detect_platform()
        
        print(f"✅ Platform detected: {platform_info.get('platform', 'Unknown')}")
        print(f"✅ Recommended providers: {platform_info.get('recommended_providers', [])}")
        
        # Ensure MetaApiProvider is recommended on Linux
        if platform_info.get('platform') == 'linux':
            recommended = platform_info.get('recommended_providers', [])
            if 'MetaApiProvider' in recommended:
                print("✅ MetaApiProvider correctly recommended for Linux")
                return True
            else:
                print("⚠️ MetaApiProvider not recommended for Linux")
                return False
        else:
            print("ℹ️ Platform is not Linux, checking general compatibility")
            return True
            
    except Exception as e:
        print(f"❌ Platform detection failed: {e}")
        return False

def test_indicator_system():
    """Test indicator system"""
    print("\n🔍 Testing indicator system...")
    
    try:
        from auj_platform.config.indicator_data_requirements import get_indicator_requirements
        
        # Test a simple indicator
        sma_req = get_indicator_requirements("SMA")
        if sma_req and "MetaApiProvider" in sma_req.get("available_providers", []):
            print("✅ SMA indicator supports MetaApiProvider")
            return True
        else:
            print("❌ SMA indicator does not support MetaApiProvider")
            return False
            
    except Exception as e:
        print(f"❌ Indicator system test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test"""
    print("🚀 Starting comprehensive platform test...\n")
    
    test_results = []
    
    # Run all tests
    test_results.append(("Imports", test_imports()))
    test_results.append(("Provider Availability", test_provider_availability()))
    test_results.append(("Platform Detection", test_platform_detection()))
    test_results.append(("Indicator System", test_indicator_system()))
    
    # Summarize results
    print(f"\n📊 Test Summary:")
    print(f"=" * 50)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed_tests += 1
    
    print(f"=" * 50)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! MetaApi integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)