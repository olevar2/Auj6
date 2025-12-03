"""
Test Script for unified_database_manager_NEW.py
Tests syntax, imports, and basic functionality
"""

import sys
sys.path.insert(0, 'e:/AUG6')

print("=" * 60)
print("Testing unified_database_manager_NEW.py")
print("=" * 60)

# Test 1: Import test
print("\n[Test 1] Testing imports...")
try:
    from auj_platform.src.core.unified_database_manager_NEW import (
        UnifiedDatabaseManager,
        BoundedMetricsCollector,
        ConnectionPool,
        QueryCache,
        DatabaseHealth,
        QueryMetrics
    )
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Class instantiation
print("\n[Test 2] Testing class instantiation...")
try:
    db = UnifiedDatabaseManager()
    print(f"✅ UnifiedDatabaseManager created: {db.database_url}")
    
    collector = BoundedMetricsCollector(max_size=100)
    print(f"✅ BoundedMetricsCollector created: max_size={collector.max_size}")
    
    pool = ConnectionPool("sqlite:///test.db", True, 10)
    print(f"✅ ConnectionPool created: max_connections={pool.max_connections}")
    
    cache = QueryCache(max_size=100, ttl_seconds=60)
    print(f"✅ QueryCache created: max_size={cache.max_size}")
    
except Exception as e:
    print(f"❌ Instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check async locks
print("\n[Test 3] Checking async locks...")
try:
    import asyncio
    
    # Check that locks are asyncio.Lock instances
    assert isinstance(collector._lock, asyncio.Lock), "BoundedMetricsCollector should use asyncio.Lock"
    print("✅ BoundedMetricsCollector uses asyncio.Lock")
    
    assert isinstance(pool._lock, asyncio.Lock), "ConnectionPool should use asyncio.Lock"
    print("✅ ConnectionPool uses asyncio.Lock")
    
    assert isinstance(cache._lock, asyncio.Lock), "QueryCache should use asyncio.Lock"
    print("✅ QueryCache uses asyncio.Lock")
    
    assert isinstance(db._init_lock, asyncio.Lock), "UnifiedDatabaseManager._init_lock should use asyncio.Lock"
    print("✅ UnifiedDatabaseManager._init_lock uses asyncio.Lock")
    
    assert isinstance(db._metrics_lock, asyncio.Lock), "UnifiedDatabaseManager._metrics_lock should use asyncio.Lock"
    print("✅ UnifiedDatabaseManager._metrics_lock uses asyncio.Lock")
    
except AssertionError as e:
    print(f"❌ Lock check failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe new file is ready for integration testing.")
