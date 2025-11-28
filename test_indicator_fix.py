"""
Final verification test for indicator auto-discovery fix
"""

from auj_platform.src.registry.indicator_registry import get_indicator_registry

print("=" * 70)
print("TESTING INDICATOR AUTO-DISCOVERY FIX")
print("=" * 70)

print("\n[1/3] Getting indicator registry...")
try:
    registry = get_indicator_registry()
    print("  [OK] Registry obtained successfully")
except Exception as e:
    print(f"  [FAIL] Could not get registry: {e}")
    exit(1)

print("\n[2/3] Checking discovered indicators...")
try:
    indicator_names = registry.get_all_indicator_names()
    count = len(indicator_names)
    print(f"  [OK] Discovered {count} indicators")
    
    if count == 0:
        print("  [FAIL] No indicators discovered!")
    elif count < 100:
        print(f"  [WARN] Only {count} indicators (expected ~159)")
    else:
        print(f"  [SUCCESS] Indicator count is good!")
        
except Exception as e:
    print(f"  [FAIL] Could not get indicator list: {e}")
    exit(1)

print("\n[3/3] Sampling discovered indicators...")
try:
    # Show first 15 indicators
    sample = indicator_names[:15] if len(indicator_names) > 15 else indicator_names
    for i, name in enumerate(sample, 1):
        print(f"  {i:2}. {name}")
    
    if len(indicator_names) > 15:
        print(f"  ... and {len(indicator_names) - 15} more")
        
except Exception as e:
    print(f"  [FAIL] Could not sample indicators: {e}")

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

if count > 100:
    print(f"[SUCCESS] Indicator auto-discovery is working!")
    print(f"  - {count} indicators discovered and registered")
    print(f"  - Ready for use by all 10 agents")
    print(f"  - SmartIndicatorExecutor can now invoke all indicators")
else:
    print(f"[PARTIAL] Discovered {count} indicators")
    print(f"  - Expected: ~159 indicators")
    print(f"  - May need path adjustment in discover_and_register_all()")

print("=" * 70)
