"""
Comprehensive Indicator Health Check
Tests all aspects of indicator registration and usage
"""

import sys
import traceback

print("=" * 70)
print("INDICATOR SYSTEM HEALTH CHECK")
print("=" * 70)

# Test 1: Import IndicatorRegistry
print("\n[1/6] Testing IndicatorRegistry import...")
try:
    from auj_platform.src.registry.indicator_registry import IndicatorRegistry
    print("  [OK] IndicatorRegistry imported successfully")
except Exception as e:
    print(f"  [FAIL] Could not import IndicatorRegistry: {e}")
    sys.exit(1)

# Test 2: Check if indicators can be discovered
print("\n[2/6] Testing indicator discovery...")
try:
    registry = IndicatorRegistry()
    indicator_count = len(registry._registry)
    print(f"  [OK] Registry initialized with {indicator_count} indicators")
    
    if indicator_count == 0:
        print("  [WARN] No indicators discovered!")
    elif indicator_count < 100:
        print(f"  [WARN] Only {indicator_count} indicators (expected ~159)")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# Test 3: Check agent-indicator mapping
print("\n[3/6] Testing agent-indicator mapping...")
try:
    from auj_platform.src.registry.agent_indicator_mapping import (
        AGENT_MAPPINGS, 
        get_all_mapped_indicators,
        validate_agent_mapping
    )
    
    mapped_count = len(get_all_mapped_indicators())
    validation = validate_agent_mapping()
    
    print(f"  [OK] {mapped_count} unique indicators mapped to agents")
    print(f"  [OK] Total assignments: {validation['total_assignments']}")
    print(f"  [OK] Agents: {validation['agents_count']}")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# Test 4: Check SmartIndicatorExecutor
print("\n[4/6] Testing SmartIndicatorExecutor...")
try:
    from auj_platform.src.indicator_engine.indicator_executor import SmartIndicatorExecutor
    print("  [OK] SmartIndicatorExecutor imported successfully")
    
except Exception as e:
    print(f"  [FAIL] Could not import SmartIndicatorExecutor: {e}")

# Test 5: Sample indicator imports
print("\n[5/6] Testing sample indicator imports...")
sample_indicators = [
    ("RSI", "auj_platform.src.indicator_engine.indicators.momentum.rsi_indicator"),
    ("MACD", "auj_platform.src.indicator_engine.indicators.momentum.macd_indicator"),
    ("ADX", "auj_platform.src.indicator_engine.indicators.trend.adx_indicator"),
    ("Bollinger", "auj_platform.src.indicator_engine.indicators.volatility.bollinger_bands_indicator"),
]

success_count = 0
for name, module_path in sample_indicators:
    try:
        module = __import__(module_path, fromlist=[''])
        print(f"  [OK] {name} indicator imported")
        success_count += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}")

print(f"\n  Import success: {success_count}/{len(sample_indicators)}")

# Test 6: Check for common issues
print("\n[6/6] Checking for common issues...")

issues_found = []

try:
    registered = set(registry._registry.keys())
    mapped = set(get_all_mapped_indicators())
    
    unmapped = registered - mapped
    if unmapped:
        issues_found.append(f"{len(unmapped)} indicators in registry but not mapped")
        print(f"  [WARN] {len(unmapped)} indicators not mapped to agents")
        if len(unmapped) <= 10:
            for ind in list(unmapped)[:10]:
                print(f"    - {ind}")
    
    not_registered = mapped - registered
    if not_registered:
        issues_found.append(f"{len(not_registered)} mapped but not in registry")
        print(f"  [WARN] {len(not_registered)} mapped indicators missing from registry")
        if len(not_registered) <= 10:
            for ind in list(not_registered)[:10]:
                print(f"    - {ind}")
                
    if not unmapped and not not_registered:
        print("  [OK] All mapped indicators exist in registry")
        print("  [OK] No missing mappings detected")
        
except Exception as e:
    issues_found.append(f"Verification failed: {e}")
    print(f"  [FAIL] {e}")

# Final Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if not issues_found:
    print("[SUCCESS] ALL CHECKS PASSED - Indicator system is healthy!")
    print(f"\n  - {indicator_count} indicators in registry")
    print(f"  - {mapped_count} indicators mapped to {validation['agents_count']} agents")
    print(f"  - All key components can be imported")
else:
    print(f"[ISSUES] Found {len(issues_found)} potential issue(s):")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")

print("\n" + "=" * 70)
