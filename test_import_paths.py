"""
Debug script to test actual imports
"""
import sys
import importlib

# Test different import approaches
test_paths = [
    "auj_platform.src.indicator_engine.indicators.momentum.rsi_indicator",
    "src.indicator_engine.indicators.momentum.rsi_indicator",
    "indicator_engine.indicators.momentum.rsi_indicator",
]

print("Testing indicator imports:")
print("=" * 70)

for path in test_paths:
    print(f"\nTrying: {path}")
    try:
        module = importlib.import_module(path)
        print(f"  [SUCCESS] Module imported!")
        print(f"  Module: {module}")
        # Try to find indicator class
        import inspect
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.endswith('Indicator'):
                print(f"  Found class: {name}")
        break
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")

print("\n" + "=" * 70)
