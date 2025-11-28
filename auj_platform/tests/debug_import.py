import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Attempting to import EconomicCalendarAgent...")
try:
    from src.agents.economic_calendar_agent import EconomicCalendarEnhancedAgent
    print("SUCCESS: Imported EconomicCalendarEnhancedAgent")
except ImportError as e:
    print(f"FAILURE: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"FAILURE (Exception): {e}")
    traceback.print_exc()
