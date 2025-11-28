import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.registry.agent_registry import get_agent_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_discovery():
    print("Verifying Agent Discovery...")
    
    registry = get_agent_registry()
    agents = registry.get_all_agent_classes()
    
    print(f"Found {len(agents)} agents:")
    found_economic = False
    
    for name, cls in agents.items():
        print(f"  - {name}: {cls.__name__}")
        if name == 'economic_calendar_agent':
            found_economic = True
            
    if found_economic:
        print("\nSUCCESS: EconomicCalendarAgent was discovered!")
    else:
        print("\nFAILURE: EconomicCalendarAgent was NOT discovered.")
        
if __name__ == "__main__":
    verify_discovery()
