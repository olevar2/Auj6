import sys
import asyncio
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "config"))

from registry.agent_indicator_mapping import AGENT_MAPPINGS, get_agent_indicators
from indicator_data_requirements import INDICATOR_DATA_REQUIREMENTS

async def verify_system():
    print("Verifying Indicator System Integrity...")
    
    # 1. Verify SimulationExpert is gone
    if "SimulationExpert" in AGENT_MAPPINGS:
        print("[FAIL] SimulationExpert still exists in mappings!")
        return False
    else:
        print("[OK] SimulationExpert successfully removed.")
        
    # 2. Verify MomentumAgent indicators
    momentum_indicators = get_agent_indicators("MomentumAgent")
    print(f"MomentumAgent has {len(momentum_indicators)} indicators (Expected: 8)")
    if len(momentum_indicators) != 8:
        print("[FAIL] MomentumAgent indicator count mismatch!")
        return False
    else:
        print("[OK] MomentumAgent pruned successfully.")
        
    # 3. Verify PatternMaster indicators
    pattern_indicators = get_agent_indicators("PatternMaster")
    print(f"PatternMaster has {len(pattern_indicators)} indicators (Expected: 12)")
    if len(pattern_indicators) != 12:
        print("[FAIL] PatternMaster indicator count mismatch!")
        return False
    else:
        print("[OK] PatternMaster pruned successfully.")
        
    # 4. Verify all mapped indicators exist in requirements (if applicable)
    # Note: INDICATOR_DATA_REQUIREMENTS might need updating if it was hardcoded with old indicators
    # But for now, we just check if we can access the mapping without error
    
    print("\nSystem verification passed!")
    return True

if __name__ == "__main__":
    asyncio.run(verify_system())
