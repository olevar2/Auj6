import sys
import asyncio
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "config"))

from registry.agent_indicator_mapping import AGENT_MAPPINGS, get_all_mapped_indicators
from indicator_data_requirements import INDICATOR_DATA_REQUIREMENTS

def verify_deep():
    print("Running Deep System Verification...")
    
    # 1. Count Unique Indicators
    all_mapped = get_all_mapped_indicators()
    unique_count = len(all_mapped)
    print(f"Total Unique Indicators Mapped: {unique_count}")
    
    # 2. Verify File Existence
    indicators_dir = project_root / "src" / "indicator_engine" / "indicators"
    missing_files = []
    
    # We need to search recursively
    all_files = {}
    for root, _, files in os.walk(indicators_dir):
        for file in files:
            if file.endswith(".py"):
                all_files[file[:-3]] = Path(root) / file
                
    for indicator in all_mapped:
        if indicator not in all_files:
            missing_files.append(indicator)
            
    if missing_files:
        print(f"[FAIL] Missing indicator files: {len(missing_files)}")
        for missing in missing_files:
            print(f"  - {missing}")
    else:
        print("[OK] All mapped indicators have corresponding files.")
        
    # 3. Verify Data Requirements
    missing_reqs = []
    for indicator in all_mapped:
        if indicator not in INDICATOR_DATA_REQUIREMENTS:
            # Some indicators might not need external data or are composite
            # But generally they should be listed if they use the executor
            missing_reqs.append(indicator)
            
    if missing_reqs:
        print(f"[WARN] Indicators missing data requirements: {len(missing_reqs)}")
        # This is not necessarily a fail, but good to know
        # for missing in missing_reqs:
        #     print(f"  - {missing}")
    else:
        print("[OK] All mapped indicators have data requirements.")

    # 4. Verify Agent Load
    print("\nVerifying Agent Configurations:")
    for agent_name, data in AGENT_MAPPINGS.items():
        count = len(data["assigned_indicators"])
        print(f"  - {agent_name}: {count} indicators")
        
    print("\nVerification Complete.")
    return unique_count

import os

if __name__ == "__main__":
    verify_deep()
