import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from registry.agent_indicator_mapping import AGENT_MAPPINGS
except ImportError as e:
    print(f"Error importing registry: {e}")
    sys.exit(1)

def get_all_assigned_indicators():
    indicators = set()
    for agent, data in AGENT_MAPPINGS.items():
        indicators.update(data["assigned_indicators"])
    return indicators

def cleanup():
    assigned = get_all_assigned_indicators()
    print(f"Found {len(assigned)} unique assigned indicators across all agents.")
    
    indicators_dir = project_root / "src" / "indicator_engine" / "indicators"
    
    deleted_count = 0
    kept_count = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(indicators_dir):
        for file in files:
            if not file.endswith(".py") or file == "__init__.py":
                continue
                
            indicator_name = file[:-3] # Remove .py
            
            # Skip base classes or specific utility files
            if indicator_name in ["standard_indicator", "indicator_base", "indicator_interface"]: 
                continue
                
            # Skip if it's in the base directory (might be abstract classes)
            if Path(root).name == "base":
                continue

            if indicator_name not in assigned:
                file_path = Path(root) / file
                print(f"Deleting orphan: {indicator_name}")
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
            else:
                kept_count += 1
                
    print(f"\nSummary:")
    print(f"Deleted: {deleted_count} files")
    print(f"Kept: {kept_count} files")
    print(f"Total Active: {len(assigned)}")

if __name__ == "__main__":
    cleanup()
