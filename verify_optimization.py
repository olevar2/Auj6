import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

# Set environment to low_resource
os.environ["AUJ_ENVIRONMENT"] = "low_resource"

try:
    from auj_platform.src.core.unified_config import UnifiedConfigManager
except ImportError as e:
    print(f"Error importing UnifiedConfigManager: {e}")
    sys.exit(1)

def verify():
    print("Verifying Low Resource Configuration...")
    
    # Initialize Config Manager
    config = UnifiedConfigManager()
    
    # Check critical values
    max_positions = config.get_int("risk_parameters.max_positions")
    logging_level = config.get_str("logging.level")
    parallel_analysis = config.get_bool("coordination.enable_parallel_analysis")
    
    print(f"   - Max Positions: {max_positions} (Expected: 3)")
    print(f"   - Logging Level: {logging_level} (Expected: WARNING)")
    print(f"   - Parallel Analysis: {parallel_analysis} (Expected: False)")
    
    if max_positions == 3 and logging_level == "WARNING" and not parallel_analysis:
        print("\nSUCCESS: Low resource configuration loaded correctly.")
    else:
        print("\nFAILURE: Configuration mismatch.")

if __name__ == "__main__":
    verify()
