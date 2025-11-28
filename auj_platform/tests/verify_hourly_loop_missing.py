
import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.containers import PlatformContainer, ApplicationContainer
from src.learning.robust_hourly_feedback_loop import RobustHourlyFeedbackLoop

async def verify_hourly_loop_wiring():
    print("=" * 80)
    print("[VERIFY] VERIFYING HOURLY FEEDBACK LOOP WIRING")
    print("=" * 80)

    try:
        # Create containers
        platform_container = PlatformContainer()
        app_container = ApplicationContainer()
        app_container.platform.override(platform_container)
        
        # Wire
        platform_container.wire(modules=[__name__])
        
        # Get platform instance
        platform = app_container.auj_platform()
        
        print(f"[OK] Platform instance created: {type(platform).__name__}")
        
        # Check for hourly loop attribute
        if hasattr(platform, 'hourly_loop'):
            print(f"[OK] 'hourly_loop' attribute found: {platform.hourly_loop}")
            if isinstance(platform.hourly_loop, RobustHourlyFeedbackLoop):
                print("[OK] 'hourly_loop' is instance of RobustHourlyFeedbackLoop")
            else:
                print(f"[FAIL] 'hourly_loop' is NOT RobustHourlyFeedbackLoop: {type(platform.hourly_loop)}")
        else:
            print("[FAIL] 'hourly_loop' attribute NOT found in AUJPlatformDI")
            
        # Check if it's in the container providers
        if hasattr(platform_container, 'robust_hourly_feedback_loop'):
             print("[OK] 'robust_hourly_feedback_loop' provider found in PlatformContainer")
        else:
             print("[FAIL] 'robust_hourly_feedback_loop' provider NOT found in PlatformContainer")

    except Exception as e:
        print(f"[ERROR] Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_hourly_loop_wiring())
