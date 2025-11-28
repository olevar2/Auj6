import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auj_platform.src.core.unified_config import UnifiedConfigManager
from auj_platform.src.core.alerting.alert_manager import AlertManager, AlertSeverity

async def test_alerting():
    print("Testing AlertManager...")
    
    # Mock config
    config = UnifiedConfigManager()
    
    # Initialize AlertManager
    alert_manager = AlertManager(config)
    
    # Test INFO alert
    print("Sending INFO alert...")
    await alert_manager.send_alert(
        title="Test Info",
        message="This is a test info alert",
        severity=AlertSeverity.INFO,
        component="test_script"
    )
    
    # Test WARNING alert
    print("Sending WARNING alert...")
    await alert_manager.send_alert(
        title="Test Warning",
        message="This is a test warning alert",
        severity=AlertSeverity.WARNING,
        component="test_script"
    )
    
    # Test CRITICAL alert
    print("Sending CRITICAL alert...")
    await alert_manager.send_alert(
        title="Test Critical",
        message="This is a test critical alert",
        severity=AlertSeverity.CRITICAL,
        component="test_script",
        details={"error_code": 500, "stack_trace": "fake_stack_trace"}
    )
    
    print("Alerts sent. Checking log file...")
    
    # Verify log file exists and contains alerts
    log_file = "logs/critical_alerts.log"
    if os.path.exists(log_file):
        print(f"Log file {log_file} exists.")
        with open(log_file, 'r') as f:
            content = f.read()
            print("Log content preview:")
            print(content[-500:])
            
            if "Test Critical" in content:
                print("✅ Critical alert found in log.")
            else:
                print("❌ Critical alert NOT found in log.")
    else:
        print(f"❌ Log file {log_file} does not exist.")

if __name__ == "__main__":
    asyncio.run(test_alerting())
