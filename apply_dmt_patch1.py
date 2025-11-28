#!/usr/bin/env python3
"""Script to apply patches to deal_monitoring_teams.py"""
import sys

def patch_file():
    file_path = "e:/AUG6/auj_platform/src/trading_engine/deal_monitoring_teams.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Patch 1: Add hierarchy_manager parameter
    content = content.replace(
        '                 risk_manager=None,\r\n'  
        '                 alert_callback: Optional[Callable] = None):',
        '                 risk_manager=None,\r\n'
        '                 hierarchy_manager=None,\r\n'
        '                 alert_callback: Optional[Callable] = None):'
    )
    
    # Patch 2: Update docstring
    content = content.replace(
        '            risk_manager: Dynamic risk manager instance\r\n'
        '            alert_callback: Optional callback for alert notifications',
        '            risk_manager: Dynamic risk manager instance\r\n'
        '            hierarchy_manager: Hierarchy manager for learning feedback loop\r\n'
        '            alert_callback: Optional callback for alert notifications'
    )
    
    # Patch 3: Store hierarchy_manager  
    content = content.replace(
        '        self.performance_tracker = performance_tracker\r\n'
        '        self.risk_manager = risk_manager\r\n'
        '        self.alert_callback = alert_callback\r\n',
        '        self.performance_tracker = performance_tracker\r\n'
        '        self.risk_manager = risk_manager\r\n'
        '        self.hierarchy_manager = hierarchy_manager\r\n'
        '        self.alert_callback = alert_callback\r\n'
    )
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("SUCCESS: Patches applied!")

if __name__ == "__main__":
    try:
        patch_file()
        print("Modified: Added hierarchy_manager parameter and storage")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
