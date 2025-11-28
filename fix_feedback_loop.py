#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix robust_hourly_feedback_loop.py by adding execution_handler integration.
"""

import shutil
from pathlib import Path

# File paths
original_file = Path(r'e:\AUG6\auj_platform\src\learning\robust_hourly_feedback_loop.py')
fixed_file = Path(r'e:\AUG6\auj_platform\src\learning\robust_hourly_feedback_loop_FIXED.py')
backup_file = Path(r'e:\AUG6\auj_platform\src\learning\robust_hourly_feedback_loop_OLD.py')

print(f"Reading original file: {original_file}")
with open(original_file, 'r', encoding='utf-8') as f:
    content = f.read()

print("Applying fixes...")

# Fix 1: Add execution_handler parameter to __init__
content = content.replace(
    '                 economic_monitor: Optional[EconomicMonitor] = None,\n'
    '                 database_path: Optional[str] = None,',
    
    '                 economic_monitor: Optional[EconomicMonitor] = None,\n'
    '                 execution_handler: Optional[Any] = None,\n'
    '                 database_path: Optional[str] = None,'
)

# Fix 2: Store execution_handler
content = content.replace(
    '        self.economic_monitor = economic_monitor\n'
    '        \n'
    '        # Configuration',
    
    '        self.economic_monitor = economic_monitor\n'
    '        self.execution_handler = execution_handler\n'
    '        \n'
    '        # Configuration'
)

# Fix 3: Add execute_trade_signal() call
old_phase10 = '''            if trade_signal:
                self.logger.info(f"Coordinator generated trade signal: {trade_signal.symbol} {trade_signal.direction}")
            phase_durations["COORDINATOR_EXECUTION"]'''

new_phase10 = '''            if trade_signal:
                self.logger.info(f"Coordinator generated trade signal: {trade_signal.symbol} {trade_signal.direction}")
                
                # Execute the signal via ExecutionHandler
                try:
                    if self.execution_handler:
                        from ..core.data_contracts import ExecutionStatus
                        execution_report = await self.execution_handler.execute_trade_signal(trade_signal)
                        if execution_report and hasattr(execution_report, 'status'):
                            if execution_report.status == ExecutionStatus.FILLED:
                                self.logger.info(f"Trade executed successfully: {execution_report.order_id}")
                            else:
                                self.logger.warning(f"Trade execution status: {execution_report.status.value}")
                        else:
                            self.logger.warning("No valid execution report returned")
                    else:
                        self.logger.error("ExecutionHandler not available!")
                except Exception as exec_error:
                    self.logger.error(f"Signal execution error: {exec_error}")
                    # Continue feedback loop even if execution fails
            phase_durations["COORDINATOR_EXECUTION"]'''

content = content.replace(old_phase10, new_phase10)

print(f"Writing fixed file: {fixed_file}")
with open(fixed_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("SUCCESS: Fixed file created!")
print(f"   Location: {fixed_file}")
print("\nNext steps:")
print("1. Verify syntax: python -m py_compile " + str(fixed_file))
print("2. Backup old file")
print("3. Replace with new file")
