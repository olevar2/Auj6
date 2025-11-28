"""
Script to add DealMonitoringTeams integration to ExecutionHandler
This creates a patched version for testing
"""

def add_deal_monitoring_integration():
    """Add DealMonitoringTeams calls to ExecutionHandler"""
    
    # Read original file
    with open('e:/AUG6/auj_platform/src/trading_engine/execution_handler.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the _process_successful_fill method and add position tracking
    modified_lines = []
    in_process_fill = False
    fill_method_line = 0
    added_monitoring = False
    
    for i, line in enumerate(lines):
        modified_lines.append(line)
        
        # Find _process_successful_fill method
        if 'async def _process_successful_fill' in line:
            in_process_fill = True
            fill_method_line = i
            print(f"Found _process_successful_fill at line {i+1}")
        
        # Look for the logger.info about successful fill
        if in_process_fill and not added_monitoring:
            if 'logger.info(f"Order {order.order_id} filled successfully:' in line:
                # Add monitoring integration after this line
                indent = '            '  # Match indentation
                monitoring_code = f'''
{indent}# Add position to monitoring system
{indent}if self.deal_monitoring_teams:
{indent}    try:
{indent}        # Store signal reference in order for monitoring
{indent}        signal_ref = getattr(order, 'signal_ref', None)
{indent}        if signal_ref:
{indent}            await self.deal_monitoring_teams.add_position(
{indent}                deal_id=order.order_id,
{indent}                trade_signal=signal_ref,
{indent}                entry_price=order.filled_price,
{indent}                quantity=order.filled_quantity
{indent}            )
{indent}            logger.info(f"✅ Position {{order.order_id}} added to monitoring")
{indent}    except Exception as e:
{indent}        logger.error(f"Failed to add position to monitoring: {{str(e)}}")
'''
                modified_lines.append(monitoring_code)
                added_monitoring = True
                print(f"Added monitoring integration after line {i+1}")
    
    # Find _create_execution_order and add signal storage
    modified_lines2 = []
    added_signal_ref = False
    
    for i, line in enumerate(modified_lines):
        modified_lines2.append(line)
        
        # Find where ExecutionOrder is created
        if 'order = ExecutionOrder(' in line and not added_signal_ref:
            # Look for the closing parenthesis area
            pass
        
        # After metadata field, add signal_ref
        if '"risk_parameters":' in line and 'metadata' in modified_lines[i-2]:
            # Add signal reference storage
            indent = '            '
            signal_storage = f'{indent}# Store signal reference for monitoring\n{indent}order.signal_ref = signal\n'
            # Find next blank line to add
            for j in range(i+1, min(i+10, len(modified_lines))):
                if modified_lines[j].strip() == '' or ')' in modified_lines[j]:
                    modified_lines2.insert(j+1, signal_storage)
                    added_signal_ref = True
                    print(f"Added signal_ref storage around line {j+1}")
                    break
"""
Script to add DealMonitoringTeams integration to ExecutionHandler
This creates a patched version for testing
"""

def add_deal_monitoring_integration():
    """Add DealMonitoringTeams calls to ExecutionHandler"""
    
    # Read original file
    with open('e:/AUG6/auj_platform/src/trading_engine/execution_handler.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the _process_successful_fill method and add position tracking
    modified_lines = []
    in_process_fill = False
    fill_method_line = 0
    added_monitoring = False
    
    for i, line in enumerate(lines):
        modified_lines.append(line)
        
        # Find _process_successful_fill method
        if 'async def _process_successful_fill' in line:
            in_process_fill = True
            fill_method_line = i
            print(f"Found _process_successful_fill at line {i+1}")
        
        # Look for the logger.info about successful fill
        if in_process_fill and not added_monitoring:
            if 'logger.info(f"Order {order.order_id} filled successfully:' in line:
                # Add monitoring integration after this line
                indent = '            '  # Match indentation
                monitoring_code = f'''
{indent}# Add position to monitoring system
{indent}if self.deal_monitoring_teams:
{indent}    try:
{indent}        # Store signal reference in order for monitoring
{indent}        signal_ref = getattr(order, 'signal_ref', None)
{indent}        if signal_ref:
{indent}            await self.deal_monitoring_teams.add_position(
{indent}                deal_id=order.order_id,
{indent}                trade_signal=signal_ref,
{indent}                entry_price=order.filled_price,
{indent}                quantity=order.filled_quantity
{indent}            )
{indent}            logger.info(f"[OK] Position {{order.order_id}} added to monitoring")
{indent}    except Exception as e:
{indent}        logger.error(f"Failed to add position to monitoring: {{str(e)}}")
'''
                modified_lines.append(monitoring_code)
                added_monitoring = True
                print(f"Added monitoring integration after line {i+1}")
    
    # Find _create_execution_order and add signal storage
    modified_lines2 = []
    added_signal_ref = False
    
    for i, line in enumerate(modified_lines):
        modified_lines2.append(line)
        
        # Find where ExecutionOrder is created
        if 'order = ExecutionOrder(' in line and not added_signal_ref:
            # Look for the closing parenthesis area
            pass
        
        # After metadata field, add signal_ref
        if '"risk_parameters":' in line and 'metadata' in modified_lines[i-2]:
            # Add signal reference storage
            indent = '            '
            signal_storage = f'{indent}# Store signal reference for monitoring\n{indent}order.signal_ref = signal\n'
            # Find next blank line to add
            for j in range(i+1, min(i+10, len(modified_lines))):
                if modified_lines[j].strip() == '' or ')' in modified_lines[j]:
                    modified_lines2.insert(j+1, signal_storage)
                    added_signal_ref = True
                    print(f"Added signal_ref storage around line {j+1}")
                    break
    
    # Write to new file for testing
    output_file = 'e:/AUG6/auj_platform/src/trading_engine/execution_handler_PATCHED.py'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines2 if added_signal_ref else modified_lines)
    
    print(f"\n[OK] Created patched version: {output_file}")
    print(f"Total lines: {len(modified_lines2 if added_signal_ref else modified_lines)}")
    print(f"Modifications:")
    print(f"  - add_position() call: {'[OK] added' if added_monitoring else '[FAIL] NOT FOUND'}")
    print(f"  - signal_ref storage: {'[OK] added' if added_signal_ref else '[WARN] skipped'}")
    
    return output_file

if __name__ == '__main__':
    try:
        output = add_deal_monitoring_integration()
        print(f"\n[SUCCESS] Next step: Test syntax with: python -m py_compile {output}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
