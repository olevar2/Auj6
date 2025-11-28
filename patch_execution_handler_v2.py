"""
Script to add DealMonitoringTeams integration to ExecutionHandler
Creates a patched version for testing before applying to production
"""

def patch_execution_handler():
    """Safely patch execution_handler.py with monitoring integration"""
    
    # Read original file
    input_file = 'e:/AUG6/auj_platform/src/trading_engine/execution_handler.py'
    output_file = 'e:/AUG6/auj_platform/src/trading_engine/execution_handler_PATCHED.py'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Read {len(lines)} lines from {input_file}")
    
    # Modification 1: Add position tracking after successful fill
    modified = []
    added_monitoring = False
    
    for i, line in enumerate(lines):
        modified.append(line)
        
        # Find the success log in _process_successful_fill
        if ('logger.info(f"Order {order.order_id} filled successfully:' in line and 
            not added_monitoring):
            # Add monitoring code after this line
            indent = '            '
            monitoring_block = [
                '\n',
                f'{indent}# Add position to monitoring system\n',
                f'{indent}if self.deal_monitoring_teams:\n',
                f'{indent}    try:\n',
                f'{indent}        signal_ref = getattr(order, "signal_ref", None)\n',
                f'{indent}        if signal_ref:\n',
                f'{indent}            await self.deal_monitoring_teams.add_position(\n',
                f'{indent}                deal_id=order.order_id,\n',
                f'{indent}                trade_signal=signal_ref,\n',
                f'{indent}                entry_price=order.filled_price,\n',
                f'{indent}                quantity=order.filled_quantity\n',
                f'{indent}            )\n',
                f'{indent}            logger.info(f"[MONITORING] Position {{order.order_id}} added")\n',
                f'{indent}    except Exception as e:\n',
                f'{indent}        logger.error(f"Failed to add position to monitoring: {{str(e)}}")\n',
            ]
            modified.extend(monitoring_block)
            added_monitoring = True
            print(f"[OK] Added monitoring integration at line {i+1}")
    
    # Modification 2: Store signal reference when creating ExecutionOrder
    final = []
    added_signal_ref = False
    
    for i, line in enumerate(modified):
        final.append(line)
        
        # Find ExecutionOrder creation and add signal_ref after
        if ('order = ExecutionOrder(' in line and 
            i + 1 < len(modified) and 
            not added_signal_ref):
            # Count forward to find closing parenthesis
            for j in range(i+1, min(i+20, len(modified))):
                final.append(modified[j])
                if ')' in modified[j] and 'metadata' in modified[j-1]:
                    # Add signal_ref storage after ExecutionOrder creation
                    indent = '            '
                    signal_storage = [
                        '\n',
                        f'{indent}# Store signal reference for monitoring\n',
                        f'{indent}order.signal_ref = signal\n',
                    ]
                    final.extend(signal_storage)
                    added_signal_ref = True
                    print(f"[OK] Added signal_ref storage at line {j+1}")
                    # Skip to avoid duplicate
                    for k in range(i+1, j+1):
                        if k < len(modified):
                            modified[k] = ''  # Mark as used
                    break
    
    # Write patched version
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(final if added_signal_ref else modified)
    
    print(f"\n[SUCCESS] Created patched version: {output_file}")
    print(f"Total lines: {len(final if added_signal_ref else modified)}")
    print(f"\nModifications applied:")
    print(f"  1. add_position() call: {'[OK]' if added_monitoring else '[FAIL]'}")
    print(f"  2. signal_ref storage: {'[OK]' if added_signal_ref else '[SKIP]'}")
    
    return output_file, added_monitoring

if __name__ == '__main__':
    try:
        output, success = patch_execution_handler()
        if success:
            print(f"\n[NEXT] Test syntax: python -m py_compile {output}")
        else:
            print(f"\n[WARN] Monitoring integration may be incomplete")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
