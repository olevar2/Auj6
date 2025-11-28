#!/usr/bin/env python3
"""
Add position tracking methods to DealMonitoringTeams safely.
"""

def main():
    file_path = "e:/AUG6/auj_platform/src/trading_engine/deal_monitoring_teams.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find shutdown method
    shutdown_line = None
    for i, line in enumerate(lines):
        if 'async def shutdown(self)' in line:
            shutdown_line = i
            break
    
    if not shutdown_line:
        print("ERROR: Could not find shutdown method")
        return False
    
    # Find the end of shutdown method (next method or class definition)
    insert_line = None
    for i in range(shutdown_line + 1, len(lines)):
        # Look for next method definition at same indentation
        if lines[i].strip().startswith('async def ') or lines[i].strip().startswith('def '):
            if not lines[i].startswith('        '):  # Check if it's at class level
                insert_line = i
                break
    
    if not insert_line:
        print("ERROR: Could not find insertion point")
        return False
    
    # Methods to add
    new_methods = '''
    def get_active_positions_count(self) -> int:
        """
        Get count of currently active positions.
        
        Returns:
            int: Number of active positions being monitored
        """
        return len(self.active_positions)
    
    def is_at_max_capacity(self, max_positions: int = 5) -> bool:
        """
        Check if we're at maximum position capacity.
        
        This is used by the coordinator to determine if new analysis
        should be performed or if we should focus on monitoring existing positions.
        
        Args:
            max_positions: Maximum allowed concurrent positions (default: 5)
            
        Returns:
            bool: True if at or above max capacity
        """
        return self.get_active_positions_count() >= max_positions
    
    def get_monitoring_mode(self, max_positions: int = 5) -> str:
        """
        Determine current monitoring mode based on active positions.
        
        Args:
            max_positions: Maximum allowed concurrent positions
            
        Returns:
            str: 'MONITORING_ONLY', 'LIGHT_ANALYSIS', or 'FULL_ANALYSIS'
        """
        active_count = self.get_active_positions_count()
        
        if active_count >= max_positions:
            return 'MONITORING_ONLY'
        elif active_count > 0:
            return 'LIGHT_ANALYSIS'
        else:
            return 'FULL_ANALYSIS'

'''
    
    # Insert the new methods
    lines.insert(insert_line, new_methods)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("SUCCESS: Added position tracking methods to DealMonitoringTeams")
    print(f"- get_active_positions_count()")
    print(f"- is_at_max_capacity()")
    print(f"- get_monitoring_mode()")
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
