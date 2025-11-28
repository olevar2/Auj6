#!/usr/bin/env python3
"""Apply remaining critical patches"""
import sys

def main():
    # Patch 1: Fix regime detection in RobustHourlyFeedbackLoop
    file1 = "e:/AUG6/auj_platform/src/learning/robust_hourly_feedback_loop.py"
    with open(file1, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace(
        '        return await self.regime_classifier.detect_current_regime()',
        '        return self.regime_classifier.get_current_regime()  # Synchronous method'
    )
    
    with open(file1, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("SUCCESS: Fixed regime detection method call")
    
    # Patch 2: Add update_elite_indicators stub to GeniusAgentCoordinator
    file2 = "e:/AUG6/auj_platform/src/coordination/genius_agent_coordinator.py"
    with open(file2, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add method at end of class before final line
    stub_method = '''
    async def update_elite_indicators(self, current_regime, indicator_updates):
        """Update elite indicators based on effectiveness analysis."""
        self.logger.info(f"Updating elite indicators for regime: {current_regime}")
        
        changes = {
            'updated_count': 0,
            'promoted': 0,
            'demoted': 0
        }
        
        try:
            # TODO: Implement elite indicator update logic
            # This will promote/demote indicators based on effectiveness
            self.logger.warning("update_elite_indicators not fully implemented yet")
            
        except Exception as e:
            self.logger.error(f"Failed to update elite indicators: {e}")
        
        return changes
'''
    
    # Find a good place to insert (before the last method or at end of class)
    if 'async def update_elite_indicators' not in content:
        # Insert before the final closing of the class
        insertion_point = content.rfind('\n\n')
        if insertion_point > 0:
            content = content[:insertion_point] + stub_method + content[insertion_point:]
            
            with open(file2, 'w', encoding='utf-8') as f:
                f.write(content)
            print("SUCCESS: Added update_elite_indicators stub to GeniusAgentCoordinator")
        else:
            print("INFO: Could not find insertion point for update_elite_indicators")
    else:
        print("INFO: update_elite_indicators already exists")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
