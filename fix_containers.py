# Fix containers.py for robust_hourly_feedback_loop
import re

file_path = r'e:\AUG6\auj_platform\src\core\containers.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the robust_hourly_feedback_loop provider and check if it has correct parameters
if 'robust_hourly_feedback_loop' not in content:
    print("ERROR: robust_hourly_feedback_loop provider not found!")
else:
    # Check current state
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'robust_hourly_feedback_loop = providers.Singleton' in line:
            print(f"Found at line {i+1}: {line}")
            # Show next 15 lines
            for j in range(i, min(i+15, len(lines))):
                print(f"    {j+1}: {lines[j]}")
            break

    # Add execution_handler if missing
    if 'execution_handler=execution_handler' not in content or 'regime_classifier=regime_classifier,' not in content:
        # Need to add the missing lines
        pattern = r'(robust_hourly_feedback_loop = providers\.Singleton\(\s+RobustHourlyFeedbackLoop,\s+(?:.*\n)*?\s+economic_monitor=economic_monitor,)'
        replacement = r'\1\n        execution_handler=execution_handler,'
        
        # Try alternative pattern
        old_block = '''    robust_hourly_feedback_loop = providers.Singleton(
        RobustHourlyFeedbackLoop,
        walk_forward_validator=walk_forward_validator,
        performance_tracker=performance_tracker,
        indicator_analyzer=indicator_effectiveness_analyzer,
        agent_optimizer=agent_behavior_optimizer,
        hierarchy_manager=hierarchy_manager,
        genius_coordinator=genius_agent_coordinator,
        regime_classifier=regime_classifier,
        economic_monitor=economic_monitor,
        config=config
    )'''
        
        new_block = '''    robust_hourly_feedback_loop = providers.Singleton(
        RobustHourlyFeedbackLoop,
        walk_forward_validator=walk_forward_validator,
        performance_tracker=performance_tracker,
        indicator_analyzer=indicator_effectiveness_analyzer,
        agent_optimizer=agent_behavior_optimizer,
        hierarchy_manager=hierarchy_manager,
        genius_coordinator=genius_agent_coordinator,
        regime_classifier=regime_classifier,
        economic_monitor=economic_monitor,
        execution_handler=execution_handler,
        config=config
    )'''
        
        if old_block in content:
            content = content.replace(old_block, new_block)
            print("\\nFixed robust_hourly_feedback_loop provider!")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("File updated successfully!")
        else:
            print("\\nPattern not found - manual edit needed")
    else:
        print("\\nAlready has execution_handler parameter!")
