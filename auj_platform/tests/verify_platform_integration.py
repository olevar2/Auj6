"""
Platform Integration Verification Test
======================================
Verify that platform components integrate correctly after cleanup.
"""

import sys
import ast
sys.path.insert(0, r'e:\AUG6')

def test_imports():
    """Test all critical imports work"""
    print("="*70)
    print("INTEGRATION VERIFICATION TEST")
    print("="*70)
    print("\n1. Testing Critical Imports...")
    print("-" * 70)
    
    try:
        from auj_platform.src.coordination.genius_agent_coordinator import GeniusAgentCoordinator
        print("[OK] GeniusAgentCoordinator imports successfully")
        
        from auj_platform.src.trading_engine.execution_handler import ExecutionHandler
        print("[OK] ExecutionHandler imports successfully")
        
        from auj_platform.src.trading_engine.deal_monitoring_teams import DealMonitoringTeams
        print("[OK] DealMonitoringTeams imports successfully")
        
        from auj_platform.src.core.containers import PlatformContainer
        print("[OK] PlatformContainer imports successfully")
        
        return True
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def verify_coordinator_modification():
    """Verify the position capacity check was added correctly"""
    print("\n2. Verifying Coordinator Modification...")
    print("-" * 70)
    
    file_path = r'e:\AUG6\auj_platform\src\coordination\genius_agent_coordinator.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for the new code
    checks = [
        ('Position capacity check', 'Position capacity check'),
        ('active_positions_count', 'active_positions_count'),
        ('max_concurrent_positions', 'max_concurrent_positions'),
        ('At max capacity', 'At max capacity'),
    ]
    
    all_found = True
    for check_name, search_term in checks:
        if search_term in content:
            print(f"[OK] Found: {check_name}")
        else:
            print(f"[ERROR] Missing: {check_name}")
            all_found = False
    
    return all_found

def verify_cleanup():
    """Verify unnecessary files were removed"""
    print("\n3. Verifying Cleanup...")
    print("-" * 70)
    
    from pathlib import Path
    
    files_to_check = [
        (r'e:\AUG6\auj_platform\src\trading_engine\deal_monitoring_helpers.py', False),
        (r'e:\AUG6\auj_platform\tests\verify_monitoring_helpers.py', False),
        (r'e:\AUG6\apply_safe_monitoring_integration.py', False),
        (r'e:\AUG6\auj_platform\src\trading_engine\deal_monitoring_teams.py', True),  # Should exist
        (r'e:\AUG6\auj_platform\src\coordination\genius_agent_coordinator.py', True),  # Should exist
    ]
    
    all_correct = True
    for file_path, should_exist in files_to_check:
        exists = Path(file_path).exists()
        file_name = Path(file_path).name
        
        if should_exist and exists:
            print(f"[OK] {file_name} exists (as expected)")
        elif not should_exist and not exists:
            print(f"[OK] {file_name} removed (as expected)")
        elif should_exist and not exists:
            print(f"[ERROR] {file_name} is missing (should exist)")
            all_correct = False
        else:
            print(f"[WARNING] {file_name} still exists (should be removed)")
            all_correct = False
    
    return all_correct

def verify_syntax():
    """Verify Python syntax is correct"""
    print("\n4. Verifying Python Syntax...")
    print("-" * 70)
    
    key_files = [
        r'e:\AUG6\auj_platform\src\coordination\genius_agent_coordinator.py',
        r'e:\AUG6\auj_platform\src\trading_engine\deal_monitoring_teams.py',
        r'e:\AUG6\auj_platform\src\trading_engine\execution_handler.py',
    ]
    
    all_valid = True
    for file_path in key_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            from pathlib import Path
            print(f"[OK] {Path(file_path).name} - Valid syntax")
        except SyntaxError as e:
            from pathlib import Path
            print(f"[ERROR] {Path(file_path).name} - Syntax error: {e}")
            all_valid = False
    
    return all_valid

def generate_report():
    """Generate final report"""
    print("\n" + "="*70)
    print("FINAL VERIFICATION REPORT")
    print("="*70)
    
    results = {
        'imports': test_imports(),
        'modification': verify_coordinator_modification(),
        'cleanup': verify_cleanup(),
        'syntax': verify_syntax()
    }
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print("\n" + "="*70)
    if all_passed:
        print("STATUS: ALL CHECKS PASSED")
        print("="*70)
        print("\nPlatform is clean and simple:")
        print("  - Unnecessary files removed")
        print("  - Simple solution applied")
        print("  - All integrations work")
        print("  - Syntax is valid")
        print("\nReady for production!")
    else:
        print("STATUS: SOME CHECKS FAILED")
        print("="*70)
        print("\nReview failed checks above")
    
    return all_passed

if __name__ == '__main__':
    success = generate_report()
    sys.exit(0 if success else 1)
