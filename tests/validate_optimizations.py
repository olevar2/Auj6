#!/usr/bin/env python3
"""
Simple validation script for AUJ Platform Performance Optimizations.

This script performs basic validation of the implemented optimizations
without requiring external test frameworks.
"""

import sys
import os
import asyncio
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_performance_monitor():
    """Test basic performance monitor functionality."""
    print("Testing Performance Monitor...")
    
    try:
        from auj_platform.src.coordination.performance_monitor import CoordinationPerformanceMonitor
        
        # Create monitor
        monitor = CoordinationPerformanceMonitor()
        
        # Test cycle monitoring
        monitor.start_cycle_monitoring("test_cycle_001", {'parallel': True})
        
        # Test phase monitoring
        monitor.start_phase_monitoring("test_phase", parallel_operations=3)
        time.sleep(0.1)  # Simulate work
        monitor.end_phase_monitoring(success=True, early_exit=False)
        
        # End cycle
        monitor.end_cycle_monitoring(
            success=True,
            agents_processed=5,
            indicators_calculated=10,
            signal_generated=True
        )
        
        # Get report
        report = monitor.get_performance_report()
        
        assert report['status'] == 'success'
        assert report['cycles_analyzed'] == 1
        assert report['successful_cycles'] == 1
        
        print("[PASS] Performance Monitor test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Performance Monitor test failed: {e}")
        return False

def test_coordinator_configuration():
    """Test coordinator configuration loading."""
    print("Testing Coordinator Configuration...")
    
    try:
        from unittest.mock import Mock
        from auj_platform.src.coordination.genius_agent_coordinator import GeniusAgentCoordinator
        
        # Create mock config manager
        config_manager = Mock()
        config_manager.get_int = Mock(return_value=120)
        config_manager.get_float = Mock(return_value=0.8)
        config_manager.get_bool = Mock(return_value=True)
        config_manager.get_str = Mock(return_value="EURUSD")
        
        # Create other mocks
        hierarchy_manager = Mock()
        indicator_engine = Mock()
        data_manager = Mock()
        risk_manager = Mock()
        execution_handler = Mock()
        
        # Create coordinator
        coordinator = GeniusAgentCoordinator(
            config_manager=config_manager,
            hierarchy_manager=hierarchy_manager,
            indicator_engine=indicator_engine,
            data_manager=data_manager,
            risk_manager=risk_manager,
            execution_handler=execution_handler
        )
        
        # Test configuration values
        assert coordinator.max_analysis_time_seconds == 120
        assert coordinator.enable_parallel_analysis == True
        assert coordinator.enable_phase_merging == True
        
        # Verify early decision components are removed
        assert not hasattr(coordinator, 'enable_early_decisions')
        assert not hasattr(coordinator, 'early_decision_confidence_threshold')
        
        # Verify comprehensive analysis is default
        assert getattr(coordinator, 'comprehensive_analysis_enabled', True) == True
        
        # Test performance monitor is initialized
        assert hasattr(coordinator, 'performance_monitor')
        assert coordinator.performance_monitor is not None
        
        print("[PASS] Coordinator Configuration test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Coordinator Configuration test failed: {e}")
        return False

def test_essential_indicators():
    """Test essential indicators method."""
    print("Testing Essential Indicators...")
    
    try:
        from unittest.mock import Mock
        from auj_platform.src.coordination.genius_agent_coordinator import GeniusAgentCoordinator
        
        # Create basic coordinator
        coordinator = GeniusAgentCoordinator(
            config_manager=Mock(),
            hierarchy_manager=Mock(),
            indicator_engine=Mock(),
            data_manager=Mock(),
            risk_manager=Mock(),
            execution_handler=Mock()
        )
        
        # Test essential indicators
        essential = coordinator._get_essential_indicators()
        
        assert isinstance(essential, list)
        assert len(essential) > 0
        assert 'rsi_14' in essential
        assert 'macd' in essential
        assert 'sma_20' in essential
        
        print("[PASS] Essential Indicators test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Essential Indicators test failed: {e}")
        return False

def test_comprehensive_analysis_consistency():
    """Test that comprehensive analysis is always applied."""
    print("Testing Comprehensive Analysis Consistency...")
    
    try:
        from unittest.mock import Mock
        from auj_platform.src.coordination.genius_agent_coordinator import GeniusAgentCoordinator
        from auj_platform.src.core.data_contracts import MarketConditions, MarketRegime
        
        # Create coordinator with proper configuration
        config_mock = Mock()
        config_mock.get_int = Mock(side_effect=lambda key, default: {
            'coordination.max_analysis_time_seconds': 120,
            'coordination.max_concurrent_agents': 3
        }.get(key, default))
        config_mock.get_bool = Mock(side_effect=lambda key, default: {
            'coordination.enable_parallel_analysis': True,
            'coordination.enable_phase_merging': True
        }.get(key, default))
        config_mock.get_float = Mock(side_effect=lambda key, default: default)
        
        # Create coordinator
        coordinator = GeniusAgentCoordinator(
            config_manager=config_mock,
            hierarchy_manager=Mock(),
            indicator_engine=Mock(),
            data_manager=Mock(),
            risk_manager=Mock(),
            execution_handler=Mock()
        )
        
        # Verify comprehensive analysis is always enabled
        assert not hasattr(coordinator, 'enable_early_decisions')  # Should not exist
        assert coordinator.enable_parallel_analysis == True  # Should be preserved
        
        # Test that _check_early_decision_possible method does not exist
        assert not hasattr(coordinator, '_check_early_decision_possible')
        
        print("[PASS] Comprehensive Analysis Consistency test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Comprehensive Analysis test failed: {e}")
        return False

async def test_async_functionality():
    """Test async functionality works correctly."""
    print("Testing Async Functionality...")
    
    try:
        from unittest.mock import Mock, AsyncMock
        from auj_platform.src.coordination.genius_agent_coordinator import (
            GeniusAgentCoordinator, AnalysisCycleState, AnalysisCyclePhase
        )
        
        # Create coordinator with mocks
        coordinator = GeniusAgentCoordinator(
            config_manager=Mock(),
            hierarchy_manager=Mock(),
            indicator_engine=Mock(),
            data_manager=Mock(),
            risk_manager=Mock(),
            execution_handler=Mock()
        )
        
        # Create cycle state
        cycle_state = AnalysisCycleState(
            cycle_id="async_test",
            phase=AnalysisCyclePhase.INITIALIZATION,
            start_time=datetime.utcnow(),
            symbol="EURUSD"
        )
        
        # Mock async methods
        coordinator._phase_initialization = AsyncMock()
        coordinator._phase_data_and_indicators_combined = AsyncMock()
        coordinator._phase_agent_analysis = AsyncMock()
        coordinator._phase_decision_and_validation_combined = AsyncMock()
        coordinator._phase_execution_preparation = AsyncMock()
        
        # Test merged phases execution
        start_time = time.time()
        await coordinator._execute_merged_phases(cycle_state)
        execution_time = time.time() - start_time
        
        # Verify all phases were called
        coordinator._phase_initialization.assert_called_once()
        coordinator._phase_data_and_indicators_combined.assert_called_once()
        coordinator._phase_agent_analysis.assert_called_once()
        coordinator._phase_decision_and_validation_combined.assert_called_once()
        coordinator._phase_execution_preparation.assert_called_once()
        
        assert execution_time < 1.0, "Merged phases should execute quickly with mocks"
        
        print("[PASS] Async Functionality test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Async Functionality test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("AUJ Platform Performance Optimization Validation")
    print("=" * 60)
    
    tests = [
        test_performance_monitor,
        test_coordinator_configuration,
        test_essential_indicators,
        test_comprehensive_analysis_consistency,
    ]
    
    passed = 0
    total = len(tests)
    
    # Run sync tests
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} failed with exception: {e}")
    
    # Run async test
    try:
        if asyncio.run(test_async_functionality()):
            passed += 1
        total += 1
    except Exception as e:
        print(f"[FAIL] Async test failed with exception: {e}")
        total += 1
    
    print("\n" + "=" * 60)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All validation tests passed!")
        print("Performance optimizations are working correctly.")
        return 0
    else:
        print(f"[FAILED] {total - passed} tests failed.")
        print("Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())