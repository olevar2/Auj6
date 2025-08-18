"""
Comprehensive test suite for parallel coordination performance optimizations.

Tests parallel agent analysis, early decision system, phase merging, and performance validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal

# Import the classes we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from auj_platform.src.coordination.genius_agent_coordinator import (
    GeniusAgentCoordinator, AnalysisCycleState, AnalysisCyclePhase
)
from auj_platform.src.coordination.performance_monitor import (
    CoordinationPerformanceMonitor, CyclePerformance, PhasePerformance
)
from auj_platform.src.core.data_contracts import (
    TradeSignal, AgentDecision, MarketConditions, AgentRank,
    MarketRegime, ConfidenceLevel, TradeDirection
)


class TestParallelCoordination:
    """Test parallel agent analysis functionality."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create mock configuration manager."""
        config = Mock()
        config.get_int = Mock(side_effect=lambda key, default: {
            'coordination.max_analysis_time_seconds': 120,
            'coordination.max_concurrent_agents': 3,
            'coordination.analysis_frequency_minutes': 60,
            'coordination.elite_set_update_hours': 24,
            'coordination.indicator_periods': 100
        }.get(key, default))
        
        config.get_float = Mock(side_effect=lambda key, default: {
            'coordination.minimum_confidence_threshold': 0.6,
            'coordination.consensus_threshold': 0.6
        }.get(key, default))
        
        config.get_bool = Mock(side_effect=lambda key, default: {
            'coordination.enable_parallel_analysis': True,
            'coordination.enable_phase_merging': True,
            'coordination.comprehensive_analysis_enabled': True
        }.get(key, default))
        
        config.get_str = Mock(side_effect=lambda key, default: {
            'coordination.primary_symbol': 'EURUSD',
            'coordination.primary_timeframe': '1H'
        }.get(key, default))
        
        return config

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        agents = {}
        
        for agent_name in ['alpha_agent', 'beta_agent_1', 'beta_agent_2', 'gamma_agent_1']:
            agent = Mock()
            agent.is_ready_for_analysis = Mock(return_value=True)
            agent.get_assigned_indicators = Mock(return_value=['rsi_14', 'macd', 'sma_20'])
            agent.get_required_data_types = Mock(return_value=['OHLCV', 'volume'])
            
            # Create realistic analysis result
            analysis_result = Mock()
            analysis_result.decision = "BUY" if 'alpha' in agent_name else "BUY"
            analysis_result.confidence = 0.8 if 'alpha' in agent_name else 0.7
            analysis_result.indicators_used = ['rsi_14', 'macd']
            
            # Make perform_analysis async
            async def mock_perform_analysis(*args, **kwargs):
                # Simulate processing time
                await asyncio.sleep(0.1)
                return analysis_result
            
            agent.perform_analysis = mock_perform_analysis
            agents[agent_name] = agent
            
        return agents

    @pytest.fixture
    def coordinator(self, mock_config_manager, mock_agents):
        """Create coordinator instance for testing."""
        # Mock other dependencies
        hierarchy_manager = Mock()
        indicator_engine = Mock()
        data_manager = Mock()
        risk_manager = Mock()
        execution_handler = Mock()
        
        # Create coordinator
        coordinator = GeniusAgentCoordinator(
            config_manager=mock_config_manager,
            hierarchy_manager=hierarchy_manager,
            indicator_engine=indicator_engine,
            data_manager=data_manager,
            risk_manager=risk_manager,
            execution_handler=execution_handler
        )
        
        # Set mock agents
        coordinator.agents = mock_agents
        
        # Mock indicator executor
        coordinator.indicator_executor = Mock()
        async def mock_execute_indicators(requests):
            results = []
            for request in requests:
                result = Mock()
                result.status.value = "success"
                result.indicator_name = request.indicator_name
                result.data = {'value': 50.0, 'signal': 'neutral'}
                results.append(result)
            return results
        coordinator.indicator_executor.execute_indicators = mock_execute_indicators
        
        # Mock regime classifier
        coordinator.regime_classifier = Mock()
        coordinator.regime_classifier.classify_regime = AsyncMock(return_value=MarketRegime.TRENDING_UP)
        
        return coordinator

    @pytest.fixture
    def sample_cycle_state(self):
        """Create sample cycle state for testing."""
        cycle_state = AnalysisCycleState(
            cycle_id="test_cycle_001",
            phase=AnalysisCyclePhase.INITIALIZATION,
            start_time=datetime.utcnow(),
            symbol="EURUSD"
        )
        
        cycle_state.alpha_agent = "alpha_agent"
        cycle_state.beta_agents = ["beta_agent_1", "beta_agent_2"]
        cycle_state.gamma_agents = ["gamma_agent_1"]
        cycle_state.performance_data = {
            'market_data': {
                'OHLCV': pd.DataFrame({
                    'open': [1.1000, 1.1010, 1.1020],
                    'high': [1.1015, 1.1025, 1.1035],
                    'low': [1.0995, 1.1005, 1.1015],
                    'close': [1.1010, 1.1020, 1.1030],
                    'volume': [1000, 1100, 1200]
                })
            }
        }
        
        return cycle_state

    @pytest.mark.asyncio
    async def test_parallel_agent_analysis_performance(self, coordinator, sample_cycle_state):
        """Test that parallel analysis is faster than sequential."""
        # Setup indicator results
        sample_cycle_state.performance_data['indicator_results'] = {
            'rsi_14': {'value': 50.0},
            'macd': {'signal': 'neutral'},
            'sma_20': {'value': 1.1015}
        }
        
        # Setup market conditions
        sample_cycle_state.market_conditions = MarketConditions(
            symbol="EURUSD",
            regime=MarketRegime.TRENDING_UP,
            volatility=0.5,
            trend_strength=0.7,
            volume_profile={'low': 0.3, 'medium': 0.4, 'high': 0.3},
            support_levels=[],
            resistance_levels=[],
            key_indicators={}
        )
        
        start_time = asyncio.get_event_loop().time()
        await coordinator._phase_agent_analysis(sample_cycle_state)
        execution_time = asyncio.get_event_loop().time() - start_time

        # Should complete in under 3 minutes (vs 15+ minutes sequential)
        assert execution_time < 180, f"Parallel analysis took {execution_time}s, expected < 180s"

        # Should have results from all agents
        assert len(sample_cycle_state.agent_decisions) >= 3
        
        # Verify agents were called
        for agent_name in coordinator.agents:
            if agent_name in ['alpha_agent', 'beta_agent_1', 'beta_agent_2', 'gamma_agent_1']:
                assert agent_name in sample_cycle_state.agent_decisions

    @pytest.mark.asyncio
    async def test_comprehensive_analysis_system(self, coordinator, sample_cycle_state):
        """Test comprehensive analysis system processes all indicators."""
        # Setup comprehensive analysis test
        sample_cycle_state.performance_data['market_data'] = {
            'OHLCV': pd.DataFrame({
                'open': [1.1000], 'high': [1.1015], 'low': [1.0995],
                'close': [1.1010], 'volume': [1000]
            })
        }

        # Mock comprehensive indicator execution
        coordinator.indicator_executor.execute_indicators = AsyncMock(return_value=[
            Mock(status=Mock(value="success"), indicator_name=f"indicator_{i}",
                 data={'value': 50.0 + i}) for i in range(20)  # All 20 indicators
        ])

        start_time = asyncio.get_event_loop().time()
        await coordinator._phase_indicator_calculation(sample_cycle_state)
        execution_time = asyncio.get_event_loop().time() - start_time

        # Should process all indicators consistently
        assert sample_cycle_state.performance_data.get('comprehensive_analysis', False)
        indicator_results = sample_cycle_state.performance_data.get('indicator_results', {})
        assert len(indicator_results) >= 8, f"Should calculate at least essential indicators, got {len(indicator_results)}"
        assert execution_time < 60, "Comprehensive analysis should complete efficiently"
        
        # Verify comprehensive analysis flag is set
        assert sample_cycle_state.performance_data.get('validation_type') == 'COMPREHENSIVE'

    @pytest.mark.asyncio
    async def test_hierarchy_integrity_maintained(self, coordinator):
        """Test that hierarchy system remains intact."""
        # Verify hierarchy manager integration
        assert hasattr(coordinator, 'hierarchy_manager')
        
        # Test agent role assignment
        cycle_state = AnalysisCycleState(
            cycle_id="test_hierarchy",
            phase=AnalysisCyclePhase.INITIALIZATION,
            start_time=datetime.utcnow(),
            symbol="EURUSD"
        )
        
        # Mock hierarchy manager responses
        coordinator.hierarchy_manager.get_current_rankings = AsyncMock(return_value={
            'alpha_agent': AgentRank.ALPHA,
            'beta_agent_1': AgentRank.BETA,
            'beta_agent_2': AgentRank.BETA,
            'gamma_agent_1': AgentRank.GAMMA
        })
        
        await coordinator._assign_agent_roles(cycle_state)
        
        # Verify hierarchy is maintained
        assert cycle_state.alpha_agent == 'alpha_agent'
        assert 'beta_agent_1' in cycle_state.beta_agents
        assert 'beta_agent_2' in cycle_state.beta_agents
        assert 'gamma_agent_1' in cycle_state.gamma_agents

    def test_configuration_integration(self, coordinator):
        """Test configuration integration."""
        # Verify configuration values are loaded correctly
        assert coordinator.max_analysis_time_seconds == 120
        assert coordinator.enable_parallel_analysis == True
        assert coordinator.enable_phase_merging == True

        # VERIFY early decision components are removed:
        assert not hasattr(coordinator, 'enable_early_decisions')
        assert not hasattr(coordinator, 'early_decision_confidence_threshold')

        # VERIFY comprehensive analysis is default:
        assert getattr(coordinator, 'comprehensive_analysis_enabled', True) == True


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""

    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing."""
        return CoordinationPerformanceMonitor(max_history=100)

    def test_cycle_monitoring_basic(self, performance_monitor):
        """Test basic cycle monitoring."""
        # Start monitoring
        performance_monitor.start_cycle_monitoring("test_cycle", {'parallel': True})
        
        assert performance_monitor.current_cycle is not None
        assert performance_monitor.current_cycle.cycle_id == "test_cycle"
        
        # End monitoring
        performance_monitor.end_cycle_monitoring(
            success=True, 
            agents_processed=5,
            indicators_calculated=10,
            signal_generated=True
        )
        
        assert performance_monitor.current_cycle is None
        assert len(performance_monitor.cycle_history) == 1
        
        cycle = performance_monitor.cycle_history[0]
        assert cycle.success == True
        assert cycle.agents_processed == 5
        assert cycle.indicators_calculated == 10
        assert cycle.signal_generated == True

    def test_performance_report_generation(self, performance_monitor):
        """Test performance report generation."""
        # Add some test data
        for i in range(10):
            performance_monitor.start_cycle_monitoring(f"cycle_{i}")
            performance_monitor.start_phase_monitoring("agent_analysis", parallel_operations=5)
            performance_monitor.end_phase_monitoring(success=True)
            performance_monitor.end_cycle_monitoring(
                success=True,
                agents_processed=5,
                indicators_calculated=8
            )
        
        report = performance_monitor.get_performance_report()
        
        assert report['status'] == 'success'
        assert report['cycles_analyzed'] == 10
        assert report['successful_cycles'] == 10
        assert 'average_cycle_time_seconds' in report
        assert 'performance_improvement_percent' in report
        assert 'parallel_efficiency_avg' in report


if __name__ == "__main__":
    # Run tests with performance reporting
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--durations=10"
    ])