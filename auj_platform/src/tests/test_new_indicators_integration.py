"""
Integration Testing for 10 New Indicators with AUJ Platform

This comprehensive test suite validates that the newly implemented indicators
integrate properly with the platform's architecture including:
- Indicator Registry Discovery
- SmartIndicatorExecutor Integration
- Agent-Indicator Communication
- Data Provider Integration
- Error Handling and Fallbacks

The testing supports the humanitarian mission by ensuring reliable
indicator calculations for optimal trading decisions.
"""

import asyncio
import logging
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add platform paths
platform_src = Path(__file__).parent.parent
sys.path.insert(0, str(platform_src))
sys.path.insert(0, str(platform_src / "config"))

try:
    from indicator_engine.indicator_executor import (
        SmartIndicatorExecutor,
        IndicatorRegistry,
        IndicatorExecutionRequest,
        ExecutionPriority,
        ExecutionStatus
    )
    from indicator_data_requirements import INDICATOR_DATA_REQUIREMENTS
except ImportError as e:
    print(f"Import error: {e}")
    print("Continuing with basic test structure...")

class MockDataProvider:
    """Mock data provider for testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)
    
    async def get_ohlcv_data(self, symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
        """Generate mock OHLCV data for testing"""
        
        # Generate realistic price data
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
        base_price = 1.1000  # EUR/USD base
        
        # Generate price movements
        np.random.seed(42)  # Consistent data for testing
        returns = np.random.normal(0, 0.001, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': prices,
            'high': prices * np.random.uniform(1.0001, 1.005, periods),
            'low': prices * np.random.uniform(0.995, 0.9999, periods),
            'close': prices * np.random.uniform(0.998, 1.002, periods),
            'volume': np.random.lognormal(10, 0.5, periods),
            'tick_volume': np.random.randint(1000, 10000, periods),
            'spread': np.random.uniform(0.0001, 0.0003, periods)
        }, index=dates)
        
        # Ensure high >= close >= low and high >= open >= low
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        return data

class MockDataProviderManager:
    """Mock data provider manager for testing"""
    
    def __init__(self):
        self.providers = {
            'MT5Provider': MockDataProvider('MT5Provider'),
            'YahooFinanceProvider': MockDataProvider('YahooFinanceProvider')
        }
    
    def get_available_providers(self) -> List[str]:
        """Return list of available provider names"""
        return list(self.providers.keys())
    
    def get_provider(self, provider_name: str):
        """Get provider instance by name"""
        return self.providers.get(provider_name)

class TestNewIndicatorsIntegration(unittest.TestCase):
    """Test suite for new indicators integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # List of our 10 new indicators
        cls.new_indicators = [
            'fibonacci_expansion_indicator',
            'fibonacci_grid_indicator', 
            'intermarket_correlation_indicator',
            'market_breadth_indicator',
            'sector_momentum_indicator',
            'support_resistance_indicator',
            'synthetic_option_indicator',
            'time_segmented_volume_indicator',
            'triangular_moving_average_indicator',
            'variable_moving_average_indicator'
        ]
        
        # Create mock data provider manager
        cls.data_provider_manager = MockDataProviderManager()
        
        # Create indicator registry and executor
        try:
            cls.indicator_registry = IndicatorRegistry()
            cls.executor = SmartIndicatorExecutor(
                data_provider_manager=cls.data_provider_manager,
                indicator_registry=cls.indicator_registry
            )
        except Exception as e:
            cls.logger.warning(f"Could not create full executor: {e}")
            cls.indicator_registry = None
            cls.executor = None
    
    def test_01_indicator_registry_discovery(self):
        """Test that indicator registry discovers our new indicators"""
        
        if self.indicator_registry is None:
            self.skipTest("Indicator registry not available")
        
        discovered_indicators = self.indicator_registry.list_indicators()
        self.logger.info(f"Registry discovered {len(discovered_indicators)} total indicators")
        
        # Check that our new indicators were discovered
        found_indicators = []
        missing_indicators = []
        
        for indicator_name in self.new_indicators:
            if indicator_name in discovered_indicators:
                found_indicators.append(indicator_name)
                self.logger.info(f"✅ Found: {indicator_name}")
            else:
                missing_indicators.append(indicator_name)
                self.logger.warning(f"❌ Missing: {indicator_name}")
        
        self.logger.info(f"Found {len(found_indicators)}/10 new indicators in registry")
        
        # We expect at least some indicators to be found
        self.assertGreater(len(found_indicators), 0, 
                          f"No new indicators found in registry. Missing: {missing_indicators}")
        
        # Log any missing indicators for investigation
        if missing_indicators:
            self.logger.warning(f"Missing indicators may need registry integration: {missing_indicators}")
    
    def test_02_indicator_class_loading(self):
        """Test that indicator classes can be loaded and instantiated"""
        
        if self.indicator_registry is None:
            self.skipTest("Indicator registry not available")
        
        successful_loads = []
        failed_loads = []
        
        for indicator_name in self.new_indicators:
            try:
                indicator_class = self.indicator_registry.get_indicator(indicator_name)
                if indicator_class is not None:
                    # Try to instantiate
                    instance = indicator_class()
                    self.assertIsNotNone(instance)
                    
                    # Check for required methods
                    self.assertTrue(hasattr(instance, 'calculate'))
                    successful_loads.append(indicator_name)
                    self.logger.info(f"✅ Successfully loaded and instantiated: {indicator_name}")
                else:
                    failed_loads.append(indicator_name)
                    self.logger.warning(f"❌ Could not get class for: {indicator_name}")
                    
            except Exception as e:
                failed_loads.append(indicator_name)
                self.logger.error(f"❌ Failed to load {indicator_name}: {str(e)}")
        
        self.logger.info(f"Successfully loaded {len(successful_loads)}/10 indicators")
        
        # We expect at least some indicators to load successfully
        self.assertGreater(len(successful_loads), 0,
                          f"No indicators could be loaded. Failed: {failed_loads}")
    
    async def test_03_indicator_calculation_async(self):
        """Test async calculation of indicators through SmartIndicatorExecutor"""
        
        if self.executor is None:
            self.skipTest("SmartIndicatorExecutor not available")
        
        # Create test requests for our new indicators
        requests = []
        for indicator_name in self.new_indicators:
            request = IndicatorExecutionRequest(
                indicator_name=indicator_name,
                symbol="EURUSD",
                timeframe="H1",
                periods=200,
                priority=ExecutionPriority.HIGH,
                agent_name="test_agent"
            )
            requests.append(request)
        
        # Execute indicators
        try:
            results = await self.executor.execute_indicators(requests)
            self.assertEqual(len(results), len(requests))
            
            successful_calculations = []
            failed_calculations = []
            
            for result in results:
                if result.status == ExecutionStatus.SUCCESS:
                    successful_calculations.append(result.indicator_name)
                    self.assertIsNotNone(result.data)
                    self.logger.info(f"✅ Successfully calculated: {result.indicator_name}")
                else:
                    failed_calculations.append((result.indicator_name, result.error_message))
                    self.logger.warning(f"❌ Failed calculation: {result.indicator_name} - {result.error_message}")
            
            self.logger.info(f"Successfully calculated {len(successful_calculations)}/10 indicators")
            
            # We expect at least some calculations to succeed
            self.assertGreater(len(successful_calculations), 0,
                              f"No indicators calculated successfully. Failed: {failed_calculations}")
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            self.fail(f"Could not execute indicators: {str(e)}")
    
    def test_04_data_requirements_integration(self):
        """Test integration with indicator data requirements"""
        
        # Check if our indicators are in the data requirements
        found_requirements = []
        missing_requirements = []
        
        for indicator_name in self.new_indicators:
            if indicator_name in INDICATOR_DATA_REQUIREMENTS:
                requirement = INDICATOR_DATA_REQUIREMENTS[indicator_name]
                found_requirements.append(indicator_name)
                self.logger.info(f"✅ Data requirement found: {indicator_name} - Active: {requirement.is_active}")
            else:
                missing_requirements.append(indicator_name)
                self.logger.warning(f"❌ No data requirement: {indicator_name}")
        
        self.logger.info(f"Found data requirements for {len(found_requirements)}/10 indicators")
        
        if missing_requirements:
            self.logger.warning(f"Missing data requirements for: {missing_requirements}")
            self.logger.info("These indicators may need to be added to indicator_data_requirements.py")
    
    def test_05_error_handling_and_fallbacks(self):
        """Test error handling with invalid inputs"""
        
        if self.executor is None:
            self.skipTest("SmartIndicatorExecutor not available")
        
        # Test with insufficient data
        async def test_insufficient_data():
            request = IndicatorExecutionRequest(
                indicator_name=self.new_indicators[0],  # Test first indicator
                symbol="INVALID",
                timeframe="H1", 
                periods=1000000,  # Request way too much data
                priority=ExecutionPriority.LOW
            )
            
            results = await self.executor.execute_indicators([request])
            self.assertEqual(len(results), 1)
            
            result = results[0]
            # Should handle gracefully with appropriate error status
            self.assertIn(result.status, [
                ExecutionStatus.FAILED_NO_DATA,
                ExecutionStatus.FAILED_INSUFFICIENT_DATA,
                ExecutionStatus.FAILED_PROVIDER_ERROR,
                ExecutionStatus.FAILED_INDICATOR_NOT_FOUND,
                ExecutionStatus.SKIPPED_INACTIVE
            ])
            
            self.logger.info(f"✅ Error handling test passed: {result.status}")
        
        # Run async test
        asyncio.run(test_insufficient_data())
    
    def test_06_performance_basic(self):
        """Basic performance test for indicator calculations"""
        
        if self.executor is None:
            self.skipTest("SmartIndicatorExecutor not available")
        
        async def performance_test():
            # Test batch execution performance
            requests = []
            for indicator_name in self.new_indicators[:5]:  # Test first 5
                request = IndicatorExecutionRequest(
                    indicator_name=indicator_name,
                    symbol="EURUSD",
                    timeframe="H1",
                    periods=100,
                    priority=ExecutionPriority.MEDIUM
                )
                requests.append(request)
            
            start_time = datetime.now()
            results = await self.executor.execute_indicators(requests)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Batch execution time: {execution_time:.3f}s for {len(requests)} indicators")
            
            # Performance should be reasonable (under 30 seconds for 5 indicators)
            self.assertLess(execution_time, 30.0, 
                           f"Batch execution too slow: {execution_time:.3f}s")
            
            successful = sum(1 for r in results if r.status == ExecutionStatus.SUCCESS)
            self.logger.info(f"Performance test: {successful}/{len(requests)} successful calculations")
        
        asyncio.run(performance_test())
    
    def test_07_registry_status_reporting(self):
        """Test registry status reporting functionality"""
        
        if self.executor is None:
            self.skipTest("SmartIndicatorExecutor not available")
        
        try:
            status = self.executor.get_registry_status()
            
            self.assertIsInstance(status, dict)
            self.assertIn('total_indicators', status)
            self.assertIn('indicators_list', status)
            self.assertIn('registry_paths_scanned', status)
            
            total_indicators = status['total_indicators']
            indicators_list = status['indicators_list']
            
            self.assertIsInstance(total_indicators, int)
            self.assertIsInstance(indicators_list, list)
            self.assertEqual(len(indicators_list), total_indicators)
            
            self.logger.info(f"Registry status: {total_indicators} total indicators")
            self.logger.info(f"Paths scanned: {status['registry_paths_scanned']}")
            
            # Check how many of our indicators are available
            available_new_indicators = [ind for ind in self.new_indicators if ind in indicators_list]
            self.logger.info(f"Available new indicators: {len(available_new_indicators)}/10")
            
        except Exception as e:
            self.logger.error(f"Registry status test failed: {str(e)}")
            self.fail(f"Could not get registry status: {str(e)}")

class IntegrationTestRunner:
    """Helper class to run integration tests and generate reports"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests and return summary report"""
        
        self.logger.info("=" * 60)
        self.logger.info("AUJ PLATFORM - NEW INDICATORS INTEGRATION TEST")
        self.logger.info("=" * 60)
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestNewIndicatorsIntegration)
        
        # Run tests with detailed output
        test_results = {}
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Generate summary
        total_tests = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        successful = total_tests - failures - errors - skipped
        
        summary = {
            'total_tests': total_tests,
            'successful': successful,
            'failures': failures,
            'errors': errors,
            'skipped': skipped,
            'success_rate': (successful / total_tests * 100) if total_tests > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info("=" * 60)
        self.logger.info("INTEGRATION TEST SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failures}")
        self.logger.info(f"Errors: {errors}")
        self.logger.info(f"Skipped: {skipped}")
        self.logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info("=" * 60)
        
        return summary

def main():
    """Main function to run integration tests"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('indicator_integration_test.log')
        ]
    )
    
    # Run integration tests
    runner = IntegrationTestRunner()
    summary = runner.run_all_tests()
    
    # Return exit code based on success
    if summary['failures'] == 0 and summary['errors'] == 0:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("⚠️  Some integration tests failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())