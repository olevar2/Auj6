"""
Example: Enhanced Exception Handling Integration
==============================================
Demonstrates how to integrate enhanced exception handling into existing components
without major code modifications.

Author: AUJ Platform Development Team
Date: 2025-07-04
Version: 1.0.0
"""
import logging
from typing import Dict, Any, Optional
from ..core.exception_utils import with_exception_handling, enhance_existing_component, safe_execution
from ..core.enhanced_exceptions import get_global_error_registry

# Example 1: Using decorator to enhance existing methods
class ExampleEnhancedComponent:
    """Example of how to enhance an existing component with minimal changes"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Register with global error registry happens automatically via decorator
    
    @with_exception_handling(component_name="ExampleComponent", recoverable=True)
    async def existing_method_with_enhancement(self, param1: str, param2: int) -> Dict[str, Any]:
        """
        Existing method enhanced with exception handling via decorator.
        No changes to method logic required.
        """
        # Original method logic remains unchanged
        result = await self._do_complex_work(param1, param2)
        return result
    
    async def _do_complex_work(self, param1: str, param2: int) -> Dict[str, Any]:
        """Simulate complex work that might fail"""
        if param2 < 0:
            raise ValueError("param2 must be positive")
        return {"result": f"processed {param1} with {param2}"}

# Example 2: Wrapping existing components without modification
def enhance_existing_coordinator(coordinator):
    """
    Example of how to enhance an existing coordinator component
    without modifying its source code.
    """
    # Wrap the existing coordinator
    enhanced_coordinator = enhance_existing_component(coordinator, "GeniusAgentCoordinator")
    
    # Now all method calls are automatically wrapped with error handling
    return enhanced_coordinator

# Example 3: Using context manager for safe execution blocks
async def safe_trading_operation(trading_engine, trade_data: Dict[str, Any]):
    """Example of using safe execution context manager"""
    
    with safe_execution('TradingEngine', 'execute_trade') as safe:
        # Risky trading operation
        result = await trading_engine.execute_trade(trade_data)
    
    if safe.success:
        logging.info("Trading operation completed successfully")
        return result
    else:
        logging.error(f"Trading operation failed: {safe.error}")
        return None

# Example 4: Applying to multiple components at startup
def enhance_platform_components(container):
    """
    Example of how to enhance multiple platform components at startup.
    This can be called during platform initialization.
    """
    enhanced_components = {}
    
    # Enhance key components
    if hasattr(container, 'genius_agent_coordinator'):
        enhanced_components['coordinator'] = enhance_existing_component(
            container.genius_agent_coordinator(), 
            "GeniusAgentCoordinator"
        )
    
    if hasattr(container, 'execution_handler'):
        enhanced_components['execution'] = enhance_existing_component(
            container.execution_handler(),
            "ExecutionHandler"
        )
    
    if hasattr(container, 'dynamic_risk_manager'):
        enhanced_components['risk_manager'] = enhance_existing_component(
            container.dynamic_risk_manager(),
            "DynamicRiskManager"
        )
    
    # Set up global error monitoring
    setup_global_error_monitoring()
    
    return enhanced_components

def setup_global_error_monitoring():
    """Setup global error monitoring and alerting"""
    registry = get_global_error_registry()
    
    # Add alert callback for critical errors
    def critical_error_alert(alert_data: Dict[str, Any]):
        severity = alert_data.get('severity', 'medium')
        if severity in ['high', 'critical']:
            component = alert_data.get('component', 'Unknown')
            error_msg = alert_data.get('error_record', {}).get('error_message', 'Unknown error')
            
            logging.critical(f"CRITICAL ERROR ALERT - Component: {component}, Error: {error_msg}")
            
            # Here you could add:
            # - Email notifications
            # - Slack/Discord alerts  
            # - System notifications
            # - Automatic recovery actions
    
    registry.add_global_alert_callback(critical_error_alert)

# Example 5: Integration with existing configuration
def apply_exception_handling_to_existing_platform(platform_instance):
    """
    Example of how to apply enhanced exception handling to an existing
    platform instance without major refactoring.
    """
    original_methods = {}
    
    # Store original methods
    for method_name in ['initialize', 'start', 'shutdown', 'process_signal']:
        if hasattr(platform_instance, method_name):
            original_methods[method_name] = getattr(platform_instance, method_name)
    
    # Create enhanced wrapper
    enhanced_platform = enhance_existing_component(platform_instance, "AUJPlatform")
    
    # The enhanced platform now has automatic error handling for all methods
    return enhanced_platform

# Example 6: Error reporting and monitoring
def create_error_dashboard_data():
    """Create data for error monitoring dashboard"""
    registry = get_global_error_registry()
    summary = registry.get_global_error_summary()
    
    dashboard_data = {
        'overview': {
            'total_components': summary['total_components'],
            'total_errors': summary['total_errors'],
            'critical_alerts': summary['critical_alerts'],
            'timestamp': summary['timestamp']
        },
        'component_health': {},
        'error_trends': {},
        'recommendations': []
    }
    
    # Process component data
    for component_name, stats in summary['components'].items():
        error_rate = stats.get('error_rate_per_hour', 0)
        
        if error_rate > 10:
            health_status = 'UNHEALTHY'
            dashboard_data['recommendations'].append(
                f"Component {component_name} has high error rate: {error_rate:.1f} errors/hour"
            )
        elif error_rate > 5:
            health_status = 'WARNING'
        else:
            health_status = 'HEALTHY'
        
        dashboard_data['component_health'][component_name] = {
            'status': health_status,
            'error_rate': error_rate,
            'total_errors': stats.get('total_errors', 0),
            'last_error': stats.get('last_error_time')
        }
    
    return dashboard_data

# Example 7: Automated recovery actions
async def setup_automated_recovery():
    """Setup automated recovery actions for common error patterns"""
    registry = get_global_error_registry()
    
    def recovery_alert_handler(alert_data: Dict[str, Any]):
        """Handle alerts with automated recovery"""
        error_record = alert_data.get('error_record', {})
        error_type = error_record.get('error_type', '')
        component = alert_data.get('component', '')
        
        # Define recovery actions for specific error patterns
        recovery_actions = {
            'ConnectionError': 'restart_connection',
            'TimeoutError': 'increase_timeout',
            'MemoryError': 'clear_cache',
            'ValidationError': 'reset_validation_rules'
        }
        
        if error_type in recovery_actions:
            action = recovery_actions[error_type]
            logging.info(f"Attempting automated recovery for {component}: {action}")
            
            # Here you would implement actual recovery logic
            # For example:
            # if action == 'restart_connection':
            #     await restart_component_connection(component)
    
    registry.add_global_alert_callback(recovery_alert_handler)

# Example 8: Testing enhanced error handling
def test_enhanced_error_handling():
    """Test the enhanced error handling system"""
    import asyncio
    
    async def test_async():
        # Create test component
        test_component = ExampleEnhancedComponent(None)
        
        # Test successful operation
        try:
            result = await test_component.existing_method_with_enhancement("test", 5)
            print(f"Success: {result}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        # Test error handling
        try:
            result = await test_component.existing_method_with_enhancement("test", -1)
            print(f"Should not reach here: {result}")
        except Exception as e:
            print(f"Expected error handled: {e}")
        
        # Get error statistics
        registry = get_global_error_registry()
        summary = registry.get_global_error_summary()
        print(f"Error summary: {summary}")
    
    # Run test
    asyncio.run(test_async())

if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_enhanced_error_handling()