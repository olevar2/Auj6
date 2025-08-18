"""
Platform-Wide Exception Integration
==================================
Utilities for applying enhanced exception handling across the entire AUJ Platform
without requiring major code modifications.

Author: AUJ Platform Development Team  
Date: 2025-07-04
Version: 1.0.0
"""
import logging
import asyncio
import inspect
from typing import Dict, Any, List, Optional, Type, Callable
from contextlib import contextmanager
from ..core.enhanced_exceptions import get_global_error_registry, AUJException
from ..core.exception_utils import enhance_existing_component, with_exception_handling


class PlatformExceptionIntegrator:
    """
    Centralized system for applying enhanced exception handling across
    the entire AUJ Platform without major code modifications.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.enhanced_components = {}
        self.integration_stats = {
            'components_enhanced': 0,
            'methods_wrapped': 0,
            'errors_prevented': 0,
            'recovery_actions': 0
        }
    
    def integrate_platform_exception_handling(self, container) -> Dict[str, Any]:
        """
        Apply enhanced exception handling to all platform components.
        
        Args:
            container: Dependency injection container
            
        Returns:
            Dictionary of enhancement results
        """
        self.logger.info("Starting platform-wide exception handling integration...")
        
        results = {
            'enhanced_components': [],
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Define critical components to enhance
            critical_components = [
                ('genius_agent_coordinator', 'GeniusAgentCoordinator'),
                ('execution_handler', 'ExecutionHandler'),
                ('dynamic_risk_manager', 'DynamicRiskManager'),
                ('signal_processor', 'SignalProcessor'),
                ('portfolio_manager', 'PortfolioManager'),
                ('mt5_client', 'MT5Client'),
                ('data_provider', 'DataProvider'),
                ('config_manager', 'ConfigManager'),
                ('account_manager', 'AccountManager'),
                ('deal_monitor', 'DealMonitor')
            ]
            
            # Enhance each component
            for component_attr, component_name in critical_components:
                try:
                    enhanced = self._enhance_component(container, component_attr, component_name)
                    if enhanced:
                        results['enhanced_components'].append(component_name)
                        self.integration_stats['components_enhanced'] += 1
                except Exception as e:
                    error_msg = f"Failed to enhance {component_name}: {str(e)}"
                    self.logger.warning(error_msg)
                    results['warnings'].append(error_msg)
            
            # Setup global monitoring
            self._setup_global_monitoring()
            
            # Setup automated recovery
            self._setup_automated_recovery()
            
            # Collect statistics
            results['statistics'] = self.integration_stats.copy()
            
            self.logger.info(f"Platform exception integration completed. Enhanced {len(results['enhanced_components'])} components.")
            
        except Exception as e:
            error_msg = f"Platform exception integration failed: {str(e)}"
            self.logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def _enhance_component(self, container, component_attr: str, component_name: str) -> bool:
        """Enhance a specific component if it exists in the container"""
        try:
            if hasattr(container, component_attr):
                # Get the component factory or instance
                component_factory = getattr(container, component_attr)
                
                if callable(component_factory):
                    # It's a factory method, get the instance
                    component_instance = component_factory()
                else:
                    # It's already an instance
                    component_instance = component_factory
                
                # Enhance the component
                enhanced_component = enhance_existing_component(component_instance, component_name)
                
                # Store enhanced component
                self.enhanced_components[component_name] = enhanced_component
                
                # Count methods wrapped
                method_count = len([m for m in dir(enhanced_component) 
                                  if callable(getattr(enhanced_component, m)) 
                                  and not m.startswith('_')])
                self.integration_stats['methods_wrapped'] += method_count
                
                self.logger.info(f"Enhanced {component_name} with {method_count} methods wrapped")
                return True
                
        except Exception as e:
            self.logger.warning(f"Could not enhance {component_name}: {str(e)}")
            return False
        
        return False
    
    def _setup_global_monitoring(self):
        """Setup global error monitoring and alerting"""
        registry = get_global_error_registry()
        
        def monitoring_alert_handler(alert_data: Dict[str, Any]):
            """Handle global error alerts with comprehensive monitoring"""
            try:
                component = alert_data.get('component', 'Unknown')
                severity = alert_data.get('severity', 'medium')
                error_record = alert_data.get('error_record', {})
                
                # Log the alert
                if severity in ['high', 'critical']:
                    self.logger.critical(f"CRITICAL ERROR - {component}: {error_record.get('error_message', 'Unknown')}")
                elif severity == 'medium':
                    self.logger.error(f"ERROR - {component}: {error_record.get('error_message', 'Unknown')}")
                else:
                    self.logger.warning(f"WARNING - {component}: {error_record.get('error_message', 'Unknown')}")
                
                # Update statistics
                if severity in ['high', 'critical']:
                    self.integration_stats['errors_prevented'] += 1
                
                # Trigger monitoring actions
                self._trigger_monitoring_actions(alert_data)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring alert handler: {str(e)}")
        
        registry.add_global_alert_callback(monitoring_alert_handler)
        self.logger.info("Global error monitoring setup completed")
    
    def _setup_automated_recovery(self):
        """Setup automated recovery actions for common error patterns"""
        registry = get_global_error_registry()
        
        def recovery_alert_handler(alert_data: Dict[str, Any]):
            """Handle alerts with automated recovery actions"""
            try:
                error_record = alert_data.get('error_record', {})
                error_type = error_record.get('error_type', '')
                component = alert_data.get('component', '')
                severity = alert_data.get('severity', 'medium')
                
                # Only attempt recovery for non-critical errors
                if severity not in ['high', 'critical']:
                    recovery_success = self._attempt_recovery(component, error_type, error_record)
                    if recovery_success:
                        self.integration_stats['recovery_actions'] += 1
                
            except Exception as e:
                self.logger.error(f"Error in recovery alert handler: {str(e)}")
        
        registry.add_global_alert_callback(recovery_alert_handler)
        self.logger.info("Automated recovery system setup completed")
    
    def _trigger_monitoring_actions(self, alert_data: Dict[str, Any]):
        """Trigger appropriate monitoring actions based on alert data"""
        component = alert_data.get('component', '')
        severity = alert_data.get('severity', 'medium')
        
        # Define monitoring actions
        if severity == 'critical':
            # Critical errors might require immediate attention
            self._send_critical_alert(alert_data)
        elif severity == 'high':
            # High severity errors should be tracked
            self._track_high_severity_error(alert_data)
        
        # Always update component health metrics
        self._update_component_health_metrics(component, alert_data)
    
    def _attempt_recovery(self, component: str, error_type: str, error_record: Dict[str, Any]) -> bool:
        """Attempt automated recovery for specific error types"""
        recovery_actions = {
            'ConnectionError': self._recover_connection,
            'TimeoutError': self._recover_timeout,
            'ValidationError': self._recover_validation,
            'ConfigurationError': self._recover_configuration,
            'DataProviderError': self._recover_data_provider
        }
        
        recovery_func = recovery_actions.get(error_type)
        if recovery_func:
            try:
                self.logger.info(f"Attempting recovery for {component} - {error_type}")
                return recovery_func(component, error_record)
            except Exception as e:
                self.logger.error(f"Recovery attempt failed for {component}: {str(e)}")
                return False
        
        return False
    
    def _recover_connection(self, component: str, error_record: Dict[str, Any]) -> bool:
        """Recover from connection errors"""
        self.logger.info(f"Attempting connection recovery for {component}")
        # Here you would implement actual connection recovery logic
        # For now, just log the attempt
        return True
    
    def _recover_timeout(self, component: str, error_record: Dict[str, Any]) -> bool:
        """Recover from timeout errors"""
        self.logger.info(f"Attempting timeout recovery for {component}")
        # Here you would implement timeout adjustment logic
        return True
    
    def _recover_validation(self, component: str, error_record: Dict[str, Any]) -> bool:
        """Recover from validation errors"""
        self.logger.info(f"Attempting validation recovery for {component}")
        # Here you would implement validation rule reset logic
        return True
    
    def _recover_configuration(self, component: str, error_record: Dict[str, Any]) -> bool:
        """Recover from configuration errors"""
        self.logger.info(f"Attempting configuration recovery for {component}")
        # Here you would implement configuration reload logic
        return True
    
    def _recover_data_provider(self, component: str, error_record: Dict[str, Any]) -> bool:
        """Recover from data provider errors"""
        self.logger.info(f"Attempting data provider recovery for {component}")
        # Here you would implement data provider reconnection logic
        return True
    
    def _send_critical_alert(self, alert_data: Dict[str, Any]):
        """Send critical error alerts"""
        # Here you would implement alert sending logic
        # (email, Slack, Discord, etc.)
        self.logger.critical(f"CRITICAL ALERT TRIGGERED: {alert_data}")
    
    def _track_high_severity_error(self, alert_data: Dict[str, Any]):
        """Track high severity errors for pattern analysis"""
        # Here you would implement error pattern tracking
        self.logger.error(f"HIGH SEVERITY ERROR TRACKED: {alert_data}")
    
    def _update_component_health_metrics(self, component: str, alert_data: Dict[str, Any]):
        """Update component health metrics"""
        # Here you would implement health metrics updating
        self.logger.debug(f"Updated health metrics for {component}")
    
    def get_integration_report(self) -> Dict[str, Any]:
        """Get comprehensive integration report"""
        registry = get_global_error_registry()
        global_summary = registry.get_global_error_summary()
        
        return {
            'integration_statistics': self.integration_stats,
            'enhanced_components': list(self.enhanced_components.keys()),
            'global_error_summary': global_summary,
            'component_health': self._get_component_health_status(),
            'recommendations': self._get_integration_recommendations()
        }
    
    def _get_component_health_status(self) -> Dict[str, str]:
        """Get health status for all enhanced components"""
        health_status = {}
        
        for component_name in self.enhanced_components.keys():
            # Here you would implement actual health checking
            # For now, assume all are healthy
            health_status[component_name] = 'HEALTHY'
        
        return health_status
    
    def _get_integration_recommendations(self) -> List[str]:
        """Get recommendations for improving exception handling integration"""
        recommendations = []
        
        if self.integration_stats['components_enhanced'] < 5:
            recommendations.append("Consider enhancing more platform components for better coverage")
        
        if self.integration_stats['errors_prevented'] > 10:
            recommendations.append("High error rate detected - investigate underlying causes")
        
        if self.integration_stats['recovery_actions'] > 5:
            recommendations.append("High recovery rate - consider improving component reliability")
        
        return recommendations


# Global integrator instance
_platform_integrator = None

def get_platform_integrator() -> PlatformExceptionIntegrator:
    """Get global platform exception integrator instance"""
    global _platform_integrator
    if _platform_integrator is None:
        _platform_integrator = PlatformExceptionIntegrator()
    return _platform_integrator

def integrate_platform_wide_exception_handling(container) -> Dict[str, Any]:
    """
    Convenience function to integrate exception handling across the platform.
    
    Args:
        container: Dependency injection container
        
    Returns:
        Integration results
    """
    integrator = get_platform_integrator()
    return integrator.integrate_platform_exception_handling(container)

def get_platform_exception_report() -> Dict[str, Any]:
    """
    Get comprehensive platform exception handling report.
    
    Returns:
        Detailed report of exception handling status
    """
    integrator = get_platform_integrator()
    return integrator.get_integration_report()


# Utility functions for quick integration
@contextmanager
def temporary_exception_enhancement(component, component_name: str):
    """
    Temporarily enhance a component with exception handling.
    Useful for testing or temporary operations.
    """
    enhanced = enhance_existing_component(component, component_name)
    try:
        yield enhanced
    finally:
        # Cleanup if needed
        pass

def quick_enhance_component(component, component_name: str):
    """
    Quick utility to enhance a single component.
    
    Args:
        component: Component instance to enhance
        component_name: Name for error tracking
        
    Returns:
        Enhanced component instance
    """
    return enhance_existing_component(component, component_name)

def apply_safe_execution_to_method(obj, method_name: str, component_name: str):
    """
    Apply safe execution to a specific method of an object.
    
    Args:
        obj: Object containing the method
        method_name: Name of the method to enhance
        component_name: Component name for error tracking
    """
    original_method = getattr(obj, method_name)
    
    @with_exception_handling(component_name=component_name, recoverable=True)
    async def enhanced_method(*args, **kwargs):
        if asyncio.iscoroutinefunction(original_method):
            return await original_method(*args, **kwargs)
        else:
            return original_method(*args, **kwargs)
    
    setattr(obj, method_name, enhanced_method)


# Testing utilities
def test_platform_exception_integration():
    """Test the platform exception integration system"""
    
    class MockContainer:
        """Mock container for testing"""
        def genius_agent_coordinator(self):
            return MockComponent("GeniusAgentCoordinator")
        
        def execution_handler(self):
            return MockComponent("ExecutionHandler")
    
    class MockComponent:
        """Mock component for testing"""
        def __init__(self, name):
            self.name = name
        
        def process_data(self, data):
            if data == "error":
                raise ValueError("Test error")
            return f"Processed {data}"
    
    # Test integration
    container = MockContainer()
    integrator = get_platform_integrator()
    
    results = integrator.integrate_platform_exception_handling(container)
    print(f"Integration results: {results}")
    
    # Test enhanced component
    enhanced = integrator.enhanced_components.get('GeniusAgentCoordinator')
    if enhanced:
        try:
            result = enhanced.process_data("test")
            print(f"Success: {result}")
        except Exception as e:
            print(f"Error handled: {e}")
    
    # Get integration report
    report = integrator.get_integration_report()
    print(f"Integration report: {report}")

if __name__ == "__main__":
    test_platform_exception_integration()