"""
Component Interfaces for AUJ Platform
=====================================
Minimal interfaces to break circular dependencies without major restructuring.

Author: AUJ Platform Development Team
Date: 2025-07-04
Version: 1.0.0
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, TYPE_CHECKING

class ComponentInterface(Protocol):
    """Minimal component interface"""
    async def initialize(self) -> bool: ...
    async def cleanup(self) -> None: ...

class ConfigurableComponent(Protocol):
    """Component that accepts configuration"""
    def __init__(self, config: Dict[str, Any], **kwargs): ...

class MonitoringComponent(Protocol):
    """Component that provides monitoring capabilities"""
    def get_status(self) -> Dict[str, Any]: ...
    async def health_check(self) -> bool: ...

class AgentComponent(Protocol):
    """Basic agent interface"""
    async def process_signal(self, signal: Any) -> Any: ...
    def get_performance_metrics(self) -> Dict[str, Any]: ...

# Type hints for lazy loading - prevents circular imports
if TYPE_CHECKING:
    from ..agents.base_agent import BaseAgent
    from ..coordination.genius_agent_coordinator import GeniusAgentCoordinator
    from trading_engine.deal_monitoring_teams import DealMonitoringTeams
    from monitoring.metrics_collector import MetricsCollector
    from monitoring.health_checker import HealthChecker

# Component registry for lazy initialization
class ComponentRegistry:
    """Registry for managing component instances without circular dependencies"""
    
    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
    
    def register_factory(self, name: str, factory: callable):
        """Register a factory function for a component"""
        self._factories[name] = factory
    
    def get_component(self, name: str, *args, **kwargs) -> Any:
        """Get or create a component instance"""
        if name not in self._components:
            if name in self._factories:
                self._components[name] = self._factories[name](*args, **kwargs)
            else:
                raise ValueError(f"Component '{name}' not registered")
        return self._components[name]
    
    def register_instance(self, name: str, instance: Any):
        """Register a pre-created instance"""
        self._components[name] = instance
    
    def clear(self):
        """Clear all registered components"""
        self._components.clear()
        self._factories.clear()

# Global component registry
_global_registry = ComponentRegistry()

def get_component_registry() -> ComponentRegistry:
    """Get the global component registry"""
    return _global_registry