"""
Indicator Registry - Dynamic Indicator Discovery and Registration
=================================================================

This module provides a centralized registry for all indicator implementations,
replacing the simple placeholder with a sophisticated auto-discovery system.

Key Features:
- Automatic discovery of all indicators from indicators directory
- Dynamic instantiation with parameter caching
- Factory pattern for indicator creation
- Backward compatible with dashboard usage

This solves the critical architectural flaw where 148 sophisticated indicators
were implemented but never called.
"""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Type, Optional, Any
import logging

logger = logging.getLogger(__name__)


class IndicatorRegistry:
    """
    Centralized registry for all indicator implementations.
    
    Automatically discovers indicator classes from the indicators directory
    and provides a factory method to instantiate them.
    
    Updated from simple placeholder to full auto-discovery system.
    """
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
        self._instances: Dict[str, Any] = {}  # Cache for singleton instances
        self.logger = logging.getLogger(__name__)
        
    def register_indicator(self, indicator_name: str, indicator_class: Type) -> None:
        """
        Register an indicator class manually.
        
        Backward compatible with old API: register_indicator(name, class)
        """
        self._registry[indicator_name] = indicator_class
        self.logger.debug(f"Registered indicator: {indicator_name}")
    
    # Alias for backward compatibility
    register = register_indicator
        
    def get_indicator(self, indicator_name: str) -> Optional[Type]:
        """
        Get the indicator class by name.
        
        Backward compatible with old API: get_indicator(name)
        """
        return self._registry.get(indicator_name)
    
    # New method for getting class
    def get_indicator_class(self, indicator_name: str) -> Optional[Type]:
        """Get the indicator class by name (new API)."""
        return self._registry.get(indicator_name)
        
    def get_indicator_instance(self, indicator_name: str, **kwargs) -> Optional[Any]:
        """
        Get or create an indicator instance with optional parameters.
        
        Args:
            indicator_name: Name of the indicator (e.g., 'super_trend_indicator')
            **kwargs: Constructor parameters for the indicator
            
        Returns:
            Indicator instance or None if not found
        """
        # Create cache key with params
        cache_key = f"{indicator_name}:{str(sorted(kwargs.items()))}"
        
        # Check cache first
        if cache_key in self._instances:
            return self._instances[cache_key]
            
        # Get class and instantiate
        indicator_class = self.get_indicator_class(indicator_name)
        if indicator_class:
            try:
                instance = indicator_class(**kwargs)
                self._instances[cache_key] = instance
                return instance
            except Exception as e:
                self.logger.error(f"Failed to instantiate {indicator_name}: {e}")
                return None
        
        return None
        
    def is_registered(self, indicator_name: str) -> bool:
        """Check if an indicator is registered."""
        return indicator_name in self._registry
    
    def list_indicators(self) -> list:
        """
        List all registered indicator names.
        
        Backward compatible with old API: list_indicators()
        """
        return list(self._registry.keys())
        
    def get_all_indicator_names(self) -> list:
        """Get all registered indicator names (new API)."""
        return list(self._registry.keys())
        
    def discover_and_register_all(self, base_path: Optional[Path] = None) -> int:
        """
        Automatically discover and register all indicators.
        
        Args:
            base_path: Base path to indicators directory
            
        Returns:
            Number of indicators registered
        """
        if base_path is None:
            # Auto-detect base path
            current_file = Path(__file__)
            # Navigate from src/registry/indicator_registry.py to src/indicator_engine/indicators
            base_path = current_file.parent.parent / "indicator_engine" / "indicators"
            
        if not base_path.exists():
            self.logger.warning(f"Indicators directory not found: {base_path}")
            return 0
            
        registered_count = 0
        
        # Scan all subdirectories
        for category_dir in base_path.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('_'):
                continue
                
            # Scan all Python files in category
            for indicator_file in category_dir.glob("*.py"):
                if indicator_file.name.startswith('_'):
                    continue
                    
                try:
                    # Build module path
                    # From: src/indicator_engine/indicators/trend/super_trend_indicator.py
                    # To: src.indicator_engine.indicators.trend.super_trend_indicator
                    relative_path = indicator_file.relative_to(base_path.parent)
                    module_path = str(relative_path).replace('/', '.').replace('\\', '.')[:-3]
                    full_module_path = f"src.indicator_engine.{module_path}"
                    
                    # Import module
                    try:
                        module = importlib.import_module(full_module_path)
                    except ModuleNotFoundError:
                        # Try alternative import path (without src prefix)
                        module_path_alt = str(relative_path).replace('/', '.').replace('\\', '.')[:-3]
                        try:
                            module = importlib.import_module(module_path_alt)
                        except:
                            # Last attempt: direct path from base
                            indicator_path = f"indicator_engine.{module_path}"
                            module = importlib.import_module(indicator_path)
                    
                    # Find indicator classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Check if it's an indicator class (ends with 'Indicator')
                        if name.endswith('Indicator') and hasattr(obj, 'calculate'):
                            # Convert class name to registry name
                            # e.g., SuperTrendIndicator -> super_trend_indicator
                            indicator_name = self._class_name_to_registry_name(name)
                            
                            self.register_indicator(indicator_name, obj)
                            registered_count += 1
                            
                except Exception as e:
                    self.logger.warning(f"Failed to import {indicator_file.name}: {e}")
                    continue
                    
        self.logger.info(f"✅ Registered {registered_count} indicators from {base_path}")
        return registered_count
        
    def _class_name_to_registry_name(self, class_name: str) -> str:
        """
        Convert class name to registry name.
        
        Examples:
            SuperTrendIndicator -> super_trend_indicator
            RSIIndicator -> rsi_indicator
            MACDIndicator -> macd_indicator
        """
        # Remove 'Indicator' suffix
        name = class_name[:-9] if class_name.endswith('Indicator') else class_name
        
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        # Add 'indicator' suffix
        return f"{name}_indicator"


# Global registry instance
_global_registry = None


def get_indicator_registry() -> IndicatorRegistry:
    """Get the global indicator registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = IndicatorRegistry()
        # Auto-discover on first access
        try:
            _global_registry.discover_and_register_all()
        except Exception as e:
            logger.error(f"Failed to auto-discover indicators: {e}")
    return _global_registry


def register_indicator(indicator_name: str, indicator_class: Type) -> None:
    """
    Register an indicator class in the global registry.
    
    Convenience function for backward compatibility.
    """
    registry = get_indicator_registry()
    registry.register_indicator(indicator_name, indicator_class)
