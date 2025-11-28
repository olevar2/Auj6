"""
Agent Registry - Dynamic Agent Discovery and Registration
=========================================================

This module provides a centralized registry for all agent implementations,
enabling automatic discovery and registration of expert agents.

Key Features:
- Automatic discovery of all agents from agents directory
- Dynamic instantiation with dependency injection
- Factory pattern for agent creation

This solves the "phantom agent" issue where valid agent files were ignored.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, Type, Optional, Any, List
import logging

from ..agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Centralized registry for all agent implementations.
    
    Automatically discovers agent classes from the agents directory
    and provides methods to instantiate them.
    """
    
    def __init__(self):
        self._registry: Dict[str, Type[BaseAgent]] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_agent(self, agent_name: str, agent_class: Type[BaseAgent]) -> None:
        """Register an agent class manually."""
        self._registry[agent_name] = agent_class
        self.logger.debug(f"Registered agent class: {agent_name}")
        
    def get_agent_class(self, agent_name: str) -> Optional[Type[BaseAgent]]:
        """Get the agent class by name."""
        return self._registry.get(agent_name)
        
    def get_all_agent_classes(self) -> Dict[str, Type[BaseAgent]]:
        """Get all registered agent classes."""
        return self._registry.copy()
        
    def discover_and_register_all(self, base_path: Optional[Path] = None) -> int:
        """
        Automatically discover and register all agents.
        
        Args:
            base_path: Base path to agents directory
            
        Returns:
            Number of agents registered
        """
        if base_path is None:
            # Auto-detect base path: src/registry/agent_registry.py -> src/agents
            current_file = Path(__file__)
            base_path = current_file.parent.parent / "agents"
            
        if not base_path.exists():
            self.logger.warning(f"Agents directory not found: {base_path}")
            return 0
            
        registered_count = 0
        
        # Scan all Python files in directory
        for agent_file in base_path.glob("*.py"):
            if agent_file.name.startswith('_') or agent_file.name == "base_agent.py":
                continue
                
            try:
                # Build module path
                # From: src/agents/trend_agent.py
                # To: src.agents.trend_agent
                relative_path = agent_file.relative_to(base_path.parent)
                module_path = str(relative_path).replace('/', '.').replace('\\', '.')[:-3]
                full_module_path = f"src.{module_path}"
                
                # Import module
                try:
                    module = importlib.import_module(full_module_path)
                except ModuleNotFoundError:
                    # Try without src prefix
                    try:
                        module = importlib.import_module(module_path)
                    except ModuleNotFoundError:
                         # Try relative import
                        try:
                            module = importlib.import_module(f"..{module_path}", package="src.registry")
                        except Exception as e:
                            self.logger.warning(f"Failed to import {agent_file.name}: {e}")
                            continue
                
                # Find agent classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Must inherit from BaseAgent but not be BaseAgent itself
                    if issubclass(obj, BaseAgent) and obj is not BaseAgent:
                        # Use snake_case name for registry key
                        # e.g., TrendAgent -> trend_agent
                        registry_name = self._class_name_to_registry_name(name)
                        
                        self.register_agent(registry_name, obj)
                        registered_count += 1
                        
            except Exception as e:
                self.logger.warning(f"Failed to process {agent_file.name}: {e}")
                continue
                
        self.logger.info(f"✅ Registered {registered_count} agents from {base_path}")
        return registered_count
        
    def _class_name_to_registry_name(self, class_name: str) -> str:
        """
        Convert class name to registry name (snake_case).
        
        Examples:
            TrendAgent -> trend_agent
            RiskGenius -> risk_genius
        """
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return name


# Global registry instance
_global_agent_registry = None

def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _global_agent_registry
    if _global_agent_registry is None:
        _global_agent_registry = AgentRegistry()
        # Auto-discover on first access
        try:
            _global_agent_registry.discover_and_register_all()
        except Exception as e:
            logger.error(f"Failed to auto-discover agents: {e}")
    return _global_agent_registry
