"""
Agent Coordinator - Main coordination interface for AUJ Platform agents
"""

from typing import Dict, List, Any, Optional
from .genius_agent_coordinator import GeniusAgentCoordinator

class AgentCoordinator(GeniusAgentCoordinator):
    """
    Main agent coordinator that inherits from GeniusAgentCoordinator.
    This provides backward compatibility and a standard interface.
    """
    
    def __init__(self):
        super().__init__()
        
    async def initialize(self):
        """Initialize the agent coordinator"""
        await super().initialize()
        
    async def start(self):
        """Start the coordination system"""
        await super().start()
        
    async def stop(self):
        """Stop the coordination system"""
        await super().stop()