#!/usr/bin/env python3
"""
Messaging Coordinator for AUJ Platform
======================================

Coordinates messaging integration with existing components, ensuring seamless
communication between agents, execution hand                 "timestamp": datetime.now(timezone.utc).isoformat(),
            }             "timestamp": datetime.now(timezone.utc).isoformat(),
            }r, and other platform components.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from ..core.logging_setup import get_logger
from ..messaging.message_broker import MessageBroker

logger = get_logger(__name__)


class MessagingCoordinator:
    """
    Coordinates messaging integration across all platform components.

    This class ensures that messaging features are properly integrated
    with existing components while maintaining backward compatibility.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the messaging coordinator."""
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.config = config or {}
        self.logger = logger

        # Messaging components
        self.message_broker: Optional[MessageBroker] = None

        # Component references
        self.agents: List[Any] = []
        self.execution_handler: Optional[Any] = None
        self.risk_manager: Optional[Any] = None
        self.hierarchy_manager: Optional[Any] = None
        self.error_handler: Optional[Any] = None

        # Configuration
        self.messaging_enabled = self._check_messaging_enabled()

        self.logger.info(f"🔧 MessagingCoordinator initialized (enabled: {self.messaging_enabled})")

    def _check_messaging_enabled(self) -> bool:
        """Check if messaging is enabled in configuration."""
        feature_flags = self.config_manager.get_dict('feature_flags', {})
        broker_config = self.config_manager.get_dict('message_broker', {})

        return (
            feature_flags.get('enable_messaging_system', False) and
            broker_config.get('enabled', False)
        )

    async def initialize(self) -> bool:
        """Initialize messaging coordination."""
        try:
            if not self.messaging_enabled:
                self.logger.info("📡 Messaging system disabled - skipping initialization")
                return True

            self.logger.info("📡 Initializing messaging coordination...")

            # Initialize message broker
            self.message_broker = MessageBroker(self.config)
            broker_success = await self.message_broker.initialize()

            if not broker_success:
                self.logger.error("❌ Failed to initialize message broker")
                return False

            self.logger.info("✅ Messaging coordination initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize messaging coordination: {e}")
            return False

    async def integrate_agent(self, agent):
        """Integrate an agent with the messaging system."""
        if not self.messaging_enabled or not self.message_broker:
            return

        try:
            # Set message broker for the agent
            if hasattr(agent, 'set_message_broker'):
                agent.set_message_broker(self.message_broker)
                self.agents.append(agent)
                self.logger.info(f"📡 Integrated agent {agent.name} with messaging")
            else:
                self.logger.warning(f"⚠️ Agent {agent.name} does not support messaging")

        except Exception as e:
            self.logger.error(f"❌ Failed to integrate agent {agent.name}: {e}")

    async def integrate_execution_handler(self, execution_handler):
        """Integrate execution handler with messaging system."""
        if not self.messaging_enabled or not self.message_broker:
            return

        try:
            if hasattr(execution_handler, 'set_message_broker'):
                execution_handler.set_message_broker(self.message_broker)
                self.execution_handler = execution_handler
                self.logger.info("📡 Integrated execution handler with messaging")
            else:
                self.logger.warning("⚠️ Execution handler does not support messaging")

        except Exception as e:
            self.logger.error(f"❌ Failed to integrate execution handler: {e}")

    async def integrate_risk_manager(self, risk_manager):
        """Integrate risk manager with messaging system."""
        if not self.messaging_enabled or not self.message_broker:
            return

        try:
            if hasattr(risk_manager, 'set_message_broker'):
                risk_manager.set_message_broker(self.message_broker)
                self.risk_manager = risk_manager
                self.logger.info("📡 Integrated risk manager with messaging")
            else:
                self.logger.warning("⚠️ Risk manager does not support messaging")

        except Exception as e:
            self.logger.error(f"❌ Failed to integrate risk manager: {e}")

    async def integrate_hierarchy_manager(self, hierarchy_manager):
        """Integrate hierarchy manager with messaging system."""
        if not self.messaging_enabled or not self.message_broker:
            return

        try:
            if hasattr(hierarchy_manager, 'set_message_broker'):
                hierarchy_manager.set_message_broker(self.message_broker)
                self.hierarchy_manager = hierarchy_manager
                self.logger.info("📡 Integrated hierarchy manager with messaging")
            else:
                self.logger.warning("⚠️ Hierarchy manager does not support messaging")

        except Exception as e:
            self.logger.error(f"❌ Failed to integrate hierarchy manager: {e}")

    async def publish_system_status(self, status_data: Dict[str, Any]):
        """Publish system status update."""
        if not self.messaging_enabled or not self.message_broker:
            return

        try:
            status_message = {
                "type": "system_status",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": status_data,
                "source": "messaging_coordinator"
            }

            await self.message_broker.publish_message(
                message_body=str(status_message),
                exchange_name="auj.platform",
                routing_key="system.status.coordinator",
                priority=5
            )

            self.logger.debug("📡 Published system status update")

        except Exception as e:
            self.logger.error(f"❌ Failed to publish system status: {e}")

    async def broadcast_system_message(self, message_type: str, message_data: Dict[str, Any]):
        """Broadcast a system-wide message."""
        if not self.messaging_enabled or not self.message_broker:
            return

        try:
            broadcast_message = {
                "type": message_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": message_data,
                "source": "messaging_coordinator"
            }

            await self.message_broker.publish_message(
                message_body=str(broadcast_message),
                exchange_name="auj.broadcast",
                routing_key="",  # Fanout exchange doesn't use routing key
                priority=8
            )

            self.logger.info(f"📡 Broadcasted system message: {message_type}")

        except Exception as e:
            self.logger.error(f"❌ Failed to broadcast system message: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of messaging coordination."""
        try:
            if not self.messaging_enabled:
                return {
                    "status": "disabled",
                    "messaging_enabled": False,
                    "message": "Messaging system is disabled"
                }

            # Check message broker health
            broker_healthy = False
            if self.message_broker:
                broker_healthy = await self.message_broker.health_check()

            # Count integrated components
            integrated_components = {
                "agents": len(self.agents),
                "execution_handler": self.execution_handler is not None,
                "risk_manager": self.risk_manager is not None,
                "hierarchy_manager": self.hierarchy_manager is not None
            }

            return {
                "status": "healthy" if broker_healthy else "unhealthy",
                "messaging_enabled": self.messaging_enabled,
                "broker_healthy": broker_healthy,
                "integrated_components": integrated_components,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def get_messaging_stats(self) -> Dict[str, Any]:
        """Get messaging system statistics."""
        try:
            if not self.messaging_enabled or not self.message_broker:
                return {"status": "disabled"}

            # Get broker stats
            broker_stats = self.message_broker.get_broker_stats()

            # Add coordinator stats
            coordinator_stats = {
                "integrated_agents": len(self.agents),
                "integrated_execution_handler": self.execution_handler is not None,
                "integrated_risk_manager": self.risk_manager is not None,
                "integrated_hierarchy_manager": self.hierarchy_manager is not None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            return {
                "status": "active",
                "broker_stats": broker_stats,
                "coordinator_stats": coordinator_stats
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get messaging stats: {e}")
            return {"status": "error", "error": str(e)}

    async def shutdown(self):
        """Shutdown messaging coordination."""
        try:
            self.logger.info("🔒 Shutting down messaging coordination...")

            if self.message_broker:
                await self.message_broker.close()
                self.logger.info("✅ Message broker closed")

            # Clear component references
            self.agents.clear()
            self.execution_handler = None
            self.risk_manager = None
            self.hierarchy_manager = None

            self.logger.info("✅ Messaging coordination shutdown complete")

        except Exception as e:
            self.logger.error(f"❌ Error during messaging coordination shutdown: {e}")

    def is_initialized(self) -> bool:
        """Check if messaging coordinator is initialized."""
        return self.messaging_enabled and self.message_broker is not None

    async def close(self):
        """Close messaging coordinator - alias for shutdown."""
        await self.shutdown()

    async def subscribe_to_messages(self, routing_key: str, handler):
        """Subscribe to messages with specific routing key."""
        if not self.messaging_enabled or not self.message_broker:
            self.logger.warning("Cannot subscribe - messaging not enabled")
            return

        try:
            # Set up consumer for the routing key
            await self.message_broker.setup_consumer("auj.coordination.consensus", handler)
            self.logger.info(f"✅ Subscribed to routing key: {routing_key}")
        except Exception as e:
            self.logger.error(f"❌ Failed to subscribe to {routing_key}: {e}")

    async def register_component(self, component_type: str, component):
        """Register a component for coordination."""
        if not self.messaging_enabled:
            self.logger.warning("Cannot register component - messaging not enabled")
            return

        try:
            # Store component reference
            if component_type == "agent":
                self.agents.append(component)
            elif component_type == "execution_handler":
                self.execution_handler = component
            elif component_type == "risk_manager":
                self.risk_manager = component
            elif component_type == "hierarchy_manager":
                self.hierarchy_manager = component

            self.logger.info(f"✅ Registered component: {component_type}")
        except Exception as e:
            self.logger.error(f"❌ Failed to register component {component_type}: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status including component information."""
        try:
            components = {}
            
            if self.agents:
                components["agent"] = {"count": len(self.agents), "status": "active"}
            if self.execution_handler:
                components["execution_handler"] = {"status": "active"}
            if self.risk_manager:
                components["risk_manager"] = {"status": "active"}
            if self.hierarchy_manager:
                components["hierarchy_manager"] = {"status": "active"}

            return {
                "messaging_enabled": self.messaging_enabled,
                "broker_connected": self.message_broker.is_connected() if self.message_broker else False,
                "components": components,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get system status: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

    async def check_component_health(self) -> Dict[str, Any]:
        """Check health of all registered components."""
        try:
            healthy_components = []
            unhealthy_components = []

            # Check agent health
            for i, agent in enumerate(self.agents):
                if hasattr(agent, 'is_healthy') and callable(agent.is_healthy):
                    if agent.is_healthy():
                        healthy_components.append(f"agent_{i}")
                    else:
                        unhealthy_components.append(f"agent_{i}")
                else:
                    healthy_components.append(f"agent_{i}")  # Assume healthy if no health check

            # Check other components
            for component_name, component in [
                ("execution_handler", self.execution_handler),
                ("risk_manager", self.risk_manager),
                ("hierarchy_manager", self.hierarchy_manager)
            ]:
                if component:
                    if hasattr(component, 'is_healthy') and callable(component.is_healthy):
                        if component.is_healthy():
                            healthy_components.append(component_name)
                        else:
                            unhealthy_components.append(component_name)
                    else:
                        healthy_components.append(component_name)  # Assume healthy if no health check

            return {
                "healthy_components": healthy_components,
                "unhealthy_components": unhealthy_components,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to check component health: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

    def set_error_handler(self, handler):
        """Set error handler for messaging errors."""
        self.error_handler = handler
        self.logger.info("✅ Error handler set")

    async def register_callback(self, event_type: str, callback):
        """Register a callback for specific event types."""
        # This is a placeholder for future event handling
        self.logger.info(f"✅ Registered callback for: {event_type}")
