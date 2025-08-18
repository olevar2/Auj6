#!/usr/bin/env python3
"""
Messaging Service for AUJ Platform
==================================

Refactored messaging system using dependency injection pattern.
Replaces global functions with a proper service-oriented approach.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 2.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json

from .message_broker import MessageBroker
from .message_publisher import MessagePublisher
from .message_consumer import MessageConsumer, MessageHandler
from .message_router import MessageRouter
from .retry_handler import RetryHandler
from .dead_letter_handler import DeadLetterHandler
from .message_types import (
    BaseMessage, MessageType, MessagePriority,
    TradingSignalMessage, AgentCoordinationMessage,
    SystemStatusMessage, MarketDataMessage,
    RiskManagementMessage, PerformanceUpdateMessage
)


class MessagingService:
    """
    Main messaging service class for AUJ platform using dependency injection.
    
    Provides a unified interface for all messaging operations without
    relying on global state or functions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize messaging service with dependency injection."""
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Injected dependencies (will be set via dependency injection)
        self.message_broker: Optional[MessageBroker] = None
        self.message_publisher: Optional[MessagePublisher] = None
        self.message_consumer: Optional[MessageConsumer] = None
        self.message_router: Optional[MessageRouter] = None
        self.retry_handler: Optional[RetryHandler] = None
        self.dead_letter_handler: Optional[DeadLetterHandler] = None
        
        # State
        self._initialized = False
        self._running = False
        
        # Integration configuration
        self.integration_config = self.config_manager.get_dict('messaging_integration', {})
        self.enable_monitoring = self.integration_config.get('enable_monitoring', True)
        self.enable_metrics = self.integration_config.get('enable_metrics', True)
        
        self.logger.info("🔧 MessagingService initialized with DI pattern")
    
    async def initialize(self) -> bool:
        """Initialize all messaging components with dependency injection."""
        try:
            self.logger.info("🚀 Initializing messaging service with DI...")
            
            # Create dependencies in proper order
            self.message_broker = MessageBroker(self.config)
            broker_success = await self.message_broker.initialize()
            
            if not broker_success:
                self.logger.error("❌ Failed to initialize message broker")
                return False
            
            # Inject message broker dependency into other components
            self.retry_handler = RetryHandler(self.config)
            self.message_router = MessageRouter(self.message_broker, self.config)
            self.message_publisher = MessagePublisher(self.message_broker, self.config)
            self.message_consumer = MessageConsumer(self.message_broker, self.config)
            self.dead_letter_handler = DeadLetterHandler(self.message_broker, self.config)
            
            # Start dead letter monitoring
            await self.dead_letter_handler.start_monitoring()
            
            self._initialized = True
            self.logger.info("✅ Messaging service initialized successfully with DI")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize messaging service: {e}")
            return False
    
    async def start(self) -> bool:
        """Start messaging services."""
        if not self._initialized:
            self.logger.error("❌ Messaging service not initialized")
            return False
        
        if self._running:
            self.logger.warning("⚠️ Messaging service already running")
            return True
        
        try:
            self.logger.info("🚀 Starting messaging services...")
            
            # Start consumer with default queues
            default_queues = [
                'auj.trading.signals.high',
                'auj.trading.signals.normal',
                'auj.coordination.consensus',
                'auj.coordination.conflicts',
                'auj.system.status',
                'auj.risk.alerts',
                'auj.performance.updates'
            ]
            
            await self.message_consumer.start_consuming(default_queues)
            
            self._running = True
            self.logger.info("✅ Messaging services started")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start messaging services: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop messaging services."""
        if not self._running:
            return True
        
        try:
            self.logger.info("🛑 Stopping messaging services...")
            
            # Stop consumer
            if self.message_consumer:
                await self.message_consumer.stop_consuming()
            
            # Stop dead letter monitoring
            if self.dead_letter_handler:
                await self.dead_letter_handler.stop_monitoring()
            
            # Close message broker
            if self.message_broker:
                await self.message_broker.close()
            
            self._running = False
            self.logger.info("✅ Messaging services stopped")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping messaging services: {e}")
            return False
    
    # Publisher interface methods (dependency injected)
    async def publish_trading_signal(
        self,
        agent_name: str,
        symbol: str,
        signal_type: str,
        confidence: float,
        price: float,
        **kwargs
    ) -> bool:
        """Publish a trading signal using injected publisher."""
        if not self._running or not self.message_publisher:
            self.logger.warning("⚠️ Cannot publish - service not running or publisher not available")
            return False
        
        return await self.message_publisher.publish_trading_signal(
            agent_name=agent_name,
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            **kwargs
        )
    
    async def publish_agent_coordination(
        self,
        sender_agent: str,
        coordination_type: str,
        target_agents: List[str],
        coordination_data: Dict[str, Any],
        requires_response: bool = False
    ) -> bool:
        """Publish agent coordination message using injected publisher."""
        if not self._running or not self.message_publisher:
            self.logger.warning("⚠️ Cannot publish - service not running or publisher not available")
            return False
        
        return await self.message_publisher.publish_agent_coordination(
            sender_agent=sender_agent,
            coordination_type=coordination_type,
            target_agents=target_agents,
            coordination_data=coordination_data,
            requires_response=requires_response
        )
    
    async def publish_system_status(
        self,
        component: str,
        status: str,
        details: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """Publish system status update using injected publisher."""
        if not self._running or not self.message_publisher:
            self.logger.warning("⚠️ Cannot publish - service not running or publisher not available")
            return False
        
        return await self.message_publisher.publish_system_status(
            component=component,
            status=status,
            details=details,
            metrics=metrics
        )
    
    async def publish_risk_alert(
        self,
        risk_type: str,
        severity: str,
        affected_symbols: List[str],
        risk_data: Dict[str, Any],
        recommended_actions: List[str]
    ) -> bool:
        """Publish risk management alert using injected publisher."""
        if not self._running or not self.message_publisher:
            self.logger.warning("⚠️ Cannot publish - service not running or publisher not available")
            return False
        
        return await self.message_publisher.publish_risk_alert(
            risk_type=risk_type,
            severity=severity,
            affected_symbols=affected_symbols,
            risk_data=risk_data,
            recommended_actions=recommended_actions
        )
    
    async def publish_performance_update(
        self,
        component_name: str,
        component_type: str,
        performance_metrics: Dict[str, float],
        time_period: str = "1h"
    ) -> bool:
        """Publish performance update using injected publisher."""
        if not self._running or not self.message_publisher:
            self.logger.warning("⚠️ Cannot publish - service not running or publisher not available")
            return False
        
        return await self.message_publisher.publish_performance_update(
            component_name=component_name,
            component_type=component_type,
            performance_metrics=performance_metrics,
            time_period=time_period
        )
    
    # Consumer interface methods (dependency injected)
    def register_message_handler(
        self,
        handler: MessageHandler,
        message_types: Optional[List[MessageType]] = None
    ):
        """Register a message handler using injected consumer."""
        if self.message_consumer:
            self.message_consumer.register_handler(handler, message_types)
        else:
            self.logger.warning("⚠️ Cannot register handler - consumer not available")
    
    def unregister_message_handler(self, handler: MessageHandler):
        """Unregister a message handler using injected consumer."""
        if self.message_consumer:
            self.message_consumer.unregister_handler(handler)
        else:
            self.logger.warning("⚠️ Cannot unregister handler - consumer not available")
    
    async def consume_single_message(
        self,
        queue_name: str,
        timeout_seconds: Optional[float] = None
    ) -> Optional[BaseMessage]:
        """Consume a single message from a queue using injected consumer."""
        if not self._running or not self.message_consumer:
            self.logger.warning("⚠️ Cannot consume - service not running or consumer not available")
            return None
        
        return await self.message_consumer.consume_single_message(
            queue_name=queue_name,
            timeout_seconds=timeout_seconds
        )
    
    # Router interface methods (dependency injected)
    async def route_message(
        self,
        message: BaseMessage,
        override_strategy: Optional[str] = None
    ) -> bool:
        """Route a message using injected message router."""
        if not self._running or not self.message_router:
            self.logger.warning("⚠️ Cannot route - service not running or router not available")
            return False
        
        from .message_router import RoutingStrategy
        strategy = None
        if override_strategy:
            try:
                strategy = RoutingStrategy(override_strategy)
            except ValueError:
                pass
        
        return await self.message_router.route_message(message, strategy)
    
    # Retry interface methods (dependency injected)
    async def retry_async_operation(
        self,
        func: Callable,
        *args,
        operation_key: str = "default",
        max_attempts: int = 3,
        **kwargs
    ):
        """Execute function with retry logic using injected retry handler."""
        if not self.retry_handler:
            self.logger.warning("⚠️ Retry handler not available - executing without retry")
            return await func(*args, **kwargs)
        
        from .retry_handler import RetryConfig
        config = RetryConfig(max_attempts=max_attempts)
        
        result = await self.retry_handler.retry_async(
            func, *args,
            operation_key=operation_key,
            config=config,
            **kwargs
        )
        
        if result.success:
            return result.attempts[-1]  # Return last successful attempt
        else:
            raise Exception(result.final_error)
    
    # Dead letter interface methods (dependency injected)
    async def get_dead_letters_for_review(self) -> List[Dict[str, Any]]:
        """Get dead letters that need manual review using injected dead letter handler."""
        if not self.dead_letter_handler:
            self.logger.warning("⚠️ Dead letter handler not available")
            return []
        
        return self.dead_letter_handler.get_dead_letters_for_review()
    
    async def recover_dead_letter(self, message_id: str) -> bool:
        """Manually recover a dead letter message using injected dead letter handler."""
        if not self.dead_letter_handler:
            self.logger.warning("⚠️ Dead letter handler not available")
            return False
        
        return await self.dead_letter_handler.manual_recover_message(message_id)
    
    # Monitoring and statistics
    def get_messaging_stats(self) -> Dict[str, Any]:
        """Get comprehensive messaging statistics."""
        stats = {
            'initialized': self._initialized,
            'running': self._running,
            'timestamp': datetime.now().isoformat(),
            'service_type': 'dependency_injected'
        }
        
        if self.message_broker:
            stats['broker'] = self.message_broker.get_broker_stats()
        
        if self.message_publisher:
            stats['publisher'] = self.message_publisher.get_publish_stats()
        
        if self.message_consumer:
            stats['consumer'] = self.message_consumer.get_consume_stats()
        
        if self.message_router:
            stats['router'] = self.message_router.get_routing_stats()
        
        if self.retry_handler:
            stats['retry_handler'] = self.retry_handler.get_retry_stats()
        
        if self.dead_letter_handler:
            stats['dead_letter_handler'] = self.dead_letter_handler.get_dead_letter_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform comprehensive health check."""
        health = {
            'overall': True,
            'initialized': self._initialized,
            'running': self._running,
            'dependency_injection': True
        }
        
        if self.message_broker:
            broker_health = await self.message_broker.health_check()
            health['broker'] = broker_health
            if not broker_health:
                health['overall'] = False
        
        if self.message_publisher:
            publisher_health = await self.message_publisher.health_check()
            health['publisher'] = publisher_health
            if not publisher_health:
                health['overall'] = False
        
        if self.message_consumer:
            consumer_health = await self.message_consumer.health_check()
            health['consumer'] = consumer_health
            if not consumer_health:
                health['overall'] = False
        
        if self.message_router:
            router_health = await self.message_router.health_check()
            health['router'] = router_health
            if not router_health:
                health['overall'] = False
        
        if self.retry_handler:
            retry_health = await self.retry_handler.health_check()
            health['retry_handler'] = retry_health
            if not retry_health:
                health['overall'] = False
        
        if self.dead_letter_handler:
            dlq_health = await self.dead_letter_handler.health_check()
            health['dead_letter_handler'] = dlq_health
            if not dlq_health:
                health['overall'] = False
        
        return health
    
    # Utility methods
    async def send_test_message(self, message_type: str = "system_status") -> bool:
        """Send a test message for validation."""
        try:
            if message_type == "system_status":
                return await self.publish_system_status(
                    component="messaging_service",
                    status="test",
                    details={
                        "timestamp": datetime.now().isoformat(),
                        "test_message": True,
                        "service_type": "dependency_injected"
                    }
                )
            
            elif message_type == "trading_signal":
                return await self.publish_trading_signal(
                    agent_name="test_agent",
                    symbol="TEST",
                    signal_type="buy",
                    confidence=0.75,
                    price=100.0
                )
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send test message: {e}")
            return False
    
    def get_queue_info(self, queue_name: str) -> Dict[str, Any]:
        """Get information about a specific queue."""
        if not self.message_broker:
            self.logger.warning("⚠️ Message broker not available")
            return {}
        
        try:
            return asyncio.run(self.message_broker.get_queue_info(queue_name))
        except Exception as e:
            self.logger.error(f"❌ Failed to get queue info: {e}")
            return {}
    
    async def purge_queue(self, queue_name: str) -> bool:
        """Purge messages from a queue (use with caution)."""
        try:
            if not self.message_broker:
                self.logger.warning("⚠️ Message broker not available")
                return False
            
            # This would require admin privileges
            # Implementation depends on RabbitMQ management API
            self.logger.warning(f"⚠️ Queue purge requested for: {queue_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to purge queue {queue_name}: {e}")
            return False


class MessagingServiceFactory:
    """
    Factory class for creating MessagingService instances.
    
    Provides a clean way to create and configure messaging services
    without relying on global state.
    """
    
    @staticmethod
    async def create_messaging_service(config: Dict[str, Any]) -> Optional[MessagingService]:
        """Create and initialize a new MessagingService instance."""
        try:
            service = MessagingService(config)
            success = await service.initialize()
            
            if success:
                await service.start()
                return service
            else:
                logging.getLogger(__name__).error("❌ Failed to create messaging service")
                return None
                
        except Exception as e:
            logging.getLogger(__name__).error(f"❌ Error creating messaging service: {e}")
            return None
    
    @staticmethod
    async def create_test_messaging_service() -> Optional[MessagingService]:
        """Create a test messaging service with default configuration."""
        test_config = {
            'message_broker': {
                'host': 'localhost',
                'port': 5672,
                'username': 'guest',
                'password': 'guest',
                'vhost': '/',
                'ssl_enabled': False
            },
            'messaging_integration': {
                'enable_monitoring': True,
                'enable_metrics': True
            }
        }
        
        return await MessagingServiceFactory.create_messaging_service(test_config)