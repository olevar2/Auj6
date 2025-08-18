"""
Messaging System for AUJ Platform
=================================

This module provides comprehensive message queue capabilities using RabbitMQ
for asynchronous communication between agents, coordination, and system components.

Refactored to use dependency injection pattern instead of global functions.

Components:
- MessageBroker: Core RabbitMQ connection and management
- MessagePublisher: Publishing messages with routing and reliability
- MessageConsumer: Consuming messages with acknowledgment and retry logic
- MessageRouter: Intelligent message routing based on content and priority
- RetryHandler: Advanced retry logic with exponential backoff
- DeadLetterHandler: Managing failed messages and error recovery
- MessagingService: Main service class using dependency injection
- MessagingServiceFactory: Factory for creating messaging service instances

Mission: Enable efficient async communication to maximize trading performance
for generating profits to help sick children and families in need.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 2.0.0 - Dependency Injection Refactor
"""

from .message_broker import MessageBroker
from .message_publisher import MessagePublisher
from .message_consumer import MessageConsumer
from .message_router import MessageRouter
from .retry_handler import RetryHandler
from .dead_letter_handler import DeadLetterHandler
from .messaging_service import MessagingService, MessagingServiceFactory
from .message_types import (
    BaseMessage,
    TradingSignalMessage,
    AgentCoordinationMessage,
    SystemStatusMessage,
    MarketDataMessage,
    RiskManagementMessage
)

# Deprecated global integration - use MessagingService instead
from .integration import MessagingIntegration

__all__ = [
    # Core components
    'MessageBroker',
    'MessagePublisher',
    'MessageConsumer',
    'MessageRouter',
    'RetryHandler',
    'DeadLetterHandler',
    
    # New dependency injection pattern (recommended)
    'MessagingService',
    'MessagingServiceFactory',
    
    # Message types
    'BaseMessage',
    'TradingSignalMessage',
    'AgentCoordinationMessage',
    'SystemStatusMessage',
    'MarketDataMessage',
    'RiskManagementMessage',
    
    # Deprecated (backward compatibility only)
    'MessagingIntegration'
]

__version__ = '2.0.0'