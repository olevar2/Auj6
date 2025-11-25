"""
Event Bus for the AUJ Platform.

This module implements a simple, asynchronous event bus for decoupling components.
It allows agents and services to publish and subscribe to events without direct dependencies.
"""

import asyncio
from typing import Dict, List, Callable, Any, Awaitable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import uuid

from .logging_setup import get_logger

logger = get_logger(__name__)

class EventType(str, Enum):
    """Types of events in the system."""
    MARKET_DATA_UPDATE = "MARKET_DATA_UPDATE"
    NEWS_UPDATE = "NEWS_UPDATE"
    ANALYSIS_REQUEST = "ANALYSIS_REQUEST"
    ANALYSIS_COMPLETED = "ANALYSIS_COMPLETED"
    TRADE_SIGNAL = "TRADE_SIGNAL"
    RISK_ALERT = "RISK_ALERT"
    ORDER_EXECUTION = "ORDER_EXECUTION"
    SYSTEM_STATUS = "SYSTEM_STATUS"

@dataclass
class Event:
    """Event data container."""
    type: EventType
    payload: Any
    source: str
    id: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()

class EventBus:
    """
    Asynchronous Event Bus.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._subscribers: Dict[EventType, List[Callable[[Event], Awaitable[None]]]] = {}
        self._initialized = True
        logger.info("EventBus initialized")

    def subscribe(self, event_type: EventType, handler: Callable[[Event], Awaitable[None]]):
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to {event_type}: {handler.__name__}")

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], Awaitable[None]]):
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed from {event_type}: {handler.__name__}")

    async def publish(self, event: Event):
        """Publish an event to all subscribers."""
        if event.type not in self._subscribers:
            return

        handlers = self._subscribers[event.type]
        if not handlers:
            return

        # Execute all handlers concurrently
        tasks = [handler(event) for handler in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.debug(f"Published event {event.type} from {event.source} to {len(handlers)} handlers")

# Global instance
event_bus = EventBus()
