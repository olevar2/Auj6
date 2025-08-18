#!/usr/bin/env python3
"""
Message Types for AUJ Platform
==============================

Defines message schemas and types for the RabbitMQ messaging system.
All messages inherit from BaseMessage and include validation, serialization,
and routing information.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageType(Enum):
    """Message type enumeration."""
    TRADING_SIGNAL = "trading_signal"
    AGENT_COORDINATION = "agent_coordination"
    SYSTEM_STATUS = "system_status"
    MARKET_DATA = "market_data"
    RISK_MANAGEMENT = "risk_management"
    PERFORMANCE_UPDATE = "performance_update"
    ERROR_NOTIFICATION = "error_notification"


@dataclass
class BaseMessage:
    """Base message class for all platform messages."""
    
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.SYSTEM_STATUS
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    priority: MessagePriority = MessagePriority.NORMAL
    routing_key: str = ""
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expiration: Optional[int] = None  # TTL in seconds
    headers: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Serialize message to JSON."""
        data = asdict(self)
        # Convert datetime to ISO format
        data['timestamp'] = self.timestamp.isoformat()
        # Convert enums to values
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseMessage':
        """Deserialize message from JSON."""
        data = json.loads(json_str)
        # Convert timestamp back
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        # Convert enums back
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate message structure and content."""
        if not self.message_id or not self.source:
            return False
        if not isinstance(self.payload, dict):
            return False
        return True
    
    def set_priority(self, priority: MessagePriority):
        """Set message priority."""
        self.priority = priority
        self.headers['priority'] = priority.value
    
    def set_expiration(self, seconds: int):
        """Set message expiration time."""
        self.expiration = seconds
        self.headers['expiration'] = str(seconds * 1000)  # RabbitMQ expects milliseconds


@dataclass
class TradingSignalMessage(BaseMessage):
    """Trading signal message for buy/sell recommendations."""
    
    def __post_init__(self):
        self.message_type = MessageType.TRADING_SIGNAL
        self.routing_key = f"trading.signal.{self.payload.get('symbol', 'unknown')}"
    
    @classmethod
    def create_signal(
        cls,
        agent_name: str,
        symbol: str,
        signal_type: str,  # 'buy', 'sell', 'hold'
        confidence: float,
        price: float,
        volume: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        indicators: Optional[Dict[str, Any]] = None
    ) -> 'TradingSignalMessage':
        """Create a trading signal message."""
        
        payload = {
            'agent_name': agent_name,
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'price': price,
            'volume': volume,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'indicators': indicators or {}
        }
        
        message = cls(
            source=agent_name,
            payload=payload,
            priority=MessagePriority.HIGH if confidence > 0.8 else MessagePriority.NORMAL
        )
        
        # Set expiration based on signal type
        if signal_type in ['buy', 'sell']:
            message.set_expiration(300)  # 5 minutes for trade signals
        else:
            message.set_expiration(600)  # 10 minutes for holds
        
        return message


@dataclass
class AgentCoordinationMessage(BaseMessage):
    """Agent coordination message for multi-agent communication."""
    
    def __post_init__(self):
        self.message_type = MessageType.AGENT_COORDINATION
        self.routing_key = f"coordination.{self.payload.get('coordination_type', 'general')}"
    
    @classmethod
    def create_coordination(
        cls,
        sender_agent: str,
        coordination_type: str,  # 'consensus', 'conflict_resolution', 'resource_allocation'
        target_agents: List[str],
        coordination_data: Dict[str, Any],
        requires_response: bool = False
    ) -> 'AgentCoordinationMessage':
        """Create an agent coordination message."""
        
        payload = {
            'sender_agent': sender_agent,
            'coordination_type': coordination_type,
            'target_agents': target_agents,
            'coordination_data': coordination_data,
            'requires_response': requires_response
        }
        
        message = cls(
            source=sender_agent,
            payload=payload,
            priority=MessagePriority.HIGH if coordination_type == 'conflict_resolution' else MessagePriority.NORMAL
        )
        
        if requires_response:
            message.reply_to = f"coordination.response.{sender_agent}"
        
        return message


@dataclass
class SystemStatusMessage(BaseMessage):
    """System status update message."""
    
    def __post_init__(self):
        self.message_type = MessageType.SYSTEM_STATUS
        self.routing_key = f"system.status.{self.payload.get('component', 'general')}"
    
    @classmethod
    def create_status(
        cls,
        component: str,
        status: str,  # 'online', 'offline', 'warning', 'error'
        details: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> 'SystemStatusMessage':
        """Create a system status message."""
        
        payload = {
            'component': component,
            'status': status,
            'details': details,
            'metrics': metrics or {}
        }
        
        priority = MessagePriority.CRITICAL if status == 'error' else (
            MessagePriority.HIGH if status == 'warning' else MessagePriority.NORMAL
        )
        
        return cls(
            source=component,
            payload=payload,
            priority=priority
        )


@dataclass
class MarketDataMessage(BaseMessage):
    """Market data update message."""
    
    def __post_init__(self):
        self.message_type = MessageType.MARKET_DATA
        self.routing_key = f"market.data.{self.payload.get('symbol', 'unknown')}"
    
    @classmethod
    def create_market_data(
        cls,
        symbol: str,
        data_type: str,  # 'price', 'volume', 'orderbook', 'news'
        data: Dict[str, Any],
        provider: str
    ) -> 'MarketDataMessage':
        """Create a market data message."""
        
        payload = {
            'symbol': symbol,
            'data_type': data_type,
            'data': data,
            'provider': provider
        }
        
        message = cls(
            source=provider,
            payload=payload,
            priority=MessagePriority.HIGH
        )
        
        # Market data has short TTL
        message.set_expiration(60)  # 1 minute
        
        return message


@dataclass
class RiskManagementMessage(BaseMessage):
    """Risk management message for position and portfolio risk."""
    
    def __post_init__(self):
        self.message_type = MessageType.RISK_MANAGEMENT
        self.routing_key = f"risk.{self.payload.get('risk_type', 'general')}"
    
    @classmethod
    def create_risk_alert(
        cls,
        risk_type: str,  # 'position_size', 'drawdown', 'exposure', 'correlation'
        severity: str,   # 'low', 'medium', 'high', 'critical'
        affected_symbols: List[str],
        risk_data: Dict[str, Any],
        recommended_actions: List[str]
    ) -> 'RiskManagementMessage':
        """Create a risk management message."""
        
        payload = {
            'risk_type': risk_type,
            'severity': severity,
            'affected_symbols': affected_symbols,
            'risk_data': risk_data,
            'recommended_actions': recommended_actions
        }
        
        priority_map = {
            'low': MessagePriority.NORMAL,
            'medium': MessagePriority.HIGH,
            'high': MessagePriority.HIGH,
            'critical': MessagePriority.CRITICAL
        }
        
        return cls(
            source='risk_manager',
            payload=payload,
            priority=priority_map.get(severity, MessagePriority.NORMAL)
        )


@dataclass
class PerformanceUpdateMessage(BaseMessage):
    """Performance update message for agent and system metrics."""
    
    def __post_init__(self):
        self.message_type = MessageType.PERFORMANCE_UPDATE
        self.routing_key = f"performance.{self.payload.get('component_type', 'general')}"
    
    @classmethod
    def create_performance_update(
        cls,
        component_name: str,
        component_type: str,  # 'agent', 'system', 'strategy'
        performance_metrics: Dict[str, float],
        time_period: str = "1h"
    ) -> 'PerformanceUpdateMessage':
        """Create a performance update message."""
        
        payload = {
            'component_name': component_name,
            'component_type': component_type,
            'performance_metrics': performance_metrics,
            'time_period': time_period
        }
        
        return cls(
            source=component_name,
            payload=payload,
            priority=MessagePriority.LOW
        )


@dataclass
class ErrorNotificationMessage(BaseMessage):
    """Error notification message for system errors and exceptions."""
    
    def __post_init__(self):
        self.message_type = MessageType.ERROR_NOTIFICATION
        self.routing_key = f"error.{self.payload.get('error_type', 'general')}"
    
    @classmethod
    def create_error_notification(
        cls,
        component: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        recovery_actions: Optional[List[str]] = None
    ) -> 'ErrorNotificationMessage':
        """Create an error notification message."""
        
        payload = {
            'component': component,
            'error_type': error_type,
            'error_message': error_message,
            'stack_trace': stack_trace,
            'recovery_actions': recovery_actions or []
        }
        
        return cls(
            source=component,
            payload=payload,
            priority=MessagePriority.CRITICAL
        )


def create_message_from_type(message_type: MessageType, **kwargs) -> BaseMessage:
    """Factory function to create messages based on type."""
    
    message_classes = {
        MessageType.TRADING_SIGNAL: TradingSignalMessage,
        MessageType.AGENT_COORDINATION: AgentCoordinationMessage,
        MessageType.SYSTEM_STATUS: SystemStatusMessage,
        MessageType.MARKET_DATA: MarketDataMessage,
        MessageType.RISK_MANAGEMENT: RiskManagementMessage,
        MessageType.PERFORMANCE_UPDATE: PerformanceUpdateMessage,
        MessageType.ERROR_NOTIFICATION: ErrorNotificationMessage
    }
    
    message_class = message_classes.get(message_type, BaseMessage)
    return message_class(**kwargs)