#!/usr/bin/env python3
"""
Message Router for AUJ Platform
===============================

Intelligent message routing based on content, priority, and system state.
Provides dynamic routing rules, load balancing, and traffic shaping.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import random

from .message_broker import MessageBroker
from .message_types import BaseMessage, MessageType, MessagePriority


class RoutingStrategy(Enum):
    """Message routing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    PRIORITY_BASED = "priority_based"
    CONTENT_BASED = "content_based"
    LOAD_BALANCED = "load_balanced"


class RouteCondition(Enum):
    """Route condition types."""
    MESSAGE_TYPE = "message_type"
    SOURCE = "source"
    PRIORITY = "priority"
    CONTENT = "content"
    HEADER = "header"
    TIME_WINDOW = "time_window"
    SYSTEM_LOAD = "system_load"


@dataclass
class RoutingRule:
    """Rule for message routing."""
    name: str
    condition_type: RouteCondition
    condition_value: Any
    target_exchange: str
    target_routing_key: str
    priority: int = 0
    enabled: bool = True
    
    # Advanced options
    weight: float = 1.0
    max_messages_per_minute: Optional[int] = None
    load_threshold: Optional[float] = None
    time_restrictions: Optional[Dict[str, Any]] = None


@dataclass
class RouteTarget:
    """Target destination for routing."""
    exchange: str
    routing_key: str
    weight: float = 1.0
    max_capacity: Optional[int] = None
    current_load: int = 0
    enabled: bool = True


@dataclass
class RoutingMetrics:
    """Metrics for routing performance."""
    total_routed: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    rule_matches: Dict[str, int] = field(default_factory=dict)
    target_usage: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate routing success rate."""
        if self.total_routed == 0:
            return 0.0
        return self.successful_routes / self.total_routed


class MessageRouter:
    """
    Intelligent message router for AUJ platform.
    
    Routes messages based on configurable rules, system load,
    and content analysis with support for multiple routing strategies.
    """
    
    def __init__(self, message_broker: MessageBroker, config: Dict[str, Any]):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        """Initialize message router."""
        self.message_broker = message_broker
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Router configuration
        self.router_config = self.config_manager.get_dict('message_router', {})
        self.default_strategy = RoutingStrategy(
            self.router_config.get('default_strategy', 'priority_based')
        )
        self.enable_load_balancing = self.router_config.get('enable_load_balancing', True)
        self.enable_traffic_shaping = self.router_config.get('enable_traffic_shaping', True)
        
        # Routing rules and targets
        self.routing_rules: List[RoutingRule] = []
        self.route_targets: Dict[str, RouteTarget] = {}
        
        # State tracking
        self.metrics = RoutingMetrics()
        self.traffic_counters: Dict[str, Dict[str, int]] = {}  # rule_name -> time_bucket -> count
        self.round_robin_indexes: Dict[str, int] = {}
        
        # Load monitoring
        self.system_load = 0.0
        self.queue_loads: Dict[str, int] = {}
        
        # Setup default routing rules
        self._setup_default_routing_rules()
        
        self.logger.info("🎯 MessageRouter initialized")
    
    def _setup_default_routing_rules(self):
        """Setup default routing rules for the platform."""
        default_rules = [
            # High priority trading signals to dedicated queue
            RoutingRule(
                name="high_priority_trading_signals",
                condition_type=RouteCondition.PRIORITY,
                condition_value=MessagePriority.HIGH.value,
                target_exchange="auj.platform",
                target_routing_key="trading.signal.{symbol}.high",
                priority=100
            ),
            
            # Critical messages bypass normal routing
            RoutingRule(
                name="critical_messages",
                condition_type=RouteCondition.PRIORITY,
                condition_value=MessagePriority.CRITICAL.value,
                target_exchange="auj.platform",
                target_routing_key="critical.{message_type}",
                priority=200
            ),
            
            # Agent coordination to dedicated exchange
            RoutingRule(
                name="agent_coordination",
                condition_type=RouteCondition.MESSAGE_TYPE,
                condition_value=MessageType.AGENT_COORDINATION.value,
                target_exchange="auj.coordination",
                target_routing_key="coordination.{coordination_type}",
                priority=80
            ),
            
            # Risk management alerts to high priority
            RoutingRule(
                name="risk_alerts",
                condition_type=RouteCondition.MESSAGE_TYPE,
                condition_value=MessageType.RISK_MANAGEMENT.value,
                target_exchange="auj.platform",
                target_routing_key="risk.alert.{severity}",
                priority=90
            ),
            
            # System status to broadcast
            RoutingRule(
                name="system_status_broadcast",
                condition_type=RouteCondition.MESSAGE_TYPE,
                condition_value=MessageType.SYSTEM_STATUS.value,
                target_exchange="auj.broadcast",
                target_routing_key="",  # Fanout
                priority=50
            ),
            
            # Market data time-sensitive routing
            RoutingRule(
                name="realtime_market_data",
                condition_type=RouteCondition.MESSAGE_TYPE,
                condition_value=MessageType.MARKET_DATA.value,
                target_exchange="auj.platform",
                target_routing_key="market.data.{symbol}",
                priority=70,
                max_messages_per_minute=1000  # Rate limit
            )
        ]
        
        self.routing_rules.extend(default_rules)
        
        # Sort rules by priority (higher priority first)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        
        self.logger.info(f"📋 Loaded {len(default_rules)} default routing rules")
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add a new routing rule."""
        self.routing_rules.append(rule)
        
        # Re-sort by priority
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        
        self.logger.info(f"📋 Added routing rule: {rule.name}")
    
    def remove_routing_rule(self, rule_name: str) -> bool:
        """Remove a routing rule by name."""
        for i, rule in enumerate(self.routing_rules):
            if rule.name == rule_name:
                del self.routing_rules[i]
                self.logger.info(f"📋 Removed routing rule: {rule_name}")
                return True
        
        return False
    
    def add_route_target(self, name: str, target: RouteTarget):
        """Add a route target."""
        self.route_targets[name] = target
        self.logger.info(f"🎯 Added route target: {name}")
    
    async def route_message(
        self,
        message: BaseMessage,
        override_strategy: Optional[RoutingStrategy] = None
    ) -> bool:
        """
        Route a message based on configured rules and strategy.
        
        Args:
            message: Message to route
            override_strategy: Override default routing strategy
            
        Returns:
            bool: True if message was routed successfully
        """
        try:
            self.metrics.total_routed += 1
            
            # Find matching routing rule
            matching_rule = await self._find_matching_rule(message)
            
            if not matching_rule:
                # Use default routing
                exchange, routing_key = self._get_default_routing(message)
            else:
                # Apply rule-based routing
                exchange, routing_key = await self._apply_routing_rule(message, matching_rule)
                
                # Update rule metrics
                if matching_rule.name not in self.metrics.rule_matches:
                    self.metrics.rule_matches[matching_rule.name] = 0
                self.metrics.rule_matches[matching_rule.name] += 1
            
            # Apply routing strategy
            strategy = override_strategy or self.default_strategy
            final_exchange, final_routing_key = await self._apply_routing_strategy(
                message, exchange, routing_key, strategy
            )
            
            # Check traffic shaping
            if self.enable_traffic_shaping and matching_rule:
                if not await self._check_traffic_limits(matching_rule):
                    self.logger.warning(f"⚠️ Traffic limit exceeded for rule: {matching_rule.name}")
                    return False
            
            # Route the message
            success = await self._send_message(message, final_exchange, final_routing_key)
            
            if success:
                self.metrics.successful_routes += 1
                
                # Update target usage
                target_key = f"{final_exchange}/{final_routing_key}"
                if target_key not in self.metrics.target_usage:
                    self.metrics.target_usage[target_key] = 0
                self.metrics.target_usage[target_key] += 1
                
                self.logger.debug(f"🎯 Routed message {message.message_id} to "
                                f"{final_exchange}/{final_routing_key}")
            else:
                self.metrics.failed_routes += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error routing message {message.message_id}: {e}")
            self.metrics.failed_routes += 1
            return False
    
    async def _find_matching_rule(self, message: BaseMessage) -> Optional[RoutingRule]:
        """Find the first matching routing rule for a message."""
        for rule in self.routing_rules:
            if not rule.enabled:
                continue
            
            if await self._evaluate_rule_condition(message, rule):
                return rule
        
        return None
    
    async def _evaluate_rule_condition(self, message: BaseMessage, rule: RoutingRule) -> bool:
        """Evaluate if a message matches a routing rule condition."""
        try:
            if rule.condition_type == RouteCondition.MESSAGE_TYPE:
                return message.message_type.value == rule.condition_value
            
            elif rule.condition_type == RouteCondition.SOURCE:
                return message.source == rule.condition_value
            
            elif rule.condition_type == RouteCondition.PRIORITY:
                return message.priority.value >= rule.condition_value
            
            elif rule.condition_type == RouteCondition.CONTENT:
                # Content-based matching (basic implementation)
                content_str = json.dumps(message.payload).lower()
                return rule.condition_value.lower() in content_str
            
            elif rule.condition_type == RouteCondition.HEADER:
                header_key, expected_value = rule.condition_value
                return message.headers.get(header_key) == expected_value
            
            elif rule.condition_type == RouteCondition.TIME_WINDOW:
                # Time window matching
                if rule.time_restrictions:
                    return self._check_time_window(rule.time_restrictions)
            
            elif rule.condition_type == RouteCondition.SYSTEM_LOAD:
                return self.system_load <= rule.condition_value
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating rule condition: {e}")
            return False
    
    def _check_time_window(self, time_restrictions: Dict[str, Any]) -> bool:
        """Check if current time is within allowed window."""
        now = datetime.now()
        
        # Check day of week
        if 'days_of_week' in time_restrictions:
            allowed_days = time_restrictions['days_of_week']
            if now.weekday() not in allowed_days:
                return False
        
        # Check hour range
        if 'hour_range' in time_restrictions:
            start_hour, end_hour = time_restrictions['hour_range']
            if not (start_hour <= now.hour <= end_hour):
                return False
        
        return True
    
    async def _apply_routing_rule(self, message: BaseMessage, rule: RoutingRule) -> tuple:
        """Apply a routing rule to determine exchange and routing key."""
        exchange = rule.target_exchange
        routing_key = rule.target_routing_key
        
        # Template variable substitution
        routing_key = self._substitute_template_variables(routing_key, message)
        
        return exchange, routing_key
    
    def _substitute_template_variables(self, template: str, message: BaseMessage) -> str:
        """Substitute template variables in routing key."""
        # Extract variables from message
        variables = {
            'message_type': message.message_type.value,
            'source': message.source,
            'priority': message.priority.value,
            **message.payload  # Include payload fields
        }
        
        # Simple template substitution
        result = template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        
        return result
    
    def _get_default_routing(self, message: BaseMessage) -> tuple:
        """Get default routing for message type."""
        type_routing = {
            MessageType.TRADING_SIGNAL: ('auj.platform', f'trading.signal.{message.payload.get("symbol", "unknown")}'),
            MessageType.AGENT_COORDINATION: ('auj.coordination', 'coordination.general'),
            MessageType.SYSTEM_STATUS: ('auj.broadcast', ''),
            MessageType.MARKET_DATA: ('auj.platform', f'market.data.{message.payload.get("symbol", "unknown")}'),
            MessageType.RISK_MANAGEMENT: ('auj.platform', 'risk.general'),
            MessageType.PERFORMANCE_UPDATE: ('auj.platform', 'performance.general'),
            MessageType.ERROR_NOTIFICATION: ('auj.platform', 'error.general')
        }
        
        return type_routing.get(message.message_type, ('auj.platform', 'general'))
    
    async def _apply_routing_strategy(
        self,
        message: BaseMessage,
        exchange: str,
        routing_key: str,
        strategy: RoutingStrategy
    ) -> tuple:
        """Apply routing strategy to determine final destination."""
        if strategy == RoutingStrategy.PRIORITY_BASED:
            return self._apply_priority_routing(message, exchange, routing_key)
        
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._apply_load_balanced_routing(exchange, routing_key)
        
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return self._apply_round_robin_routing(exchange, routing_key)
        
        elif strategy == RoutingStrategy.WEIGHTED:
            return self._apply_weighted_routing(exchange, routing_key)
        
        elif strategy == RoutingStrategy.CONTENT_BASED:
            return self._apply_content_based_routing(message, exchange, routing_key)
        
        else:
            return exchange, routing_key
    
    def _apply_priority_routing(self, message: BaseMessage, exchange: str, routing_key: str) -> tuple:
        """Apply priority-based routing modifications."""
        if message.priority == MessagePriority.CRITICAL:
            routing_key = f"critical.{routing_key}"
        elif message.priority == MessagePriority.HIGH:
            routing_key = f"high.{routing_key}"
        
        return exchange, routing_key
    
    async def _apply_load_balanced_routing(self, exchange: str, routing_key: str) -> tuple:
        """Apply load-balanced routing."""
        if not self.enable_load_balancing:
            return exchange, routing_key
        
        # Find available targets for this exchange
        available_targets = []
        for name, target in self.route_targets.items():
            if target.exchange == exchange and target.enabled:
                # Check capacity
                if target.max_capacity is None or target.current_load < target.max_capacity:
                    available_targets.append((name, target))
        
        if not available_targets:
            return exchange, routing_key
        
        # Select target with lowest load
        best_target = min(available_targets, key=lambda x: x[1].current_load)
        target = best_target[1]
        
        # Update load
        target.current_load += 1
        
        return target.exchange, target.routing_key
    
    def _apply_round_robin_routing(self, exchange: str, routing_key: str) -> tuple:
        """Apply round-robin routing strategy."""
        available_targets = [
            target for target in self.route_targets.values()
            if target.exchange == exchange and target.enabled
        ]
        
        if not available_targets:
            return exchange, routing_key
        
        # Get or initialize round-robin index
        key = f"{exchange}_{routing_key}"
        if key not in self.round_robin_indexes:
            self.round_robin_indexes[key] = 0
        
        # Select next target
        index = self.round_robin_indexes[key] % len(available_targets)
        target = available_targets[index]
        
        # Update index
        self.round_robin_indexes[key] = (index + 1) % len(available_targets)
        
        return target.exchange, target.routing_key
    
    def _apply_weighted_routing(self, exchange: str, routing_key: str) -> tuple:
        """Apply weighted routing strategy."""
        available_targets = [
            target for target in self.route_targets.values()
            if target.exchange == exchange and target.enabled
        ]
        
        if not available_targets:
            return exchange, routing_key
        
        # Calculate weighted selection
        total_weight = sum(target.weight for target in available_targets)
        
        if total_weight == 0:
            return exchange, routing_key
        
        # Random weighted selection
        r = random.uniform(0, total_weight)
        current = 0
        
        for target in available_targets:
            current += target.weight
            if r <= current:
                return target.exchange, target.routing_key
        
        # Fallback to first target
        return available_targets[0].exchange, available_targets[0].routing_key
    
    def _apply_content_based_routing(self, message: BaseMessage, exchange: str, routing_key: str) -> tuple:
        """Apply content-based routing modifications."""
        # Example: Route based on message content
        payload = message.payload
        
        # Trading signals based on symbol
        if message.message_type == MessageType.TRADING_SIGNAL:
            symbol = payload.get('symbol', 'unknown')
            routing_key = f"trading.signal.{symbol.lower()}"
        
        # Risk alerts based on severity
        elif message.message_type == MessageType.RISK_MANAGEMENT:
            severity = payload.get('severity', 'medium')
            routing_key = f"risk.alert.{severity}"
        
        return exchange, routing_key
    
    async def _check_traffic_limits(self, rule: RoutingRule) -> bool:
        """Check if traffic limits are exceeded for a rule."""
        if not rule.max_messages_per_minute:
            return True
        
        now = datetime.now()
        minute_bucket = now.replace(second=0, microsecond=0)
        
        # Initialize tracking
        if rule.name not in self.traffic_counters:
            self.traffic_counters[rule.name] = {}
        
        rule_counters = self.traffic_counters[rule.name]
        
        # Clean old buckets (keep only last 5 minutes)
        cutoff = minute_bucket - timedelta(minutes=5)
        to_remove = [bucket for bucket in rule_counters.keys() if bucket < cutoff]
        for bucket in to_remove:
            del rule_counters[bucket]
        
        # Check current minute
        current_count = rule_counters.get(minute_bucket, 0)
        
        if current_count >= rule.max_messages_per_minute:
            return False
        
        # Increment counter
        rule_counters[minute_bucket] = current_count + 1
        return True
    
    async def _send_message(self, message: BaseMessage, exchange: str, routing_key: str) -> bool:
        """Send message to the determined destination."""
        try:
            await self.message_broker.publish_message(
                message_body=message.to_json(),
                exchange_name=exchange,
                routing_key=routing_key,
                priority=message.priority.value,
                expiration=message.expiration,
                headers=message.headers
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send message to {exchange}/{routing_key}: {e}")
            return False
    
    async def update_system_load(self, load: float):
        """Update system load metric."""
        self.system_load = load
    
    async def update_queue_load(self, queue_name: str, load: int):
        """Update queue load metric."""
        self.queue_loads[queue_name] = load
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            'metrics': {
                'total_routed': self.metrics.total_routed,
                'successful_routes': self.metrics.successful_routes,
                'failed_routes': self.metrics.failed_routes,
                'success_rate': self.metrics.success_rate
            },
            'rule_matches': self.metrics.rule_matches,
            'target_usage': self.metrics.target_usage,
            'active_rules': len([r for r in self.routing_rules if r.enabled]),
            'total_rules': len(self.routing_rules),
            'route_targets': len(self.route_targets),
            'system_load': self.system_load,
            'default_strategy': self.default_strategy.value
        }
    
    def get_routing_rules(self) -> List[Dict[str, Any]]:
        """Get list of routing rules."""
        return [
            {
                'name': rule.name,
                'condition_type': rule.condition_type.value,
                'condition_value': rule.condition_value,
                'target_exchange': rule.target_exchange,
                'target_routing_key': rule.target_routing_key,
                'priority': rule.priority,
                'enabled': rule.enabled,
                'weight': rule.weight,
                'max_messages_per_minute': rule.max_messages_per_minute
            }
            for rule in self.routing_rules
        ]
    
    async def health_check(self) -> bool:
        """Perform health check of message router."""
        try:
            # Check if basic routing works
            test_message = BaseMessage(
                source='health_check',
                payload={'test': True}
            )
            
            # Find routing destination (don't actually send)
            matching_rule = await self._find_matching_rule(test_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Message router health check failed: {e}")
            return False