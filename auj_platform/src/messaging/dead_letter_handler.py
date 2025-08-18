#!/usr/bin/env python3
"""
Dead Letter Handler for AUJ Platform
====================================

Manages failed messages, provides error analysis, and implements
recovery strategies for dead lettered messages.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from ..core.unified_config import get_unified_config

from .message_broker import MessageBroker
from .message_types import BaseMessage, MessageType


class FailureReason(Enum):
    """Reasons for message failure."""
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    DESERIALIZATION_ERROR = "deserialization_error"
    HANDLER_ERROR = "handler_error"
    TIMEOUT = "timeout"
    INVALID_MESSAGE = "invalid_message"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery actions for failed messages."""
    RETRY = "retry"
    REQUEUE = "requeue"
    DISCARD = "discard"
    MANUAL_REVIEW = "manual_review"
    ARCHIVE = "archive"


@dataclass
class DeadLetterMessage:
    """Information about a dead lettered message."""
    message_id: str
    original_queue: str
    failure_reason: FailureReason
    failure_time: datetime
    retry_count: int
    original_message: BaseMessage
    error_details: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    last_recovery_attempt: Optional[datetime] = None


@dataclass
class RecoveryRule:
    """Rule for automatic recovery of dead letter messages."""
    failure_reason: FailureReason
    action: RecoveryAction
    conditions: Dict[str, Any] = field(default_factory=dict)
    max_recovery_attempts: int = 3
    recovery_delay: float = 300.0  # 5 minutes


class DeadLetterHandler:
    """
    Dead letter queue handler for AUJ platform.
    
    Manages failed messages, analyzes failure patterns, and implements
    automatic and manual recovery strategies.
    """
    
    def __init__(self, message_broker: MessageBroker, config: Dict[str, Any]):
        """Initialize dead letter handler."""
        self.message_broker = message_broker
        self.config = config
        self.config_manager = get_unified_config()
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.dlq_config = self.config_manager.get_dict('dead_letter_handler', {})
        self.dead_letter_queue = self.dlq_config.get('queue_name', 'auj.deadletter.queue')
        self.max_storage_days = self.dlq_config.get('max_storage_days', 7)
        self.analysis_interval = self.dlq_config.get('analysis_interval', 3600)  # 1 hour
        
        # Dead letter storage
        self.dead_letters: Dict[str, DeadLetterMessage] = {}
        
        # Recovery rules
        self.recovery_rules: List[RecoveryRule] = []
        self._setup_default_recovery_rules()
        
        # Statistics
        self.stats = {
            'total_dead_letters': 0,
            'recovered_messages': 0,
            'discarded_messages': 0,
            'manual_reviews': 0,
            'failure_reasons': {reason.value: 0 for reason in FailureReason}
        }
        
        # Background tasks
        self._monitoring_task = None
        self._cleanup_task = None
        
        self.logger.info("💀 DeadLetterHandler initialized")
    
    def _setup_default_recovery_rules(self):
        """Setup default recovery rules."""
        default_rules = [
            # Retry deserialization errors after delay
            RecoveryRule(
                failure_reason=FailureReason.DESERIALIZATION_ERROR,
                action=RecoveryAction.DISCARD,  # Usually not recoverable
                max_recovery_attempts=0
            ),
            
            # Retry handler errors with exponential backoff
            RecoveryRule(
                failure_reason=FailureReason.HANDLER_ERROR,
                action=RecoveryAction.RETRY,
                conditions={'min_delay': 300, 'max_delay': 3600},
                max_recovery_attempts=3,
                recovery_delay=300.0
            ),
            
            # Timeout errors can be retried
            RecoveryRule(
                failure_reason=FailureReason.TIMEOUT,
                action=RecoveryAction.RETRY,
                max_recovery_attempts=2,
                recovery_delay=600.0
            ),
            
            # Circuit breaker errors should wait longer
            RecoveryRule(
                failure_reason=FailureReason.CIRCUIT_BREAKER_OPEN,
                action=RecoveryAction.RETRY,
                max_recovery_attempts=5,
                recovery_delay=1800.0  # 30 minutes
            ),
            
            # Invalid messages need manual review
            RecoveryRule(
                failure_reason=FailureReason.INVALID_MESSAGE,
                action=RecoveryAction.MANUAL_REVIEW,
                max_recovery_attempts=0
            ),
            
            # Max retries exceeded - manual review for critical, discard others
            RecoveryRule(
                failure_reason=FailureReason.MAX_RETRIES_EXCEEDED,
                action=RecoveryAction.MANUAL_REVIEW,
                conditions={'critical_only': True},
                max_recovery_attempts=1
            )
        ]
        
        self.recovery_rules.extend(default_rules)
        self.logger.info(f"📋 Loaded {len(default_rules)} default recovery rules")
    
    async def start_monitoring(self):
        """Start monitoring dead letter queue."""
        if self._monitoring_task:
            self.logger.warning("⚠️ Dead letter monitoring already running")
            return
        
        self.logger.info("🔍 Starting dead letter queue monitoring...")
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitor_dead_letters())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_old_messages())
        
        self.logger.info("✅ Dead letter monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring dead letter queue."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        self.logger.info("🛑 Dead letter monitoring stopped")
    
    async def _monitor_dead_letters(self):
        """Monitor dead letter queue for new messages."""
        try:
            while True:
                async with self.message_broker.get_channel() as channel:
                    queue = self.message_broker.queues.get(self.dead_letter_queue)
                    if not queue:
                        self.logger.error(f"❌ Dead letter queue not found: {self.dead_letter_queue}")
                        await asyncio.sleep(60)
                        continue
                    
                    # Process messages in the dead letter queue
                    async with queue.iterator() as queue_iter:
                        async for raw_message in queue_iter:
                            try:
                                await self._process_dead_letter_message(raw_message)
                                await raw_message.ack()
                                
                            except Exception as e:
                                self.logger.error(f"❌ Error processing dead letter: {e}")
                                await raw_message.nack(requeue=False)
                            
                            # Break if monitoring stopped
                            if not self._monitoring_task:
                                break
                
                # Check for recovery opportunities
                await self._check_recovery_opportunities()
                
                # Wait before next iteration
                await asyncio.sleep(30)
                
        except asyncio.CancelledError:
            self.logger.info("🔍 Dead letter monitoring cancelled")
        except Exception as e:
            self.logger.error(f"❌ Error in dead letter monitoring: {e}")
    
    async def _process_dead_letter_message(self, raw_message):
        """Process a message from the dead letter queue."""
        try:
            # Extract message information
            message_body = raw_message.body.decode()
            headers = getattr(raw_message, 'headers', {}) or {}
            
            # Try to deserialize the original message
            try:
                original_message = BaseMessage.from_json(message_body)
            except Exception as e:
                self.logger.error(f"❌ Failed to deserialize dead letter message: {e}")
                # Create a minimal message representation
                original_message = BaseMessage(
                    source="unknown",
                    payload={'raw_body': message_body[:1000]}  # Truncate for storage
                )
            
            # Determine failure reason
            failure_reason = self._determine_failure_reason(headers)
            
            # Extract retry count
            retry_count = int(headers.get('x-delivery-count', 0))
            
            # Create dead letter record
            dead_letter = DeadLetterMessage(
                message_id=original_message.message_id,
                original_queue=headers.get('x-original-queue', 'unknown'),
                failure_reason=failure_reason,
                failure_time=datetime.now(),
                retry_count=retry_count,
                original_message=original_message,
                error_details={
                    'headers': headers,
                    'routing_key': getattr(raw_message, 'routing_key', ''),
                    'delivery_count': retry_count
                }
            )
            
            # Store dead letter
            self.dead_letters[original_message.message_id] = dead_letter
            
            # Update statistics
            self.stats['total_dead_letters'] += 1
            self.stats['failure_reasons'][failure_reason.value] += 1
            
            self.logger.warning(f"💀 Dead letter recorded: {original_message.message_id} "
                              f"({failure_reason.value}) after {retry_count} retries")
            
            # Analyze for immediate recovery
            await self._analyze_for_recovery(dead_letter)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing dead letter message: {e}")
    
    def _determine_failure_reason(self, headers: Dict[str, Any]) -> FailureReason:
        """Determine the reason for message failure from headers."""
        # Check for specific error indicators in headers
        error_type = headers.get('x-error-type', '').lower()
        
        if 'deserialization' in error_type or 'json' in error_type:
            return FailureReason.DESERIALIZATION_ERROR
        elif 'timeout' in error_type:
            return FailureReason.TIMEOUT
        elif 'circuit' in error_type:
            return FailureReason.CIRCUIT_BREAKER_OPEN
        elif 'invalid' in error_type:
            return FailureReason.INVALID_MESSAGE
        elif 'handler' in error_type:
            return FailureReason.HANDLER_ERROR
        elif headers.get('x-delivery-count', 0) > 0:
            return FailureReason.MAX_RETRIES_EXCEEDED
        else:
            return FailureReason.UNKNOWN
    
    async def _analyze_for_recovery(self, dead_letter: DeadLetterMessage):
        """Analyze dead letter for potential immediate recovery."""
        # Find applicable recovery rule
        recovery_rule = self._find_recovery_rule(dead_letter)
        
        if not recovery_rule:
            self.logger.debug(f"🔍 No recovery rule found for {dead_letter.message_id}")
            return
        
        # Check if recovery should be attempted
        if (dead_letter.recovery_attempts >= recovery_rule.max_recovery_attempts or
            recovery_rule.action == RecoveryAction.DISCARD):
            
            if recovery_rule.action == RecoveryAction.DISCARD:
                await self._discard_message(dead_letter)
            elif recovery_rule.action == RecoveryAction.MANUAL_REVIEW:
                await self._mark_for_manual_review(dead_letter)
            
            return
        
        # Schedule recovery attempt
        if recovery_rule.action == RecoveryAction.RETRY:
            await self._schedule_recovery(dead_letter, recovery_rule)
    
    def _find_recovery_rule(self, dead_letter: DeadLetterMessage) -> Optional[RecoveryRule]:
        """Find applicable recovery rule for dead letter."""
        for rule in self.recovery_rules:
            if rule.failure_reason == dead_letter.failure_reason:
                # Check additional conditions
                if self._check_rule_conditions(dead_letter, rule):
                    return rule
        
        return None
    
    def _check_rule_conditions(self, dead_letter: DeadLetterMessage, rule: RecoveryRule) -> bool:
        """Check if rule conditions are met for dead letter."""
        conditions = rule.conditions
        
        # Check if message is critical (for max retries exceeded rule)
        if conditions.get('critical_only'):
            message_type = dead_letter.original_message.message_type
            critical_types = [MessageType.TRADING_SIGNAL, MessageType.RISK_MANAGEMENT]
            return message_type in critical_types
        
        return True
    
    async def _schedule_recovery(self, dead_letter: DeadLetterMessage, rule: RecoveryRule):
        """Schedule recovery attempt for dead letter."""
        # Calculate delay based on recovery attempts
        base_delay = rule.recovery_delay
        exponential_delay = base_delay * (2 ** dead_letter.recovery_attempts)
        max_delay = rule.conditions.get('max_delay', 3600)
        delay = min(exponential_delay, max_delay)
        
        self.logger.info(f"⏰ Scheduling recovery for {dead_letter.message_id} "
                        f"in {delay:.0f} seconds")
        
        # Schedule recovery task
        asyncio.create_task(self._perform_delayed_recovery(dead_letter, rule, delay))
    
    async def _perform_delayed_recovery(
        self,
        dead_letter: DeadLetterMessage,
        rule: RecoveryRule,
        delay: float
    ):
        """Perform delayed recovery attempt."""
        try:
            await asyncio.sleep(delay)
            
            # Update recovery attempt info
            dead_letter.recovery_attempts += 1
            dead_letter.last_recovery_attempt = datetime.now()
            
            if rule.action == RecoveryAction.RETRY:
                success = await self._retry_message(dead_letter)
                
                if success:
                    self.stats['recovered_messages'] += 1
                    # Remove from dead letters
                    if dead_letter.message_id in self.dead_letters:
                        del self.dead_letters[dead_letter.message_id]
                    
                    self.logger.info(f"✅ Successfully recovered message: {dead_letter.message_id}")
                else:
                    self.logger.warning(f"⚠️ Recovery attempt failed: {dead_letter.message_id}")
                    
                    # Check if more attempts should be made
                    if dead_letter.recovery_attempts < rule.max_recovery_attempts:
                        await self._schedule_recovery(dead_letter, rule)
                    else:
                        await self._mark_for_manual_review(dead_letter)
            
            elif rule.action == RecoveryAction.REQUEUE:
                await self._requeue_message(dead_letter)
            
        except Exception as e:
            self.logger.error(f"❌ Error in delayed recovery: {e}")
    
    async def _retry_message(self, dead_letter: DeadLetterMessage) -> bool:
        """Retry processing a dead letter message."""
        try:
            # Republish the message to its original routing
            message = dead_letter.original_message
            
            # Determine original exchange and routing key
            original_exchange = self._get_original_exchange(message)
            
            # Publish message back to the system
            await self.message_broker.publish_message(
                message_body=message.to_json(),
                exchange_name=original_exchange,
                routing_key=message.routing_key,
                priority=message.priority.value
            )
            
            self.logger.info(f"🔄 Retried message: {message.message_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to retry message {dead_letter.message_id}: {e}")
            return False
    
    def _get_original_exchange(self, message: BaseMessage) -> str:
        """Determine original exchange for message type."""
        type_exchange_map = {
            MessageType.TRADING_SIGNAL: 'auj.platform',
            MessageType.AGENT_COORDINATION: 'auj.coordination',
            MessageType.SYSTEM_STATUS: 'auj.broadcast',
            MessageType.MARKET_DATA: 'auj.platform',
            MessageType.RISK_MANAGEMENT: 'auj.platform',
            MessageType.PERFORMANCE_UPDATE: 'auj.platform',
            MessageType.ERROR_NOTIFICATION: 'auj.platform'
        }
        
        return type_exchange_map.get(message.message_type, 'auj.platform')
    
    async def _requeue_message(self, dead_letter: DeadLetterMessage):
        """Requeue message to original queue."""
        try:
            # This would require knowing the original queue name
            # For now, we'll retry to the appropriate exchange
            await self._retry_message(dead_letter)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to requeue message: {e}")
    
    async def _discard_message(self, dead_letter: DeadLetterMessage):
        """Discard a dead letter message."""
        self.stats['discarded_messages'] += 1
        
        if dead_letter.message_id in self.dead_letters:
            del self.dead_letters[dead_letter.message_id]
        
        self.logger.info(f"🗑️ Discarded message: {dead_letter.message_id}")
    
    async def _mark_for_manual_review(self, dead_letter: DeadLetterMessage):
        """Mark message for manual review."""
        self.stats['manual_reviews'] += 1
        
        # Add manual review flag
        dead_letter.error_details['manual_review'] = True
        dead_letter.error_details['review_time'] = datetime.now().isoformat()
        
        self.logger.warning(f"👤 Marked for manual review: {dead_letter.message_id}")
    
    async def _check_recovery_opportunities(self):
        """Check for recovery opportunities among stored dead letters."""
        now = datetime.now()
        recovery_count = 0
        
        try:
            for dead_letter in list(self.dead_letters.values()):
                # Skip if already marked for manual review
                if dead_letter.error_details.get('manual_review'):
                    continue
                
                # Find recovery rule
                rule = self._find_recovery_rule(dead_letter)
                if not rule or rule.action != RecoveryAction.RETRY:
                    continue
                
                # Check if enough time has passed since last attempt
                if dead_letter.last_recovery_attempt:
                    time_since_last = (now - dead_letter.last_recovery_attempt).total_seconds()
                    if time_since_last < rule.recovery_delay:
                        continue
                
                # Check if more attempts are allowed
                if dead_letter.recovery_attempts >= rule.max_recovery_attempts:
                    continue
                
                # Schedule recovery
                await self._schedule_recovery(dead_letter, rule)
                recovery_count += 1
                
                # Limit recovery attempts per cycle to avoid overload
                if recovery_count >= 10:
                    break
            
            if recovery_count > 0:
                self.logger.info(f"🔄 Scheduled {recovery_count} message recoveries")
                
        except Exception as e:
            self.logger.error(f"❌ Error checking recovery opportunities: {e}")
    
    async def _cleanup_old_messages(self):
        """Clean up old dead letter messages and perform maintenance."""
        try:
            while True:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old messages
                cutoff_time = datetime.now() - timedelta(days=self.max_storage_days)
                
                to_remove = []
                archived_count = 0
                
                for message_id, dead_letter in self.dead_letters.items():
                    if dead_letter.failure_time < cutoff_time:
                        # Archive important messages before removal
                        if dead_letter.original_message.message_type in [
                            MessageType.TRADING_SIGNAL, 
                            MessageType.RISK_MANAGEMENT
                        ]:
                            await self._archive_dead_letter(dead_letter)
                            archived_count += 1
                        
                        to_remove.append(message_id)
                
                # Remove old messages
                for message_id in to_remove:
                    del self.dead_letters[message_id]
                
                if to_remove:
                    self.logger.info(f"🧹 Cleaned up {len(to_remove)} old dead letters "
                                   f"(archived {archived_count} critical messages)")
                
                # Perform periodic maintenance
                await self._perform_maintenance()
                
        except asyncio.CancelledError:
            self.logger.info("🧹 Cleanup task cancelled")
        except Exception as e:
            self.logger.error(f"❌ Error in cleanup task: {e}")
    
    async def _archive_dead_letter(self, dead_letter: DeadLetterMessage):
        """Archive critical dead letter messages before cleanup."""
        try:
            # Create archive record
            archive_data = {
                'message_id': dead_letter.message_id,
                'original_queue': dead_letter.original_queue,
                'failure_reason': dead_letter.failure_reason.value,
                'failure_time': dead_letter.failure_time.isoformat(),
                'retry_count': dead_letter.retry_count,
                'recovery_attempts': dead_letter.recovery_attempts,
                'message_type': dead_letter.original_message.message_type.value,
                'error_details': dead_letter.error_details,
                'archived_at': datetime.now().isoformat()
            }
            
            # In a real implementation, this would save to persistent storage
            # For now, we'll log the archive action
            self.logger.info(f"📦 Archived dead letter: {dead_letter.message_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to archive dead letter {dead_letter.message_id}: {e}")
    
    async def _perform_maintenance(self):
        """Perform periodic maintenance tasks."""
        try:
            # Update statistics
            self._update_failure_statistics()
            
            # Check for patterns in failures
            await self._analyze_failure_patterns()
            
            # Clean up old recovery rules if needed
            self._cleanup_recovery_rules()
            
            # Log health status
            self.logger.debug(f"💀 Dead letter handler maintenance completed. "
                            f"Active dead letters: {len(self.dead_letters)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error in maintenance: {e}")
    
    def _update_failure_statistics(self):
        """Update failure reason statistics."""
        # Reset counts
        for reason in FailureReason:
            self.stats['failure_reasons'][reason.value] = 0
        
        # Count current dead letters by failure reason
        for dead_letter in self.dead_letters.values():
            self.stats['failure_reasons'][dead_letter.failure_reason.value] += 1
    
    async def _analyze_failure_patterns(self):
        """Analyze patterns in dead letter failures for insights."""
        try:
            # Count failures by hour of day to detect patterns
            hourly_failures = [0] * 24
            
            for dead_letter in self.dead_letters.values():
                hour = dead_letter.failure_time.hour
                hourly_failures[hour] += 1
            
            # Check for unusual patterns (more than 2x average)
            avg_failures = sum(hourly_failures) / 24
            if avg_failures > 0:
                for hour, count in enumerate(hourly_failures):
                    if count > avg_failures * 2:
                        self.logger.warning(f"⚠️ High failure rate detected at hour {hour}: "
                                          f"{count} failures (avg: {avg_failures:.1f})")
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing failure patterns: {e}")
    
    def _cleanup_recovery_rules(self):
        """Clean up unused or ineffective recovery rules."""
        try:
            # This could implement logic to remove rules that never match
            # or have low success rates
            pass
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up recovery rules: {e}")
    
    def get_dead_letter_stats(self) -> Dict[str, Any]:
        """Get dead letter statistics."""
        now = datetime.now()
        
        # Calculate age distribution
        age_distribution = {'< 1h': 0, '1-24h': 0, '1-7d': 0, '> 7d': 0}
        
        for dead_letter in self.dead_letters.values():
            age = (now - dead_letter.failure_time).total_seconds()
            
            if age < 3600:  # < 1 hour
                age_distribution['< 1h'] += 1
            elif age < 86400:  # < 24 hours
                age_distribution['1-24h'] += 1
            elif age < 604800:  # < 7 days
                age_distribution['1-7d'] += 1
            else:
                age_distribution['> 7d'] += 1
        
        return {
            **self.stats,
            'current_dead_letters': len(self.dead_letters),
            'age_distribution': age_distribution,
            'recovery_rate': (
                self.stats['recovered_messages'] / 
                max(self.stats['total_dead_letters'], 1)
            ),
            'monitoring_active': self._monitoring_task is not None
        }
    
    def get_dead_letters_for_review(self) -> List[Dict[str, Any]]:
        """Get dead letters marked for manual review."""
        review_items = []
        
        for dead_letter in self.dead_letters.values():
            if dead_letter.error_details.get('manual_review'):
                review_items.append({
                    'message_id': dead_letter.message_id,
                    'failure_reason': dead_letter.failure_reason.value,
                    'failure_time': dead_letter.failure_time.isoformat(),
                    'retry_count': dead_letter.retry_count,
                    'recovery_attempts': dead_letter.recovery_attempts,
                    'original_queue': dead_letter.original_queue,
                    'message_type': dead_letter.original_message.message_type.value,
                    'source': dead_letter.original_message.source,
                    'error_details': dead_letter.error_details
                })
        
        return review_items
    
    async def manual_recover_message(self, message_id: str) -> bool:
        """Manually recover a specific dead letter message."""
        if message_id not in self.dead_letters:
            return False
        
        dead_letter = self.dead_letters[message_id]
        
        try:
            success = await self._retry_message(dead_letter)
            
            if success:
                self.stats['recovered_messages'] += 1
                del self.dead_letters[message_id]
                self.logger.info(f"✅ Manually recovered message: {message_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Manual recovery failed for {message_id}: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Perform health check of dead letter handler."""
        try:
            # Check if monitoring is running
            if not self._monitoring_task:
                return False
            
            # Check if dead letter queue exists
            queue_info = await self.message_broker.get_queue_info(self.dead_letter_queue)
            return bool(queue_info)
            
        except Exception as e:
            self.logger.error(f"❌ Dead letter handler health check failed: {e}")
            return False