#!/usr/bin/env python3
"""
Message Consumer for AUJ Platform
=================================

High-level message consumer with automatic message deserialization,
error handling, retry logic, and dead letter queue management.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime, timedelta
import json
import traceback
from abc import ABC, abstractmethod

from .message_broker import MessageBroker
from .message_types import (
    BaseMessage, MessageType,
    TradingSignalMessage, AgentCoordinationMessage,
    SystemStatusMessage, MarketDataMessage,
    RiskManagementMessage, PerformanceUpdateMessage,
    ErrorNotificationMessage
)


class MessageHandler(ABC):
    """Abstract base class for message handlers."""
    
    @abstractmethod
    async def handle_message(self, message: BaseMessage) -> bool:
        """
        Handle a received message.
        
        Args:
            message: The received message
            
        Returns:
            bool: True if message was handled successfully
        """
        try:
            # Process message based on type and content
            processed_result = await self._process_message_content(message)
            
            # Update message status if processing was successful
            if processed_result:
                await self._update_message_status(message.id, 'processed')
                return True
            else:
                await self._update_message_status(message.id, 'failed')
                return False
                
        except Exception as e:
            # Handle message error with proper logging and recovery
            await self._handle_message_error(message, e)
            raise
    
    async def _process_message_content(self, message: BaseMessage) -> bool:
        """
        Process message based on type and content.
        
        Args:
            message: The message to process
            
        Returns:
            bool: True if processing was successful
        """
        try:
            # Log message processing start
            logger = logging.getLogger(__name__)
            logger.debug(f"Processing message {message.id} of type {message.message_type}")
            
            # Default processing - subclasses should override this method
            # for specific message type handling
            if hasattr(message, 'message_type'):
                if message.message_type == MessageType.TRADING_SIGNAL:
                    return await self._handle_trading_signal(message)
                elif message.message_type == MessageType.AGENT_COORDINATION:
                    return await self._handle_agent_coordination(message)
                elif message.message_type == MessageType.SYSTEM_STATUS:
                    return await self._handle_system_status(message)
                elif message.message_type == MessageType.MARKET_DATA:
                    return await self._handle_market_data(message)
                elif message.message_type == MessageType.RISK_MANAGEMENT:
                    return await self._handle_risk_management(message)
                elif message.message_type == MessageType.PERFORMANCE_UPDATE:
                    return await self._handle_performance_update(message)
                elif message.message_type == MessageType.ERROR_NOTIFICATION:
                    return await self._handle_error_notification(message)
                else:
                    logger.warning(f"Unknown message type: {message.message_type}")
                    return False
            else:
                # Generic message processing
                logger.info(f"Processing generic message: {message.id}")
                return True
                
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            return False
    
    async def _handle_trading_signal(self, message: BaseMessage) -> bool:
        """Handle trading signal messages."""
        logger = logging.getLogger(__name__)
        logger.info(f"Handling trading signal: {message.id}")
        # Base implementation - subclasses should override
        return True
    
    async def _handle_agent_coordination(self, message: BaseMessage) -> bool:
        """Handle agent coordination messages."""
        logger = logging.getLogger(__name__)
        logger.info(f"Handling agent coordination: {message.id}")
        # Base implementation - subclasses should override
        return True
    
    async def _handle_system_status(self, message: BaseMessage) -> bool:
        """Handle system status messages."""
        logger = logging.getLogger(__name__)
        logger.info(f"Handling system status: {message.id}")
        # Base implementation - subclasses should override
        return True
    
    async def _handle_market_data(self, message: BaseMessage) -> bool:
        """Handle market data messages."""
        logger = logging.getLogger(__name__)
        logger.info(f"Handling market data: {message.id}")
        # Base implementation - subclasses should override
        return True
    
    async def _handle_risk_management(self, message: BaseMessage) -> bool:
        """Handle risk management messages."""
        logger = logging.getLogger(__name__)
        logger.info(f"Handling risk management: {message.id}")
        # Base implementation - subclasses should override
        return True
    
    async def _handle_performance_update(self, message: BaseMessage) -> bool:
        """Handle performance update messages."""
        logger = logging.getLogger(__name__)
        logger.info(f"Handling performance update: {message.id}")
        # Base implementation - subclasses should override
        return True
    
    async def _handle_error_notification(self, message: BaseMessage) -> bool:
        """Handle error notification messages."""
        logger = logging.getLogger(__name__)
        logger.info(f"Handling error notification: {message.id}")
        # Base implementation - subclasses should override
        return True
    
    async def _update_message_status(self, message_id: str, status: str):
        """
        Update message processing status.
        
        Args:
            message_id: ID of the message
            status: New status ('processed', 'failed', etc.)
        """
        try:
            logger = logging.getLogger(__name__)
            logger.debug(f"Updating message {message_id} status to: {status}")
            
            # This would typically update a database or tracking system
            # For now, we'll just log it
            logger.info(f"Message {message_id} status updated to: {status}")
            
        except Exception as e:
            logger.error(f"Failed to update message status for {message_id}: {e}")
    
    async def _handle_message_error(self, message: BaseMessage, error: Exception):
        """
        Handle message processing error with proper logging and recovery.
        
        Args:
            message: The message that failed to process
            error: The exception that occurred
        """
        try:
            logger = logging.getLogger(__name__)
            
            # Log the error with full context
            logger.error(
                f"Message processing failed for {message.id}: {str(error)}\n"
                f"Message type: {getattr(message, 'message_type', 'unknown')}\n"
                f"Error type: {type(error).__name__}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            
            # Update message status to failed
            await self._update_message_status(message.id, 'failed')
            
            # Determine if message should be retried or sent to dead letter queue
            retry_count = getattr(message, 'retry_count', 0)
            max_retries = 3  # Could be configurable
            
            if retry_count < max_retries:
                # Schedule retry
                message.retry_count = retry_count + 1
                await self._schedule_retry(message, error)
            else:
                # Send to dead letter queue
                await self._send_to_dead_letter_queue(message, error)
            
        except Exception as nested_error:
            logger.critical(f"Error in error handler for message {message.id}: {nested_error}")
    
    async def _schedule_retry(self, message: BaseMessage, error: Exception):
        """Schedule message for retry."""
        logger = logging.getLogger(__name__)
        logger.info(f"Scheduling retry for message {message.id} (attempt {message.retry_count})")
        
        # This would typically put the message back in the queue with a delay
        # For now, we'll just log it
        
    async def _send_to_dead_letter_queue(self, message: BaseMessage, error: Exception):
        """Send message to dead letter queue."""
        logger = logging.getLogger(__name__)
        logger.warning(f"Sending message {message.id} to dead letter queue after {message.retry_count} retries")
        
        # This would typically send to an actual dead letter queue
        # For now, we'll just log it
    
    def can_handle(self, message_type: MessageType) -> bool:
        """Check if this handler can handle a specific message type."""
        return True


class MessageConsumer:
    """
    High-level message consumer for AUJ platform.
    
    Provides automatic message deserialization, routing to handlers,
    error handling, and retry logic with dead letter queue support.
    """
    
    def __init__(self, message_broker: MessageBroker, config: Dict[str, Any] = None):
        """Initialize message consumer."""
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.message_broker = message_broker
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Consumer configuration
        self.consumer_config = self.config_manager.get_dict('message_consumer', {})
        self.auto_ack = self.consumer_config.get('auto_ack', False)
        self.prefetch_count = self.consumer_config.get('prefetch_count', 100)
        self.max_retry_attempts = self.consumer_config.get('max_retry_attempts', 3)
        self.retry_delay = self.consumer_config.get('retry_delay', 5.0)
        
        # Message handlers
        self._handlers: Dict[MessageType, List[MessageHandler]] = {}
        self._default_handlers: List[MessageHandler] = []
        
        # Consumer state
        self._consumers = {}
        self._consuming = False
        
        # Statistics
        self._consume_stats = {
            'total_consumed': 0,
            'successful_handles': 0,
            'failed_handles': 0,
            'retry_attempts': 0,
            'dead_lettered': 0
        }
        
        self.logger.info("📥 MessageConsumer initialized")
    
    def register_handler(
        self,
        handler: MessageHandler,
        message_types: Optional[List[MessageType]] = None
    ):
        """
        Register a message handler for specific message types.
        
        Args:
            handler: The message handler instance
            message_types: List of message types to handle (None for all)
        """
        if message_types is None:
            self._default_handlers.append(handler)
            self.logger.info(f"📋 Registered default handler: {handler.__class__.__name__}")
        else:
            for message_type in message_types:
                if message_type not in self._handlers:
                    self._handlers[message_type] = []
                self._handlers[message_type].append(handler)
                
                self.logger.info(f"📋 Registered handler: {handler.__class__.__name__} "
                               f"for {message_type.value}")
    
    def unregister_handler(self, handler: MessageHandler):
        """Unregister a message handler."""
        # Remove from default handlers
        if handler in self._default_handlers:
            self._default_handlers.remove(handler)
        
        # Remove from specific type handlers
        for message_type, handlers in self._handlers.items():
            if handler in handlers:
                handlers.remove(handler)
        
        self.logger.info(f"📋 Unregistered handler: {handler.__class__.__name__}")
    
    async def start_consuming(self, queue_names: List[str]):
        """
        Start consuming messages from specified queues.
        
        Args:
            queue_names: List of queue names to consume from
        """
        if self._consuming:
            self.logger.warning("⚠️ Consumer already running")
            return
        
        self._consuming = True
        self.logger.info(f"🚀 Starting message consumption from queues: {queue_names}")
        
        # Start consumers for each queue
        for queue_name in queue_names:
            consumer_task = asyncio.create_task(
                self._consume_from_queue(queue_name)
            )
            self._consumers[queue_name] = consumer_task
        
        self.logger.info("✅ Message consumption started")
    
    async def stop_consuming(self):
        """Stop all message consumers."""
        if not self._consuming:
            return
        
        self.logger.info("🛑 Stopping message consumption...")
        self._consuming = False
        
        # Cancel all consumer tasks
        for queue_name, task in self._consumers.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._consumers.clear()
        self.logger.info("✅ Message consumption stopped")
    
    async def _consume_from_queue(self, queue_name: str):
        """Consume messages from a specific queue."""
        try:
            self.logger.info(f"📥 Starting consumer for queue: {queue_name}")
            
            async with self.message_broker.get_channel() as channel:
                # Set QoS
                await channel.set_qos(prefetch_count=self.prefetch_count)
                
                # Get queue
                queue = self.message_broker.queues.get(queue_name)
                if not queue:
                    self.logger.error(f"❌ Queue not found: {queue_name}")
                    return
                
                # Consume messages
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        if not self._consuming:
                            break
                        
                        await self._process_message(message, queue_name)
                        
        except asyncio.CancelledError:
            self.logger.info(f"📥 Consumer cancelled for queue: {queue_name}")
        except Exception as e:
            self.logger.error(f"❌ Error in consumer for {queue_name}: {e}")
            
            # Restart consumer if still consuming
            if self._consuming:
                self.logger.info(f"🔄 Restarting consumer for {queue_name}...")
                await asyncio.sleep(self.retry_delay)
                self._consumers[queue_name] = asyncio.create_task(
                    self._consume_from_queue(queue_name)
                )
    
    async def _process_message(self, raw_message, queue_name: str):
        """Process a received message."""
        try:
            self._consume_stats['total_consumed'] += 1
            
            # Decode message
            message_body = raw_message.body.decode()
            
            # Deserialize to BaseMessage
            try:
                message = BaseMessage.from_json(message_body)
            except Exception as e:
                self.logger.error(f"❌ Failed to deserialize message: {e}")
                await raw_message.nack(requeue=False)
                return
            
            # Convert to specific message type if needed
            message = self._convert_to_specific_type(message)
            
            self.logger.debug(f"📥 Processing message: {message.message_id} "
                            f"({message.message_type.value}) from {queue_name}")
            
            # Handle message with retry logic
            success = await self._handle_message_with_retry(message, raw_message)
            
            if success:
                self._consume_stats['successful_handles'] += 1
                if not self.auto_ack:
                    await raw_message.ack()
            else:
                self._consume_stats['failed_handles'] += 1
                if not self.auto_ack:
                    # Check retry count and decide whether to requeue or dead letter
                    retry_count = self._get_retry_count(raw_message)
                    
                    if retry_count < self.max_retry_attempts:
                        await raw_message.nack(requeue=True)
                        self.logger.warning(f"⚠️ Requeuing message {message.message_id} "
                                          f"(attempt {retry_count + 1})")
                    else:
                        await raw_message.nack(requeue=False)
                        self._consume_stats['dead_lettered'] += 1
                        self.logger.error(f"💀 Dead lettering message {message.message_id} "
                                        f"after {retry_count} attempts")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing message: {e}")
            if not self.auto_ack:
                await raw_message.nack(requeue=False)
    
    def _convert_to_specific_type(self, base_message: BaseMessage) -> BaseMessage:
        """Convert BaseMessage to specific message type based on message_type."""
        type_mapping = {
            MessageType.TRADING_SIGNAL: TradingSignalMessage,
            MessageType.AGENT_COORDINATION: AgentCoordinationMessage,
            MessageType.SYSTEM_STATUS: SystemStatusMessage,
            MessageType.MARKET_DATA: MarketDataMessage,
            MessageType.RISK_MANAGEMENT: RiskManagementMessage,
            MessageType.PERFORMANCE_UPDATE: PerformanceUpdateMessage,
            MessageType.ERROR_NOTIFICATION: ErrorNotificationMessage
        }
        
        message_class = type_mapping.get(base_message.message_type, BaseMessage)
        
        if message_class == BaseMessage:
            return base_message
        
        # Create new instance of specific type with base message data
        try:
            return message_class(
                message_id=base_message.message_id,
                message_type=base_message.message_type,
                timestamp=base_message.timestamp,
                source=base_message.source,
                priority=base_message.priority,
                routing_key=base_message.routing_key,
                correlation_id=base_message.correlation_id,
                reply_to=base_message.reply_to,
                expiration=base_message.expiration,
                headers=base_message.headers,
                payload=base_message.payload
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to convert to specific type: {e}")
            return base_message
    
    def _get_retry_count(self, raw_message) -> int:
        """Get retry count from message headers."""
        headers = getattr(raw_message, 'headers', {}) or {}
        return int(headers.get('x-delivery-count', 0))
    
    async def _handle_message_with_retry(
        self,
        message: BaseMessage,
        raw_message
    ) -> bool:
        """Handle message with retry logic."""
        retry_count = 0
        
        while retry_count <= self.max_retry_attempts:
            try:
                # Find appropriate handlers
                handlers = self._get_handlers_for_message(message)
                
                if not handlers:
                    self.logger.warning(f"⚠️ No handlers found for message type: "
                                      f"{message.message_type.value}")
                    return False
                
                # Execute all handlers
                all_successful = True
                for handler in handlers:
                    try:
                        success = await handler.handle_message(message)
                        if not success:
                            all_successful = False
                            self.logger.warning(f"⚠️ Handler {handler.__class__.__name__} "
                                              f"failed for message {message.message_id}")
                    except Exception as e:
                        all_successful = False
                        self.logger.error(f"❌ Handler {handler.__class__.__name__} "
                                        f"error: {e}")
                        self.logger.debug(traceback.format_exc())
                
                if all_successful:
                    return True
                
                # If handlers failed, retry
                retry_count += 1
                self._consume_stats['retry_attempts'] += 1
                
                if retry_count <= self.max_retry_attempts:
                    delay = self.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    self.logger.warning(f"⚠️ Retrying message {message.message_id} "
                                      f"in {delay}s (attempt {retry_count})")
                    await asyncio.sleep(delay)
                
            except Exception as e:
                retry_count += 1
                self.logger.error(f"❌ Error handling message {message.message_id}: {e}")
                
                if retry_count <= self.max_retry_attempts:
                    await asyncio.sleep(self.retry_delay)
        
        return False
    
    def _get_handlers_for_message(self, message: BaseMessage) -> List[MessageHandler]:
        """Get handlers that can process the given message."""
        handlers = []
        
        # Get type-specific handlers
        if message.message_type in self._handlers:
            for handler in self._handlers[message.message_type]:
                if handler.can_handle(message.message_type):
                    handlers.append(handler)
        
        # Add default handlers that can handle this type
        for handler in self._default_handlers:
            if handler.can_handle(message.message_type):
                handlers.append(handler)
        
        return handlers
    
    async def consume_single_message(
        self,
        queue_name: str,
        timeout_seconds: Optional[float] = None
    ) -> Optional[BaseMessage]:
        """
        Consume a single message from a queue.
        
        Args:
            queue_name: Queue to consume from
            timeout_seconds: Timeout for waiting for message
            
        Returns:
            BaseMessage or None if timeout/error
        """
        try:
            async with self.message_broker.get_channel() as channel:
                queue = self.message_broker.queues.get(queue_name)
                if not queue:
                    self.logger.error(f"❌ Queue not found: {queue_name}")
                    return None
                
                # Get single message
                if timeout_seconds:
                    try:
                        raw_message = await asyncio.wait_for(
                            queue.get(),
                            timeout=timeout_seconds
                        )
                    except asyncio.TimeoutError:
                        return None
                else:
                    raw_message = await queue.get()
                
                if raw_message is None:
                    return None
                
                # Decode and convert message
                message_body = raw_message.body.decode()
                message = BaseMessage.from_json(message_body)
                message = self._convert_to_specific_type(message)
                
                # Auto-acknowledge if configured
                if self.auto_ack:
                    await raw_message.ack()
                
                return message
                
        except Exception as e:
            self.logger.error(f"❌ Error consuming single message: {e}")
            return None
    
    def get_consume_stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        total = max(self._consume_stats['total_consumed'], 1)
        
        return {
            **self._consume_stats,
            'success_rate': self._consume_stats['successful_handles'] / total,
            'failure_rate': self._consume_stats['failed_handles'] / total,
            'active_consumers': len(self._consumers),
            'consuming': self._consuming,
            'registered_handlers': {
                msg_type.value: len(handlers)
                for msg_type, handlers in self._handlers.items()
            },
            'default_handlers': len(self._default_handlers)
        }
    
    async def health_check(self) -> bool:
        """Perform health check of the consumer."""
        try:
            # Check if consuming and consumers are healthy
            if not self._consuming:
                return False
            
            # Check if any consumers have failed
            failed_consumers = []
            for queue_name, task in self._consumers.items():
                if task.done() and not task.cancelled():
                    failed_consumers.append(queue_name)
            
            if failed_consumers:
                self.logger.warning(f"⚠️ Failed consumers detected: {failed_consumers}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Consumer health check failed: {e}")
            return False


# Example message handlers
class TradingSignalHandler(MessageHandler):
    """Example handler for trading signals."""
    
    def __init__(self):
        """Initialize trading signal handler."""
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.logger = logging.getLogger(__name__)
    
    async def handle_message(self, message: BaseMessage) -> bool:
        """Handle trading signal message."""
        if not isinstance(message, TradingSignalMessage):
            return False
        
        try:
            payload = message.payload
            self.logger.info(f"📊 Trading signal: {payload['signal_type']} "
                           f"{payload['symbol']} at {payload['price']} "
                           f"(confidence: {payload['confidence']:.1%})")
            
            # Process trading signal
            # This would integrate with the trading engine
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error handling trading signal: {e}")
            return False
    
    def can_handle(self, message_type: MessageType) -> bool:
        """Check if can handle message type."""
        return message_type == MessageType.TRADING_SIGNAL


class SystemStatusHandler(MessageHandler):
    """Example handler for system status messages."""
    
    def __init__(self):
        """Initialize system status handler."""
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.logger = logging.getLogger(__name__)
    
    async def handle_message(self, message: BaseMessage) -> bool:
        """Handle system status message."""
        if not isinstance(message, SystemStatusMessage):
            return False
        
        try:
            payload = message.payload
            self.logger.info(f"🔧 System status: {payload['component']} is "
                           f"{payload['status']}")
            
            # Process system status
            # This would update monitoring systems
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error handling system status: {e}")
            return False
    
    def can_handle(self, message_type: MessageType) -> bool:
        """Check if can handle message type."""
        return message_type == MessageType.SYSTEM_STATUS