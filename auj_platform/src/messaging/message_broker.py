#!/usr/bin/env python3
"""
Message Broker for AUJ Platform
===============================

Core RabbitMQ connection management and messaging infrastructure.
Handles connection pooling, exchanges, queues, and routing setup.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List

try:
    import aio_pika
    from aio_pika import connect_robust, ExchangeType, Message, DeliveryMode
    from aio_pika.pool import Pool
    RABBITMQ_AVAILABLE = True
except ImportError:
    # Graceful degradation when RabbitMQ is not available
    aio_pika = None
    connect_robust = None
    ExchangeType = None
    Message = None
    Pool = None
    RABBITMQ_AVAILABLE = False

    # Create mock classes for graceful degradation
    class MockDeliveryMode:
        PERSISTENT = 2
        NOT_PERSISTENT = 1

    DeliveryMode = MockDeliveryMode
    RABBITMQ_AVAILABLE = False

import ssl
import json
from datetime import datetime


class MessageBroker:
    """
    Core message broker managing RabbitMQ connections and infrastructure.

    Provides robust connection management, exchange/queue setup, and
    message routing for the AUJ platform's async communication needs.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the message broker."""
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Connection configuration
        self.broker_config = self.config_manager.get_dict('message_broker', {})
        self.host = self.broker_config.get('host', 'localhost')
        self.port = self.broker_config.get('port', 5672)
        self.username = self.broker_config.get('username', 'guest')
        self.password = self.broker_config.get('password', 'guest')

        # Log warning if using default credentials
        if self.username == 'guest' and self.password == 'guest':
            logging.warning("Using default RabbitMQ credentials - consider configuring explicit credentials")

        self.vhost = self.broker_config.get('vhost', '/')
        self.ssl_enabled = self.broker_config.get('ssl_enabled', False)

        # Connection pools
        self.connection_pool = None
        self.channel_pool = None

        # Exchanges and queues
        self.exchanges = {}
        self.queues = {}
        self.bindings = {}

        # State management
        self._initialized = False
        self._connected = False

        self.logger.info(f"🔧 MessageBroker configured for {self.host}:{self.port}")

    async def initialize(self):
        """Initialize the message broker and create infrastructure."""
        # Check if RabbitMQ is available
        if not RABBITMQ_AVAILABLE:
            self.logger.warning("⚠️ RabbitMQ (aio_pika) not available - operating in degraded mode")
            self.logger.info("📝 Messages will be logged instead of queued")
            self._initialized = True
            self._connected = False  # Mark as not connected but initialized
            return True

        try:
            self.logger.info("🔧 Initializing message broker...")

            # Create connection string
            connection_url = self._build_connection_url()

            # Create connection pool
            self.connection_pool = Pool(
                self._create_connection,
                max_size=self.broker_config.get('max_connections', 10),
                loop=asyncio.get_event_loop()
            )

            # Create channel pool
            self.channel_pool = Pool(
                self._create_channel,
                max_size=self.broker_config.get('max_channels', 50),
                loop=asyncio.get_event_loop()
            )

            # Test connection
            await self._test_connection()

            # Setup exchanges, queues, and bindings
            await self._setup_infrastructure()

            self._initialized = True
            self._connected = True

            self.logger.info("✅ Message broker initialized successfully")
            return True

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize message broker: {e}")
            self.logger.info("📝 Falling back to degraded mode - messages will be logged")
            self._initialized = True
            self._connected = False  # Mark as not connected but initialized
            return True  # Return True to allow system to continue

    def _build_connection_url(self) -> str:
        """Build RabbitMQ connection URL."""
        protocol = "amqps" if self.ssl_enabled else "amqp"
        return f"{protocol}://{self.username}:{self.password}@{self.host}:{self.port}{self.vhost}"

    async def _create_connection(self):
        """Create a new RabbitMQ connection."""
        connection_url = self._build_connection_url()

        # SSL context for secure connections
        ssl_context = None
        if self.ssl_enabled:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        connection = await connect_robust(
            connection_url,
            ssl_context=ssl_context,
            heartbeat=self.broker_config.get('heartbeat', 600),
            connection_attempts=self.broker_config.get('connection_attempts', 5),
            retry_delay=self.broker_config.get('retry_delay', 2)
        )

        return connection

    async def _create_channel(self):
        """Create a new channel from the connection pool."""
        async with self.connection_pool.acquire() as connection:
            channel = await connection.channel()

            # Set QoS for fair message distribution
            await channel.set_qos(
                prefetch_count=self.broker_config.get('prefetch_count', 100)
            )

            return channel

    async def _test_connection(self):
        """Test broker connection."""
        try:
            async with self.connection_pool.acquire() as connection:
                self.logger.info(f"✅ Connected to RabbitMQ at {self.host}:{self.port}")

        except Exception as e:
            self.logger.error(f"❌ Connection test failed: {e}")
            raise

    async def _setup_infrastructure(self):
        """Setup exchanges, queues, and bindings."""
        async with self.channel_pool.acquire() as channel:

            # Setup exchanges
            await self._setup_exchanges(channel)

            # Setup queues
            await self._setup_queues(channel)

            # Setup bindings
            await self._setup_bindings(channel)

            self.logger.info("🔗 Message broker infrastructure setup complete")

    async def _setup_exchanges(self, channel):
        """Setup RabbitMQ exchanges."""
        exchange_configs = [
            # Main topic exchange for routing
            {
                'name': 'auj.platform',
                'type': ExchangeType.TOPIC,
                'durable': True,
                'description': 'Main platform exchange for all message routing'
            },
            # Direct exchange for agent coordination
            {
                'name': 'auj.coordination',
                'type': ExchangeType.DIRECT,
                'durable': True,
                'description': 'Agent coordination exchange'
            },
            # Fanout exchange for system broadcasts
            {
                'name': 'auj.broadcast',
                'type': ExchangeType.FANOUT,
                'durable': True,
                'description': 'System-wide broadcast exchange'
            },
            # Headers exchange for complex routing
            {
                'name': 'auj.headers',
                'type': ExchangeType.HEADERS,
                'durable': True,
                'description': 'Headers-based routing exchange'
            },
            # Dead letter exchange
            {
                'name': 'auj.deadletter',
                'type': ExchangeType.DIRECT,
                'durable': True,
                'description': 'Dead letter exchange for failed messages'
            }
        ]

        for exchange_config in exchange_configs:
            exchange = await channel.declare_exchange(
                name=exchange_config['name'],
                type=exchange_config['type'],
                durable=exchange_config['durable']
            )

            self.exchanges[exchange_config['name']] = exchange

            self.logger.debug(f"📢 Exchange created: {exchange_config['name']} "
                            f"({exchange_config['type'].value})")

    async def _setup_queues(self, channel):
        """Setup RabbitMQ queues."""
        queue_configs = [
            # Trading signal queues
            {
                'name': 'auj.trading.signals.high',
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 300000,  # 5 minutes
                    'x-max-priority': 10,
                    'x-dead-letter-exchange': 'auj.deadletter',
                    'x-dead-letter-routing-key': 'trading.signals.failed'
                }
            },
            {
                'name': 'auj.trading.signals.normal',
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 600000,  # 10 minutes
                    'x-max-priority': 5,
                    'x-dead-letter-exchange': 'auj.deadletter',
                    'x-dead-letter-routing-key': 'trading.signals.failed'
                }
            },
            # Agent coordination queues
            {
                'name': 'auj.coordination.consensus',
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 60000,  # 1 minute
                    'x-dead-letter-exchange': 'auj.deadletter'
                }
            },
            {
                'name': 'auj.coordination.conflicts',
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 30000,  # 30 seconds
                    'x-max-priority': 10,
                    'x-dead-letter-exchange': 'auj.deadletter'
                }
            },
            # System status and monitoring
            {
                'name': 'auj.system.status',
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 300000,  # 5 minutes
                    'x-dead-letter-exchange': 'auj.deadletter'
                }
            },
            # Market data queues
            {
                'name': 'auj.market.data.realtime',
                'durable': False,  # Real-time data doesn't need persistence
                'exclusive': False,
                'auto_delete': True,
                'arguments': {
                    'x-message-ttl': 60000,  # 1 minute
                    'x-max-length': 10000  # Limit queue size
                }
            },
            # Risk management
            {
                'name': 'auj.risk.alerts',
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 1800000,  # 30 minutes
                    'x-max-priority': 10,
                    'x-dead-letter-exchange': 'auj.deadletter'
                }
            },
            # Dead letter queue
            {
                'name': 'auj.deadletter.queue',
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 86400000,  # 24 hours
                    'x-max-length': 100000
                }
            },
            # Performance updates
            {
                'name': 'auj.performance.updates',
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 3600000,  # 1 hour
                    'x-max-length': 50000
                }
            }
        ]

        for queue_config in queue_configs:
            queue = await channel.declare_queue(
                name=queue_config['name'],
                durable=queue_config['durable'],
                exclusive=queue_config['exclusive'],
                auto_delete=queue_config['auto_delete'],
                arguments=queue_self.config_manager.get_dict('arguments', {})
            )

            self.queues[queue_config['name']] = queue

            self.logger.debug(f"📬 Queue created: {queue_config['name']}")

    async def _setup_bindings(self, channel):
        """Setup queue bindings to exchanges."""
        bindings = [
            # Trading signals
            ('auj.platform', 'auj.trading.signals.high', 'trading.signal.*.high'),
            ('auj.platform', 'auj.trading.signals.normal', 'trading.signal.*'),

            # Agent coordination
            ('auj.coordination', 'auj.coordination.consensus', 'consensus'),
            ('auj.coordination', 'auj.coordination.conflicts', 'conflict_resolution'),

            # System status
            ('auj.platform', 'auj.system.status', 'system.status.*'),
            ('auj.broadcast', 'auj.system.status', ''),  # Fanout binding

            # Market data
            ('auj.platform', 'auj.market.data.realtime', 'market.data.*'),

            # Risk management
            ('auj.platform', 'auj.risk.alerts', 'risk.*'),

            # Performance updates
            ('auj.platform', 'auj.performance.updates', 'performance.*'),

            # Dead letter bindings
            ('auj.deadletter', 'auj.deadletter.queue', '*')
        ]

        for exchange_name, queue_name, routing_key in bindings:
            if exchange_name in self.exchanges and queue_name in self.queues:
                await self.queues[queue_name].bind(
                    exchange=self.exchanges[exchange_name],
                    routing_key=routing_key
                )

                self.logger.debug(f"🔗 Binding: {exchange_name} -> {queue_name} "
                                f"({routing_key})")

    def get_channel(self):
        """Get a channel from the pool as async context manager."""
        if not self._initialized:
            raise RuntimeError("Message broker not initialized")

        return self.channel_pool.acquire()

    def is_connected(self) -> bool:
        """Check if the message broker is connected."""
        return self._connected and self._initialized

    async def publish_message(
        self,
        message_body: str,
        exchange_name: str = 'auj.platform',
        routing_key: str = '',
        priority: int = 1,
        delivery_mode: DeliveryMode = DeliveryMode.PERSISTENT,
        expiration: Optional[int] = None,
        headers: Optional[Dict[str, Any]] = None
    ):
        """Publish a message to the broker."""
        # Check if we're in degraded mode (RabbitMQ unavailable)
        if not RABBITMQ_AVAILABLE or not self._connected:
            self.logger.info(f"📝 [DEGRADED] Message logged: {exchange_name}/{routing_key} - {message_body[:100]}...")
            return  # Gracefully handle by logging

        try:
            async with self.channel_pool.acquire() as channel:
                exchange = self.exchanges.get(exchange_name)
                if not exchange:
                    raise ValueError(f"Exchange {exchange_name} not found")

                # Prepare message
                message = Message(
                    body=message_body.encode(),
                    delivery_mode=delivery_mode,
                    priority=priority,
                    expiration=expiration,
                    headers=headers or {},
                    timestamp=datetime.now()
                )

                # Publish message
                await exchange.publish(
                    message=message,
                    routing_key=routing_key
                )

                self.logger.debug(f"📤 Message published: {exchange_name}/{routing_key}")

        except Exception as e:
            self.logger.error(f"❌ Failed to publish message: {e}")
            # In degraded mode, log instead of raising
            if not self._connected:
                self.logger.info(f"📝 [FALLBACK] Message logged due to connection failure: {message_body[:100]}...")
            else:
                raise

    async def setup_consumer(
        self,
        queue_name: str,
        callback: Callable,
        auto_ack: bool = False,
        exclusive: bool = False
    ):
        """Setup a message consumer for a queue."""
        # Check if we're in degraded mode (RabbitMQ unavailable)
        if not RABBITMQ_AVAILABLE or not self._connected:
            self.logger.warning(f"⚠️ [DEGRADED] Consumer setup skipped for queue {queue_name} - RabbitMQ unavailable")
            return  # Gracefully handle by skipping consumer setup

        try:
            async with self.channel_pool.acquire() as channel:
                queue = self.queues.get(queue_name)
                if not queue:
                    raise ValueError(f"Queue {queue_name} not found")

                # Setup consumer
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        try:
                            await callback(message)

                            if not auto_ack:
                                await message.ack()

                        except Exception as e:
                            self.logger.error(f"❌ Consumer callback failed: {e}")

                            if not auto_ack:
                                await message.nack(requeue=False)

        except Exception as e:
            self.logger.error(f"❌ Failed to setup consumer: {e}")
            # In degraded mode, log instead of raising
            if not self._connected:
                self.logger.warning(f"⚠️ [FALLBACK] Consumer setup failed due to connection failure: {queue_name}")
            else:
                raise

    async def get_queue_info(self, queue_name: str) -> Dict[str, Any]:
        """Get queue information and statistics."""
        try:
            async with self.channel_pool.acquire() as channel:
                queue = self.queues.get(queue_name)
                if not queue:
                    return {}

                # This would typically use management API
                # For now, return basic info
                return {
                    'name': queue_name,
                    'durable': True,
                    'message_count': 0,  # Would be fetched from management API
                    'consumer_count': 0
                }

        except Exception as e:
            self.logger.error(f"❌ Failed to get queue info: {e}")
            return {}

    async def close(self):
        """Close all connections and clean up."""
        try:
            self.logger.info("🔒 Closing message broker connections...")

            if not RABBITMQ_AVAILABLE or not self._connected:
                self.logger.info("📝 No active connections to close (degraded mode)")
                self._connected = False
                return

            if self.channel_pool:
                await self.channel_pool.close()

            if self.connection_pool:
                await self.connection_pool.close()

            self._connected = False
            self.logger.info("✅ Message broker closed successfully")

        except Exception as e:
            self.logger.error(f"❌ Error closing message broker: {e}")

    def is_connected(self) -> bool:
        """Check if the message broker is connected."""
        return self._connected and self._initialized

    async def health_check(self) -> bool:
        """Perform health check of the message broker."""
        if not RABBITMQ_AVAILABLE:
            self.logger.debug("📝 Health check: RabbitMQ not available (degraded mode)")
            return self._initialized  # Return True if initialized in degraded mode

        try:
            if not self._connected:
                return False

            # Test connection by creating a temporary channel
            async with self.channel_pool.acquire() as channel:
                # Simple operation to test connectivity
                await channel.get_exchange('auj.platform')

            return True

        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return False

    def get_broker_stats(self) -> Dict[str, Any]:
        """Get broker statistics and status."""
        return {
            'initialized': self._initialized,
            'connected': self._connected,
            'host': self.host,
            'port': self.port,
            'vhost': self.vhost,
            'exchanges_count': len(self.exchanges),
            'queues_count': len(self.queues),
            'ssl_enabled': self.ssl_enabled
        }
