"""
Unified Database Manager for AUJ Platform
=========================================

This module provides a unified database abstraction layer that resolves
mixed sync/async patterns, connection pooling issues, and transaction
management inconsistencies identified in Phase 1.1B analysis.

Key Features:
- Unified sync/async interface
- Intelligent connection pooling
- Transaction management with automatic rollback
- Query caching and performance monitoring
- Graceful degradation between SQLite and PostgreSQL
- Connection health monitoring and recovery

Author: AUJ Platform Development Team
Date: 2025-07-04
Version: 2.1.0 - Fixed Bug #28: Database Deadlock (threading.Lock → asyncio.Lock)
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Union, AsyncContextManager, ContextManager
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import logging
import sqlite3
from pathlib import Path

import asyncpg
from sqlalchemy import create_engine, MetaData, text, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError

from .exceptions import DatabaseError, ConfigurationError
from .unified_config import get_unified_config
from .logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class DatabaseHealth:
    """Database health metrics."""
    is_healthy: bool
    connection_count: int
    active_connections: int
    idle_connections: int
    total_queries: int
    failed_queries: int
    avg_query_time: float
    last_health_check: datetime
    error_messages: List[str]


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_hash: str
    query_type: str
    execution_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    last_executed: datetime
    error_count: int


class BoundedMetricsCollector:
    """Memory-bounded metrics collector to prevent memory leaks."""

    def __init__(self, max_size: int = 10000, cleanup_threshold: float = 0.8):
        self.max_size = max_size
        self.cleanup_threshold = cleanup_threshold
        self.metrics = {}
        self.access_queue = deque(maxlen=max_size)
        self.queue_set = set()  # For O(1) membership checking
        self.last_cleanup = datetime.utcnow()
        self._lock = asyncio.Lock()  # ✅ FIXED: Changed from threading.Lock to asyncio.Lock

    async def record_metric(self, query_hash: str, metric: QueryMetrics):
        """Record a metric with automatic cleanup when needed."""
        async with self._lock:  # ✅ FIXED: async with instead of with
            # Add or update metric
            self.metrics[query_hash] = metric

            # Update access order efficiently
            if query_hash not in self.queue_set:
                # Add to queue if not already present
                if len(self.access_queue) >= self.max_size:
                    # Remove oldest if queue is full
                    old_hash = self.access_queue.popleft()
                    self.queue_set.discard(old_hash)

                self.access_queue.append(query_hash)
                self.queue_set.add(query_hash)

            # Cleanup if needed
            if len(self.metrics) >= int(self.max_size * self.cleanup_threshold):
                await self._cleanup_old_metrics()  # ✅ FIXED: await added

    async def get_metric(self, query_hash: str) -> Optional[QueryMetrics]:
        """Get metric without updating access order to avoid performance issues."""
        async with self._lock:  # ✅ FIXED: async with instead of with
            return self.metrics.get(query_hash)

    async def _cleanup_old_metrics(self):
        """Remove least recently used metrics."""
        # Remove oldest 25% of metrics
        cleanup_count = len(self.metrics) // 4

        for _ in range(cleanup_count):
            if self.access_queue:
                old_hash = self.access_queue.popleft()
                self.queue_set.discard(old_hash)
                self.metrics.pop(old_hash, None)

        self.last_cleanup = datetime.utcnow()

    async def get_all_metrics(self) -> Dict[str, QueryMetrics]:
        """Get all current metrics."""
        async with self._lock:  # ✅ FIXED: async with instead of with
            return self.metrics.copy()

    async def get_memory_usage_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        async with self._lock:  # ✅ FIXED: async with instead of with
            return {
                'total_metrics': len(self.metrics),
                'max_size': self.max_size,
                'memory_usage_pct': (len(self.metrics) / self.max_size) * 100,
                'last_cleanup': self.last_cleanup,
                'queue_length': len(self.access_queue)
            }


class ConnectionPool:
    """Enhanced connection pool with health monitoring."""

    def __init__(self, database_url: str, is_sqlite: bool, max_connections: int = 20):
        self.database_url = database_url
        self.is_sqlite = is_sqlite
        self.max_connections = max_connections
        self.active_connections = 0
        self.total_connections = 0
        self.failed_connections = 0
        self.connection_times = []
        self._lock = asyncio.Lock()  # ✅ FIXED: Changed from threading.Lock to asyncio.Lock

    async def get_health(self) -> DatabaseHealth:
        """Get connection pool health metrics."""
        async with self._lock:  # ✅ FIXED: async with instead of with
            avg_connection_time = (
                sum(self.connection_times[-100:]) / len(self.connection_times[-100:])
                if self.connection_times else 0
            )

            return DatabaseHealth(
                is_healthy=self.failed_connections / max(self.total_connections, 1) < 0.1,
                connection_count=self.total_connections,
                active_connections=self.active_connections,
                idle_connections=self.max_connections - self.active_connections,
                total_queries=0,  # Will be filled by parent
                failed_queries=0,  # Will be filled by parent
                avg_query_time=avg_connection_time,
                last_health_check=datetime.utcnow(),
                error_messages=[]
            )

    async def record_connection(self, success: bool, duration: float):
        """Record connection attempt metrics."""
        async with self._lock:  # ✅ FIXED: async with instead of with
            self.total_connections += 1
            if success:
                self.active_connections += 1
                self.connection_times.append(duration)
                # Keep only last 1000 connection times
                if len(self.connection_times) > 1000:
                    self.connection_times = self.connection_times[-1000:]
            else:
                self.failed_connections += 1


class QueryCache:
    """Intelligent query result caching."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self._lock = asyncio.Lock()  # ✅ FIXED: Changed from threading.Lock to asyncio.Lock

    async def get(self, query_hash: str) -> Optional[Any]:
        """Get cached query result if valid."""
        async with self._lock:  # ✅ FIXED: async with instead of with
            if query_hash not in self.cache:
                return None

            # Check TTL
            if time.time() - self.creation_times[query_hash] > self.ttl_seconds:
                await self._remove(query_hash)  # ✅ FIXED: await added
                return None

            # Update access time
            self.access_times[query_hash] = time.time()
            return self.cache[query_hash]

    async def set(self, query_hash: str, result: Any):
        """Cache query result."""
        async with self._lock:  # ✅ FIXED: async with instead of with
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                await self._evict_lru()  # ✅ FIXED: await added

            self.cache[query_hash] = result
            self.access_times[query_hash] = time.time()
            self.creation_times[query_hash] = time.time()

    async def _remove(self, query_hash: str):
        """Remove item from cache."""
        self.cache.pop(query_hash, None)
        self.access_times.pop(query_hash, None)
        self.creation_times.pop(query_hash, None)

    async def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        await self._remove(lru_key)  # ✅ FIXED: await added


class UnifiedDatabaseManager:
    """
    Unified Database Manager that resolves all database abstraction issues.

    Features:
    - Unified sync/async interface
    - Intelligent connection pooling
    - Transaction management
    - Query caching and monitoring
    - Health monitoring and recovery
    - Graceful degradation
    """

    def __init__(self, database_url: Optional[str] = None):
        """Initialize the unified database manager."""
        self.config = get_unified_config()
        self.database_url = database_url or self.config.get('database.url', 'sqlite:///data/auj_platform.db')
        self.is_sqlite = self.database_url.startswith('sqlite')

        # Connection management
        self.sync_engine = None
        self.async_engine = None
        self.sync_session_factory = None
        self.async_session_factory = None

        # Monitoring and optimization
        self.connection_pool = ConnectionPool(self.database_url, self.is_sqlite)
        self.query_cache = QueryCache()
        self.query_metrics_collector = BoundedMetricsCollector(max_size=10000)

        # Health monitoring
        self.total_queries = 0
        self.failed_queries = 0
        self.last_health_check = datetime.utcnow()
        self.is_initialized = False

        # Thread safety
        self._init_lock = asyncio.Lock()  # ✅ FIXED: Changed from threading.Lock to asyncio.Lock
        self._metrics_lock = asyncio.Lock()  # ✅ FIXED: Changed from threading.Lock to asyncio.Lock

    async def initialize(self) -> bool:
        """Initialize database connections with enhanced error handling."""
        if self.is_initialized:
            return True

        async with self._init_lock:  # ✅ FIXED: async with instead of with
            if self.is_initialized:  # Double-check after acquiring lock
                return True

            try:
                start_time = time.time()

                if self.is_sqlite:
                    success = await self._initialize_sqlite()
                else:
                    success = await self._initialize_postgresql()

                duration = time.time() - start_time
                await self.connection_pool.record_connection(success, duration)  # ✅ FIXED: await added

                if success:
                    self.is_initialized = True
                    logger.info(f"Unified database manager initialized: {self.database_url}")

                    # Setup health monitoring
                    asyncio.create_task(self._health_monitor_loop())

                return success

            except Exception as e:
                logger.error(f"Failed to initialize unified database manager: {e}")
                await self.connection_pool.record_connection(False, 0)  # ✅ FIXED: await added
                return False

    async def _initialize_sqlite(self) -> bool:
        """Initialize SQLite with optimized settings."""
        try:
            # Ensure directory exists
            db_path = self.database_url.replace('sqlite:///', '').replace('sqlite://', '')
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            # Optimized SQLite engine
            self.sync_engine = create_engine(
                self.database_url,
                echo=False,
                poolclass=StaticPool,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30,
                    "isolation_level": None  # Enable autocommit mode
                }
            )

            # Configure SQLite for performance
            @event.listens_for(self.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                cursor.close()

            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                expire_on_commit=False
            )

            # Test connection
            with self.sync_session_factory() as session:
                session.execute(text("SELECT 1"))

            logger.info("SQLite database initialized with optimized settings")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            return False

    async def _initialize_postgresql(self) -> bool:
        """Initialize PostgreSQL with connection pooling."""
        try:
            # Convert to async URL
            if not self.database_url.startswith('postgresql+asyncpg'):
                async_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')
            else:
                async_url = self.database_url

            # Async engine with connection pooling
            self.async_engine = create_async_engine(
                async_url,
                echo=False,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_timeout=30
            )

            self.async_session_factory = async_sessionmaker(
                self.async_engine,
                expire_on_commit=False
            )

            # Sync engine for compatibility
            sync_url = async_url.replace('postgresql+asyncpg://', 'postgresql://')
            self.sync_engine = create_engine(
                sync_url,
                echo=False,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                expire_on_commit=False
            )

            # Test connection
            async with self.async_session_factory() as session:
                await session.execute(text("SELECT 1"))

            logger.info("PostgreSQL database initialized with connection pooling")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            return False

    @asynccontextmanager
    async def get_async_session(self) -> AsyncContextManager[AsyncSession]:
        """Get async database session with automatic transaction management."""
        if not self.is_initialized:
            await self.initialize()

        if self.is_sqlite:
            # For SQLite, use sync session in async context
            session = self.sync_session_factory()
            try:
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                await self._record_failed_query()  # ✅ FIXED: await added
                raise DatabaseError(f"Database session error: {str(e)}")
            finally:
                session.close()
        else:
            # PostgreSQL async session
            if not self.async_session_factory:
                raise DatabaseError("Async session factory not initialized")

            async with self.async_session_factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception as e:
                    await session.rollback()
                    await self._record_failed_query()  # ✅ FIXED: await added
                    raise DatabaseError(f"Database session error: {str(e)}")

    @contextmanager
    def get_sync_session(self) -> ContextManager[Session]:
        """Get synchronous database session with automatic transaction management."""
        if not self.is_initialized:
            # Initialize synchronously for sync usage - check if event loop is running
            try:
                loop = asyncio.get_running_loop()
                # If we're in an event loop, we can't use asyncio.run()
                # Instead, create a new thread to run the async initialization
                import threading
                import concurrent.futures
                
                def init_db():
                    # Create a new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.initialize())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(init_db)
                    result = future.result(timeout=30)
                    if not result:
                        raise DatabaseError("Database initialization failed")
            except RuntimeError:
                # No event loop running, can use asyncio.run()
                asyncio.run(self.initialize())

        if not self.sync_session_factory:
            raise DatabaseError("Sync session factory not initialized")

        session = self.sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            # For sync context, we need to handle _record_failed_query without await
            # We'll use a helper to run it in the event loop if available
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._record_failed_query())
            except RuntimeError:
                # No event loop, run synchronously
                asyncio.run(self._record_failed_query())
            raise DatabaseError(f"Database session error: {str(e)}")
        finally:
            session.close()

    async def execute_async_query(self,
                                 query: str,
                                 parameters: Optional[Dict] = None,
                                 use_cache: bool = True) -> List[Dict[str, Any]]:
        """Execute async query with caching and monitoring."""
        start_time = time.time()
        query_hash = self._get_query_hash(query, parameters)

        # Check cache first
        if use_cache:
            cached_result = await self.query_cache.get(query_hash)  # ✅ FIXED: await added
            if cached_result is not None:
                logger.debug(f"Query cache hit: {query_hash[:16]}...")
                return cached_result

        try:
            async with self.get_async_session() as session:
                if parameters:
                    result = await session.execute(text(query), parameters)
                else:
                    result = await session.execute(text(query))

                # Convert to list of dicts
                rows = []
                for row in result.fetchall():
                    if hasattr(row, '_mapping'):
                        rows.append(dict(row._mapping))
                    else:
                        rows.append(dict(row))

                # Cache result if appropriate
                if use_cache and self._is_cacheable_query(query):
                    await self.query_cache.set(query_hash, rows)  # ✅ FIXED: await added

                # Record metrics
                execution_time = time.time() - start_time
                await self._record_query_metrics(query_hash, query, execution_time, True)  # ✅ FIXED: await added

                return rows

        except Exception as e:
            execution_time = time.time() - start_time
            await self._record_query_metrics(query_hash, query, execution_time, False)  # ✅ FIXED: await added
            logger.error(f"Async query failed: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}")

    def execute_sync_query(self,
                          query: str,
                          parameters: Optional[Dict] = None,
                          use_cache: bool = True) -> List[Dict[str, Any]]:
        """Execute sync query with caching and monitoring."""
        start_time = time.time()
        query_hash = self._get_query_hash(query, parameters)

        # Check cache first - need to handle async cache access
        # FIXED BUG #3: Removed unnecessary task creation that was causing memory leak
        if use_cache:
            try:
                # Check if we're in async context
                loop = asyncio.get_running_loop()
                # Skip cache for sync methods in async context
                cached_result = None
            except RuntimeError:
                # No event loop, run cache check synchronously
                cached_result = asyncio.run(self.query_cache.get(query_hash))
                if cached_result is not None:
                    logger.debug(f"Query cache hit: {query_hash[:16]}...")
                    return cached_result

        try:
            with self.get_sync_session() as session:
                if parameters:
                    result = session.execute(text(query), parameters)
                else:
                    result = session.execute(text(query))

                # Convert to list of dicts
                rows = []
                for row in result.fetchall():
                    if hasattr(row, '_mapping'):
                        rows.append(dict(row._mapping))
                    else:
                        rows.append(dict(row))

                # Cache result if appropriate
                if use_cache and self._is_cacheable_query(query):
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.create_task(self.query_cache.set(query_hash, rows))
                    except RuntimeError:
                        asyncio.run(self.query_cache.set(query_hash, rows))

                # Record metrics
                execution_time = time.time() - start_time
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(self._record_query_metrics(query_hash, query, execution_time, True))
                except RuntimeError:
                    asyncio.run(self._record_query_metrics(query_hash, query, execution_time, True))

                return rows

        except Exception as e:
            execution_time = time.time() - start_time
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self._record_query_metrics(query_hash, query, execution_time, False))
            except RuntimeError:
                asyncio.run(self._record_query_metrics(query_hash, query, execution_time, False))
            logger.error(f"Sync query failed: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}")

    async def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute multiple operations in a single transaction."""
        try:
            async with self.get_async_session() as session:
                for operation in operations:
                    query = operation.get('query')
                    parameters = operation.get('parameters', {})

                    if not query:
                        continue

                    if parameters:
                        await session.execute(text(query), parameters)
                    else:
                        await session.execute(text(query))

                # Transaction is automatically committed by context manager
                logger.debug(f"Transaction completed successfully with {len(operations)} operations")
                return True

        except Exception as e:
            logger.error(f"Transaction failed: {str(e)}")
            raise DatabaseError(f"Transaction failed: {str(e)}")

    async def get_health_status(self) -> DatabaseHealth:
        """Get comprehensive database health status."""
        pool_health = await self.connection_pool.get_health()  # ✅ FIXED: await added

        # Add query metrics
        async with self._metrics_lock:  # ✅ FIXED: async with instead of with
            pool_health.total_queries = self.total_queries
            pool_health.failed_queries = self.failed_queries

            if self.total_queries > 0:
                all_metrics = await self.query_metrics_collector.get_all_metrics()  # ✅ FIXED: await added
                total_time = sum(metric.total_time for metric in all_metrics.values())
                pool_health.avg_query_time = total_time / self.total_queries

        pool_health.last_health_check = datetime.utcnow()

        return pool_health

    async def get_query_performance_report(self) -> Dict[str, Any]:
        """Get detailed query performance report."""
        async with self._metrics_lock:  # ✅ FIXED: async with instead of with
            report = {
                'total_queries': self.total_queries,
                'failed_queries': self.failed_queries,
                'success_rate': (self.total_queries - self.failed_queries) / max(self.total_queries, 1),
                'cache_stats': {
                    'size': len(self.query_cache.cache),
                    'max_size': self.query_cache.max_size,
                    'ttl_seconds': self.query_cache.ttl_seconds
                },
                'top_queries': []
            }

            # Get top 10 slowest queries
            all_metrics = await self.query_metrics_collector.get_all_metrics()  # ✅ FIXED: await added
            sorted_metrics = sorted(
                all_metrics.values(),
                key=lambda m: m.avg_time,
                reverse=True
            )[:10]

            for metric in sorted_metrics:
                report['top_queries'].append({
                    'query_hash': metric.query_hash,
                    'execution_count': metric.execution_count,
                    'avg_time': metric.avg_time,
                    'total_time': metric.total_time,
                    'error_count': metric.error_count
                })

            return report

    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data with performance monitoring."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            cleanup_operations = [
                {
                    'query': "DELETE FROM trade_signals WHERE timestamp < :cutoff_date",
                    'parameters': {'cutoff_date': cutoff_date}
                },
                {
                    'query': "DELETE FROM agent_decisions WHERE timestamp < :cutoff_date",
                    'parameters': {'cutoff_date': cutoff_date}
                },
                {
                    'query': "DELETE FROM platform_status WHERE timestamp < :cutoff_date",
                    'parameters': {'cutoff_date': cutoff_date}
                }
            ]

            await self.execute_transaction(cleanup_operations)
            logger.info(f"Cleaned up data older than {days_to_keep} days")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            raise DatabaseError(f"Failed to cleanup old data: {str(e)}")

    async def optimize_database(self):
        """Perform database optimization operations."""
        try:
            if self.is_sqlite:
                await self._optimize_sqlite()
            else:
                await self._optimize_postgresql()

            logger.info("Database optimization completed")

        except Exception as e:
            logger.error(f"Database optimization failed: {str(e)}")

    async def _optimize_sqlite(self):
        """SQLite-specific optimizations."""
        optimization_queries = [
            "VACUUM;",
            "ANALYZE;",
            "PRAGMA optimize;"
        ]

        for query in optimization_queries:
            await self.execute_async_query(query, use_cache=False)

    async def _optimize_postgresql(self):
        """PostgreSQL-specific optimizations."""
        optimization_queries = [
            "VACUUM ANALYZE;",
            "REINDEX DATABASE current_database();"
        ]

        for query in optimization_queries:
            try:
                await self.execute_async_query(query, use_cache=False)
            except Exception as e:
                logger.warning(f"Optimization query failed: {query} - {e}")

    def _get_query_hash(self, query: str, parameters: Optional[Dict] = None) -> str:
        """Generate hash for query caching."""
        import hashlib
        content = query + str(sorted(parameters.items()) if parameters else "")
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cacheable_query(self, query: str) -> bool:
        """Determine if query results should be cached."""
        query_lower = query.lower().strip()

        # Don't cache write operations
        if any(keyword in query_lower for keyword in ['insert', 'update', 'delete', 'create', 'drop', 'alter']):
            return False

        # Don't cache queries with current timestamp functions
        if any(func in query_lower for func in ['now()', 'current_timestamp', 'current_time']):
            return False

        return True

    async def _record_query_metrics(self, query_hash: str, query: str, execution_time: float, success: bool):
        """Record query performance metrics."""
        async with self._metrics_lock:  # ✅ FIXED: async with instead of with
            self.total_queries += 1
            if not success:
                self.failed_queries += 1

            # Get existing metric or create new one
            metric = await self.query_metrics_collector.get_metric(query_hash)  # ✅ FIXED: await added
            if metric is None:  # First time
                metric = QueryMetrics(
                    query_hash=query_hash,
                    query_type=self._get_query_type(query),
                    execution_count=0,
                    total_time=0.0,
                    avg_time=0.0,
                    min_time=float('inf'),
                    max_time=0.0,
                    last_executed=datetime.utcnow(),
                    error_count=0
                )

            # Update metrics
            metric.execution_count += 1
            metric.total_time += execution_time
            metric.avg_time = metric.total_time / metric.execution_count
            metric.min_time = min(metric.min_time, execution_time)
            metric.max_time = max(metric.max_time, execution_time)
            metric.last_executed = datetime.utcnow()

            if not success:
                metric.error_count += 1

            # Record the updated metric
            await self.query_metrics_collector.record_metric(query_hash, metric)  # ✅ FIXED: await added

    async def _record_failed_query(self):
        """Record a failed query for metrics."""
        async with self._metrics_lock:  # ✅ FIXED: async with instead of with
            self.failed_queries += 1

    def _get_query_type(self, query: str) -> str:
        """Determine query type for categorization."""
        query_lower = query.lower().strip()

        if query_lower.startswith('select'):
            return 'SELECT'
        elif query_lower.startswith('insert'):
            return 'INSERT'
        elif query_lower.startswith('update'):
            return 'UPDATE'
        elif query_lower.startswith('delete'):
            return 'DELETE'
        elif query_lower.startswith('create'):
            return 'CREATE'
        elif query_lower.startswith('drop'):
            return 'DROP'
        elif query_lower.startswith('alter'):
            return 'ALTER'
        else:
            return 'OTHER'

    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while self.is_initialized:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                health = await self.get_health_status()  # ✅ FIXED: await added

                if not health.is_healthy:
                    logger.warning(f"Database health check failed: {health.error_messages}")

                    # Attempt recovery
                    try:
                        if self.is_sqlite:
                            # Test SQLite connection
                            with self.get_sync_session() as session:
                                session.execute(text("SELECT 1"))
                        else:
                            # Test PostgreSQL connection
                            async with self.get_async_session() as session:
                                await session.execute(text("SELECT 1"))

                        logger.info("Database connection recovery successful")

                    except Exception as recovery_error:
                        logger.error(f"Database recovery failed: {recovery_error}")

                # Log performance metrics periodically
                if self.total_queries % 1000 == 0 and self.total_queries > 0:
                    report = await self.get_query_performance_report()  # ✅ FIXED: await added
                    logger.info(f"Database performance: {report['success_rate']:.2%} success rate, "
                              f"avg query time: {health.avg_query_time:.3f}s")

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)  # Shorter delay on error

    async def get_memory_usage_info(self) -> Dict[str, Any]:
        """Get memory usage information for the metrics collector."""
        memory_info = await self.query_metrics_collector.get_memory_usage_info()  # ✅ FIXED: await added
        async with self._metrics_lock:  # ✅ FIXED: async with instead of with
            return {
                'query_metrics': memory_info,
                'total_queries_processed': self.total_queries,
                'failed_queries': self.failed_queries,
                'memory_efficiency': {
                    'bounded_collection': True,
                    'max_metrics_stored': memory_info['max_size'],
                    'cleanup_automatic': True
                }
            }

    async def close(self):
        """Close all database connections and cleanup resources."""
        try:
            self.is_initialized = False

            if self.async_engine:
                await self.async_engine.dispose()

            if self.sync_engine:
                self.sync_engine.dispose()

            logger.info("Unified database manager closed successfully")

        except Exception as e:
            logger.error(f"Error closing unified database manager: {e}")

    async def fetch_one(self, query: str, parameters: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row from the database."""
        try:
            if self.is_sqlite:
                # For SQLite, use sync operations
                session = self.sync_session_factory()
                try:
                    if parameters:
                        # Convert tuple to dict for named parameters
                        param_dict = {f'param_{i}': param for i, param in enumerate(parameters)}
                        # Replace ? with :param_0, :param_1, etc.
                        formatted_query = query
                        for i in range(len(parameters)):
                            formatted_query = formatted_query.replace('?', f':param_{i}', 1)
                        result = session.execute(text(formatted_query), param_dict)
                    else:
                        result = session.execute(text(query))
                    row = result.fetchone()
                    if row:
                        return dict(row._mapping)
                    return None
                finally:
                    session.close()
            else:
                # For PostgreSQL, use async operations
                async with self.async_session_factory() as session:
                    if parameters:
                        param_dict = {f'param_{i}': param for i, param in enumerate(parameters)}
                        formatted_query = query
                        for i in range(len(parameters)):
                            formatted_query = formatted_query.replace('?', f':param_{i}', 1)
                        result = await session.execute(text(formatted_query), param_dict)
                    else:
                        result = await session.execute(text(query))
                    row = result.fetchone()
                    if row:
                        return dict(row._mapping)
                    return None
        except Exception as e:
            logger.error(f"Fetch one failed: {e}")
            raise DatabaseError(f"Failed to fetch data: {str(e)}")

    async def _fetch_one_sqlite(self, query: str, parameters: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch one row from SQLite."""
        try:
            session = self.sync_session_factory()
            try:
                result = session.execute(text(query), parameters or ())
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
                return None
            finally:
                session.close()
        except Exception as e:
            logger.error(f"SQLite fetch one failed: {e}")
            raise

    async def _fetch_one_postgresql(self, query: str, parameters: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch one row from PostgreSQL."""
        try:
            async with self.async_session_factory() as session:
                result = await session.execute(text(query), parameters or ())
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
                return None
        except Exception as e:
            logger.error(f"PostgreSQL fetch one failed: {e}")
            raise

    async def execute_query(self, query: str, parameters: Optional[tuple] = None) -> None:
        """Execute a query without returning results."""
        try:
            if self.is_sqlite:
                # For SQLite, use sync operations
                session = self.sync_session_factory()
                try:
                    if parameters:
                        # Convert tuple to dict for named parameters
                        param_dict = {f'param_{i}': param for i, param in enumerate(parameters)}
                        # Replace ? with :param_0, :param_1, etc.
                        formatted_query = query
                        for i in range(len(parameters)):
                            formatted_query = formatted_query.replace('?', f':param_{i}', 1)
                        session.execute(text(formatted_query), param_dict)
                    else:
                        session.execute(text(query))
                    session.commit()
                finally:
                    session.close()
            else:
                # For PostgreSQL, use async operations
                async with self.async_session_factory() as session:
                    if parameters:
                        param_dict = {f'param_{i}': param for i, param in enumerate(parameters)}
                        formatted_query = query
                        for i in range(len(parameters)):
                            formatted_query = formatted_query.replace('?', f':param_{i}', 1)
                        await session.execute(text(formatted_query), param_dict)
                    else:
                        await session.execute(text(query))
                    await session.commit()
        except Exception as e:
            logger.error(f"Execute query failed: {e}")
            raise DatabaseError(f"Failed to execute query: {str(e)}")

    async def _execute_sqlite(self, query: str, parameters: Optional[tuple] = None) -> None:
        """Execute query in SQLite."""
        try:
            session = self.sync_session_factory()
            try:
                session.execute(text(query), parameters or ())
                session.commit()
            finally:
                session.close()
        except Exception as e:
            logger.error(f"SQLite execute failed: {e}")
            raise

    async def _execute_postgresql(self, query: str, parameters: Optional[tuple] = None) -> None:
        """Execute query in PostgreSQL."""
        try:
            async with self.async_session_factory() as session:
                await session.execute(text(query), parameters or ())
                await session.commit()
        except Exception as e:
            logger.error(f"PostgreSQL execute failed: {e}")
            raise


# Global unified database manager instance
_unified_db_manager: Optional[UnifiedDatabaseManager] = None


async def get_unified_database() -> UnifiedDatabaseManager:
    """
    Get the global unified database manager instance.

    Returns:
        UnifiedDatabaseManager instance
    """
    global _unified_db_manager

    if _unified_db_manager is None:
        _unified_db_manager = UnifiedDatabaseManager()
        await _unified_db_manager.initialize()

    return _unified_db_manager


def get_unified_database_sync() -> UnifiedDatabaseManager:
    """
    Get the global unified database manager instance (synchronous).

    Returns:
        UnifiedDatabaseManager instance
    """
    global _unified_db_manager

    if _unified_db_manager is None:
        _unified_db_manager = UnifiedDatabaseManager()
        # Initialize synchronously - handle event loop conflicts
        try:
            loop = asyncio.get_running_loop()
            # If we're in an event loop, use threading approach
            import threading
            import concurrent.futures
            
            def init_db():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(_unified_db_manager.initialize())
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(init_db)
                result = future.result(timeout=30)
                if not result:
                    raise DatabaseError("Database initialization failed")
        except RuntimeError:
            # No event loop running, can use asyncio.run()
            asyncio.run(_unified_db_manager.initialize())

    return _unified_db_manager


async def close_unified_database():
    """Close the global unified database manager."""
    global _unified_db_manager

    if _unified_db_manager:
        await _unified_db_manager.close()
        _unified_db_manager = None
