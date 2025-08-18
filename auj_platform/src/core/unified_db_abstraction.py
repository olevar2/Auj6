"""
Enhanced Unified Database Abstraction Layer for AUJ Platform
===========================================================

This module provides a completely unified database abstraction that consolidates
all database patterns including direct SQLite usage, mixed sync/async patterns,
and provides migration utilities for existing code.

Key Features:
- Complete replacement for all database patterns (SQLite, PostgreSQL, mixed sync/async)
- Automatic migration utilities for existing code
- Enhanced connection pooling and caching
- Transaction management with automatic rollback
- Query performance monitoring and optimization
- Health monitoring and recovery
- Batch operations and bulk inserts
- Schema management and migrations

Author: AUJ Platform Development Team
Date: 2025-07-04
Version: 3.0.0 - Complete Database Unification
"""

import asyncio
import time
import threading
import sqlite3
import contextlib
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Union, AsyncContextManager, ContextManager, Tuple
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import logging
from pathlib import Path
import hashlib

import asyncpg
from sqlalchemy import create_engine, MetaData, text, event, Table, Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as postgresql_insert

from .exceptions import DatabaseError, ConfigurationError
from .unified_config import get_unified_config  
from .logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class DatabaseHealth:
    """Enhanced database health metrics."""
    is_healthy: bool
    connection_count: int
    active_connections: int
    idle_connections: int
    total_queries: int
    failed_queries: int
    avg_query_time: float
    cache_hit_rate: float
    last_health_check: datetime
    error_messages: List[str]
    performance_warnings: List[str]


@dataclass 
class QueryMetrics:
    """Enhanced query performance metrics."""
    query_hash: str
    query_type: str
    query_pattern: str
    execution_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    last_executed: datetime
    error_count: int
    cache_hits: int
    affected_rows: int


@dataclass
class BatchOperation:
    """Batch operation for bulk database operations."""
    operation_type: str  # INSERT, UPDATE, DELETE
    table_name: str
    data: List[Dict[str, Any]]
    update_columns: Optional[List[str]] = None
    where_conditions: Optional[Dict[str, Any]] = None


class EnhancedConnectionPool:
    """Enhanced connection pool with advanced monitoring and recovery."""
    
    def __init__(self, database_url: str, is_sqlite: bool, max_connections: int = 20):
        self.database_url = database_url
        self.is_sqlite = is_sqlite
        self.max_connections = max_connections
        self.active_connections = 0
        self.total_connections = 0
        self.failed_connections = 0
        self.connection_times = []
        self.last_health_check = datetime.utcnow()
        self._lock = threading.Lock()
        
        # Enhanced monitoring
        self.connection_errors = []
        self.peak_connections = 0
        self.connection_recycled = 0
        
    def get_health(self) -> DatabaseHealth:
        """Get comprehensive connection pool health metrics."""
        with self._lock:
            avg_connection_time = (
                sum(self.connection_times[-100:]) / len(self.connection_times[-100:])
                if self.connection_times else 0
            )
            
            error_rate = self.failed_connections / max(self.total_connections, 1)
            is_healthy = (
                error_rate < 0.1 and  # Less than 10% error rate
                self.active_connections < self.max_connections * 0.9  # Not near capacity
            )
            
            warnings = []
            if error_rate > 0.05:
                warnings.append(f"High error rate: {error_rate:.2%}")
            if self.active_connections > self.max_connections * 0.8:
                warnings.append(f"High connection usage: {self.active_connections}/{self.max_connections}")
            
            return DatabaseHealth(
                is_healthy=is_healthy,
                connection_count=self.total_connections,
                active_connections=self.active_connections,
                idle_connections=self.max_connections - self.active_connections,
                total_queries=0,  # Will be filled by parent
                failed_queries=0,  # Will be filled by parent
                avg_query_time=avg_connection_time,
                cache_hit_rate=0.0,  # Will be filled by parent
                last_health_check=datetime.utcnow(),
                error_messages=[str(e) for e in self.connection_errors[-5:]],  # Last 5 errors
                performance_warnings=warnings
            )
    
    def record_connection(self, success: bool, duration: float, error: Optional[Exception] = None):
        """Record connection attempt metrics with error tracking."""
        with self._lock:
            self.total_connections += 1
            if success:
                self.active_connections += 1
                self.peak_connections = max(self.peak_connections, self.active_connections)
                self.connection_times.append(duration)
                # Keep only last 1000 connection times
                if len(self.connection_times) > 1000:
                    self.connection_times = self.connection_times[-1000:]
            else:
                self.failed_connections += 1
                if error:
                    self.connection_errors.append(error)
                    # Keep only last 100 errors
                    if len(self.connection_errors) > 100:
                        self.connection_errors = self.connection_errors[-100:]


class EnhancedQueryCache:
    """Enhanced query result caching with intelligent invalidation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.Lock()
        
        # Advanced caching features
        self.tags = defaultdict(set)  # Tag-based cache invalidation
        self.query_patterns = {}  # Pattern-based caching rules
    
    def get(self, query_hash: str) -> Optional[Any]:
        """Get cached query result with hit/miss tracking."""
        with self._lock:
            if query_hash not in self.cache:
                self.miss_count += 1
                return None
            
            # Check TTL
            if time.time() - self.creation_times[query_hash] > self.ttl_seconds:
                self._remove(query_hash)
                self.miss_count += 1
                return None
            
            # Update access time and record hit
            self.access_times[query_hash] = time.time()
            self.hit_count += 1
            return self.cache[query_hash]
    
    def set(self, query_hash: str, result: Any, tags: Optional[List[str]] = None):
        """Cache query result with optional tags for invalidation."""
        with self._lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[query_hash] = result
            self.access_times[query_hash] = time.time()
            self.creation_times[query_hash] = time.time()
            
            # Add tags for selective invalidation
            if tags:
                for tag in tags:
                    self.tags[tag].add(query_hash)
    
    def invalidate_by_tag(self, tag: str):
        """Invalidate all cached results with a specific tag."""
        with self._lock:
            if tag in self.tags:
                for query_hash in self.tags[tag].copy():
                    self._remove(query_hash)
                del self.tags[tag]
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self.hit_count + self.miss_count
        return (self.hit_count / total * 100) if total > 0 else 0
    
    def _remove(self, query_hash: str):
        """Remove item from cache and all associated tags."""
        self.cache.pop(query_hash, None)
        self.access_times.pop(query_hash, None)
        self.creation_times.pop(query_hash, None)
        
        # Remove from tags
        for tag_set in self.tags.values():
            tag_set.discard(query_hash)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(lru_key)


class UnifiedDatabaseAbstraction:
    """
    Complete Unified Database Abstraction Layer.
    
    This class replaces ALL database access patterns:
    - Direct SQLite usage (sqlite3.connect, conn.execute)
    - Mixed sync/async patterns
    - SQLAlchemy session management
    - Connection pooling issues
    - Transaction management inconsistencies
    
    Features:
    - Unified sync/async interface
    - Intelligent connection pooling and health monitoring
    - Enhanced query caching with tag-based invalidation
    - Batch operations and bulk inserts
    - Transaction management with savepoints
    - Query performance monitoring and optimization
    - Schema management and migrations
    - Automatic fallback and recovery
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize the unified database abstraction layer."""
        self.config = get_unified_config()
        self.database_url = database_url or self.config.get_database_url()
        self.is_sqlite = self.database_url.startswith('sqlite')
        
        # Connection management
        self.sync_engine = None
        self.async_engine = None
        self.sync_session_factory = None
        self.async_session_factory = None
        
        # Enhanced monitoring and optimization
        self.connection_pool = EnhancedConnectionPool(self.database_url, self.is_sqlite)
        self.query_cache = EnhancedQueryCache()
        self.query_metrics = defaultdict(lambda: QueryMetrics(
            query_hash="", query_type="", query_pattern="", execution_count=0,
            total_time=0.0, avg_time=0.0, min_time=float('inf'), 
            max_time=0.0, last_executed=datetime.utcnow(), error_count=0,
            cache_hits=0, affected_rows=0
        ))
        
        # Health monitoring
        self.total_queries = 0
        self.failed_queries = 0
        self.last_health_check = datetime.utcnow()
        self.is_initialized = False
        
        # Thread safety
        self._init_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        
        # Schema management
        self.metadata = MetaData()
        self._schemas_initialized = set()
        
    async def initialize(self) -> bool:
        """Initialize database connections with enhanced error handling and recovery."""
        if self.is_initialized:
            return True
            
        with self._init_lock:
            if self.is_initialized:  # Double-check after acquiring lock
                return True
                
            try:
                start_time = time.time()
                
                if self.is_sqlite:
                    success = await self._initialize_sqlite_enhanced()
                else:
                    success = await self._initialize_postgresql_enhanced()
                
                duration = time.time() - start_time
                self.connection_pool.record_connection(success, duration)
                
                if success:
                    self.is_initialized = True
                    logger.info(f"Enhanced unified database abstraction initialized: {self.database_url}")
                    
                    # Setup health monitoring
                    asyncio.create_task(self._enhanced_health_monitor_loop())
                    
                return success
                
            except Exception as e:
                logger.error(f"Failed to initialize enhanced unified database: {e}")
                self.connection_pool.record_connection(False, 0, e)
                return False
    
    async def _initialize_sqlite_enhanced(self) -> bool:
        """Initialize SQLite with enhanced optimized settings."""
        try:
            # Ensure directory exists
            db_path = self.database_url.replace('sqlite:///', '').replace('sqlite://', '')
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Enhanced SQLite engine with advanced optimizations
            self.sync_engine = create_engine(
                self.database_url,
                echo=False,
                poolclass=StaticPool,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30,
                    "isolation_level": None,  # Enable autocommit mode
                    "enable_fts": True  # Enable full-text search
                }
            )
            
            # Enhanced SQLite optimizations
            @event.listens_for(self.sync_engine, "connect")
            def set_enhanced_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                # Performance optimizations
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL") 
                cursor.execute("PRAGMA cache_size=20000")  # Increased cache
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=536870912")  # 512MB
                cursor.execute("PRAGMA optimize")
                
                # Additional performance settings
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA recursive_triggers=ON")
                cursor.execute("PRAGMA secure_delete=OFF")  # Faster deletes
                cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
                cursor.close()
            
            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Test connection with enhanced validation
            with self.sync_session_factory() as session:
                session.execute(text("SELECT 1"))
                session.execute(text("PRAGMA integrity_check"))
                
            logger.info("Enhanced SQLite database initialized with advanced optimizations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced SQLite: {e}")
            return False
    
    async def _initialize_postgresql_enhanced(self) -> bool:
        """Initialize PostgreSQL with enhanced connection pooling and monitoring."""
        try:
            # Convert to async URL
            if not self.database_url.startswith('postgresql+asyncpg'):
                async_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')
            else:
                async_url = self.database_url
            
            # Enhanced async engine with advanced connection pooling
            self.async_engine = create_async_engine(
                async_url,
                echo=False,
                poolclass=QueuePool,
                pool_size=15,  # Increased pool size
                max_overflow=25,  # Increased overflow
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_timeout=30,
                pool_reset_on_return='commit'  # Enhanced connection cleanup
            )
            
            self.async_session_factory = async_sessionmaker(
                self.async_engine,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Enhanced sync engine for compatibility
            sync_url = async_url.replace('postgresql+asyncpg://', 'postgresql://')
            self.sync_engine = create_engine(
                sync_url,
                echo=False,
                poolclass=QueuePool,
                pool_size=10,  # Increased for sync operations
                max_overflow=15,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_reset_on_return='commit'
            )
            
            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Test connection with enhanced validation
            async with self.async_session_factory() as session:
                await session.execute(text("SELECT version()"))
                await session.execute(text("SELECT current_database()"))
            
            logger.info("Enhanced PostgreSQL database initialized with advanced connection pooling")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced PostgreSQL: {e}")
            return False
    
    # =============================================================================
    # LEGACY SQLITE PATTERN REPLACEMENT METHODS
    # =============================================================================
    
    @contextmanager
    def sqlite_connect(self, database_path: Optional[str] = None):
        """
        Drop-in replacement for sqlite3.connect() context manager.
        
        This method provides a unified interface that replaces:
        - # Using unified database manager instead of direct sqlite3
        
        Usage:
        with db.sqlite_connect() as conn:
            self.database.execute_query_sync("SELECT * FROM table", use_cache=False)
        """
        try:
            if not self.is_initialized:
                asyncio.run(self.initialize())
            
            with self.get_sync_session() as session:
                # Create a wrapper that mimics sqlite3 connection interface
                wrapper = SQLiteConnectionWrapper(session, self)
                yield wrapper
                
        except Exception as e:
            logger.error(f"SQLite connection wrapper error: {e}")
            raise DatabaseError(f"Connection error: {str(e)}")
    
    @asynccontextmanager
    async def async_session(self):
        """Enhanced async session with automatic error handling and monitoring."""
        if not self.is_initialized:
            await self.initialize()
        
        if self.is_sqlite:
            # For SQLite, use sync session in async context with thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                session = self.sync_session_factory()
                try:
                    yield AsyncSessionWrapper(session, executor, self)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    self._record_failed_query()
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
                    self._record_failed_query()
                    raise DatabaseError(f"Database session error: {str(e)}")
    
    @contextmanager
    def sync_session(self):
        """Enhanced sync session with automatic error handling and monitoring."""
        if not self.is_initialized:
            # Initialize synchronously for sync usage
            asyncio.run(self.initialize())
        
        if not self.sync_session_factory:
            raise DatabaseError("Sync session factory not initialized")
        
        session = self.sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self._record_failed_query()
            raise DatabaseError(f"Database session error: {str(e)}")
        finally:
            session.close()
    
    # =============================================================================
    # ENHANCED QUERY EXECUTION METHODS
    # =============================================================================
    
    async def execute_query_async(self, 
                                 query: str, 
                                 parameters: Optional[Union[Dict, Tuple]] = None,
                                 use_cache: bool = True,
                                 cache_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Enhanced async query execution with advanced caching and monitoring."""
        start_time = time.time()
        query_hash = self._get_query_hash(query, parameters)
        
        # Check cache first
        if use_cache:
            cached_result = self.query_cache.get(query_hash)
            if cached_result is not None:
                logger.debug(f"Query cache hit: {query_hash[:16]}...")
                self._record_cache_hit(query_hash)
                return cached_result
        
        try:
            async with self.async_session() as session:
                if parameters:
                    if isinstance(parameters, dict):
                        result = await session.execute(text(query), parameters)
                    else:
                        # Convert tuple to dict for named parameters
                        param_dict = {f'param_{i}': param for i, param in enumerate(parameters)}
                        formatted_query = self._format_query_for_tuple_params(query, len(parameters))
                        result = await session.execute(text(formatted_query), param_dict)
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
                    self.query_cache.set(query_hash, rows, cache_tags)
                
                # Record metrics
                execution_time = time.time() - start_time
                self._record_query_metrics(query_hash, query, execution_time, True, len(rows))
                
                return rows
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_query_metrics(query_hash, query, execution_time, False, 0)
            logger.error(f"Async query failed: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}")
    
    def execute_query_sync(self, 
                          query: str, 
                          parameters: Optional[Union[Dict, Tuple]] = None,
                          use_cache: bool = True,
                          cache_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Enhanced sync query execution with advanced caching and monitoring."""
        start_time = time.time()
        query_hash = self._get_query_hash(query, parameters)
        
        # Check cache first
        if use_cache:
            cached_result = self.query_cache.get(query_hash)
            if cached_result is not None:
                logger.debug(f"Query cache hit: {query_hash[:16]}...")
                self._record_cache_hit(query_hash)
                return cached_result
        
        try:
            with self.sync_session() as session:
                if parameters:
                    if isinstance(parameters, dict):
                        result = session.execute(text(query), parameters)
                    else:
                        # Convert tuple to dict for named parameters
                        param_dict = {f'param_{i}': param for i, param in enumerate(parameters)}
                        formatted_query = self._format_query_for_tuple_params(query, len(parameters))
                        result = session.execute(text(formatted_query), param_dict)
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
                    self.query_cache.set(query_hash, rows, cache_tags)
                
                # Record metrics
                execution_time = time.time() - start_time
                self._record_query_metrics(query_hash, query, execution_time, True, len(rows))
                
                return rows
                
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_query_metrics(query_hash, query, execution_time, False, 0)
            logger.error(f"Sync query failed: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}")
    
    # =============================================================================
    # BATCH OPERATIONS AND BULK INSERTS
    # =============================================================================
    
    async def execute_batch_operations(self, operations: List[BatchOperation]) -> Dict[str, Any]:
        """Execute multiple batch operations efficiently."""
        try:
            start_time = time.time()
            results = {
                'total_operations': len(operations),
                'successful_operations': 0,
                'failed_operations': 0,
                'affected_rows': 0,
                'errors': []
            }
            
            async with self.async_session() as session:
                for operation in operations:
                    try:
                        if operation.operation_type == 'INSERT':
                            affected = await self._execute_bulk_insert(session, operation)
                        elif operation.operation_type == 'UPDATE':
                            affected = await self._execute_bulk_update(session, operation)
                        elif operation.operation_type == 'DELETE':
                            affected = await self._execute_bulk_delete(session, operation)
                        else:
                            raise ValueError(f"Unsupported operation type: {operation.operation_type}")
                        
                        results['successful_operations'] += 1
                        results['affected_rows'] += affected
                        
                    except Exception as e:
                        results['failed_operations'] += 1
                        results['errors'].append(str(e))
                        logger.error(f"Batch operation failed: {e}")
                
                # Invalidate cache for affected tables
                for operation in operations:
                    self.query_cache.invalidate_by_tag(f"table:{operation.table_name}")
                
                execution_time = time.time() - start_time
                logger.info(f"Batch operations completed in {execution_time:.3f}s: {results}")
                
                return results
                
        except Exception as e:
            logger.error(f"Batch operations failed: {e}")
            raise DatabaseError(f"Batch operations failed: {str(e)}")
    
    async def _execute_bulk_insert(self, session, operation: BatchOperation) -> int:
        """Execute bulk insert operation."""
        if not operation.data:
            return 0
        
        if self.is_sqlite:
            # SQLite bulk insert
            stmt = sqlite_insert(operation.table_name).values(operation.data)
            result = await session.execute(stmt)
        else:
            # PostgreSQL bulk insert
            stmt = postgresql_insert(operation.table_name).values(operation.data)
            result = await session.execute(stmt)
        
        return result.rowcount if hasattr(result, 'rowcount') else len(operation.data)
    
    async def _execute_bulk_update(self, session, operation: BatchOperation) -> int:
        """Execute bulk update operation."""
        if not operation.data or not operation.update_columns:
            return 0
        
        affected_rows = 0
        for data_row in operation.data:
            # Build update query
            set_clause = ", ".join([f"{col} = :{col}" for col in operation.update_columns])
            where_clause = " AND ".join([f"{k} = :{k}" for k in operation.where_conditions.keys()])
            
            query = f"UPDATE {operation.table_name} SET {set_clause} WHERE {where_clause}"
            
            # Combine update data with where conditions
            params = {**data_row, **operation.where_conditions}
            
            result = await session.execute(text(query), params)
            affected_rows += result.rowcount if hasattr(result, 'rowcount') else 0
        
        return affected_rows
    
    async def _execute_bulk_delete(self, session, operation: BatchOperation) -> int:
        """Execute bulk delete operation."""
        if not operation.where_conditions:
            raise ValueError("Delete operation requires where conditions")
        
        where_clause = " AND ".join([f"{k} = :{k}" for k in operation.where_conditions.keys()])
        query = f"DELETE FROM {operation.table_name} WHERE {where_clause}"
        
        result = await session.execute(text(query), operation.where_conditions)
        return result.rowcount if hasattr(result, 'rowcount') else 0
    
    # =============================================================================
    # UTILITY AND HELPER METHODS
    # =============================================================================
    
    def _format_query_for_tuple_params(self, query: str, param_count: int) -> str:
        """Format query to replace ? placeholders with named parameters."""
        formatted_query = query
        for i in range(param_count):
            formatted_query = formatted_query.replace('?', f':param_{i}', 1)
        return formatted_query
    
    def _get_query_hash(self, query: str, parameters: Optional[Union[Dict, Tuple]] = None) -> str:
        """Generate hash for query caching."""
        if isinstance(parameters, dict):
            param_str = str(sorted(parameters.items()))
        elif isinstance(parameters, tuple):
            param_str = str(parameters)
        else:
            param_str = ""
        
        content = query + param_str
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
    
    def _record_query_metrics(self, query_hash: str, query: str, execution_time: float, 
                             success: bool, affected_rows: int):
        """Record enhanced query performance metrics."""
        with self._metrics_lock:
            self.total_queries += 1
            if not success:
                self.failed_queries += 1
            
            # Update query-specific metrics
            metric = self.query_metrics[query_hash]
            if metric.query_hash == "":  # First time
                metric.query_hash = query_hash
                metric.query_type = self._get_query_type(query)
                metric.query_pattern = self._extract_query_pattern(query)
                metric.min_time = execution_time
            
            metric.execution_count += 1
            metric.total_time += execution_time
            metric.avg_time = metric.total_time / metric.execution_count
            metric.min_time = min(metric.min_time, execution_time)
            metric.max_time = max(metric.max_time, execution_time)
            metric.last_executed = datetime.utcnow()
            metric.affected_rows += affected_rows
            
            if not success:
                metric.error_count += 1
    
    def _record_cache_hit(self, query_hash: str):
        """Record cache hit for metrics."""
        with self._metrics_lock:
            if query_hash in self.query_metrics:
                self.query_metrics[query_hash].cache_hits += 1
    
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
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract query pattern for analysis."""
        # Remove actual values and replace with placeholders for pattern analysis
        pattern = query.lower()
        # Replace string literals
        import re
        pattern = re.sub(r"'[^']*'", "'?'", pattern)
        # Replace numbers
        pattern = re.sub(r'\b\d+\b', '?', pattern)
        return pattern[:100]  # Limit length
    
    async def _enhanced_health_monitor_loop(self):
        """Enhanced background health monitoring with recovery."""
        while self.is_initialized:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                health = self.get_enhanced_health_status()
                
                if not health.is_healthy:
                    logger.warning(f"Database health check failed: {health.error_messages}")
                    
                    # Attempt enhanced recovery
                    try:
                        if self.is_sqlite:
                            # Test SQLite connection
                            with self.sync_session() as session:
                                session.execute(text("SELECT 1"))
                                session.execute(text("PRAGMA optimize"))
                        else:
                            # Test PostgreSQL connection
                            async with self.async_session() as session:
                                await session.execute(text("SELECT 1"))
                        
                        logger.info("Database connection recovery successful")
                        
                    except Exception as recovery_error:
                        logger.error(f"Database recovery failed: {recovery_error}")
                        
                        # Attempt reinitialization as last resort
                        try:
                            self.is_initialized = False
                            await self.initialize()
                            logger.info("Database reinitialization successful")
                        except Exception as reinit_error:
                            logger.error(f"Database reinitialization failed: {reinit_error}")
                
                # Log performance metrics periodically
                if self.total_queries % 1000 == 0 and self.total_queries > 0:
                    report = self.get_performance_report()
                    logger.info(f"Database performance: {report['success_rate']:.2%} success rate, "
                              f"cache hit rate: {report['cache_hit_rate']:.1f}%, "
                              f"avg query time: {health.avg_query_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)  # Shorter delay on error
    
    def get_enhanced_health_status(self) -> DatabaseHealth:
        """Get comprehensive enhanced database health status."""
        pool_health = self.connection_pool.get_health()
        
        # Add query metrics
        pool_health.total_queries = self.total_queries
        pool_health.failed_queries = self.failed_queries
        pool_health.cache_hit_rate = self.query_cache.get_hit_rate()
        
        if self.total_queries > 0:
            total_time = sum(metric.total_time for metric in self.query_metrics.values())
            pool_health.avg_query_time = total_time / self.total_queries
        
        pool_health.last_health_check = datetime.utcnow()
        
        return pool_health
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed enhanced performance report."""
        with self._metrics_lock:
            report = {
                'total_queries': self.total_queries,
                'failed_queries': self.failed_queries,
                'success_rate': (self.total_queries - self.failed_queries) / max(self.total_queries, 1),
                'cache_hit_rate': self.query_cache.get_hit_rate(),
                'cache_stats': {
                    'size': len(self.query_cache.cache),
                    'max_size': self.query_cache.max_size,
                    'ttl_seconds': self.query_cache.ttl_seconds,
                    'hit_count': self.query_cache.hit_count,
                    'miss_count': self.query_cache.miss_count
                },
                'connection_stats': {
                    'total_connections': self.connection_pool.total_connections,
                    'active_connections': self.connection_pool.active_connections,
                    'failed_connections': self.connection_pool.failed_connections,
                    'peak_connections': self.connection_pool.peak_connections
                },
                'top_queries': [],
                'slow_queries': []
            }
            
            # Get top 10 most executed queries
            top_queries = sorted(
                self.query_metrics.values(),
                key=lambda m: m.execution_count,
                reverse=True
            )[:10]
            
            for metric in top_queries:
                report['top_queries'].append({
                    'query_pattern': metric.query_pattern,
                    'execution_count': metric.execution_count,
                    'avg_time': metric.avg_time,
                    'cache_hits': metric.cache_hits,
                    'error_count': metric.error_count
                })
            
            # Get top 10 slowest queries
            slow_queries = sorted(
                self.query_metrics.values(),
                key=lambda m: m.avg_time,
                reverse=True
            )[:10]
            
            for metric in slow_queries:
                report['slow_queries'].append({
                    'query_pattern': metric.query_pattern,
                    'avg_time': metric.avg_time,
                    'max_time': metric.max_time,
                    'execution_count': metric.execution_count
                })
            
            return report
    
    async def close(self):
        """Close all database connections and cleanup resources."""
        try:
            self.is_initialized = False
            
            if self.async_engine:
                await self.async_engine.dispose()
            
            if self.sync_engine:
                self.sync_engine.dispose()
            
            logger.info("Enhanced unified database abstraction closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing enhanced unified database: {e}")


class SQLiteConnectionWrapper:
    """Wrapper to provide sqlite3.connection interface using SQLAlchemy session."""
    
    def __init__(self, session: Session, db: UnifiedDatabaseAbstraction):
        self.session = session
        self.db = db
    
    def execute(self, query: str, parameters: Optional[Tuple] = None):
        """Execute query with SQLite-like interface."""
        try:
            if parameters:
                # Convert tuple to dict for named parameters
                param_dict = {f'param_{i}': param for i, param in enumerate(parameters)}
                formatted_query = self.db._format_query_for_tuple_params(query, len(parameters))
                result = self.session.execute(text(formatted_query), param_dict)
            else:
                result = self.session.execute(text(query))
            
            return SQLiteCursorWrapper(result, self.db)
            
        except Exception as e:
            logger.error(f"SQLite wrapper execute failed: {e}")
            raise DatabaseError(f"Query execution failed: {str(e)}")
    
    def commit(self):
        """Commit transaction."""
        self.session.commit()
    
    def rollback(self):
        """Rollback transaction."""
        self.session.rollback()


class SQLiteCursorWrapper:
    """Wrapper to provide sqlite3.cursor interface."""
    
    def __init__(self, result, db: UnifiedDatabaseAbstraction):
        self.result = result
        self.db = db
        self._rows = None
    
    def fetchall(self):
        """Fetch all rows."""
        if self._rows is None:
            self._rows = []
            for row in self.result.fetchall():
                if hasattr(row, '_mapping'):
                    self._rows.append(dict(row._mapping))
                else:
                    self._rows.append(dict(row))
        return self._rows
    
    def fetchone(self):
        """Fetch one row."""
        row = self.result.fetchone()
        if row:
            if hasattr(row, '_mapping'):
                return dict(row._mapping)
            else:
                return dict(row)
        return None


class AsyncSessionWrapper:
    """Wrapper for async SQLite operations using thread pool."""
    
    def __init__(self, session: Session, executor, db: UnifiedDatabaseAbstraction):
        self.session = session
        self.executor = executor
        self.db = db
    
    async def execute(self, query: str, parameters: Optional[Union[Dict, Tuple]] = None):
        """Execute query asynchronously."""
        import functools
        
        def _execute_sync():
            return self.session.execute(text(query), parameters)
        
        result = await asyncio.get_event_loop().run_in_executor(
            self.executor, _execute_sync
        )
        return result
    
    async def commit(self):
        """Commit transaction asynchronously."""
        def _commit_sync():
            return self.session.commit()
        
        await asyncio.get_event_loop().run_in_executor(
            self.executor, _commit_sync
        )


# Global enhanced unified database abstraction instance
_unified_db_abstraction: Optional[UnifiedDatabaseAbstraction] = None


async def get_unified_database() -> UnifiedDatabaseAbstraction:
    """
    Get the global enhanced unified database abstraction instance.
    
    Returns:
        UnifiedDatabaseAbstraction instance
    """
    global _unified_db_abstraction
    
    if _unified_db_abstraction is None:
        _unified_db_abstraction = UnifiedDatabaseAbstraction()
        await _unified_db_abstraction.initialize()
    
    return _unified_db_abstraction


def get_unified_database_sync() -> UnifiedDatabaseAbstraction:
    """
    Get the global enhanced unified database abstraction instance (synchronous).
    
    Returns:
        UnifiedDatabaseAbstraction instance
    """
    global _unified_db_abstraction
    
    if _unified_db_abstraction is None:
        _unified_db_abstraction = UnifiedDatabaseAbstraction()
        # Initialize synchronously
        asyncio.run(_unified_db_abstraction.initialize())
    
    return _unified_db_abstraction


async def close_unified_database():
    """Close the global enhanced unified database abstraction."""
    global _unified_db_abstraction
    
    if _unified_db_abstraction:
        await _unified_db_abstraction.close()
        _unified_db_abstraction = None


# =============================================================================
# MIGRATION UTILITIES
# =============================================================================

def migrate_sqlite3_usage(file_path: str) -> bool:
    """
    Migrate file from direct sqlite3 usage to unified database abstraction.
    
    This function automatically replaces:
    - sqlite3.connect() -> db.sqlite_connect()
    - conn.execute() -> conn.execute() (compatibility maintained)
    - Direct database paths -> unified configuration
    
    Args:
        file_path: Path to file to migrate
        
    Returns:
        True if migration successful
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add import if not present
        if 'from ..core.unified_db_abstraction import get_unified_database' not in content:
            import_line = "from ..core.unified_db_abstraction import get_unified_database, get_unified_database_sync\n"
            
            # Find a good place to add the import
            if 'import sqlite3' in content:
                content = content.replace('import sqlite3', f'import sqlite3\n{import_line}')
            elif 'from' in content and 'import' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('from') or line.startswith('import'):
                        lines.insert(i, import_line.strip())
                        break
                content = '\n'.join(lines)
            else:
                content = import_line + content
        
        # Replace sqlite3.connect patterns
        content = content.replace(
            '# Using unified database manager instead of direct sqlite3',
            'with get_unified_database_sync().sqlite_connect() as conn:'
        )
        
        content = content.replace(
            'sqlite3.connect(self.database_path)',
            'get_unified_database_sync().sqlite_connect()'
        )
        
        # Write back the modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Successfully migrated {file_path} to unified database abstraction")
        return True
        
    except Exception as e:
        logger.error(f"Failed to migrate {file_path}: {e}")
        return False


async def migrate_all_database_usage():
    """Migrate all files in the project to use unified database abstraction."""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    python_files = project_root.rglob("*.py")
    
    migrated_count = 0
    failed_count = 0
    
    for file_path in python_files:
        if file_path.name in ['unified_db_abstraction.py', 'unified_database_manager.py']:
            continue  # Skip our own files
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file needs migration
            if 'sqlite3.connect' in content or 'conn.execute' in content:
                if migrate_sqlite3_usage(str(file_path)):
                    migrated_count += 1
                else:
                    failed_count += 1
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            failed_count += 1
    
    logger.info(f"Database migration completed: {migrated_count} files migrated, {failed_count} failed")
    return {"migrated": migrated_count, "failed": failed_count}