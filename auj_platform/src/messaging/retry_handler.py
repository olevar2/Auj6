#!/usr/bin/env python3
"""
Retry Handler for AUJ Platform
==============================

Advanced retry logic with exponential backoff, circuit breaker pattern,
and sophisticated failure handling for message processing.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import random
import json

from ..core.unified_config import get_unified_config


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    JITTERED_BACKOFF = "jittered_backoff"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit open, calls fail fast
    HALF_OPEN = "half_open"  # Test mode, limited calls allowed


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Circuit breaker configuration
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 30.0  # Time before trying half-open
    success_threshold: int = 3  # Successes needed to close circuit


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    timestamp: datetime
    delay: float
    error: Optional[str] = None
    success: bool = False


@dataclass
class RetryResult:
    """Result of retry operation."""
    success: bool
    attempts: List[RetryAttempt] = field(default_factory=list)
    total_time: float = 0.0
    final_error: Optional[str] = None
    circuit_breaker_triggered: bool = False


class CircuitBreaker:
    """Circuit breaker implementation for failure protection."""
    
    def __init__(self, config: RetryConfig):
        """Initialize circuit breaker."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_opens': 0,
            'circuit_closes': 0
        }
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        self.stats['total_calls'] += 1
        
        # Check circuit state
        await self._check_circuit_state()
        
        if self.state == CircuitState.OPEN:
            self.logger.warning("⚡ Circuit breaker OPEN - call rejected")
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Record success
            await self._record_success()
            self.stats['successful_calls'] += 1
            
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure()
            self.stats['failed_calls'] += 1
            raise
    
    async def _check_circuit_state(self):
        """Check and update circuit state."""
        now = datetime.now()
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                now - self.last_failure_time > timedelta(seconds=self.config.recovery_timeout)):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info("🔄 Circuit breaker moved to HALF_OPEN")
    
    async def _record_success(self):
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.stats['circuit_closes'] += 1
                self.logger.info("✅ Circuit breaker CLOSED")
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    async def _record_failure(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if (self.state == CircuitState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self.state = CircuitState.OPEN
            self.stats['circuit_opens'] += 1
            self.logger.warning("❌ Circuit breaker OPENED")
        
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.logger.warning("❌ Circuit breaker OPENED from HALF_OPEN")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            **self.stats,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryHandler:
    """
    Advanced retry handler with multiple strategies and circuit breaker.
    
    Provides sophisticated retry logic for the AUJ platform's message
    processing and external service calls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize retry handler."""
        self.config = config
        self.config_manager = get_unified_config()
        self.logger = logging.getLogger(__name__)
        
        # Retry configuration
        retry_config_data = self.config_manager.get_dict('retry_handler', {})
        self.default_config = RetryConfig(
            max_attempts=retry_config_data.get('max_attempts', 3),
            initial_delay=retry_config_data.get('initial_delay', 1.0),
            max_delay=retry_config_data.get('max_delay', 60.0),
            backoff_multiplier=retry_config_data.get('backoff_multiplier', 2.0),
            jitter_range=retry_config_data.get('jitter_range', 0.1),
            strategy=RetryStrategy(retry_config_data.get('strategy', 'exponential_backoff')),
            failure_threshold=retry_config_data.get('failure_threshold', 5),
            recovery_timeout=retry_config_data.get('recovery_timeout', 30.0),
            success_threshold=retry_config_data.get('success_threshold', 3)
        )
        
        # Circuit breakers for different operations
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Retry statistics
        self.stats = {
            'total_retries': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'circuit_breaker_trips': 0
        }
        
        self.logger.info("🔄 RetryHandler initialized")
    
    def get_circuit_breaker(self, operation_key: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation_key not in self.circuit_breakers:
            self.circuit_breakers[operation_key] = CircuitBreaker(self.default_config)
        return self.circuit_breakers[operation_key]
    
    async def retry_async(
        self,
        func: Callable,
        *args,
        operation_key: str = "default",
        config: Optional[RetryConfig] = None,
        **kwargs
    ) -> RetryResult:
        """
        Execute function with retry logic and circuit breaker.
        
        Args:
            func: Async function to execute
            operation_key: Key for circuit breaker grouping
            config: Custom retry configuration
            *args, **kwargs: Arguments for the function
            
        Returns:
            RetryResult with attempt details
        """
        retry_config = config or self.default_config
        circuit_breaker = self.get_circuit_breaker(operation_key)
        
        start_time = datetime.now()
        attempts = []
        
        self.stats['total_retries'] += 1
        
        for attempt in range(retry_config.max_attempts):
            attempt_start = datetime.now()
            
            try:
                # Calculate delay for this attempt
                if attempt > 0:
                    delay = self._calculate_delay(attempt, retry_config)
                    
                    self.logger.debug(f"🔄 Retry attempt {attempt + 1} for {operation_key} "
                                    f"after {delay:.2f}s delay")
                    
                    await asyncio.sleep(delay)
                else:
                    delay = 0.0
                
                # Execute function with circuit breaker
                result = await circuit_breaker.call(func, *args, **kwargs)
                
                # Success
                attempt_info = RetryAttempt(
                    attempt_number=attempt + 1,
                    timestamp=attempt_start,
                    delay=delay,
                    success=True
                )
                attempts.append(attempt_info)
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                self.stats['successful_retries'] += 1
                
                return RetryResult(
                    success=True,
                    attempts=attempts,
                    total_time=total_time
                )
                
            except CircuitBreakerOpenError as e:
                # Circuit breaker is open
                attempt_info = RetryAttempt(
                    attempt_number=attempt + 1,
                    timestamp=attempt_start,
                    delay=delay if attempt > 0 else 0.0,
                    error=str(e),
                    success=False
                )
                attempts.append(attempt_info)
                
                self.stats['circuit_breaker_trips'] += 1
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                return RetryResult(
                    success=False,
                    attempts=attempts,
                    total_time=total_time,
                    final_error=str(e),
                    circuit_breaker_triggered=True
                )
                
            except Exception as e:
                # Function failed
                attempt_info = RetryAttempt(
                    attempt_number=attempt + 1,
                    timestamp=attempt_start,
                    delay=delay if attempt > 0 else 0.0,
                    error=str(e),
                    success=False
                )
                attempts.append(attempt_info)
                
                self.logger.warning(f"⚠️ Attempt {attempt + 1} failed for {operation_key}: {e}")
                
                # If this is the last attempt, return failure
                if attempt == retry_config.max_attempts - 1:
                    break
        
        # All attempts failed
        total_time = (datetime.now() - start_time).total_seconds()
        final_error = attempts[-1].error if attempts else "Unknown error"
        
        self.stats['failed_retries'] += 1
        
        return RetryResult(
            success=False,
            attempts=attempts,
            total_time=total_time,
            final_error=final_error
        )
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt based on strategy."""
        if config.strategy == RetryStrategy.FIXED_DELAY:
            base_delay = config.initial_delay
            
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            base_delay = config.initial_delay * attempt
            
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            base_delay = config.initial_delay * (config.backoff_multiplier ** (attempt - 1))
            
        elif config.strategy == RetryStrategy.JITTERED_BACKOFF:
            # Exponential backoff with jitter
            exp_delay = config.initial_delay * (config.backoff_multiplier ** (attempt - 1))
            jitter = random.uniform(-config.jitter_range, config.jitter_range) * exp_delay
            base_delay = exp_delay + jitter
            
        else:
            base_delay = config.initial_delay
        
        # Apply maximum delay limit
        return min(base_delay, config.max_delay)
    
    async def retry_with_conditions(
        self,
        func: Callable,
        should_retry: Callable[[Exception], bool],
        *args,
        operation_key: str = "conditional",
        config: Optional[RetryConfig] = None,
        **kwargs
    ) -> RetryResult:
        """
        Retry with custom conditions for when to retry.
        
        Args:
            func: Function to execute
            should_retry: Function that determines if error should trigger retry
            operation_key: Key for circuit breaker grouping
            config: Custom retry configuration
        """
        retry_config = config or self.default_config
        circuit_breaker = self.get_circuit_breaker(operation_key)
        
        start_time = datetime.now()
        attempts = []
        
        for attempt in range(retry_config.max_attempts):
            attempt_start = datetime.now()
            
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt, retry_config)
                    await asyncio.sleep(delay)
                else:
                    delay = 0.0
                
                result = await circuit_breaker.call(func, *args, **kwargs)
                
                # Success
                attempt_info = RetryAttempt(
                    attempt_number=attempt + 1,
                    timestamp=attempt_start,
                    delay=delay,
                    success=True
                )
                attempts.append(attempt_info)
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                return RetryResult(
                    success=True,
                    attempts=attempts,
                    total_time=total_time
                )
                
            except Exception as e:
                attempt_info = RetryAttempt(
                    attempt_number=attempt + 1,
                    timestamp=attempt_start,
                    delay=delay if attempt > 0 else 0.0,
                    error=str(e),
                    success=False
                )
                attempts.append(attempt_info)
                
                # Check if we should retry this error
                if not should_retry(e) or attempt == retry_config.max_attempts - 1:
                    break
        
        # Failed
        total_time = (datetime.now() - start_time).total_seconds()
        final_error = attempts[-1].error if attempts else "Unknown error"
        
        return RetryResult(
            success=False,
            attempts=attempts,
            total_time=total_time,
            final_error=final_error
        )
    
    async def bulk_retry(
        self,
        operations: List[Dict[str, Any]],
        max_concurrent: int = 10
    ) -> List[RetryResult]:
        """
        Execute multiple operations with retry logic concurrently.
        
        Args:
            operations: List of operation dictionaries with 'func' and 'args'
            max_concurrent: Maximum concurrent operations
            
        Returns:
            List of RetryResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_operation(operation: Dict[str, Any]) -> RetryResult:
            async with semaphore:
                func = operation['func']
                args = operation.get('args', [])
                kwargs = operation.get('kwargs', {})
                operation_key = operation.get('operation_key', 'bulk')
                config = operation.get('config')
                
                return await self.retry_async(
                    func, *args, 
                    operation_key=operation_key,
                    config=config,
                    **kwargs
                )
        
        # Execute all operations concurrently
        tasks = [execute_operation(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to RetryResult objects
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(RetryResult(
                    success=False,
                    attempts=[],
                    total_time=0.0,
                    final_error=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get retry handler statistics."""
        circuit_stats = {}
        for key, breaker in self.circuit_breakers.items():
            circuit_stats[key] = breaker.get_stats()
        
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_retries'] / 
                max(self.stats['total_retries'], 1)
            ),
            'circuit_breakers': circuit_stats,
            'active_circuit_breakers': len(self.circuit_breakers)
        }
    
    async def health_check(self) -> bool:
        """Perform health check of retry handler."""
        try:
            # Test basic retry functionality
            async def test_func():
                return "success"
            
            result = await self.retry_async(test_func, operation_key="health_check")
            return result.success
            
        except Exception as e:
            self.logger.error(f"❌ Retry handler health check failed: {e}")
            return False
    
    def reset_circuit_breaker(self, operation_key: str) -> bool:
        """Manually reset a circuit breaker."""
        if operation_key in self.circuit_breakers:
            breaker = self.circuit_breakers[operation_key]
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            
            self.logger.info(f"🔄 Circuit breaker reset: {operation_key}")
            return True
        
        return False