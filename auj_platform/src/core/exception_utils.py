"""
Exception Handling Utilities
===========================
Utilities to apply enhanced exception handling to existing components without major code changes.

Author: AUJ Platform Development Team
Date: 2025-07-04
Version: 1.0.0
"""
import asyncio
import functools
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from .enhanced_exceptions import ExceptionHandler, create_error_wrapper, get_global_error_registry

T = TypeVar('T')

def with_exception_handling(
    component_name: Optional[str] = None,
    operation_name: Optional[str] = None,
    recoverable: bool = True,
    logger: Optional[logging.Logger] = None
):
    """
    Decorator to add enhanced exception handling to methods.
    
    Args:
        component_name: Name of the component (auto-detected if None)
        operation_name: Name of the operation (uses function name if None)
        recoverable: Whether errors should be recoverable
        logger: Logger to use (auto-detected if None)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Auto-detect component name from class if not provided
        actual_component_name = component_name
        if actual_component_name is None:
            actual_component_name = getattr(func, '__qualname__', func.__name__).split('.')[0]
        
        # Create or get exception handler
        actual_logger = logger or logging.getLogger(actual_component_name)
        handler = create_error_wrapper(actual_component_name, actual_logger)
        
        # Register with global registry
        get_global_error_registry().register_handler(actual_component_name, handler)
        
        actual_operation_name = operation_name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context = {
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                    
                    is_recoverable = handler.handle_exception(
                        e, actual_operation_name, context=context
                    )
                    
                    if not is_recoverable or not recoverable:
                        raise
                    
                    # Return safe fallback for recoverable errors
                    return None
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                    
                    is_recoverable = handler.handle_exception(
                        e, actual_operation_name, context=context
                    )
                    
                    if not is_recoverable or not recoverable:
                        raise
                    
                    # Return safe fallback for recoverable errors
                    return None
            
            return sync_wrapper
    
    return decorator

def safe_method_call(
    obj: Any,
    method_name: str,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely call a method on an object with enhanced error handling.
    
    Args:
        obj: Object to call method on
        method_name: Name of method to call
        default_return: Value to return if method fails
        log_errors: Whether to log errors
        **kwargs: Arguments to pass to method
    
    Returns:
        Method result or default_return if method fails
    """
    component_name = obj.__class__.__name__
    logger = logging.getLogger(component_name)
    handler = create_error_wrapper(component_name, logger)
    
    try:
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            if callable(method):
                return method(**kwargs)
        else:
            if log_errors:
                logger.warning(f"Method {method_name} not found on {component_name}")
    except Exception as e:
        if log_errors:
            handler.handle_exception(e, f"safe_call_{method_name}")
    
    return default_return

async def safe_async_method_call(
    obj: Any,
    method_name: str,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely call an async method on an object with enhanced error handling.
    """
    component_name = obj.__class__.__name__
    logger = logging.getLogger(component_name)
    handler = create_error_wrapper(component_name, logger)
    
    try:
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            if callable(method):
                if asyncio.iscoroutinefunction(method):
                    return await method(**kwargs)
                else:
                    return method(**kwargs)
        else:
            if log_errors:
                logger.warning(f"Method {method_name} not found on {component_name}")
    except Exception as e:
        if log_errors:
            handler.handle_exception(e, f"safe_async_call_{method_name}")
    
    return default_return

class SafeComponentWrapper:
    """
    Wrapper that adds safe method calling to any component.
    Can be used to wrap existing components without modifying their code.
    """
    
    def __init__(self, wrapped_component: Any, component_name: Optional[str] = None):
        self.wrapped_component = wrapped_component
        self.component_name = component_name or wrapped_component.__class__.__name__
        self.logger = logging.getLogger(self.component_name)
        self.exception_handler = create_error_wrapper(self.component_name, self.logger)
        
        # Register with global registry
        get_global_error_registry().register_handler(self.component_name, self.exception_handler)
    
    def __getattr__(self, name: str):
        """Intercept attribute access and wrap methods with error handling"""
        attr = getattr(self.wrapped_component, name)
        
        if callable(attr):
            return self._wrap_method(attr, name)
        return attr
    
    def _wrap_method(self, method: Callable, method_name: str) -> Callable:
        """Wrap a method with enhanced error handling"""
        if asyncio.iscoroutinefunction(method):
            async def async_safe_method(*args, **kwargs):
                try:
                    return await method(*args, **kwargs)
                except Exception as e:
                    context = {
                        'method': method_name,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                    
                    recoverable = self.exception_handler.handle_exception(
                        e, method_name, context=context
                    )
                    
                    if not recoverable:
                        raise
                    
                    return None
            
            return async_safe_method
        else:
            def safe_method(*args, **kwargs):
                try:
                    return method(*args, **kwargs)
                except Exception as e:
                    context = {
                        'method': method_name,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                    
                    recoverable = self.exception_handler.handle_exception(
                        e, method_name, context=context
                    )
                    
                    if not recoverable:
                        raise
                    
                    return None
            
            return safe_method
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for this component"""
        return self.exception_handler.get_error_statistics()

def enhance_existing_component(component: Any, component_name: Optional[str] = None) -> SafeComponentWrapper:
    """
    Enhance an existing component with advanced error handling.
    
    Args:
        component: The component to enhance
        component_name: Optional name for the component
    
    Returns:
        Enhanced component wrapper
    """
    return SafeComponentWrapper(component, component_name)

def create_error_monitoring_middleware():
    """
    Create middleware for error monitoring that can be added to existing systems.
    """
    class ErrorMonitoringMiddleware:
        def __init__(self):
            self.global_registry = get_global_error_registry()
            self.logger = logging.getLogger('ErrorMonitoring')
        
        def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process incoming request with error monitoring"""
            try:
                # Add request tracking
                return request_data
            except Exception as e:
                handler = create_error_wrapper('ErrorMonitoring', self.logger)
                handler.handle_exception(e, 'process_request', context={'request': request_data})
                raise
        
        def process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process outgoing response with error monitoring"""
            try:
                # Add response validation
                return response_data
            except Exception as e:
                handler = create_error_wrapper('ErrorMonitoring', self.logger)
                handler.handle_exception(e, 'process_response', context={'response': response_data})
                raise
        
        def get_error_summary(self) -> Dict[str, Any]:
            """Get comprehensive error summary"""
            return self.global_registry.get_global_error_summary()
    
    return ErrorMonitoringMiddleware()

# Context manager for safe execution blocks
class safe_execution:
    """
    Context manager for safe execution of code blocks.
    
    Usage:
        with safe_execution('component_name', 'operation_name') as safe:
            # risky code here
            pass
        
        if safe.success:
            # execution succeeded
            pass
        else:
            # execution failed
            print(safe.error)
    """
    
    def __init__(self, component_name: str, operation_name: str, logger: Optional[logging.Logger] = None):
        self.component_name = component_name
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(component_name)
        self.exception_handler = create_error_wrapper(component_name, self.logger)
        self.success = False
        self.error = None
        self.result = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
            self.error = exc_val
            
            recoverable = self.exception_handler.handle_exception(
                exc_val, self.operation_name
            )
            
            # Suppress exception if recoverable
            if recoverable:
                return True
        else:
            self.success = True
        
        return False

# Utility functions for common patterns
def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    component_name: str = 'RetryHandler'
):
    """
    Retry function with exponential backoff and error handling.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        backoff_seconds: Base backoff time in seconds
        component_name: Component name for error tracking
    
    Returns:
        Function result or None if all retries failed
    """
    import time
    
    logger = logging.getLogger(component_name)
    handler = create_error_wrapper(component_name, logger)
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            is_last_attempt = attempt == max_retries
            
            recoverable = handler.handle_exception(
                e, f'retry_attempt_{attempt}',
                context={'attempt': attempt, 'max_retries': max_retries}
            )
            
            if is_last_attempt or not recoverable:
                if is_last_attempt:
                    logger.error(f"All {max_retries} retry attempts failed")
                raise
            
            # Wait before retry with exponential backoff
            wait_time = backoff_seconds * (2 ** attempt)
            logger.info(f"Retrying in {wait_time} seconds (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
    
    return None