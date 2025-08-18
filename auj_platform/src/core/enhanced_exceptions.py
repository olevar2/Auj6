"""
Enhanced Exception Handling System for AUJ Platform
==================================================
Provides specific exceptions and enhanced error handling without disrupting existing code.

Author: AUJ Platform Development Team
Date: 2025-07-04
Version: 1.0.0
"""
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from collections import defaultdict, deque

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    CONFIGURATION = "configuration"
    DATA_PROVIDER = "data_provider"
    TRADING = "trading"
    SECURITY = "security"
    VALIDATION = "validation"
    MESSAGING = "messaging"
    AGENT = "agent"
    SYSTEM = "system"
    COORDINATION = "coordination"
    MONITORING = "monitoring"

class AUJException(Exception):
    """Base exception for AUJ Platform with enhanced metadata"""
    
    def __init__(self, 
                 message: str, 
                 error_code: Optional[str] = None,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 component: Optional[str] = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None,
                 recoverable: bool = True):
        super().__init__(message)
        self.error_code = error_code
        self.category = category
        self.component = component
        self.severity = severity
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/storage"""
        return {
            'message': str(self),
            'error_code': self.error_code,
            'category': self.category.value,
            'component': self.component,
            'severity': self.severity.value,
            'context': self.context,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp.isoformat(),
            'traceback': traceback.format_exc()
        }

class ConfigurationError(AUJException):
    """Configuration-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )

class DataProviderError(AUJException):
    """Data provider errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_PROVIDER,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class TradingError(AUJException):
    """Trading execution errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TRADING,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class SecurityError(AUJException):
    """Security-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )

class ValidationError(AUJException):
    """Validation errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class MessagingError(AUJException):
    """Messaging system errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MESSAGING,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class AgentError(AUJException):
    """Agent-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AGENT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class CoordinationError(AUJException):
    """Coordination system errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.COORDINATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class MonitoringError(AUJException):
    """Monitoring system errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MONITORING,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class ErrorPattern:
    """Represents a pattern of errors for analysis"""
    def __init__(self, pattern_id: str, description: str):
        self.pattern_id = pattern_id
        self.description = description
        self.occurrences = 0
        self.last_occurrence = None
        self.components = set()
        self.severity_distribution = defaultdict(int)

class ExceptionHandler:
    """Enhanced exception handler that can wrap existing code"""
    
    def __init__(self, logger: logging.Logger, component_name: str):
        self.logger = logger
        self.component_name = component_name
        self.error_count = 0
        self.last_error_time = None
        self.error_history: deque = deque(maxlen=100)
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.alert_callbacks: List[Callable] = []
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for error alerts"""
        self.alert_callbacks.append(callback)
    
    def handle_exception(self, 
                        exception: Exception, 
                        operation: str,
                        context: Optional[Dict[str, Any]] = None,
                        **kwargs) -> bool:
        """
        Handle exception with enhanced logging and context
        Returns True if operation can continue, False if it should stop
        """
        self.error_count += 1
        self.last_error_time = datetime.utcnow()
        context = context or {}
        context.update(kwargs)
        
        # Create error record
        error_record = self._create_error_record(exception, operation, context)
        
        # Store in history
        self.error_history.append(error_record)
        
        # Pattern analysis
        self._analyze_error_patterns(error_record)
        
        # Check circuit breakers
        should_break = self._check_circuit_breaker(operation, error_record)
        
        # Log with appropriate level
        self._log_error(error_record)
        
        # Send alerts if necessary
        self._send_alerts(error_record)
        
        # Determine if recoverable
        is_recoverable = self._is_recoverable(exception) and not should_break
        
        return is_recoverable
    
    def _create_error_record(self, exception: Exception, operation: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed error record"""
        error_record = {
            'component': self.component_name,
            'operation': operation,
            'error_type': type(exception).__name__,
            'error_message': str(exception),
            'context': context,
            'timestamp': self.last_error_time.isoformat(),
            'error_count': self.error_count,
            'traceback': traceback.format_exc()
        }
        
        # Add AUJ-specific information if available
        if isinstance(exception, AUJException):
            error_record.update({
                'error_code': exception.error_code,
                'category': exception.category.value,
                'severity': exception.severity.value,
                'auj_component': exception.component,
                'auj_context': exception.context,
                'recoverable': exception.recoverable
            })
        
        return error_record
    
    def _analyze_error_patterns(self, error_record: Dict[str, Any]):
        """Analyze error patterns for trending"""
        error_type = error_record['error_type']
        operation = error_record['operation']
        pattern_key = f"{error_type}:{operation}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = ErrorPattern(
                pattern_id=pattern_key,
                description=f"{error_type} in {operation}"
            )
        
        pattern = self.error_patterns[pattern_key]
        pattern.occurrences += 1
        pattern.last_occurrence = self.last_error_time
        pattern.components.add(self.component_name)
        
        severity = error_record.get('severity', 'medium')
        pattern.severity_distribution[severity] += 1
    
    def _check_circuit_breaker(self, operation: str, error_record: Dict[str, Any]) -> bool:
        """Check if circuit breaker should activate"""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {
                'error_count': 0,
                'last_reset': datetime.utcnow(),
                'state': 'CLOSED',  # CLOSED, OPEN, HALF_OPEN
                'threshold': 5,
                'timeout_minutes': 5
            }
        
        breaker = self.circuit_breakers[operation]
        
        # Reset if timeout has passed
        time_since_reset = datetime.utcnow() - breaker['last_reset']
        if time_since_reset.total_seconds() > (breaker['timeout_minutes'] * 60):
            breaker['error_count'] = 0
            breaker['last_reset'] = datetime.utcnow()
            breaker['state'] = 'CLOSED'
        
        breaker['error_count'] += 1
        
        # Check if threshold exceeded
        if breaker['error_count'] >= breaker['threshold']:
            if breaker['state'] == 'CLOSED':
                breaker['state'] = 'OPEN'
                self.logger.critical(f"Circuit breaker OPENED for operation: {operation}")
                return True
        
        return breaker['state'] == 'OPEN'
    
    def _log_error(self, error_record: Dict[str, Any]):
        """Log error with appropriate level"""
        severity = error_record.get('severity', 'medium')
        log_level = self._get_log_level(severity)
        
        # Create formatted message
        message = (f"Exception in {error_record['operation']}: "
                  f"{error_record['error_type']}: {error_record['error_message']}")
        
        # Add context if available
        if error_record.get('context'):
            message += f" | Context: {error_record['context']}"
        
        self.logger.log(log_level, message)
        
        # Log full details at debug level
        self.logger.debug(f"Full error details: {error_record}")
    
    def _send_alerts(self, error_record: Dict[str, Any]):
        """Send alerts for critical errors"""
        severity = error_record.get('severity', 'medium')
        
        if severity in ['high', 'critical'] or self.error_count % 10 == 0:
            alert_data = {
                'type': 'ERROR_ALERT',
                'component': self.component_name,
                'severity': severity,
                'error_record': error_record,
                'error_count': self.error_count
            }
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
    
    def _get_log_level(self, severity: str) -> int:
        """Map severity to log level"""
        mapping = {
            'low': logging.WARNING,
            'medium': logging.ERROR,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }
        return mapping.get(severity.lower(), logging.ERROR)
    
    def _is_recoverable(self, exception: Exception) -> bool:
        """Determine if exception is recoverable"""
        # AUJ exceptions have explicit recoverability
        if isinstance(exception, AUJException):
            return exception.recoverable
        
        # System-level exceptions that are typically non-recoverable
        non_recoverable = (SecurityError, ConfigurationError, MemoryError, SystemExit, KeyboardInterrupt)
        
        return not isinstance(exception, non_recoverable)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        if not self.error_history:
            return {'status': 'no_errors'}
        
        recent_errors = list(self.error_history)
        
        # Calculate statistics
        error_types = defaultdict(int)
        severities = defaultdict(int)
        operations = defaultdict(int)
        
        for error in recent_errors:
            error_types[error['error_type']] += 1
            severities[error.get('severity', 'medium')] += 1
            operations[error['operation']] += 1
        
        # Calculate error rate (errors per hour)
        if len(recent_errors) >= 2:
            time_span = (datetime.fromisoformat(recent_errors[-1]['timestamp'].replace('Z', '+00:00')) -
                        datetime.fromisoformat(recent_errors[0]['timestamp'].replace('Z', '+00:00')))
            hours = max(time_span.total_seconds() / 3600, 0.1)  # Minimum 0.1 hour
            error_rate = len(recent_errors) / hours
        else:
            error_rate = 0
        
        return {
            'total_errors': self.error_count,
            'recent_errors': len(recent_errors),
            'error_rate_per_hour': error_rate,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'component': self.component_name,
            'error_types': dict(error_types),
            'severities': dict(severities),
            'operations': dict(operations),
            'patterns': {
                pattern_id: {
                    'occurrences': pattern.occurrences,
                    'last_occurrence': pattern.last_occurrence.isoformat() if pattern.last_occurrence else None,
                    'components': list(pattern.components),
                    'severity_distribution': dict(pattern.severity_distribution)
                }
                for pattern_id, pattern in self.error_patterns.items()
            },
            'circuit_breakers': {
                operation: {
                    'state': breaker['state'],
                    'error_count': breaker['error_count'],
                    'last_reset': breaker['last_reset'].isoformat()
                }
                for operation, breaker in self.circuit_breakers.items()
            }
        }

def safe_execute(func, exception_handler: ExceptionHandler, operation_name: str, **kwargs):
    """
    Safely execute a function with enhanced exception handling
    Can be used to wrap existing code without major changes
    """
    try:
        return func(**kwargs)
    except Exception as e:
        recoverable = exception_handler.handle_exception(e, operation_name, **kwargs)
        if not recoverable:
            raise
        return None

def create_error_wrapper(component_name: str, logger: Optional[logging.Logger] = None):
    """Create an error handling wrapper for a component"""
    if logger is None:
        logger = logging.getLogger(component_name)
    
    return ExceptionHandler(logger, component_name)

# Global error registry for cross-component analysis
class GlobalErrorRegistry:
    """Global registry for tracking errors across all components"""
    
    def __init__(self):
        self.component_handlers: Dict[str, ExceptionHandler] = {}
        self.global_patterns: Dict[str, ErrorPattern] = {}
        self.alert_callbacks: List[Callable] = []
    
    def register_handler(self, component_name: str, handler: ExceptionHandler):
        """Register a component's error handler"""
        self.component_handlers[component_name] = handler
        
        # Add global alert callback
        handler.add_alert_callback(self._handle_global_alert)
    
    def _handle_global_alert(self, alert_data: Dict[str, Any]):
        """Handle alerts from any component"""
        # Forward to global alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logging.getLogger(__name__).error(f"Global alert callback error: {e}")
    
    def add_global_alert_callback(self, callback: Callable):
        """Add global alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_global_error_summary(self) -> Dict[str, Any]:
        """Get error summary across all components"""
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_components': len(self.component_handlers),
            'components': {},
            'global_patterns': {},
            'critical_alerts': 0,
            'total_errors': 0
        }
        
        for component_name, handler in self.component_handlers.items():
            stats = handler.get_error_statistics()
            summary['components'][component_name] = stats
            summary['total_errors'] += stats.get('total_errors', 0)
            
            # Count critical errors
            severities = stats.get('severities', {})
            summary['critical_alerts'] += severities.get('critical', 0)
        
        return summary

# Global instance
_global_error_registry = GlobalErrorRegistry()

def get_global_error_registry() -> GlobalErrorRegistry:
    """Get the global error registry"""
    return _global_error_registry