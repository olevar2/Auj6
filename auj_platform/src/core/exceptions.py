"""
Custom exceptions for the AUJ Platform.

This module defines all custom exceptions used throughout the platform
to provide clear error handling and debugging capabilities.
"""

from typing import Optional, Any, Dict


class AUJPlatformError(Exception):
    """Base exception class for all AUJ Platform errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(AUJPlatformError):
    """Raised when there's an error in configuration loading or validation."""
    pass


class DataProviderError(AUJPlatformError):
    """Raised when there's an error with data providers."""
    pass


class DataNotAvailableError(DataProviderError):
    """Raised when requested data is not available from any provider."""
    pass


class IndicatorCalculationError(AUJPlatformError):
    """Raised when there's an error calculating indicators."""
    pass


class IndicatorNotFoundError(AUJPlatformError):
    """Raised when a requested indicator is not found or registered."""
    pass


class AgentError(AUJPlatformError):
    """Raised when there's an error with agent operations."""
    pass


class HierarchyError(AUJPlatformError):
    """Raised when there's an error with agent hierarchy operations."""
    pass


class TradingEngineError(AUJPlatformError):
    """Raised when there's an error with trading engine operations."""
    pass


class RiskManagementError(TradingEngineError):
    """Raised when there's an error with risk management."""
    pass


class ExecutionError(TradingEngineError):
    """Raised when there's an error with trade execution."""
    pass


class DatabaseError(AUJPlatformError):
    """Raised when there's an error with database operations."""
    pass


class ValidationError(AUJPlatformError):
    """Raised when there's an error with data validation."""
    pass


class SecurityError(AUJPlatformError):
    """Raised when there's a security-related error."""
    pass


class OptimizationError(AUJPlatformError):
    """Raised when there's an error with optimization processes."""
    pass


class APIError(AUJPlatformError):
    """Raised when there's an error with API operations."""
    pass


class InsufficientDataError(DataProviderError):
    """Raised when there's insufficient data for analysis."""
    pass


class MarketDataError(DataProviderError):
    """Raised when there's an error with market data."""
    pass


class ConnectionError(DataProviderError):
    """Raised when there's a connection error with data sources."""
    pass


class AuthenticationError(AUJPlatformError):
    """Raised when there's an authentication error."""
    pass


class PermissionError(AUJPlatformError):
    """Raised when there's a permission error."""
    pass


class PerformanceTrackingError(AUJPlatformError):
    """Raised when there's an error with performance tracking operations."""
    pass


class CoordinationError(AUJPlatformError):
    """Raised when there's an error with agent coordination operations."""
    pass


class ForecastingError(AUJPlatformError):
    """Raised when there's an error with forecasting operations."""
    pass


# Aliases for backward compatibility and consistency
CalculationError = IndicatorCalculationError
AUJDataProviderError = DataProviderError


class BrokerError(TradingEngineError):
    """Raised when there's an error with broker operations."""
    pass


class TradingError(TradingEngineError):
    """Raised when there's an error with trading operations."""
    pass


class AUJException(AUJPlatformError):
    """Alias for AUJPlatformError for backward compatibility."""
    pass


# Phase 5 Anti-Overfitting Framework Exceptions
class WalkForwardValidationError(ValidationError):
    """Raised when there's an error with walk-forward validation operations."""
    pass


class IndicatorEffectivenessError(ValidationError):
    """Raised when there's an error with indicator effectiveness analysis."""
    pass


class AgentBehaviorOptimizationError(OptimizationError):
    """Raised when there's an error with agent behavior optimization."""
    pass


class FeedbackLoopError(AUJPlatformError):
    """Raised when there's an error with the hourly feedback loop operations."""
    pass


class AntiOverfittingError(ValidationError):
    """Raised when there's an error with anti-overfitting mechanisms."""
    pass


class AUJSystemError(AUJPlatformError):
    """Raised when there's a critical system-level error."""
    pass


# Enhanced Exception Classes for Verification
class AUJException(AUJPlatformError):
    """Enhanced base exception with severity levels and error codes"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, severity: str = "medium", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, context)
        self.severity = severity  # low, medium, high, critical


class AUJValidationError(ValidationError):
    """Enhanced validation error for verification compatibility"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, severity: str = "medium", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, context)
        self.severity = severity


class AUJConfigurationError(ConfigurationError):
    """Enhanced configuration error for verification compatibility"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, severity: str = "high", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, context)
        self.severity = severity


class AUJTradingError(TradingEngineError):
    """Enhanced trading error for verification compatibility"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, severity: str = "high", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, context)
        self.severity = severity


class AUJConnectionError(AUJPlatformError):
    """Enhanced connection error for verification compatibility"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, severity: str = "medium", context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, context)
        self.severity = severity
