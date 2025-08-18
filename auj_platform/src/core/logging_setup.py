"""
Logging configuration for the AUJ Platform.

This module sets up comprehensive logging for all platform components
with different log levels and output destinations.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .exceptions import ConfigurationError
from .unified_config import get_unified_config


class AUJFormatter(logging.Formatter):
    """Custom formatter for AUJ Platform logs."""

    def __init__(self):
        """Initialize AUJ formatter."""
        super().__init__()

    def format(self, record):
        # Create custom format based on log level
        if record.levelno >= logging.ERROR:
            fmt = '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s'
        elif record.levelno >= logging.WARNING:
            fmt = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        else:
            fmt = '[%(asctime)s] [%(levelname)s] %(message)s'

        self._style._fmt = fmt
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging for the AUJ Platform.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: project_root/logs)
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        max_file_size: Maximum size of each log file in bytes
        backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance

    Raises:
        ConfigurationError: If logging setup fails
    """
    try:
        # Validate log level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')

        # Set up log directory
        if log_dir is None:
            # Default to project root/logs
            project_root = Path(__file__).parent.parent.parent.parent
            log_dir = project_root / "logs"
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(exist_ok=True)

        # Create root logger
        root_logger = logging.getLogger("auj_platform")
        root_logger.setLevel(numeric_level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatter
        formatter = AUJFormatter()

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File handlers
        if enable_file:
            # Main log file (all levels)
            main_log_file = log_dir / "auj_platform.log"
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            main_handler.setLevel(numeric_level)
            main_handler.setFormatter(formatter)
            root_logger.addHandler(main_handler)

            # Error log file (errors and critical only)
            error_log_file = log_dir / "auj_platform_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)

            # Daily log file (for daily operations)
            daily_log_file = log_dir / f"auj_platform_daily_{datetime.now().strftime('%Y%m%d')}.log"
            daily_handler = logging.FileHandler(daily_log_file, encoding='utf-8')
            daily_handler.setLevel(logging.INFO)
            daily_handler.setFormatter(formatter)
            root_logger.addHandler(daily_handler)

        # Log the successful setup
        root_logger.info(f"AUJ Platform logging initialized - Level: {log_level}, Log Dir: {log_dir}")

        return root_logger

    except Exception as e:
        raise ConfigurationError(f"Failed to setup logging: {str(e)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"auj_platform.{name}")


def log_trade_signal(logger: logging.Logger, signal_data: dict):
    """
    Log trade signal with structured format.

    Args:
        logger: Logger instance
        signal_data: Trade signal data dictionary
    """
    logger.info(
        f"TRADE_SIGNAL: {signal_data.get('symbol', 'UNKNOWN')} "
        f"{signal_data.get('direction', 'UNKNOWN')} "
        f"Confidence: {signal_data.get('confidence', 0.0):.2f} "
        f"Agent: {signal_data.get('generating_agent', 'UNKNOWN')}"
    )


def log_performance_metric(logger: logging.Logger, metric_data: dict):
    """
    Log performance metrics with structured format.

    Args:
        logger: Logger instance
        metric_data: Performance metric data dictionary
    """
    logger.info(
        f"PERFORMANCE: Agent: {metric_data.get('agent_name', 'UNKNOWN')} "
        f"Win Rate: {metric_data.get('win_rate', 0.0):.2f} "
        f"PnL: {metric_data.get('total_pnl', 0.0)} "
        f"Rank: {metric_data.get('current_rank', 'UNKNOWN')}"
    )


def log_system_health(logger: logging.Logger, health_data: dict):
    """
    Log system health status with structured format.

    Args:
        logger: Logger instance
        health_data: System health data dictionary
    """
    logger.info(
        f"SYSTEM_HEALTH: Active: {health_data.get('is_active', False)} "
        f"Alpha: {health_data.get('alpha_agent', 'NONE')} "
        f"Positions: {health_data.get('active_positions', 0)} "
        f"Daily PnL: {health_data.get('daily_pnl', 0.0)}"
    )


class LoggingSetup:
    """
    Logging setup manager for the AUJ Platform.

    Provides a convenient interface for setting up and managing
    platform-wide logging configuration.
    """

    def __init__(self, config=None):
        """
        Initialize LoggingSetup with optional configuration.

        Args:
            config: Configuration object containing logging settings
        """
        self.config = config
        self.config_manager = get_unified_config()
        self.logger = None
        self._initialized = False

    def setup(self, **kwargs) -> logging.Logger:
        """
        Set up logging with configuration.

        Args:
            **kwargs: Additional logging configuration options

        Returns:
            Configured root logger
        """
        # Get settings from config if available
        if self.config:
            log_level = self.config_manager.get_dict('logging', {}).get('level', 'INFO')
            log_dir = self.config_manager.get_dict('logging', {}).get('directory', None)
            enable_console = self.config_manager.get_dict('logging', {}).get('console', True)
            enable_file = self.config_manager.get_dict('logging', {}).get('file', True)
        else:
            log_level = kwargs.get('log_level', 'INFO')
            log_dir = kwargs.get('log_dir', None)
            enable_console = kwargs.get('enable_console', True)
            enable_file = kwargs.get('enable_file', True)

        # Setup logging
        self.logger = setup_logging(
            log_level=log_level,
            log_dir=log_dir,
            enable_console=enable_console,
            enable_file=enable_file
        )

        self._initialized = True
        return self.logger

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific component.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        if not self._initialized:
            # Setup with defaults if not already initialized
            self.setup()

        return get_logger(name)

    def is_initialized(self) -> bool:
        """Check if logging is initialized."""
        return self._initialized
