"""
Dashboard Logging Configuration
Professional logging system for AUJ Platform Dashboard
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional

class DashboardLogger:
    """Professional logging system for AUJ Platform Dashboard"""
    
    def __init__(self, name: str = 'auj_platform_dashboard', log_dir: str = None):
        self.name = name
        self.log_dir = log_dir or os.path.join(os.path.dirname(__file__), '..', 'logs')
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup comprehensive logging system"""
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handlers
        self._add_file_handlers(detailed_formatter)
        
        # Console handler
        self._add_console_handler(simple_formatter)
        
        # Error handler for critical issues
        self._add_error_handler(detailed_formatter)
    
    def _add_file_handlers(self, formatter):
        """Add file-based logging handlers"""
        # Main log file with rotation
        main_log_file = os.path.join(self.log_dir, f'{self.name}.log')
        file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Daily log files
        daily_log_file = os.path.join(self.log_dir, f'{self.name}_daily.log')
        daily_handler = TimedRotatingFileHandler(
            daily_log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        daily_handler.setLevel(logging.DEBUG)
        daily_handler.setFormatter(formatter)
        self.logger.addHandler(daily_handler)
    
    def _add_console_handler(self, formatter):
        """Add console logging handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _add_error_handler(self, formatter):
        """Add dedicated error logging handler"""
        error_log_file = os.path.join(self.log_dir, f'{self.name}_errors.log')
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger
    
    def log_performance(self, operation: str, duration: float, details: dict = None):
        """Log performance metrics"""
        message = f"PERFORMANCE - {operation}: {duration:.3f}s"
        if details:
            message += f" | Details: {details}"
        self.logger.info(message)
    
    def log_error_with_context(self, error: Exception, context: dict = None):
        """Log error with additional context"""
        message = f"ERROR - {type(error).__name__}: {str(error)}"
        if context:
            message += f" | Context: {context}"
        self.logger.error(message, exc_info=True)
    
    def log_user_action(self, action: str, user_id: str = None, details: dict = None):
        """Log user actions for audit trail"""
        message = f"USER_ACTION - {action}"
        if user_id:
            message += f" | User: {user_id}"
        if details:
            message += f" | Details: {details}"
        self.logger.info(message)
    
    def log_api_call(self, endpoint: str, method: str, status_code: int, duration: float):
        """Log API calls"""
        message = f"API_CALL - {method} {endpoint} - Status: {status_code} - Duration: {duration:.3f}s"
        
        if status_code >= 400:
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def log_indicator_calculation(self, indicator: str, symbol: str, timeframe: str, 
                                success: bool, duration: float = None):
        """Log indicator calculations"""
        status = "SUCCESS" if success else "FAILED"
        message = f"INDICATOR - {indicator} | {symbol} | {timeframe} | {status}"
        
        if duration:
            message += f" | Duration: {duration:.3f}s"
        
        if success:
            self.logger.debug(message)
        else:
            self.logger.warning(message)
    
    def log_data_fetch(self, source: str, symbol: str, timeframe: str, 
                      records: int, success: bool):
        """Log data fetching operations"""
        status = "SUCCESS" if success else "FAILED"
        message = f"DATA_FETCH - {source} | {symbol} | {timeframe} | Records: {records} | {status}"
        
        if success:
            self.logger.info(message)
        else:
            self.logger.error(message)
    
    def log_session_info(self, session_id: str, action: str, details: dict = None):
        """Log session-related information"""
        message = f"SESSION - {session_id} | {action}"
        if details:
            message += f" | {details}"
        self.logger.info(message)

class PerformanceLogger:
    """Specialized logger for performance monitoring"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str, details: dict = None):
        """End timing and log the duration"""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            self.logger.info(f"TIMER - {operation}: {duration:.3f}s" + 
                           (f" | {details}" if details else ""))
            del self.start_times[operation]
            return duration
        return None
    
    def log_memory_usage(self, operation: str = "Memory Check"):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.logger.info(f"MEMORY - {operation}: {memory_mb:.2f} MB")
            return memory_mb
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return None

# Global logger instance
_dashboard_logger = None

def get_dashboard_logger(name: str = 'auj_platform_dashboard') -> DashboardLogger:
    """Get or create the global dashboard logger"""
    global _dashboard_logger
    if _dashboard_logger is None:
        _dashboard_logger = DashboardLogger(name)
    return _dashboard_logger

def setup_streamlit_logging():
    """Setup logging for Streamlit applications"""
    # Reduce Streamlit's verbose logging
    logging.getLogger('streamlit').setLevel(logging.WARNING)
    logging.getLogger('streamlit.runtime').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Get dashboard logger
    dashboard_logger = get_dashboard_logger()
    return dashboard_logger.get_logger()

# Convenience functions
def log_info(message: str, context: dict = None):
    """Quick info logging"""
    logger = get_dashboard_logger().get_logger()
    if context:
        message += f" | Context: {context}"
    logger.info(message)

def log_warning(message: str, context: dict = None):
    """Quick warning logging"""
    logger = get_dashboard_logger().get_logger()
    if context:
        message += f" | Context: {context}"
    logger.warning(message)

def log_error(message: str, error: Exception = None, context: dict = None):
    """Quick error logging"""
    logger = get_dashboard_logger().get_logger()
    if context:
        message += f" | Context: {context}"
    if error:
        logger.error(message, exc_info=error)
    else:
        logger.error(message)

def log_debug(message: str, context: dict = None):
    """Quick debug logging"""
    logger = get_dashboard_logger().get_logger()
    if context:
        message += f" | Context: {context}"
    logger.debug(message)