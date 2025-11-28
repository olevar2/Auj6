"""
Alert Manager for AUJ Platform.

This module provides a centralized system for handling critical platform alerts,
routing them to appropriate channels (logging, messaging service, etc.) based on severity.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from ..logging_setup import get_logger
from ..unified_config import UnifiedConfigManager

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class AlertManager:
    """
    Centralized manager for platform alerts.
    
    Handles dispatching alerts to:
    1. Dedicated alert log file
    2. Messaging service (if available)
    3. Console (for critical/emergency)
    """
    
    def __init__(self, 
                 config_manager: UnifiedConfigManager,
                 messaging_service: Optional[Any] = None):
        """
        Initialize AlertManager.
        
        Args:
            config_manager: Unified configuration manager
            messaging_service: Optional messaging service for remote alerts
        """
        self.config_manager = config_manager
        self.messaging_service = messaging_service
        self.logger = get_logger("alert_manager")
        
        # Setup dedicated alert logger
        self._setup_alert_logging()
        
        self.logger.info("Alert Manager initialized")

    def _setup_alert_logging(self):
        """Setup dedicated logger for alerts."""
        self.alert_logger = logging.getLogger("auj_platform.alerts")
        self.alert_logger.propagate = False  # Don't propagate to root to avoid double logging
        
        # Get log directory from config or default
        log_dir = "logs"
        if hasattr(self.config_manager, 'get_dict'):
            logging_config = self.config_manager.get_dict('logging', {})
            log_dir = logging_config.get('directory', 'logs')
            
        import os
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler for alerts
        handler = logging.FileHandler(os.path.join(log_dir, "critical_alerts.log"), encoding='utf-8')
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)
        self.alert_logger.addHandler(handler)
        self.alert_logger.setLevel(logging.INFO)

    async def send_alert(self, 
                       title: str, 
                       message: str, 
                       severity: AlertSeverity = AlertSeverity.INFO,
                       details: Optional[Dict[str, Any]] = None,
                       component: str = "system"):
        """
        Send an alert through configured channels.
        
        Args:
            title: Short summary of the alert
            message: Detailed alert message
            severity: Alert severity level
            details: Optional dictionary with technical details
            component: Component generating the alert
        """
        try:
            timestamp = datetime.now().isoformat()
            alert_data = {
                "timestamp": timestamp,
                "title": title,
                "message": message,
                "severity": severity.value,
                "component": component,
                "details": details or {}
            }
            
            # 1. Log to dedicated alert file
            log_msg = f"[{component.upper()}] {title}: {message}"
            if details:
                log_msg += f" | Details: {json.dumps(details)}"
                
            if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                self.alert_logger.error(log_msg)
                self.logger.error(f"CRITICAL ALERT: {log_msg}") # Also log to main log
            elif severity == AlertSeverity.WARNING:
                self.alert_logger.warning(log_msg)
            else:
                self.alert_logger.info(log_msg)
                
            # 2. Send via Messaging Service (if available)
            if self.messaging_service:
                try:
                    # Determine appropriate channel/topic based on severity
                    if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                        topic = "auj.risk.alerts"
                        priority = "high"
                    else:
                        topic = "auj.system.status"
                        priority = "normal"
                        
                    # Use publish_system_status or publish_risk_alert based on context
                    if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                        await self.messaging_service.publish_risk_alert(
                            risk_type="SYSTEM_ALERT",
                            severity=severity.value,
                            affected_symbols=[],
                            risk_data=alert_data,
                            recommended_actions=["Check system logs", "Verify component status"]
                        )
                    else:
                        await self.messaging_service.publish_system_status(
                            component=component,
                            status=severity.value,
                            details=alert_data
                        )
                        
                except Exception as msg_err:
                    self.logger.error(f"Failed to send alert via messaging service: {msg_err}")
                    
        except Exception as e:
            # Fallback print to ensure critical errors are seen even if logging fails
            print(f"!!! ALERT SYSTEM FAILURE !!! Failed to process alert: {title} - {e}")
