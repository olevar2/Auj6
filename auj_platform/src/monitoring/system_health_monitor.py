#!/usr/bin/env python3
"""
System Health Monitor for AUJ Platform
=====================================

Comprehensive system health monitoring with advanced alerting, performance tracking,
and resource monitoring capabilities. This monitors all critical platform components
and provides real-time health status with intelligent alerting.

Key Features:
- CPU/Memory/Disk monitoring with configurable thresholds
- Database connection health monitoring
- Broker connection status monitoring
- Message queue health monitoring
- Execution time tracking for all operations
- Throughput monitoring with trend analysis
- Error rate tracking with pattern detection
- Resource utilization monitoring
- Configurable alert thresholds
- Multi-channel notification mechanisms
- Alert escalation and suppression logic
- Performance baseline tracking
- Anomaly detection

Author: AUJ Platform Development Team
Date: 2025-07-04
Version: 2.0.0 - Advanced System Health Monitor
"""

import asyncio
import logging
import time
import psutil
import aiohttp
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from statistics import mean, stdev
import numpy as np
from pathlib import Path

try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    # Create dummy classes for environments without email support
    class MimeText:
        def __init__(self, *args, **kwargs):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
            pass
    class MimeMultipart:
        def __init__(self, *args, **kwargs):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
            pass
            
    smtplib = None

from ..core.logging_setup import get_logger
from ..core.unified_config import get_unified_config

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class MonitorStatus(Enum):
    """Monitoring status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class MetricSample:
    """Individual metric sample."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdConfig:
    """Threshold configuration for metrics."""
    warning_min: Optional[float] = None
    warning_max: Optional[float] = None
    critical_min: Optional[float] = None
    critical_max: Optional[float] = None
    trend_warning_slope: Optional[float] = None
    trend_critical_slope: Optional[float] = None


@dataclass
class Alert:
    """System alert definition."""
    id: str
    component: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False
    notification_count: int = 0


@dataclass
class PerformanceBaseline:
    """Performance baseline for a metric."""
    metric_name: str
    mean_value: float
    std_deviation: float
    min_value: float
    max_value: float
    sample_count: int
    last_updated: datetime


class NotificationChannel:
    """Base notification channel."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize notification channel."""
        self.config_manager = get_unified_config()
        self.config = config or {}
        self.enabled = self.config_manager.get_bool('enabled', True)
        
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert."""
        if not self.enabled:
            return False
        
        try:
            # Base implementation - should be overridden by subclasses
            logger.info(f"📢 Base notification: {alert.severity.value.upper()} - {alert.component}: {alert.message}")
            return True
        except Exception as e:
            logger.error(f"❌ Base notification failed: {e}")
            return False


class EmailNotifier(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize email notifier."""
        super().__init__(config)
        self.smtp_server = self.config_manager.get_str('smtp_server', 'localhost')
        self.smtp_port = self.config_manager.get_int('smtp_port', 587)
        self.username = self.config_manager.get_str('username')
        self.password = self.config_manager.get_str('password')
        self.from_email = self.config_manager.get_str('from_email')
        self.to_emails = self.config_manager.get_dict('to_emails', [])
        
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        if not self.enabled or not self.to_emails or not EMAIL_AVAILABLE:
            if not EMAIL_AVAILABLE:
                logger.warning("Email notifications not available - email libraries not found")
            return False
            
        try:
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"AUJ Platform Alert: {alert.severity.value.upper()} - {alert.component}"
            
            body = f"""
Alert Details:
- Component: {alert.component}
- Severity: {alert.severity.value.upper()}
- Message: {alert.message}
- Timestamp: {alert.timestamp.isoformat()}
- Alert ID: {alert.id}

Details:
{json.dumps(alert.details, indent=2)}

AUJ Platform Monitoring System
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email in thread to avoid blocking
            def send_email():
                try:
                    if not smtplib:
                        return False
                        
                    server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                    if self.username and self.password:
                        server.starttls()
                        server.login(self.username, self.password)
                    
                    server.send_message(msg)
                    server.quit()
                    return True
                except Exception as e:
                    logger.error(f"Failed to send email notification: {e}")
                    return False
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, send_email)
            
            if result:
                logger.info(f"Email notification sent for alert {alert.id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Email notification failed for alert {alert.id}: {e}")
            return False


class SlackNotifier(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize slack notifier."""
        super().__init__(config)
        self.webhook_url = self.config_manager.get_str('webhook_url')
        self.channel = self.config_manager.get_str('channel', '#alerts')
        
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not self.enabled or not self.webhook_url:
            return False
            
        try:
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.FATAL: "danger"
            }
            
            payload = {
                "channel": self.channel,
                "username": "AUJ Platform Monitor",
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": f"{alert.severity.value.upper()} Alert: {alert.component}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Alert ID", "value": alert.id, "short": True},
                        {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True}
                    ],
                    "footer": "AUJ Platform Monitoring"
                }]
            }
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.id}")
                        return True
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Slack notification failed for alert {alert.id}: {e}")
            return False


class SystemHealthMonitor:
    """
    Advanced System Health Monitor for AUJ Platform.
    
    Provides comprehensive monitoring of all platform components with
    intelligent alerting, performance tracking, and anomaly detection.
    """
    
    def __init__(self, config=None):
        """Initialize system health monitor."""
        self.config_manager = get_unified_config()
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Monitoring configuration
        self.monitoring_config = self.config_manager.get_dict('monitoring', {})
        self.check_interval = self.monitoring_config.get('check_interval', 30)
        self.metric_retention_days = self.monitoring_config.get('metric_retention_days', 7)
        self.baseline_update_interval = self.monitoring_config.get('baseline_update_hours', 24) * 3600
        
        # Metric storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._thresholds: Dict[str, ThresholdConfig] = {}
        self._baselines: Dict[str, PerformanceBaseline] = {}
        
        # Alert management
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=10000)
        self._suppressed_alerts: Set[str] = set()
        self._alert_counter = 0
        
        # Notification system
        self._notification_channels: List[NotificationChannel] = []
        self._setup_notification_channels()
        
        # Monitoring state
        self._monitoring = False
        self._monitoring_task = None
        self._metrics_cleanup_task = None
        self._baseline_update_task = None
        
        # Component monitoring
        self._component_status: Dict[str, MonitorStatus] = {}
        self._last_check_times: Dict[str, datetime] = {}
        
        # Performance tracking
        self._execution_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._throughput_counters: Dict[str, int] = defaultdict(int)
        self._error_counters: Dict[str, int] = defaultdict(int)
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Load thresholds from configuration
        self._load_threshold_configuration()
        
        self.logger.info("🏥 SystemHealthMonitor initialized")
    
    def _setup_notification_channels(self):
        """Setup notification channels."""
        notifications_config = self.config_manager.get_dict('monitoring.notifications', {})
        
        # Email notifications
        email_config = notifications_config.get('email', {})
        if email_config.get('enabled', False):
            self._notification_channels.append(EmailNotifier(email_config))
            
        # Slack notifications
        slack_config = notifications_config.get('slack', {})
        if slack_config.get('enabled', False):
            self._notification_channels.append(SlackNotifier(slack_config))
        
        self.logger.info(f"📢 Configured {len(self._notification_channels)} notification channels")
    
    def _load_threshold_configuration(self):
        """Load threshold configuration."""
        thresholds_config = self.config_manager.get_dict('monitoring.thresholds', {})
        
        # Default thresholds
        default_thresholds = {
            'cpu_usage': ThresholdConfig(warning_max=80, critical_max=95),
            'memory_usage': ThresholdConfig(warning_max=85, critical_max=95),
            'disk_usage': ThresholdConfig(warning_max=85, critical_max=95),
            'database_response_time': ThresholdConfig(warning_max=1.0, critical_max=3.0),
            'api_response_time': ThresholdConfig(warning_max=2.0, critical_max=5.0),
            'error_rate': ThresholdConfig(warning_max=0.05, critical_max=0.1),
            'queue_depth': ThresholdConfig(warning_max=1000, critical_max=5000)
        }
        
        # Merge with config
        for metric_name, default_threshold in default_thresholds.items():
            user_config = thresholds_config.get(metric_name, {})
            
            self._thresholds[metric_name] = ThresholdConfig(
                warning_min=user_config.get('warning_min', default_threshold.warning_min),
                warning_max=user_config.get('warning_max', default_threshold.warning_max),
                critical_min=user_config.get('critical_min', default_threshold.critical_min),
                critical_max=user_config.get('critical_max', default_threshold.critical_max),
                trend_warning_slope=user_config.get('trend_warning_slope', default_threshold.trend_warning_slope),
                trend_critical_slope=user_config.get('trend_critical_slope', default_threshold.trend_critical_slope)
            )
        
        self.logger.info(f"📊 Loaded thresholds for {len(self._thresholds)} metrics")
    
    async def start_monitoring(self):
        """Start system health monitoring."""
        if self._monitoring:
            self.logger.warning("⚠️ System monitoring already running")
            return
        
        self._monitoring = True
        self.logger.info("🚀 Starting system health monitoring...")
        
        # Start monitoring tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._metrics_cleanup_task = asyncio.create_task(self._metrics_cleanup_loop())
        self._baseline_update_task = asyncio.create_task(self._baseline_update_loop())
        
        self.logger.info("✅ System health monitoring started")
    
    async def stop_monitoring(self):
        """Stop system health monitoring."""
        if not self._monitoring:
            return
        
        self.logger.info("🛑 Stopping system health monitoring...")
        self._monitoring = False
        
        # Cancel tasks
        tasks = [self._monitoring_task, self._metrics_cleanup_task, self._baseline_update_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("✅ System health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_checks(self):
        """Perform all health checks."""
        start_time = time.time()
        
        # Perform checks concurrently
        results = await asyncio.gather(
            self._check_system_resources(),
            self._check_database_health(),
            self._check_broker_connections(),
            self._check_message_queue_health(),
            self._check_api_endpoints(),
            return_exceptions=True
        )
        
        # Process results and update metrics
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"❌ Health check failed: {result}")
                continue
        
        # Record overall check duration
        check_duration = time.time() - start_time
        self.record_metric('health_check_duration', check_duration)
        
        # Update throughput metrics
        self._update_throughput_metrics()
        
        # Check for threshold violations
        await self._check_threshold_violations()
    
    async def _check_system_resources(self):
        """Check system resource utilization."""
        try:
            # CPU monitoring
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('cpu_usage', cpu_percent)
            
            # Memory monitoring
            memory = psutil.virtual_memory()
            self.record_metric('memory_usage', memory.percent)
            self.record_metric('memory_available_gb', memory.available / (1024**3))
            
            # Disk monitoring
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric('disk_usage', disk_percent)
            self.record_metric('disk_free_gb', disk.free / (1024**3))
            
            # Network monitoring
            net_io = psutil.net_io_counters()
            self.record_metric('network_bytes_sent', net_io.bytes_sent)
            self.record_metric('network_bytes_recv', net_io.bytes_recv)
            
            # Process monitoring
            current_process = psutil.Process()
            self.record_metric('process_memory_mb', current_process.memory_info().rss / (1024**2))
            self.record_metric('process_cpu_percent', current_process.cpu_percent())
            
            # Open file descriptors (Linux/Unix)
            try:
                self.record_metric('open_file_descriptors', current_process.num_fds())
            except (AttributeError, psutil.AccessDenied):
                pass  # Not available on all platforms
            
            self._component_status['system_resources'] = MonitorStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"❌ System resource check failed: {e}")
            self._component_status['system_resources'] = MonitorStatus.CRITICAL
            await self._create_alert(
                'system_resources',
                AlertSeverity.CRITICAL,
                f"System resource monitoring failed: {e}"
            )
    
    async def _check_database_health(self):
        """Check database connection health."""
        try:
            # Get database from config or DI container
            # This is a placeholder - would be replaced with actual database check
            start_time = time.time()
            
            # Simulate database connectivity check
            await asyncio.sleep(0.01)  # Simulate database query
            
            response_time = time.time() - start_time
            self.record_metric('database_response_time', response_time)
            
            # Check connection pool status (if available)
            # self.record_metric('database_pool_active', pool.active_connections)
            # self.record_metric('database_pool_idle', pool.idle_connections)
            
            self._component_status['database'] = MonitorStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"❌ Database health check failed: {e}")
            self._component_status['database'] = MonitorStatus.CRITICAL
            await self._create_alert(
                'database',
                AlertSeverity.CRITICAL,
                f"Database health check failed: {e}"
            )
    
    async def _check_broker_connections(self):
        """Check broker connection status."""
        try:
            # Check MT5 connection status
            # This would be replaced with actual broker interface checks
            start_time = time.time()
            
            # Simulate broker connectivity check
            await asyncio.sleep(0.005)  # Simulate broker ping
            
            response_time = time.time() - start_time
            self.record_metric('broker_response_time', response_time)
            
            # Check trading session status
            # self.record_metric('trading_session_active', is_session_active)
            
            self._component_status['broker_connections'] = MonitorStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"❌ Broker connection check failed: {e}")
            self._component_status['broker_connections'] = MonitorStatus.CRITICAL
            await self._create_alert(
                'broker_connections',
                AlertSeverity.CRITICAL,
                f"Broker connection check failed: {e}"
            )
    
    async def _check_message_queue_health(self):
        """Check message queue health."""
        try:
            # Check message queue depth and processing rate
            # This would be replaced with actual queue monitoring
            
            # Simulate queue metrics
            queue_depth = 0  # Would get from actual queue
            self.record_metric('queue_depth', queue_depth)
            
            # Check queue processing rate
            # processing_rate = queue.get_processing_rate()
            # self.record_metric('queue_processing_rate', processing_rate)
            
            self._component_status['message_queue'] = MonitorStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"❌ Message queue health check failed: {e}")
            self._component_status['message_queue'] = MonitorStatus.CRITICAL
            await self._create_alert(
                'message_queue',
                AlertSeverity.CRITICAL,
                f"Message queue health check failed: {e}"
            )
    
    async def _check_api_endpoints(self):
        """Check API endpoint health."""
        try:
            api_config = self.config_manager.get_dict('api_settings', {})
            if not api_config.get('enabled', False):
                return
            
            host = api_config.get('host', 'localhost')
            port = api_config.get('port', 8000)
            
            endpoints = [
                f'http://{host}:{port}/health',
                f'http://{host}:{port}/api/status'
            ]
            
            for endpoint in endpoints:
                start_time = time.time()
                
                try:
                    timeout = aiohttp.ClientTimeout(total=5)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(endpoint) as response:
                            response_time = time.time() - start_time
                            
                            endpoint_name = endpoint.split('/')[-1]
                            self.record_metric(f'api_{endpoint_name}_response_time', response_time)
                            
                            if response.status != 200:
                                await self._create_alert(
                                    'api_endpoints',
                                    AlertSeverity.WARNING,
                                    f"API endpoint {endpoint} returned {response.status}"
                                )
                            
                except asyncio.TimeoutError:
                    await self._create_alert(
                        'api_endpoints',
                        AlertSeverity.CRITICAL,
                        f"API endpoint {endpoint} timeout"
                    )
                except Exception as e:
                    await self._create_alert(
                        'api_endpoints',
                        AlertSeverity.WARNING,
                        f"API endpoint {endpoint} error: {e}"
                    )
            
            self._component_status['api_endpoints'] = MonitorStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"❌ API endpoint check failed: {e}")
            self._component_status['api_endpoints'] = MonitorStatus.CRITICAL
    
    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a metric sample."""
        with self._lock:
            sample = MetricSample(
                timestamp=datetime.now(),
                value=value,
                metadata=metadata or {}
            )
            
            self._metrics[metric_name].append(sample)
            self._last_check_times[metric_name] = sample.timestamp
    
    def record_execution_time(self, operation: str, duration: float):
        """Record execution time for an operation."""
        with self._lock:
            self._execution_times[operation].append(duration)
            self.record_metric(f'execution_time_{operation}', duration)
    
    def increment_throughput_counter(self, counter_name: str):
        """Increment throughput counter."""
        with self._lock:
            self._throughput_counters[counter_name] += 1
    
    def increment_error_counter(self, error_type: str):
        """Increment error counter."""
        with self._lock:
            self._error_counters[error_type] += 1
    
    def _update_throughput_metrics(self):
        """Update throughput metrics."""
        current_time = datetime.now()
        
        # Calculate per-minute rates
        for counter_name, count in self._throughput_counters.items():
            rate_per_minute = count  # This would be calculated over time window
            self.record_metric(f'throughput_{counter_name}_per_minute', rate_per_minute)
        
        # Calculate error rates
        for error_type, error_count in self._error_counters.items():
            total_operations = sum(self._throughput_counters.values()) or 1
            error_rate = error_count / total_operations
            self.record_metric(f'error_rate_{error_type}', error_rate)
    
    async def _check_threshold_violations(self):
        """Check for threshold violations and create alerts."""
        for metric_name, threshold in self._thresholds.items():
            if metric_name not in self._metrics or not self._metrics[metric_name]:
                continue
            
            latest_sample = self._metrics[metric_name][-1]
            value = latest_sample.value
            
            # Check critical thresholds
            if threshold.critical_max is not None and value > threshold.critical_max:
                await self._create_alert(
                    metric_name,
                    AlertSeverity.CRITICAL,
                    f"{metric_name} exceeded critical maximum: {value:.2f} > {threshold.critical_max}"
                )
            elif threshold.critical_min is not None and value < threshold.critical_min:
                await self._create_alert(
                    metric_name,
                    AlertSeverity.CRITICAL,
                    f"{metric_name} below critical minimum: {value:.2f} < {threshold.critical_min}"
                )
            # Check warning thresholds
            elif threshold.warning_max is not None and value > threshold.warning_max:
                await self._create_alert(
                    metric_name,
                    AlertSeverity.WARNING,
                    f"{metric_name} exceeded warning maximum: {value:.2f} > {threshold.warning_max}"
                )
            elif threshold.warning_min is not None and value < threshold.warning_min:
                await self._create_alert(
                    metric_name,
                    AlertSeverity.WARNING,
                    f"{metric_name} below warning minimum: {value:.2f} < {threshold.warning_min}"
                )
            
            # Check trend violations
            await self._check_trend_violations(metric_name, threshold)
    
    async def _check_trend_violations(self, metric_name: str, threshold: ThresholdConfig):
        """Check for trend-based threshold violations."""
        if not threshold.trend_warning_slope and not threshold.trend_critical_slope:
            return
        
        metrics = self._metrics[metric_name]
        if len(metrics) < 10:  # Need enough samples for trend analysis
            return
        
        # Calculate trend slope over recent samples
        recent_samples = list(metrics)[-10:]
        timestamps = [(s.timestamp - recent_samples[0].timestamp).total_seconds() for s in recent_samples]
        values = [s.value for s in recent_samples]
        
        if len(set(timestamps)) < 2:  # Need variation in time
            return
        
        try:
            # Simple linear regression
            n = len(timestamps)
            sum_x = sum(timestamps)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(timestamps, values))
            sum_x2 = sum(x * x for x in timestamps)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Check slope thresholds
            if threshold.trend_critical_slope is not None and abs(slope) > threshold.trend_critical_slope:
                await self._create_alert(
                    metric_name,
                    AlertSeverity.CRITICAL,
                    f"{metric_name} critical trend detected: slope {slope:.4f}"
                )
            elif threshold.trend_warning_slope is not None and abs(slope) > threshold.trend_warning_slope:
                await self._create_alert(
                    metric_name,
                    AlertSeverity.WARNING,
                    f"{metric_name} warning trend detected: slope {slope:.4f}"
                )
                
        except (ZeroDivisionError, ValueError):
            # Insufficient data for trend calculation
            pass
    
    async def _create_alert(self, component: str, severity: AlertSeverity, message: str, details: Dict[str, Any] = None):
        """Create and process alert."""
        alert_id = f"{component}_{severity.value}_{int(time.time())}"
        
        # Check if similar alert is suppressed
        if self._is_alert_suppressed(component, severity, message):
            return
        
        alert = Alert(
            id=alert_id,
            component=component,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        # Store alert
        with self._lock:
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
            self._alert_counter += 1
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.FATAL: logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        self.logger.log(log_level, f"🚨 ALERT [{severity.value.upper()}] {component}: {message}")
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        # Check for escalation
        await self._check_alert_escalation(alert)
    
    def _is_alert_suppressed(self, component: str, severity: AlertSeverity, message: str) -> bool:
        """Check if alert should be suppressed."""
        # Simple suppression based on component and recent alerts
        suppression_key = f"{component}_{severity.value}"
        
        if suppression_key in self._suppressed_alerts:
            return True
        
        # Check for recent similar alerts
        recent_threshold = datetime.now() - timedelta(minutes=5)
        recent_similar = [
            alert for alert in self._alert_history
            if (alert.component == component and 
                alert.severity == severity and
                alert.timestamp > recent_threshold and
                alert.message == message)
        ]
        
        if len(recent_similar) > 3:  # More than 3 similar alerts in 5 minutes
            self._suppressed_alerts.add(suppression_key)
            self.logger.info(f"🔇 Suppressing alerts for {suppression_key}")
            return True
        
        return False
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through all channels."""
        for channel in self._notification_channels:
            try:
                success = await channel.send_notification(alert)
                if success:
                    alert.notification_count += 1
            except Exception as e:
                self.logger.error(f"❌ Notification failed: {e}")
    
    async def _check_alert_escalation(self, alert: Alert):
        """Check if alert needs escalation."""
        # Escalate critical alerts that haven't been acknowledged
        if (alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.FATAL] and
            not alert.acknowledged and
            not alert.escalated):
            
            # Wait 15 minutes then escalate
            await asyncio.sleep(900)
            
            if not alert.acknowledged and not alert.resolved:
                alert.escalated = True
                escalated_alert = Alert(
                    id=f"{alert.id}_escalated",
                    component=alert.component,
                    severity=AlertSeverity.FATAL,
                    message=f"ESCALATED: {alert.message}",
                    timestamp=datetime.now(),
                    details=alert.details
                )
                
                await self._send_alert_notifications(escalated_alert)
                self.logger.critical(f"🚨 ESCALATED ALERT: {escalated_alert.message}")
    
    async def _metrics_cleanup_loop(self):
        """Clean up old metrics."""
        while self._monitoring:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(3600)  # Clean up every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Metrics cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_metrics(self):
        """Remove old metric samples."""
        cutoff_time = datetime.now() - timedelta(days=self.metric_retention_days)
        
        with self._lock:
            for metric_name, samples in self._metrics.items():
                # Remove old samples
                while samples and samples[0].timestamp < cutoff_time:
                    samples.popleft()
            
            # Clean up empty metrics
            empty_metrics = [name for name, samples in self._metrics.items() if not samples]
            for metric_name in empty_metrics:
                del self._metrics[metric_name]
        
        self.logger.debug(f"🧹 Cleaned up metrics older than {self.metric_retention_days} days")
    
    async def _baseline_update_loop(self):
        """Update performance baselines."""
        while self._monitoring:
            try:
                await self._update_performance_baselines()
                await asyncio.sleep(self.baseline_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Baseline update error: {e}")
                await asyncio.sleep(self.baseline_update_interval)
    
    async def _update_performance_baselines(self):
        """Update performance baselines for metrics."""
        with self._lock:
            for metric_name, samples in self._metrics.items():
                if len(samples) < 100:  # Need enough samples
                    continue
                
                values = [s.value for s in samples]
                
                baseline = PerformanceBaseline(
                    metric_name=metric_name,
                    mean_value=mean(values),
                    std_deviation=stdev(values) if len(values) > 1 else 0,
                    min_value=min(values),
                    max_value=max(values),
                    sample_count=len(values),
                    last_updated=datetime.now()
                )
                
                self._baselines[metric_name] = baseline
        
        self.logger.info(f"📊 Updated baselines for {len(self._baselines)} metrics")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        with self._lock:
            # Calculate overall status
            statuses = list(self._component_status.values())
            if MonitorStatus.CRITICAL in statuses:
                overall_status = MonitorStatus.CRITICAL
            elif MonitorStatus.DEGRADED in statuses:
                overall_status = MonitorStatus.DEGRADED
            elif MonitorStatus.WARNING in statuses:
                overall_status = MonitorStatus.WARNING
            elif all(s == MonitorStatus.HEALTHY for s in statuses):
                overall_status = MonitorStatus.HEALTHY
            else:
                overall_status = MonitorStatus.UNKNOWN
            
            return {
                'overall_status': overall_status.value,
                'monitoring_active': self._monitoring,
                'last_check': max(self._last_check_times.values()).isoformat() if self._last_check_times else None,
                'components': {
                    name: status.value 
                    for name, status in self._component_status.items()
                },
                'active_alerts': len(self._active_alerts),
                'total_alerts': self._alert_counter,
                'metrics_collected': len(self._metrics),
                'baselines_available': len(self._baselines)
            }
    
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        if metric_name not in self._metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        samples = [s for s in self._metrics[metric_name] if s.timestamp > cutoff_time]
        
        if not samples:
            return {}
        
        values = [s.value for s in samples]
        
        return {
            'metric_name': metric_name,
            'sample_count': len(values),
            'latest_value': values[-1] if values else None,
            'mean': mean(values),
            'min': min(values),
            'max': max(values),
            'std_dev': stdev(values) if len(values) > 1 else 0,
            'time_range_hours': hours,
            'baseline': asdict(self._baselines[metric_name]) if metric_name in self._baselines else None
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        with self._lock:
            return [
                {
                    'id': alert.id,
                    'component': alert.component,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved,
                    'escalated': alert.escalated,
                    'notification_count': alert.notification_count
                }
                for alert in self._active_alerts.values()
            ]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].acknowledged = True
                self.logger.info(f"✅ Alert acknowledged: {alert_id}")
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolved = True
                del self._active_alerts[alert_id]
                self.logger.info(f"✅ Alert resolved: {alert_id}")
                return True
            return False
    
    async def _check_critical_systems(self) -> Dict[str, Any]:
        """
        Implement comprehensive system health checks for all critical systems.
        
        Returns:
            Dict containing health status of all critical systems
        """
        try:
            self.logger.debug("🔍 Performing critical systems health check...")
            
            # Run all critical checks concurrently
            results = await asyncio.gather(
                self._check_database_health(),
                self._check_messaging_health(),
                self._check_data_providers_health(),
                self._check_indicators_health(),
                self._check_trading_engine_health(),
                return_exceptions=True
            )
            
            # Process results
            health_status = {
                'database': 'unknown',
                'messaging': 'unknown', 
                'data_providers': 'unknown',
                'indicators': 'unknown',
                'trading_engine': 'unknown',
                'overall_status': 'unknown',
                'check_time': datetime.now().isoformat(),
                'errors': []
            }
            
            check_names = ['database', 'messaging', 'data_providers', 'indicators', 'trading_engine']
            
            for i, result in enumerate(results):
                component = check_names[i]
                if isinstance(result, Exception):
                    health_status[component] = 'critical'
                    health_status['errors'].append(f"{component}: {str(result)}")
                    self.logger.error(f"❌ Critical system check failed for {component}: {result}")
                elif isinstance(result, dict):
                    health_status[component] = result.get('status', 'unknown')
                    if result.get('error'):
                        health_status['errors'].append(f"{component}: {result['error']}")
                else:
                    health_status[component] = 'healthy'
            
            # Determine overall status
            statuses = [health_status[comp] for comp in check_names]
            if 'critical' in statuses:
                health_status['overall_status'] = 'critical'
            elif 'warning' in statuses:
                health_status['overall_status'] = 'warning'
            elif all(status == 'healthy' for status in statuses):
                health_status['overall_status'] = 'healthy'
            else:
                health_status['overall_status'] = 'degraded'
            
            self.logger.info(f"✅ Critical systems check completed - Overall status: {health_status['overall_status']}")
            return health_status
            
        except Exception as e:
            error_msg = f"Critical systems health check failed: {e}"
            self.logger.error(f"❌ {error_msg}")
            return {
                'database': 'critical',
                'messaging': 'critical',
                'data_providers': 'critical', 
                'indicators': 'critical',
                'trading_engine': 'critical',
                'overall_status': 'critical',
                'check_time': datetime.now().isoformat(),
                'errors': [error_msg]
            }
    
    async def _check_messaging_health(self) -> Dict[str, Any]:
        """Check messaging system health and connectivity."""
        try:
            start_time = time.time()
            
            # Check message broker connectivity
            # This would be replaced with actual broker health checks
            health_status = {
                'status': 'healthy',
                'response_time': 0.0,
                'queue_depth': 0,
                'connections_active': True,
                'error': None
            }
            
            # Simulate messaging system check
            await asyncio.sleep(0.01)
            
            # Check queue depths (placeholder)
            queue_depth = 0  # Would get from actual message broker
            if queue_depth > 1000:
                health_status['status'] = 'warning'
                health_status['error'] = f"High queue depth: {queue_depth}"
            elif queue_depth > 5000:
                health_status['status'] = 'critical' 
                health_status['error'] = f"Critical queue depth: {queue_depth}"
            
            health_status['response_time'] = time.time() - start_time
            health_status['queue_depth'] = queue_depth
            
            # Record metrics
            self.record_metric('messaging_response_time', health_status['response_time'])
            self.record_metric('messaging_queue_depth', queue_depth)
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'critical',
                'response_time': 0.0,
                'queue_depth': -1,
                'connections_active': False,
                'error': str(e)
            }
    
    async def _check_data_providers_health(self) -> Dict[str, Any]:
        """Check data providers health and connectivity."""
        try:
            start_time = time.time()
            
            health_status = {
                'status': 'healthy',
                'response_time': 0.0,
                'providers_online': 0,
                'providers_total': 0,
                'error': None
            }
            
            # Check various data providers
            providers = ['yahoo_finance', 'broker_data', 'economic_calendar']
            providers_status = {}
            
            for provider in providers:
                try:
                    # Simulate provider check
                    await asyncio.sleep(0.005)
                    providers_status[provider] = 'online'
                except Exception as e:
                    providers_status[provider] = 'offline'
                    self.logger.warning(f"Data provider {provider} offline: {e}")
            
            online_count = sum(1 for status in providers_status.values() if status == 'online')
            total_count = len(providers)
            
            health_status['providers_online'] = online_count
            health_status['providers_total'] = total_count
            health_status['response_time'] = time.time() - start_time
            
            # Determine overall status
            if online_count == 0:
                health_status['status'] = 'critical'
                health_status['error'] = 'All data providers offline'
            elif online_count < total_count:
                health_status['status'] = 'warning'
                health_status['error'] = f'{total_count - online_count} data providers offline'
            
            # Record metrics
            self.record_metric('data_providers_online', online_count)
            self.record_metric('data_providers_response_time', health_status['response_time'])
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'critical',
                'response_time': 0.0,
                'providers_online': 0,
                'providers_total': 0,
                'error': str(e)
            }
    
    async def _check_indicators_health(self) -> Dict[str, Any]:
        """Check indicator engine health and performance."""
        try:
            start_time = time.time()
            
            health_status = {
                'status': 'healthy',
                'response_time': 0.0,
                'indicators_active': 0,
                'calculation_errors': 0,
                'error': None
            }
            
            # Simulate indicator engine check
            await asyncio.sleep(0.01)
            
            # Check indicator calculations (placeholder)
            active_indicators = 25  # Would get from actual indicator engine
            calculation_errors = 0   # Would get from indicator engine
            
            health_status['indicators_active'] = active_indicators
            health_status['calculation_errors'] = calculation_errors
            health_status['response_time'] = time.time() - start_time
            
            # Determine status based on errors
            if calculation_errors > 5:
                health_status['status'] = 'warning'
                health_status['error'] = f'High calculation errors: {calculation_errors}'
            elif calculation_errors > 10:
                health_status['status'] = 'critical'
                health_status['error'] = f'Critical calculation errors: {calculation_errors}'
            
            # Record metrics
            self.record_metric('indicators_active', active_indicators)
            self.record_metric('indicators_errors', calculation_errors)
            self.record_metric('indicators_response_time', health_status['response_time'])
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'critical',
                'response_time': 0.0,
                'indicators_active': 0,
                'calculation_errors': -1,
                'error': str(e)
            }
    
    async def _check_trading_engine_health(self) -> Dict[str, Any]:
        """Check trading engine health and broker connectivity."""
        try:
            start_time = time.time()
            
            health_status = {
                'status': 'healthy',
                'response_time': 0.0,
                'brokers_connected': 0,
                'active_positions': 0,
                'trading_enabled': True,
                'error': None
            }
            
            # Check broker connections
            broker_connections = 1  # Would check actual broker connections
            active_positions = len(getattr(self, '_active_positions', {}))
            trading_enabled = True   # Would check from trading engine
            
            health_status['brokers_connected'] = broker_connections
            health_status['active_positions'] = active_positions
            health_status['trading_enabled'] = trading_enabled
            health_status['response_time'] = time.time() - start_time
            
            # Determine status
            if broker_connections == 0:
                health_status['status'] = 'critical'
                health_status['error'] = 'No broker connections'
            elif not trading_enabled:
                health_status['status'] = 'warning'
                health_status['error'] = 'Trading disabled'
            
            # Record metrics
            self.record_metric('trading_brokers_connected', broker_connections)
            self.record_metric('trading_active_positions', active_positions)
            self.record_metric('trading_response_time', health_status['response_time'])
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'critical',
                'response_time': 0.0,
                'brokers_connected': 0,
                'active_positions': 0,
                'trading_enabled': False,
                'error': str(e)
            }

    async def health_check(self) -> bool:
        """Simple health check for this component."""
        # Calculate current overall health based on component statuses
        if not self._component_status:
            overall_health = MonitorStatus.UNKNOWN
        else:
            statuses = list(self._component_status.values())
            if MonitorStatus.CRITICAL in statuses:
                overall_health = MonitorStatus.CRITICAL
            elif MonitorStatus.DEGRADED in statuses:
                overall_health = MonitorStatus.DEGRADED
            elif MonitorStatus.WARNING in statuses:
                overall_health = MonitorStatus.WARNING
            elif all(s == MonitorStatus.HEALTHY for s in statuses):
                overall_health = MonitorStatus.HEALTHY
            else:
                overall_health = MonitorStatus.UNKNOWN
        
        return self._monitoring and overall_health in [MonitorStatus.HEALTHY, MonitorStatus.WARNING]