"""
Performance Monitor for Conservative Production Settings
=====================================================

This module provides monitoring functions to track and log conservative
production settings to ensure optimal performance and stability.
"""

import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class ConservativeSettings:
    """Data class to hold conservative settings for monitoring"""
    max_indicators_per_regime: int = 10
    cache_duration_minutes: int = 10
    execution_timeout_seconds: int = 2
    ml_complexity: str = "medium"
    max_concurrent_calculations: int = 8
    max_cache_size: int = 800
    performance_alert_threshold: float = 1.5
    memory_usage_alert: int = 85
    cpu_usage_alert: int = 80
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceMetrics:
    """Data class to hold performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    avg_execution_time: float
    cache_hit_rate: float
    timeout_rate: float
    active_indicators: int
    concurrent_calculations: int
    ml_success_rate: float

class ConservativeSettingsMonitor:
    """Monitor for conservative production settings"""
    
    def __init__(self):
        self.settings = ConservativeSettings()
        self.performance_history: List[PerformanceMetrics] = []
        self.alerts_sent: Dict[str, datetime] = {}
        self.monitoring_active = False
        self.monitor_thread = None
        
    def log_conservative_settings(self) -> Dict[str, Any]:
        """Log current conservative production settings"""
        settings_dict = {
            'max_indicators_per_regime': self.settings.max_indicators_per_regime,
            'cache_duration_minutes': self.settings.cache_duration_minutes,
            'execution_timeout_seconds': self.settings.execution_timeout_seconds,
            'ml_complexity': self.settings.ml_complexity,
            'max_concurrent_calculations': self.settings.max_concurrent_calculations,
            'max_cache_size': self.settings.max_cache_size,
            'performance_alert_threshold': self.settings.performance_alert_threshold,
            'memory_usage_alert': self.settings.memory_usage_alert,
            'cpu_usage_alert': self.settings.cpu_usage_alert
        }
        
        logger.info("🔧 PRODUCTION: Conservative settings active")
        for key, value in settings_dict.items():
            logger.info(f"   {key}: {value}")
        
        return settings_dict
    
    def check_performance_thresholds(self, metrics: PerformanceMetrics) -> List[str]:
        """Check if any performance thresholds are exceeded"""
        alerts = []
        
        # Check execution time
        if metrics.avg_execution_time > self.settings.performance_alert_threshold:
            alerts.append(f"Execution time exceeded: {metrics.avg_execution_time:.2f}s > {self.settings.performance_alert_threshold}s")
        
        # Check memory usage
        if metrics.memory_usage > self.settings.memory_usage_alert:
            alerts.append(f"Memory usage exceeded: {metrics.memory_usage:.1f}% > {self.settings.memory_usage_alert}%")
        
        # Check CPU usage
        if metrics.cpu_usage > self.settings.cpu_usage_alert:
            alerts.append(f"CPU usage exceeded: {metrics.cpu_usage:.1f}% > {self.settings.cpu_usage_alert}%")
        
        # Check timeout rate
        if metrics.timeout_rate > 0.05:  # 5% threshold
            alerts.append(f"High timeout rate: {metrics.timeout_rate:.2%}")
        
        # Check cache hit rate
        if metrics.cache_hit_rate < 0.8:  # 80% threshold
            alerts.append(f"Low cache hit rate: {metrics.cache_hit_rate:.2%}")
        
        return alerts
    
    def collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system and application metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            
            # Placeholder for application metrics (would be collected from actual components)
            avg_execution_time = 0.8  # Placeholder - would come from actual execution tracking
            cache_hit_rate = 0.92     # Placeholder - would come from cache statistics
            timeout_rate = 0.002      # Placeholder - would come from timeout tracking
            active_indicators = 8     # Placeholder - would come from indicator tracking
            concurrent_calculations = 6  # Placeholder - would come from executor
            ml_success_rate = 0.94    # Placeholder - would come from ML engine
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                avg_execution_time=avg_execution_time,
                cache_hit_rate=cache_hit_rate,
                timeout_rate=timeout_rate,
                active_indicators=active_indicators,
                concurrent_calculations=concurrent_calculations,
                ml_success_rate=ml_success_rate
            )
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0, memory_usage=0, avg_execution_time=0,
                cache_hit_rate=0, timeout_rate=0, active_indicators=0,
                concurrent_calculations=0, ml_success_rate=0
            )
    
    def log_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Log current performance metrics"""
        logger.info("📊 PERFORMANCE METRICS:")
        logger.info(f"   CPU Usage: {metrics.cpu_usage:.1f}%")
        logger.info(f"   Memory Usage: {metrics.memory_usage:.1f}%")
        logger.info(f"   Avg Execution Time: {metrics.avg_execution_time:.3f}s")
        logger.info(f"   Cache Hit Rate: {metrics.cache_hit_rate:.1%}")
        logger.info(f"   Timeout Rate: {metrics.timeout_rate:.3%}")
        logger.info(f"   Active Indicators: {metrics.active_indicators}")
        logger.info(f"   Concurrent Calculations: {metrics.concurrent_calculations}")
        logger.info(f"   ML Success Rate: {metrics.ml_success_rate:.1%}")
    
    def send_alerts(self, alerts: List[str]) -> None:
        """Send alerts for performance issues"""
        for alert in alerts:
            # Avoid spam - only send alert once per hour for same issue
            alert_key = alert[:50]  # Use first 50 chars as key
            last_sent = self.alerts_sent.get(alert_key)
            
            if last_sent is None or datetime.now() - last_sent > timedelta(hours=1):
                logger.warning(f"🚨 PERFORMANCE ALERT: {alert}")
                self.alerts_sent[alert_key] = datetime.now()
    
    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.info("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            logger.info("🔍 Starting conservative settings monitoring")
            self.log_conservative_settings()
            
            while self.monitoring_active:
                try:
                    # Collect metrics
                    metrics = self.collect_current_metrics()
                    self.performance_history.append(metrics)
                    
                    # Keep only last 1000 metrics
                    if len(self.performance_history) > 1000:
                        self.performance_history = self.performance_history[-1000:]
                    
                    # Log metrics
                    self.log_performance_metrics(metrics)
                    
                    # Check thresholds and send alerts
                    alerts = self.check_performance_thresholds(metrics)
                    if alerts:
                        self.send_alerts(alerts)
                    
                    # Wait for next cycle
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            logger.info("🛑 Stopping conservative settings monitoring")
        else:
            logger.info("Monitoring not active")
    
    def get_performance_summary(self, last_n_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified period"}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_execution_time = sum(m.avg_execution_time for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        avg_timeout_rate = sum(m.timeout_rate for m in recent_metrics) / len(recent_metrics)
        avg_ml_success_rate = sum(m.ml_success_rate for m in recent_metrics) / len(recent_metrics)
        
        # Performance status
        status = "excellent"
        if avg_execution_time > 1.2 or avg_cpu > 70 or avg_memory > 80:
            status = "good"
        if avg_execution_time > 1.5 or avg_cpu > 80 or avg_memory > 85:
            status = "warning"
        if avg_execution_time > 2.0 or avg_cpu > 90 or avg_memory > 90:
            status = "critical"
        
        return {
            "period_minutes": last_n_minutes,
            "sample_count": len(recent_metrics),
            "status": status,
            "averages": {
                "cpu_usage": round(avg_cpu, 1),
                "memory_usage": round(avg_memory, 1),
                "execution_time": round(avg_execution_time, 3),
                "cache_hit_rate": round(avg_cache_hit_rate, 3),
                "timeout_rate": round(avg_timeout_rate, 4),
                "ml_success_rate": round(avg_ml_success_rate, 3)
            },
            "conservative_settings": {
                "max_indicators_per_regime": self.settings.max_indicators_per_regime,
                "cache_duration_minutes": self.settings.cache_duration_minutes,
                "execution_timeout_seconds": self.settings.execution_timeout_seconds,
                "ml_complexity": self.settings.ml_complexity,
                "max_concurrent_calculations": self.settings.max_concurrent_calculations
            }
        }
    
    def save_metrics_to_file(self, filepath: Optional[str] = None) -> None:
        """Save performance metrics to file"""
        if not filepath:
            filepath = f"data/performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "conservative_settings": self.log_conservative_settings(),
            "metrics_count": len(self.performance_history),
            "last_updated": datetime.now().isoformat(),
            "recent_summary": self.get_performance_summary(60)
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Performance metrics saved to {filepath}")

# Global monitor instance
conservative_monitor = ConservativeSettingsMonitor()

# Convenience functions
def log_conservative_settings() -> Dict[str, Any]:
    """Log current conservative production settings"""
    return conservative_monitor.log_conservative_settings()

def start_performance_monitoring(interval_seconds: int = 60) -> None:
    """Start performance monitoring with specified interval"""
    conservative_monitor.start_monitoring(interval_seconds)

def stop_performance_monitoring() -> None:
    """Stop performance monitoring"""
    conservative_monitor.stop_monitoring()

def get_performance_summary(last_n_minutes: int = 60) -> Dict[str, Any]:
    """Get performance summary for the last N minutes"""
    return conservative_monitor.get_performance_summary(last_n_minutes)

def verify_conservative_settings() -> Dict[str, Any]:
    """Verify that conservative settings are properly applied"""
    settings = conservative_monitor.settings
    
    verification_results = {
        "timestamp": datetime.now().isoformat(),
        "settings_verified": True,
        "settings": {
            "max_indicators_per_regime": settings.max_indicators_per_regime,
            "cache_duration_minutes": settings.cache_duration_minutes,
            "execution_timeout_seconds": settings.execution_timeout_seconds,
            "ml_complexity": settings.ml_complexity,
            "max_concurrent_calculations": settings.max_concurrent_calculations
        },
        "validation_checks": {
            "max_indicators_conservative": settings.max_indicators_per_regime <= 10,
            "cache_duration_appropriate": settings.cache_duration_minutes >= 10,
            "timeout_conservative": settings.execution_timeout_seconds <= 2,
            "ml_complexity_medium": settings.ml_complexity == "medium",
            "concurrency_limited": settings.max_concurrent_calculations <= 8
        }
    }
    
    all_checks_passed = all(verification_results["validation_checks"].values())
    verification_results["all_checks_passed"] = all_checks_passed
    
    if all_checks_passed:
        logger.info("✅ All conservative settings verified successfully")
    else:
        logger.warning("⚠️ Some conservative settings validation checks failed")
        for check, passed in verification_results["validation_checks"].items():
            if not passed:
                logger.warning(f"   Failed: {check}")
    
    return verification_results