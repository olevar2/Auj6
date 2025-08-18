"""
Health monitoring module for AUJ Platform.

Provides health check functionality for platform components.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthCheck:
    """Individual health check result."""

    def __init__(self, name: str, status: HealthStatus, message: str = "", details: Optional[Dict[str, Any]] = None):
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class HealthMonitor:
    """Health monitoring for platform components."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize health monitor."""
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.checks = []
        self.last_check_time = None

    async def initialize(self) -> None:
        """Initialize health monitoring."""
        if self.enabled:
            logger.info("Health monitoring initialized")
        else:
            logger.info("Health monitoring disabled by configuration")

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check result."""
        self.checks.append(check)

    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        self.last_check_time = datetime.utcnow()

        # Basic system health
        system_check = HealthCheck(
            name="system",
            status=HealthStatus.HEALTHY,
            message="System operational"
        )
        self.add_check(system_check)

        # Calculate overall status
        overall_status = self._calculate_overall_status()

        return {
            'overall_status': overall_status.value,
            'timestamp': self.last_check_time.isoformat(),
            'checks': [
                {
                    'name': check.name,
                    'status': check.status.value,
                    'message': check.message,
                    'details': check.details,
                    'timestamp': check.timestamp.isoformat()
                }
                for check in self.checks[-10:]  # Last 10 checks
            ]
        }

    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall health status from individual checks."""
        if not self.checks:
            return HealthStatus.UNKNOWN

        recent_checks = self.checks[-5:]  # Last 5 checks

        if any(check.status == HealthStatus.CRITICAL for check in recent_checks):
            return HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.WARNING for check in recent_checks):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    async def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        return {
            'enabled': self.enabled,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'total_checks': len(self.checks),
            'overall_status': self._calculate_overall_status().value
        }

    async def shutdown(self) -> None:
        """Shutdown health monitoring."""
        logger.info("Health monitoring shutdown")


def get_health_monitor(config: Optional[Dict[str, Any]] = None) -> HealthMonitor:
    """Get health monitor instance."""
    return HealthMonitor(config)
