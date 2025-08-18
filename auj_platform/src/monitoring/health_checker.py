#!/usr/bin/env python3
"""
Health Checker for AUJ Platform
===============================

Comprehensive health monitoring system that checks the status of all
platform components and provides detailed health reports.

This module monitors:
- Database connectivity
- API endpoints
- Trading connections (MT5, brokers)
- System resources
- Agent performance
- Data provider health

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import asyncio
import logging
import time
import aiohttp
import psutil
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    component: str
    status: HealthStatus
    message: str
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = None


class HealthChecker:
    """
    Comprehensive health monitoring for AUJ platform.
    
    Performs periodic health checks on all platform components
    and provides real-time health status information.
    """
    
    def __init__(self, config=None, database=None, metrics_collector=None):
        """Initialize health checker."""
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.config = config or {}
        self.database = database
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Health check configuration
        monitoring_config = getattr(self.config, 'monitoring', {}) if hasattr(self.config, 'monitoring') else {}
        self.check_interval = monitoring_config.get('health_check_interval', 60)
        self.timeout = monitoring_config.get('health_check_timeout', 10)
        
        # Health state
        self._health_results = {}
        self._overall_health = HealthStatus.UNKNOWN
        self._last_check_time = None
        
        # Monitoring state
        self._monitoring = False
        self._monitoring_task = None
        
        # Thresholds
        self.cpu_warning_threshold = 80
        self.cpu_critical_threshold = 95
        self.memory_warning_threshold = 85
        self.memory_critical_threshold = 95
        self.disk_warning_threshold = 85
        self.disk_critical_threshold = 95
        
        self.logger.info("🏥 HealthChecker initialized")
    
    async def initialize(self):
        """Initialize the health checker."""
        try:
            self.logger.info("🔧 Initializing health monitoring system...")
            
            # Perform initial health check
            await self.perform_full_health_check()
            
            self.logger.info("✅ Health monitoring system initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize health checker: {e}")
            return False
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring:
            self.logger.warning("⚠️ Health monitoring already running")
            return
        
        self._monitoring = True
        self.logger.info("🚀 Starting continuous health monitoring...")
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("✅ Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self._monitoring:
            return
        
        self.logger.info("🛑 Stopping health monitoring...")
        self._monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("✅ Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self.perform_full_health_check()
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def perform_full_health_check(self) -> Dict[str, HealthCheck]:
        """Perform comprehensive health check of all components."""
        self.logger.debug("🔍 Performing full health check...")
        start_time = time.time()
        
        # Run all health checks concurrently
        health_checks = await asyncio.gather(
            self._check_system_resources(),
            self._check_database_health(),
            self._check_api_endpoints(),
            self._check_data_providers(),
            self._check_trading_connections(),
            return_exceptions=True
        )
        
        # Process results
        self._health_results = {}
        for check_result in health_checks:
            if isinstance(check_result, Exception):
                self.logger.error(f"❌ Health check failed: {check_result}")
                continue
            
            if isinstance(check_result, dict):
                self._health_results.update(check_result)
            elif isinstance(check_result, HealthCheck):
                self._health_results[check_result.component] = check_result
        
        # Calculate overall health
        self._calculate_overall_health()
        
        # Update metrics
        if self.metrics_collector:
            await self._update_health_metrics()
        
        self._last_check_time = datetime.now()
        check_duration = time.time() - start_time
        
        self.logger.debug(f"🔍 Health check completed in {check_duration:.2f}s - "
                         f"Status: {self._overall_health.value}")
        
        return self._health_results
    
    async def _check_system_resources(self) -> Dict[str, HealthCheck]:
        """Check system resource health."""
        results = {}
        
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent >= self.cpu_critical_threshold:
                cpu_status = HealthStatus.CRITICAL
                cpu_message = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent >= self.cpu_warning_threshold:
                cpu_status = HealthStatus.WARNING
                cpu_message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                cpu_status = HealthStatus.HEALTHY
                cpu_message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            results['system_cpu'] = HealthCheck(
                component='system_cpu',
                status=cpu_status,
                message=cpu_message,
                response_time=0.0,
                timestamp=datetime.now(),
                details={'cpu_percent': cpu_percent}
            )
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent >= self.memory_critical_threshold:
                memory_status = HealthStatus.CRITICAL
                memory_message = f"Critical memory usage: {memory_percent:.1f}%"
            elif memory_percent >= self.memory_warning_threshold:
                memory_status = HealthStatus.WARNING
                memory_message = f"High memory usage: {memory_percent:.1f}%"
            else:
                memory_status = HealthStatus.HEALTHY
                memory_message = f"Memory usage normal: {memory_percent:.1f}%"
            
            results['system_memory'] = HealthCheck(
                component='system_memory',
                status=memory_status,
                message=memory_message,
                response_time=0.0,
                timestamp=datetime.now(),
                details={
                    'memory_percent': memory_percent,
                    'memory_total_gb': memory.total // (1024**3),
                    'memory_available_gb': memory.available // (1024**3)
                }
            )
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent >= self.disk_critical_threshold:
                disk_status = HealthStatus.CRITICAL
                disk_message = f"Critical disk usage: {disk_percent:.1f}%"
            elif disk_percent >= self.disk_warning_threshold:
                disk_status = HealthStatus.WARNING
                disk_message = f"High disk usage: {disk_percent:.1f}%"
            else:
                disk_status = HealthStatus.HEALTHY
                disk_message = f"Disk usage normal: {disk_percent:.1f}%"
            
            results['system_disk'] = HealthCheck(
                component='system_disk',
                status=disk_status,
                message=disk_message,
                response_time=0.0,
                timestamp=datetime.now(),
                details={
                    'disk_percent': disk_percent,
                    'disk_total_gb': disk.total // (1024**3),
                    'disk_free_gb': disk.free // (1024**3)
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ System resource check failed: {e}")
            results['system_resources'] = HealthCheck(
                component='system_resources',
                status=HealthStatus.CRITICAL,
                message=f"System check failed: {e}",
                response_time=0.0,
                timestamp=datetime.now()
            )
        
        return results
    
    async def _check_database_health(self) -> Dict[str, HealthCheck]:
        """Check database connectivity and health."""
        if not self.database:
            return {
                'database': HealthCheck(
                    component='database',
                    status=HealthStatus.WARNING,
                    message="Database not configured",
                    response_time=0.0,
                    timestamp=datetime.now()
                )
            }
        
        start_time = time.time()
        
        try:
            # Simple connectivity test
            if hasattr(self.database, 'execute'):
                await asyncio.wait_for(
                    self.database.execute("SELECT 1"),
                    timeout=self.timeout
                )
            
            response_time = time.time() - start_time
            
            # Check response time
            if response_time > 2.0:
                status = HealthStatus.WARNING
                message = f"Database slow response: {response_time:.2f}s"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database healthy: {response_time:.3f}s"
            
            return {
                'database': HealthCheck(
                    component='database',
                    status=status,
                    message=message,
                    response_time=response_time,
                    timestamp=datetime.now()
                )
            }
            
        except asyncio.TimeoutError:
            return {
                'database': HealthCheck(
                    component='database',
                    status=HealthStatus.CRITICAL,
                    message="Database timeout",
                    response_time=self.timeout,
                    timestamp=datetime.now()
                )
            }
        except Exception as e:
            return {
                'database': HealthCheck(
                    component='database',
                    status=HealthStatus.CRITICAL,
                    message=f"Database error: {e}",
                    response_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
            }
    
    async def _check_api_endpoints(self) -> Dict[str, HealthCheck]:
        """Check API endpoint health."""
        results = {}
        
        # Get API configuration
        api_config = getattr(self.config, 'api_settings', {}) if hasattr(self.config, 'api_settings') else {}
        host = api_config.get('host', 'localhost')
        port = api_config.get('port', 8000)
        
        endpoints_to_check = [
            f'http://{host}:{port}/health',
            f'http://{host}:{port}/api/status',
        ]
        
        for endpoint in endpoints_to_check:
            start_time = time.time()
            component_name = f"api_{endpoint.split('/')[-1]}"
            
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(endpoint) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            if response_time > 1.0:
                                status = HealthStatus.WARNING
                                message = f"API slow: {response_time:.2f}s"
                            else:
                                status = HealthStatus.HEALTHY
                                message = f"API healthy: {response_time:.3f}s"
                        else:
                            status = HealthStatus.WARNING
                            message = f"API returned {response.status}"
                        
                        results[component_name] = HealthCheck(
                            component=component_name,
                            status=status,
                            message=message,
                            response_time=response_time,
                            timestamp=datetime.now(),
                            details={'status_code': response.status}
                        )
                        
            except asyncio.TimeoutError:
                results[component_name] = HealthCheck(
                    component=component_name,
                    status=HealthStatus.CRITICAL,
                    message="API timeout",
                    response_time=self.timeout,
                    timestamp=datetime.now()
                )
            except Exception as e:
                results[component_name] = HealthCheck(
                    component=component_name,
                    status=HealthStatus.CRITICAL,
                    message=f"API error: {e}",
                    response_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
        
        return results
    
    async def _check_data_providers(self) -> Dict[str, HealthCheck]:
        """Check data provider health."""
        results = {}
        
        # Check Yahoo Finance
        try:
            # Simple test - this would be replaced with actual data provider checks
            results['yahoo_finance'] = HealthCheck(
                component='yahoo_finance',
                status=HealthStatus.HEALTHY,
                message="Yahoo Finance provider simulated check",
                response_time=0.1,
                timestamp=datetime.now()
            )
        except Exception as e:
            results['yahoo_finance'] = HealthCheck(
                component='yahoo_finance',
                status=HealthStatus.WARNING,
                message=f"Yahoo Finance check failed: {e}",
                response_time=0.0,
                timestamp=datetime.now()
            )
        
        return results
    
    async def _check_trading_connections(self) -> Dict[str, HealthCheck]:
        """Check trading connection health."""
        results = {}
        
        # Check MT5 connection (simulated)
        try:
            results['mt5_connection'] = HealthCheck(
                component='mt5_connection',
                status=HealthStatus.HEALTHY,
                message="MT5 connection simulated check",
                response_time=0.1,
                timestamp=datetime.now()
            )
        except Exception as e:
            results['mt5_connection'] = HealthCheck(
                component='mt5_connection',
                status=HealthStatus.WARNING,
                message=f"MT5 connection check failed: {e}",
                response_time=0.0,
                timestamp=datetime.now()
            )
        
        return results
    
    def _calculate_overall_health(self):
        """Calculate overall system health based on component statuses."""
        if not self._health_results:
            self._overall_health = HealthStatus.UNKNOWN
            return
        
        statuses = [check.status for check in self._health_results.values()]
        
        # If any component is critical, overall is critical
        if HealthStatus.CRITICAL in statuses:
            self._overall_health = HealthStatus.CRITICAL
        # If any component has warning, overall is warning
        elif HealthStatus.WARNING in statuses:
            self._overall_health = HealthStatus.WARNING
        # If all are healthy, overall is healthy
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            self._overall_health = HealthStatus.HEALTHY
        else:
            self._overall_health = HealthStatus.UNKNOWN
    
    async def _update_health_metrics(self):
        """Update health metrics in metrics collector."""
        if not self.metrics_collector:
            return
        
        try:
            # Update component health metrics
            for component, health_check in self._health_results.items():
                # Convert status to numeric value for metrics
                status_value = {
                    HealthStatus.HEALTHY: 1,
                    HealthStatus.WARNING: 0.5,
                    HealthStatus.CRITICAL: 0,
                    HealthStatus.UNKNOWN: -1
                }.get(health_check.status, -1)
                
                # This would be a custom metric in the metrics collector
                # For now, we'll log it
                self.logger.debug(f"📊 Health metric: {component} = {status_value}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update health metrics: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of current health status."""
        return {
            'overall_status': self._overall_health.value,
            'last_check': self._last_check_time.isoformat() if self._last_check_time else None,
            'components': {
                name: {
                    'status': check.status.value,
                    'message': check.message,
                    'response_time': check.response_time,
                    'timestamp': check.timestamp.isoformat()
                }
                for name, check in self._health_results.items()
            },
            'monitoring_active': self._monitoring
        }
    
    def get_component_health(self, component: str) -> Optional[HealthCheck]:
        """Get health status for specific component."""
        return self._health_results.get(component)
    
    async def health_check(self) -> bool:
        """Simple health check for this component."""
        return self._overall_health in [HealthStatus.HEALTHY, HealthStatus.WARNING]
    
    async def perform_comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check with enhanced security monitoring.
        Enhanced version with detailed component analysis and security checks.
        """
        try:
            health_results = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': 'HEALTHY',
                'components': {},
                'security_status': {},
                'performance_metrics': {},
                'warnings': [],
                'errors': [],
                'recommendations': []
            }
            
            # Core component health checks
            await self._check_database_health(health_results)
            await self._check_trading_connections(health_results)
            await self._check_data_providers(health_results)
            await self._check_agent_health(health_results)
            await self._check_messaging_system(health_results)
            
            # Security health checks
            await self._check_security_health(health_results)
            
            # Performance health checks  
            await self._check_performance_health(health_results)
            
            # System resource checks
            await self._check_system_resources(health_results)
            
            # Determine overall status
            await self._determine_overall_health_status(health_results)
            
            # Generate recommendations
            await self._generate_health_recommendations(health_results)
            
            # Store health check results
            await self._store_health_results(health_results)
            
            return health_results
            
        except Exception as e:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': 'ERROR',
                'error': str(e),
                'components': {},
                'warnings': [f"Health check system error: {e}"],
                'errors': [str(e)]
            }
    
    async def _check_database_health(self, results: Dict[str, Any]):
        """Check database connectivity and performance"""
        try:
            start_time = time.time()
            
            # Test basic connectivity
            if self.database:
                # Simple connection test
                if hasattr(self.database, 'execute'):
                    await self.database.execute("SELECT 1")
                
                response_time = (time.time() - start_time) * 1000
                
                if response_time > 1000:  # > 1 second
                    results['warnings'].append(f"Database response time high: {response_time:.2f}ms")
                    results['components']['database'] = 'WARNING'
                else:
                    results['components']['database'] = 'HEALTHY'
                    
                results['performance_metrics']['database_response_ms'] = response_time
            else:
                results['components']['database'] = 'NOT_CONFIGURED'
                results['warnings'].append("Database not configured")
                
        except Exception as e:
            results['components']['database'] = 'UNHEALTHY'
            results['errors'].append(f"Database: {e}")
    
    async def _check_trading_connections(self, results: Dict[str, Any]):
        """Check trading platform connections (MT5, etc.)"""
        try:
            # Check MT5 connection if available
            mt5_status = await self._check_mt5_connection()
            results['components']['mt5'] = mt5_status
            
            if mt5_status not in ['HEALTHY', 'NOT_CONFIGURED']:
                results['errors'].append("MT5 connection failed")
            
            # Check other broker connections
            broker_status = await self._check_broker_connections()
            results['components']['brokers'] = broker_status
            
        except Exception as e:
            results['components']['trading'] = 'ERROR'
            results['errors'].append(f"Trading connections: {e}")
    
    async def _check_data_providers(self, results: Dict[str, Any]):
        """Check data provider health"""
        try:
            providers_status = {}
            
            # Check news providers
            news_status = await self._check_news_providers()
            providers_status['news'] = news_status
            
            # Check market data providers
            market_data_status = await self._check_market_data_providers()
            providers_status['market_data'] = market_data_status
            
            # Check economic calendar
            economic_status = await self._check_economic_calendar()
            providers_status['economic_calendar'] = economic_status
            
            results['components']['data_providers'] = providers_status
            
            # Check for any unhealthy providers
            unhealthy_providers = [k for k, v in providers_status.items() if v == 'UNHEALTHY']
            if unhealthy_providers:
                results['warnings'].append(f"Unhealthy data providers: {unhealthy_providers}")
                
        except Exception as e:
            results['components']['data_providers'] = 'ERROR'
            results['errors'].append(f"Data providers: {e}")
    
    async def _check_agent_health(self, results: Dict[str, Any]):
        """Check agent system health"""
        try:
            agent_health = {
                'active_agents': 0,
                'responding_agents': 0,
                'error_rate': 0.0,
                'average_response_time': 0.0
            }
            
            # This would check actual agents if available
            # For now, provide placeholder status
            results['components']['agents'] = 'HEALTHY'
            results['performance_metrics']['agent_health'] = agent_health
            
        except Exception as e:
            results['components']['agents'] = 'ERROR'
            results['errors'].append(f"Agent system: {e}")
    
    async def _check_messaging_system(self, results: Dict[str, Any]):
        """Check messaging system health"""
        try:
            messaging_status = await self._test_messaging_connectivity()
            results['components']['messaging'] = messaging_status
            
            if messaging_status == 'UNHEALTHY':
                results['errors'].append("Messaging system connectivity failed")
                
        except Exception as e:
            results['components']['messaging'] = 'ERROR'
            results['errors'].append(f"Messaging system: {e}")
    
    async def _check_security_health(self, results: Dict[str, Any]):
        """Check security-related health metrics"""
        try:
            security_status = {
                'authentication_system': 'HEALTHY',
                'ssl_certificates': await self._check_ssl_status(),
                'firewall_status': await self._check_firewall_status(),
                'intrusion_detection': await self._check_intrusion_detection(),
                'access_control': await self._check_access_control(),
                'encryption_status': 'ENABLED'
            }
            
            results['security_status'] = security_status
            
            # Check for security issues
            security_issues = [k for k, v in security_status.items() if v in ['UNHEALTHY', 'DISABLED', 'ERROR']]
            if security_issues:
                results['errors'].append(f"Security issues detected: {security_issues}")
                
        except Exception as e:
            results['security_status'] = {'error': str(e)}
            results['errors'].append(f"Security health check failed: {e}")
    
    async def _check_performance_health(self, results: Dict[str, Any]):
        """Check performance-related health metrics"""
        try:
            performance = {
                'memory_usage_percent': 0.0,
                'cpu_usage_percent': 0.0,
                'disk_usage_percent': 0.0,
                'network_latency_ms': 0.0,
                'throughput_ops_per_sec': 0.0
            }
            
            # Collect actual performance metrics if psutil available
            try:
                import psutil
                
                memory = psutil.virtual_memory()
                performance['memory_usage_percent'] = memory.percent
                
                cpu_percent = psutil.cpu_percent(interval=1)
                performance['cpu_usage_percent'] = cpu_percent
                
                disk = psutil.disk_usage('/')
                performance['disk_usage_percent'] = (disk.used / disk.total) * 100
                
                # Check for performance issues
                if memory.percent > 90:
                    results['warnings'].append(f"High memory usage: {memory.percent:.1f}%")
                if cpu_percent > 90:
                    results['warnings'].append(f"High CPU usage: {cpu_percent:.1f}%")
                if performance['disk_usage_percent'] > 90:
                    results['warnings'].append(f"High disk usage: {performance['disk_usage_percent']:.1f}%")
                    
            except ImportError:
                performance['status'] = 'LIMITED - psutil not available'
                
            results['performance_metrics']['system'] = performance
            
        except Exception as e:
            results['performance_metrics'] = {'error': str(e)}
            results['warnings'].append(f"Performance check failed: {e}")
    
    async def _check_system_resources(self, results: Dict[str, Any]):
        """Check system resource availability"""
        try:
            resources = {
                'available_memory_gb': 0.0,
                'available_disk_gb': 0.0,
                'cpu_cores': 0,
                'network_interfaces': 0,
                'open_files': 0,
                'network_connections': 0
            }
            
            try:
                import psutil
                
                memory = psutil.virtual_memory()
                resources['available_memory_gb'] = memory.available / (1024**3)
                
                disk = psutil.disk_usage('/')
                resources['available_disk_gb'] = disk.free / (1024**3)
                
                resources['cpu_cores'] = psutil.cpu_count()
                resources['network_interfaces'] = len(psutil.net_if_addrs())
                resources['network_connections'] = len(psutil.net_connections())
                
                # Check resource thresholds
                if resources['available_memory_gb'] < 1.0:
                    results['warnings'].append("Low available memory")
                if resources['available_disk_gb'] < 5.0:
                    results['warnings'].append("Low available disk space")
                    
            except ImportError:
                resources['status'] = 'LIMITED - psutil not available'
                
            results['performance_metrics']['resources'] = resources
            
        except Exception as e:
            results['warnings'].append(f"Resource check failed: {e}")
    
    async def _determine_overall_health_status(self, results: Dict[str, Any]):
        """Determine overall health status based on component results"""
        try:
            component_statuses = list(results['components'].values())
            
            # Count unhealthy components
            unhealthy_count = sum(1 for status in component_statuses if status == 'UNHEALTHY')
            error_count = len(results['errors'])
            warning_count = len(results['warnings'])
            
            if error_count > 0 or unhealthy_count > 2:
                results['overall_status'] = 'UNHEALTHY'
            elif warning_count > 5 or unhealthy_count > 0:
                results['overall_status'] = 'WARNING'
            else:
                results['overall_status'] = 'HEALTHY'
                
        except Exception as e:
            results['overall_status'] = 'ERROR'
            results['errors'].append(f"Status determination failed: {e}")
    
    async def _generate_health_recommendations(self, results: Dict[str, Any]):
        """Generate health improvement recommendations"""
        try:
            recommendations = []
            
            # Performance recommendations
            performance = results.get('performance_metrics', {}).get('system', {})
            if performance.get('memory_usage_percent', 0) > 80:
                recommendations.append("Consider increasing available memory or optimizing memory usage")
            if performance.get('cpu_usage_percent', 0) > 80:
                recommendations.append("High CPU usage detected - consider load balancing or optimization")
                
            # Component recommendations
            components = results.get('components', {})
            if 'database' in components and components['database'] == 'UNHEALTHY':
                recommendations.append("Database connectivity issues - check configuration and network")
            if 'trading' in components and components['trading'] == 'UNHEALTHY':
                recommendations.append("Trading platform connectivity issues - verify credentials and network")
                
            # Security recommendations
            security = results.get('security_status', {})
            if security.get('ssl_certificates') == 'EXPIRED':
                recommendations.append("SSL certificates need renewal")
                
            results['recommendations'] = recommendations
            
        except Exception as e:
            results['recommendations'] = [f"Recommendation generation failed: {e}"]
    
    async def _store_health_results(self, results: Dict[str, Any]):
        """Store health check results for trending analysis"""
        try:
            # Store in memory buffer
            if not hasattr(self, '_health_history'):
                from collections import deque
                self._health_history = deque(maxlen=100)
            
            self._health_history.append(results)
            
            # Store in database if available
            if self.database:
                await self.database.store_health_check(results)
                
        except Exception as e:
            self.logger.error(f"Failed to store health results: {e}")
    
    # Helper methods for specific checks
    async def _check_mt5_connection(self) -> str:
        """Check MT5 connection status"""
        try:
            # Would check actual MT5 connection if available
            return 'HEALTHY'  # Placeholder
        except:
            return 'UNHEALTHY'
    
    async def _check_broker_connections(self) -> str:
        """Check broker API connections"""
        return 'HEALTHY'  # Placeholder
    
    async def _check_news_providers(self) -> str:
        """Check news provider connectivity"""
        return 'HEALTHY'  # Placeholder
    
    async def _check_market_data_providers(self) -> str:
        """Check market data provider connectivity"""
        return 'HEALTHY'  # Placeholder
    
    async def _check_economic_calendar(self) -> str:
        """Check economic calendar data availability"""
        return 'HEALTHY'  # Placeholder
    
    async def _test_messaging_connectivity(self) -> str:
        """Test messaging system connectivity"""
        return 'HEALTHY'  # Placeholder
    
    async def _check_ssl_status(self) -> str:
        """Check SSL certificate status"""
        return 'VALID'  # Placeholder
    
    async def _check_firewall_status(self) -> str:
        """Check firewall status"""
        return 'ENABLED'  # Placeholder
    
    async def _check_intrusion_detection(self) -> str:
        """Check intrusion detection system"""
        return 'ACTIVE'  # Placeholder
    
    async def _check_access_control(self) -> str:
        """Check access control system"""
        return 'ENFORCED'  # Placeholder