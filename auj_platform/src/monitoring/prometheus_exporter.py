#!/usr/bin/env python3
"""
Prometheus Exporter for AUJ Platform
====================================

HTTP server that exposes metrics in Prometheus format for scraping.
Provides a standardized endpoint for all platform metrics including
agent performance, system health, and trading statistics.

Author: AUJ Platform Development Team
Date: 2025-07-01
Version: 1.0.0
"""

import asyncio
import logging
from typing import Optional
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from aiohttp import web, ClientTimeout
import threading
import time


class PrometheusExporter:
    """
    Prometheus metrics exporter for AUJ platform.
    
    Provides HTTP endpoint for Prometheus to scrape metrics.
    Integrates with MetricsCollector to expose all platform metrics.
    """
    
    def __init__(self, config=None, metrics_collector=None):
        """Initialize Prometheus exporter."""
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.config = config or {}
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Server configuration
        self.host = self.config_manager.get_dict('monitoring', {}).get('prometheus_host', '0.0.0.0')
        self.port = self.config_manager.get_dict('monitoring', {}).get('prometheus_port', 9090)
        
        # Server state
        self.app = None
        self.runner = None
        self.site = None
        self._server_started = False
        
        self.logger.info(f"🔧 PrometheusExporter configured on {self.host}:{self.port}")
    
    async def initialize(self):
        """Initialize the Prometheus exporter."""
        try:
            self.logger.info("🔧 Initializing Prometheus exporter...")
            
            # Create aiohttp application
            self.app = web.Application()
            
            # Add routes
            self.app.router.add_get('/metrics', self._metrics_handler)
            self.app.router.add_get('/health', self._health_handler)
            self.app.router.add_get('/', self._index_handler)
            
            self.logger.info("✅ Prometheus exporter initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Prometheus exporter: {e}")
            return False
    
    async def start_server(self):
        """Start the Prometheus metrics server."""
        try:
            if self._server_started:
                self.logger.warning("⚠️ Prometheus server already running")
                return
            
            self.logger.info(f"🚀 Starting Prometheus server on {self.host}:{self.port}")
            
            # Create runner
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            # Create site
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            
            self._server_started = True
            self.logger.info(f"✅ Prometheus server started on http://{self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start Prometheus server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the Prometheus metrics server."""
        try:
            if not self._server_started:
                return
            
            self.logger.info("🛑 Stopping Prometheus server...")
            
            if self.site:
                await self.site.stop()
            
            if self.runner:
                await self.runner.cleanup()
            
            self._server_started = False
            self.logger.info("✅ Prometheus server stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping Prometheus server: {e}")
    
    async def _metrics_handler(self, request):
        """Handle /metrics endpoint for Prometheus scraping."""
        try:
            start_time = time.time()
            
            # Generate metrics from collector's registry
            metrics_data = generate_latest(self.metrics_collector.registry)
            
            # Record metrics generation time
            generation_time = time.time() - start_time
            if generation_time > 0.1:  # Log if it takes more than 100ms
                self.logger.warning(f"⚠️ Slow metrics generation: {generation_time:.3f}s")
            
            # Return Prometheus formatted metrics
            return web.Response(
                body=metrics_data,
                content_type=CONTENT_TYPE_LATEST
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error generating metrics: {e}")
            return web.Response(
                text=f"Error generating metrics: {e}",
                status=500
            )
    
    async def _health_handler(self, request):
        """Handle /health endpoint for health checks."""
        try:
            # Check metrics collector health
            collector_healthy = await self.metrics_collector.health_check()
            
            # Check server health
            server_healthy = self._server_started
            
            # Overall health
            overall_healthy = collector_healthy and server_healthy
            
            health_data = {
                'status': 'healthy' if overall_healthy else 'unhealthy',
                'timestamp': time.time(),
                'components': {
                    'metrics_collector': 'healthy' if collector_healthy else 'unhealthy',
                    'prometheus_server': 'healthy' if server_healthy else 'unhealthy'
                }
            }
            
            status_code = 200 if overall_healthy else 503
            
            return web.json_response(health_data, status=status_code)
            
        except Exception as e:
            self.logger.error(f"❌ Error in health check: {e}")
            return web.json_response(
                {'status': 'error', 'error': str(e)},
                status=500
            )
    
    async def _index_handler(self, request):
        """Handle root endpoint with information."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AUJ Platform Metrics</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { color: #2c3e50; }
                .mission { color: #e74c3c; font-weight: bold; }
                .endpoint { background: #ecf0f1; padding: 10px; margin: 10px 0; }
                a { color: #3498db; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1 class="header">🌟 AUJ Platform Metrics Exporter</h1>
            <p class="mission">💝 Mission: Generate sustainable profits for sick children and families</p>
            
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <strong><a href="/metrics">/metrics</a></strong> - Prometheus metrics endpoint
            </div>
            <div class="endpoint">
                <strong><a href="/health">/health</a></strong> - Health check endpoint
            </div>
            
            <h2>Metrics Categories:</h2>
            <ul>
                <li><strong>Agent Performance:</strong> Win rates, execution times, P&L</li>
                <li><strong>System Health:</strong> CPU, memory, disk usage</li>
                <li><strong>Trading Metrics:</strong> Positions, orders, daily P&L</li>
                <li><strong>Indicator Performance:</strong> Calculation times, effectiveness</li>
                <li><strong>API Metrics:</strong> Request times, error rates</li>
            </ul>
            
            <p><em>AUJ Platform - Advanced Learning & Anti-Overfitting Framework</em></p>
        </body>
        </html>
        """
        
        return web.Response(text=html_content, content_type='text/html')
    
    def get_server_info(self):
        """Get server information."""
        return {
            'host': self.host,
            'port': self.port,
            'running': self._server_started,
            'metrics_url': f'http://{self.host}:{self.port}/metrics',
            'health_url': f'http://{self.host}:{self.port}/health'
        }