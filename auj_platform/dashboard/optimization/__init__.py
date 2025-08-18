"""
AUJ Platform Optimization Dashboard Module

This module provides comprehensive dashboard components for
AUJ Platform's strategic optimization features.

Components:
- optimization_dashboard: Main optimization overview
- optimization_metrics: Real-time metrics and analytics  
- optimization_controls: Configuration and control interface

Author: AUJ Platform Development Team
Version: 1.0.0
"""

from .optimization_dashboard import optimization_dashboard_tab
from .optimization_metrics import optimization_metrics_tab
from .optimization_controls import optimization_controls_tab

__all__ = [
    'optimization_dashboard_tab',
    'optimization_metrics_tab', 
    'optimization_controls_tab'
]