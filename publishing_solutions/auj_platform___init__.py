# AUJ Platform Package
"""
Advanced AI Trading Platform for Humanitarian Impact
===================================================

Mission: Generate sustainable profits through intelligent trading
to help sick children and families in need.
"""

__version__ = "1.0.0"
__author__ = "AUJ Platform Development Team"
__email__ = "support@auj-platform.com"

# Core exports
from .src.core.unified_config import get_unified_config, UnifiedConfigManager
from .src.config import get_config, Config

__all__ = [
    '__version__',
    'get_unified_config',
    'UnifiedConfigManager',
    'get_config',
    'Config'
]
