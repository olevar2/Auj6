# AUJ Platform Package
# This file makes the auj_platform directory a Python package

__version__ = "1.0.0"
__author__ = "AUJ Platform Development Team"
__email__ = "support@auj-platform.com"

# Make core modules available at package level
from .src.core.unified_config import get_unified_config, UnifiedConfigManager
from .src.config import get_config, Config

__all__ = ['__version__', 'get_unified_config', 'UnifiedConfigManager', 'get_config', 'Config']
