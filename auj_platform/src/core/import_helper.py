"""
AUJ Platform Import Helper
=========================
Provides convenient import utilities and path management for the AUJ Platform.

Usage:
    from core.import_helper import setup_environment, safe_import
    
    # Setup environment (call once at application start)
    setup_environment()
    
    # Safe imports with fallback
    coordinator = safe_import('coordination.genius_agent_coordinator', 'GeniusAgentCoordinator')
"""
import sys
import os
import logging
from pathlib import Path
from typing import Any, Optional, Union


def setup_environment(base_path: Optional[str] = None) -> bool:
    """
    Setup the AUJ Platform environment including paths and configuration.
    
    Args:
        base_path: Optional base path for the platform
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Determine platform root
        if base_path:
            platform_root = Path(base_path).resolve()
        else:
            current_file = Path(__file__).resolve()
            platform_root = current_file.parent.parent.parent.parent
        
        # Add critical paths to sys.path
        critical_paths = [
            str(platform_root),
            str(platform_root / "auj_platform"),
            str(platform_root / "auj_platform" / "src"),
            str(platform_root / "auj_platform" / "src" / "core"),
        ]
        
        for path in critical_paths:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
        
        # Set environment variables
        os.environ['AUJ_PLATFORM_ROOT'] = str(platform_root)
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to setup environment: {str(e)}")
        return False


def safe_import(module_path: str, class_name: Optional[str] = None, 
               fallback: Any = None) -> Any:
    """
    Safely import a module or class with fallback handling.
    
    Args:
        module_path: Module path (e.g., 'coordination.genius_agent_coordinator')
        class_name: Optional class name to import from module
        fallback: Fallback value if import fails
        
    Returns:
        Imported module/class or fallback value
    """
    try:
        module = __import__(module_path, fromlist=[class_name] if class_name else [])
        
        if class_name:
            return getattr(module, class_name, fallback)
        else:
            return module
            
    except ImportError as e:
        logging.warning(f"Import failed for {module_path}.{class_name or ''}: {str(e)}")
        return fallback
    except AttributeError as e:
        logging.warning(f"Attribute error for {module_path}.{class_name}: {str(e)}")
        return fallback


def get_platform_root() -> Path:
    """Get the platform root directory"""
    return Path(os.environ.get('AUJ_PLATFORM_ROOT', Path(__file__).parent.parent.parent.parent))


def get_src_path() -> Path:
    """Get the source code directory"""
    return get_platform_root() / "auj_platform" / "src"


def get_config_path() -> Path:
    """Get the configuration directory"""
    return get_platform_root() / "config"


def get_logs_path() -> Path:
    """Get the logs directory"""
    return get_platform_root() / "logs"


# Convenience imports for common components
def import_coordinator():
    """Import GeniusAgentCoordinator with fallback"""
    return safe_import('coordination.genius_agent_coordinator', 'GeniusAgentCoordinator')


def import_config_manager():
    """Import ConfigManager with fallback"""
    return safe_import('core.config_manager', 'ConfigManager')


def import_container():
    """Import DI container with fallback"""
    return safe_import('core.containers', 'ApplicationContainer')


def import_base_agent():
    """Import BaseAgent with fallback"""
    return safe_import('agents.base_agent', 'BaseAgent')
