"""
AUJ Platform Environment Setup Script
====================================
Comprehensive environment configuration and path management for the AUJ Platform.
This script ensures proper Python path setup, dependency management, and 
environment configuration without disrupting existing architecture.

Author: AUJ Platform Development Team  
Date: 2025-07-04
Version: 1.0.0
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class AUJEnvironmentSetup:
    """
    Comprehensive environment setup for the AUJ Platform.
    Handles path configuration, dependency management, and environment validation.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize environment setup with base platform path"""
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self.platform_root = self.base_path.resolve()
        self.src_path = self.platform_root / "auj_platform" / "src"
        self.config_path = self.platform_root / "config"
        self.logs_path = self.platform_root / "logs"
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Environment configuration
        self.env_config = {
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONUNBUFFERED': '1',
            'AUJ_PLATFORM_ROOT': str(self.platform_root),
            'AUJ_SRC_PATH': str(self.src_path),
            'AUJ_CONFIG_PATH': str(self.config_path),
            'AUJ_LOGS_PATH': str(self.logs_path)
        }
    
    def setup_logging(self):
        """Setup logging for environment setup process"""
        log_level = os.getenv('AUJ_LOG_LEVEL', 'INFO')
        
        # Ensure logs directory exists
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.logs_path / 'environment_setup.log', mode='a')
            ]
        )
    
    def setup_python_path(self) -> bool:
        """
        Setup Python path for proper module imports.
        Returns True if successful, False otherwise.
        """
        try:
            self.logger.info("Setting up Python path configuration...")
            
            # Define critical paths for the platform
            critical_paths = [
                str(self.platform_root),
                str(self.platform_root / "auj_platform"),
                str(self.src_path),
                str(self.src_path / "core"),
                str(self.src_path / "agents"),
                str(self.src_path / "coordination"),
                str(self.src_path / "trading_engine"),
                str(self.src_path / "broker_interfaces"),
                str(self.src_path / "data_providers"),
                str(self.src_path / "analytics"),
                str(self.src_path / "monitoring"),
                str(self.src_path / "api"),
            ]
            
            # Add paths to sys.path if not already present
            paths_added = []
            for path in critical_paths:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)
                    paths_added.append(path)
            
            if paths_added:
                self.logger.info(f"Added {len(paths_added)} paths to Python path")
                for path in paths_added:
                    self.logger.debug(f"  - {path}")
            else:
                self.logger.info("All critical paths already in Python path")
            
            # Set environment variables
            for key, value in self.env_config.items():
                os.environ[key] = value
                self.logger.debug(f"Set environment variable: {key} = {value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup Python path: {str(e)}")
            return False
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """
        Validate that all required dependencies are available.
        Returns validation results.
        """
        self.logger.info("Validating platform dependencies...")
        
        validation_results = {
            'required_packages': {},
            'optional_packages': {},
            'missing_required': [],
            'missing_optional': [],
            'status': 'unknown'
        }
        
        # Required packages for core functionality
        required_packages = [
            'numpy',
            'pandas',
            'asyncio',
            'typing',
            'logging',
            'pathlib',
            'dataclasses',
            'enum',
            'abc',
            'datetime',
            'json',
            'yaml',
            'sqlite3'
        ]
        
        # Optional packages for enhanced functionality
        optional_packages = [
            'MetaTrader5',
            'yfinance',
            'requests',
            'aiohttp',
            'fastapi',
            'uvicorn',
            'prometheus_client',
            'psutil',
            'dependency_injector'
        ]
        
        # Check required packages
        for package in required_packages:
            try:
                __import__(package)
                validation_results['required_packages'][package] = True
                self.logger.debug(f"✓ Required package available: {package}")
            except ImportError:
                validation_results['required_packages'][package] = False
                validation_results['missing_required'].append(package)
                self.logger.warning(f"✗ Required package missing: {package}")
        
        # Check optional packages
        for package in optional_packages:
            try:
                __import__(package)
                validation_results['optional_packages'][package] = True
                self.logger.debug(f"✓ Optional package available: {package}")
            except ImportError:
                validation_results['optional_packages'][package] = False
                validation_results['missing_optional'].append(package)
                self.logger.debug(f"✗ Optional package missing: {package}")
        
        # Determine overall status
        if not validation_results['missing_required']:
            if not validation_results['missing_optional']:
                validation_results['status'] = 'complete'
                self.logger.info("[OK] All dependencies available")
            else:
                validation_results['status'] = 'partial'
                self.logger.info(f"[OK] Core dependencies available, {len(validation_results['missing_optional'])} optional missing")
        else:
            validation_results['status'] = 'incomplete'
            self.logger.error(f"✗ Missing {len(validation_results['missing_required'])} required dependencies")
        
        return validation_results
    
    def validate_directory_structure(self) -> Dict[str, Any]:
        """
        Validate that all required directories exist.
        Creates missing directories if needed.
        """
        self.logger.info("Validating directory structure...")
        
        validation_results = {
            'directories': {},
            'created': [],
            'missing': [],
            'status': 'unknown'
        }
        
        # Required directories
        required_directories = [
            "auj_platform",
            "auj_platform/src",
            "auj_platform/src/core",
            "auj_platform/src/agents",
            "auj_platform/src/coordination",
            "auj_platform/src/trading_engine",
            "auj_platform/src/broker_interfaces",
            "auj_platform/src/data_providers",
            "auj_platform/src/analytics",
            "auj_platform/src/monitoring",
            "auj_platform/src/api",
            "auj_platform/src/learning",
            "auj_platform/src/validation",
            "auj_platform/src/forecasting",
            "auj_platform/src/optimization",
            "config",
            "logs",
            "data",
            "tests",
            "docs"
        ]
        
        # Check and create directories
        for directory in required_directories:
            dir_path = self.platform_root / directory
            if dir_path.exists():
                validation_results['directories'][directory] = True
                self.logger.debug(f"✓ Directory exists: {directory}")
            else:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    validation_results['directories'][directory] = True
                    validation_results['created'].append(directory)
                    self.logger.info(f"✓ Created directory: {directory}")
                except Exception as e:
                    validation_results['directories'][directory] = False
                    validation_results['missing'].append(directory)
                    self.logger.error(f"✗ Failed to create directory {directory}: {str(e)}")
        
        # Determine status
        if not validation_results['missing']:
            validation_results['status'] = 'complete'
            self.logger.info("✓ Directory structure validation complete")
        else:
            validation_results['status'] = 'incomplete'
            self.logger.error(f"✗ Missing {len(validation_results['missing'])} directories")
        
        return validation_results
    
    def create_environment_config(self) -> bool:
        """
        Create environment configuration file with all settings.
        Returns True if successful, False otherwise.
        """
        try:
            self.logger.info("Creating environment configuration...")
            
            config_data = {
                'platform': {
                    'name': 'AUJ Platform',
                    'version': '1.0.0',
                    'root_path': str(self.platform_root),
                    'src_path': str(self.src_path),
                    'config_path': str(self.config_path),
                    'logs_path': str(self.logs_path)
                },
                'paths': {
                    'python_path': [
                        str(self.platform_root),
                        str(self.platform_root / "auj_platform"),
                        str(self.src_path),
                        str(self.src_path / "core"),
                        str(self.src_path / "agents"),
                        str(self.src_path / "coordination"),
                        str(self.src_path / "trading_engine"),
                        str(self.src_path / "broker_interfaces"),
                        str(self.src_path / "data_providers"),
                        str(self.src_path / "analytics"),
                        str(self.src_path / "monitoring"),
                        str(self.src_path / "api")
                    ]
                },
                'environment_variables': self.env_config,
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file': str(self.logs_path / 'auj_platform.log')
                }
            }
            
            # Write configuration file
            config_file = self.config_path / 'environment_config.json'
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"✓ Environment configuration saved to: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create environment configuration: {str(e)}")
            return False
    
    def setup_import_helpers(self) -> bool:
        """
        Create import helper utilities for easier module imports.
        Returns True if successful, False otherwise.
        """
        try:
            self.logger.info("Creating import helper utilities...")
            
            # Import helper content
            import_helper_content = '''"""
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
'''
            
            # Write import helper
            import_helper_file = self.src_path / 'core' / 'import_helper.py'
            with open(import_helper_file, 'w') as f:
                f.write(import_helper_content)
            
            self.logger.info(f"✓ Import helper created: {import_helper_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create import helpers: {str(e)}")
            return False
    
    def run_full_setup(self) -> Dict[str, Any]:
        """
        Run complete environment setup process.
        Returns comprehensive setup results.
        """
        self.logger.info("=== Starting AUJ Platform Environment Setup ===")
        
        setup_results = {
            'python_path': False,
            'dependencies': {},
            'directories': {},
            'config_file': False,
            'import_helpers': False,
            'overall_status': 'failed',
            'summary': {}
        }
        
        try:
            # 1. Setup Python path
            setup_results['python_path'] = self.setup_python_path()
            
            # 2. Validate dependencies
            setup_results['dependencies'] = self.validate_dependencies()
            
            # 3. Validate/create directory structure
            setup_results['directories'] = self.validate_directory_structure()
            
            # 4. Create environment configuration
            setup_results['config_file'] = self.create_environment_config()
            
            # 5. Setup import helpers
            setup_results['import_helpers'] = self.setup_import_helpers()
            
            # Calculate overall status
            core_success = (
                setup_results['python_path'] and
                setup_results['dependencies']['status'] in ['complete', 'partial'] and
                setup_results['directories']['status'] == 'complete'
            )
            
            if core_success and setup_results['config_file'] and setup_results['import_helpers']:
                setup_results['overall_status'] = 'complete'
            elif core_success:
                setup_results['overall_status'] = 'partial'
            else:
                setup_results['overall_status'] = 'failed'
            
            # Generate summary
            setup_results['summary'] = {
                'status': setup_results['overall_status'],
                'python_path_configured': setup_results['python_path'],
                'required_dependencies': len([k for k, v in setup_results['dependencies']['required_packages'].items() if v]),
                'missing_required': len(setup_results['dependencies']['missing_required']),
                'directories_created': len(setup_results['directories']['created']),
                'config_file_created': setup_results['config_file'],
                'import_helpers_created': setup_results['import_helpers']
            }
            
            # Log final status
            if setup_results['overall_status'] == 'complete':
                self.logger.info("=== ✓ Environment setup completed successfully ===")
            elif setup_results['overall_status'] == 'partial':
                self.logger.warning("=== ⚠ Environment setup completed with warnings ===")
            else:
                self.logger.error("=== ✗ Environment setup failed ===")
            
        except Exception as e:
            self.logger.error(f"Environment setup failed with exception: {str(e)}")
            setup_results['overall_status'] = 'error'
            setup_results['error'] = str(e)
        
        return setup_results


def setup_auj_environment(base_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to setup AUJ Platform environment.
    
    Args:
        base_path: Optional base path for the platform
        
    Returns:
        Setup results dictionary
    """
    setup = AUJEnvironmentSetup(base_path)
    return setup.run_full_setup()


def quick_setup() -> bool:
    """
    Quick environment setup that returns success/failure status.
    
    Returns:
        True if setup successful, False otherwise
    """
    try:
        results = setup_auj_environment()
        return results['overall_status'] in ['complete', 'partial']
    except Exception:
        return False


if __name__ == "__main__":
    # Run environment setup when script is executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description='AUJ Platform Environment Setup')
    parser.add_argument('--base-path', help='Base path for the platform')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run setup
    results = setup_auj_environment(args.base_path)
    
    # Print results
    print(f"\\nSetup Status: {results['overall_status'].upper()}")
    print(f"Summary: {results['summary']}")
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_status'] in ['complete', 'partial'] else 1
    sys.exit(exit_code)