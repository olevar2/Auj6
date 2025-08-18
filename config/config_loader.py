"""
Configuration Loader for AUJ Platform
Cross-platform configuration management with MetaApi support
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationPaths:
    """Configuration file paths for different components"""
    main_config: str = "config/main_config.yaml"
    metaapi_config: str = "config/metaapi_config.yaml"
    mt5_config: str = "config/mt5_config.yaml"
    monitoring_config: str = "config/monitoring_config.yaml"
    production_conservative: str = "config/production_conservative.yaml"
    linux_deployment: str = "config/linux_deployment.yaml"
    env_template: str = "config/.env.template"

class ConfigurationLoader:
    """
    Enhanced configuration loader with cross-platform support
    Handles MetaApi and legacy MT5 configurations
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize configuration loader
        
        Args:
            base_path: Base path for configuration files
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.paths = ConfigurationPaths()
        self.configs: Dict[str, Any] = {}
        self.platform = self._detect_platform()
        
        logger.info(f"Initialized ConfigurationLoader for platform: {self.platform}")
    
    def _detect_platform(self) -> str:
        """Detect current platform"""
        import platform
        system = platform.system().lower()
        
        # Check for container environment
        if os.path.exists('/.dockerenv') or os.environ.get('CONTAINER') == 'true':
            return 'container'
        
        # Map platform names
        platform_map = {
            'linux': 'linux',
            'darwin': 'macos',
            'windows': 'windows'
        }
        
        return platform_map.get(system, 'unknown')
    
    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load all configuration files
        
        Returns:
            Dictionary containing all configurations
        """
        config_files = {
            'main': self.paths.main_config,
            'metaapi': self.paths.metaapi_config,
            'mt5': self.paths.mt5_config,
            'monitoring': self.paths.monitoring_config,
            'production_conservative': self.paths.production_conservative,
            'linux_deployment': self.paths.linux_deployment
        }
        
        for config_name, config_path in config_files.items():
            try:
                self.configs[config_name] = self._load_yaml_file(config_path)
                logger.info(f"Loaded {config_name} configuration")
            except Exception as e:
                logger.warning(f"Failed to load {config_name} config: {e}")
                self.configs[config_name] = {}
        
        # Apply platform-specific overrides
        self._apply_platform_overrides()
        
        return self.configs
    
    def _load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Configuration dictionary
        """
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            logger.warning(f"Configuration file not found: {full_path}")
            return {}
        
        with open(full_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file) or {}
    
    def _apply_platform_overrides(self) -> None:
        """Apply platform-specific configuration overrides"""
        
        if self.platform == 'linux' or self.platform == 'container':
            self._apply_linux_overrides()
        elif self.platform == 'windows':
            self._apply_windows_overrides()
        elif self.platform == 'macos':
            self._apply_macos_overrides()
    
    def _apply_linux_overrides(self) -> None:
        """Apply Linux-specific configuration overrides"""
        logger.info("Applying Linux-specific configuration overrides")
        
        # Main config overrides
        if 'main' in self.configs:
            main_config = self.configs['main']
            
            # Prioritize MetaApi provider
            if 'data_providers' in main_config:
                providers = main_config['data_providers']
                
                # Enable MetaApi, disable MT5
                if 'metaapi' in providers:
                    providers['metaapi']['enabled'] = True
                    providers['metaapi']['priority'] = 1
                
                if 'mt5' in providers:
                    providers['mt5']['enabled'] = False
                    providers['mt5']['priority'] = 999
            
            # Linux platform settings
            if 'platform_settings' in main_config:
                platform_settings = main_config['platform_settings']
                if 'linux_deployment' in platform_settings:
                    platform_settings['linux_deployment']['enabled'] = True
                    platform_settings['linux_deployment']['preferred_provider'] = 'metaapi'
                    platform_settings['linux_deployment']['disable_mt5_direct'] = True
            
            # Feature flags for Linux
            if 'feature_flags' in main_config:
                flags = main_config['feature_flags']
                flags['enable_metaapi_provider'] = True
                flags['enable_linux_optimization'] = True
                flags['enable_mt5_direct_fallback'] = False
                flags['enable_cross_platform_deployment'] = True
    
    def _apply_windows_overrides(self) -> None:
        """Apply Windows-specific configuration overrides"""
        logger.info("Applying Windows-specific configuration overrides")
        
        # Main config overrides
        if 'main' in self.configs:
            main_config = self.configs['main']
            
            # Enable both MetaApi and MT5
            if 'data_providers' in main_config:
                providers = main_config['data_providers']
                
                # Keep MT5 as primary on Windows if available
                if 'mt5' in providers:
                    providers['mt5']['enabled'] = True
                    providers['mt5']['priority'] = 1
                
                if 'metaapi' in providers:
                    providers['metaapi']['enabled'] = True
                    providers['metaapi']['priority'] = 2
            
            # Feature flags for Windows
            if 'feature_flags' in main_config:
                flags = main_config['feature_flags']
                flags['enable_metaapi_provider'] = True
                flags['enable_mt5_direct_fallback'] = True
                flags['enable_cross_platform_deployment'] = True
    
    def _apply_macos_overrides(self) -> None:
        """Apply macOS-specific configuration overrides"""
        logger.info("Applying macOS-specific configuration overrides")
        
        # Similar to Linux - prioritize MetaApi
        self._apply_linux_overrides()
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific provider
        
        Args:
            provider_name: Name of the provider (metaapi, mt5, yahoo, etc.)
            
        Returns:
            Provider configuration
        """
        if provider_name == 'metaapi':
            return self.configs.get('metaapi', {})
        elif provider_name == 'mt5':
            return self.configs.get('mt5', {})
        else:
            # Look in main config data_providers section
            main_config = self.configs.get('main', {})
            providers = main_config.get('data_providers', {})
            return providers.get(provider_name, {})
    
    def get_platform_specific_config(self) -> Dict[str, Any]:
        """
        Get platform-specific configuration
        
        Returns:
            Platform-specific configuration
        """
        if self.platform == 'linux' or self.platform == 'container':
            return self.configs.get('linux_deployment', {})
        else:
            return {}
    
    def get_enabled_providers(self) -> List[str]:
        """
        Get list of enabled providers in priority order
        
        Returns:
            List of enabled provider names
        """
        main_config = self.configs.get('main', {})
        providers = main_config.get('data_providers', {})
        
        enabled_providers = []
        provider_priorities = []
        
        for provider_name, provider_config in providers.items():
            if provider_config.get('enabled', False):
                priority = provider_config.get('priority', 999)
                provider_priorities.append((priority, provider_name))
        
        # Sort by priority (lower number = higher priority)
        provider_priorities.sort(key=lambda x: x[0])
        enabled_providers = [provider[1] for provider in provider_priorities]
        
        logger.info(f"Enabled providers (priority order): {enabled_providers}")
        return enabled_providers
    
    def is_provider_supported(self, provider_name: str) -> bool:
        """
        Check if provider is supported on current platform
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            True if provider is supported
        """
        if provider_name == 'metaapi':
            return True  # MetaApi supports all platforms
        elif provider_name == 'mt5':
            # MT5 direct only supported on Windows
            return self.platform == 'windows'
        elif provider_name in ['yahoo', 'binance']:
            return True  # These work on all platforms
        else:
            return False
    
    def load_environment_variables(self) -> Dict[str, str]:
        """
        Load environment variables from .env file if it exists
        
        Returns:
            Dictionary of environment variables
        """
        env_file = self.base_path / '.env'
        env_vars = {}
        
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                logger.info("Loaded environment variables from .env file")
            except ImportError:
                logger.warning("python-dotenv not installed, cannot load .env file")
            except Exception as e:
                logger.error(f"Failed to load .env file: {e}")
        
        # Get relevant environment variables
        env_prefixes = ['AUJ_', 'METAAPI_', 'MT5_']
        for key, value in os.environ.items():
            if any(key.startswith(prefix) for prefix in env_prefixes):
                env_vars[key] = value
        
        return env_vars
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration for current platform
        
        Returns:
            List of validation errors/warnings
        """
        issues = []
        
        # Check if any providers are enabled
        enabled_providers = self.get_enabled_providers()
        if not enabled_providers:
            issues.append("No data providers are enabled")
        
        # Platform-specific validation
        if self.platform in ['linux', 'container']:
            # Check MetaApi configuration
            if 'metaapi' not in enabled_providers:
                issues.append("MetaApi provider should be enabled for Linux deployment")
            
            # Check if MT5 is accidentally enabled
            if 'mt5' in enabled_providers:
                issues.append("MT5 direct provider is not supported on Linux")
        
        # Check environment variables
        env_vars = self.load_environment_variables()
        
        if 'metaapi' in enabled_providers:
            if not env_vars.get('AUJ_METAAPI_TOKEN'):
                issues.append("MetaApi token not found in environment variables")
            if not env_vars.get('AUJ_METAAPI_ACCOUNT_ID'):
                issues.append("MetaApi account ID not found in environment variables")
        
        return issues
    
    def get_merged_config(self) -> Dict[str, Any]:
        """
        Get merged configuration from all sources
        
        Returns:
            Merged configuration dictionary
        """
        # Start with main config
        merged_config = self.configs.get('main', {}).copy()
        
        # Add platform-specific overrides
        platform_config = self.get_platform_specific_config()
        if platform_config:
            self._deep_merge(merged_config, platform_config)
        
        # Add provider-specific configurations
        merged_config['providers'] = {}
        for provider in self.get_enabled_providers():
            merged_config['providers'][provider] = self.get_provider_config(provider)
        
        # Add environment variables
        merged_config['environment'] = self.load_environment_variables()
        
        return merged_config
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
        """
        Deep merge two dictionaries
        
        Args:
            dict1: Target dictionary (modified in place)
            dict2: Source dictionary
        """
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                self._deep_merge(dict1[key], value)
            else:
                dict1[key] = value

# Global configuration loader instance
config_loader = ConfigurationLoader()

def get_config() -> Dict[str, Any]:
    """
    Get the global configuration
    
    Returns:
        Global configuration dictionary
    """
    if not config_loader.configs:
        config_loader.load_all_configs()
    
    return config_loader.get_merged_config()

def get_provider_config(provider_name: str) -> Dict[str, Any]:
    """
    Get configuration for specific provider
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Provider configuration
    """
    return config_loader.get_provider_config(provider_name)

def get_enabled_providers() -> List[str]:
    """
    Get list of enabled providers
    
    Returns:
        List of enabled provider names
    """
    return config_loader.get_enabled_providers()

def is_linux_deployment() -> bool:
    """
    Check if running on Linux deployment
    
    Returns:
        True if Linux deployment
    """
    return config_loader.platform in ['linux', 'container']

def validate_config() -> List[str]:
    """
    Validate current configuration
    
    Returns:
        List of validation issues
    """
    return config_loader.validate_configuration()