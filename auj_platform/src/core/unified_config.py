"""
Unified Configuration Management System for AUJ Platform
======================================================

This module provides a comprehensive unified configuration management system that
consolidates all configuration patterns, implements caching, environment overrides,
validation, hot reloading, secure secrets handling, and configuration audit trails.

Key Features:
- Unified configuration interface replacing all config.get() and config_loader.get() calls
- Environment-specific configuration management
- Configuration validation and schema enforcement
- Hot reloading with change detection
- Secure secrets handling with encryption
- Configuration audit trail and access logging
- Caching with intelligent invalidation
- Configuration versioning and rollback
- Configuration migration and compatibility

Author: AUJ Platform Development Team
Date: 2025-07-04
Version: 2.0.0 - Unified Configuration Management
"""

import os
import yaml
import json
import asyncio
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from contextlib import contextmanager
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    # Define dummy classes/functions if needed or handle in logic
    Fernet = None
    
import base64
import logging
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = object
    FileSystemEventHandler = object
    
import tempfile
import shutil

# Use basic logging to avoid circular import with logging_setup
import logging

# Define minimal exceptions to avoid circular imports
class ConfigurationError(Exception):
    """Configuration related error."""
    pass

class ValidationError(Exception):
    """Validation related error."""
    pass

class SecurityError(Exception):
    """Security related error."""
    pass
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ConfigurationAccess:
    """Configuration access record for audit trail."""
    timestamp: datetime
    key: str
    value_hash: str
    accessor: str
    environment: str
    source_file: Optional[str] = None
    validation_status: str = "SUCCESS"
    error_message: Optional[str] = None


@dataclass
class ConfigurationChange:
    """Configuration change record for versioning."""
    timestamp: datetime
    key: str
    old_value_hash: Optional[str]
    new_value_hash: str
    changed_by: str
    change_reason: str
    environment: str
    rollback_data: Optional[Dict[str, Any]] = None


@dataclass
class ConfigurationValidation:
    """Configuration validation rules."""
    required: bool = False
    data_type: Optional[type] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    description: Optional[str] = None


class ConfigurationSchema:
    """Configuration schema definition and validation."""

    def __init__(self):
        self.schema: Dict[str, ConfigurationValidation] = {}
        self._load_default_schema()

    def _load_default_schema(self):
        """Load default configuration schema for AUJ Platform."""
        # Trading Configuration
        self.add_validation("trading.risk_percentage", ConfigurationValidation(
            required=True, data_type=float, min_value=0.1, max_value=10.0,
            description="Risk percentage per trade (0.1-10.0%)"
        ))

        self.add_validation("trading.max_positions", ConfigurationValidation(
            required=True, data_type=int, min_value=1, max_value=20,
            description="Maximum number of open positions"
        ))

        self.add_validation("trading.confidence_threshold", ConfigurationValidation(
            required=True, data_type=float, min_value=0.1, max_value=1.0,
            description="Minimum confidence threshold for trades"
        ))

        # Database Configuration
        self.add_validation("database.url", ConfigurationValidation(
            required=True, data_type=str,
            description="Database connection URL"
        ))

        self.add_validation("database.pool_size", ConfigurationValidation(
            required=False, data_type=int, min_value=5, max_value=100,
            description="Database connection pool size"
        ))

        # MT5 Configuration
        self.add_validation("mt5.server", ConfigurationValidation(
            required=True, data_type=str,
            description="MT5 server name"
        ))

        self.add_validation("mt5.login", ConfigurationValidation(
            required=True, data_type=int, min_value=1,
            description="MT5 login number"
        ))

        self.add_validation("mt5.timeout", ConfigurationValidation(
            required=False, data_type=int, min_value=1000, max_value=60000,
            description="MT5 operation timeout in milliseconds"
        ))

        # Monitoring Configuration
        self.add_validation("monitoring.enabled", ConfigurationValidation(
            required=False, data_type=bool,
            description="Enable system monitoring"
        ))

        self.add_validation("monitoring.check_interval", ConfigurationValidation(
            required=False, data_type=int, min_value=10, max_value=3600,
            description="Monitoring check interval in seconds"
        ))

        # Logging Configuration
        self.add_validation("logging.level", ConfigurationValidation(
            required=False, data_type=str,
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            description="Logging level"
        ))

        # Environment Configuration
        self.add_validation("environment", ConfigurationValidation(
            required=True, data_type=str,
            allowed_values=["development", "testing", "staging", "production"],
            description="Environment type"
        ))

    def add_validation(self, key: str, validation: ConfigurationValidation):
        """Add validation rule for configuration key."""
        self.schema[key] = validation

    def validate(self, key: str, value: Any) -> tuple[bool, Optional[str]]:
        """Validate configuration value against schema."""
        if key not in self.schema:
            return True, None  # No validation rule = valid

        validation = self.schema[key]

        # Check if required
        if validation.required and value is None:
            return False, f"Configuration key '{key}' is required"

        if value is None:
            return True, None  # Optional and None is valid

        # Check data type
        if validation.data_type and not isinstance(value, validation.data_type):
            return False, f"Configuration key '{key}' must be of type {validation.data_type.__name__}"

        # Check numeric ranges
        if validation.min_value is not None and value < validation.min_value:
            return False, f"Configuration key '{key}' must be >= {validation.min_value}"

        if validation.max_value is not None and value > validation.max_value:
            return False, f"Configuration key '{key}' must be <= {validation.max_value}"

        # Check allowed values
        if validation.allowed_values and value not in validation.allowed_values:
            return False, f"Configuration key '{key}' must be one of {validation.allowed_values}"

        # Check pattern (for strings)
        if validation.pattern and isinstance(value, str):
            import re
            if not re.match(validation.pattern, value):
                return False, f"Configuration key '{key}' does not match required pattern"

        # Custom validation
        if validation.custom_validator:
            try:
                if not validation.custom_validator(value):
                    return False, f"Configuration key '{key}' failed custom validation"
            except Exception as e:
                return False, f"Configuration key '{key}' validation error: {str(e)}"

        return True, None


class ConfigurationEncryption:
    """Secure configuration encryption and decryption."""

    def __init__(self, master_key: Optional[str] = None):
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography module not available. Encryption disabled.")
            self.master_key = None
            self.cipher_suite = None
            return

        self.master_key = master_key or self._get_or_create_master_key()
        self.cipher_suite = self._create_cipher_suite()

    def _get_or_create_master_key(self) -> str:
        """Get or create master encryption key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return ""
            
        key_file = Path.home() / ".auj_platform" / "master.key"

        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read().decode()
        else:
            # Create new master key
            key_file.parent.mkdir(parents=True, exist_ok=True)
            master_key = Fernet.generate_key().decode()

            with open(key_file, 'w') as f:
                f.write(master_key)

            # Set restrictive permissions
            try:
                key_file.chmod(0o600)
            except Exception:
                pass # Windows might not support this

            return master_key

    def _create_cipher_suite(self) -> Any:
        """Create cipher suite from master key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            return None
        return Fernet(self.master_key.encode())

    def encrypt(self, value: str) -> str:
        """Encrypt configuration value."""
        if not self.cipher_suite:
            logger.warning("Encryption disabled, returning raw value")
            return str(value)

        if not isinstance(value, str):
            value = str(value)

        encrypted = self.cipher_suite.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_value: str) -> str:
        """Decrypt configuration value."""
        if not self.cipher_suite:
            logger.warning("Encryption disabled, returning raw value")
            return encrypted_value

        try:
            encrypted_data = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self.cipher_suite.decrypt(encrypted_data)
            return decrypted.decode()
        except Exception as e:
            raise SecurityError(f"Failed to decrypt configuration value: {str(e)}")

    def is_encrypted(self, value: str) -> bool:
        """Check if value is encrypted."""
        try:
            # Encrypted values have specific format
            base64.urlsafe_b64decode(value.encode())
            return True
        except:
            return False


class ConfigurationWatcher(FileSystemEventHandler):
    """File system watcher for configuration hot reloading."""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.last_reload = {}
        self.reload_delay = 1.0  # 1 second delay to avoid multiple reloads

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if it's a configuration file
        if file_path.suffix in ['.yaml', '.yml', '.json', '.conf']:
            now = datetime.utcnow()
            last_reload = self.last_reload.get(str(file_path), datetime.min)

            # Avoid rapid reloads
            if (now - last_reload).total_seconds() > self.reload_delay:
                self.last_reload[str(file_path)] = now

                try:
                    self.config_manager._reload_file(file_path)
                    logger.info(f"Configuration file reloaded: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to reload configuration file {file_path}: {str(e)}")


class UnifiedConfigManager:
    """
    Unified Configuration Manager for AUJ Platform.

    This class provides a comprehensive configuration management system that:
    - Consolidates all configuration patterns
    - Implements environment-specific configurations
    - Provides configuration validation and schema enforcement
    - Supports hot reloading with change detection
    - Handles secure secrets with encryption
    - Maintains configuration audit trail
    - Implements caching with intelligent invalidation
    - Supports configuration versioning and rollback
    """

    def __init__(self,
                 config_paths: Optional[List[Union[str, Path]]] = None,
                 environment: Optional[str] = None,
                 enable_encryption: bool = True,
                 enable_hot_reload: bool = True,
                 enable_audit: bool = True):
        """
        Initialize Unified Configuration Manager.

        Args:
            config_paths: List of configuration file paths
            environment: Environment name (development, testing, staging, production)
            enable_encryption: Enable configuration encryption for secrets
            enable_hot_reload: Enable hot reloading of configuration files
            enable_audit: Enable configuration access audit logging
        """
        self.environment = environment or os.getenv("AUJ_ENVIRONMENT", "development")
        self.enable_encryption = enable_encryption
        self.enable_hot_reload = enable_hot_reload
        self.enable_audit = enable_audit

        # Configuration storage
        self.config_data: Dict[str, Any] = {}
        self.file_timestamps: Dict[str, datetime] = {}
        self.config_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=5)

        # Configuration paths
        self.config_paths = self._resolve_config_paths(config_paths)

        # Schema and validation
        self.schema = ConfigurationSchema()

        # Encryption
        self.encryption = ConfigurationEncryption() if enable_encryption else None

        # Audit trail
        self.access_log: List[ConfigurationAccess] = []
        self.change_log: List[ConfigurationChange] = []

        # Hot reloading
        self.watcher = None
        self.observer = None

        # Thread safety
        self._lock = threading.RLock()

        # Load initial configuration
        self._load_all_configurations()

        # Start file watcher if enabled
        if self.enable_hot_reload:
            self._start_file_watcher()

        logger.info(f"UnifiedConfigManager initialized for environment: {self.environment}")

    def _resolve_config_paths(self, config_paths: Optional[List[Union[str, Path]]]) -> List[Path]:
        """Resolve configuration file paths."""
        if config_paths:
            return [Path(p) for p in config_paths]

        # Default configuration paths
        base_path = Path(__file__).parent.parent.parent
        paths = [
            base_path / "config" / "main_config.yaml",
            base_path / "config" / "mt5_config.yaml",
            base_path / "auj_platform" / "config" / "main_config.yaml",
            base_path / "auj_platform" / "config" / "mt5_config.yaml",
        ]

        # Environment-specific configurations
        env_paths = [
            base_path / "config" / f"{self.environment}_config.yaml",
            base_path / "auj_platform" / "config" / f"{self.environment}_config.yaml",
        ]

        # Return existing paths
        existing_paths = []
        for path in paths + env_paths:
            if path.exists():
                existing_paths.append(path)

        return existing_paths

    def _load_all_configurations(self):
        """Load all configuration files."""
        with self._lock:
            for config_path in self.config_paths:
                try:
                    self._load_configuration_file(config_path)
                except Exception as e:
                    logger.error(f"Failed to load configuration from {config_path}: {str(e)}")

            # Load environment variables
            self._load_environment_variables()

            # Validate configuration
            self._validate_all_configurations()

    def _load_configuration_file(self, config_path: Path):
        """Load configuration from a file."""
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return

        # Check if file has been modified
        file_stat = config_path.stat()
        file_time = datetime.fromtimestamp(file_stat.st_mtime)

        if config_path in self.file_timestamps:
            if file_time <= self.file_timestamps[config_path]:
                return  # File hasn't changed

        self.file_timestamps[config_path] = file_time

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    logger.warning(f"Unsupported configuration file format: {config_path}")
                    return

            if data:
                # Merge configuration data
                self._merge_configuration_data(data, str(config_path))

                logger.info(f"Loaded configuration from: {config_path}")

        except Exception as e:
            logger.error(f"Error loading configuration file {config_path}: {str(e)}")
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {str(e)}")

    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        # Load AUJ_* environment variables
        env_config = {}

        for key, value in os.environ.items():
            if key.startswith('AUJ_'):
                # Convert AUJ_TRADING_RISK_PERCENTAGE to trading.risk_percentage
                config_key = key[4:].lower().replace('_', '.')

                # Try to parse value
                parsed_value = self._parse_env_value(value)
                env_config[config_key] = parsed_value

        if env_config:
            self._merge_configuration_data(env_config, "environment_variables")
            logger.info(f"Loaded {len(env_config)} configuration values from environment variables")

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _merge_configuration_data(self, new_data: Dict[str, Any], source: str):
        """Merge new configuration data into existing configuration."""
        def merge_dict(base_dict: Dict[str, Any], new_dict: Dict[str, Any], prefix: str = ""):
            for key, value in new_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    if key not in base_dict:
                        base_dict[key] = {}
                    elif not isinstance(base_dict[key], dict):
                        base_dict[key] = {}

                    merge_dict(base_dict[key], value, full_key)
                else:
                    # Record change if value is different
                    old_value = base_dict.get(key)
                    if old_value != value:
                        self._record_configuration_change(full_key, old_value, value, source)

                    base_dict[key] = value

        merge_dict(self.config_data, new_data)

        # Clear cache after configuration change
        self.config_cache.clear()
        self.cache_timestamps.clear()

    def _validate_all_configurations(self):
        """Validate all loaded configurations against schema."""
        def validate_nested(data: Dict[str, Any], prefix: str = ""):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    validate_nested(value, full_key)
                else:
                    is_valid, error_message = self.schema.validate(full_key, value)
                    if not is_valid:
                        logger.error(f"Configuration validation error: {error_message}")
                        if full_key in [k for k, v in self.schema.schema.items() if v.required]:
                            raise ConfigurationError(error_message)

        validate_nested(self.config_data)

    def _start_file_watcher(self):
        """Start file system watcher for hot reloading."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("Watchdog module not available. Hot reloading disabled.")
            return

        try:
            self.watcher = ConfigurationWatcher(self)
            self.observer = Observer()

            # Watch all configuration directories
            watched_dirs = set()
            for config_path in self.config_paths:
                config_dir = config_path.parent
                if config_dir not in watched_dirs:
                    self.observer.schedule(self.watcher, str(config_dir), recursive=False)
                    watched_dirs.add(config_dir)

            self.observer.start()
            logger.info("Configuration file watcher started")

        except Exception as e:
            logger.warning(f"Failed to start configuration file watcher: {str(e)}")
            self.watcher = None
            self.observer = None

    def _reload_file(self, file_path: Path):
        """Reload specific configuration file."""
        with self._lock:
            if file_path in [p for p in self.config_paths]:
                self._load_configuration_file(file_path)
                self._validate_all_configurations()

    def _record_configuration_access(self, key: str, value: Any, accessor: str = "unknown"):
        """Record configuration access for audit trail."""
        if not self.enable_audit:
            return

        access_record = ConfigurationAccess(
            timestamp=datetime.utcnow(),
            key=key,
            value_hash=hashlib.sha256(str(value).encode()).hexdigest()[:16],
            accessor=accessor,
            environment=self.environment
        )

        self.access_log.append(access_record)

        # Keep only last 10000 access records
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-10000:]

    def _record_configuration_change(self, key: str, old_value: Any, new_value: Any, changed_by: str):
        """Record configuration change for versioning."""
        change_record = ConfigurationChange(
            timestamp=datetime.utcnow(),
            key=key,
            old_value_hash=hashlib.sha256(str(old_value).encode()).hexdigest()[:16] if old_value is not None else None,
            new_value_hash=hashlib.sha256(str(new_value).encode()).hexdigest()[:16],
            changed_by=changed_by,
            change_reason="configuration_reload",
            environment=self.environment,
            rollback_data={"old_value": old_value}
        )

        self.change_log.append(change_record)

        # Keep only last 1000 change records
        if len(self.change_log) > 1000:
            self.change_log = self.change_log[-1000:]

    def get(self, key: str, default: Any = None, decrypt: bool = False) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            decrypt: Whether to decrypt the value if encrypted

        Returns:
            Configuration value
        """
        # Check cache first
        cache_key = f"{key}:{decrypt}"
        if cache_key in self.config_cache:
            cache_time = self.cache_timestamps.get(cache_key, datetime.min)
            if datetime.utcnow() - cache_time < self.cache_duration:
                value = self.config_cache[cache_key]
                self._record_configuration_access(key, value, "cache")
                return value

        with self._lock:
            # Navigate through nested dictionary
            keys = key.split('.')
            current = self.config_data

            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    self._record_configuration_access(key, default, "default")
                    return default

            value = current

            # Decrypt if requested and encryption is enabled
            if decrypt and self.encryption and isinstance(value, str):
                if self.encryption.is_encrypted(value):
                    try:
                        value = self.encryption.decrypt(value)
                    except SecurityError as e:
                        logger.error(f"Failed to decrypt configuration value for key '{key}': {str(e)}")
                        raise

            # Cache the result
            self.config_cache[cache_key] = value
            self.cache_timestamps[cache_key] = datetime.utcnow()

            self._record_configuration_access(key, value)

            return value

    def set(self, key: str, value: Any, encrypt: bool = False, persist: bool = False):
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Configuration value
            encrypt: Whether to encrypt the value
            persist: Whether to persist the change to file
        """
        with self._lock:
            # Validate the new value
            is_valid, error_message = self.schema.validate(key, value)
            if not is_valid:
                raise ValidationError(error_message)

            # Encrypt if requested
            if encrypt and self.encryption:
                if not isinstance(value, str):
                    value = str(value)
                value = self.encryption.encrypt(value)

            # Navigate and set value
            keys = key.split('.')
            current = self.config_data

            # Create nested structure if needed
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                elif not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]

            # Record change
            old_value = current.get(keys[-1])
            self._record_configuration_change(key, old_value, value, "manual_set")

            # Set the value
            current[keys[-1]] = value

            # Clear cache
            self.config_cache.clear()
            self.cache_timestamps.clear()

            # Persist if requested
            if persist:
                self._persist_configuration()

            logger.info(f"Configuration value set: {key}")

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key, default)

        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        elif isinstance(value, (int, float)):
            return bool(value)
        else:
            return bool(value)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        value = self.get(key, default)

        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Configuration value '{key}' cannot be converted to int, using default: {default}")
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        value = self.get(key, default)

        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Configuration value '{key}' cannot be converted to float, using default: {default}")
            return default

    def get_list(self, key: str, default: List[Any] = None) -> List[Any]:
        """Get list configuration value."""
        if default is None:
            default = []

        value = self.get(key, default)

        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            # Try to parse as comma-separated values
            return [item.strip() for item in value.split(',') if item.strip()]
        else:
            return [value] if value is not None else default

    def get_dict(self, key: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get dictionary configuration value."""
        if default is None:
            default = {}

        value = self.get(key, default)

        if isinstance(value, dict):
            return value
        else:
            logger.warning(f"Configuration value '{key}' is not a dictionary, using default")
            return default

    def get_secret(self, key: str, default: str = None) -> str:
        """Get encrypted secret configuration value."""
        return self.get(key, default, decrypt=True)

    def has(self, key: str) -> bool:
        """Check if configuration key exists."""
        return self.get(key, sentinel := object()) is not sentinel

    def delete(self, key: str):
        """Delete configuration key."""
        with self._lock:
            keys = key.split('.')
            current = self.config_data

            # Navigate to parent
            for k in keys[:-1]:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return  # Key doesn't exist

            # Delete the key
            if isinstance(current, dict) and keys[-1] in current:
                old_value = current[keys[-1]]
                del current[keys[-1]]

                # Record change
                self._record_configuration_change(key, old_value, None, "manual_delete")

                # Clear cache
                self.config_cache.clear()
                self.cache_timestamps.clear()

                logger.info(f"Configuration key deleted: {key}")

    def get_all(self, prefix: str = "") -> Dict[str, Any]:
        """Get all configuration values with optional prefix filter."""
        if not prefix:
            return self.config_data.copy()

        # Get values with prefix
        result = {}
        prefix_parts = prefix.split('.')
        current = self.config_data

        # Navigate to prefix
        for part in prefix_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return {}

        return current if isinstance(current, dict) else {}

    def reload(self):
        """Manually reload all configuration files."""
        logger.info("Manually reloading configuration files")
        self._load_all_configurations()

    def get_audit_log(self, key_filter: Optional[str] = None,
                     limit: int = 100) -> List[ConfigurationAccess]:
        """Get configuration access audit log."""
        if not self.enable_audit:
            return []

        filtered_log = self.access_log

        if key_filter:
            filtered_log = [
                record for record in filtered_log
                if key_filter in record.key
            ]

        return filtered_log[-limit:] if limit else filtered_log

    def get_change_log(self, key_filter: Optional[str] = None,
                      limit: int = 100) -> List[ConfigurationChange]:
        """Get configuration change log."""
        filtered_log = self.change_log

        if key_filter:
            filtered_log = [
                record for record in filtered_log
                if key_filter in record.key
            ]

        return filtered_log[-limit:] if limit else filtered_log

    def export_configuration(self, file_path: Optional[Union[str, Path]] = None,
                           include_secrets: bool = False) -> Dict[str, Any]:
        """Export current configuration to file or return as dictionary."""
        config_copy = self.config_data.copy()

        # Remove or mask secrets if not including them
        if not include_secrets and self.encryption:
            config_copy = self._mask_secrets(config_copy)

        if file_path:
            file_path = Path(file_path)

            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.safe_dump(config_copy, f, default_flow_style=False, indent=2)
                elif file_path.suffix.lower() == '.json':
                    json.dump(config_copy, f, indent=2, default=str)
                else:
                    raise ConfigurationError(f"Unsupported export format: {file_path.suffix}")

            logger.info(f"Configuration exported to: {file_path}")

        return config_copy

    def _mask_secrets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask encrypted secrets in configuration data."""
        masked_data = {}

        for key, value in data.items():
            if isinstance(value, dict):
                masked_data[key] = self._mask_secrets(value)
            elif isinstance(value, str) and self.encryption and self.encryption.is_encrypted(value):
                masked_data[key] = "***ENCRYPTED***"
            else:
                masked_data[key] = value

        return masked_data

    def _persist_configuration(self):
        """Persist configuration changes to file."""
        # Create backup
        backup_path = self._create_configuration_backup()

        try:
            # Export to main configuration file
            if self.config_paths:
                main_config = self.config_paths[0]
                self.export_configuration(main_config, include_secrets=True)
                logger.info(f"Configuration persisted to: {main_config}")
        except Exception as e:
            # Restore from backup if persistence fails
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, self.config_paths[0])
                logger.error(f"Configuration persistence failed, restored from backup: {str(e)}")
            raise ConfigurationError(f"Failed to persist configuration: {str(e)}")

    def _create_configuration_backup(self) -> Optional[Path]:
        """Create backup of current configuration file."""
        if not self.config_paths:
            return None

        main_config = self.config_paths[0]
        if not main_config.exists():
            return None

        backup_path = main_config.with_suffix(f"{main_config.suffix}.backup")
        shutil.copy2(main_config, backup_path)

        return backup_path

    def close(self):
        """Close configuration manager and cleanup resources."""
        if self.observer:
            self.observer.stop()
            self.observer.join()

        logger.info("UnifiedConfigManager closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global unified configuration instance
_unified_config: Optional[UnifiedConfigManager] = None


def get_unified_config() -> UnifiedConfigManager:
    """
    Get the global unified configuration manager instance.

    Returns:
        UnifiedConfigManager instance
    """
    global _unified_config

    if _unified_config is None:
        _unified_config = UnifiedConfigManager()

    return _unified_config


def configure_unified_config(config_paths: Optional[List[Union[str, Path]]] = None,
                           environment: Optional[str] = None,
                           enable_encryption: bool = True,
                           enable_hot_reload: bool = True,
                           enable_audit: bool = True) -> UnifiedConfigManager:
    """
    Configure and get the global unified configuration manager.

    Args:
        config_paths: List of configuration file paths
        environment: Environment name
        enable_encryption: Enable configuration encryption
        enable_hot_reload: Enable hot reloading
        enable_audit: Enable audit logging

    Returns:
        Configured UnifiedConfigManager instance
    """
    global _unified_config

    if _unified_config:
        _unified_config.close()

    _unified_config = UnifiedConfigManager(
        config_paths=config_paths,
        environment=environment,
        enable_encryption=enable_encryption,
        enable_hot_reload=enable_hot_reload,
        enable_audit=enable_audit
    )

    return _unified_config


def close_unified_config():
    """Close the global unified configuration manager."""
    global _unified_config

    if _unified_config:
        _unified_config.close()
        _unified_config = None


# Legacy compatibility functions for migration
def get_config(key: str, default: Any = None) -> Any:
    """Legacy compatibility: Get configuration value."""
    return get_unified_config().get(key, default)


def get_config_bool(key: str, default: bool = False) -> bool:
    """Legacy compatibility: Get boolean configuration value."""
    return get_unified_config().get_bool(key, default)


def get_config_int(key: str, default: int = 0) -> int:
    """Legacy compatibility: Get integer configuration value."""
    return get_unified_config().get_int(key, default)


def get_config_float(key: str, default: float = 0.0) -> float:
    """Legacy compatibility: Get float configuration value."""
    return get_unified_config().get_float(key, default)


# Configuration decorators for easy injection
def config_value(key: str, default: Any = None, decrypt: bool = False):
    """Decorator to inject configuration value into function parameter."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if key not in kwargs:
                kwargs[key] = get_unified_config().get(key, default, decrypt=decrypt)
            return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def temporary_config(key: str, value: Any):
    """Context manager for temporary configuration changes."""
    config_manager = get_unified_config()
    original_value = config_manager.get(key)

    try:
        config_manager.set(key, value)
        yield
    finally:
        if original_value is not None:
            config_manager.set(key, original_value)
        else:
            config_manager.delete(key)
