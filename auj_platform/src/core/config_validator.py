"""
Configuration Validator for Conservative Production Settings
==========================================================

This module provides validation functions to ensure that conservative
settings are properly applied across the AUJ Platform.
"""

import logging
import os
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import yaml
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a configuration validation"""
    setting_name: str
    expected_value: Any
    actual_value: Any
    is_valid: bool
    message: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO

@dataclass
class ConservativeConfigRequirements:
    """Requirements for conservative production configuration"""

    # Performance settings
    max_indicators_per_regime: int = 10
    cache_duration_minutes_min: int = 10
    execution_timeout_seconds_max: int = 2
    ml_complexity_allowed: List[str] = None
    max_concurrent_calculations_max: int = 8

    # ML settings
    n_estimators_max: int = 75
    max_depth_max: int = 8
    ensemble_size_max: int = 3
    max_features_max: int = 15

    # Cache settings
    max_cache_size_max: int = 800
    cache_cleanup_interval_min: int = 300

    # Risk management - Updated for balanced risk approach
    max_daily_risk_max: float = 0.03
    max_trade_risk_max: float = 0.015

    # Performance thresholds
    performance_alert_threshold_max: float = 1.5
    memory_usage_alert_max: int = 85
    cpu_usage_alert_max: int = 80

    def __post_init__(self):
        if self.ml_complexity_allowed is None:
            self.ml_complexity_allowed = ["low", "medium"]

class ConservativeConfigValidator:
    """Validator for conservative production configuration"""

    def __init__(self):
        self.requirements = ConservativeConfigRequirements()
        self.validation_results: List[ValidationResult] = []

    def validate_conservative_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate that conservative settings are properly applied"""

        self.validation_results.clear()

        if config is None:
            config = self._load_config_from_multiple_sources()

        # Validate different configuration sections
        self._validate_performance_settings(config.get('performance_settings', {}))
        self._validate_selective_engine_settings(config.get('selective_indicator_engine', {}))
        self._validate_ml_settings(config.get('ml_settings', {}))
        self._validate_execution_settings(config.get('execution_settings', {}))
        self._validate_data_caching_settings(config.get('data_caching', {}))
        self._validate_risk_management_settings(config.get('risk_management', {}))
        self._validate_monitoring_settings(config.get('monitoring', {}))
        self._validate_environment_variables()

        # Generate validation summary
        return self._generate_validation_summary()

    def _load_config_from_multiple_sources(self) -> Dict[str, Any]:
        """Load configuration from multiple sources"""
        config = {}

        # Load from YAML file
        yaml_config = self._load_yaml_config()
        if yaml_config:
            config.update(yaml_config)

        # Load from environment variables
        env_config = self._load_env_config()
        config.update(env_config)

        # Load from Python config
        try:
            from auj_platform.src.config import load_production_config
            python_config = load_production_config()
            config.update(python_config)
        except Exception as e:
            logger.warning(f"Could not load Python config: {e}")

        return config

    def _load_yaml_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file"""
        yaml_path = Path(__file__).parent.parent.parent.parent / "config" / "production_conservative.yaml"

        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as file:
                    return yaml.safe_load(file)
            except Exception as e:
                logger.warning(f"Could not load YAML config: {e}")

        return None

    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}

        # Map environment variables to config structure
        env_mappings = {
            'AUJ_MAX_INDICATORS_PER_REGIME': ('performance_settings', 'max_indicators_per_regime', int),
            'AUJ_CACHE_DURATION_MINUTES': ('performance_settings', 'cache_duration_minutes', int),
            'AUJ_EXECUTION_TIMEOUT_SECONDS': ('performance_settings', 'execution_timeout_seconds', int),
            'AUJ_ML_COMPLEXITY': ('performance_settings', 'ml_complexity_level', str),
            'AUJ_MAX_CONCURRENT_CALCULATIONS': ('performance_settings', 'max_concurrent_calculations', int),
            'AUJ_ML_N_ESTIMATORS': ('ml_settings', 'n_estimators', int),
            'AUJ_ML_MAX_DEPTH': ('ml_settings', 'max_depth', int),
            'AUJ_MAX_CACHE_SIZE': ('data_caching', 'max_cache_size', int),
            'AUJ_MAX_DAILY_RISK': ('risk_management', 'max_daily_risk', float),
            'AUJ_MAX_TRADE_RISK': ('risk_management', 'max_trade_risk', float),
            'AUJ_PERFORMANCE_ALERT_THRESHOLD': ('monitoring', 'performance_alert_threshold', float),
            'AUJ_MEMORY_USAGE_ALERT': ('monitoring', 'memory_usage_alert', int),
            'AUJ_CPU_USAGE_ALERT': ('monitoring', 'cpu_usage_alert', int),
        }

        for env_var, (section, key, data_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = data_type(value)
                    if section not in env_config:
                        env_config[section] = {}
                    env_config[section][key] = converted_value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {value} ({e})")

        return env_config

    def _validate_performance_settings(self, settings: Dict[str, Any]) -> None:
        """Validate performance settings"""

        # Max indicators per regime
        self._add_validation_result(
            "max_indicators_per_regime",
            self.requirements.max_indicators_per_regime,
            settings.get('max_indicators_per_regime'),
            lambda x: x is not None and x <= self.requirements.max_indicators_per_regime,
            f"Must be <= {self.requirements.max_indicators_per_regime} for conservative settings"
        )

        # Cache duration
        self._add_validation_result(
            "cache_duration_minutes",
            f">= {self.requirements.cache_duration_minutes_min}",
            settings.get('cache_duration_minutes'),
            lambda x: x is not None and x >= self.requirements.cache_duration_minutes_min,
            f"Must be >= {self.requirements.cache_duration_minutes_min} minutes for conservative caching"
        )

        # Execution timeout
        self._add_validation_result(
            "execution_timeout_seconds",
            f"<= {self.requirements.execution_timeout_seconds_max}",
            settings.get('execution_timeout_seconds'),
            lambda x: x is not None and x <= self.requirements.execution_timeout_seconds_max,
            f"Must be <= {self.requirements.execution_timeout_seconds_max} seconds for conservative execution"
        )

        # ML complexity
        self._add_validation_result(
            "ml_complexity_level",
            self.requirements.ml_complexity_allowed,
            settings.get('ml_complexity_level'),
            lambda x: x in self.requirements.ml_complexity_allowed if x else False,
            f"Must be one of {self.requirements.ml_complexity_allowed} for conservative ML"
        )

        # Max concurrent calculations
        self._add_validation_result(
            "max_concurrent_calculations",
            f"<= {self.requirements.max_concurrent_calculations_max}",
            settings.get('max_concurrent_calculations'),
            lambda x: x is not None and x <= self.requirements.max_concurrent_calculations_max,
            f"Must be <= {self.requirements.max_concurrent_calculations_max} for conservative concurrency"
        )

    def _validate_selective_engine_settings(self, settings: Dict[str, Any]) -> None:
        """Validate selective indicator engine settings"""

        # Max indicators per regime (in selective engine)
        self._add_validation_result(
            "selective_engine.max_indicators_per_regime",
            self.requirements.max_indicators_per_regime,
            settings.get('max_indicators_per_regime'),
            lambda x: x is not None and x <= self.requirements.max_indicators_per_regime,
            f"Selective engine must limit indicators to <= {self.requirements.max_indicators_per_regime}"
        )

        # Correlation threshold
        self._add_validation_result(
            "selective_engine.correlation_threshold",
            ">= 0.7",
            settings.get('correlation_threshold'),
            lambda x: x is not None and x >= 0.7,
            "Conservative correlation threshold should be >= 0.7",
            "WARNING"
        )

        # Stability threshold
        self._add_validation_result(
            "selective_engine.stability_threshold",
            ">= 0.6",
            settings.get('stability_threshold'),
            lambda x: x is not None and x >= 0.6,
            "Conservative stability threshold should be >= 0.6",
            "WARNING"
        )

    def _validate_ml_settings(self, settings: Dict[str, Any]) -> None:
        """Validate ML settings"""

        # Complexity level
        self._add_validation_result(
            "ml_settings.complexity_level",
            self.requirements.ml_complexity_allowed,
            settings.get('complexity_level'),
            lambda x: x in self.requirements.ml_complexity_allowed if x else False,
            f"ML complexity must be {self.requirements.ml_complexity_allowed}"
        )

        # N estimators
        self._add_validation_result(
            "ml_settings.n_estimators",
            f"<= {self.requirements.n_estimators_max}",
            settings.get('n_estimators'),
            lambda x: x is None or x <= self.requirements.n_estimators_max,
            f"N estimators should be <= {self.requirements.n_estimators_max} for conservative ML",
            "WARNING"
        )

        # Max depth
        self._add_validation_result(
            "ml_settings.max_depth",
            f"<= {self.requirements.max_depth_max}",
            settings.get('max_depth'),
            lambda x: x is None or x <= self.requirements.max_depth_max,
            f"Max depth should be <= {self.requirements.max_depth_max} for conservative ML",
            "WARNING"
        )

        # Ensemble size
        self._add_validation_result(
            "ml_settings.ensemble_size",
            f"<= {self.requirements.ensemble_size_max}",
            settings.get('ensemble_size'),
            lambda x: x is None or x <= self.requirements.ensemble_size_max,
            f"Ensemble size should be <= {self.requirements.ensemble_size_max} for conservative ML",
            "WARNING"
        )

    def _validate_execution_settings(self, settings: Dict[str, Any]) -> None:
        """Validate execution settings"""

        # Timeout seconds
        self._add_validation_result(
            "execution_settings.timeout_seconds",
            f"<= {self.requirements.execution_timeout_seconds_max}",
            settings.get('timeout_seconds'),
            lambda x: x is not None and x <= self.requirements.execution_timeout_seconds_max,
            f"Execution timeout must be <= {self.requirements.execution_timeout_seconds_max} seconds"
        )

        # Concurrent limit
        self._add_validation_result(
            "execution_settings.concurrent_limit",
            f"<= {self.requirements.max_concurrent_calculations_max}",
            settings.get('concurrent_limit'),
            lambda x: x is not None and x <= self.requirements.max_concurrent_calculations_max,
            f"Concurrent limit must be <= {self.requirements.max_concurrent_calculations_max}"
        )

    def _validate_data_caching_settings(self, settings: Dict[str, Any]) -> None:
        """Validate data caching settings"""

        # Default expiry minutes
        self._add_validation_result(
            "data_caching.default_expiry_minutes",
            f">= {self.requirements.cache_duration_minutes_min}",
            settings.get('default_expiry_minutes'),
            lambda x: x is not None and x >= self.requirements.cache_duration_minutes_min,
            f"Cache expiry must be >= {self.requirements.cache_duration_minutes_min} minutes"
        )

        # Max cache entries
        self._add_validation_result(
            "data_caching.max_cache_entries",
            f"<= {self.requirements.max_cache_size_max}",
            settings.get('max_cache_entries'),
            lambda x: x is not None and x <= self.requirements.max_cache_size_max,
            f"Max cache entries should be <= {self.requirements.max_cache_size_max} for conservative memory usage"
        )

    def _validate_risk_management_settings(self, settings: Dict[str, Any]) -> None:
        """Validate risk management settings"""

        # Max daily risk
        self._add_validation_result(
            "risk_management.max_daily_risk",
            f"<= {self.requirements.max_daily_risk_max}",
            settings.get('max_daily_risk'),
            lambda x: x is not None and x <= self.requirements.max_daily_risk_max,
            f"Max daily risk must be <= {self.requirements.max_daily_risk_max} ({self.requirements.max_daily_risk_max*100}%) for conservative trading"
        )

        # Max trade risk
        self._add_validation_result(
            "risk_management.max_trade_risk",
            f"<= {self.requirements.max_trade_risk_max}",
            settings.get('max_trade_risk'),
            lambda x: x is not None and x <= self.requirements.max_trade_risk_max,
            f"Max trade risk must be <= {self.requirements.max_trade_risk_max} ({self.requirements.max_trade_risk_max*100}%) for conservative trading"
        )

    def _validate_monitoring_settings(self, settings: Dict[str, Any]) -> None:
        """Validate monitoring settings"""

        # Performance alert threshold
        self._add_validation_result(
            "monitoring.performance_alert_threshold",
            f"<= {self.requirements.performance_alert_threshold_max}",
            settings.get('performance_alert_threshold'),
            lambda x: x is not None and x <= self.requirements.performance_alert_threshold_max,
            f"Performance alert threshold should be <= {self.requirements.performance_alert_threshold_max} seconds",
            "WARNING"
        )

        # Memory usage alert
        self._add_validation_result(
            "monitoring.memory_usage_alert",
            f"<= {self.requirements.memory_usage_alert_max}",
            settings.get('memory_usage_alert'),
            lambda x: x is not None and x <= self.requirements.memory_usage_alert_max,
            f"Memory usage alert should be <= {self.requirements.memory_usage_alert_max}%",
            "WARNING"
        )

        # CPU usage alert
        self._add_validation_result(
            "monitoring.cpu_usage_alert",
            f"<= {self.requirements.cpu_usage_alert_max}",
            settings.get('cpu_usage_alert'),
            lambda x: x is not None and x <= self.requirements.cpu_usage_alert_max,
            f"CPU usage alert should be <= {self.requirements.cpu_usage_alert_max}%",
            "WARNING"
        )

    def _validate_environment_variables(self) -> None:
        """Validate critical environment variables"""

        critical_env_vars = [
            'AUJ_PRODUCTION_MODE',
            'AUJ_MAX_INDICATORS_PER_REGIME',
            'AUJ_EXECUTION_TIMEOUT_SECONDS',
            'AUJ_ML_COMPLEXITY',
            'AUJ_MAX_CONCURRENT_CALCULATIONS'
        ]

        for env_var in critical_env_vars:
            value = os.getenv(env_var)
            self._add_validation_result(
                f"env.{env_var}",
                "set",
                value,
                lambda x: x is not None,
                f"Critical environment variable {env_var} must be set",
                "WARNING" if env_var not in ['AUJ_PRODUCTION_MODE'] else "ERROR"
            )

    def _add_validation_result(self, setting_name: str, expected: Any, actual: Any,
                             validator: callable, message: str, severity: str = "ERROR") -> None:
        """Add a validation result"""

        is_valid = validator(actual)

        result = ValidationResult(
            setting_name=setting_name,
            expected_value=expected,
            actual_value=actual,
            is_valid=is_valid,
            message=message,
            severity=severity
        )

        self.validation_results.append(result)

    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""

        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r.is_valid)
        failed_checks = total_checks - passed_checks

        errors = [r for r in self.validation_results if not r.is_valid and r.severity == "ERROR"]
        warnings = [r for r in self.validation_results if not r.is_valid and r.severity == "WARNING"]

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "overall_status": "PASS" if len(errors) == 0 else "FAIL",
            "conservative_compliance": "COMPLIANT" if len(errors) == 0 and len(warnings) <= 2 else "NON_COMPLIANT",
            "errors": [{"setting": e.setting_name, "message": e.message, "expected": e.expected_value, "actual": e.actual_value} for e in errors],
            "warnings": [{"setting": w.setting_name, "message": w.message, "expected": w.expected_value, "actual": w.actual_value} for w in warnings],
            "all_results": [
                {
                    "setting": r.setting_name,
                    "expected": r.expected_value,
                    "actual": r.actual_value,
                    "valid": r.is_valid,
                    "severity": r.severity,
                    "message": r.message
                } for r in self.validation_results
            ]
        }

        # Log results
        logger.info(f"🔍 Configuration Validation Complete:")
        logger.info(f"   Total Checks: {total_checks}")
        logger.info(f"   Passed: {passed_checks}")
        logger.info(f"   Failed: {failed_checks}")
        logger.info(f"   Errors: {len(errors)}")
        logger.info(f"   Warnings: {len(warnings)}")
        logger.info(f"   Overall Status: {summary['overall_status']}")
        logger.info(f"   Conservative Compliance: {summary['conservative_compliance']}")

        # Log errors and warnings
        for error in errors:
            logger.error(f"❌ {error['setting']}: {error['message']}")

        for warning in warnings:
            logger.warning(f"⚠️ {warning['setting']}: {warning['message']}")

        if summary['overall_status'] == "PASS":
            logger.info("✅ All critical conservative settings validated successfully")
        else:
            logger.error("❌ Conservative settings validation failed - please review errors")

        return summary

# Global validator instance
config_validator = ConservativeConfigValidator()

# Convenience functions
def validate_conservative_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate that conservative settings are properly applied"""
    return config_validator.validate_conservative_config(config)

def quick_validation_check() -> bool:
    """Quick validation check returning True if all critical settings are valid"""
    results = validate_conservative_config()
    return results['overall_status'] == "PASS"

def get_setting_value(setting_path: str) -> Any:
    """Get a setting value from configuration sources"""
    config = config_validator._load_config_from_multiple_sources()

    # Navigate the setting path
    parts = setting_path.split('.')
    value = config

    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None

    return value
