"""
Configuration module for AUJ Platform.

This module provides access to configuration settings throughout the platform.
"""

from auj_platform.src.core.unified_config import get_unified_config
from pathlib import Path
import yaml
import os

def get_config():
    """Get the global configuration instance."""
    return get_unified_config()

def reload_config():
    """Reload configuration from file."""
    config = get_unified_config()
    config.reload()
    return config

def load_production_config():
    """Load conservative production configuration"""
    config = {
        'performance': {
            'max_indicators_per_regime': 10,
            'cache_duration_minutes': 10,
            'execution_timeout_seconds': 2,
            'ml_complexity_level': 'medium',
            'max_concurrent_calculations': 8
        },
        'selective_engine': {
            'max_indicators': 10,
            'correlation_threshold': 0.75,
            'stability_threshold': 0.65
        },
        'ml_settings': {
            'complexity': 'medium',
            'ensemble_size': 3,
            'max_features': 15,
            'n_estimators': 75,
            'max_depth': 8,
            'hidden_layer_sizes': [75, 40]
        },
        'execution_settings': {
            'timeout_seconds': 2,
            'retry_attempts': 2,
            'batch_size': 8,
            'concurrent_limit': 8
        },
        'data_caching': {
            'default_expiry_minutes': 10,
            'max_cache_entries': 800,
            'cleanup_frequency': 300
        },
        'risk_management': {
            'conservative_mode': True,
            'position_sizing_conservative': False,
            'max_daily_risk': 0.03,  # 3% max daily risk - balanced for opportunities
            'max_trade_risk': 0.015,  # 1.5% max risk per trade - allows meaningful positions
            'max_portfolio_risk': 0.20,  # 20% max portfolio risk
            'stop_loss_percentage': 0.02,  # 2% stop loss
            'take_profit_ratio': 2.5,  # 2.5:1 reward to risk ratio
            'max_open_positions': 5  # Allow up to 5 concurrent positions
        },
        'monitoring': {
            'performance_alert_threshold': 1.5,
            'memory_usage_alert': 85,
            'cpu_usage_alert': 80
        }
    }
    return config

def load_production_config_from_file():
    """Load production configuration from YAML file"""
    config_path = Path(__file__).parent.parent.parent / "config" / "production_conservative.yaml"

    if config_path.exists():
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        # Fallback to hardcoded config
        return load_production_config()

# Make Config available as both a class and instance for compatibility
class Config:
    """Configuration class for accessing settings."""

    @classmethod
    def get(cls, key, default=None):
        """Get configuration value by key."""
        config = get_config()
        return config.get(key, default)

    @classmethod
    def database(cls):
        """Get database configuration."""
        config = get_config()
        return config.get_dict('database', {})

    @classmethod
    def trading(cls):
        """Get trading configuration."""
        config = get_config()
        return config.get_dict('trading', {})

    @classmethod
    def risk_parameters(cls):
        """Get risk parameters."""
        config = get_config()
        return config.get_dict('risk_parameters', {})

# Create default instance for compatibility
config = Config()
