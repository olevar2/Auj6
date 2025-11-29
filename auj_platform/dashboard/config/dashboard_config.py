"""
Dashboard Configuration Settings
Professional configuration for AUJ Platform Chart Analysis Dashboard

Updated for Linux Migration Phase 2:
- MetaApi as primary data provider
- MT5 explicitly disabled and deprecated
- Aligned with linux_deployment.yaml settings
"""

# Chart settings
CHART_CONFIG = {
    'default_timeframe': '1D',
    'default_pair': 'EUR/USD',
    'max_indicators': 10,
    'chart_height': 600,
    'subplot_height': 150,
    'refresh_intervals': {
        '1s': 1,
        '5s': 5,
        '15s': 15,
        '30s': 30,
        '1m': 60,
        '5m': 300
    },
    'color_scheme': {
        'bullish': '#00ff88',
        'bearish': '#ff4444',
        'neutral': '#ffaa00',
        'background': 'rgba(255, 255, 255, 0.05)',
        'grid': 'rgba(128, 128, 128, 0.3)',
        'gann': '#FFD700',
        'fibonacci': '#FF6B6B',
        'pivot': '#4ECDC4',
        'elliott': '#45B7D1',
        'ai': '#9C27B0',
        'fractal': '#00CED1',
        'volume': '#FFA726',
        'trend': '#8BC34A',
        'momentum': '#E91E63'
    },
    'line_styles': {
        'solid': 'solid',
        'dash': 'dash',
        'dot': 'dot',
        'dashdot': 'dashdot'
    }
}

# Performance settings
PERFORMANCE_CONFIG = {
    'max_refresh_count': 100,
    'cache_ttl': 60,
    'max_retry_attempts': 3,
    'retry_delay': 1,
    'memory_cleanup_interval': 50,
    'max_data_points': 5000,
    'compression_threshold': 1000
}

# Indicator settings
INDICATOR_CONFIG = {
    'max_overlay_indicators': 5,
    'max_subplot_indicators': 4,
    'default_periods': {
        'RSI': 14,
        'MACD_fast': 12,
        'MACD_slow': 26,
        'MACD_signal': 9,
        'MA': 20,
        'EMA': 20,
        'SMA': 20,
        'Fractal': 5,
        'Gann_swing': 20,
        'Fibonacci_swing': 20,
        'Elliott_wave': 50,
        'Bollinger': 20,
        'ATR': 14,
        'Stochastic': 14,
        'Williams': 14,
        'CCI': 20,
        'Momentum': 10,
        'ROC': 12,
        'TRIX': 14,
        'Ultimate': 14,
        'Awesome': 34,
        'Accelerator': 34
    },
    'overlay_categories': [
        'Moving Average', 'Trend', 'Gann', 'Fibonacci', 
        'Elliott', 'Pivot', 'Support Resistance', 'Bollinger'
    ],
    'subplot_categories': [
        'RSI', 'MACD', 'Stochastic', 'CCI', 'Williams', 
        'Momentum', 'Volume', 'Oscillator', 'AI', 'Neural'
    ]
}

# API settings
API_CONFIG = {
    'timeout': 30,
    'max_retries': 3,
    'base_url': 'http://localhost:8000',
    'endpoints': {
        'live_data': '/api/v1/data/live',
        'indicators': '/api/v1/indicators',
        'analysis': '/api/v1/analysis',
        'signals': '/api/v1/signals'
    },
    'headers': {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
}

# UI Configuration
UI_CONFIG = {
    'theme': 'dark',
    'layout': 'wide',
    'sidebar_width': 300,
    'main_panel_ratio': 0.75,
    'control_panel_ratio': 0.25,
    'animation_duration': 500,
    'toast_duration': 3000,
    'loading_spinner': {
        'color': '#00ff88',
        'size': 'medium'
    }
}

# Error Handling Configuration
ERROR_CONFIG = {
    'max_errors_per_session': 50,
    'error_display_duration': 5000,
    'auto_recovery': True,
    'fallback_enabled': True,
    'debug_mode': False,
    'log_level': 'INFO'
}

# Data Validation Rules
VALIDATION_CONFIG = {
    'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'min_data_points': 50,
    'max_data_points': 10000,
    'price_validation': {
        'check_ohlc_relationships': True,
        'check_negative_prices': True,
        'check_outliers': True,
        'outlier_threshold': 3.0  # Standard deviations
    },
    'volume_validation': {
        'check_negative_volume': True,
        'min_volume': 0,
        'max_volume_multiplier': 1000
    }
}

# Professional Features
PROFESSIONAL_CONFIG = {
    'watermark': 'AUJ Platform Dashboard',
    'branding': {
        'logo_url': None,
        'company_name': 'AUJ Platform',
        'tagline': 'Advanced AI Trading Analytics'
    },
    'export_formats': ['CSV', 'Excel', 'JSON', 'PDF'],
    'sharing': {
        'enable_sharing': True,
        'share_formats': ['PNG', 'SVG', 'PDF'],
        'watermark_exports': True
    },
    'alerts': {
        'enable_alerts': True,
        'alert_channels': ['Email', 'SMS', 'Webhook'],
        'max_alerts_per_user': 100
    }
}

# Security Configuration
SECURITY_CONFIG = {
    'enable_authentication': False,
    'session_timeout': 3600,  # 1 hour
    'max_sessions_per_user': 5,
    'rate_limiting': {
        'requests_per_minute': 60,
        'requests_per_hour': 1000
    },
    'data_encryption': False,
    'audit_logging': True
}

# Feature Flags
FEATURE_FLAGS = {
    'ai_indicators': True,
    'real_time_data': True,
    'auto_refresh': True,
    'advanced_charting': True,
    'pattern_recognition': True,
    'signal_generation': True,
    'backtesting': True,
    'portfolio_analysis': True,
    'news_integration': True,
    'social_sentiment': True,
    'economic_calendar': True,
    'multi_timeframe': True,
    'cross_asset_analysis': True,
    'options_analysis': False,
    'crypto_support': True,
    'mobile_responsive': True
}

# =============================================================================
# NEW: Data Provider Configuration (Aligned with Linux Migration Phase 2)
# =============================================================================
# This section reflects the platform's migration to MetaApi as the primary
# data provider, with MT5 explicitly disabled as per linux_deployment.yaml
DATA_PROVIDER_CONFIG = {
    # Primary data provider - aligned with config/linux_deployment.yaml
    'primary_provider': 'metaapi',
    
    # MT5 Configuration - DEPRECATED as per Linux migration
    'mt5': {
        'enabled': False,           # Explicitly disabled
        'deprecated': True,         # Marked as deprecated
        'migration_complete': True, # Migration to MetaApi complete
        'reason': 'Platform migrated to Linux with MetaApi as primary provider'
    },
    
    # MetaApi Configuration - PRIMARY PROVIDER
    'metaapi': {
        'enabled': True,
        'trading': True,
        'streaming': True,
        'websocket_enabled': True,
        'auto_reconnect': True,
        'connection_pooling': True,
        'max_connections': 10
    },
    
    # Supported provider types for dashboard UI
    'supported_providers': ['metaapi', 'ctrader', 'tradingview'],
    
    # Provider display names for UI
    'provider_display_names': {
        'metaapi': 'MetaApi',
        'ctrader': 'cTrader',
        'tradingview': 'TradingView'
    },
    
    # Migration notes
    'migration_info': {
        'phase': 'Phase 2 - Configuration Optimization',
        'date': '2025-11-29',
        'status': 'Complete',
        'details': 'MT5 deprecated, MetaApi is now the sole primary provider'
    }
}

# Default Configuration Function
def get_default_config():
    """Get default configuration dictionary"""
    return {
        'chart': CHART_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'indicators': INDICATOR_CONFIG,
        'api': API_CONFIG,
        'ui': UI_CONFIG,
        'error': ERROR_CONFIG,
        'validation': VALIDATION_CONFIG,
        'professional': PROFESSIONAL_CONFIG,
        'security': SECURITY_CONFIG,
        'features': FEATURE_FLAGS,
        'data_provider': DATA_PROVIDER_CONFIG  # NEW: Added data provider config
    }

# Configuration Validation
def validate_config(config: dict) -> bool:
    """Validate configuration settings"""
    required_sections = [
        'chart', 'performance', 'indicators', 
        'api', 'ui', 'error', 'validation'
    ]
    
    for section in required_sections:
        if section not in config:
            return False
    
    # Validate numeric ranges
    if config['performance']['cache_ttl'] < 1:
        return False
    
    if config['chart']['chart_height'] < 300:
        return False
    
    return True

# Environment-specific configurations
ENVIRONMENTS = {
    'development': {
        'debug': True,
        'cache_ttl': 10,
        'log_level': 'DEBUG',
        'auto_refresh_default': True
    },
    'staging': {
        'debug': False,
        'cache_ttl': 30,
        'log_level': 'INFO',
        'auto_refresh_default': True
    },
    'production': {
        'debug': False,
        'cache_ttl': 60,
        'log_level': 'WARNING',
        'auto_refresh_default': False
    }
}

def get_environment_config(env: str = 'production'):
    """Get environment-specific configuration"""
    base_config = get_default_config()
    env_config = ENVIRONMENTS.get(env, ENVIRONMENTS['production'])
    
    # Merge environment settings
    base_config['error']['debug_mode'] = env_config['debug']
    base_config['performance']['cache_ttl'] = env_config['cache_ttl']
    base_config['error']['log_level'] = env_config['log_level']
    
    return base_config
