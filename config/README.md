# AUJ Platform Configuration Guide

This directory contains all configuration files for the AUJ Trading Platform, with enhanced support for Linux deployment and MetaApi integration.

## Configuration Files Overview

### Core Configuration Files

#### 1. `main_config.yaml`
The primary configuration file containing all platform settings, enhanced with MetaApi support and Linux deployment options.

**Key Features:**
- Cross-platform provider priority configuration
- MetaApi integration settings
- Linux deployment optimization flags
- Environment variable mappings

#### 2. `metaapi_config.yaml` 
**NEW**: Comprehensive MetaApi configuration for Linux deployment.

**Purpose:**
- Replace MT5 direct connection on Linux
- WebSocket streaming configuration
- Trading parameters and risk management
- Linux-specific optimizations

#### 3. `mt5_config.yaml` (Legacy)
**DEPRECATED for Linux**: Legacy MT5 configuration, maintained only for Windows fallback.

**Status:**
- Marked as deprecated for Linux deployment
- Windows-only fallback support
- Migration guide provided

### Platform-Specific Configuration

#### 4. `linux_deployment.yaml`
**NEW**: Linux-specific deployment configuration.

**Contains:**
- System requirements and dependencies
- Service configuration (systemd)
- Security settings
- Container support
- Performance optimization

#### 5. `production_conservative.yaml`
Enhanced with MetaApi and Linux support for conservative production deployment.

#### 6. `monitoring_config.yaml`
Updated with MetaApi monitoring and Linux system monitoring capabilities.

### Environment and Templates

#### 7. `.env.template`
**NEW**: Comprehensive environment variables template for Linux deployment.

**Includes:**
- MetaApi credentials
- Database configuration
- Security keys
- System limits
- Feature flags

#### 8. `config_loader.py`
**NEW**: Advanced configuration management system with cross-platform support.

## Quick Start Guide

### For Linux Deployment

1. **Copy Environment Template:**
   ```bash
   cp config/.env.template .env
   ```

2. **Configure MetaApi Credentials:**
   ```bash
   # Edit .env file
   AUJ_METAAPI_TOKEN=your_token_here
   AUJ_METAAPI_ACCOUNT_ID=your_account_id_here
   ```

3. **Verify Configuration:**
   ```python
   from config.config_loader import validate_config
   issues = validate_config()
   print(issues)  # Should be empty for valid config
   ```

### For Windows Deployment

1. **Use Existing MT5 Configuration:**
   ```bash
   # MT5 direct connection still works on Windows
   AUJ_MT5_PASSWORD=your_mt5_password_here
   ```

2. **Optional MetaApi Setup:**
   ```bash
   # Can also use MetaApi on Windows as primary or fallback
   AUJ_METAAPI_TOKEN=your_token_here
   ```

## Configuration Hierarchy

The configuration system uses the following priority order:

1. **Environment Variables** (highest priority)
2. **Platform-Specific Overrides** (linux_deployment.yaml)
3. **Provider-Specific Configs** (metaapi_config.yaml, mt5_config.yaml)
4. **Main Configuration** (main_config.yaml)
5. **Conservative Defaults** (production_conservative.yaml)

## Provider Priority by Platform

### Linux/Container:
1. **MetaApi** (primary)
2. **Yahoo Finance** (fallback)
3. **Binance** (optional)

### Windows:
1. **MT5 Direct** (if available)
2. **MetaApi** (primary fallback)
3. **Yahoo Finance** (secondary fallback)
4. **Binance** (optional)

### macOS:
1. **MetaApi** (primary)
2. **Yahoo Finance** (fallback)
3. **Binance** (optional)

## Environment Variables

### Critical Variables for MetaApi:

```bash
# MetaApi Authentication
AUJ_METAAPI_TOKEN=your_metaapi_token_here
AUJ_METAAPI_ACCOUNT_ID=your_metaapi_account_id_here

# Security
AUJ_ENCRYPTION_KEY=your_256_bit_key_here
AUJ_API_KEY=your_api_key_here

# Platform Detection
AUJ_PLATFORM=linux  # auto-detected
AUJ_DEPLOYMENT_TYPE=production
```

### Optional Variables:

```bash
# Database (SQLite by default)
AUJ_DB_URL=sqlite:///data/auj_platform.db

# Redis (optional caching)
AUJ_REDIS_URL=redis://localhost:6379/0

# Monitoring
AUJ_LOG_LEVEL=INFO
AUJ_PROMETHEUS_ENABLED=true
```

## Configuration Loading Examples

### Basic Usage:

```python
from config.config_loader import get_config, get_enabled_providers

# Load all configuration
config = get_config()

# Get enabled providers
providers = get_enabled_providers()
print(f"Enabled providers: {providers}")
```

### Provider-Specific Configuration:

```python
from config.config_loader import get_provider_config

# Get MetaApi configuration
metaapi_config = get_provider_config('metaapi')

# Get MT5 configuration (if enabled)
mt5_config = get_provider_config('mt5')
```

### Platform Detection:

```python
from config.config_loader import is_linux_deployment

if is_linux_deployment():
    print("Running on Linux - using MetaApi")
else:
    print("Running on Windows - MT5 available")
```

## Migration from MT5 to MetaApi

### Step 1: Get MetaApi Credentials
1. Register at [MetaApi.cloud](https://app.metaapi.cloud/)
2. Connect your trading account
3. Generate API token
4. Note your account ID

### Step 2: Update Environment
```bash
# Add to .env
AUJ_METAAPI_TOKEN=your_token_here
AUJ_METAAPI_ACCOUNT_ID=your_account_id_here

# Disable MT5 direct (for Linux)
AUJ_MT5_ENABLED=false
```

### Step 3: Test Configuration
```python
from config.config_loader import validate_config
issues = validate_config()
if not issues:
    print("Configuration is valid!")
else:
    print("Issues found:", issues)
```

## Troubleshooting

### Common Issues:

1. **MetaApi Token Missing:**
   ```
   Error: MetaApi token not found in environment variables
   Solution: Set AUJ_METAAPI_TOKEN in .env file
   ```

2. **MT5 Enabled on Linux:**
   ```
   Error: MT5 direct provider is not supported on Linux
   Solution: Set AUJ_MT5_ENABLED=false
   ```

3. **No Providers Enabled:**
   ```
   Error: No data providers are enabled
   Solution: Enable at least MetaApi or Yahoo Finance
   ```

### Debug Mode:

```bash
# Enable debug logging
AUJ_DEBUG_MODE=true
AUJ_LOG_LEVEL=DEBUG
```

### Configuration Validation:

```python
from config.config_loader import config_loader

# Load and validate all configs
config_loader.load_all_configs()
issues = config_loader.validate_configuration()

for issue in issues:
    print(f"⚠️  {issue}")
```

## Security Considerations

### Environment Variables:
- **Never commit .env files to version control**
- Use secure key generation for encryption keys
- Rotate API keys regularly

### File Permissions:
```bash
# Secure configuration files
chmod 600 .env
chmod 644 config/*.yaml
```

### Production Deployment:
- Use environment variable injection
- Enable SSL/TLS for all connections
- Regular security audits

## Performance Optimization

### Linux-Specific:
- Container-optimized settings in `linux_deployment.yaml`
- Memory and CPU limits configured
- File descriptor limits increased

### MetaApi Optimization:
- Connection pooling enabled
- WebSocket streaming for real-time data
- Batch requests for historical data
- Intelligent caching and retry logic

## Monitoring and Alerting

### Health Checks:
- MetaApi connection monitoring
- System resource tracking
- Provider failover detection

### Metrics Collection:
- Prometheus integration
- Grafana dashboards
- Custom alerts for trading issues

## Support and Documentation

### Further Reading:
- [MetaApi Documentation](https://metaapi.cloud/docs/)
- [AUJ Platform Architecture](../docs/architecture/)
- [Linux Deployment Guide](../docs/deployment/)

### Getting Help:
- Check logs in `logs/` directory
- Use configuration validation tools
- Review environment variable template

---

**Last Updated:** January 2025
**Version:** 2.0.0 (MetaApi Integration)