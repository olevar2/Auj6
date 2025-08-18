"""
Platform Detection and Broker Selection

Automatically detects the operating system and selects
the appropriate broker interface accordingly.

Enhanced for Linux deployment with MetaApi as primary provider.
"""

import os
import platform
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict


class PlatformType(Enum):
    """Supported platform types."""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


class BrokerType(Enum):
    """Available broker types."""
    MT5_DIRECT = "mt5_direct"  # Windows only
    METAAPI = "metaapi"        # Cross-platform
    OANDA = "oanda"           # Cross-platform
    UNKNOWN = "unknown"


class PlatformDetection:
    """Detect platform and determine available broker interfaces."""

    @staticmethod
    def get_platform() -> PlatformType:
        """Get current platform type."""
        system = platform.system().lower()

        if system == 'windows':
            return PlatformType.WINDOWS
        elif system == 'linux':
            return PlatformType.LINUX
        elif system == 'darwin':
            return PlatformType.MACOS
        else:
            return PlatformType.UNKNOWN

    @staticmethod
    def is_windows() -> bool:
        """Check if running on Windows."""
        return PlatformDetection.get_platform() == PlatformType.WINDOWS

    @staticmethod
    def is_linux() -> bool:
        """Check if running on Linux."""
        return PlatformDetection.get_platform() == PlatformType.LINUX

    @staticmethod
    def is_macos() -> bool:
        """Check if running on macOS."""
        return PlatformDetection.get_platform() == PlatformType.MACOS

    @staticmethod
    def can_use_mt5_direct() -> bool:
        """Check if direct MT5 library can be used."""
        try:
            # Only Windows supports direct MT5
            if not PlatformDetection.is_windows():
                return False

            # Try importing MT5 library
            import MetaTrader5
            return True
        except ImportError:
            return False

    @staticmethod
    def get_python_version() -> str:
        """Get Python version information."""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    @staticmethod
    def get_architecture() -> str:
        """Get system architecture."""
        return platform.machine()

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "platform": PlatformDetection.get_platform().value,
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": PlatformDetection.get_architecture(),
            "processor": platform.processor(),
            "python_version": PlatformDetection.get_python_version(),
            "python_implementation": platform.python_implementation(),
            "can_use_mt5_direct": PlatformDetection.can_use_mt5_direct()
        }

    @staticmethod
    def get_recommended_broker_type() -> BrokerType:
        """Get recommended broker type based on platform."""
        if PlatformDetection.is_linux():
            # Linux: Force MetaApi
            return BrokerType.METAAPI
        elif PlatformDetection.is_windows():
            # Windows: Try MT5 direct first, fallback to MetaApi
            if PlatformDetection.can_use_mt5_direct():
                return BrokerType.MT5_DIRECT
            else:
                return BrokerType.METAAPI
        elif PlatformDetection.is_macos():
            # macOS: MetaApi only
            return BrokerType.METAAPI
        else:
            return BrokerType.UNKNOWN

    @staticmethod
    def get_recommended_broker_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommended broker configuration based on platform."""
        platform_type = PlatformDetection.get_platform()
        recommended_broker = PlatformDetection.get_recommended_broker_type()

        # Base configuration
        recommended_config = {
            "platform_detected": platform_type.value,
            "primary_broker": recommended_broker.value,
            "fallback_broker": None,
            "deployment_mode": "linux_optimized" if platform_type == PlatformType.LINUX else "multi_platform",
            "force_metaapi": platform_type in [PlatformType.LINUX, PlatformType.MACOS],
            "system_info": PlatformDetection.get_system_info(),
            "brokers": {}
        }

        if platform_type == PlatformType.LINUX:
            # Linux: MetaApi only - optimized for production deployment
            recommended_config.update({
                "fallback_broker": None,
                "deployment_note": "Linux deployment using MetaApi cloud service for optimal performance",
                "brokers": {
                    "metaapi": {
                        "enabled": True,
                        "priority": "primary",
                        "reason": "Linux deployment - MetaApi cloud service",
                        "timeout": 30,
                        "region": "london",  # Default region
                        "retry_attempts": 3,
                        "websocket_enabled": True,
                        **config.get("brokers", {}).get("metaapi", {})
                    },
                    "mt5_direct": {
                        "enabled": False,
                        "priority": "disabled",
                        "reason": "Not supported on Linux - use MetaApi instead"
                    }
                }
            })

        elif platform_type == PlatformType.WINDOWS:
            if PlatformDetection.can_use_mt5_direct():
                # Windows with MT5 library available
                recommended_config.update({
                    "fallback_broker": BrokerType.METAAPI.value,
                    "deployment_note": "Windows with MT5 direct support, MetaApi as fallback",
                    "brokers": {
                        "mt5_direct": {
                            "enabled": True,
                            "priority": "primary",
                            "reason": "Windows with MT5 library available",
                            "timeout": 60,
                            **config.get("brokers", {}).get("mt5", {})
                        },
                        "metaapi": {
                            "enabled": True,
                            "priority": "fallback",
                            "reason": "Fallback for MT5 direct",
                            "timeout": 30,
                            "region": "london",
                            **config.get("brokers", {}).get("metaapi", {})
                        }
                    }
                })
            else:
                # Windows without MT5 library
                recommended_config.update({
                    "fallback_broker": None,
                    "deployment_note": "Windows without MT5 library - using MetaApi",
                    "brokers": {
                        "metaapi": {
                            "enabled": True,
                            "priority": "primary",
                            "reason": "MT5 library not available",
                            "timeout": 30,
                            "region": "london",
                            **config.get("brokers", {}).get("metaapi", {})
                        },
                        "mt5_direct": {
                            "enabled": False,
                            "priority": "disabled",
                            "reason": "MT5 library not found"
                        }
                    }
                })

        elif platform_type == PlatformType.MACOS:
            # macOS: MetaApi only (similar to Linux)
            recommended_config.update({
                "fallback_broker": None,
                "deployment_note": "macOS deployment using MetaApi cloud service",
                "brokers": {
                    "metaapi": {
                        "enabled": True,
                        "priority": "primary",
                        "reason": "macOS deployment - MetaApi cloud service",
                        "timeout": 30,
                        "region": "london",
                        **config.get("brokers", {}).get("metaapi", {})
                    },
                    "mt5_direct": {
                        "enabled": False,
                        "priority": "disabled",
                        "reason": "Not supported on macOS"
                    }
                }
            })

        else:
            # Unknown platform
            recommended_config.update({
                "fallback_broker": None,
                "deployment_note": "Unknown platform - manual configuration required",
                "brokers": {
                    "metaapi": {
                        "enabled": False,
                        "priority": "disabled",
                        "reason": "Unknown platform - manual configuration required"
                    },
                    "mt5_direct": {
                        "enabled": False,
                        "priority": "disabled",
                        "reason": "Unknown platform - manual configuration required"
                    }
                }
            })

        return recommended_config


    @staticmethod
    def check_metaapi_requirements() -> Dict[str, Any]:
        """Check if MetaApi requirements are met."""
        requirements_check = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "required_env_vars": ["AUJ_METAAPI_TOKEN", "AUJ_METAAPI_ACCOUNT_ID"],
            "optional_env_vars": ["AUJ_METAAPI_REGION", "AUJ_METAAPI_TIMEOUT"]
        }

        # Check required environment variables
        for env_var in requirements_check["required_env_vars"]:
            if not os.getenv(env_var):
                requirements_check["valid"] = False
                requirements_check["errors"].append(f"Missing required environment variable: {env_var}")

        # Check optional environment variables
        for env_var in requirements_check["optional_env_vars"]:
            if not os.getenv(env_var):
                requirements_check["warnings"].append(f"Optional environment variable not set: {env_var}")

        # Check internet connectivity (simplified)
        try:
            import urllib.request
            urllib.request.urlopen('https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai', timeout=5)
            requirements_check["internet_connection"] = True
        except:
            requirements_check["valid"] = False
            requirements_check["errors"].append("Cannot reach MetaApi servers - check internet connection")
            requirements_check["internet_connection"] = False

        return requirements_check

    @staticmethod
    def get_linux_deployment_config() -> Dict[str, Any]:
        """Get Linux-specific deployment configuration."""
        return {
            "deployment_type": "linux_production",
            "service_management": "systemd",
            "log_directory": "/var/log/auj",
            "data_directory": "/var/lib/auj",
            "config_directory": "/etc/auj",
            "user": "auj",
            "group": "auj",
            "permissions": {
                "log_files": "644",
                "config_files": "600",
                "data_files": "644"
            },
            "systemd_service": {
                "name": "auj-platform",
                "description": "AUJ Trading Platform",
                "working_directory": "/opt/auj",
                "environment_file": "/etc/auj/environment",
                "restart_policy": "always"
            },
            "firewall": {
                "required_outbound": [
                    "443/tcp",  # HTTPS for MetaApi
                    "80/tcp",   # HTTP for basic connectivity
                    "53/tcp",   # DNS
                    "53/udp"    # DNS
                ],
                "optional_inbound": [
                    "8080/tcp"  # Dashboard if needed
                ]
            }
        }

    @staticmethod
    def generate_environment_file() -> str:
        """Generate environment file content for Linux deployment."""
        env_template = """# AUJ Platform Environment Configuration
# Generated by Platform Detection System

# Platform Configuration
AUJ_PLATFORM_MODE=linux
AUJ_DEPLOYMENT_ENV=production
AUJ_BROKER_TYPE=metaapi

# MetaApi Configuration (REQUIRED)
AUJ_METAAPI_TOKEN=your_metaapi_token_here
AUJ_METAAPI_ACCOUNT_ID=your_account_id_here
AUJ_METAAPI_REGION=london
AUJ_METAAPI_TIMEOUT=30

# Database Configuration
AUJ_DATABASE_URL=sqlite:///var/lib/auj/platform.db
AUJ_DATABASE_BACKUP_DIR=/var/lib/auj/backups

# Logging Configuration
AUJ_LOG_LEVEL=INFO
AUJ_LOG_FILE=/var/log/auj/platform.log
AUJ_LOG_MAX_SIZE=100MB
AUJ_LOG_BACKUP_COUNT=5

# Security Configuration
AUJ_SECRET_KEY=generate_your_secret_key_here
AUJ_ENCRYPTION_KEY=generate_your_encryption_key_here

# Performance Configuration
AUJ_MAX_WORKERS=4
AUJ_WORKER_TIMEOUT=300
AUJ_MEMORY_LIMIT=2GB

# Optional Dashboard Configuration
AUJ_DASHBOARD_ENABLED=false
AUJ_DASHBOARD_PORT=8080
AUJ_DASHBOARD_HOST=127.0.0.1

# Monitoring Configuration
AUJ_METRICS_ENABLED=true
AUJ_HEALTH_CHECK_INTERVAL=60
"""
        return env_template

    @staticmethod
    def is_containerized() -> bool:
        """Check if running in a container (Docker, Podman, etc.)."""
        # Check for container indicators
        container_indicators = [
            Path("/.dockerenv").exists(),
            Path("/run/.containerenv").exists(),
            os.getenv("container") is not None,
            os.getenv("KUBERNETES_SERVICE_HOST") is not None
        ]
        return any(container_indicators)

    @staticmethod
    def get_container_info() -> Dict[str, Any]:
        """Get container information if running in container."""
        if not PlatformDetection.is_containerized():
            return {"containerized": False}

        container_info = {"containerized": True}

        # Docker detection
        if Path("/.dockerenv").exists():
            container_info["type"] = "docker"

        # Podman detection
        elif Path("/run/.containerenv").exists():
            container_info["type"] = "podman"

        # Kubernetes detection
        elif os.getenv("KUBERNETES_SERVICE_HOST"):
            container_info["type"] = "kubernetes"
            container_info["namespace"] = os.getenv("KUBERNETES_NAMESPACE", "default")
            container_info["pod_name"] = os.getenv("HOSTNAME")

        # Generic container
        else:
            container_info["type"] = "generic"

        return container_info

    @staticmethod
    def get_required_dependencies() -> Dict[str, list]:
        """Get required dependencies based on platform."""
        platform_type = PlatformDetection.get_platform()

        # Common dependencies for all platforms
        common_deps = [
            "asyncio>=3.4.3",
            "aiohttp>=3.8.0",
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "pydantic>=2.0.0",
            "requests>=2.28.0",
            "sqlalchemy>=2.0.0",
            "metaapi-cloud-sdk>=21.0.0"  # MetaApi is now required for all platforms
        ]

        # Platform-specific dependencies
        if platform_type == PlatformType.WINDOWS:
            windows_deps = [
                "pywin32>=306"
            ]
            optional_deps = [
                "MetaTrader5>=5.0.0"  # Optional for direct MT5 on Windows
            ]
            return {
                "common": common_deps,
                "platform_specific": windows_deps,
                "optional": optional_deps,
                "note": "MetaTrader5 is optional on Windows - MetaApi will be used as fallback"
            }

        elif platform_type in [PlatformType.LINUX, PlatformType.MACOS]:
            unix_deps = [
                "python-daemon>=3.0.0",
                "psutil>=5.9.0"
            ]
            return {
                "common": common_deps,
                "platform_specific": unix_deps,
                "optional": [],
                "note": "Linux/macOS uses MetaApi exclusively - no MT5 direct support"
            }

        else:
            return {
                "common": common_deps,
                "platform_specific": [],
                "optional": [],
                "note": "Unknown platform - may require manual dependency management"
            }

    @staticmethod
    def validate_platform_requirements() -> Dict[str, Any]:
        """Validate if platform meets requirements for Linux deployment."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }

        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Python 3.8+ required, found {PlatformDetection.get_python_version()}"
            )

        # Platform-specific checks
        platform_type = PlatformDetection.get_platform()

        if platform_type == PlatformType.LINUX:
            validation_result["info"].append("✅ Linux detected - optimal for MetaApi deployment")

            # Check MetaApi requirements for Linux
            metaapi_check = PlatformDetection.check_metaapi_requirements()
            if not metaapi_check["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(metaapi_check["errors"])
            validation_result["warnings"].extend(metaapi_check["warnings"])

            # Check container environment
            if PlatformDetection.is_containerized():
                container_info = PlatformDetection.get_container_info()
                validation_result["info"].append(f"🐳 Container detected: {container_info.get('type', 'unknown')}")

        elif platform_type == PlatformType.WINDOWS:
            # Check if MT5 can be used
            if PlatformDetection.can_use_mt5_direct():
                validation_result["info"].append("MT5 direct library available - will use as primary")
                validation_result["info"].append("MetaApi will be configured as fallback")
            else:
                validation_result["warnings"].append(
                    "MT5 direct library not available - will use MetaApi only"
                )

                # Check MetaApi requirements for Windows fallback
                metaapi_check = PlatformDetection.check_metaapi_requirements()
                if not metaapi_check["valid"]:
                    validation_result["valid"] = False
                    validation_result["errors"].extend(metaapi_check["errors"])
                validation_result["warnings"].extend(metaapi_check["warnings"])

        elif platform_type == PlatformType.MACOS:
            validation_result["info"].append("macOS detected - using MetaApi cloud service")

            # Check MetaApi requirements for macOS
            metaapi_check = PlatformDetection.check_metaapi_requirements()
            if not metaapi_check["valid"]:
                validation_result["valid"] = False
                validation_result["errors"].extend(metaapi_check["errors"])
            validation_result["warnings"].extend(metaapi_check["warnings"])

        else:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Unsupported platform: {platform.system()}")

        return validation_result

    @staticmethod
    def log_platform_info() -> str:
        """Generate comprehensive platform information log for Linux deployment."""
        system_info = PlatformDetection.get_system_info()
        recommended_broker = PlatformDetection.get_recommended_broker_type()
        validation = PlatformDetection.validate_platform_requirements()

        log_lines = [
            "=" * 70,
            "🚀 AUJ PLATFORM - LINUX DEPLOYMENT DETECTION",
            "=" * 70,
            f"📱 Platform: {system_info['platform'].title()}",
            f"🖥️  System: {system_info['system']} {system_info['release']}",
            f"🏗️  Architecture: {system_info['architecture']}",
            f"🐍 Python: {system_info['python_version']} ({system_info['python_implementation']})",
            f"📊 Recommended Broker: {recommended_broker.value.title()}",
            f"🔗 MT5 Direct Available: {'✅ Yes' if system_info['can_use_mt5_direct'] else '❌ No (Linux - using MetaApi)'}",
            "-" * 70,
        ]

        # Add container information if applicable
        if PlatformDetection.is_containerized():
            container_info = PlatformDetection.get_container_info()
            log_lines.extend([
                f"🐳 Container: {container_info.get('type', 'unknown').title()}",
                "-" * 70,
            ])

        # Add Linux deployment info
        if PlatformDetection.is_linux():
            linux_config = PlatformDetection.get_linux_deployment_config()
            log_lines.extend([
                "🐧 LINUX DEPLOYMENT CONFIGURATION:",
                f"   Service Management: {linux_config['service_management']}",
                f"   Log Directory: {linux_config['log_directory']}",
                f"   Data Directory: {linux_config['data_directory']}",
                f"   Config Directory: {linux_config['config_directory']}",
                "-" * 70,
            ])

        log_lines.append("🔍 PLATFORM VALIDATION:")

        if validation["valid"]:
            log_lines.append("✅ Platform validation PASSED")
        else:
            log_lines.append("❌ Platform validation FAILED")

        for error in validation["errors"]:
            log_lines.append(f"❌ ERROR: {error}")

        for warning in validation["warnings"]:
            log_lines.append(f"⚠️  WARNING: {warning}")

        for info in validation["info"]:
            log_lines.append(f"ℹ️  INFO: {info}")

        # Add MetaApi specific info
        if recommended_broker == BrokerType.METAAPI:
            log_lines.extend([
                "-" * 70,
                "🌐 METAAPI CONFIGURATION:",
                "   ✅ Cross-platform cloud trading service",
                "   ✅ WebSocket real-time data streaming",
                "   ✅ Complete MT5 API functionality",
                "   ✅ Linux production ready",
                "   📝 Requires: METAAPI_TOKEN and ACCOUNT_ID",
            ])

        log_lines.append("=" * 70)

        return "\n".join(log_lines)


def main():
    """Test platform detection functionality for Linux deployment."""
    print(PlatformDetection.log_platform_info())

    # Test configuration generation
    test_config = {
        "brokers": {
            "metaapi": {
                "api_token": "test_token",
                "account_id": "test_account",
                "region": "london",
                "timeout": 30
            },
            "mt5": {
                "login": "12345",
                "password": "password",
                "server": "demo-server"
            }
        }
    }

    recommended_config = PlatformDetection.get_recommended_broker_config(test_config)
    print("\n🔧 RECOMMENDED CONFIGURATION:")
    print("-" * 50)
    for key, value in recommended_config.items():
        if key != "brokers":
            print(f"{key}: {value}")

    print("\n📊 BROKER CONFIGURATIONS:")
    for broker_name, broker_config in recommended_config["brokers"].items():
        status = "✅ ENABLED" if broker_config.get("enabled") else "❌ DISABLED"
        priority = broker_config.get("priority", "unknown").upper()
        print(f"\n{broker_name.upper()} ({status} - {priority}):")
        for config_key, config_value in broker_config.items():
            if config_key not in ["enabled", "priority"]:
                print(f"  {config_key}: {config_value}")

    # Test MetaApi requirements
    print("\n🌐 METAAPI REQUIREMENTS CHECK:")
    print("-" * 50)
    metaapi_check = PlatformDetection.check_metaapi_requirements()
    if metaapi_check["valid"]:
        print("✅ MetaApi requirements satisfied")
    else:
        print("❌ MetaApi requirements NOT satisfied")
        for error in metaapi_check["errors"]:
            print(f"   ❌ {error}")

    for warning in metaapi_check["warnings"]:
        print(f"   ⚠️  {warning}")

    # Test dependencies
    print("\n📦 REQUIRED DEPENDENCIES:")
    print("-" * 50)
    deps = PlatformDetection.get_required_dependencies()
    print("Common dependencies:")
    for dep in deps["common"]:
        print(f"  ✓ {dep}")

    if deps["platform_specific"]:
        print(f"\nPlatform-specific dependencies ({PlatformDetection.get_platform().value}):")
        for dep in deps["platform_specific"]:
            print(f"  ✓ {dep}")

    if deps["optional"]:
        print("\nOptional dependencies:")
        for dep in deps["optional"]:
            print(f"  ? {dep}")

    if "note" in deps:
        print(f"\n📝 Note: {deps['note']}")

    # Generate environment file template
    if PlatformDetection.is_linux():
        print("\n🐧 LINUX ENVIRONMENT FILE TEMPLATE:")
        print("-" * 50)
        env_content = PlatformDetection.generate_environment_file()
        print(env_content[:500] + "..." if len(env_content) > 500 else env_content)


if __name__ == "__main__":
    main()
