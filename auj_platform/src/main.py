#!/usr/bin/env python3
"""
Main Entry Point for AUJ Platform - Advanced Learning & Anti-Overfitting Framework
====================================================================================

This module serves as the primary entry point for the AUJ automated trading platform,
using dependency injection for clean component management and initialization.

Mission Statement: Generate sustainable profits through intelligent learning while
preventing overfitting to support sick children and families in need.

Author: AUJ Platform Development Team
Date: 2025-07-03
Version: 2.0.0 (With Dependency Injection)
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
import traceback

# Core dependency injection
from .core.containers import PlatformContainer, ApplicationContainer, AUJPlatformDI
from .core.exceptions import AUJSystemError


async def main():
    """
    Main entry function for the AUJ Platform using Dependency Injection.

    Creates the DI container, wires dependencies, and starts the platform.
    """
    platform = None

    try:
        print("=" * 80)
        print("🌟 AUJ PLATFORM - ADVANCED LEARNING & ANTI-OVERFITTING FRAMEWORK 🌟")
        print("=" * 80)
        print("💝 Mission: Generate sustainable profits for sick children and families")
        print("🛡️ Focus: Intelligent learning while preventing overfitting")
        print("🔧 Architecture: Dependency Injection Container")
        print("=" * 80)

        # Create and wire the dependency injection container
        print("📋 Creating dependency injection container...")

        # Create platform container
        platform_container = PlatformContainer()

        # Create application container
        app_container = ApplicationContainer()
        app_container.platform.override(platform_container)

        # Wire the containers
        platform_container.wire(modules=[__name__])
        app_container.wire(modules=[__name__])

        print("✅ Dependency injection container created and wired")

        # Get the platform instance from the container
        print("🚀 Retrieving platform instance from DI container...")
        platform = app_container.auj_platform()

        print("✅ Platform instance created via dependency injection")

        # Initialize platform
        print("📋 Initializing platform components...")
        success = await platform.initialize()

        if not success:
            print("❌ Platform initialization failed")
            return 1

        print("✅ Platform initialization completed successfully")
        print("🚀 Starting main execution loop...")
        print("=" * 80)

        # Register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\n📢 Received signal {signum}, initiating graceful shutdown...")
            platform.shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start platform operation
        await platform.start()

        return 0

    except KeyboardInterrupt:
        print("\n⏹️ Shutdown requested by user")
        return 0

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1

    finally:
        # Ensure graceful shutdown
        if platform:
            try:
                await platform.shutdown()
            except Exception as e:
                print(f"❌ Error during shutdown: {e}")


if __name__ == "__main__":
    """
    Entry point when script is run directly.

    Uses dependency injection for clean component management.
    """
    try:
        # Run the platform with dependency injection
        exit_code = asyncio.run(main())
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n⏹️ Platform stopped by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
