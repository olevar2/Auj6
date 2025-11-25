#!/usr/bin/env python3
"""
Dependency Injection Container for AUJ Platform
===============================================

This module implements the dependency injection container that manages
all platform components and their inter-dependencies using the
dependency-injector library.

Enhanced with circular dependency prevention through lazy loading and
component interfaces.

Author: AUJ Platform Development Team
Date: 2025-07-04
Version: 1.1.0
"""

from dependency_injector import containers, providers
from typing import Dict, Any, Optional, TYPE_CHECKING

# Import component interfaces for circular dependency prevention
from .component_interfaces import ComponentRegistry, get_component_registry

# Core system imports
from .unified_config import get_unified_config, UnifiedConfigManager
from .logging_setup import LoggingSetup
from .unified_database_manager import get_unified_database_sync, get_unified_database
from .unified_database_manager import UnifiedDatabaseManager

# Validation and anti-overfitting components
from ..validation.walk_forward_validator import WalkForwardValidator
from ..analytics.performance_tracker import PerformanceTracker
from ..analytics.indicator_effectiveness_analyzer import IndicatorEffectivenessAnalyzer
from ..learning.agent_behavior_optimizer import AgentBehaviorOptimizer
from ..regime_detection.regime_classifier import RegimeClassifier

# Agent and coordination systems
from ..hierarchy.hierarchy_manager import HierarchyManager
from ..coordination.genius_agent_coordinator import GeniusAgentCoordinator
# Monitoring system
from ..monitoring.economic_monitor import EconomicMonitor


# ============================================================================
# FACTORY FUNCTIONS FOR UNIFIED PROVIDERS
# ============================================================================

# Legacy MT5 provider factory removed as part of MetaApi migration


def _create_unified_news_provider(config_manager):
    """Factory function for unified news provider."""
    from ..data_providers.unified_news_economic_provider import UnifiedNewsEconomicProvider
    news_config = config_manager.get('news', {}) if hasattr(config_manager, 'get') else {}
    return UnifiedNewsEconomicProvider(news_config)


def _create_smart_indicator_executor(data_provider_manager):
    """Factory function for SmartIndicatorExecutor."""
    from ..indicator_engine.indicator_executor import SmartIndicatorExecutor, IndicatorRegistry
    registry = IndicatorRegistry()
    return SmartIndicatorExecutor(data_provider_manager, registry)


class PlatformContainer(containers.DeclarativeContainer):
    """
    Main dependency injection container for the AUJ Platform.

    This container defines all major components and their dependencies,
    enabling clean separation of concerns and easier testing.
    """

    # Configuration - the foundation of everything using unified config system
    unified_config_manager = providers.Singleton(get_unified_config)

    # Configuration dictionary provider
    config = providers.Callable(
        lambda: get_unified_config().get_all(),
    )

    # Logging setup
    logging_setup = providers.Singleton(
        LoggingSetup,
        config=unified_config_manager
    )

    # Database manager (legacy for compatibility)
    database_manager = providers.Singleton(
        UnifiedDatabaseManager
    )

    # Unified database manager (new improved version)
    unified_database_manager = providers.Singleton(
        UnifiedDatabaseManager
    )

    # Anti-overfitting validation components
    walk_forward_validator = providers.Singleton(
        WalkForwardValidator
    )

    performance_tracker = providers.Singleton(
        PerformanceTracker,
        config=unified_config_manager,
        database=database_manager,
        walk_forward_validator=walk_forward_validator
    )

    indicator_effectiveness_analyzer = providers.Singleton(
        IndicatorEffectivenessAnalyzer,
        config=unified_config_manager,
        walk_forward_validator=walk_forward_validator,
        performance_tracker=performance_tracker
    )

    agent_behavior_optimizer = providers.Singleton(
        AgentBehaviorOptimizer,
        config=unified_config_manager,
        walk_forward_validator=walk_forward_validator,
        performance_tracker=performance_tracker,
        indicator_analyzer=indicator_effectiveness_analyzer
    )

    regime_classifier = providers.Singleton(
        RegimeClassifier,
        config=unified_config_manager
    )

    # Data and indicator systems
    data_provider_manager = providers.Singleton(
        DataProviderManager,
        config_manager=unified_config_manager
    )

    # Unified news provider 
    unified_news_provider = providers.Singleton(
        lambda: _create_unified_news_provider(get_unified_config()),
    )

    selective_indicator_engine = providers.Singleton(
        SelectiveIndicatorEngine,
        config=unified_config_manager,
        data_manager=data_provider_manager
    )

    # Smart Indicator Executor - Bridge between data providers and indicators
    smart_indicator_executor = providers.Factory(
        lambda data_provider_manager: _create_smart_indicator_executor(data_provider_manager),
        data_provider_manager=data_provider_manager
    )

    # Messaging system
    messaging_service = providers.Singleton(
        MessagingServiceFactory.create_messaging_service,
        config=config
    )

    messaging_coordinator = providers.Singleton(
        MessagingCoordinator,
        config=unified_config_manager
    )

    # Economic monitoring system
    economic_monitor = providers.Singleton(
        EconomicMonitor,
        config_manager=unified_config_manager,
        data_provider_manager=data_provider_manager
    )

    # Hierarchy management
    hierarchy_manager = providers.Singleton(
        HierarchyManager,
        config=unified_config_manager,
        performance_tracker=performance_tracker
    )

    # Trading engine components
    risk_state_repository = providers.Singleton(
        RiskStateRepository,
        db_manager=unified_database_manager
    )

    dynamic_risk_manager = providers.Singleton(
        DynamicRiskManager,
        config_manager=unified_config_manager,
        risk_repository=risk_state_repository
    )

    execution_handler = providers.Singleton(
        ExecutionHandler,
        config_manager=unified_config_manager,
        risk_manager=dynamic_risk_manager,
        messaging_service=messaging_service
    )

    deal_monitoring_teams = providers.Singleton(
        DealMonitoringTeams,
        config=unified_config_manager,
        performance_tracker=performance_tracker
    )

    # Genius Agent Coordinator
    genius_agent_coordinator = providers.Singleton(
        GeniusAgentCoordinator,
        config_manager=unified_config_manager,
        hierarchy_manager=hierarchy_manager,
        indicator_engine=selective_indicator_engine,
        data_manager=data_provider_manager,
        risk_manager=dynamic_risk_manager,
        execution_handler=execution_handler,
        smart_indicator_executor=smart_indicator_executor,
        messaging_service=messaging_service,
        config=config
    )


class ApplicationContainer(containers.DeclarativeContainer):
    """
    Application-level container that provides the main AUJ Platform application.

    This container is responsible for wiring all the components together
    and providing the main application instance.
    """

    # Include the platform container
    platform = providers.DependenciesContainer()

    # Main application factory
    auj_platform = providers.Factory(
        lambda **deps: AUJPlatformDI(**deps),
        config_loader=platform.unified_config_manager,
        database_manager=platform.database_manager,
        walk_forward_validator=platform.walk_forward_validator,
        performance_tracker=platform.performance_tracker,
        indicator_effectiveness_analyzer=platform.indicator_effectiveness_analyzer,
        agent_behavior_optimizer=platform.agent_behavior_optimizer,
        regime_classifier=platform.regime_classifier,
        hierarchy_manager=platform.hierarchy_manager,
        genius_agent_coordinator=platform.genius_agent_coordinator,
        data_provider_manager=platform.data_provider_manager,
        selective_indicator_engine=platform.selective_indicator_engine,
        dynamic_risk_manager=platform.dynamic_risk_manager,
        execution_handler=platform.execution_handler,
        deal_monitoring_teams=platform.deal_monitoring_teams,
        messaging_service=platform.messaging_service,
        messaging_coordinator=platform.messaging_coordinator,
        economic_monitor=platform.economic_monitor
    )


class AUJPlatformDI:
    """
    Main AUJ Platform application class using dependency injection.

    This is a leaner version of the original AUJPlatform class that
    accepts all dependencies via constructor injection.
    """

    def __init__(self,
                 config_loader: 'UnifiedConfigManager',
                 database_manager: UnifiedDatabaseManager,
                 walk_forward_validator: WalkForwardValidator,
                 performance_tracker: PerformanceTracker,
                 indicator_effectiveness_analyzer: IndicatorEffectivenessAnalyzer,
                 agent_behavior_optimizer: AgentBehaviorOptimizer,
                 regime_classifier: RegimeClassifier,
                 hierarchy_manager: HierarchyManager,
                 genius_agent_coordinator: GeniusAgentCoordinator,
                 data_provider_manager: DataProviderManager,
                 selective_indicator_engine: SelectiveIndicatorEngine,
                 dynamic_risk_manager: DynamicRiskManager,
                 execution_handler: ExecutionHandler,
                 deal_monitoring_teams: DealMonitoringTeams,
                 messaging_service: Optional[MessagingService],
                 messaging_coordinator: Optional[MessagingCoordinator],
                 economic_monitor: 'EconomicMonitor'):
        """Initialize AUJ Platform with injected dependencies."""

        # Store injected dependencies
        self.config_loader = config_loader
        self.config = config_loader
        self.database = database_manager
        self.logger = LoggingSetup(config_loader).get_logger(__name__)

        # Anti-overfitting components
        self.walk_forward_validator = walk_forward_validator
        self.performance_tracker = performance_tracker
        self.indicator_analyzer = indicator_effectiveness_analyzer
        self.behavior_optimizer = agent_behavior_optimizer
        self.regime_classifier = regime_classifier

        # Platform systems
        self.hierarchy_manager = hierarchy_manager
        self.coordinator = genius_agent_coordinator
        self.data_manager = data_provider_manager
        self.indicator_engine = selective_indicator_engine
        self.risk_manager = dynamic_risk_manager
        self.execution_handler = execution_handler
        self.deal_monitoring = deal_monitoring_teams

        # Messaging system
        self.messaging_service = messaging_service
        self.messaging_coordinator = messaging_coordinator

        # Monitoring systems
        self.economic_monitor = economic_monitor

        # System state
        self.initialized = False
        self.running = False
        self.shutdown_requested = False

        # Will be created during initialization
        self.feedback_loop = None

    async def initialize(self) -> bool:
        """Initialize all platform components."""
        try:
            self.logger.info("🚀 Initializing AUJ Platform with Dependency Injection")

            # All components are already created and injected by the container
            # We just need to call their initialize methods

            # Initialize core components
            await self.config_loader.load_configuration()
            await self.database.initialize()

            # Initialize anti-overfitting components
            await self.walk_forward_validator.initialize()
            await self.performance_tracker.initialize()
            await self.indicator_analyzer.initialize()
            await self.behavior_optimizer.initialize()

            # Initialize platform systems
            await self.data_manager.initialize()
            await self.indicator_engine.initialize()
            await self.hierarchy_manager.initialize()
            await self.risk_manager.initialize()
            await self.execution_handler.initialize()
            await self.deal_monitoring.initialize()
            await self.coordinator.initialize()

            # Initialize messaging system
            if self.messaging_coordinator:
                await self.messaging_coordinator.initialize()

            # Validate integration
            await self._validate_integration()

            # Initialize daily feedback loop
            await self._initialize_daily_feedback_loop()

            self.initialized = True
            self.logger.info("✅ AUJ Platform initialized successfully with DI")
            self.logger.info("💝 Mission: Generate sustainable profits for sick children and families")

            return True

        except Exception as e:
            self.logger.error(f"❌ Platform initialization failed: {e}")
            return False

    async def _validate_integration(self):
        """Validate that all components are properly integrated."""
        self.logger.info("🔍 Validating system integration...")

        # All components should be initialized by now
        components = {
            "Config Loader": self.config_loader,
            "Database": self.database,
            "Walk-Forward Validator": self.walk_forward_validator,
            "Performance Tracker": self.performance_tracker,
            "Indicator Analyzer": self.indicator_analyzer,
            "Behavior Optimizer": self.behavior_optimizer,
            "Hierarchy Manager": self.hierarchy_manager,
            "Coordinator": self.coordinator,
            "Data Manager": self.data_manager,
            "Indicator Engine": self.indicator_engine,
            "Risk Manager": self.risk_manager,
            "Execution Handler": self.execution_handler,
            "Deal Monitoring": self.deal_monitoring
        }

        # Add messaging components if available
        if self.messaging_service:
            components["Messaging Service"] = self.messaging_service
        if self.messaging_coordinator:
            components["Messaging Coordinator"] = self.messaging_coordinator

        for name, component in components.items():
            if not component:
                raise Exception(f"Component not initialized: {name}")

            # Check if component has health check method
            if hasattr(component, 'health_check'):
                health_status = await component.health_check()
                if not health_status:
                    raise Exception(f"Component health check failed: {name}")

        self.logger.info("✅ All components passed integration validation")

    async def _initialize_daily_feedback_loop(self):
        """Initialize the daily feedback loop with injected dependencies."""
        # Import here to avoid circular imports
        from ..learning.daily_feedback_loop import DailyFeedbackLoop

        self.feedback_loop = DailyFeedbackLoop(
            walk_forward_validator=self.walk_forward_validator,
            performance_tracker=self.performance_tracker,
            indicator_analyzer=self.indicator_analyzer,
            agent_optimizer=self.behavior_optimizer,
            hierarchy_manager=self.hierarchy_manager,
            genius_coordinator=self.coordinator,
            regime_classifier=self.regime_classifier,
            economic_monitor=self.economic_monitor,
            config=self.config_loader
        )

        await self.feedback_loop.initialize()
        self.logger.info("✅ Robust Feedback Loop initialized with DI")

    async def start(self):
        """Start the platform's main execution loop."""
        if not self.initialized:
            raise Exception("Platform not initialized. Call initialize() first.")

        self.running = True
        self.logger.info("🚀 Starting AUJ Platform main execution loop with DI")
        self.logger.info("💰 Mission: Sustainable profits for humanitarian aid")
        self.logger.info("🛡️ Anti-overfitting framework active")

        # Start the daily feedback loop
        await self.feedback_loop.start()

    async def shutdown(self):
        """Gracefully shutdown the platform."""
        if not self.running:
            return

        self.logger.info("🛑 Initiating platform shutdown with DI...")
        self.running = False

        try:
            # Stop feedback loop first
            if self.feedback_loop:
                await self.feedback_loop.stop()
                self.logger.info("✅ Feedback loop stopped")

            # Save critical data
            if self.performance_tracker:
                await self.performance_tracker.save_critical_data()
                self.logger.info("✅ Performance data saved")

            if self.hierarchy_manager:
                await self.hierarchy_manager.save_agent_rankings()
                self.logger.info("✅ Agent rankings saved")

            # Close systems
            if self.deal_monitoring:
                await self.deal_monitoring.shutdown()
                self.logger.info("✅ Deal monitoring stopped")

            if self.messaging_service:
                await self.messaging_service.stop()
                self.logger.info("✅ Messaging service stopped")

            if self.messaging_coordinator:
                await self.messaging_coordinator.shutdown()
                self.logger.info("✅ Messaging coordinator stopped")

            if self.data_manager:
                await self.data_manager.shutdown()
                self.logger.info("✅ Data provider manager stopped")

            if self.database:
                await self.database.close()
                self.logger.info("✅ Database connections closed")

            self.logger.info("✅ Platform shutdown completed successfully with DI")
            self.logger.info("💝 Thank you for supporting sick children and families")

        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
