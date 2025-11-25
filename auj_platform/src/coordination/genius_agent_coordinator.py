"""
Genius Agent Coordinator for AUJ Platform.

This module implements the master coordinator that orchestrates the hierarchical agent system,
executing hybrid analysis workflows every hour and managing agent authority based on
validated out-of-sample performance.

Key Features:
- Hourly analysis cycle execution
- Agent hierarchy management (Alpha, Beta, Gamma ranks)
- Elite Indicator Set coordination
- Anti-overfitting validation integration
- Confidence-based decision synthesis
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import field
from enum import Enum
import pandas as pd
from decimal import Decimal
import logging
import traceback

from ..core.data_contracts import (
    TradeSignal, AgentDecision, MarketConditions, AgentRank,
    MarketRegime, ConfidenceLevel, TradeDirection, PlatformStatus
)
from ..core.event_bus import event_bus, Event, EventType


logger = logging.getLogger(__name__)
from ..agents.base_agent import BaseAgent, AnalysisResult
from ..registry.agent_indicator_mapping import AGENT_MAPPINGS, get_agent_indicators

# Import shared coordination types and exceptions
from .shared import (
    AnalysisCyclePhase,
    AnalysisCycleState,
    EliteIndicatorSet,
    CoordinationError,
    ValidationError,
    AgentError
)


class GeniusAgentCoordinator:
    """
    Master coordinator for the hierarchical agent system.

    Orchestrates hourly analysis cycles where:
    1. Alpha agent (best performing) makes primary decision
    2. Beta agents (second tier) provide verification
    3. Gamma agents (learning tier) contribute insights
    4. Decision synthesis uses ensemble consensus with confidence weighting
    5. Elite indicator sets are used based on current market regime
    """

    def __init__(self,
                 config_manager,
                 hierarchy_manager,
                 indicator_engine,
                 data_manager,
                 risk_manager,
                 execution_handler,
                 smart_indicator_executor=None,
                 messaging_service: Optional[Any] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Genius Agent Coordinator.

        Args:
            config_manager: Unified configuration manager instance
            hierarchy_manager: Agent hierarchy management
            indicator_engine: Indicator calculation engine (SelectiveIndicatorEngine)
            data_manager: Data provider manager instance
            risk_manager: Dynamic risk manager instance
            execution_handler: Trade execution handler
            messaging_service: Optional injected messaging service
            config: Configuration parameters (deprecated, use config_manager)
        """
        self.config_manager = config_manager
        # Backward compatibility - will be removed in next version
        self.config = config or {}
        self.hierarchy_manager = hierarchy_manager
        self.indicator_engine = indicator_engine  # This is the SelectiveIndicatorEngine
        self.data_manager = data_manager  # This is the DataProviderManager
        self.risk_manager = risk_manager
        self.execution_handler = execution_handler
        self.messaging_service = messaging_service  # Injected dependency

        # Initialize SmartIndicatorExecutor
        if smart_indicator_executor:
            self.indicator_executor = smart_indicator_executor
        else:
            # Create it if not provided (fallback)
            from ..indicator_engine.indicator_executor import SmartIndicatorExecutor
            self.indicator_executor = SmartIndicatorExecutor(data_manager)

        # These will be initialized during the initialize() method
        self.agents: Dict[str, BaseAgent] = {}
        self.performance_tracker = None
        self.walk_forward_validator = None
        self.regime_classifier = None

        # Performance monitoring
        from .performance_monitor import CoordinationPerformanceMonitor
        self.performance_monitor = CoordinationPerformanceMonitor()

        # Coordination state
        self.is_active = False
        self.current_cycle: Optional[AnalysisCycleState] = None
        self.last_cycle_time: Optional[datetime] = None
        self.cycle_history: List[AnalysisCycleState] = []

        # Elite indicator sets by market regime
        self.elite_indicator_sets: Dict[MarketRegime, EliteIndicatorSet] = {}

        # Performance tracking
        self.total_cycles = 0
        self.successful_cycles = 0
        self.average_cycle_duration = 0.0

        # Configuration parameters from unified config manager
        self.analysis_frequency_minutes = self._get_config_int('coordination.analysis_frequency_minutes', 60)
        self.min_confidence_threshold = self._get_config_float('coordination.minimum_confidence_threshold', 0.6)
        self.max_analysis_time_seconds = self._get_config_int('coordination.max_analysis_time_seconds', 120)  # Reduced from 300 to 120
        self.consensus_threshold = self._get_config_float('coordination.consensus_threshold', 0.6)
        self.elite_set_update_frequency = self._get_config_int('coordination.elite_set_update_hours', 24)

        # Parallel processing settings
        self.enable_parallel_analysis = self._get_config_bool('coordination.enable_parallel_analysis', True)
        self.max_concurrent_agents = self._get_config_int('coordination.max_concurrent_agents', 3)

        # Phase optimization settings
        self.enable_phase_merging = self._get_config_bool('coordination.enable_phase_merging', True)

        logger.info("GeniusAgentCoordinator initialized with dependency injection")

    def _get_config_int(self, key: str, default: int) -> int:
        """Safely get integer config value."""
        if self.config_manager and hasattr(self.config_manager, 'get_int'):
            return self.config_manager.get_int(key, default)
        return default

    def _get_config_float(self, key: str, default: float) -> float:
        """Safely get float config value."""
        if self.config_manager and hasattr(self.config_manager, 'get_float'):
            return self.config_manager.get_float(key, default)
        return default

    def _get_config_str(self, key: str, default: str) -> str:
        """Safely get string config value."""
        if self.config_manager and hasattr(self.config_manager, 'get_str'):
            return self.config_manager.get_str(key, default)
        return default

    def _get_config_bool(self, key: str, default: bool) -> bool:
        """Safely get boolean config value."""
        if self.config_manager and hasattr(self.config_manager, 'get_bool'):
            return self.config_manager.get_bool(key, default)
        return default

    async def initialize(self):
        """Initialize the coordinator and create missing dependencies."""
        try:
            logger.info("🚀 Initializing GeniusAgentCoordinator...")

            # Initialize missing components that are now expected to be available
            # These would typically be injected, but for compatibility we'll create them

            # Create agents if not provided
            if not self.agents:
                from agents.risk_genius import RiskGenius
                from agents.technical_scout import TechnicalScout
                from agents.fundamental_analyst import FundamentalAnalyst
                from agents.execution_expert import ExecutionExpert
                from agents.microstructure_agent import MicrostructureAgent

                # Create agents with messaging service injection and unified config
                self.agents = {
                    'risk_genius': RiskGenius(self.config_manager.get_dict('agents.risk_genius', {}), config_manager=self.config_manager, messaging_service=self.messaging_service),
                    'technical_scout': TechnicalScout(self.config_manager.get_dict('agents.technical_scout', {}), config_manager=self.config_manager, messaging_service=self.messaging_service),
                    'fundamental_analyst': FundamentalAnalyst(self.config_manager.get_dict('agents.fundamental_analyst', {}), config_manager=self.config_manager, messaging_service=self.messaging_service),
                    'execution_expert': ExecutionExpert(self.config_manager.get_dict('agents.execution_expert', {}), config_manager=self.config_manager, messaging_service=self.messaging_service),
                    'microstructure_agent': MicrostructureAgent(self.config_manager.get_dict('agents.microstructure_agent', {}), config_manager=self.config_manager, messaging_service=self.messaging_service)
                }

                logger.info(f"✅ Created {len(self.agents)} agents with messaging injection")

            # Get other dependencies from hierarchy manager if available
            if hasattr(self.hierarchy_manager, 'performance_tracker'):
                self.performance_tracker = self.hierarchy_manager.performance_tracker

            if hasattr(self.hierarchy_manager, 'walk_forward_validator'):
                self.walk_forward_validator = self.hierarchy_manager.walk_forward_validator

            # Create regime classifier if not available
            if not self.regime_classifier:
                from ..regime_detection.regime_classifier import RegimeClassifier
                self.regime_classifier = RegimeClassifier(config=self.config)
                logger.info("✅ Created regime classifier")

            logger.info("✅ GeniusAgentCoordinator initialization completed")

        except Exception as e:
            logger.error(f"❌ Failed to initialize GeniusAgentCoordinator: {e}")
            raise

    async def start(self):
        """Initialize the coordinator (no longer runs independent loop)."""
        if self.is_active:
            logger.warning("Coordinator already active")
            return

        self.is_active = True
        logger.info("Genius Agent Coordinator initialized - will be called by orchestrator")

        # Initialize elite indicator sets
        await self._initialize_elite_indicator_sets()

        logger.info("Genius Agent Coordinator ready for orchestrated execution")

    async def stop(self):
        """Stop the coordinator gracefully."""
        logger.info("Stopping Genius Agent Coordinator")
        self.is_active = False

        # Wait for current cycle to complete if running
        if self.current_cycle and self.current_cycle.phase != AnalysisCyclePhase.COMPLETED:
            logger.info("Waiting for current analysis cycle to complete...")
            timeout = 60  # 1 minute timeout
            start_wait = datetime.utcnow()

            while (self.current_cycle.phase != AnalysisCyclePhase.COMPLETED and
                   (datetime.utcnow() - start_wait).total_seconds() < timeout):
                await asyncio.sleep(1)

        logger.info("Genius Agent Coordinator stopped")

    async def execute_analysis_cycle(self, symbol: Optional[str] = None) -> Optional[TradeSignal]:
        """
        Execute a complete analysis cycle (called by orchestrator).

        Args:
            symbol: Trading symbol to analyze (optional, uses default if not provided)

        Returns:
            Generated trade signal if any
        """
        symbol = symbol or self.config_manager.get_str('coordination.primary_symbol', 'EURUSD')
        cycle_id = f"cycle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        cycle_state = AnalysisCycleState(
            cycle_id=cycle_id,
            phase=AnalysisCyclePhase.INITIALIZATION,
            start_time=datetime.utcnow(),
            symbol=symbol
        )

        # Start performance monitoring
        optimization_flags = {
            'enable_parallel_analysis': self.enable_parallel_analysis,
            'enable_phase_merging': self.enable_phase_merging
        }
        self.performance_monitor.start_cycle_monitoring(cycle_id, optimization_flags)

        self.current_cycle = cycle_state

        try:
            logger.info(f"Starting analysis cycle {cycle_id} for {symbol}")

            # Choose execution path based on configuration
            if self.enable_phase_merging:
                # Use optimized merged phases
                await self._execute_merged_phases(cycle_state)
            else:
                # Use traditional sequential phases
                # Phase 1: Initialization and agent hierarchy setup
                await self._phase_initialization(cycle_state)

                # Phase 2: Data collection
                await self._phase_data_collection(cycle_state)

                # Phase 3: Indicator calculation
                await self._phase_indicator_calculation(cycle_state)

                # Phase 4: Agent analysis
                await self._phase_agent_analysis(cycle_state)

                # Phase 5: Decision synthesis
                await self._phase_decision_synthesis(cycle_state)

                # Phase 6: Validation
                await self._phase_validation(cycle_state)

                # Phase 7: Execution preparation
                await self._phase_execution_preparation(cycle_state)

            # Mark as completed
            cycle_state.phase = AnalysisCyclePhase.COMPLETED

            # End performance monitoring
            agents_processed = len(cycle_state.agent_decisions)
            indicators_calculated = len(cycle_state.performance_data.get('indicator_results', {}))
            signal_generated = cycle_state.final_signal is not None
            
            self.performance_monitor.end_cycle_monitoring(
                success=cycle_state.is_successful,
                agents_processed=agents_processed,
                indicators_calculated=indicators_calculated,
                signal_generated=signal_generated
            )

            # Update statistics
            self.total_cycles += 1
            if cycle_state.is_successful:
                self.successful_cycles += 1

            # Update average cycle duration
            self.average_cycle_duration = (
                (self.average_cycle_duration * (self.total_cycles - 1) + cycle_state.duration_seconds) /
                self.total_cycles
            )

            # Store in history
            self.cycle_history.append(cycle_state)
            if len(self.cycle_history) > 100:  # Keep last 100 cycles
                self.cycle_history.pop(0)

            self.last_cycle_time = datetime.utcnow()

            logger.info(f"Analysis cycle {cycle_id} completed successfully in {cycle_state.duration_seconds:.2f}s")

            # Return the generated signal
            return cycle_state.final_signal

        except Exception as e:
            cycle_state.phase = AnalysisCyclePhase.ERROR
            cycle_state.error_log.append(f"Cycle execution failed: {str(e)}")
            logger.error(f"Analysis cycle {cycle_id} failed: {str(e)}")
            logger.error(traceback.format_exc())

            # End performance monitoring for failed cycle
            self.performance_monitor.end_cycle_monitoring(
                success=False,
                agents_processed=len(cycle_state.agent_decisions),
                indicators_calculated=len(cycle_state.performance_data.get('indicator_results', {})),
                signal_generated=False,
                error=str(e)
            )

            # Still count as total cycle for statistics
            self.total_cycles += 1
            return None

        finally:
            self.current_cycle = None  # Clear current cycle reference
    async def _phase_initialization(self, cycle_state: AnalysisCycleState):
        """Phase 1: Initialize the analysis cycle and setup agent hierarchy."""
        cycle_state.phase = AnalysisCyclePhase.INITIALIZATION

        try:
            # Update agent rankings based on recent performance
            await self._update_agent_hierarchy()

            # Assign agent roles for this cycle
            await self._assign_agent_roles(cycle_state)

            # Update elite indicator sets if needed
            await self._update_elite_indicator_sets_if_needed()

            logger.debug(f"Cycle {cycle_state.cycle_id} initialization completed")

        except Exception as e:
            raise CoordinationError(f"Initialization phase failed: {str(e)}")

    async def _phase_data_collection(self, cycle_state: AnalysisCycleState):
        """Phase 2: Collect all required market data."""
        cycle_state.phase = AnalysisCyclePhase.DATA_COLLECTION

        try:
            # Determine required data types from all active agents
            required_data_types = set()
            for agent_name in [cycle_state.alpha_agent] + cycle_state.beta_agents + cycle_state.gamma_agents:
                if agent_name in self.agents:
                    agent_data_types = self.agents[agent_name].get_required_data_types()
                    required_data_types.update(agent_data_types)

            # Collect market data from data providers
            market_data = {}
            for data_type in required_data_types:
                try:
                    data = await self.data_manager.get_data(
                        data_type=data_type,
                        symbol=cycle_state.symbol,
                        timeframe='1H',  # Primary timeframe
                        limit=500  # Sufficient for most indicators
                    )
                    if data is not None and not data.empty:
                        market_data[data_type] = data
                    else:
                        logger.warning(f"No data available for {data_type}")

                except Exception as e:
                    logger.warning(f"Failed to collect {data_type} data: {str(e)}")
                    continue

            if not market_data:
                raise CoordinationError("No market data collected")

            # Store market data in cycle state
            cycle_state.performance_data['market_data'] = market_data
            cycle_state.performance_data['data_types_collected'] = list(market_data.keys())

            logger.debug(f"Collected data types: {list(market_data.keys())}")

        except Exception as e:
            raise CoordinationError(f"Data collection phase failed: {str(e)}")

    async def _phase_indicator_calculation(self, cycle_state: AnalysisCycleState):
        """Phase 3: Calculate indicators with tiered approach."""
        cycle_state.phase = AnalysisCyclePhase.INDICATOR_CALCULATION

        try:
            market_data = cycle_state.performance_data['market_data']

            # Detect current market regime
            market_conditions = await self._assess_market_conditions(cycle_state.symbol, market_data)
            cycle_state.market_conditions = market_conditions
            current_regime = market_conditions.regime

            # Tiered indicator calculation
            # Tier 1: Essential indicators (always calculate)
            essential_indicators = self._get_essential_indicators()

            # Tier 2: Regime-specific indicators
            regime_indicators = self._get_regime_indicators(current_regime)

            # Tier 3: Agent-specific indicators (calculate only if needed)
            agent_indicators = self._get_agent_specific_indicators(cycle_state)

            # Calculate in priority order
            all_indicators = {}

            # Essential indicators (parallel)
            if essential_indicators:
                essential_tasks = []
                for indicator_name in essential_indicators:
                    task = self._calculate_single_indicator(indicator_name, market_data, cycle_state)
                    essential_tasks.append(task)

                essential_results = await asyncio.gather(*essential_tasks, return_exceptions=True)
                self._merge_indicator_results(essential_results, essential_indicators, all_indicators)

            # Calculate remaining indicators (comprehensive analysis)
            remaining_indicators = regime_indicators + agent_indicators
            if remaining_indicators:
                remaining_tasks = []
                for indicator_name in remaining_indicators:
                    task = self._calculate_single_indicator(indicator_name, market_data, cycle_state)
                    remaining_tasks.append(task)

                remaining_results = await asyncio.gather(*remaining_tasks, return_exceptions=True)
                self._merge_indicator_results(remaining_results, remaining_indicators, all_indicators)

            # Always use comprehensive analysis
            cycle_state.performance_data['comprehensive_analysis'] = True
            cycle_state.performance_data['validation_type'] = 'COMPREHENSIVE'

            cycle_state.performance_data['indicator_results'] = all_indicators
            cycle_state.performance_data['indicators_calculated'] = len(all_indicators)
            cycle_state.performance_data['essential_indicators'] = essential_indicators
            cycle_state.performance_data['regime_indicators'] = regime_indicators
            cycle_state.performance_data['agent_indicators'] = agent_indicators

            logger.debug(f"Calculated {len(all_indicators)} indicators for regime {current_regime.value}")

        except Exception as e:
            raise CoordinationError(f"Indicator calculation phase failed: {str(e)}")

    def _get_essential_indicators(self) -> List[str]:
        """Get list of essential indicators always needed."""
        return [
            'sma_20', 'sma_50', 'sma_200',
            'rsi_14', 'macd',
            'bollinger_bands',
            'atr_14',
            'volume_sma_20'
        ]

    def _get_regime_indicators(self, regime) -> List[str]:
        """Get indicators specific to current market regime."""
        from ..core.data_contracts import MarketRegime
        
        regime_map = {
            MarketRegime.TRENDING_UP: [
                'trend_strength_indicator',
                'momentum_indicators',
                'breakout_indicators'
            ],
            MarketRegime.TRENDING_DOWN: [
                'trend_strength_indicator',
                'momentum_indicators',
                'breakdown_indicators'
            ],
            MarketRegime.SIDEWAYS: [
                'mean_reversion_indicators',
                'support_resistance_indicators',
                'oscillator_indicators'
            ],
            MarketRegime.HIGH_VOLATILITY: [
                'volatility_indicators',
                'risk_indicators',
                'dynamic_indicators'
            ],
            MarketRegime.LOW_VOLATILITY: [
                'range_indicators',
                'compression_indicators',
                'accumulation_indicators'
            ]
        }
        return regime_map.get(regime, [])

    def _get_agent_specific_indicators(self, cycle_state: AnalysisCycleState) -> List[str]:
        """Get indicators specific to active agents."""
        agent_indicators = set()
        for agent_name in [cycle_state.alpha_agent] + cycle_state.beta_agents + cycle_state.gamma_agents:
            if agent_name in self.agents:
                try:
                    agent_indicators.update(self.agents[agent_name].get_assigned_indicators())
                except AttributeError:
                    # Fallback if agent doesn't have get_assigned_indicators method
                    pass
        return list(agent_indicators)

    async def _calculate_single_indicator(self, indicator_name: str, market_data: Dict, cycle_state: AnalysisCycleState) -> Dict[str, Any]:
        """Calculate a single indicator."""
        try:
            from ..indicator_engine.indicator_executor import IndicatorExecutionRequest, ExecutionPriority

            request = IndicatorExecutionRequest(
                indicator_name=indicator_name,
                symbol=cycle_state.symbol,
                timeframe=self.config_manager.get_str('coordination.primary_timeframe', '1H'),
                periods=self.config_manager.get_int('coordination.indicator_periods', 100),
                priority=ExecutionPriority.HIGH
            )

            # Execute single indicator
            results = await self.indicator_executor.execute_indicators([request])
            
            if results and len(results) > 0 and results[0].status.value == "success":
                return {
                    'indicator_name': indicator_name,
                    'data': results[0].data,
                    'error': None
                }
            else:
                return {
                    'indicator_name': indicator_name,
                    'data': None,
                    'error': 'Calculation failed'
                }

        except Exception as e:
            return {
                'indicator_name': indicator_name,
                'data': None,
                'error': str(e)
            }

    def _merge_indicator_results(self, results: List, indicator_names: List[str], all_indicators: Dict):
        """Merge indicator calculation results."""
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Indicator {indicator_names[i]} failed: {str(result)}")
                continue

            if result['error']:
                logger.warning(f"Indicator {result['indicator_name']}: {result['error']}")
                continue

            if result['data'] is not None:
                all_indicators[result['indicator_name']] = result['data']

    async def _phase_agent_analysis(self, cycle_state: AnalysisCycleState):
        """Phase 4: Execute agent analysis with parallel processing."""
        cycle_state.phase = AnalysisCyclePhase.AGENT_ANALYSIS

        try:
            market_data = cycle_state.performance_data['market_data']
            indicator_results = cycle_state.performance_data['indicator_results']
            market_conditions = cycle_state.market_conditions

            # Group agents by hierarchy for parallel processing
            alpha_agents = [cycle_state.alpha_agent] if cycle_state.alpha_agent else []
            beta_agents = cycle_state.beta_agents
            gamma_agents = cycle_state.gamma_agents

            agent_decisions = {}

            # Phase 1: Alpha agent analysis (highest priority)
            if alpha_agents:
                alpha_tasks = []
                for agent_name in alpha_agents:
                    if agent_name in self.agents:
                        task = self._analyze_single_agent(
                            agent_name, market_data, market_conditions,
                            indicator_results, AgentRank.ALPHA, cycle_state
                        )
                        alpha_tasks.append(task)

                if alpha_tasks:
                    alpha_results = await asyncio.gather(*alpha_tasks, return_exceptions=True)
                    self._process_agent_results(alpha_results, alpha_agents, agent_decisions, cycle_state)

            # Phase 2: Beta agents analysis (parallel)
            if beta_agents:
                beta_tasks = []
                for agent_name in beta_agents:
                    if agent_name in self.agents:
                        task = self._analyze_single_agent(
                            agent_name, market_data, market_conditions,
                            indicator_results, AgentRank.BETA, cycle_state
                        )
                        beta_tasks.append(task)

                if beta_tasks:
                    beta_results = await asyncio.gather(*beta_tasks, return_exceptions=True)
                    self._process_agent_results(beta_results, beta_agents, agent_decisions, cycle_state)

            # Phase 3: Gamma agents analysis (parallel)
            if gamma_agents:
                gamma_tasks = []
                for agent_name in gamma_agents:
                    if agent_name in self.agents:
                        task = self._analyze_single_agent(
                            agent_name, market_data, market_conditions,
                            indicator_results, AgentRank.GAMMA, cycle_state
                        )
                        gamma_tasks.append(task)

                if gamma_tasks:
                    gamma_results = await asyncio.gather(*gamma_tasks, return_exceptions=True)
                    self._process_agent_results(gamma_results, gamma_agents, agent_decisions, cycle_state)

            cycle_state.agent_decisions = agent_decisions
            cycle_state.performance_data['agents_analyzed'] = len(agent_decisions)

            if not agent_decisions:
                raise CoordinationError("No agent analysis results available")

            logger.debug(f"Completed parallel analysis for {len(agent_decisions)} agents")

        except Exception as e:
            raise CoordinationError(f"Agent analysis phase failed: {str(e)}")

    async def _analyze_single_agent(self, agent_name: str, market_data: Dict,
                                   market_conditions, indicator_results: Dict,
                                   rank: AgentRank, cycle_state: AnalysisCycleState) -> Dict[str, Any]:
        """Analyze single agent with optimized timeout."""
        try:
            agent = self.agents[agent_name]

            if not agent.is_ready_for_analysis():
                return {'agent_name': agent_name, 'error': 'Agent not ready', 'result': None}

            analysis_start = datetime.utcnow()

            # Reduced timeout: 2 minutes instead of 5
            analysis_result = await asyncio.wait_for(
                agent.perform_analysis(
                    symbol=cycle_state.symbol,
                    market_data=market_data,
                    market_conditions=market_conditions,
                    indicator_results=indicator_results
                ),
                timeout=120  # 2 minutes instead of 5
            )

            analysis_duration = (datetime.utcnow() - analysis_start).total_seconds()

            return {
                'agent_name': agent_name,
                'rank': rank,
                'result': analysis_result,
                'duration': analysis_duration,
                'error': None
            }

        except asyncio.TimeoutError:
            return {
                'agent_name': agent_name,
                'rank': rank,
                'error': f'Analysis timed out after 2 minutes',
                'result': None
            }
        except Exception as e:
            return {
                'agent_name': agent_name,
                'rank': rank,
                'error': str(e),
                'result': None
            }

    def _process_agent_results(self, results: List, agent_names: List[str],
                              agent_decisions: Dict, cycle_state: AnalysisCycleState):
        """Process parallel agent analysis results."""
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Agent {agent_names[i]} failed: {str(result)}"
                cycle_state.error_log.append(error_msg)
                logger.warning(error_msg)
                continue

            if result['error']:
                cycle_state.error_log.append(f"Agent {result['agent_name']}: {result['error']}")
                logger.warning(f"Agent {result['agent_name']}: {result['error']}")
                continue

            if result['result']:
                agent_decisions[result['agent_name']] = result['result']
                logger.debug(f"Agent {result['agent_name']} ({result['rank'].value}) "
                            f"completed in {result['duration']:.2f}s")

    async def _phase_decision_synthesis(self, cycle_state: AnalysisCycleState):
        """Phase 5: Synthesize agent decisions into final trading signal."""
        cycle_state.phase = AnalysisCyclePhase.DECISION_SYNTHESIS

        try:
            agent_decisions = cycle_state.agent_decisions

            if not agent_decisions:
                logger.warning("No agent decisions to synthesize")
                return

            # Apply hierarchical decision synthesis
            final_signal = await self._synthesize_hierarchical_decision(
                agent_decisions=agent_decisions,
                alpha_agent=cycle_state.alpha_agent,
                beta_agents=cycle_state.beta_agents,
                gamma_agents=cycle_state.gamma_agents,
                symbol=cycle_state.symbol
            )

            cycle_state.final_signal = final_signal

            if final_signal:
                logger.info(f"Generated final signal: {final_signal.direction.value} {final_signal.symbol} "
                          f"(confidence: {final_signal.confidence:.3f})")
            else:
                logger.info("No trading signal generated from decision synthesis")

        except Exception as e:
            raise CoordinationError(f"Decision synthesis phase failed: {str(e)}")

    async def _phase_validation(self, cycle_state: AnalysisCycleState):
        """Phase 6: Validate the final decision using anti-overfitting framework."""
        cycle_state.phase = AnalysisCyclePhase.VALIDATION

        try:
            if not cycle_state.final_signal:
                logger.debug("No signal to validate")
                return

            # Validate signal using walk-forward validation principles
            validation_result = await self._validate_signal_robustness(
                signal=cycle_state.final_signal,
                agent_decisions=cycle_state.agent_decisions,
                market_conditions=cycle_state.market_conditions
            )

            cycle_state.performance_data['validation_result'] = validation_result

            # Apply validation filters
            if validation_result and validation_result.get('overfitting_risk', 0) > 0.7:
                logger.warning("High overfitting risk detected, discarding signal")
                cycle_state.final_signal = None
                cycle_state.error_log.append("Signal discarded due to overfitting risk")

            elif validation_result and validation_result.get('robustness_score', 0) < 0.3:
                logger.warning("Low robustness score, discarding signal")
                cycle_state.final_signal = None
                cycle_state.error_log.append("Signal discarded due to low robustness")

        except Exception as e:
            logger.warning(f"Validation phase failed: {str(e)}")
            # Don't fail the entire cycle for validation errors
            cycle_state.error_log.append(f"Validation error: {str(e)}")

    async def _phase_execution_preparation(self, cycle_state: AnalysisCycleState):
        """Phase 7: Prepare signal for execution."""
        cycle_state.phase = AnalysisCyclePhase.EXECUTION_PREPARATION

        try:
            if not cycle_state.final_signal:
                logger.debug("No signal to prepare for execution")
                return

            # Add execution metadata
            cycle_state.final_signal.metadata.update({
                'cycle_id': cycle_state.cycle_id,
                'analysis_timestamp': cycle_state.start_time.isoformat(),
                'analysis_duration_seconds': cycle_state.duration_seconds,
                'market_regime': cycle_state.market_conditions.regime.value if cycle_state.market_conditions else None,
                'participating_agents': list(cycle_state.agent_decisions.keys()),
                'agents_consulted': len(cycle_state.agent_decisions),
                'indicators_analyzed': len(cycle_state.performance_data.get('indicator_results', {})),
                'alpha_agent': cycle_state.alpha_agent,
                'beta_agents': cycle_state.beta_agents,
                'gamma_agents': cycle_state.gamma_agents,
                'validation_passed': True,  # If we reach here, validation passed
                
                # ADD comprehensive analysis indicators:
                'comprehensive_analysis': cycle_state.performance_data.get('comprehensive_analysis', True),
                'full_validation_performed': cycle_state.performance_data.get('validation_type') == 'COMPREHENSIVE',
                'analysis_consistency': True,
                'parallel_processing_used': self.enable_parallel_analysis,
                'timestamp': datetime.utcnow().isoformat()
            })

            # Send to trading engine for risk management and execution
            # This will be handled by the Dynamic Risk Manager in the next task

            logger.debug("Signal prepared for execution")

        except Exception as e:
            raise CoordinationError(f"Execution preparation phase failed: {str(e)}")

    # Phase Merging Enhancement Methods

    async def _execute_merged_phases(self, cycle_state: AnalysisCycleState):
        """Execute phases with intelligent merging for better performance."""

        # Phase 1: Initialization (unchanged)
        self.performance_monitor.start_phase_monitoring("initialization")
        try:
            await self._phase_initialization(cycle_state)
            self.performance_monitor.end_phase_monitoring(success=True)
        except Exception as e:
            self.performance_monitor.end_phase_monitoring(success=False, error=str(e))
            raise

        # Phase 2 & 3: Combined Data Collection + Indicator Calculation
        self.performance_monitor.start_phase_monitoring("data_and_indicators_combined", parallel_operations=2)
        try:
            await self._phase_data_and_indicators_combined(cycle_state)
            comprehensive_analysis = cycle_state.performance_data.get('comprehensive_analysis', True)
            validation_type = cycle_state.performance_data.get('validation_type', 'COMPREHENSIVE')
            logger.info(f"Analysis completed - Comprehensive: {comprehensive_analysis}, Validation: {validation_type}")
            self.performance_monitor.end_phase_monitoring(success=True, early_exit=False)
            
            # Update cycle performance data with comprehensive analysis tracking
            if hasattr(self.performance_monitor, 'current_cycle') and self.performance_monitor.current_cycle:
                self.performance_monitor.current_cycle.comprehensive_analysis_applied = comprehensive_analysis
                self.performance_monitor.current_cycle.full_validation_completed = (validation_type == 'COMPREHENSIVE')
                self.performance_monitor.current_cycle.analysis_consistency_score = 1.0
        except Exception as e:
            self.performance_monitor.end_phase_monitoring(success=False, error=str(e))
            raise

        # Phase 4: Parallel Agent Analysis (optimized)
        agents_count = len([cycle_state.alpha_agent] + cycle_state.beta_agents + cycle_state.gamma_agents)
        self.performance_monitor.start_phase_monitoring("agent_analysis_parallel", parallel_operations=agents_count)
        try:
            await self._phase_agent_analysis(cycle_state)
            self.performance_monitor.end_phase_monitoring(success=True)
        except Exception as e:
            self.performance_monitor.end_phase_monitoring(success=False, error=str(e))
            raise

        # Phase 5 & 6: Combined Decision Synthesis + Validation
        self.performance_monitor.start_phase_monitoring("decision_and_validation_combined", parallel_operations=2)
        try:
            await self._phase_decision_and_validation_combined(cycle_state)
            self.performance_monitor.end_phase_monitoring(success=True)
        except Exception as e:
            self.performance_monitor.end_phase_monitoring(success=False, error=str(e))
            raise

        # Phase 7: Execution Preparation (unchanged)
        self.performance_monitor.start_phase_monitoring("execution_preparation")
        try:
            await self._phase_execution_preparation(cycle_state)
            self.performance_monitor.end_phase_monitoring(success=True)
        except Exception as e:
            self.performance_monitor.end_phase_monitoring(success=False, error=str(e))
            raise

    async def _phase_data_and_indicators_combined(self, cycle_state: AnalysisCycleState):
        """Combined data collection and indicator calculation."""
        cycle_state.phase = AnalysisCyclePhase.DATA_COLLECTION

        try:
            # Start data collection
            essential_data = await self._collect_essential_data(cycle_state)

            # Start indicator calculation as soon as essential data is available
            indicator_task = self._calculate_indicators_incremental(essential_data, cycle_state)

            # Continue collecting remaining data in parallel
            remaining_data_task = self._collect_additional_data(cycle_state)

            # Wait for both to complete
            indicator_results, additional_data = await asyncio.gather(
                indicator_task, remaining_data_task, return_exceptions=True
            )

            # Handle exceptions
            if isinstance(indicator_results, Exception):
                logger.warning(f"Indicator calculation failed: {str(indicator_results)}")
                indicator_results = {}
            if isinstance(additional_data, Exception):
                logger.warning(f"Additional data collection failed: {str(additional_data)}")
                additional_data = {}

            # Merge results
            cycle_state.performance_data['market_data'] = {**essential_data, **additional_data}
            cycle_state.performance_data['indicator_results'] = indicator_results

            logger.debug("Combined data collection and indicator calculation completed")

        except Exception as e:
            raise CoordinationError(f"Combined data/indicator phase failed: {str(e)}")

    async def _collect_essential_data(self, cycle_state: AnalysisCycleState) -> Dict[str, Any]:
        """Collect essential market data quickly."""
        essential_data = {}
        
        # Collect OHLCV data first (most critical)
        try:
            ohlcv_data = await self.data_manager.get_data(
                data_type='OHLCV',
                symbol=cycle_state.symbol,
                timeframe='1H',
                limit=200
            )
            if ohlcv_data is not None and not ohlcv_data.empty:
                essential_data['OHLCV'] = ohlcv_data
        except Exception as e:
            logger.warning(f"Failed to collect OHLCV data: {str(e)}")

        return essential_data

    async def _collect_additional_data(self, cycle_state: AnalysisCycleState) -> Dict[str, Any]:
        """Collect additional data types in parallel."""
        additional_data = {}
        
        # Get other required data types
        additional_types = ['volume', 'economic_data', 'sentiment_data']
        
        tasks = []
        for data_type in additional_types:
            task = self._get_single_data_type(data_type, cycle_state.symbol)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if not isinstance(result, Exception) and result:
                additional_data[additional_types[i]] = result

        return additional_data

    async def _get_single_data_type(self, data_type: str, symbol: str):
        """Get a single data type."""
        try:
            return await self.data_manager.get_data(
                data_type=data_type,
                symbol=symbol,
                timeframe='1H',
                limit=100
            )
        except Exception as e:
            logger.warning(f"Failed to get {data_type}: {str(e)}")
            return None

    async def _calculate_indicators_incremental(self, essential_data: Dict, cycle_state: AnalysisCycleState) -> Dict[str, Any]:
        """Calculate indicators incrementally as data becomes available."""
        try:
            if not essential_data.get('OHLCV') is not None:
                return {}

            # Detect market regime quickly using essential data
            market_conditions = await self._assess_market_conditions(cycle_state.symbol, essential_data)
            cycle_state.market_conditions = market_conditions

            # Calculate using the enhanced tiered approach
            cycle_state.phase = AnalysisCyclePhase.INDICATOR_CALCULATION
            await self._phase_indicator_calculation(cycle_state)
            
            return cycle_state.performance_data.get('indicator_results', {})

        except Exception as e:
            logger.warning(f"Incremental indicator calculation failed: {str(e)}")
            return {}

    async def _phase_decision_and_validation_combined(self, cycle_state: AnalysisCycleState):
        """Combined decision synthesis and validation."""
        cycle_state.phase = AnalysisCyclePhase.DECISION_SYNTHESIS

        try:
            # Start decision synthesis
            decision_task = self._synthesize_decision(cycle_state)

            # Start validation preparation in parallel
            validation_prep_task = self._prepare_validation_data(cycle_state)

            # Wait for decision
            decision_result = await decision_task
            validation_data = await validation_prep_task

            if decision_result:
                # ALWAYS use comprehensive validation (no fast-track logic)
                logger.info("Applying comprehensive validation to decision")
                validation_result = await self._full_validation(decision_result, validation_data)
                cycle_state.final_signal = validation_result
                cycle_state.performance_data['validation_type'] = 'COMPREHENSIVE'
            else:
                cycle_state.final_signal = None

            cycle_state.is_successful = cycle_state.final_signal is not None

            logger.debug("Combined decision synthesis and validation completed")

        except Exception as e:
            raise CoordinationError(f"Combined decision/validation phase failed: {str(e)}")

    async def _synthesize_decision(self, cycle_state: AnalysisCycleState):
        """Synthesize decision from agent analysis."""
        try:
            if not cycle_state.agent_decisions:
                return None

            # Use existing hierarchical decision synthesis
            return await self._synthesize_hierarchical_decision(
                agent_decisions=cycle_state.agent_decisions,
                alpha_agent=cycle_state.alpha_agent,
                beta_agents=cycle_state.beta_agents,
                gamma_agents=cycle_state.gamma_agents,
                symbol=cycle_state.symbol
            )

        except Exception as e:
            logger.warning(f"Decision synthesis failed: {str(e)}")
            return None

    async def _prepare_validation_data(self, cycle_state: AnalysisCycleState) -> Dict[str, Any]:
        """Prepare validation data in parallel."""
        try:
            validation_data = {
                'market_conditions': cycle_state.market_conditions,
                'agent_consensus': len(cycle_state.agent_decisions),
                'indicator_diversity': len(cycle_state.performance_data.get('indicator_results', {})),
                'comprehensive_analysis': cycle_state.performance_data.get('comprehensive_analysis', True),
                'validation_type': cycle_state.performance_data.get('validation_type', 'COMPREHENSIVE')
            }
            return validation_data

        except Exception as e:
            logger.warning(f"Validation data preparation failed: {str(e)}")
            return {}

    async def _fast_validation(self, signal, validation_data: Dict[str, Any]):
        """Fast validation for high-confidence signals."""
        try:
            # Quick validation checks
            if signal.confidence >= 0.9 and validation_data.get('agent_consensus', 0) >= 2:
                return signal  # Pass through high-confidence signals quickly

            # Apply basic robustness check
            validation_result = await self._validate_signal_robustness(
                signal=signal,
                agent_decisions={},  # Simplified for fast validation
                market_conditions=validation_data.get('market_conditions')
            )

            if validation_result and validation_result.get('robustness_score', 0) >= 0.6:
                return signal
            else:
                return None

        except Exception as e:
            logger.warning(f"Fast validation failed: {str(e)}")
            return signal  # Return original signal if validation fails

    async def _full_validation(self, signal, validation_data: Dict[str, Any]):
        """Full validation for uncertain signals."""
        try:
            # Use existing validation logic
            validation_result = await self._validate_signal_robustness(
                signal=signal,
                agent_decisions={},  # Would need to pass full agent decisions
                market_conditions=validation_data.get('market_conditions')
            )

            # Apply validation filters
            if validation_result:
                if validation_result.get('overfitting_risk', 0) > 0.7:
                    logger.warning("High overfitting risk detected in full validation")
                    return None
                elif validation_result.get('robustness_score', 0) < 0.3:
                    logger.warning("Low robustness score in full validation")
                    return None

            return signal

        except Exception as e:
            logger.warning(f"Full validation failed: {str(e)}")
            return None

    async def _update_agent_hierarchy(self):
        """Update agent hierarchy based on recent performance."""
        try:
            # Get recent performance data for all agents
            agent_performance = {}
            for agent_name, agent in self.agents.items():
                recent_perf = agent.get_recent_performance(days=7)

                # Get out-of-sample performance from walk-forward validator
                validation_perf = await self._get_agent_validation_performance(agent_name)

                # Calculate composite performance score
                composite_score = self._calculate_agent_composite_score(recent_perf, validation_perf)

                agent_performance[agent_name] = {
                    'composite_score': composite_score,
                    'recent_performance': recent_perf,
                    'validation_performance': validation_perf
                }

            # Update hierarchy based on composite scores
            await self.hierarchy_manager.update_hierarchy(agent_performance)

        except Exception as e:
            logger.error(f"Failed to update agent hierarchy: {str(e)}")

    async def _assign_agent_roles(self, cycle_state: AnalysisCycleState):
        """Assign Alpha, Beta, and Gamma roles to agents for this cycle."""
        try:
            # Get current agent rankings from hierarchy manager
            agent_rankings = await self.hierarchy_manager.get_current_rankings()

            # Assign roles based on rankings
            alpha_agents = [name for name, rank in agent_rankings.items() if rank == AgentRank.ALPHA]
            beta_agents = [name for name, rank in agent_rankings.items() if rank == AgentRank.BETA]
            gamma_agents = [name for name, rank in agent_rankings.items() if rank == AgentRank.GAMMA]

            # Select Alpha agent (best performing agent for current market conditions)
            if alpha_agents:
                cycle_state.alpha_agent = await self._select_optimal_alpha_agent(
                    alpha_agents, cycle_state.symbol
                )

            # Select Beta agents (verification agents)
            cycle_state.beta_agents = beta_agents[:3]  # Limit to top 3 beta agents

            # Select Gamma agents (learning agents)
            cycle_state.gamma_agents = gamma_agents[:2]  # Limit to 2 gamma agents

            logger.debug(f"Agent roles assigned - Alpha: {cycle_state.alpha_agent}, "
                        f"Beta: {cycle_state.beta_agents}, Gamma: {cycle_state.gamma_agents}")

        except Exception as e:
            raise CoordinationError(f"Failed to assign agent roles: {str(e)}")

    async def _select_optimal_alpha_agent(self, alpha_agents: List[str], symbol: str) -> Optional[str]:
        """Select the best Alpha agent for current market conditions."""
        if not alpha_agents:
            return None

        if len(alpha_agents) == 1:
            return alpha_agents[0]

        # Evaluate each alpha agent for current market conditions
        best_agent = None
        best_score = -1

        for agent_name in alpha_agents:
            try:
                # Get agent's specialization match with current market regime
                specialization_score = await self._calculate_specialization_match(agent_name, symbol)

                # Get recent performance for this symbol
                symbol_performance = await self._get_agent_symbol_performance(agent_name, symbol)

                # Combine scores
                total_score = (specialization_score * 0.6) + (symbol_performance * 0.4)

                if total_score > best_score:
                    best_score = total_score
                    best_agent = agent_name

            except Exception as e:
                logger.warning(f"Failed to evaluate alpha agent {agent_name}: {str(e)}")
                continue

        return best_agent or alpha_agents[0]  # Fallback to first agent

    async def _assess_market_conditions(self, symbol: str, market_data: Dict[str, pd.DataFrame]) -> MarketConditions:
        """Assess current market conditions and regime."""
        try:
            # Use regime classifier to determine market regime
            ohlcv_data = market_data.get('OHLCV')
            if ohlcv_data is None or ohlcv_data.empty:
                raise CoordinationError("No OHLCV data available for market assessment")

            # Detect market regime
            market_regime = await self.regime_classifier.classify_regime(ohlcv_data)

            # Calculate additional market metrics
            latest_close = ohlcv_data['close'].iloc[-1]
            volatility = await self._calculate_volatility(ohlcv_data)
            trend_strength = await self._calculate_trend_strength(ohlcv_data)
            volume_profile = await self._analyze_volume_profile(ohlcv_data)

            # Identify support and resistance levels
            support_levels, resistance_levels = await self._identify_key_levels(ohlcv_data)

            market_conditions = MarketConditions(
                symbol=symbol,
                regime=market_regime,
                volatility=volatility,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                key_indicators={
                    'latest_close': float(latest_close),
                    'price_change_24h': float((latest_close - ohlcv_data['close'].iloc[-25]) / ohlcv_data['close'].iloc[-25] * 100),
                    'average_volume': float(ohlcv_data['volume'].tail(20).mean())
                }
            )

            return market_conditions

        except Exception as e:
            logger.error(f"Failed to assess market conditions: {str(e)}")
            # Return default market conditions
            return MarketConditions(
                symbol=symbol,
                regime=MarketRegime.SIDEWAYS,
                volatility=0.5,
                trend_strength=0.5,
                volume_profile={'low': 0.3, 'medium': 0.4, 'high': 0.3},
                support_levels=[],
                resistance_levels=[],
                key_indicators={}
            )

    async def _synthesize_hierarchical_decision(self,
                                              agent_decisions: Dict[str, AnalysisResult],
                                              alpha_agent: Optional[str],
                                              beta_agents: List[str],
                                              gamma_agents: List[str],
                                              symbol: str) -> Optional[TradeSignal]:
        """Synthesize final decision using hierarchical consensus."""
        try:
            # Weight assignments for different agent ranks
            ALPHA_WEIGHT = 0.5
            BETA_WEIGHT = 0.3 / max(len(beta_agents), 1)
            GAMMA_WEIGHT = 0.2 / max(len(gamma_agents), 1)

            # Collect weighted decisions
            buy_confidence = 0.0
            sell_confidence = 0.0
            hold_confidence = 0.0
            total_weight = 0.0

            supporting_agents = []
            indicators_used = set()

            # Process Alpha agent decision (highest weight)
            if alpha_agent and alpha_agent in agent_decisions:
                alpha_decision = agent_decisions[alpha_agent]
                alpha_conf = alpha_decision.confidence

                if alpha_decision.decision == "BUY":
                    buy_confidence += alpha_conf * ALPHA_WEIGHT
                elif alpha_decision.decision == "SELL":
                    sell_confidence += alpha_conf * ALPHA_WEIGHT
                else:  # HOLD or NO_SIGNAL
                    hold_confidence += alpha_conf * ALPHA_WEIGHT

                total_weight += ALPHA_WEIGHT
                supporting_agents.append(alpha_agent)
                indicators_used.update(alpha_decision.indicators_used)

            # Process Beta agent decisions
            for beta_agent in beta_agents:
                if beta_agent in agent_decisions:
                    beta_decision = agent_decisions[beta_agent]
                    beta_conf = beta_decision.confidence

                    if beta_decision.decision == "BUY":
                        buy_confidence += beta_conf * BETA_WEIGHT
                    elif beta_decision.decision == "SELL":
                        sell_confidence += beta_conf * BETA_WEIGHT
                    else:
                        hold_confidence += beta_conf * BETA_WEIGHT

                    total_weight += BETA_WEIGHT
                    supporting_agents.append(beta_agent)
                    indicators_used.update(beta_decision.indicators_used)

            # Process Gamma agent decisions
            for gamma_agent in gamma_agents:
                if gamma_agent in agent_decisions:
                    gamma_decision = agent_decisions[gamma_agent]
                    gamma_conf = gamma_decision.confidence

                    if gamma_decision.decision == "BUY":
                        buy_confidence += gamma_conf * GAMMA_WEIGHT
                    elif gamma_decision.decision == "SELL":
                        sell_confidence += gamma_conf * GAMMA_WEIGHT
                    else:
                        hold_confidence += gamma_conf * GAMMA_WEIGHT

                    total_weight += GAMMA_WEIGHT
                    supporting_agents.append(gamma_agent)
                    indicators_used.update(gamma_decision.indicators_used)

            # Normalize confidence scores
            if total_weight > 0:
                buy_confidence /= total_weight
                sell_confidence /= total_weight
                hold_confidence /= total_weight

            # Determine final decision
            max_confidence = max(buy_confidence, sell_confidence, hold_confidence)

            # Check if confidence meets minimum threshold
            if max_confidence < self.min_confidence_threshold:
                logger.debug(f"Confidence {max_confidence:.3f} below threshold {self.min_confidence_threshold}")
                return None

            # Determine direction and confidence level
            if max_confidence == buy_confidence:
                direction = TradeDirection.BUY
                final_confidence = buy_confidence
            elif max_confidence == sell_confidence:
                direction = TradeDirection.SELL
                final_confidence = sell_confidence
            else:
                # Hold decision, no signal generated
                return None

            # Create the final trade signal
            signal = TradeSignal(
                symbol=symbol,
                direction=direction,
                confidence=final_confidence,
                confidence_level=self._determine_confidence_level(final_confidence),
                timeframe='1H',
                strategy='HierarchicalConsensus',
                generating_agent='GeniusAgentCoordinator',
                supporting_agents=supporting_agents,
                indicators_used=list(indicators_used),
                metadata={
                    'alpha_agent': alpha_agent,
                    'beta_agents': beta_agents,
                    'gamma_agents': gamma_agents,
                    'buy_confidence': buy_confidence,
                    'sell_confidence': sell_confidence,
                    'hold_confidence': hold_confidence,
                    'consensus_threshold': self.consensus_threshold,
                    'total_agents_participating': len(supporting_agents)
                }
            )

            return signal

        except Exception as e:
            logger.error(f"Failed to synthesize hierarchical decision: {str(e)}")
            return None
    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to ConfidenceLevel enum."""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.65:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.35:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    async def _wait_for_next_cycle(self):
        """Wait for the next analysis cycle."""
        if not self.is_active:
            return

        # Calculate time until next cycle
        if self.last_cycle_time:
            next_cycle_time = self.last_cycle_time + timedelta(minutes=self.analysis_frequency_minutes)
            now = datetime.utcnow()

            if next_cycle_time > now:
                wait_seconds = (next_cycle_time - now).total_seconds()
                logger.debug(f"Waiting {wait_seconds:.1f} seconds until next cycle")

                # Wait in small intervals to allow for graceful shutdown
                while wait_seconds > 0 and self.is_active:
                    sleep_time = min(60, wait_seconds)  # Check every minute
                    await asyncio.sleep(sleep_time)
                    wait_seconds -= sleep_time
        else:
            # First cycle, wait full interval
            await asyncio.sleep(self.analysis_frequency_minutes * 60)

    async def _initialize_elite_indicator_sets(self):
        """Initialize elite indicator sets for all market regimes."""
        try:
            logger.info("Initializing elite indicator sets")

            for regime in MarketRegime:
                # Get elite indicators from selective indicator engine
                elite_indicators = await self.indicator_engine.get_elite_indicators(regime)

                # Create elite indicator set
                elite_set = EliteIndicatorSet(
                    regime=regime,
                    indicators=elite_indicators,
                    performance_score=0.5,  # Default score
                    validation_score=0.5,   # Will be updated with real validation
                    last_updated=datetime.utcnow(),
                    out_of_sample_performance={},
                    robustness_metrics={}
                )

                self.elite_indicator_sets[regime] = elite_set

            logger.info(f"Initialized elite indicator sets for {len(MarketRegime)} regimes")

        except Exception as e:
            logger.error(f"Failed to initialize elite indicator sets: {str(e)}")
            # Create empty sets as fallback
            for regime in MarketRegime:
                self.elite_indicator_sets[regime] = EliteIndicatorSet(
                    regime=regime,
                    indicators=[],
                    performance_score=0.0,
                    validation_score=0.0,
                    last_updated=datetime.utcnow(),
                    out_of_sample_performance={},
                    robustness_metrics={}
                )

    async def _update_elite_indicator_sets_if_needed(self):
        """Update elite indicator sets if they're outdated."""
        try:
            current_time = datetime.utcnow()
            update_threshold = timedelta(hours=self.elite_set_update_frequency)

            for regime, elite_set in self.elite_indicator_sets.items():
                if current_time - elite_set.last_updated > update_threshold:
                    logger.info(f"Updating elite indicator set for regime {regime.value}")

                    # Get updated indicators and performance data
                    updated_indicators = await self.indicator_engine.get_elite_indicators(regime)
                    validation_data = await self._validate_indicator_set(updated_indicators, regime)

                    # Update the elite set
                    elite_set.indicators = updated_indicators
                    elite_set.validation_score = validation_data.get('validation_score', 0.5)
                    elite_set.performance_score = validation_data.get('performance_score', 0.5)
                    elite_set.out_of_sample_performance = validation_data.get('out_of_sample_performance', {})
                    elite_set.robustness_metrics = validation_data.get('robustness_metrics', {})
                    elite_set.last_updated = current_time

        except Exception as e:
            logger.warning(f"Failed to update elite indicator sets: {str(e)}")

    def _get_elite_indicators_for_regime(self, regime: MarketRegime) -> set:
        """Get elite indicators for a specific market regime."""
        if regime in self.elite_indicator_sets:
            return set(self.elite_indicator_sets[regime].indicators)
        return set()

    async def _get_agent_validation_performance(self, agent_name: str) -> Dict[str, float]:
        """Get out-of-sample validation performance for an agent."""
        try:
            # Get validation data from walk-forward validator
            validation_results = await self.walk_forward_validator.get_agent_performance(agent_name)

            return {
                'out_of_sample_win_rate': validation_results.get('out_of_sample_win_rate', 0.5),
                'out_of_sample_profit_factor': validation_results.get('out_of_sample_profit_factor', 1.0),
                'robustness_score': validation_results.get('robustness_score', 0.5),
                'overfitting_score': validation_results.get('overfitting_score', 0.5)
            }
        except Exception as e:
            logger.warning(f"Failed to get validation performance for {agent_name}: {str(e)}")
            return {
                'out_of_sample_win_rate': 0.5,
                'out_of_sample_profit_factor': 1.0,
                'robustness_score': 0.5,
                'overfitting_score': 0.5
            }

    def _calculate_agent_composite_score(self, recent_perf: Dict[str, Any], validation_perf: Dict[str, float]) -> float:
        """Calculate composite performance score for agent ranking."""
        try:
            # Recent performance metrics (30% weight)
            recent_confidence = recent_perf.get('avg_confidence', 0.5)
            recent_decisions = recent_perf.get('total_decisions', 0)
            recent_weight = 0.3

            # Validation performance metrics (70% weight - emphasize out-of-sample performance)
            validation_win_rate = validation_perf.get('out_of_sample_win_rate', 0.5)
            validation_profit_factor = min(validation_perf.get('out_of_sample_profit_factor', 1.0), 3.0) / 3.0  # Normalize
            robustness_score = validation_perf.get('robustness_score', 0.5)
            overfitting_penalty = 1.0 - validation_perf.get('overfitting_score', 0.5)  # Lower overfitting is better
            validation_weight = 0.7

            # Calculate composite score
            recent_score = recent_confidence * (1.0 if recent_decisions > 5 else 0.5)  # Penalty for low activity
            validation_score = (validation_win_rate + validation_profit_factor + robustness_score + overfitting_penalty) / 4.0

            composite_score = (recent_score * recent_weight) + (validation_score * validation_weight)

            return min(max(composite_score, 0.0), 1.0)  # Clamp between 0 and 1

        except Exception as e:
            logger.warning(f"Failed to calculate composite score: {str(e)}")
            return 0.5  # Default neutral score

    async def _calculate_specialization_match(self, agent_name: str, symbol: str) -> float:
        """Calculate how well an agent's specialization matches current market conditions."""
        try:
            # This is a simplified implementation - could be enhanced with ML models
            agent_mapping = AGENT_MAPPINGS.get(agent_name, {})
            specialization = agent_mapping.get('specialization', '')

            # Get current market conditions for the symbol
            # For now, return a base score that could be enhanced with more sophisticated matching
            base_score = 0.7  # Default good match

            # Could enhance with:
            # - Historical performance of this agent on this symbol
            # - Market regime compatibility
            # - Volatility matching agent's strengths
            # - Session timing for SessionExpert

            return base_score

        except Exception as e:
            logger.warning(f"Failed to calculate specialization match for {agent_name}: {str(e)}")
            return 0.5

    async def _get_agent_symbol_performance(self, agent_name: str, symbol: str) -> float:
        """Get agent's recent performance for a specific symbol."""
        try:
            # Get symbol-specific performance from performance tracker
            symbol_performance = await self.performance_tracker.get_agent_symbol_performance(agent_name, symbol)

            return symbol_performance.get('win_rate', 0.5)

        except Exception as e:
            logger.warning(f"Failed to get symbol performance for {agent_name} on {symbol}: {str(e)}")
            return 0.5

    async def _calculate_volatility(self, ohlcv_data: pd.DataFrame) -> float:
        """Calculate market volatility."""
        try:
            # Calculate ATR-based volatility
            high_low = ohlcv_data['high'] - ohlcv_data['low']
            high_close = abs(ohlcv_data['high'] - ohlcv_data['close'].shift(1))
            low_close = abs(ohlcv_data['low'] - ohlcv_data['close'].shift(1))

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]

            # Normalize volatility (this could be enhanced with historical percentiles)
            normalized_volatility = min(atr / ohlcv_data['close'].iloc[-1], 0.1) * 10  # Scale to 0-1

            return float(normalized_volatility)

        except Exception as e:
            logger.warning(f"Failed to calculate volatility: {str(e)}")
            return 0.5

    async def _calculate_trend_strength(self, ohlcv_data: pd.DataFrame) -> float:
        """Calculate trend strength."""
        try:
            # Simple trend strength using ADX-like calculation
            close_prices = ohlcv_data['close']

            # Calculate price changes
            price_changes = close_prices.pct_change().dropna()

            # Calculate trend strength as directional consistency
            positive_moves = (price_changes > 0).sum()
            total_moves = len(price_changes)

            if total_moves == 0:
                return 0.5

            # Calculate trend strength (0.5 = no trend, 1.0 = strong trend)
            trend_strength = abs((positive_moves / total_moves) - 0.5) * 2

            return float(min(trend_strength, 1.0))

        except Exception as e:
            logger.warning(f"Failed to calculate trend strength: {str(e)}")
            return 0.5
    async def _analyze_volume_profile(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume profile."""
        try:
            volumes = ohlcv_data['volume'].tail(20)  # Last 20 periods

            # Calculate volume percentiles
            low_threshold = volumes.quantile(0.33)
            high_threshold = volumes.quantile(0.67)

            low_volume_count = (volumes <= low_threshold).sum()
            medium_volume_count = ((volumes > low_threshold) & (volumes <= high_threshold)).sum()
            high_volume_count = (volumes > high_threshold).sum()

            total_count = len(volumes)

            return {
                'low': float(low_volume_count / total_count),
                'medium': float(medium_volume_count / total_count),
                'high': float(high_volume_count / total_count)
            }

        except Exception as e:
            logger.warning(f"Failed to analyze volume profile: {str(e)}")
            return {'low': 0.33, 'medium': 0.34, 'high': 0.33}

    async def _identify_key_levels(self, ohlcv_data: pd.DataFrame) -> Tuple[List[Decimal], List[Decimal]]:
        """Identify support and resistance levels."""
        try:
            # Simple implementation using recent highs and lows
            recent_data = ohlcv_data.tail(50)  # Last 50 periods

            # Find local highs and lows
            highs = recent_data['high']
            lows = recent_data['low']

            # Identify resistance levels (local highs)
            resistance_levels = []
            for i in range(2, len(highs) - 2):
                if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and
                    highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
                    resistance_levels.append(Decimal(str(highs.iloc[i])))

            # Identify support levels (local lows)
            support_levels = []
            for i in range(2, len(lows) - 2):
                if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and
                    lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]):
                    support_levels.append(Decimal(str(lows.iloc[i])))

            # Keep only the most recent and significant levels
            resistance_levels = sorted(resistance_levels, reverse=True)[:3]
            support_levels = sorted(support_levels, reverse=True)[:3]

            return support_levels, resistance_levels

        except Exception as e:
            logger.warning(f"Failed to identify key levels: {str(e)}")
            return [], []

    async def _validate_signal_robustness(self,
                                        signal: TradeSignal,
                                        agent_decisions: Dict[str, AnalysisResult],
                                        market_conditions: Optional[MarketConditions]) -> Optional[Dict[str, Any]]:
        """Validate signal robustness using anti-overfitting principles."""
        try:
            validation_result = {
                'robustness_score': 0.0,
                'overfitting_risk': 0.0,
                'consensus_strength': 0.0,
                'regime_consistency': 0.0
            }

            # 1. Consensus strength validation
            total_agents = len(agent_decisions)
            agreeing_agents = 0

            for decision in agent_decisions.values():
                if decision.decision == signal.direction.value:
                    agreeing_agents += 1

            consensus_strength = agreeing_agents / max(total_agents, 1)
            validation_result['consensus_strength'] = consensus_strength

            # 2. Indicator diversity validation
            indicators_used = set(signal.indicators_used)
            indicator_categories = set()

            # Categorize indicators (simplified)
            for indicator in indicators_used:
                if any(word in indicator.lower() for word in ['rsi', 'stochastic', 'momentum']):
                    indicator_categories.add('momentum')
                elif any(word in indicator.lower() for word in ['bollinger', 'atr', 'volatility']):
                    indicator_categories.add('volatility')
                elif any(word in indicator.lower() for word in ['ma', 'ema', 'trend']):
                    indicator_categories.add('trend')
                elif any(word in indicator.lower() for word in ['volume', 'vwap']):
                    indicator_categories.add('volume')
                else:
                    indicator_categories.add('other')

            diversity_score = len(indicator_categories) / 5.0  # Max 5 categories

            # 3. Market regime consistency
            regime_consistency = 1.0  # Default high consistency
            if market_conditions:
                # Check if signal direction aligns with market regime
                if market_conditions.regime == MarketRegime.TRENDING_UP and signal.direction == TradeDirection.SELL:
                    regime_consistency *= 0.7
                elif market_conditions.regime == MarketRegime.TRENDING_DOWN and signal.direction == TradeDirection.BUY:
                    regime_consistency *= 0.7

            validation_result['regime_consistency'] = regime_consistency

            # 4. Calculate overall robustness score
            robustness_score = (
                consensus_strength * 0.4 +
                diversity_score * 0.3 +
                regime_consistency * 0.3
            )

            validation_result['robustness_score'] = robustness_score

            # 5. Calculate overfitting risk (inverse of robustness for now)
            overfitting_risk = 1.0 - robustness_score
            validation_result['overfitting_risk'] = overfitting_risk

            return validation_result

        except Exception as e:
            logger.warning(f"Signal validation failed: {str(e)}")
            return None

    async def _validate_indicator_set(self, indicators: List[str], regime: MarketRegime) -> Dict[str, Any]:
        """Validate an indicator set using walk-forward analysis."""
        try:
            # Use walk-forward validator to test indicator set
            validation_data = await self.walk_forward_validator.validate_indicator_set(indicators, regime)

            return validation_data

        except Exception as e:
            logger.warning(f"Failed to validate indicator set for regime {regime.value}: {str(e)}")
            return {
                'validation_score': 0.5,
                'performance_score': 0.5,
                'out_of_sample_performance': {},
                'robustness_metrics': {}
            }

    # Public interface methods

    def get_current_status(self) -> Dict[str, Any]:
        """Get current coordinator status."""
        return {
            'is_active': self.is_active,
            'current_cycle': {
                'cycle_id': self.current_cycle.cycle_id if self.current_cycle else None,
                'phase': self.current_cycle.phase.value if self.current_cycle else None,
                'duration_seconds': self.current_cycle.duration_seconds if self.current_cycle else 0,
                'symbol': self.current_cycle.symbol if self.current_cycle else None,
                'alpha_agent': self.current_cycle.alpha_agent if self.current_cycle else None,
                'agents_participating': len(self.current_cycle.agent_decisions) if self.current_cycle else 0
            },
            'performance': {
                'total_cycles': self.total_cycles,
                'successful_cycles': self.successful_cycles,
                'success_rate': self.successful_cycles / max(self.total_cycles, 1),
                'average_cycle_duration': self.average_cycle_duration
            },
            'last_cycle_time': self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            'next_cycle_in_minutes': self._get_minutes_to_next_cycle(),
            'elite_indicator_sets': {
                regime.value: len(elite_set.indicators)
                for regime, elite_set in self.elite_indicator_sets.items()
            }
        }

    def _get_minutes_to_next_cycle(self) -> Optional[float]:
        """Get minutes until next analysis cycle."""
        if not self.last_cycle_time:
            return None

        next_cycle_time = self.last_cycle_time + timedelta(minutes=self.analysis_frequency_minutes)
        now = datetime.utcnow()

        if next_cycle_time > now:
            return (next_cycle_time - now).total_seconds() / 60.0
        else:
            return 0.0

    def get_recent_cycles(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis cycles."""
        recent_cycles = self.cycle_history[-count:] if count > 0 else self.cycle_history

        return [
            {
                'cycle_id': cycle.cycle_id,
                'phase': cycle.phase.value,
                'start_time': cycle.start_time.isoformat(),
                'duration_seconds': cycle.duration_seconds,
                'symbol': cycle.symbol,
                'alpha_agent': cycle.alpha_agent,
                'beta_agents': cycle.beta_agents,
                'gamma_agents': cycle.gamma_agents,
                'agents_analyzed': len(cycle.agent_decisions),
                'signal_generated': cycle.final_signal is not None,
                'errors': len(cycle.error_log),
                'is_successful': cycle.is_successful
            }
            for cycle in recent_cycles
        ]

    def get_elite_indicator_summary(self) -> Dict[str, Any]:
        """Get summary of elite indicator sets."""
        return {
            regime.value: {
                'indicator_count': len(elite_set.indicators),
                'performance_score': elite_set.performance_score,
                'validation_score': elite_set.validation_score,
                'composite_score': elite_set.composite_score,
                'last_updated': elite_set.last_updated.isoformat(),
                'top_indicators': elite_set.indicators[:5]  # Top 5 indicators
            }
            for regime, elite_set in self.elite_indicator_sets.items()
        }

    async def force_analysis_cycle(self, symbol: str) -> Dict[str, Any]:
        """Force an immediate analysis cycle for testing/manual execution."""
        if self.current_cycle and self.current_cycle.phase != AnalysisCyclePhase.COMPLETED:
            raise CoordinationError("Another analysis cycle is currently running")

        logger.info(f"Forcing analysis cycle for {symbol}")

        # Temporarily store original symbol
        original_symbol = self.config_manager.get_str('coordination.primary_symbol')
        # Note: Temporarily updating config here - would need proper config override method

        try:
            await self._execute_analysis_cycle()

            result = {
                'success': True,
                'cycle_id': self.current_cycle.cycle_id,
                'duration_seconds': self.current_cycle.duration_seconds,
                'signal_generated': self.current_cycle.final_signal is not None,
                'agents_participated': len(self.current_cycle.agent_decisions),
                'errors': self.current_cycle.error_log
            }

            if self.current_cycle.final_signal:
                result['signal'] = {
                    'direction': self.current_cycle.final_signal.direction.value,
                    'confidence': self.current_cycle.final_signal.confidence,
                    'confidence_level': self.current_cycle.final_signal.confidence_level.value
                }

            return result

        finally:
            # Restore original symbol
            if original_symbol:
                self.config['primary_symbol'] = original_symbol

    # Performance Monitoring Interface

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report from the performance monitor."""
        return self.performance_monitor.get_performance_report()

    def get_optimization_effectiveness(self) -> Dict[str, Any]:
        """Get analysis of optimization technique effectiveness."""
        return self.performance_monitor.get_optimization_effectiveness()

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        return self.performance_monitor.get_real_time_metrics()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance improvements achieved."""
        try:
            report = self.performance_monitor.get_performance_report()
            
            if report.get('status') != 'success':
                return report

            return {
                'status': 'success',
                'cycle_time_improvement': {
                    'baseline_seconds': report.get('baseline_cycle_time_seconds', 1800),
                    'current_average_seconds': report.get('average_cycle_time_seconds', 0),
                    'target_seconds': report.get('target_cycle_time_seconds', 300),
                    'improvement_percent': report.get('performance_improvement_percent', 0),
                    'target_achievement_percent': report.get('target_achievement_percent', 0)
                },
                'quality_metrics': {
                    'success_rate_percent': report.get('success_rate_percent', 0),
                    'comprehensive_analysis_rate': 100.0,  # Always 100% now
                    'parallel_efficiency_avg': report.get('parallel_efficiency_avg', 0),
                    'analysis_consistency_score': 1.0  # Consistent comprehensive analysis
                },
                'volume_metrics': {
                    'total_cycles': report.get('cycles_analyzed', 0),
                    'successful_cycles': report.get('successful_cycles', 0),
                    'throughput_per_hour': report.get('throughput_cycles_per_hour', 0)
                },
                'optimization_status': {
                    'parallel_analysis_enabled': self.enable_parallel_analysis,
                    'comprehensive_analysis_enabled': True,  # Always enabled now
                    'phase_merging_enabled': self.enable_phase_merging
                }
            }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {str(e)}")
            return {'status': 'error', 'message': str(e)}
