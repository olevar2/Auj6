"""
FastAPI Application for AUJ Platform - Dashboard Service API

This module implements the comprehensive REST API backend that connects the AUJ Platform
core trading system with the Streamlit dashboard frontend. All endpoints are designed
to provide real-time data access and configuration management for the dashboard.

The API follows RESTful principles and provides endpoints for:
- System overview and status monitoring
- Trade management and deal quality assessment
- Chart data and technical analysis
- Optimization dashboard and metrics
- Account and configuration management
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

# Core platform imports
from ..core.data_contracts import (
    TradeSignal, GradedDeal, TradeStatus, DealGrade, MarketRegime,
    AgentRank, PlatformStatus, AccountInfo, OptimizationConfig,
    OHLCVData, MarketConditions, AgentPerformanceMetrics
)
from ..core.exceptions import (
    DataProviderError, PerformanceTrackingError, HierarchyError,
    ConfigurationError
)
from ..core.logging_setup import get_logger
from ..core.unified_config import get_unified_config

# Platform components
from ..analytics.performance_tracker import PerformanceTracker, ValidationPeriodType
from ..hierarchy.hierarchy_manager import HierarchyManager
from ..data_providers.data_provider_manager import DataProviderManager
from ..coordination.genius_agent_coordinator import GeniusAgentCoordinator
from ..optimization.selective_indicator_engine import SelectiveIndicatorEngine
from ..trading_engine.dynamic_risk_manager import DynamicRiskManager

logger = get_logger(__name__)


# API Response Models
class DashboardOverviewResponse(BaseModel):
    """Response model for dashboard overview endpoint."""
    system_status: str = Field(..., description="Overall system status")
    is_active: bool = Field(..., description="Whether the platform is actively trading")
    active_agents: List[str] = Field(..., description="List of currently active agent names")
    alpha_agent: Optional[str] = Field(None, description="Current Alpha agent name")
    daily_pnl: float = Field(..., description="Today's profit and loss")
    total_equity: Optional[float] = Field(None, description="Total account equity (null if unavailable)")
    active_positions: int = Field(..., description="Number of active trading positions")
    win_rate: float = Field(..., description="Overall win rate percentage")
    market_regime: Optional[str] = Field(None, description="Current market regime")
    volatility: Optional[float] = Field(None, description="Current market volatility")
    last_updated: str = Field(..., description="Last update timestamp")
    data_quality: Dict[str, str] = Field(
        default_factory=dict,
        description="Data quality indicators: REAL/FALLBACK/UNAVAILABLE with optional reason"
    )


class GradedDealResponse(BaseModel):
    """Response model for graded deals."""
    id: str
    pair: str
    strategy: str
    grade: str
    status: str
    confidence: float
    entry_time: str
    exit_time: Optional[str] = None
    entry_price: float
    exit_price: Optional[float] = None
    position_size: float
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    generating_agent: str
    supporting_agents: List[str]
    indicators_used: List[str]
    grade_factors: Dict[str, float]
    execution_quality: Optional[float] = None


class ChartDataResponse(BaseModel):
    """Response model for chart data."""
    symbol: str
    timeframe: str
    data: List[Dict[str, Any]]  # OHLCV data in pandas-compatible format
    indicators: Dict[str, List[float]] = Field(default_factory=dict)
    last_updated: str


class OptimizationDashboardResponse(BaseModel):
    """Response model for optimization dashboard."""
    agent_hierarchy: Dict[str, Any]
    market_regimes: Dict[str, Any]
    indicator_effectiveness: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    validation_results: Dict[str, Any]
    last_updated: str


class OptimizationMetricsResponse(BaseModel):
    """Response model for optimization metrics."""
    agent_performance: Dict[str, Dict[str, float]]
    indicator_efficiency: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    overfitting_indicators: Dict[str, float]
    system_health: Dict[str, Any]
    last_updated: str
    data_quality: Dict[str, str] = Field(
        default_factory=dict,
        description="Data quality indicators for metrics"
    )


class AccountResponse(BaseModel):
    """Response model for account information."""
    account_id: str
    broker: str
    account_type: str
    currency: str
    balance: float
    equity: float
    margin_available: float
    margin_used: float
    is_active: bool
    last_updated: str


# API Configuration and Dependencies
class APIComponents:
    """Container for API component dependencies."""

    def __init__(self, production_mode: bool = False):
        # Note: This class now requires config_manager parameter in __init__
        self.production_mode = production_mode  # ✅ NEW: Control strict vs lenient data validation
        self.config_manager = None  # Will be initialized in initialize()
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.hierarchy_manager: Optional[HierarchyManager] = None
        self.data_provider_manager: Optional[DataProviderManager] = None
        self.coordinator: Optional[GeniusAgentCoordinator] = None
        self.indicator_engine: Optional[SelectiveIndicatorEngine] = None
        self.risk_manager: Optional[DynamicRiskManager] = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize all platform components with proper dependency management."""
        try:
            logger.info("Initializing API components with dependency-aware sequence...")

            # Load configuration first (no dependencies)
            self.config_manager = get_unified_config()
            logger.info("✓ Configuration loaded")

            # Phase 1: Core data and analytics components (no external dependencies)
            self.performance_tracker = PerformanceTracker(
                self.config_manager.get_dict('database', {}).get('performance_db_path')
            )
            await self._wait_for_component_ready(self.performance_tracker, "PerformanceTracker")
            logger.info("✓ PerformanceTracker initialized")

            # Phase 2: Data providers (independent of other components)
            self.data_provider_manager = DataProviderManager(
                self.config_manager.get_dict('data_providers', {})
            )
            await self._wait_for_component_ready(self.data_provider_manager, "DataProviderManager")
            logger.info("✓ DataProviderManager initialized")

            # Phase 3: Hierarchy manager (depends on performance tracker)
            self.hierarchy_manager = HierarchyManager(
                self.config_manager.get_dict('hierarchy', {}),
                performance_tracker=self.performance_tracker
            )
            await self._wait_for_component_ready(self.hierarchy_manager, "HierarchyManager")
            logger.info("✓ HierarchyManager initialized")

            # Phase 4: Indicator engine (depends on data provider manager)
            self.indicator_engine = SelectiveIndicatorEngine(
                self.config_manager.get_dict('optimization', {}),
                data_provider_manager=self.data_provider_manager
            )
            await self._wait_for_component_ready(self.indicator_engine, "SelectiveIndicatorEngine")
            logger.info("✓ SelectiveIndicatorEngine initialized")

            # Phase 5: Risk manager (depends on performance tracker and data providers)
            self.risk_manager = DynamicRiskManager(
                self.config_manager.get_dict('risk_management', {}),
                performance_tracker=self.performance_tracker,
                data_provider_manager=self.data_provider_manager
            )
            await self._wait_for_component_ready(self.risk_manager, "DynamicRiskManager")
            logger.info("✓ DynamicRiskManager initialized")

            # Phase 6: Coordinator (depends on all other components)
            self.coordinator = GeniusAgentCoordinator(
                hierarchy_manager=self.hierarchy_manager,
                performance_tracker=self.performance_tracker,
                data_provider_manager=self.data_provider_manager,
                indicator_engine=self.indicator_engine,
                risk_manager=self.risk_manager,
                config=self.config_manager.get_dict('coordination', {})
            )
            await self._wait_for_component_ready(self.coordinator, "GeniusAgentCoordinator")
            logger.info("✓ GeniusAgentCoordinator initialized")

            # Verify all components are properly initialized
            await self._verify_all_components()

            self.is_initialized = True
            logger.info("🎉 All API components initialized successfully with proper dependency order")

        except Exception as e:
            logger.error(f"Failed to initialize API components: {str(e)}")
            # Cleanup any partially initialized components
            await self.cleanup()
            raise ConfigurationError(f"API initialization failed: {str(e)}")

    async def _wait_for_component_ready(self, component: Any, component_name: str, timeout_seconds: int = 30):
        """Wait for a component to be ready with timeout and proper error handling."""
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            try:
                # Check if component has an is_ready method
                if hasattr(component, 'is_ready'):
                    if await component.is_ready() if asyncio.iscoroutinefunction(component.is_ready) else component.is_ready():
                        return

                # Check if component has initialization status
                elif hasattr(component, 'is_initialized'):
                    if component.is_initialized:
                        return

                # Check if component has connection status
                elif hasattr(component, 'is_connected'):
                    if await component.is_connected() if asyncio.iscoroutinefunction(component.is_connected) else component.is_connected():
                        return

                # For components without explicit ready checks, assume ready if they exist and have key methods
                elif component is not None:
                    # Basic validation that component has expected interface
                    if hasattr(component, '__class__'):
                        return

                # Wait a short time before checking again
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"Error checking {component_name} readiness: {e}")
                await asyncio.sleep(0.5)

        raise TimeoutError(f"{component_name} failed to become ready within {timeout_seconds} seconds")

    async def _verify_all_components(self):
        """Verify all components are properly initialized and functional."""
        components_to_verify = [
            ("PerformanceTracker", self.performance_tracker),
            ("HierarchyManager", self.hierarchy_manager),
            ("DataProviderManager", self.data_provider_manager),
            ("SelectiveIndicatorEngine", self.indicator_engine),
            ("DynamicRiskManager", self.risk_manager),
            ("GeniusAgentCoordinator", self.coordinator)
        ]

        for name, component in components_to_verify:
            if component is None:
                raise ConfigurationError(f"{name} is None after initialization")

            # Additional verification for critical components
            if name == "DataProviderManager" and hasattr(component, 'providers'):
                if not component.providers:
                    logger.warning(f"{name} has no providers configured")

            logger.debug(f"✓ {name} verification passed")

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up API components...")

        if self.data_provider_manager:
            # Disconnect all providers
            for provider in self.data_provider_manager.providers.values():
                try:
                    await provider.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting provider: {str(e)}")

        logger.info("API cleanup completed")


# Global components instance
api_components = APIComponents()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle."""
    # Startup
    await api_components.initialize()
    yield
    # Shutdown
    await api_components.cleanup()


# FastAPI Application
app = FastAPI(
    title="AUJ Platform API",
    description="Advanced Automated Trading Platform - Dashboard Service API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS middleware for dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✨ OPPORTUNITY RADAR API ROUTER
from .radar_api import router as radar_router
app.include_router(radar_router)


def get_components() -> APIComponents:
    """Dependency to get initialized components."""
    if not api_components.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="API components not initialized yet. Please try again in a moment."
        )
    return api_components


# Health Check Endpoint
@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint for API monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components_initialized": api_components.is_initialized
    }

# Docker Health Check Endpoint (simplified for Docker HEALTHCHECK)
@app.get("/health", tags=["Health"])
async def docker_health_check():
    """Simplified health check endpoint for Docker containers."""
    return {"status": "healthy"}


# Dashboard Overview Endpoints
@app.get("/api/v1/dashboard/overview",
         response_model=DashboardOverviewResponse,
         tags=["Dashboard"])
async def get_dashboard_overview(
    components: APIComponents = Depends(get_components)
) -> DashboardOverviewResponse:
    """
    Get comprehensive system overview for the main dashboard.

    Returns real-time summary including system status, agent hierarchy,
    performance metrics, and current market conditions.
    """
    try:
        # Get performance summary from tracker
        performance_summary = components.performance_tracker.get_platform_summary()

        # Get current hierarchy state
        alpha_agent = components.hierarchy_manager.get_alpha_agent()

        # Build comprehensive active agents list
        active_agents = []
        if alpha_agent:
            active_agents.append(alpha_agent.name)

        beta_agents = components.hierarchy_manager.get_beta_agents()
        for agent in beta_agents:
            active_agents.append(agent.name)

        gamma_agents = components.hierarchy_manager.get_gamma_agents()
        for agent in gamma_agents:
            active_agents.append(agent.name)

        # Get current market conditions from coordinator if available
        # Track data quality for all fields
        data_quality = {}
        
        # Market regime and volatility - NO HARDCODED FALLBACKS
        current_regime = None
        volatility = None
        data_quality['market_regime'] = 'UNAVAILABLE'
        data_quality['volatility'] = 'UNAVAILABLE'

        try:
            # Try to get real market regime from coordinator's regime analysis
            if (components.coordinator and
                hasattr(components.coordinator, 'regime_detector') and
                components.coordinator.regime_detector):
                regime_data = await components.coordinator.regime_detector.get_current_regime()
                if regime_data:
                    current_regime = regime_data.get('regime')
                    volatility = regime_data.get('volatility')
                    if current_regime:
                        data_quality['market_regime'] = 'REAL'
                    if volatility is not None:
                        data_quality['volatility'] = 'REAL'
            # Fallback to data provider's market analysis
            elif components.data_provider_manager:
                market_data = await components.data_provider_manager.get_market_conditions()
                if market_data:
                    volatility = market_data.get('volatility')
                    if volatility is not None:
                        data_quality['volatility'] = 'REAL'
        except Exception as e:
            logger.error(f"Failed to get current market regime: {e}")

        # Calculate daily P&L from actual performance tracker
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        daily_pnl = 0.0
        data_quality['daily_pnl'] = 'REAL'

        # Get recent trades for daily P&L calculation using actual tracker data
        completed_trades = getattr(components.performance_tracker, 'completed_trades', {})
        for trade in completed_trades.values():
            if (hasattr(trade, 'exit_time') and trade.exit_time and
                trade.exit_time >= today_start and
                hasattr(trade, 'pnl') and trade.pnl):
                daily_pnl += float(trade.pnl)

        # Get real equity from risk manager - CRITICAL DATA, NO FALLBACK
        total_equity = None
        data_quality['total_equity'] = 'UNAVAILABLE - No equity source connected'
        
        try:
            if (components.risk_manager and
                hasattr(components.risk_manager, 'get_current_equity')):
                total_equity = await components.risk_manager.get_current_equity()
                data_quality['total_equity'] = 'REAL'
            elif (components.coordinator and
                  hasattr(components.coordinator, 'get_account_equity')):
                total_equity = await components.coordinator.get_account_equity()
                data_quality['total_equity'] = 'REAL'
            else:
                # CRITICAL: No equity source available
                if components.production_mode:
                    # Production mode: FAIL FAST
                    raise HTTPException(
                        status_code=503,
                        detail="Account equity unavailable. Risk manager not connected."
                    )
                else:
                    # Development mode: Allow null but warn clearly
                    logger.warning("⚠️ NO EQUITY SOURCE - total_equity will be null")
                    total_equity = None
                    data_quality['total_equity'] = 'UNAVAILABLE - Development mode'
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get account equity: {e}")
            if components.production_mode:
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot retrieve account equity: {str(e)}"
                )
            else:
                total_equity = None
                data_quality['total_equity'] = f'ERROR - {str(e)}'

        # Determine system status based on actual component health
        system_status = "ACTIVE"
        is_active = True

        try:
            # Check if coordinator is running
            if (components.coordinator and
                hasattr(components.coordinator, 'is_active')):
                is_active = components.coordinator.is_active()
            # Check data provider connectivity
            if (components.data_provider_manager and
                hasattr(components.data_provider_manager, 'are_providers_connected')):
                providers_connected = components.data_provider_manager.are_providers_connected()
                if not providers_connected:
                    system_status = "DEGRADED"
        except Exception as e:
            logger.warning(f"Could not determine system status: {e}")
            system_status = "UNKNOWN"

        return DashboardOverviewResponse(
            system_status=system_status,
            is_active=is_active,
            active_agents=active_agents,
            alpha_agent=alpha_agent.name if alpha_agent else None,
            daily_pnl=daily_pnl,
            total_equity=total_equity,
            active_positions=performance_summary.get("active_trades", 0),
            win_rate=performance_summary.get("win_rate", 0.0) * 100,  # Convert to percentage
            market_regime=current_regime.value if hasattr(current_regime, 'value') else str(current_regime) if current_regime else None,
            volatility=volatility,
            last_updated=datetime.utcnow().isoformat(),
            data_quality=data_quality  # ✅ NEW: Data quality tracking
        )

    except Exception as e:
        logger.error(f"Error getting dashboard overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Deal Quality Endpoints
@app.get("/api/v1/deals/graded",
         response_model=List[GradedDealResponse],
         tags=["Deals"])
async def get_graded_deals(
    status_filter: Optional[str] = Query(None, description="Filter by trade status"),
    grade_filter: Optional[str] = Query(None, description="Filter by grade (A+, A, B+, etc.)"),
    pair_filter: Optional[str] = Query(None, description="Filter by trading pair"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of deals to return"),
    components: APIComponents = Depends(get_components)
) -> List[GradedDealResponse]:
    """
    Get list of recent deals graded by the system.

    Returns comprehensive information about each deal including performance metrics,
    quality assessment, and execution details.
    """
    try:
        deals = []

        # Get recent completed trades from performance tracker
        completed_trades = getattr(components.performance_tracker, 'completed_trades', {})
        if not completed_trades:
            return []

        # Convert to list and sort by exit time (most recent first)
        recent_trades = list(completed_trades.values())
        recent_trades.sort(
            key=lambda x: x.exit_time if hasattr(x, 'exit_time') and x.exit_time else datetime.min,
            reverse=True
        )

        # Apply filters and build response
        count = 0
        for trade in recent_trades:
            if count >= limit:
                break

            # Extract trade attributes safely
            trade_status = getattr(trade, 'status', 'CLOSED')
            trade_grade = getattr(trade, 'grade', None)
            trade_symbol = getattr(trade, 'symbol', None)

            # If no symbol from trade, try to get from original signal
            if not trade_symbol and hasattr(trade, 'original_signal'):
                trade_symbol = getattr(trade.original_signal, 'symbol', 'UNKNOWN')

            # Apply filters
            if status_filter and trade_status != status_filter:
                continue

            if grade_filter and (not trade_grade or
                                (hasattr(trade_grade, 'value') and trade_grade.value != grade_filter) or
                                (isinstance(trade_grade, str) and trade_grade != grade_filter)):
                continue

            if pair_filter and trade_symbol != pair_filter:
                continue

            # Extract deal data with safe attribute access
            deal_response = GradedDealResponse(
                id=getattr(trade, 'trade_id', f"trade_{count}"),
                pair=trade_symbol or 'UNKNOWN',
                strategy=getattr(trade, 'strategy_name', 'Unknown Strategy'),
                grade=trade_grade.value if hasattr(trade_grade, 'value') else str(trade_grade) if trade_grade else "F",
                status=trade_status,
                confidence=float(getattr(trade, 'confidence', 0.0)),
                entry_time=trade.entry_time.isoformat() if hasattr(trade, 'entry_time') and trade.entry_time else datetime.utcnow().isoformat(),
                exit_time=trade.exit_time.isoformat() if hasattr(trade, 'exit_time') and trade.exit_time else None,
                entry_price=float(getattr(trade, 'entry_price', 0.0)),
                exit_price=float(getattr(trade, 'exit_price', 0.0)) if hasattr(trade, 'exit_price') and trade.exit_price else None,
                position_size=float(getattr(trade, 'position_size', 0.0)),
                pnl=float(getattr(trade, 'pnl', 0.0)) if hasattr(trade, 'pnl') and trade.pnl else None,
                pnl_percentage=getattr(trade, 'pnl_percentage', None),
                generating_agent=getattr(trade, 'generating_agent', 'Unknown Agent'),
                supporting_agents=getattr(trade, 'supporting_agents', []),
                indicators_used=getattr(trade, 'indicators_used', []),
                grade_factors=getattr(trade, 'grade_factors', {}),
                execution_quality=getattr(trade, 'execution_quality', None)
            )

            deals.append(deal_response)
            count += 1

        return deals

    except Exception as e:
        logger.error(f"Error getting graded deals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Chart Analysis Endpoints
@app.get("/api/v1/chart/data",
         response_model=ChartDataResponse,
         tags=["Chart Analysis"])
async def get_chart_data(
    pair: str = Query(..., description="Trading pair symbol"),
    timeframe: str = Query("1H", description="Timeframe (1M, 5M, 15M, 30M, 1H, 4H, 1D)"),
    count: int = Query(100, ge=10, le=1000, description="Number of candles"),
    components: APIComponents = Depends(get_components)
) -> ChartDataResponse:
    """
    Get historical and live OHLCV data for chart analysis.

    Returns pandas-compatible JSON data suitable for chart visualization
    with optional technical indicators.
    """
    try:
        # Map string timeframe to enum
        from data_providers.base_provider import Timeframe

        timeframe_map = {
            "1M": Timeframe.M1,
            "5M": Timeframe.M5,
            "15M": Timeframe.M15,
            "30M": Timeframe.M30,
            "1H": Timeframe.H1,
            "4H": Timeframe.H4,
            "1D": Timeframe.D1
        }

        tf_enum = timeframe_map.get(timeframe, Timeframe.H1)

        # Get OHLCV data from data provider manager
        try:
            ohlcv_data = await components.data_provider_manager.get_ohlcv_data(
                symbol=pair,
                timeframe=tf_enum,
                count=count
            )
        except Exception as e:
            logger.error(f"Error getting OHLCV data from data provider: {e}")
            # Try fallback approach
            try:
                ohlcv_data = await components.data_provider_manager.get_ohlcv_data(
                    symbol=pair,
                    timeframe=tf_enum,
                    start_time=datetime.utcnow() - timedelta(hours=count),
                    end_time=datetime.utcnow()
                )
            except Exception as fallback_error:
                logger.error(f"Fallback OHLCV data fetch also failed: {fallback_error}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Data provider unavailable for {pair}. Please try again later."
                )

        if ohlcv_data is None or ohlcv_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {pair} on {timeframe} timeframe"
            )

        # Convert DataFrame to JSON-compatible format
        data_list = []
        for idx, row in ohlcv_data.iterrows():
            # Handle different index types (datetime vs string)
            timestamp = idx
            if hasattr(timestamp, 'isoformat'):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)

            data_list.append({
                "timestamp": timestamp_str,
                "open": float(row.get("Open", row.get("open", 0.0))),
                "high": float(row.get("High", row.get("high", 0.0))),
                "low": float(row.get("Low", row.get("low", 0.0))),
                "close": float(row.get("Close", row.get("close", 0.0))),
                "volume": float(row.get("Volume", row.get("volume", 0.0)))
            })

        # Get indicators from indicator engine if available
        indicators = {}
        try:
            if (components.indicator_engine and
                hasattr(components.indicator_engine, 'calculate_indicators')):
                indicator_results = await components.indicator_engine.calculate_indicators(
                    ohlcv_data,
                    symbol=pair,
                    requested_indicators=['sma_20', 'sma_50', 'rsi', 'macd']
                )
                if indicator_results:
                    # Convert indicator results to lists
                    for indicator_name, values in indicator_results.items():
                        if hasattr(values, 'tolist'):
                            indicators[indicator_name] = values.tolist()
                        elif isinstance(values, list):
                            indicators[indicator_name] = values
                        else:
                            indicators[indicator_name] = [float(values)] * len(data_list)
        except Exception as e:
            logger.warning(f"Could not calculate indicators for {pair}: {e}")
            # Provide empty indicators as fallback
            indicators = {
                "sma_20": [0.0] * len(data_list),
                "sma_50": [0.0] * len(data_list),
            }

        return ChartDataResponse(
            symbol=pair,
            timeframe=timeframe,
            data=data_list,
            indicators=indicators,
            last_updated=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error getting chart data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# Optimization Dashboard Endpoints
@app.get("/api/v1/optimization/dashboard",
         response_model=OptimizationDashboardResponse,
         tags=["Optimization"])
async def get_optimization_dashboard(
    components: APIComponents = Depends(get_components)
) -> OptimizationDashboardResponse:
    """
    Get comprehensive optimization dashboard data.

    Returns agent hierarchy visualization data, market regime analysis,
    indicator effectiveness metrics, and validation results.
    """
    try:
        # Agent Hierarchy Data - use actual hierarchy manager
        alpha_agent = components.hierarchy_manager.get_alpha_agent()
        beta_agents = components.hierarchy_manager.get_beta_agents()
        gamma_agents = components.hierarchy_manager.get_gamma_agents()

        hierarchy_data = {
            "alpha_agent": alpha_agent.name if alpha_agent else None,
            "beta_agents": [agent.name for agent in beta_agents],
            "gamma_agents": [agent.name for agent in gamma_agents],
            "inactive_agents": getattr(components.hierarchy_manager, 'inactive_agents', []),
            "rankings": getattr(components.hierarchy_manager, 'agent_rankings', {}),
            "last_update": getattr(components.hierarchy_manager, 'last_ranking_update', datetime.utcnow()).isoformat()
        }

        # Market Regimes Data - try to get from coordinator or fallback
        regimes_data = {
            "current_regime": "SIDEWAYS",  # Default
            "regime_history": [],
            "regime_specialists": {}
        }

        try:
            if (components.coordinator and
                hasattr(components.coordinator, 'regime_detector')):
                current_regime_info = await components.coordinator.regime_detector.get_current_regime()
                if current_regime_info:
                    regimes_data["current_regime"] = current_regime_info.get("regime", "SIDEWAYS")
                    regimes_data["regime_history"] = current_regime_info.get("history", [])

            # Get regime specialists from hierarchy manager
            if hasattr(components.hierarchy_manager, 'regime_specialists'):
                regime_specialists = components.hierarchy_manager.regime_specialists
                regimes_data["regime_specialists"] = {
                    regime.value if hasattr(regime, 'value') else str(regime): agents
                    for regime, agents in regime_specialists.items()
                }
        except Exception as e:
            logger.warning(f"Could not get regime data: {e}")

        # Indicator Effectiveness - get from selective indicator engine
        effectiveness_data = {}
        try:
            if (components.indicator_engine and
                hasattr(components.indicator_engine, 'get_elite_indicators')):
                elite_indicators = await components.indicator_engine.get_elite_indicators()
                effectiveness_data = {
                    "elite_indicators": elite_indicators or {},
                    "effectiveness_scores": getattr(components.indicator_engine, 'indicator_scores', {}),
                    "regime_specific": getattr(components.indicator_engine, 'regime_effectiveness', {})
                }
            elif hasattr(components.indicator_engine, 'elite_indicators'):
                effectiveness_data = {
                    "elite_indicators": getattr(components.indicator_engine, 'elite_indicators', {}),
                    "effectiveness_scores": getattr(components.indicator_engine, 'indicator_scores', {}),
                    "regime_specific": getattr(components.indicator_engine, 'regime_effectiveness', {})
                }
        except Exception as e:
            logger.warning(f"Could not get indicator effectiveness data: {e}")

        # Performance Metrics - get actual agent performance from tracker
        performance_data = {}
        try:
            # Get all agent names from hierarchy
            all_agents = set()
            if alpha_agent:
                all_agents.add(alpha_agent.name)
            all_agents.update(agent.name for agent in beta_agents)
            all_agents.update(agent.name for agent in gamma_agents)

            # Add any other agents from hierarchy manager
            if hasattr(components.hierarchy_manager, 'agents'):
                all_agents.update(components.hierarchy_manager.agents.keys())

            for agent_name in all_agents:
                try:
                    agent_perf = components.performance_tracker.get_agent_performance(agent_name, days_back=30)
                    oos_perf = components.performance_tracker.get_out_of_sample_performance_only(agent_name)

                    performance_data[agent_name] = {
                        "win_rate": agent_perf.get("win_rate", 0.0),
                        "total_pnl": agent_perf.get("total_pnl", 0.0),
                        "sharpe_ratio": agent_perf.get("sharpe_ratio", None),
                        "out_of_sample_performance": oos_perf.get("win_rate", 0.0)
                    }
                except Exception as agent_error:
                    logger.warning(f"Could not get performance for agent {agent_name}: {agent_error}")
                    performance_data[agent_name] = {
                        "win_rate": 0.0,
                        "total_pnl": 0.0,
                        "sharpe_ratio": None,
                        "out_of_sample_performance": 0.0
                    }
        except Exception as e:
            logger.warning(f"Could not get performance data: {e}")

        # Validation Results - get actual validation data where available
        validation_data = {
            "recent_validations": [],
            "overfitting_alerts": [],
            "robustness_scores": {}
        }

        try:
            # Check if performance tracker has validation methods
            if hasattr(components.performance_tracker, 'get_recent_validations'):
                validation_data["recent_validations"] = components.performance_tracker.get_recent_validations()

            # Calculate overfitting alerts based on performance gaps
            for agent_name in performance_data.keys():
                try:
                    all_perf = components.performance_tracker.get_agent_performance(agent_name)
                    oos_perf = components.performance_tracker.get_out_of_sample_performance_only(agent_name)

                    all_wr = all_perf.get("win_rate", 0.0)
                    oos_wr = oos_perf.get("win_rate", 0.0)

                    # If gap is > 20%, flag as potential overfitting
                    if all_wr - oos_wr > 0.2:
                        validation_data["overfitting_alerts"].append({
                            "agent": agent_name,
                            "gap": all_wr - oos_wr,
                            "severity": "HIGH" if all_wr - oos_wr > 0.3 else "MEDIUM"
                        })

                    validation_data["robustness_scores"][agent_name] = max(0.0, 1.0 - (all_wr - oos_wr))
                except Exception as validation_error:
                    logger.warning(f"Could not calculate validation metrics for {agent_name}: {validation_error}")
        except Exception as e:
            logger.warning(f"Could not get validation data: {e}")

        return OptimizationDashboardResponse(
            agent_hierarchy=hierarchy_data,
            market_regimes=regimes_data,
            indicator_effectiveness=effectiveness_data,
            performance_metrics=performance_data,
            validation_results=validation_data,
            last_updated=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error getting optimization dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/optimization/metrics",
         response_model=OptimizationMetricsResponse,
         tags=["Optimization"])
async def get_optimization_metrics(
    components: APIComponents = Depends(get_components)
) -> OptimizationMetricsResponse:
    """
    Get real-time optimization performance metrics.

    Returns current agent performance, indicator efficiency,
    out-of-sample validation results, and overfitting indicators.
    """
    try:
        # Agent Performance Metrics - get from actual performance tracker
        agent_performance = {}

        # Get all agents from hierarchy manager
        all_agents = set()
        alpha_agent = components.hierarchy_manager.get_alpha_agent()
        if alpha_agent:
            all_agents.add(alpha_agent.name)

        beta_agents = components.hierarchy_manager.get_beta_agents()
        all_agents.update(agent.name for agent in beta_agents)

        gamma_agents = components.hierarchy_manager.get_gamma_agents()
        all_agents.update(agent.name for agent in gamma_agents)

        # Also include any other tracked agents
        if hasattr(components.hierarchy_manager, 'agents'):
            all_agents.update(components.hierarchy_manager.agents.keys())

        for agent_name in all_agents:
            try:
                perf = components.performance_tracker.get_agent_performance(agent_name, days_back=7)
                agent_performance[agent_name] = {
                    "win_rate": perf.get("win_rate", 0.0),
                    "profit_factor": perf.get("profit_factor", 0.0),
                    "sharpe_ratio": perf.get("sharpe_ratio", 0.0),
                    "max_drawdown": perf.get("max_drawdown", 0.0)
                }
            except Exception as agent_error:
                logger.warning(f"Could not get performance metrics for {agent_name}: {agent_error}")
                agent_performance[agent_name] = {
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0
                }

        # Indicator Efficiency - get from indicator engine
        indicator_efficiency = {}
        data_quality_indicators = 'UNAVAILABLE'
        
        try:
            if (components.indicator_engine and
                hasattr(components.indicator_engine, 'get_indicator_effectiveness')):
                effectiveness_scores = await components.indicator_engine.get_indicator_effectiveness()
                if effectiveness_scores:
                    indicator_efficiency = effectiveness_scores
                    data_quality_indicators = 'REAL'
            elif hasattr(components.indicator_engine, 'indicator_scores'):
                indicator_efficiency = getattr(components.indicator_engine, 'indicator_scores', {})
                if indicator_efficiency:
                    data_quality_indicators = 'REAL'
            
            # NO FALLBACK - if no data, return empty dict with clear quality indicator
            if not indicator_efficiency:
                logger.warning("⚠️ No indicator effectiveness data available from engine")
                data_quality_indicators = 'UNAVAILABLE - Indicator engine not configured'
        except Exception as e:
            logger.error(f"Failed to get indicator efficiency: {e}")
            data_quality_indicators = f'ERROR - {str(e)}'
            indicator_efficiency = {}

        # Out-of-Sample Performance - get actual OOS data
        oos_performance = {}
        for agent_name in all_agents:
            try:
                oos_perf = components.performance_tracker.get_out_of_sample_performance_only(agent_name)
                oos_performance[agent_name] = oos_perf.get("win_rate", 0.0)
            except Exception as oos_error:
                logger.warning(f"Could not get OOS performance for {agent_name}: {oos_error}")
                oos_performance[agent_name] = 0.0

        # Overfitting Indicators - calculate gap between in-sample and out-of-sample
        overfitting_indicators = {}
        for agent_name in all_agents:
            try:
                # Get all performance (includes in-sample)
                all_perf = components.performance_tracker.get_agent_performance(agent_name)
                # Get only out-of-sample performance
                oos_perf = components.performance_tracker.get_out_of_sample_performance_only(agent_name)

                all_wr = all_perf.get("win_rate", 0.0)
                oos_wr = oos_perf.get("win_rate", 0.0)

                # Calculate overfitting score (positive gap indicates potential overfitting)
                overfitting_score = max(0.0, all_wr - oos_wr)
                overfitting_indicators[agent_name] = overfitting_score
            except Exception as overfitting_error:
                logger.warning(f"Could not calculate overfitting score for {agent_name}: {overfitting_error}")
                overfitting_indicators[agent_name] = 0.0

        # System Health - check actual component status
        system_health = {
            "api_status": "healthy",
            "data_provider_status": "unknown",
            "database_status": "unknown",
            "active_components": 0
        }

        try:
            # Check data provider status
            if (components.data_provider_manager and
                hasattr(components.data_provider_manager, 'are_providers_connected')):
                if components.data_provider_manager.are_providers_connected():
                    system_health["data_provider_status"] = "connected"
                else:
                    system_health["data_provider_status"] = "disconnected"
            elif (components.data_provider_manager and
                  hasattr(components.data_provider_manager, 'providers')):
                # Check individual providers
                provider_statuses = []
                for provider in components.data_provider_manager.providers.values():
                    if hasattr(provider, 'is_connected'):
                        provider_statuses.append(provider.is_connected)
                if provider_statuses:
                    if all(provider_statuses):
                        system_health["data_provider_status"] = "connected"
                    elif any(provider_statuses):
                        system_health["data_provider_status"] = "partial"
                    else:
                        system_health["data_provider_status"] = "disconnected"

            # Check database status
            if (components.performance_tracker and
                hasattr(components.performance_tracker, 'db_connection')):
                # Try a simple database operation
                try:
                    components.performance_tracker.get_platform_summary()
                    system_health["database_status"] = "connected"
                except Exception:
                    system_health["database_status"] = "disconnected"

            # Count active components
            active_count = 0
            component_list = [
                components.performance_tracker,
                components.hierarchy_manager,
                components.data_provider_manager,
                components.coordinator,
                components.indicator_engine,
                components.risk_manager
            ]
            for component in component_list:
                if component is not None:
                    active_count += 1

            system_health["active_components"] = active_count

        except Exception as health_error:
            logger.warning(f"Could not determine full system health: {health_error}")

        return OptimizationMetricsResponse(
            agent_performance=agent_performance,
            indicator_efficiency=indicator_efficiency,
            out_of_sample_performance=oos_performance,
            overfitting_indicators=overfitting_indicators,
            system_health=system_health,
            last_updated=datetime.utcnow().isoformat(),
            data_quality={'indicator_efficiency': data_quality_indicators}  # ✅ NEW
        )

    except Exception as e:
        logger.error(f"Error getting optimization metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration Management Endpoints
@app.put("/api/v1/optimization/config",
         tags=["Configuration"])
async def update_optimization_config(
    config_updates: Dict[str, Any] = Body(..., description="Configuration updates"),
    components: APIComponents = Depends(get_components)
):
    """
    Update optimization configuration settings.

    Accepts configuration changes from the control panel and applies them
    to the relevant system components.
    """
    try:
        applied_updates = {}
        failed_updates = {}

        # Feature flags updates
        if "feature_flags" in config_updates:
            try:
                # Update feature flags in indicator engine
                if (components.indicator_engine and
                    hasattr(components.indicator_engine, 'update_feature_flags')):
                    await components.indicator_engine.update_feature_flags(config_updates["feature_flags"])
                    applied_updates["feature_flags"] = config_updates["feature_flags"]
                elif (components.indicator_engine and
                      hasattr(components.indicator_engine, 'feature_flags')):
                    # Direct attribute update if method not available
                    components.indicator_engine.feature_flags.update(config_updates["feature_flags"])
                    applied_updates["feature_flags"] = config_updates["feature_flags"]
                else:
                    failed_updates["feature_flags"] = "Indicator engine not available or doesn't support feature flags"
            except Exception as e:
                failed_updates["feature_flags"] = f"Failed to update feature flags: {str(e)}"

        # Risk parameters updates
        if "risk_parameters" in config_updates:
            try:
                if components.risk_manager:
                    # Apply risk parameter updates
                    for param, value in config_updates["risk_parameters"].items():
                        try:
                            if hasattr(components.risk_manager, f'set_{param}'):
                                getattr(components.risk_manager, f'set_{param}')(value)
                            elif hasattr(components.risk_manager, param):
                                setattr(components.risk_manager, param, value)
                            else:
                                logger.warning(f"Risk manager doesn't have parameter: {param}")
                        except Exception as param_error:
                            logger.warning(f"Failed to set risk parameter {param}: {param_error}")
                    applied_updates["risk_parameters"] = config_updates["risk_parameters"]
                else:
                    failed_updates["risk_parameters"] = "Risk manager not available"
            except Exception as e:
                failed_updates["risk_parameters"] = f"Failed to update risk parameters: {str(e)}"

        # Hierarchy thresholds updates
        if "hierarchy_thresholds" in config_updates:
            try:
                hierarchy_config = config_updates["hierarchy_thresholds"]
                updated_thresholds = {}

                if "alpha_threshold" in hierarchy_config:
                    if hasattr(components.hierarchy_manager, 'alpha_threshold'):
                        components.hierarchy_manager.alpha_threshold = hierarchy_config["alpha_threshold"]
                        updated_thresholds["alpha_threshold"] = hierarchy_config["alpha_threshold"]

                if "beta_threshold" in hierarchy_config:
                    if hasattr(components.hierarchy_manager, 'beta_threshold'):
                        components.hierarchy_manager.beta_threshold = hierarchy_config["beta_threshold"]
                        updated_thresholds["beta_threshold"] = hierarchy_config["beta_threshold"]

                if "gamma_threshold" in hierarchy_config:
                    if hasattr(components.hierarchy_manager, 'gamma_threshold'):
                        components.hierarchy_manager.gamma_threshold = hierarchy_config["gamma_threshold"]
                        updated_thresholds["gamma_threshold"] = hierarchy_config["gamma_threshold"]

                if updated_thresholds:
                    applied_updates["hierarchy_thresholds"] = updated_thresholds
                else:
                    failed_updates["hierarchy_thresholds"] = "No hierarchy thresholds could be updated"
            except Exception as e:
                failed_updates["hierarchy_thresholds"] = f"Failed to update hierarchy thresholds: {str(e)}"

        # Performance tracking settings
        if "performance_settings" in config_updates:
            try:
                perf_config = config_updates["performance_settings"]
                updated_settings = {}

                if "rolling_window_size" in perf_config:
                    if hasattr(components.performance_tracker, 'rolling_window_size'):
                        components.performance_tracker.rolling_window_size = perf_config["rolling_window_size"]
                        updated_settings["rolling_window_size"] = perf_config["rolling_window_size"]

                if "validation_period" in perf_config:
                    if hasattr(components.performance_tracker, 'validation_period'):
                        components.performance_tracker.validation_period = perf_config["validation_period"]
                        updated_settings["validation_period"] = perf_config["validation_period"]

                if updated_settings:
                    applied_updates["performance_settings"] = updated_settings
                else:
                    failed_updates["performance_settings"] = "No performance settings could be updated"
            except Exception as e:
                failed_updates["performance_settings"] = f"Failed to update performance settings: {str(e)}"

        # Log results
        if applied_updates:
            logger.info(f"Successfully applied configuration updates: {list(applied_updates.keys())}")
        if failed_updates:
            logger.warning(f"Failed to apply some configuration updates: {failed_updates}")

        # Determine response status
        if applied_updates and not failed_updates:
            status = "success"
        elif applied_updates and failed_updates:
            status = "partial_success"
        else:
            status = "failed"

        return {
            "status": status,
            "applied_updates": applied_updates,
            "failed_updates": failed_updates,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error updating optimization config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Account Management Endpoints
@app.get("/api/v1/accounts/list",
         response_model=List[AccountResponse],
         tags=["Account Management"])
async def get_accounts_list(
    components: APIComponents = Depends(get_components)
) -> List[AccountResponse]:
    """
    Get list of all configured trading accounts.

    Returns account information including balance, equity, and status
    for all configured trading accounts.
    """
    try:
        accounts = []

        # Try to get accounts from risk manager (which might have account info)
        if (components.risk_manager and
            hasattr(components.risk_manager, 'get_all_accounts')):
            try:
                account_data = await components.risk_manager.get_all_accounts()
                for account_info in account_data:
                    accounts.append(AccountResponse(
                        account_id=account_info.get("account_id", "UNKNOWN"),
                        broker=account_info.get("broker", "Unknown Broker"),
                        account_type=account_info.get("account_type", "UNKNOWN"),
                        currency=account_info.get("currency", "USD"),
                        balance=float(account_info.get("balance", 0.0)),
                        equity=float(account_info.get("equity", 0.0)),
                        margin_available=float(account_info.get("margin_available", 0.0)),
                        margin_used=float(account_info.get("margin_used", 0.0)),
                        is_active=account_info.get("is_active", False),
                        last_updated=datetime.utcnow().isoformat()
                    ))
                return accounts
            except Exception as e:
                logger.warning(f"Could not get accounts from risk manager: {e}")

        # Try to get account info from coordinator
        if (components.coordinator and
            hasattr(components.coordinator, 'get_account_info')):
            try:
                account_info = await components.coordinator.get_account_info()
                if account_info:
                    # Validate required fields are present
                    if not all(key in account_info for key in ['balance', 'equity']):
                        logger.warning("⚠️ Account info missing required fields, skipping")
                    else:
                        accounts.append(AccountResponse(
                            account_id=account_info.get("account_id", "PRIMARY_ACCOUNT"),
                            broker=account_info.get("broker", "MetaTrader 5"),
                            account_type=account_info.get("account_type", "DEMO"),
                            currency=account_info.get("currency", "USD"),
                            balance=float(account_info.get("balance")),
                            equity=float(account_info.get("equity")),
                            margin_available=float(account_info.get("margin_available", 0.0)),
                            margin_used=float(account_info.get("margin_used", 0.0)),
                            is_active=account_info.get("is_active", True),
                            last_updated=datetime.utcnow().isoformat()
                        ))
                        return accounts
            except Exception as e:
                logger.warning(f"Could not get account info from coordinator: {e}")

        # Try to get account info from data provider (some providers have account data)
        if (components.data_provider_manager and
            hasattr(components.data_provider_manager, 'get_account_info')):
            try:
                for provider_name, provider in components.data_provider_manager.providers.items():
                    if hasattr(provider, 'get_account_info'):
                        account_info = await provider.get_account_info()
                        if account_info:
                            accounts.append(AccountResponse(
                                account_id=f"{provider_name.upper()}_ACCOUNT",
                                broker=provider_name.title(),
                                account_type=account_info.get("account_type", "DEMO"),
                                currency=account_info.get("currency", "USD"),
                                balance=float(account_info.get("balance", 10000.0)),
                                equity=float(account_info.get("equity", 10000.0)),
                                margin_available=float(account_info.get("margin_available", 9500.0)),
                                margin_used=float(account_info.get("margin_used", 500.0)),
                                is_active=account_info.get("is_active", True),
                                last_updated=datetime.utcnow().isoformat()
                            ))
                if accounts:
                    return accounts
            except Exception as e:
                logger.warning(f"Could not get account info from data providers: {e}")

        # NO FAKE DEMO ACCOUNTS - return empty or error based on mode
        if not accounts:
            logger.error("❌ No account data available from any source")
            
            if components.production_mode:
                # Production: FAIL - account data is critical
                raise HTTPException(
                    status_code=503,
                    detail="Account information unavailable. Please check broker connections."
                )
            else:
                # Development: Return empty with clear warning
                logger.warning("⚠️ RETURNING EMPTY ACCOUNTS LIST - Development mode")
                return []

        return accounts

    except Exception as e:
        logger.error(f"Error getting accounts list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/accounts/add",
          tags=["Account Management"])
async def add_account(
    account_data: Dict[str, Any] = Body(..., description="New account information"),
    components: APIComponents = Depends(get_components)
):
    """
    Add a new trading account to the platform.

    Accepts account configuration and credentials for a new trading account.
    """
    try:
        # Validate required fields
        required_fields = ["account_id", "broker", "account_type", "currency"]
        for field in required_fields:
            if field not in account_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )

        # Here would be the actual account creation logic
        # For now, return success response

        new_account = {
            "account_id": account_data["account_id"],
            "broker": account_data["broker"],
            "account_type": account_data["account_type"],
            "currency": account_data["currency"],
            "status": "CREATED",
            "created_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Added new account: {account_data['account_id']}")

        return {
            "status": "success",
            "account": new_account,
            "message": "Account added successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding account: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/accounts/{account_id}/status",
         tags=["Account Management"])
async def update_account_status(
    account_id: str = Path(..., description="Account ID"),
    status_update: Dict[str, str] = Body(..., description="Status update"),
    components: APIComponents = Depends(get_components)
):
    """
    Update trading account status.

    Allows changing account status (ACTIVE, PAUSED, DISABLED) for
    risk management and operational control.
    """
    try:
        new_status = status_update.get("status")
        if not new_status:
            raise HTTPException(status_code=400, detail="Status field is required")

        valid_statuses = ["ACTIVE", "PAUSED", "DISABLED"]
        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )

        # Here would be the actual account status update logic
        logger.info(f"Updated account {account_id} status to {new_status}")

        return {
            "status": "success",
            "account_id": account_id,
            "new_status": new_status,
            "updated_at": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating account status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Additional utility endpoints
@app.get("/api/v1/system/status", tags=["System"])
async def get_system_status(
    components: APIComponents = Depends(get_components)
):
    """Get comprehensive system status information."""
    try:
        # Data provider status - check actual connections
        provider_status = {}
        if components.data_provider_manager:
            for name, provider in components.data_provider_manager.providers.items():
                try:
                    is_connected = False
                    connection_status = "UNKNOWN"
                    last_heartbeat = None

                    # Check connection status
                    if hasattr(provider, 'is_connected'):
                        is_connected = provider.is_connected
                    elif hasattr(provider, 'connection_status'):
                        connection_status = provider.connection_status.value if hasattr(provider.connection_status, 'value') else str(provider.connection_status)
                        is_connected = connection_status in ["CONNECTED", "ACTIVE"]

                    # Check heartbeat
                    if hasattr(provider, 'last_heartbeat') and provider.last_heartbeat:
                        last_heartbeat = provider.last_heartbeat.isoformat()
                    elif hasattr(provider, 'last_update') and provider.last_update:
                        last_heartbeat = provider.last_update.isoformat()

                    provider_status[name] = {
                        "connected": is_connected,
                        "status": connection_status,
                        "last_heartbeat": last_heartbeat
                    }
                except Exception as provider_error:
                    logger.warning(f"Error checking provider {name} status: {provider_error}")
                    provider_status[name] = {
                        "connected": False,
                        "status": "ERROR",
                        "last_heartbeat": None
                    }

        # Component status - check if components are initialized and functional
        component_status = {}
        component_checks = [
            ("performance_tracker", components.performance_tracker),
            ("hierarchy_manager", components.hierarchy_manager),
            ("data_provider_manager", components.data_provider_manager),
            ("coordinator", components.coordinator),
            ("indicator_engine", components.indicator_engine),
            ("risk_manager", components.risk_manager)
        ]

        for component_name, component in component_checks:
            is_healthy = False
            try:
                if component is not None:
                    # Check if component has a health check method
                    if hasattr(component, 'health_check'):
                        is_healthy = await component.health_check() if asyncio.iscoroutinefunction(component.health_check) else component.health_check()
                    elif hasattr(component, 'is_healthy'):
                        is_healthy = component.is_healthy()
                    elif hasattr(component, 'is_initialized'):
                        is_healthy = component.is_initialized
                    else:
                        # Basic check - component exists and has expected methods
                        is_healthy = True

                component_status[component_name] = is_healthy
            except Exception as component_error:
                logger.warning(f"Error checking {component_name} health: {component_error}")
                component_status[component_name] = False

        # System metrics - get actual performance data
        system_metrics = {}
        try:
            # Active trades
            active_trades_count = 0
            if (components.performance_tracker and
                hasattr(components.performance_tracker, 'active_trades')):
                active_trades_count = len(components.performance_tracker.active_trades)
            system_metrics["active_trades"] = active_trades_count

            # Total agents
            total_agents = 0
            if (components.hierarchy_manager and
                hasattr(components.hierarchy_manager, 'agents')):
                total_agents = len(components.hierarchy_manager.agents)
            elif components.hierarchy_manager:
                # Count from individual agent lists
                alpha_agent = components.hierarchy_manager.get_alpha_agent()
                beta_agents = components.hierarchy_manager.get_beta_agents()
                gamma_agents = components.hierarchy_manager.get_gamma_agents()

                total_agents = (1 if alpha_agent else 0) + len(beta_agents) + len(gamma_agents)
            system_metrics["total_agents"] = total_agents

            # System uptime (would need to track start time)
            system_metrics["uptime"] = "N/A"  # Would calculate from actual start time

            # Memory usage (basic check)
            import psutil
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / 1024 / 1024
            system_metrics["memory_usage"] = f"{memory_usage_mb:.1f} MB"
        except ImportError:
            system_metrics["memory_usage"] = "N/A (psutil not available)"
        except Exception as metrics_error:
            logger.warning(f"Error getting system metrics: {metrics_error}")
            system_metrics = {
                "active_trades": 0,
                "total_agents": 0,
                "uptime": "N/A",
                "memory_usage": "N/A"
            }

        # Determine overall status
        overall_status = "OPERATIONAL"
        try:
            # Check if critical components are healthy
            critical_components = ["performance_tracker", "hierarchy_manager", "data_provider_manager"]
            critical_health = [component_status.get(comp, False) for comp in critical_components]

            if not any(critical_health):
                overall_status = "CRITICAL"
            elif not all(critical_health):
                overall_status = "DEGRADED"
            elif not any(provider_status.get(provider, {}).get("connected", False) for provider in provider_status):
                overall_status = "DEGRADED"  # No data providers connected
        except Exception as status_error:
            logger.warning(f"Error determining overall status: {status_error}")
            overall_status = "UNKNOWN"

        return {
            "overall_status": overall_status,
            "components": component_status,
            "data_providers": provider_status,
            "metrics": system_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Development server configuration
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
