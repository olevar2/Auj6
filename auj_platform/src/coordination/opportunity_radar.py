"""
Opportunity Radar - Crown Jewel Component for AUJ Platform.

This module implements intelligent multi-pair opportunity scanning and ranking
to replace the blind rotation approach with data-driven pair selection.

Instead of analyzing pairs one-by-one in rotation:
1. Quick-scan ALL pairs with 5 lightweight indicators (~30s total)
2. Rank by Opportunity Score
3. Deep-analyze TOP 3 only with full 10-agent analysis
4. Compare & select BEST opportunity considering correlation

Author: Antigravity AI - Crown Jewel Innovation
Date: 2025-12-05
Version: 1.0.0
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal
import logging

from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager
from ..core.data_contracts import MarketRegime, TradeDirection

logger = get_logger(__name__)


class OpportunityGrade(Enum):
    """Opportunity quality grades."""
    A_PLUS = "A+"   # Exceptional (>85)
    A = "A"         # Excellent (75-85)
    B = "B"         # Good (60-75)
    C = "C"         # Average (45-60)
    D = "D"         # Poor (<45)


@dataclass
class QuickScanResult:
    """Result from quick scan of a single pair."""
    symbol: str
    opportunity_score: float  # 0-100
    grade: OpportunityGrade
    direction: TradeDirection  # BUY or SELL
    
    # Quick indicators
    trend_clarity: float      # 0-1, ADX-based
    momentum_strength: float  # 0-1, RSI-based
    volatility_fit: float     # 0-1, ATR in sweet spot
    regime: MarketRegime
    
    # Timing
    scan_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Flags
    is_tradeable: bool = True
    skip_reason: Optional[str] = None


@dataclass
class DeepAnalysisResult:
    """Result from full 10-agent analysis."""
    symbol: str
    direction: TradeDirection
    confidence: float  # 0-1
    
    # Agent consensus
    consensus_score: float
    agents_bullish: int
    agents_bearish: int
    agents_neutral: int
    
    # Risk metrics
    risk_reward_ratio: float
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    
    # Correlation with existing positions
    max_correlation: float = 0.0
    correlation_penalty: float = 1.0
    
    # Final adjusted score
    adjusted_score: float = 0.0
    
    # Analysis details
    analysis_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OpportunityRadarResult:
    """Final result from Opportunity Radar scan."""
    # Winner
    best_pair: str
    best_direction: TradeDirection
    best_score: float
    best_analysis: Optional[DeepAnalysisResult] = None
    
    # All scanned pairs
    quick_scan_results: List[QuickScanResult] = field(default_factory=list)
    deep_analysis_results: List[DeepAnalysisResult] = field(default_factory=list)
    
    # Timing
    total_time_seconds: float = 0.0
    quick_scan_time_seconds: float = 0.0
    deep_analysis_time_seconds: float = 0.0
    
    # Statistics
    pairs_scanned: int = 0
    pairs_deep_analyzed: int = 0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OpportunityRadar:
    """
    Intelligent multi-pair opportunity scanner.
    
    Crown Jewel Innovation: Instead of blind rotation, this component:
    1. Scans ALL pairs quickly with lightweight indicators
    2. Ranks them by Opportunity Score
    3. Deep-analyzes only TOP N candidates
    4. Compares and selects the BEST opportunity
    
    Integrates with:
    - DataProviderManager: For market data
    - RegimeClassifier: For regime detection
    - DynamicRiskManager: For correlation checking
    - GeniusAgentCoordinator: For deep analysis
    """
    
    # Extended pair list (10 forex + metals + crypto)
    DEFAULT_PAIRS = [
        # Major Pairs
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
        # Cross Pairs
        'EURJPY', 'GBPJPY', 'EURGBP',
        # Commodities (if supported by broker)
        'XAUUSD',  # Gold
        # Crypto (if supported)
        'BTCUSD'
    ]
    
    # Quick scan indicators (lightweight, fast calculation)
    QUICK_INDICATORS = [
        'atr_14',       # Volatility
        'rsi_14',       # Momentum/Overbought/Oversold
        'adx_14',       # Trend strength
        'sma_50',       # Trend direction
        'volume_ratio'  # Liquidity
    ]
    
    def __init__(self,
                 data_provider,
                 regime_classifier,
                 risk_manager,
                 config_manager: UnifiedConfigManager,
                 genius_coordinator=None):
        """
        Initialize Opportunity Radar.
        
        Args:
            data_provider: DataProviderManager for market data
            regime_classifier: RegimeClassifier for regime detection
            risk_manager: DynamicRiskManager for correlation checking
            config_manager: Configuration manager
            genius_coordinator: GeniusAgentCoordinator for deep analysis (optional)
        """
        self.data_provider = data_provider
        self.regime_classifier = regime_classifier
        self.risk_manager = risk_manager
        self.config_manager = config_manager
        self.genius_coordinator = genius_coordinator
        
        # Configuration
        self.enabled = config_manager.get_bool('opportunity_radar.enabled', True)
        self.top_n = config_manager.get_int('opportunity_radar.top_n', 3)
        self.min_score_threshold = config_manager.get_float('opportunity_radar.min_score', 45.0)
        self.max_correlation = config_manager.get_float('opportunity_radar.max_correlation', 0.7)
        self.quick_scan_timeout = config_manager.get_int('opportunity_radar.quick_scan_timeout', 10)
        
        # Get trading pairs from config or use defaults
        self.trading_pairs = config_manager.get_list(
            'opportunity_radar.pairs',
            self.DEFAULT_PAIRS
        )
        
        # Score weights
        self.weights = {
            'trend_clarity': 0.25,
            'momentum_strength': 0.20,
            'entry_quality': 0.20,
            'regime_suitability': 0.20,
            'volatility_fit': 0.15
        }
        
        # Statistics
        self.total_scans = 0
        self.successful_scans = 0
        self.average_scan_time = 0.0
        
        logger.info(f"🎯 OpportunityRadar initialized with {len(self.trading_pairs)} pairs")
        logger.info(f"   Pairs: {', '.join(self.trading_pairs)}")
    
    async def find_best_opportunity(self) -> OpportunityRadarResult:
        """
        Main entry point: Find the best trading opportunity across all pairs.
        
        Returns:
            OpportunityRadarResult with best pair and analysis
        """
        start_time = datetime.utcnow()
        result = OpportunityRadarResult(
            best_pair="",
            best_direction=TradeDirection.HOLD,
            best_score=0.0,
            pairs_scanned=len(self.trading_pairs)
        )
        
        try:
            logger.info("=" * 60)
            logger.info("🎯 OPPORTUNITY RADAR: Starting multi-pair scan...")
            logger.info(f"📊 Scanning {len(self.trading_pairs)} pairs")
            logger.info("=" * 60)
            
            # Phase 1: Quick scan all pairs in parallel
            quick_scan_start = datetime.utcnow()
            quick_results = await self._parallel_quick_scan()
            result.quick_scan_results = quick_results
            result.quick_scan_time_seconds = (datetime.utcnow() - quick_scan_start).total_seconds()
            
            logger.info(f"⚡ Quick scan completed in {result.quick_scan_time_seconds:.2f}s")
            
            # Filter and rank results
            tradeable_pairs = [r for r in quick_results if r.is_tradeable and r.opportunity_score >= self.min_score_threshold]
            tradeable_pairs.sort(key=lambda x: x.opportunity_score, reverse=True)
            
            if not tradeable_pairs:
                logger.warning("⚠️ No pairs meet minimum score threshold")
                return result
            
            # Log top opportunities
            logger.info(f"🏆 Top opportunities after quick scan:")
            for i, r in enumerate(tradeable_pairs[:5], 1):
                logger.info(f"   #{i} {r.symbol}: {r.opportunity_score:.1f} ({r.grade.value}) - {r.direction.value}")
            
            # Phase 2: Deep analysis on top N pairs
            top_pairs = tradeable_pairs[:self.top_n]
            
            deep_analysis_start = datetime.utcnow()
            deep_results = await self._deep_analyze_top_pairs(top_pairs)
            result.deep_analysis_results = deep_results
            result.pairs_deep_analyzed = len(deep_results)
            result.deep_analysis_time_seconds = (datetime.utcnow() - deep_analysis_start).total_seconds()
            
            logger.info(f"🔬 Deep analysis completed in {result.deep_analysis_time_seconds:.2f}s")
            
            # Phase 3: Compare and select winner
            if deep_results:
                winner = await self._select_winner(deep_results)
                if winner:
                    result.best_pair = winner.symbol
                    result.best_direction = winner.direction
                    result.best_score = winner.adjusted_score
                    result.best_analysis = winner
                    
                    logger.info("=" * 60)
                    logger.info(f"🏆 WINNER: {winner.symbol} ({winner.direction.value})")
                    logger.info(f"   Score: {winner.adjusted_score:.2f}")
                    logger.info(f"   Confidence: {winner.confidence:.2%}")
                    logger.info(f"   R:R Ratio: {winner.risk_reward_ratio:.2f}")
                    logger.info("=" * 60)
            
            # Update statistics
            result.total_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            self.total_scans += 1
            if result.best_pair:
                self.successful_scans += 1
            self.average_scan_time = (
                (self.average_scan_time * (self.total_scans - 1) + result.total_time_seconds) / 
                self.total_scans
            )
            
            logger.info(f"⏱️ Total scan time: {result.total_time_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ OpportunityRadar scan failed: {e}")
            result.total_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            return result
    
    async def _parallel_quick_scan(self) -> List[QuickScanResult]:
        """
        Quick scan all pairs in parallel.
        
        Returns:
            List of QuickScanResult for each pair
        """
        tasks = []
        for symbol in self.trading_pairs:
            task = asyncio.create_task(self._quick_scan_single(symbol))
            tasks.append(task)
        
        # Wait for all with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        scan_results = []
        for i, result in enumerate(results):
            symbol = self.trading_pairs[i]
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Quick scan failed for {symbol}: {result}")
                scan_results.append(QuickScanResult(
                    symbol=symbol,
                    opportunity_score=0.0,
                    grade=OpportunityGrade.D,
                    direction=TradeDirection.HOLD,
                    trend_clarity=0.0,
                    momentum_strength=0.0,
                    volatility_fit=0.0,
                    regime=MarketRegime.UNKNOWN,
                    scan_time_ms=0.0,
                    is_tradeable=False,
                    skip_reason=str(result)
                ))
            else:
                scan_results.append(result)
        
        return scan_results
    
    async def _quick_scan_single(self, symbol: str) -> QuickScanResult:
        """
        Quick scan a single pair with lightweight indicators.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            QuickScanResult
        """
        start_time = datetime.utcnow()
        
        try:
            # Fetch recent OHLCV data (minimal - 100 bars)
            from ..data_providers.base_provider import Timeframe
            
            ohlcv = await asyncio.wait_for(
                self.data_provider.get_ohlcv(
                    symbol=symbol,
                    timeframe=Timeframe.H1,
                    count=100
                ),
                timeout=self.quick_scan_timeout
            )
            
            if ohlcv is None or ohlcv.empty or len(ohlcv) < 50:
                return QuickScanResult(
                    symbol=symbol,
                    opportunity_score=0.0,
                    grade=OpportunityGrade.D,
                    direction=TradeDirection.HOLD,
                    trend_clarity=0.0,
                    momentum_strength=0.0,
                    volatility_fit=0.0,
                    regime=MarketRegime.UNKNOWN,
                    scan_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                    is_tradeable=False,
                    skip_reason="Insufficient data"
                )
            
            # Calculate quick indicators
            close = ohlcv['close'].astype(float).values
            high = ohlcv['high'].astype(float).values
            low = ohlcv['low'].astype(float).values
            volume = ohlcv['volume'].astype(float).values if 'volume' in ohlcv.columns else np.ones(len(close))
            
            # 1. Trend Clarity (ADX-based)
            trend_clarity, trend_direction = self._calculate_trend_clarity(high, low, close)
            
            # 2. Momentum Strength (RSI-based)
            momentum_strength, momentum_direction = self._calculate_momentum_strength(close)
            
            # 3. Volatility Fit (ATR in sweet spot)
            volatility_fit = self._calculate_volatility_fit(high, low, close)
            
            # 4. Entry Quality (distance from MA, recent reversal)
            entry_quality = self._calculate_entry_quality(close)
            
            # 5. Regime Detection (simplified)
            regime = self._detect_regime_quick(close, volatility_fit, trend_clarity)
            
            # 6. Regime Suitability (does current regime suit our strategy?)
            regime_suitability = self._calculate_regime_suitability(regime, trend_clarity)
            
            # Calculate Opportunity Score
            opportunity_score = (
                trend_clarity * self.weights['trend_clarity'] +
                momentum_strength * self.weights['momentum_strength'] +
                entry_quality * self.weights['entry_quality'] +
                regime_suitability * self.weights['regime_suitability'] +
                volatility_fit * self.weights['volatility_fit']
            ) * 100
            
            # Determine direction
            if trend_direction > 0 and momentum_direction > 0:
                direction = TradeDirection.BUY
            elif trend_direction < 0 and momentum_direction < 0:
                direction = TradeDirection.SELL
            else:
                direction = TradeDirection.HOLD
            
            # Assign grade
            grade = self._score_to_grade(opportunity_score)
            
            scan_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return QuickScanResult(
                symbol=symbol,
                opportunity_score=opportunity_score,
                grade=grade,
                direction=direction,
                trend_clarity=trend_clarity,
                momentum_strength=momentum_strength,
                volatility_fit=volatility_fit,
                regime=regime,
                scan_time_ms=scan_time_ms,
                is_tradeable=direction != TradeDirection.HOLD
            )
            
        except asyncio.TimeoutError:
            return QuickScanResult(
                symbol=symbol,
                opportunity_score=0.0,
                grade=OpportunityGrade.D,
                direction=TradeDirection.HOLD,
                trend_clarity=0.0,
                momentum_strength=0.0,
                volatility_fit=0.0,
                regime=MarketRegime.UNKNOWN,
                scan_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                is_tradeable=False,
                skip_reason="Timeout"
            )
        except Exception as e:
            return QuickScanResult(
                symbol=symbol,
                opportunity_score=0.0,
                grade=OpportunityGrade.D,
                direction=TradeDirection.HOLD,
                trend_clarity=0.0,
                momentum_strength=0.0,
                volatility_fit=0.0,
                regime=MarketRegime.UNKNOWN,
                scan_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                is_tradeable=False,
                skip_reason=str(e)
            )
    
    def _calculate_trend_clarity(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[float, int]:
        """
        Calculate trend clarity using ADX-like calculation.
        
        Returns:
            (clarity: 0-1, direction: -1/0/1)
        """
        try:
            period = 14
            
            # True Range
            tr = np.maximum(high[1:] - low[1:], 
                          np.abs(high[1:] - close[:-1]),
                          np.abs(low[1:] - close[:-1]))
            
            # Directional Movement
            plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                              np.maximum(high[1:] - high[:-1], 0), 0)
            minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                               np.maximum(low[:-1] - low[1:], 0), 0)
            
            # Smoothed averages
            atr = pd.Series(tr).rolling(period).mean().iloc[-1]
            plus_di = (pd.Series(plus_dm).rolling(period).mean().iloc[-1] / atr * 100) if atr > 0 else 0
            minus_di = (pd.Series(minus_dm).rolling(period).mean().iloc[-1] / atr * 100) if atr > 0 else 0
            
            # ADX
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
            adx = dx / 50  # Normalize to 0-1 range (ADX > 50 = very strong)
            adx = min(adx, 1.0)
            
            # Direction
            direction = 1 if plus_di > minus_di else (-1 if minus_di > plus_di else 0)
            
            return adx, direction
            
        except Exception:
            return 0.0, 0
    
    def _calculate_momentum_strength(self, close: np.ndarray) -> Tuple[float, int]:
        """
        Calculate momentum strength using RSI.
        
        Returns:
            (strength: 0-1, direction: -1/0/1)
        """
        try:
            period = 14
            
            # Calculate RSI
            delta = np.diff(close)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            avg_gain = pd.Series(gains).rolling(period).mean().iloc[-1]
            avg_loss = pd.Series(losses).rolling(period).mean().iloc[-1]
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Convert RSI to strength (extreme values = stronger)
            if rsi >= 70:
                # Overbought - strong bullish momentum (but potentially reversal)
                strength = (rsi - 50) / 50
                direction = 1
            elif rsi <= 30:
                # Oversold - strong bearish momentum (but potentially reversal)
                strength = (50 - rsi) / 50
                direction = -1
            elif rsi >= 50:
                # Mild bullish
                strength = (rsi - 50) / 50 * 0.7
                direction = 1
            else:
                # Mild bearish
                strength = (50 - rsi) / 50 * 0.7
                direction = -1
            
            return min(strength, 1.0), direction
            
        except Exception:
            return 0.0, 0
    
    def _calculate_volatility_fit(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """
        Calculate if volatility is in sweet spot for trading.
        Too low = no movement, too high = too risky.
        
        Returns:
            fit: 0-1 (1 = perfect for trading)
        """
        try:
            # ATR
            tr = np.maximum(high[1:] - low[1:],
                          np.abs(high[1:] - close[:-1]),
                          np.abs(low[1:] - close[:-1]))
            atr = pd.Series(tr).rolling(14).mean().iloc[-1]
            
            # ATR as percentage of price
            atr_pct = (atr / close[-1]) * 100
            
            # Sweet spot: 0.1% - 1% for forex
            # Below 0.1% = too quiet, above 1% = too volatile
            if 0.1 <= atr_pct <= 0.5:
                return 1.0  # Perfect
            elif 0.05 <= atr_pct < 0.1:
                return 0.7  # Slightly low
            elif 0.5 < atr_pct <= 1.0:
                return 0.8  # Slightly high but still good
            elif atr_pct > 1.0:
                return max(0.3, 1.0 - (atr_pct - 1.0) * 0.3)  # Too volatile
            else:
                return 0.3  # Too quiet
                
        except Exception:
            return 0.5
    
    def _calculate_entry_quality(self, close: np.ndarray) -> float:
        """
        Calculate entry quality (pullback to MA, reversal signs).
        
        Returns:
            quality: 0-1
        """
        try:
            # SMA 20 and 50
            sma_20 = pd.Series(close).rolling(20).mean().iloc[-1]
            sma_50 = pd.Series(close).rolling(50).mean().iloc[-1]
            current_price = close[-1]
            
            # Distance from SMA 20 as percentage
            distance_20 = abs(current_price - sma_20) / sma_20 * 100
            distance_50 = abs(current_price - sma_50) / sma_50 * 100
            
            # Best entry: close to MA (pullback), but not too far
            # Ideal: 0.1% - 0.5% from SMA20
            if distance_20 <= 0.5:
                entry_score = 1.0  # Perfect pullback entry
            elif distance_20 <= 1.0:
                entry_score = 0.8
            elif distance_20 <= 2.0:
                entry_score = 0.5  # Extended
            else:
                entry_score = 0.3  # Too extended
            
            # Bonus if price crossed SMA recently
            sma_20_series = pd.Series(close).rolling(20).mean()
            if len(sma_20_series) >= 3:
                prev_above = close[-3] > sma_20_series.iloc[-3]
                curr_above = close[-1] > sma_20
                if prev_above != curr_above:
                    entry_score = min(entry_score + 0.2, 1.0)  # Fresh crossover
            
            return entry_score
            
        except Exception:
            return 0.5
    
    def _detect_regime_quick(self, close: np.ndarray, volatility: float, trend_clarity: float) -> MarketRegime:
        """
        Quick regime detection without full classifier.
        """
        try:
            # Check trend
            sma_50 = pd.Series(close).rolling(50).mean().iloc[-1]
            sma_20 = pd.Series(close).rolling(20).mean().iloc[-1]
            
            trend_up = close[-1] > sma_20 > sma_50
            trend_down = close[-1] < sma_20 < sma_50
            
            if trend_clarity > 0.5:
                if trend_up:
                    return MarketRegime.TRENDING_UP
                elif trend_down:
                    return MarketRegime.TRENDING_DOWN
            
            if volatility > 0.8:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.4:
                return MarketRegime.LOW_VOLATILITY
            
            return MarketRegime.SIDEWAYS
            
        except Exception:
            return MarketRegime.UNKNOWN
    
    def _calculate_regime_suitability(self, regime: MarketRegime, trend_clarity: float) -> float:
        """
        Calculate how suitable current regime is for our strategy.
        """
        # Our strategy works best in trending markets
        suitability = {
            MarketRegime.TRENDING_UP: 1.0,
            MarketRegime.TRENDING_DOWN: 1.0,
            MarketRegime.HIGH_VOLATILITY: 0.6,  # Risky but opportunities
            MarketRegime.SIDEWAYS: 0.4,  # Range-bound
            MarketRegime.LOW_VOLATILITY: 0.3,  # Dead market
            MarketRegime.UNKNOWN: 0.5
        }
        
        base = suitability.get(regime, 0.5)
        
        # Boost if trend is clear
        if trend_clarity > 0.6:
            base = min(base + 0.2, 1.0)
        
        return base
    
    def _score_to_grade(self, score: float) -> OpportunityGrade:
        """Convert numeric score to letter grade."""
        if score >= 85:
            return OpportunityGrade.A_PLUS
        elif score >= 75:
            return OpportunityGrade.A
        elif score >= 60:
            return OpportunityGrade.B
        elif score >= 45:
            return OpportunityGrade.C
        else:
            return OpportunityGrade.D
    
    async def _deep_analyze_top_pairs(self, top_pairs: List[QuickScanResult]) -> List[DeepAnalysisResult]:
        """
        Perform deep 10-agent analysis on top pairs.
        
        Args:
            top_pairs: Top N pairs from quick scan
            
        Returns:
            List of DeepAnalysisResult
        """
        results = []
        
        for quick_result in top_pairs:
            try:
                if self.genius_coordinator:
                    # Use full GeniusAgentCoordinator for deep analysis
                    start_time = datetime.utcnow()
                    
                    signal = await self.genius_coordinator.execute_analysis_cycle(
                        symbol=quick_result.symbol
                    )
                    
                    analysis_time = (datetime.utcnow() - start_time).total_seconds()
                    
                    if signal:
                        # Calculate correlation with existing positions
                        correlation_penalty = 1.0
                        max_correlation = 0.0
                        
                        if self.risk_manager:
                            try:
                                correlation_penalty = await self.risk_manager._calculate_correlation_penalty(
                                    quick_result.symbol
                                )
                                # Get max correlation from risk manager's internal calculation
                                max_correlation = 1.0 - correlation_penalty if correlation_penalty < 1.0 else 0.0
                            except Exception:
                                pass
                        
                        deep_result = DeepAnalysisResult(
                            symbol=quick_result.symbol,
                            direction=signal.direction,
                            confidence=signal.confidence,
                            consensus_score=signal.metadata.get('consensus_score', 0.6),
                            agents_bullish=signal.metadata.get('agents_bullish', 0),
                            agents_bearish=signal.metadata.get('agents_bearish', 0),
                            agents_neutral=signal.metadata.get('agents_neutral', 0),
                            risk_reward_ratio=signal.risk_reward if hasattr(signal, 'risk_reward') else 2.0,
                            suggested_stop_loss=signal.stop_loss if hasattr(signal, 'stop_loss') else None,
                            suggested_take_profit=signal.take_profit if hasattr(signal, 'take_profit') else None,
                            max_correlation=max_correlation,
                            correlation_penalty=correlation_penalty,
                            adjusted_score=quick_result.opportunity_score * correlation_penalty * signal.confidence,
                            analysis_time_seconds=analysis_time
                        )
                        results.append(deep_result)
                    else:
                        logger.warning(f"No signal generated for {quick_result.symbol}")
                        
                else:
                    # Fallback: Use quick scan data + simple analysis
                    results.append(DeepAnalysisResult(
                        symbol=quick_result.symbol,
                        direction=quick_result.direction,
                        confidence=quick_result.opportunity_score / 100,
                        consensus_score=quick_result.opportunity_score / 100,
                        agents_bullish=0,
                        agents_bearish=0,
                        agents_neutral=0,
                        risk_reward_ratio=2.0,
                        adjusted_score=quick_result.opportunity_score
                    ))
                    
            except Exception as e:
                logger.error(f"Deep analysis failed for {quick_result.symbol}: {e}")
        
        return results
    
    async def _select_winner(self, deep_results: List[DeepAnalysisResult]) -> Optional[DeepAnalysisResult]:
        """
        Select the winning opportunity from deep analysis results.
        
        Considers:
        - Adjusted score (includes correlation penalty)
        - Confidence
        - Risk/Reward ratio
        """
        if not deep_results:
            return None
        
        # Sort by adjusted score
        deep_results.sort(key=lambda x: x.adjusted_score, reverse=True)
        
        # Filter out high correlation pairs
        valid_results = [r for r in deep_results if r.max_correlation <= self.max_correlation]
        
        if not valid_results:
            logger.warning("All top pairs have high correlation - selecting best anyway with penalty")
            return deep_results[0]
        
        return valid_results[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Opportunity Radar statistics."""
        return {
            'total_scans': self.total_scans,
            'successful_scans': self.successful_scans,
            'success_rate': (self.successful_scans / self.total_scans * 100) if self.total_scans > 0 else 0,
            'average_scan_time_seconds': self.average_scan_time,
            'pairs_configured': len(self.trading_pairs),
            'enabled': self.enabled
        }
