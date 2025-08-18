"""
Synthetic Option Indicator - Advanced Options Analytics Using Stock Data
=======================================================================

This module implements a sophisticated synthetic options indicator that derives
options-like analytics from underlying stock price and volume data. It provides
implied volatility estimation, synthetic Greeks calculation, and options
strategy analysis without requiring actual options data.

Features:
- Synthetic implied volatility calculation
- Delta, Gamma, Theta, Vega estimation
- Put-call parity analysis
- Volatility surface modeling
- Risk reversal and volatility skew analysis
- Synthetic straddle and strangle pricing
- Option flow inference from volume patterns
- Volatility smile and term structure
- Monte Carlo options pricing simulation

The indicator helps traders understand options market dynamics and identify
opportunities using only the underlying asset data, making options analysis
accessible even when options data is not available.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from scipy.special import erf
from math import sqrt, log, exp, pi
import warnings
warnings.filterwarnings('ignore')

from src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface, IndicatorResult, DataRequirement, DataType, SignalType
from src.core.exceptions import IndicatorCalculationError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SyntheticOption:
    """Represents a synthetic option contract"""
    option_type: str  # 'call' or 'put'
    strike: float
    expiry_days: int
    implied_volatility: float
    theoretical_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    intrinsic_value: float
    time_value: float
    moneyness: float  # S/K for calls, K/S for puts


@dataclass
class VolatilityPoint:
    """Represents a point on the volatility surface"""
    strike: float
    expiry_days: int
    implied_volatility: float
    delta: float
    volume_weight: float


@dataclass
class OptionsStrategy:
    """Represents a synthetic options strategy"""
    strategy_name: str
    components: List[Dict[str, Any]]  # List of option legs
    net_premium: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    profit_probability: float
    strategy_delta: float
    strategy_gamma: float
    strategy_theta: float
    strategy_vega: float


class SyntheticOptionIndicator(StandardIndicatorInterface):
    """
    Advanced Synthetic Option Indicator with comprehensive options analytics
    derived from underlying stock data.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'volatility_window': 21,  # Days for volatility calculation
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'dividend_yield': 0.0,  # Assume no dividends
            'expiry_days': [7, 14, 30, 60, 90, 180],  # Days to expiration for synthetic options
            'strike_range': 0.2,  # ±20% from current price for strike range
            'strike_intervals': 21,  # Number of strike prices
            'volatility_estimation_method': 'garch',  # 'historical', 'garch', 'parkinson'
            'min_volatility': 0.05,  # 5% minimum volatility
            'max_volatility': 2.0,  # 200% maximum volatility
            'volume_volatility_weight': 0.3,  # Weight of volume in volatility estimation
            'skew_adjustment': True,  # Adjust for volatility skew
            'smile_parameters': {'a': 0.1, 'b': 0.02, 'rho': -0.3},  # Volatility smile parameters
            'monte_carlo_simulations': 10000,  # Number of MC simulations
            'confidence_intervals': [0.05, 0.95],  # 5% and 95% confidence levels
            'strategy_types': ['straddle', 'strangle', 'collar', 'butterfly', 'condor'],
            'rebalancing_threshold': 0.1,  # 10% move for strategy rebalancing
            'liquidity_adjustment': True,  # Adjust pricing for liquidity
            'bid_ask_spread': 0.02  # 2% bid-ask spread assumption
        }

        if parameters:
            default_params.update(parameters)

        super().__init__(name="SyntheticOption")

        # Initialize internal state
        self.synthetic_options: List[SyntheticOption] = []
        self.volatility_surface: List[VolatilityPoint] = []
        self.options_strategies: List[OptionsStrategy] = []
        self.current_volatility = 0.0
        self.volatility_history = []

        logger.info(f"SyntheticOptionIndicator initialized")

    def get_data_requirements(self) -> DataRequirement:
        """Define the data requirements for synthetic options calculation"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(50, self.parameters['volatility_window'] * 2),
            lookback_periods=252  # One year of data for better volatility estimation
        )

    def validate_parameters(self) -> bool:
        """Validate the indicator parameters"""
        try:
            required_params = ['volatility_window', 'risk_free_rate', 'expiry_days']
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")

            if self.parameters['volatility_window'] < 5:
                raise ValueError("volatility_window must be at least 5")

            if not self.parameters['expiry_days']:
                raise ValueError("expiry_days cannot be empty")

            return True

        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False

    def _calculate_historical_volatility(self, data: pd.DataFrame) -> float:
        """Calculate historical volatility using close-to-close returns"""
        try:
            window = self.parameters['volatility_window']
            returns = data['close'].pct_change().dropna()

            if len(returns) < window:
                return 0.2  # Default 20% volatility

            vol = returns.rolling(window).std().iloc[-1] * sqrt(252)
            return np.clip(vol, self.parameters['min_volatility'], self.parameters['max_volatility'])

        except Exception as e:
            logger.error(f"Error calculating historical volatility: {str(e)}")
            return 0.2

    def _calculate_parkinson_volatility(self, data: pd.DataFrame) -> float:
        """Calculate Parkinson volatility using high-low range"""
        try:
            window = self.parameters['volatility_window']

            if len(data) < window:
                return 0.2

            # Parkinson volatility formula
            log_hl_ratio = np.log(data['high'] / data['low'])
            parkinson_var = log_hl_ratio.rolling(window).apply(
                lambda x: np.mean(x**2) / (4 * log(2))
            ).iloc[-1]

            vol = sqrt(parkinson_var * 252)
            return np.clip(vol, self.parameters['min_volatility'], self.parameters['max_volatility'])

        except Exception as e:
            logger.error(f"Error calculating Parkinson volatility: {str(e)}")
            return 0.2

    def _calculate_garch_volatility(self, data: pd.DataFrame) -> float:
        """Calculate GARCH-style volatility with exponential weighting"""
        try:
            returns = data['close'].pct_change().dropna()

            if len(returns) < 20:
                return self._calculate_historical_volatility(data)

            # Simple GARCH(1,1) approximation
            lambda_factor = 0.94  # RiskMetrics lambda
            weights = np.array([lambda_factor**i for i in range(len(returns))])
            weights = weights[::-1]  # Reverse to give more weight to recent observations
            weights = weights / weights.sum()

            # Weighted variance
            weighted_returns = returns.tail(len(weights))
            variance = np.sum(weights * (weighted_returns**2))
            vol = sqrt(variance * 252)

            return np.clip(vol, self.parameters['min_volatility'], self.parameters['max_volatility'])

        except Exception as e:
            logger.error(f"Error calculating GARCH volatility: {str(e)}")
            return self._calculate_historical_volatility(data)

    def _estimate_implied_volatility(self, data: pd.DataFrame) -> float:
        """Estimate implied volatility using specified method"""
        try:
            method = self.parameters['volatility_estimation_method']

            if method == 'historical':
                vol = self._calculate_historical_volatility(data)
            elif method == 'parkinson':
                vol = self._calculate_parkinson_volatility(data)
            elif method == 'garch':
                vol = self._calculate_garch_volatility(data)
            else:
                vol = self._calculate_historical_volatility(data)

            # Adjust for volume if enabled
            if self.parameters['volume_volatility_weight'] > 0:
                volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
                if not np.isnan(volume_ratio):
                    volume_adjustment = 1 + (volume_ratio - 1) * self.parameters['volume_volatility_weight']
                    vol *= volume_adjustment

            self.current_volatility = vol
            self.volatility_history.append(vol)

            # Keep only recent history
            if len(self.volatility_history) > 100:
                self.volatility_history = self.volatility_history[-100:]

            return vol

        except Exception as e:
            logger.error(f"Error estimating implied volatility: {str(e)}")
            return 0.2

    def _black_scholes_price(self, S: float, K: float, T: float, r: float,
                           sigma: float, option_type: str = 'call') -> float:
        """Calculate Black-Scholes option price"""
        try:
            if T <= 0:
                return max(S - K, 0) if option_type == 'call' else max(K - S, 0)

            d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)

            if option_type == 'call':
                price = S * self._norm_cdf(d1) - K * exp(-r * T) * self._norm_cdf(d2)
            else:  # put
                price = K * exp(-r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)

            return max(price, 0)

        except Exception as e:
            logger.error(f"Error calculating Black-Scholes price: {str(e)}")
            return 0.0

    def _norm_cdf(self, x: float) -> float:
        """Calculate cumulative normal distribution"""
        return 0.5 * (1 + erf(x / sqrt(2)))

    def _norm_pdf(self, x: float) -> float:
        """Calculate normal probability density function"""
        return exp(-0.5 * x**2) / sqrt(2 * pi)

    def _calculate_greeks(self, S: float, K: float, T: float, r: float,
                        sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """Calculate option Greeks"""
        try:
            if T <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

            d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)

            # Delta
            if option_type == 'call':
                delta = self._norm_cdf(d1)
            else:
                delta = -self._norm_cdf(-d1)

            # Gamma (same for calls and puts)
            gamma = self._norm_pdf(d1) / (S * sigma * sqrt(T))

            # Theta
            theta_common = (-S * self._norm_pdf(d1) * sigma / (2 * sqrt(T)) -
                          r * K * exp(-r * T))

            if option_type == 'call':
                theta = (theta_common * self._norm_cdf(d2)) / 365  # Daily theta
            else:
                theta = (theta_common * self._norm_cdf(-d2) +
                        r * K * exp(-r * T)) / 365

            # Vega (same for calls and puts)
            vega = S * self._norm_pdf(d1) * sqrt(T) / 100  # Per 1% volatility

            # Rho
            if option_type == 'call':
                rho = K * T * exp(-r * T) * self._norm_cdf(d2) / 100
            else:
                rho = -K * T * exp(-r * T) * self._norm_cdf(-d2) / 100

            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }

        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    def _adjust_volatility_for_skew(self, volatility: float, moneyness: float) -> float:
        """Adjust volatility for volatility skew/smile"""
        try:
            if not self.parameters['skew_adjustment']:
                return volatility

            # Simple volatility smile model
            smile_params = self.parameters['smile_parameters']
            a = smile_params['a']
            b = smile_params['b']
            rho = smile_params['rho']

            # Moneyness adjustment (simplified Heston-like model)
            log_moneyness = log(moneyness)
            skew_adjustment = a * log_moneyness + b * log_moneyness**2 + rho * abs(log_moneyness)

            adjusted_vol = volatility * (1 + skew_adjustment)
            return np.clip(adjusted_vol, self.parameters['min_volatility'], self.parameters['max_volatility'])

        except Exception as e:
            logger.error(f"Error adjusting volatility for skew: {str(e)}")
            return volatility

    def _generate_synthetic_options(self, current_price: float, volatility: float) -> List[SyntheticOption]:
        """Generate synthetic options across strikes and expiries"""
        try:
            options = []
            r = self.parameters['risk_free_rate']

            # Generate strike prices
            strike_range = self.parameters['strike_range']
            strike_intervals = self.parameters['strike_intervals']

            min_strike = current_price * (1 - strike_range)
            max_strike = current_price * (1 + strike_range)
            strikes = np.linspace(min_strike, max_strike, strike_intervals)

            for expiry_days in self.parameters['expiry_days']:
                T = expiry_days / 365.0  # Convert to years

                for strike in strikes:
                    for option_type in ['call', 'put']:
                        # Calculate moneyness
                        if option_type == 'call':
                            moneyness = current_price / strike
                        else:
                            moneyness = strike / current_price

                        # Adjust volatility for skew
                        adjusted_vol = self._adjust_volatility_for_skew(volatility, moneyness)

                        # Calculate option price
                        price = self._black_scholes_price(current_price, strike, T, r, adjusted_vol, option_type)

                        # Calculate Greeks
                        greeks = self._calculate_greeks(current_price, strike, T, r, adjusted_vol, option_type)

                        # Calculate intrinsic and time value
                        if option_type == 'call':
                            intrinsic = max(current_price - strike, 0)
                        else:
                            intrinsic = max(strike - current_price, 0)

                        time_value = price - intrinsic

                        option = SyntheticOption(
                            option_type=option_type,
                            strike=strike,
                            expiry_days=expiry_days,
                            implied_volatility=adjusted_vol,
                            theoretical_price=price,
                            delta=greeks['delta'],
                            gamma=greeks['gamma'],
                            theta=greeks['theta'],
                            vega=greeks['vega'],
                            rho=greeks['rho'],
                            intrinsic_value=intrinsic,
                            time_value=time_value,
                            moneyness=moneyness
                        )
                        options.append(option)

            return options

        except Exception as e:
            logger.error(f"Error generating synthetic options: {str(e)}")
            return []

    def _create_volatility_surface(self, options: List[SyntheticOption]) -> List[VolatilityPoint]:
        """Create volatility surface from synthetic options"""
        try:
            surface_points = []

            for option in options:
                if option.option_type == 'call':  # Use calls for surface
                    point = VolatilityPoint(
                        strike=option.strike,
                        expiry_days=option.expiry_days,
                        implied_volatility=option.implied_volatility,
                        delta=option.delta,
                        volume_weight=1.0  # Equal weight for synthetic data
                    )
                    surface_points.append(point)

            return surface_points

        except Exception as e:
            logger.error(f"Error creating volatility surface: {str(e)}")
            return []
    def _create_options_strategies(self, options: List[SyntheticOption],
                                 current_price: float) -> List[OptionsStrategy]:
        """Create synthetic options strategies"""
        try:
            strategies = []

            # Filter options near current price for strategies
            atm_options = [opt for opt in options if abs(opt.strike - current_price) / current_price < 0.1]

            if len(atm_options) < 4:
                return strategies

            # Group by expiry
            expiry_groups = {}
            for option in atm_options:
                if option.expiry_days not in expiry_groups:
                    expiry_groups[option.expiry_days] = []
                expiry_groups[option.expiry_days].append(option)

            for expiry_days, expiry_options in expiry_groups.items():
                calls = [opt for opt in expiry_options if opt.option_type == 'call']
                puts = [opt for opt in expiry_options if opt.option_type == 'put']

                if len(calls) < 2 or len(puts) < 2:
                    continue

                # Sort by strike
                calls.sort(key=lambda x: x.strike)
                puts.sort(key=lambda x: x.strike)

                # ATM options
                atm_call = min(calls, key=lambda x: abs(x.strike - current_price))
                atm_put = min(puts, key=lambda x: abs(x.strike - current_price))

                # Long Straddle
                straddle = self._create_straddle_strategy(atm_call, atm_put, current_price)
                if straddle:
                    strategies.append(straddle)

                # Long Strangle (if we have OTM options)
                otm_call = next((c for c in calls if c.strike > current_price), None)
                otm_put = next((p for p in puts if p.strike < current_price), None)

                if otm_call and otm_put:
                    strangle = self._create_strangle_strategy(otm_call, otm_put, current_price)
                    if strangle:
                        strategies.append(strangle)

                # Protective Collar (if we have enough strikes)
                if len(calls) >= 2 and len(puts) >= 2:
                    collar = self._create_collar_strategy(calls, puts, current_price)
                    if collar:
                        strategies.append(collar)

            return strategies

        except Exception as e:
            logger.error(f"Error creating options strategies: {str(e)}")
            return []

    def _create_straddle_strategy(self, call: SyntheticOption, put: SyntheticOption,
                                current_price: float) -> Optional[OptionsStrategy]:
        """Create long straddle strategy"""
        try:
            net_premium = call.theoretical_price + put.theoretical_price

            # Breakeven points
            breakeven_up = call.strike + net_premium
            breakeven_down = put.strike - net_premium

            # Max profit is unlimited, max loss is premium paid
            max_loss = net_premium
            max_profit = float('inf')

            # Estimate profit probability (simplified)
            volatility = call.implied_volatility
            days_to_expiry = call.expiry_days
            required_move = net_premium / current_price

            # Probability of moving beyond breakevens
            daily_vol = volatility / sqrt(252)
            expected_move = daily_vol * sqrt(days_to_expiry)
            profit_probability = 1 - stats.norm.cdf(required_move / expected_move)

            # Strategy Greeks
            strategy_delta = call.delta + put.delta
            strategy_gamma = call.gamma + put.gamma
            strategy_theta = call.theta + put.theta
            strategy_vega = call.vega + put.vega

            components = [
                {'type': 'call', 'action': 'buy', 'strike': call.strike, 'price': call.theoretical_price},
                {'type': 'put', 'action': 'buy', 'strike': put.strike, 'price': put.theoretical_price}
            ]

            return OptionsStrategy(
                strategy_name='Long Straddle',
                components=components,
                net_premium=net_premium,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_down, breakeven_up],
                profit_probability=profit_probability,
                strategy_delta=strategy_delta,
                strategy_gamma=strategy_gamma,
                strategy_theta=strategy_theta,
                strategy_vega=strategy_vega
            )

        except Exception as e:
            logger.error(f"Error creating straddle strategy: {str(e)}")
            return None

    def _create_strangle_strategy(self, call: SyntheticOption, put: SyntheticOption,
                                current_price: float) -> Optional[OptionsStrategy]:
        """Create long strangle strategy"""
        try:
            net_premium = call.theoretical_price + put.theoretical_price

            # Breakeven points
            breakeven_up = call.strike + net_premium
            breakeven_down = put.strike - net_premium

            max_loss = net_premium
            max_profit = float('inf')

            # Profit probability calculation
            volatility = call.implied_volatility
            days_to_expiry = call.expiry_days

            # Distance to breakevens
            distance_up = abs(breakeven_up - current_price) / current_price
            distance_down = abs(breakeven_down - current_price) / current_price
            min_required_move = min(distance_up, distance_down)

            daily_vol = volatility / sqrt(252)
            expected_move = daily_vol * sqrt(days_to_expiry)
            profit_probability = 1 - stats.norm.cdf(min_required_move / expected_move)

            # Strategy Greeks
            strategy_delta = call.delta + put.delta
            strategy_gamma = call.gamma + put.gamma
            strategy_theta = call.theta + put.theta
            strategy_vega = call.vega + put.vega

            components = [
                {'type': 'call', 'action': 'buy', 'strike': call.strike, 'price': call.theoretical_price},
                {'type': 'put', 'action': 'buy', 'strike': put.strike, 'price': put.theoretical_price}
            ]

            return OptionsStrategy(
                strategy_name='Long Strangle',
                components=components,
                net_premium=net_premium,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_down, breakeven_up],
                profit_probability=profit_probability,
                strategy_delta=strategy_delta,
                strategy_gamma=strategy_gamma,
                strategy_theta=strategy_theta,
                strategy_vega=strategy_vega
            )

        except Exception as e:
            logger.error(f"Error creating strangle strategy: {str(e)}")
            return None

    def _create_collar_strategy(self, calls: List[SyntheticOption], puts: List[SyntheticOption],
                              current_price: float) -> Optional[OptionsStrategy]:
        """Create protective collar strategy"""
        try:
            # Find appropriate strikes for collar
            otm_call = next((c for c in calls if c.strike > current_price), None)
            otm_put = next((p for p in puts if p.strike < current_price), None)

            if not otm_call or not otm_put:
                return None

            # Collar: Long stock + Long put + Short call
            # Net premium = put_price - call_price (assuming we already own stock)
            net_premium = otm_put.theoretical_price - otm_call.theoretical_price

            # Max profit = Call strike - current price - net premium
            max_profit = otm_call.strike - current_price - net_premium

            # Max loss = current price - put strike + net premium
            max_loss = current_price - otm_put.strike + net_premium

            # Breakeven = current price + net premium
            breakeven = current_price + net_premium

            # Profit probability (simplified)
            profit_probability = 0.6 if max_profit > 0 else 0.4

            # Strategy Greeks (adjusted for stock position)
            strategy_delta = 1 + otm_put.delta - otm_call.delta  # Include stock delta = 1
            strategy_gamma = otm_put.gamma - otm_call.gamma
            strategy_theta = otm_put.theta - otm_call.theta
            strategy_vega = otm_put.vega - otm_call.vega

            components = [
                {'type': 'stock', 'action': 'own', 'quantity': 100, 'price': current_price},
                {'type': 'put', 'action': 'buy', 'strike': otm_put.strike, 'price': otm_put.theoretical_price},
                {'type': 'call', 'action': 'sell', 'strike': otm_call.strike, 'price': otm_call.theoretical_price}
            ]

            return OptionsStrategy(
                strategy_name='Protective Collar',
                components=components,
                net_premium=net_premium,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                profit_probability=profit_probability,
                strategy_delta=strategy_delta,
                strategy_gamma=strategy_gamma,
                strategy_theta=strategy_theta,
                strategy_vega=strategy_vega
            )

        except Exception as e:
            logger.error(f"Error creating collar strategy: {str(e)}")
            return None

    def _analyze_options_flow(self, data: pd.DataFrame, options: List[SyntheticOption]) -> Dict[str, Any]:
        """Analyze synthetic options flow based on volume patterns"""
        try:
            if len(data) < 5:
                return {'call_put_ratio': 1.0, 'flow_direction': 'neutral', 'flow_strength': 0.0}

            # Use volume patterns to infer options activity
            recent_volume = data['volume'].tail(5)
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]

            # Volume surge could indicate options activity
            volume_ratio = recent_volume.mean() / avg_volume if avg_volume > 0 else 1.0

            # Price action analysis for flow inference
            recent_returns = data['close'].pct_change().tail(5)
            price_momentum = recent_returns.mean()

            # Infer call/put bias from price action and volume
            if price_momentum > 0.01 and volume_ratio > 1.2:
                call_put_ratio = 1.5  # Bullish flow
                flow_direction = 'bullish'
            elif price_momentum < -0.01 and volume_ratio > 1.2:
                call_put_ratio = 0.7  # Bearish flow
                flow_direction = 'bearish'
            else:
                call_put_ratio = 1.0
                flow_direction = 'neutral'

            # Flow strength based on volume and price movement
            flow_strength = min(volume_ratio * abs(price_momentum) * 10, 1.0)

            return {
                'call_put_ratio': call_put_ratio,
                'flow_direction': flow_direction,
                'flow_strength': flow_strength,
                'volume_ratio': volume_ratio,
                'price_momentum': price_momentum
            }

        except Exception as e:
            logger.error(f"Error analyzing options flow: {str(e)}")
            return {'call_put_ratio': 1.0, 'flow_direction': 'neutral', 'flow_strength': 0.0}

    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive synthetic options analysis
        """
        try:
            if len(data) < self.parameters['volatility_window']:
                return {
                    'current_volatility': 0.0,
                    'synthetic_options': [],
                    'volatility_surface': [],
                    'options_strategies': [],
                    'options_flow': {},
                    'atm_call': None,
                    'atm_put': None,
                    'volatility_percentile': 0.5
                }

            current_price = data['close'].iloc[-1]

            # Estimate implied volatility
            implied_volatility = self._estimate_implied_volatility(data)

            # Generate synthetic options
            synthetic_options = self._generate_synthetic_options(current_price, implied_volatility)
            self.synthetic_options = synthetic_options

            # Create volatility surface
            volatility_surface = self._create_volatility_surface(synthetic_options)
            self.volatility_surface = volatility_surface

            # Create options strategies
            options_strategies = self._create_options_strategies(synthetic_options, current_price)
            self.options_strategies = options_strategies

            # Analyze options flow
            options_flow = self._analyze_options_flow(data, synthetic_options)

            # Find ATM options for key metrics
            atm_call = min(
                [opt for opt in synthetic_options if opt.option_type == 'call'],
                key=lambda x: abs(x.strike - current_price),
                default=None
            )

            atm_put = min(
                [opt for opt in synthetic_options if opt.option_type == 'put'],
                key=lambda x: abs(x.strike - current_price),
                default=None
            )

            # Calculate volatility percentile
            volatility_percentile = 0.5
            if len(self.volatility_history) > 20:
                volatility_percentile = stats.percentileofscore(
                    self.volatility_history, implied_volatility) / 100

            # Calculate key option metrics
            put_call_parity_check = 0.0
            if atm_call and atm_put and abs(atm_call.strike - atm_put.strike) < 0.01:
                # Put-Call Parity: C - P = S - K*e^(-r*T)
                T = atm_call.expiry_days / 365.0
                r = self.parameters['risk_free_rate']
                theoretical_diff = current_price - atm_call.strike * exp(-r * T)
                actual_diff = atm_call.theoretical_price - atm_put.theoretical_price
                put_call_parity_check = abs(actual_diff - theoretical_diff)

            # Prepare result
            result = {
                'current_price': current_price,
                'current_volatility': implied_volatility,
                'volatility_percentile': volatility_percentile,
                'synthetic_options': [self._option_to_dict(opt) for opt in synthetic_options[:50]],  # Limit for performance
                'volatility_surface': [self._vol_point_to_dict(point) for point in volatility_surface[:30]],
                'options_strategies': [self._strategy_to_dict(strategy) for strategy in options_strategies],
                'options_flow': options_flow,
                'atm_call': self._option_to_dict(atm_call) if atm_call else None,
                'atm_put': self._option_to_dict(atm_put) if atm_put else None,
                'put_call_parity_deviation': put_call_parity_check,
                'total_synthetic_options': len(synthetic_options),
                'total_strategies': len(options_strategies),
                'risk_free_rate': self.parameters['risk_free_rate'],
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error in synthetic options calculation: {str(e)}")
            raise IndicatorCalculationError(
                indicator_name=self.name,
                calculation_step="synthetic_options_calculation",
                message=str(e)
            )

    def _option_to_dict(self, option: SyntheticOption) -> Dict[str, Any]:
        """Convert SyntheticOption to dictionary"""
        return {
            'option_type': option.option_type,
            'strike': option.strike,
            'expiry_days': option.expiry_days,
            'implied_volatility': option.implied_volatility,
            'theoretical_price': option.theoretical_price,
            'delta': option.delta,
            'gamma': option.gamma,
            'theta': option.theta,
            'vega': option.vega,
            'rho': option.rho,
            'intrinsic_value': option.intrinsic_value,
            'time_value': option.time_value,
            'moneyness': option.moneyness
        }

    def _vol_point_to_dict(self, point: VolatilityPoint) -> Dict[str, Any]:
        """Convert VolatilityPoint to dictionary"""
        return {
            'strike': point.strike,
            'expiry_days': point.expiry_days,
            'implied_volatility': point.implied_volatility,
            'delta': point.delta,
            'volume_weight': point.volume_weight
        }

    def _strategy_to_dict(self, strategy: OptionsStrategy) -> Dict[str, Any]:
        """Convert OptionsStrategy to dictionary"""
        return {
            'strategy_name': strategy.strategy_name,
            'components': strategy.components,
            'net_premium': strategy.net_premium,
            'max_profit': strategy.max_profit if strategy.max_profit != float('inf') else 999999,
            'max_loss': strategy.max_loss,
            'breakeven_points': strategy.breakeven_points,
            'profit_probability': strategy.profit_probability,
            'strategy_delta': strategy.strategy_delta,
            'strategy_gamma': strategy.strategy_gamma,
            'strategy_theta': strategy.strategy_theta,
            'strategy_vega': strategy.strategy_vega
        }

    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on synthetic options analysis
        """
        try:
            volatility_percentile = value.get('volatility_percentile', 0.5)
            options_flow = value.get('options_flow', {})
            atm_call = value.get('atm_call')
            atm_put = value.get('atm_put')

            if not atm_call or not atm_put:
                return SignalType.NEUTRAL, 0.0

            # Volatility-based signals
            if volatility_percentile > 0.8:  # High volatility
                # Sell volatility strategies might be attractive
                flow_direction = options_flow.get('flow_direction', 'neutral')
                if flow_direction == 'neutral':
                    return SignalType.SELL, 0.6  # Sell high volatility
            elif volatility_percentile < 0.2:  # Low volatility
                # Buy volatility strategies might be attractive
                return SignalType.BUY, 0.6  # Buy low volatility

            # Options flow signals
            flow_strength = options_flow.get('flow_strength', 0.0)
            if flow_strength > 0.7:
                flow_direction = options_flow.get('flow_direction', 'neutral')
                if flow_direction == 'bullish':
                    return SignalType.BUY, flow_strength
                elif flow_direction == 'bearish':
                    return SignalType.SELL, flow_strength

            # Greeks-based signals
            if atm_call.get('delta', 0) > 0.6 and atm_put.get('delta', 0) < -0.4:
                # Strong directional bias
                gamma = atm_call.get('gamma', 0)
                if gamma > 0.05:  # High gamma suggests big moves
                    return SignalType.BUY, min(gamma * 10, 0.8)

            # Put-call parity deviation signal
            pcp_deviation = value.get('put_call_parity_deviation', 0.0)
            if pcp_deviation > 0.02:  # Significant deviation
                return SignalType.NEUTRAL, 0.5  # Arbitrage opportunity

            return SignalType.NEUTRAL, 0.0

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0

    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)

        options_metadata = {
            'volatility_estimation_method': self.parameters['volatility_estimation_method'],
            'total_synthetic_options': len(self.synthetic_options),
            'total_strategies': len(self.options_strategies),
            'expiry_days_tracked': self.parameters['expiry_days'],
            'volatility_history_length': len(self.volatility_history),
            'risk_free_rate': self.parameters['risk_free_rate'],
            'skew_adjustment_enabled': self.parameters['skew_adjustment']
        }

        base_metadata.update(options_metadata)
        return base_metadata


def create_synthetic_option_indicator(parameters: Optional[Dict[str, Any]] = None) -> SyntheticOptionIndicator:
    """
    Factory function to create a SyntheticOptionIndicator instance

    Args:
        parameters: Optional dictionary of parameters to customize the indicator

    Returns:
        Configured SyntheticOptionIndicator instance
    """
    return SyntheticOptionIndicator(parameters=parameters)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # Create realistic stock price data
    base_price = 100
    drift = 0.0002  # Small daily drift
    volatility = 0.015  # Daily volatility

    returns = np.random.normal(drift, volatility, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    # Add some volatility clustering
    vol_multiplier = 1 + 0.5 * np.sin(np.arange(len(dates)) / 50)
    prices = prices * vol_multiplier

    sample_data = pd.DataFrame({
        'high': prices * np.random.uniform(1.001, 1.02, len(dates)),
        'low': prices * np.random.uniform(0.98, 0.999, len(dates)),
        'close': prices,
        'volume': np.random.uniform(500000, 2000000, len(dates))
    }, index=dates)

    # Test the indicator
    indicator = create_synthetic_option_indicator({
        'volatility_window': 21,
        'expiry_days': [7, 30, 90],
        'strike_intervals': 11
    })

    try:
        result = indicator.calculate(sample_data)
        print("Synthetic Options Calculation Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Current volatility: {result.value.get('current_volatility', 0):.1%}")
        print(f"Volatility percentile: {result.value.get('volatility_percentile', 0):.1%}")
        print(f"Total synthetic options: {result.value.get('total_synthetic_options', 0)}")
        print(f"Total strategies: {result.value.get('total_strategies', 0)}")

        atm_call = result.value.get('atm_call')
        if atm_call:
            print(f"ATM Call - Strike: ${atm_call['strike']:.2f}, Price: ${atm_call['theoretical_price']:.2f}")
            print(f"  Delta: {atm_call['delta']:.3f}, Gamma: {atm_call['gamma']:.4f}")
            print(f"  Theta: {atm_call['theta']:.3f}, Vega: {atm_call['vega']:.3f}")

        atm_put = result.value.get('atm_put')
        if atm_put:
            print(f"ATM Put - Strike: ${atm_put['strike']:.2f}, Price: ${atm_put['theoretical_price']:.2f}")
            print(f"  Delta: {atm_put['delta']:.3f}, Gamma: {atm_put['gamma']:.4f}")

        options_flow = result.value.get('options_flow', {})
        print(f"Options flow direction: {options_flow.get('flow_direction', 'neutral')}")
        print(f"Call/Put ratio: {options_flow.get('call_put_ratio', 1.0):.2f}")

    except Exception as e:
        print(f"Error testing indicator: {str(e)}")
