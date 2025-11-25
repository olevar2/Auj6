"""
Indicator Data Requirements Contract for AUJ Platform

This module defines the exact data requirements for each of the 230+ technical indicators,
mapping them to available data providers (MetaApiProvider, YahooFinanceProvider).

The system is designed for graceful degradation - indicators requiring unavailable
data types (like news, order book) are configured as inactive to prevent system errors.

Core Design Principles:
1. Data-Indicator Specialization: Each indicator knows exactly what data it needs
2. Provider Mapping: Clear mapping to available providers (MetaApi, Yahoo Finance)
3. Graceful Degradation: Unavailable data types result in inactive indicators, not crashes
4. Real-World Adaptation: Focus on price action, volume, and tick data from available sources
"""

from typing import Dict, List, Optional, Set, Literal
from enum import Enum
from dataclasses import dataclass

# Available Data Provider Types (migrated to MetaApi for Linux deployment)
DataProviderType = Literal["MetaApiProvider", "YahooFinanceProvider", "NewsProvider", "OrderBookProvider"]

# Available Data Types from our providers
class DataType(Enum):
    OHLCV = "ohlcv"  # Open, High, Low, Close, Volume
    TICK = "tick"    # Tick-by-tick price data
    BID_ASK = "bid_ask"  # Bid/Ask spreads
    NEWS = "news"    # News sentiment (UNAVAILABLE)
    ORDER_BOOK = "order_book"  # Order book depth (UNAVAILABLE)
    VOLUME_PROFILE = "volume_profile"  # Volume at price levels

# Required Data Columns for different data types
class RequiredColumns(Enum):
    BASIC_OHLCV = ["open", "high", "low", "close", "volume"]
    EXTENDED_OHLCV = ["open", "high", "low", "close", "volume", "tick_volume", "spread"]
    TICK_DATA = ["price", "volume", "time", "bid", "ask"]
    BID_ASK_DATA = ["bid", "ask", "spread", "time"]
    NEWS_DATA = ["sentiment", "impact", "title", "time"]  # UNAVAILABLE
    ORDER_BOOK_DATA = ["bid_levels", "ask_levels", "depth"]  # UNAVAILABLE

@dataclass
class IndicatorDataRequirement:
    """Defines data requirements for a specific indicator"""
    indicator_name: str
    primary_data_type: DataType
    required_columns: List[str]
    available_providers: List[DataProviderType]
    fallback_providers: List[DataProviderType]
    is_active: bool  # False if no available providers
    minimum_periods: int = 1
    additional_requirements: Optional[Dict] = None

# Main Indicator Requirements Mapping
INDICATOR_DATA_REQUIREMENTS: Dict[str, IndicatorDataRequirement] = {

    # ==================== MOMENTUM INDICATORS ====================

    "awesome_oscillator_indicator": IndicatorDataRequirement(
        indicator_name="awesome_oscillator_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=34
    ),

    "macd_indicator": IndicatorDataRequirement(
        indicator_name="macd_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=26
    ),


    "rate_of_change_indicator": IndicatorDataRequirement(
        indicator_name="rate_of_change_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=12
    ),

    "rsi_indicator": IndicatorDataRequirement(
        indicator_name="rsi_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),


    "stochastic_rsi_indicator": IndicatorDataRequirement(
        indicator_name="stochastic_rsi_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=21
    ),







    # ==================== TREND INDICATORS ====================
    "adx_indicator": IndicatorDataRequirement(
        indicator_name="adx_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "super_trend_indicator": IndicatorDataRequirement(
        indicator_name="super_trend_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=10
    ),

    "trend_direction_indicator": IndicatorDataRequirement(
        indicator_name="trend_direction_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "trend_following_system_indicator": IndicatorDataRequirement(
        indicator_name="trend_following_system_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "trend_strength_indicator": IndicatorDataRequirement(
        indicator_name="trend_strength_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "vortex_indicator": IndicatorDataRequirement(
        indicator_name="vortex_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "parabolic_sar_indicator": IndicatorDataRequirement(
        indicator_name="parabolic_sar_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=2
    ),

    "directional_movement_index_indicator": IndicatorDataRequirement(
        indicator_name="directional_movement_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "directional_movement_system_indicator": IndicatorDataRequirement(
        indicator_name="directional_movement_system_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "aroon_oscillator_indicator": IndicatorDataRequirement(
        indicator_name="aroon_oscillator_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "aroon_indicator": IndicatorDataRequirement(
        indicator_name="aroon_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),
    # ==================== VOLATILITY INDICATORS ====================
    "average_true_range_indicator": IndicatorDataRequirement(
        indicator_name="average_true_range_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "bollinger_bands_indicator": IndicatorDataRequirement(
        indicator_name="bollinger_bands_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "keltner_channel_indicator": IndicatorDataRequirement(
        indicator_name="keltner_channel_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "historical_volatility_indicator": IndicatorDataRequirement(
        indicator_name="historical_volatility_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "chaikin_volatility_indicator": IndicatorDataRequirement(
        indicator_name="chaikin_volatility_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=10
    ),

    "relative_volatility_index_indicator": IndicatorDataRequirement(
        indicator_name="relative_volatility_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "donchian_channels_indicator": IndicatorDataRequirement(
        indicator_name="donchian_channels_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),


    "standard_deviation_channels_indicator": IndicatorDataRequirement(
        indicator_name="standard_deviation_channels_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    # ==================== VOLUME INDICATORS ====================
    "accumulation_distribution_line_indicator": IndicatorDataRequirement(
        indicator_name="accumulation_distribution_line_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=1
    ),

    "anchored_vwap_indicator": IndicatorDataRequirement(
        indicator_name="anchored_vwap_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=1
    ),

    "vwap_indicator": IndicatorDataRequirement(
        indicator_name="vwap_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=1
    ),

    "on_balance_volume_indicator": IndicatorDataRequirement(
        indicator_name="on_balance_volume_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=1
    ),

    "volume_profile_indicator": IndicatorDataRequirement(
        indicator_name="volume_profile_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "chaikin_money_flow_indicator": IndicatorDataRequirement(
        indicator_name="chaikin_money_flow_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "ease_of_movement_indicator": IndicatorDataRequirement(
        indicator_name="ease_of_movement_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "price_volume_trend_indicator": IndicatorDataRequirement(
        indicator_name="price_volume_trend_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=1
    ),

    "volume_oscillator_indicator": IndicatorDataRequirement(
        indicator_name="volume_oscillator_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "volume_rate_of_change_indicator": IndicatorDataRequirement(
        indicator_name="volume_rate_of_change_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=12
    ),

    "negative_volume_index_indicator": IndicatorDataRequirement(
        indicator_name="negative_volume_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=1
    ),

    "positive_volume_index_indicator": IndicatorDataRequirement(
        indicator_name="positive_volume_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=1
    ),

    "klinger_oscillator_indicator": IndicatorDataRequirement(
        indicator_name="klinger_oscillator_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=34
    ),


    # ==================== MOVING AVERAGE INDICATORS ====================
    "simple_moving_average_indicator": IndicatorDataRequirement(
        indicator_name="simple_moving_average_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "exponential_moving_average_indicator": IndicatorDataRequirement(
        indicator_name="exponential_moving_average_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=12
    ),


    "hull_moving_average_indicator": IndicatorDataRequirement(
        indicator_name="hull_moving_average_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "kaufman_adaptive_moving_average_indicator": IndicatorDataRequirement(
        indicator_name="kaufman_adaptive_moving_average_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=10
    ),

    "triple_ema_indicator": IndicatorDataRequirement(
        indicator_name="triple_ema_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=21
    ),

    "zero_lag_ema_indicator": IndicatorDataRequirement(
        indicator_name="zero_lag_ema_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "alligator_indicator": IndicatorDataRequirement(
        indicator_name="alligator_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=21
    ),
    # ==================== PATTERN RECOGNITION INDICATORS ====================



    "doji_indicator": IndicatorDataRequirement(
        indicator_name="doji_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["open", "high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=1
    ),



    "engulfing_pattern_indicator": IndicatorDataRequirement(
        indicator_name="engulfing_pattern_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["open", "high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=2
    ),



    "hammer_indicator": IndicatorDataRequirement(
        indicator_name="hammer_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["open", "high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=1
    ),




    "head_and_shoulders_indicator": IndicatorDataRequirement(
        indicator_name="head_and_shoulders_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "triangle_pattern_indicator": IndicatorDataRequirement(
        indicator_name="triangle_pattern_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=15
    ),


    # ==================== FIBONACCI INDICATORS ====================
    "fibonacci_retracement_indicator": IndicatorDataRequirement(
        indicator_name="fibonacci_retracement_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "fibonacci_extension_indicator": IndicatorDataRequirement(
        indicator_name="fibonacci_extension_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),




    # ==================== GANN INDICATORS ====================
    "gann_angles_indicator": IndicatorDataRequirement(
        indicator_name="gann_angles_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=30
    ),

    "gann_fan_indicator": IndicatorDataRequirement(
        indicator_name="gann_fan_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "gann_square_of_nine_indicator": IndicatorDataRequirement(
        indicator_name="gann_square_of_nine_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=9
    ),

    # ==================== STATISTICAL INDICATORS ====================
    "zscore_indicator": IndicatorDataRequirement(
        indicator_name="zscore_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "skewness_indicator": IndicatorDataRequirement(
        indicator_name="skewness_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "variance_ratio_indicator": IndicatorDataRequirement(
        indicator_name="variance_ratio_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=30
    ),

    "autocorrelation_indicator": IndicatorDataRequirement(
        indicator_name="autocorrelation_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "hurst_exponent_indicator": IndicatorDataRequirement(
        indicator_name="hurst_exponent_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=50
    ),

    "beta_coefficient_indicator": IndicatorDataRequirement(
        indicator_name="beta_coefficient_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=30,
        additional_requirements={"benchmark_data": "required"}
    ),

    # ==================== CORRELATION INDICATORS ====================
    "correlation_matrix_indicator": IndicatorDataRequirement(
        indicator_name="correlation_matrix_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20,
        additional_requirements={"multiple_symbols": "required"}
    ),

    "correlation_analysis_indicator": IndicatorDataRequirement(
        indicator_name="correlation_analysis_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20,
        additional_requirements={"multiple_symbols": "required"}
    ),

    "cointegration_indicator": IndicatorDataRequirement(
        indicator_name="cointegration_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=50,
        additional_requirements={"multiple_symbols": "required"}
    ),
    # ==================== TICK DATA INDICATORS (MetaApi Specialized) ====================
    "bid_ask_spread_analyzer_indicator": IndicatorDataRequirement(
        indicator_name="bid_ask_spread_analyzer_indicator",
        primary_data_type=DataType.TICK,
        required_columns=["bid", "ask", "time"],
        available_providers=["MetaApiProvider", "MetaApiProvider"],
        fallback_providers=[],
        is_active=True,
        minimum_periods=100
    ),

    "order_flow_imbalance_indicator": IndicatorDataRequirement(
        indicator_name="order_flow_imbalance_indicator",
        primary_data_type=DataType.TICK,
        required_columns=["price", "volume", "time"],
        available_providers=["MetaApiProvider", "MetaApiProvider"],
        fallback_providers=[],
        is_active=True,
        minimum_periods=50
    ),

    "tick_volume_analyzer": IndicatorDataRequirement(
        indicator_name="tick_volume_analyzer",
        primary_data_type=DataType.TICK,
        required_columns=["price", "volume", "time"],
        available_providers=["MetaApiProvider", "MetaApiProvider"],
        fallback_providers=[],
        is_active=True,
        minimum_periods=100
    ),

    "institutional_flow_detector": IndicatorDataRequirement(
        indicator_name="institutional_flow_detector",
        primary_data_type=DataType.TICK,
        required_columns=["price", "volume", "time"],
        available_providers=["MetaApiProvider", "MetaApiProvider"],
        fallback_providers=[],
        is_active=True,
        minimum_periods=200
    ),

    "liquidity_flow_indicator": IndicatorDataRequirement(
        indicator_name="liquidity_flow_indicator",
        primary_data_type=DataType.TICK,
        required_columns=["bid", "ask", "volume", "time"],
        available_providers=["MetaApiProvider", "MetaApiProvider"],
        fallback_providers=[],
        is_active=True,
        minimum_periods=100
    ),

    # ==================== AI/ML INDICATORS (DecisionMaster Focused) ====================
    "adaptive_indicators": IndicatorDataRequirement(
        indicator_name="adaptive_indicators",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=100
    ),

    "neural_network_predictor_indicator": IndicatorDataRequirement(
        indicator_name="neural_network_predictor_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["open", "high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=200
    ),

    "lstm_price_predictor_indicator": IndicatorDataRequirement(
        indicator_name="lstm_price_predictor_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["open", "high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=250
    ),

    "genetic_algorithm_optimizer_indicator": IndicatorDataRequirement(
        indicator_name="genetic_algorithm_optimizer_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["open", "high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=300
    ),

    # ==================== UNAVAILABLE DATA INDICATORS (INACTIVE) ====================
    "news_article_indicator": IndicatorDataRequirement(
        indicator_name="news_article_indicator",
        primary_data_type=DataType.NEWS,
        required_columns=["sentiment", "impact", "title", "time"],
        available_providers=[],
        fallback_providers=[],
        is_active=False,  # No news provider available
        minimum_periods=1
    ),

    "social_media_post_indicator": IndicatorDataRequirement(
        indicator_name="social_media_post_indicator",
        primary_data_type=DataType.NEWS,
        required_columns=["sentiment", "volume", "platform"],
        available_providers=[],
        fallback_providers=[],
        is_active=False,  # No social media provider available
        minimum_periods=1
    ),

    "sentiment_integration_indicator": IndicatorDataRequirement(
        indicator_name="sentiment_integration_indicator",
        primary_data_type=DataType.NEWS,
        required_columns=["sentiment", "impact"],
        available_providers=[],
        fallback_providers=[],
        is_active=False,  # No news provider available
        minimum_periods=1
    ),

    # ==================== ICHIMOKU SYSTEM ====================
    "ichimoku_indicator": IndicatorDataRequirement(
        indicator_name="ichimoku_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=52
    ),

    "ichimoku_kinko_hyo_indicator": IndicatorDataRequirement(
        indicator_name="ichimoku_kinko_hyo_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=52
    ),

    "cloud_position_indicator": IndicatorDataRequirement(
        indicator_name="cloud_position_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=52
    ),

    # ==================== OSCILLATOR INDICATORS ====================
    "commodity_channel_index_indicator": IndicatorDataRequirement(
        indicator_name="commodity_channel_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "money_flow_index_indicator": IndicatorDataRequirement(
        indicator_name="money_flow_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    "fisher_transform_indicator": IndicatorDataRequirement(
        indicator_name="fisher_transform_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=10
    ),


    # ==================== FRACTAL INDICATORS ====================



    # ==================== ELLIOTT WAVE INDICATORS ====================
    "elliott_wave_oscillator_indicator": IndicatorDataRequirement(
        indicator_name="elliott_wave_oscillator_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=34
    ),


    # ==================== SUPPORT/RESISTANCE INDICATORS ====================

    "central_pivot_range_indicator": IndicatorDataRequirement(
        indicator_name="central_pivot_range_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=1
    ),

    # ==================== MISCELLANEOUS INDICATORS ====================

    "force_index_indicator": IndicatorDataRequirement(
        indicator_name="force_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=13
    ),

    "mass_index_indicator": IndicatorDataRequirement(
        indicator_name="mass_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=25
    ),

    "ulcer_index_indicator": IndicatorDataRequirement(
        indicator_name="ulcer_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=14
    ),

    # ==================== RENAMED INDICATORS (Duplicates Resolution) ====================
    "ai_commodity_channel_index_indicator": IndicatorDataRequirement(
        indicator_name="ai_commodity_channel_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),


    "volume_mass_index_indicator": IndicatorDataRequirement(
        indicator_name="volume_mass_index_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=25
    ),

    # ==================== NEW INDICATORS - 10 MISSING IMPLEMENTATIONS ====================
    "fibonacci_expansion_indicator": IndicatorDataRequirement(
        indicator_name="fibonacci_expansion_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20
    ),

    "fibonacci_grid_indicator": IndicatorDataRequirement(
        indicator_name="fibonacci_grid_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["high", "low", "close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=30
    ),

    "intermarket_correlation_indicator": IndicatorDataRequirement(
        indicator_name="intermarket_correlation_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=50,
        additional_requirements={"multiple_symbols": "required"}
    ),

    "market_breadth_indicator": IndicatorDataRequirement(
        indicator_name="market_breadth_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=20,
        additional_requirements={"multiple_symbols": "required"}
    ),

    "sector_momentum_indicator": IndicatorDataRequirement(
        indicator_name="sector_momentum_indicator",
        primary_data_type=DataType.OHLCV,
        required_columns=["close", "volume"],
        available_providers=["MetaApiProvider", "MetaApiProvider", "YahooFinanceProvider"],
        fallback_providers=["YahooFinanceProvider"],
        is_active=True,
        minimum_periods=30,
        additional_requirements={"sector_data": "required"}
    ),





}

# Provider Capability Mapping
PROVIDER_CAPABILITIES: Dict[str, Dict] = {
    "MetaApiProvider": {
        "supported_data_types": [DataType.OHLCV, DataType.TICK, DataType.BID_ASK],
        "real_time": True,
        "historical": True,
        "tick_data": True,
        "volume_data": True,
        "spread_data": True,
        "priority": 1,  # Highest priority for Linux deployment
        "status": "ACTIVE",
        "platform_compatibility": ["linux", "windows", "macos"]
    },

    "MetaApiProvider": {
        "supported_data_types": [DataType.OHLCV],
        "real_time": True,
        "historical": True,
        "tick_data": False,
        "volume_data": True,
        "spread_data": True,
        "priority": 2,  # Lower priority, Windows fallback
        "status": "LEGACY",
        "platform_compatibility": ["windows"]
    },

    "MetaApiProvider": {
        "supported_data_types": [DataType.TICK, DataType.BID_ASK],
        "real_time": True,
        "historical": True,
        "tick_data": True,
        "volume_data": True,
        "spread_data": True,
        "priority": 2,  # Lower priority, Windows fallback
        "status": "LEGACY",
        "platform_compatibility": ["windows"]
    },

    "YahooFinanceProvider": {
        "supported_data_types": [DataType.OHLCV],
        "real_time": False,
        "historical": True,
        "tick_data": False,
        "volume_data": True,
        "spread_data": False,
        "priority": 3,  # Fallback provider
        "status": "ACTIVE",
        "platform_compatibility": ["linux", "windows", "macos"]
    },

    "NewsProvider": {
        "supported_data_types": [DataType.NEWS],
        "real_time": False,
        "historical": False,
        "tick_data": False,
        "volume_data": False,
        "spread_data": False,
        "priority": 999,  # Unavailable
        "status": "INACTIVE"
    },

    "OrderBookProvider": {
        "supported_data_types": [DataType.ORDER_BOOK],
        "real_time": False,
        "historical": False,
        "tick_data": False,
        "volume_data": False,
        "spread_data": False,
        "priority": 999,  # Unavailable
        "status": "INACTIVE"
    }
}

# Utility Functions for Indicator Data Requirements
def get_active_indicators() -> List[str]:
    """Get list of all active indicators that have available data providers"""
    return [name for name, req in INDICATOR_DATA_REQUIREMENTS.items() if req.is_active]

def get_inactive_indicators() -> List[str]:
    """Get list of inactive indicators (no available data providers)"""
    return [name for name, req in INDICATOR_DATA_REQUIREMENTS.items() if not req.is_active]

def get_indicators_by_provider(provider_name: str) -> List[str]:
    """Get list of indicators that can be served by a specific provider"""
    return [
        name for name, req in INDICATOR_DATA_REQUIREMENTS.items()
        if provider_name in req.available_providers and req.is_active
    ]

def get_indicators_by_data_type(data_type: DataType) -> List[str]:
    """Get list of indicators that require a specific data type"""
    return [
        name for name, req in INDICATOR_DATA_REQUIREMENTS.items()
        if req.primary_data_type == data_type and req.is_active
    ]

def validate_indicator_requirements(indicator_name: str, available_providers: List[str]) -> bool:
    """Validate if an indicator can be calculated with available providers"""
    if indicator_name not in INDICATOR_DATA_REQUIREMENTS:
        return False

    req = INDICATOR_DATA_REQUIREMENTS[indicator_name]
    if not req.is_active:
        return False

    # Check if at least one required provider is available
    return any(provider in available_providers for provider in req.available_providers)

def get_provider_priority_for_indicator(indicator_name: str) -> List[str]:
    """Get providers for an indicator sorted by priority (highest first)"""
    if indicator_name not in INDICATOR_DATA_REQUIREMENTS:
        return []

    req = INDICATOR_DATA_REQUIREMENTS[indicator_name]
    if not req.is_active:
        return []

    # Sort providers by priority from PROVIDER_CAPABILITIES
    return sorted(
        req.available_providers,
        key=lambda p: PROVIDER_CAPABILITIES.get(p, {}).get("priority", 999)
    )

# Data Requirements Summary
REQUIREMENTS_SUMMARY = {
    "total_indicators": len(INDICATOR_DATA_REQUIREMENTS),
    "active_indicators": len(get_active_indicators()),
    "inactive_indicators": len(get_inactive_indicators()),
    "metaapi_indicators": len(get_indicators_by_provider("MetaApiProvider")),
    "yahoo_indicators": len(get_indicators_by_provider("YahooFinanceProvider")),
    "tick_only_indicators": len([
        name for name, req in INDICATOR_DATA_REQUIREMENTS.items()
        if req.primary_data_type == DataType.TICK and req.is_active
    ])
}
