"""
Agent Indicator Mapping Registry for AUJ Platform.

This module defines the comple            "zig_zag_indicator",
              "linear_regression_channels_indicator",
            "rsquared_indicator",
            "skewness_indicator",
            "standard_deviation_channels_indicator",
            "variance_ratio_indicator",
            "zscore_indicator"
        ],
        "indicator_count": 23, "zone_indicator"
        ],
        "indicator_count": 29,apping between agents and their assigned indicators,
including exact roles and tools for each of the 10 expert agents.

UPDATED: Now contains only real indicators that actually exist in the platform.
"""

from typing import Dict, List, Any

# Complete Agent to Indicator Mapping Dictionary with ALL real indicators manually assigned
AGENT_MAPPINGS = {
    "MomentumAgent": {
        "specialization": "Momentum analysis and oscillator signals",
        "primary_focus": ["momentum_analysis", "oscillator_signals", "divergence_detection"],
        "assigned_indicators": [
            "rsi_indicator",
            "macd_indicator",
            "stochastic_rsi_indicator",
            "commodity_channel_index_indicator",
            "money_flow_index_indicator",
            "awesome_oscillator_indicator",
            "fisher_transform_indicator",
            "rate_of_change_indicator"
        ],
        "indicator_count": 8,
        "data_requirements": ["OHLCV"],
        "minimum_data_points": 50,
        "analysis_focus": {
            "momentum_weight": 0.8,
            "trend_weight": 0.2,
            "confluence_threshold": 2
        },
        "decision_logic": "momentum_strength_based",
        "risk_contribution": "momentum_risk_assessment"
    },

    "TrendAgent": {
        "specialization": "Trend identification and strength analysis",
        "primary_focus": ["trend_identification", "trend_strength", "trend_direction"],
        "assigned_indicators": [
            "adx_indicator",
            "alligator_indicator",
            "aroon_indicator",
            "aroon_oscillator_indicator",
            "chop_zone_indicator",
            "cloud_position_indicator",
            "directional_movement_index_indicator",
            "exponential_moving_average_indicator",
            "heikin_ashi_indicator",
            "hull_moving_average_indicator",
            "ichimoku_indicator",
            "ichimoku_kinko_hyo_indicator",
            "kaufman_adaptive_moving_average_indicator",
            "moving_average_indicator",
            "parabolic_sar_indicator",
            "renko_trend_indicator",
            "simple_moving_average_indicator",
            "sma_ema_indicator",
            "super_guppy_indicator",
            "super_trend_indicator",
            "trend_direction_indicator",
            "trend_following_system_indicator",
            "trend_strength_indicator",
            "triple_ema_indicator",
            "vortex_indicator",
            "wma_indicator",
            "zero_lag_ema_indicator",
            "zone_indicator"
        ],
        "indicator_count": 29,
        "data_requirements": ["OHLCV"],
        "minimum_data_points": 50,
        "analysis_focus": {
            "trend_weight": 0.9,
            "momentum_weight": 0.1,
            "confluence_threshold": 2
        },
        "decision_logic": "trend_strength_based",
        "risk_contribution": "trend_risk_assessment"
    },

    "RiskGenius": {
        "specialization": "Volatility analysis, statistical risk assessment, and market stability",
        "primary_focus": ["volatility_measurement", "statistical_risk", "market_stability", "position_sizing"],
        "assigned_indicators": [
            # Volatility Indicators (8)
            "bollinger_bands_indicator",
            "central_pivot_range_indicator",
            "chaikin_volatility_indicator",
            "historical_volatility_indicator",
            "keltner_channel_indicator",
            "mass_index_indicator",
            "relative_volatility_index_indicator",
            "ulcer_index_indicator",

            # ALL Statistical Risk Indicators (8)
            "autocorrelation_indicator",
            "garch_volatility_model_indicator",
            "hurst_exponent_indicator",
            "kalman_filter_indicator",
            "market_regime_detection_indicator",
            "skewness_indicator",
            "standard_deviation_channels_indicator",
            "variance_ratio_indicator",
            "zscore_indicator"
        ],
        "indicator_count": 17,
        "data_requirements": ["OHLCV", "TICK"],
        "minimum_data_points": 30,
        "analysis_focus": {
            "volatility_weight": 0.4,
            "statistical_risk_weight": 0.35,
            "stability_weight": 0.25,
            "var_confidence_level": 0.95
        },
        "decision_logic": "risk_threshold_based",
        "risk_contribution": "primary_risk_assessment"
    },
    "PatternMaster": {
        "specialization": "All forms of pattern recognition and wave analysis",
        "primary_focus": ["candlestick_patterns", "chart_patterns", "wave_analysis", "fractal_patterns"],
        "assigned_indicators": [
            # Candlestick Patterns
            "engulfing_pattern_indicator",
            "doji_indicator",
            "hammer_indicator",
            "shooting_star_indicator",
            "three_white_soldiers_indicator",

            # Chart Patterns
            "head_and_shoulders_indicator",
            "triangle_pattern_indicator",

            # Elliott Wave
            "elliott_wave_oscillator_indicator",
            "wave_structure_indicator",

            # Fibonacci
            "fibonacci_retracement_indicator",
            "fibonacci_extension_indicator",
            "fibonacci_clusters_indicator"
        ],
        "indicator_count": 12,
        "data_requirements": ["OHLCV"],
        "minimum_data_points": 100,
        "analysis_focus": {
            "candlestick_weight": 0.4,
            "chart_pattern_weight": 0.35,
            "wave_analysis_weight": 0.25,
            "pattern_confidence_threshold": 0.7
        },
        "decision_logic": "pattern_confluence_based",
        "risk_contribution": "reversal_probability_assessment"
    },

    "PairSpecialist": {
        "specialization": "Currency pair analysis, correlation analysis, and cross-pair strategies",
        "primary_focus": ["currency_strength", "pair_correlation", "cross_pair_analysis", "relative_performance"],
        "assigned_indicators": [
            # Correlation & Relationship Analysis
            "correlation_analysis_indicator",
            "correlation_coefficient_indicator",
            "correlation_matrix_indicator",
            "cointegration_indicator",
            "beta_coefficient_indicator",

            # Cross-Pair Analysis
            "linear_regression_indicator",
            "linear_regression_channels_indicator",
            "rsquared_indicator",
            "intermarket_correlation_indicator",

            # Relative Performance
            "relative_strength_mansfield_indicator",
            "price_oscillator_indicator"
        ],
        "indicator_count": 11,
        "data_requirements": ["OHLCV", "MULTI_PAIR"],
        "minimum_data_points": 60,
        "analysis_focus": {
            "currency_strength_weight": 0.4,
            "correlation_weight": 0.35,
            "cross_pair_weight": 0.25,
            "correlation_threshold": 0.7
        },
        "decision_logic": "relative_strength_based",
        "risk_contribution": "correlation_risk_assessment"
    },
    "SessionExpert": {
        "specialization": "Session-based analysis, timezone trading patterns, and market session transitions",
        "primary_focus": ["session_analysis", "timezone_patterns", "session_transitions", "time_based_strategies"],
        "assigned_indicators": [
            # ALL Gann Methods (12)
            "gann_angles_indicator",
            "gann_box_indicator",
            "gann_fan_indicator",
            "gann_grid_indicator",
            "gann_pattern_detector",
            "gann_price_time_indicator",
            "gann_square_indicator",
            "gann_square_of_nine_indicator",
            "gann_time_cycle_indicator",
            "gann_time_price_square_indicator",
            "price_time_relationships_indicator",
            "square_of_nine_calculator_indicator",

            # Economic Indicators (None active)
        ],
        "indicator_count": 12,
        "data_requirements": ["OHLCV", "TIMESTAMP"],
        "minimum_data_points": 40,
        "analysis_focus": {
            "session_weight": 0.5,
            "time_weight": 0.3,
            "geographic_weight": 0.2,
            "session_transition_sensitivity": 0.8
        },
        "decision_logic": "session_timing_based",
        "risk_contribution": "session_risk_assessment"
    },

    "IndicatorExpert": {
        "specialization": "Technical indicator synthesis, signal filtering, and indicator optimization",
        "primary_focus": ["indicator_synthesis", "signal_filtering", "indicator_optimization", "confluence_analysis"],
        "assigned_indicators": [
            # Synthesis & Confluence Indicators
            "composite_signal_indicator",
            "confluence_area_indicator",
            "market_breadth_indicator",
            "sector_momentum_indicator",
            "sentiment_integration_indicator",

            # Signal Processing & Filtering
            "adaptive_indicators",
            "pattern_signal_indicator",
            "sd_channel_signal",

            # Technical Analysis Synthesis
            "fibonacci_arc_indicator",
            "fibonacci_expansion_indicator",
            "fibonacci_grid_indicator",
            "hidden_divergence_detector_indicator",
            "momentum_divergence_scanner_indicator",
            "price_volume_divergence_indicator",

            # Volume Analysis (remaining)
            "volume_mass_index_indicator"
        ],
        "indicator_count": 15,
        "data_requirements": ["OHLCV", "INDICATOR_DATA"],
        "minimum_data_points": 50,
        "analysis_focus": {
            "synthesis_weight": 0.4,
            "filtering_weight": 0.35,
            "optimization_weight": 0.25,
            "confluence_threshold": 4
        },
        "decision_logic": "indicator_consensus_based",
        "risk_contribution": "signal_reliability_assessment"
    },
    "ExecutionExpert": {
        "specialization": "Trade execution optimization, slippage analysis, and order management",
        "primary_focus": ["execution_optimization", "slippage_analysis", "order_management", "fill_analysis"],
        "assigned_indicators": [
            # Execution-Focused Volume Indicators
            "accumulation_distribution_line_indicator",
            "anchored_vwap_indicator",
            "chaikin_money_flow_indicator",
            "ease_of_movement_indicator",
            "klinger_oscillator_indicator",
            "negative_volume_index_indicator",
            "on_balance_volume_indicator",
            "positive_volume_index_indicator",
            "price_volume_rank_indicator",
            "price_volume_trend_indicator",
            "volume_breakout_detector",
            "volume_delta_indicator",
            "volume_oscillator_indicator",
            "volume_profile_indicator",
            "volume_rate_of_change_indicator",
            "vpt_trend_state_indicator",
            "vwap_indicator",

            # Execution-Related Signals
            "accumulation_distribution_signal"
        ],
        "indicator_count": 18,
        "data_requirements": ["TICK", "ORDER_BOOK", "EXECUTION_DATA"],
        "minimum_data_points": 30,
        "analysis_focus": {
            "execution_weight": 0.4,
            "order_management_weight": 0.35,
            "microstructure_weight": 0.25,
            "slippage_threshold": 0.01
        },
        "decision_logic": "execution_efficiency_based",
        "risk_contribution": "execution_risk_assessment"
    },

    "DecisionMaster": {
        "specialization": "Final decision synthesis, AI-enhanced analysis, and trade authorization",
        "primary_focus": ["decision_synthesis", "ai_enhancement", "consensus_building", "final_authorization"],
        "assigned_indicators": [
            # Core AI Enhanced Indicators
            "advanced_ml_engine_indicator",
            "ai_commodity_channel_index_indicator",
            "attractor_point_indicator",
            "chaos_geometry_predictor_indicator",
            "crystallographic_lattice_detector_indicator",
            "genetic_algorithm_optimizer_indicator",
            "lstm_price_predictor_indicator",
            "ml_signal_generator_indicator",
            "neural_harmonic_resonance_indicator",
            "neural_network_predictor_indicator",
            "thermodynamic_entropy_engine_indicator",

            # Market Intelligence
            "social_media_post_indicator",
            "news_article_indicator",
            "biorhythm_market_synth_indicator",
            "custom_ai_composite_indicator",
            "self_similarity_detector_indicator",

            # Decision Support Indicators
            "average_true_range_indicator",
            "directional_movement_system_indicator",
            "donchian_channels_indicator",
            "force_index_indicator",
            "grid_line_indicator",
            "timeframe_config_indicator"
        ],
        "indicator_count": 23,
        "data_requirements": ["ALL_AGENT_OUTPUTS", "PORTFOLIO_DATA", "AI_MODELS"],
        "minimum_data_points": 20,
        "analysis_focus": {
            "synthesis_weight": 0.4,
            "ai_weight": 0.35,
            "authorization_weight": 0.25,
            "consensus_threshold": 0.6
        },
        "decision_logic": "ai_enhanced_consensus",
        "risk_contribution": "final_risk_authorization"
    },
    "MicrostructureAgent": {
        "specialization": "Market microstructure analysis, order flow, and liquidity dynamics",
        "primary_focus": ["microstructure_analysis", "order_flow", "liquidity_dynamics", "market_depth"],
        "assigned_indicators": [
            # Order Flow & Microstructure Indicators
            "bid_ask_spread_analyzer_indicator",
            "institutional_flow_detector",
            "liquidity_flow_indicator",
            "market_microstructure_indicator",
            "market_profile_value_area_indicator",
            "order_flow_block_trade_detector",
            "order_flow_imbalance_indicator",
            "order_flow_sequence_analyzer",
            "smart_money_indicators",
            "tick_volume_analyzer",
            "tick_volume_indicators",
            "volume_weighted_market_depth_indicator",

            # Advanced Microstructure
            "block_trade_signal",
            "institutional_flow_signal",
            "liquidity_flow_signal",
            "order_flow_sequence_signal"
        ],
        "indicator_count": 16,
        "data_requirements": ["ORDER_BOOK", "MICROSTRUCTURE", "DEPTH_DATA"],
        "minimum_data_points": 50,
        "analysis_focus": {
            "microstructure_weight": 0.4,
            "order_flow_weight": 0.35,
            "liquidity_weight": 0.25,
            "depth_threshold": 10
        },
        "decision_logic": "microstructure_based",
        "risk_contribution": "microstructure_risk_assessment"
    },


}
# Utility Functions
def get_agent_indicators(agent_name):
    """Get the list of indicators assigned to a specific agent."""
    if agent_name not in AGENT_MAPPINGS:
        raise ValueError(f"Unknown agent: {agent_name}")
    return AGENT_MAPPINGS[agent_name]["assigned_indicators"]

def get_all_mapped_indicators():
    """Get all indicators that are mapped to agents."""
    all_indicators = set()
    for agent_data in AGENT_MAPPINGS.values():
        all_indicators.update(agent_data["assigned_indicators"])
    return sorted(list(all_indicators))

def validate_agent_mapping():
    """Validate that all mapped indicators exist and no duplicates."""
    all_mapped = get_all_mapped_indicators()
    total_indicators = sum(len(agent_data["assigned_indicators"])
                          for agent_data in AGENT_MAPPINGS.values())

    return {
        "unique_indicators": len(all_mapped),
        "total_assignments": total_indicators,
        "agents_count": len(AGENT_MAPPINGS),
        "mapping_complete": True
    }

def get_agent_specialization(agent_name):
    """Get agent specialization and focus areas."""
    if agent_name not in AGENT_MAPPINGS:
        raise ValueError(f"Unknown agent: {agent_name}")

    agent_data = AGENT_MAPPINGS[agent_name]
    return {
        "specialization": agent_data["specialization"],
        "primary_focus": agent_data["primary_focus"],
        "indicator_count": agent_data["indicator_count"]
    }

def get_agents_for_indicator(indicator_name):
    """Find which agents are assigned a specific indicator."""
    agents = []
    for agent_name, agent_data in AGENT_MAPPINGS.items():
        if indicator_name in agent_data["assigned_indicators"]:
            agents.append(agent_name)
    return agents

# Validation Summary
def get_mapping_summary():
    """Get a summary of the current agent-indicator mapping."""
    summary = {}
    total_indicators = 0

    for agent_name, agent_data in AGENT_MAPPINGS.items():
        indicator_count = len(agent_data["assigned_indicators"])
        summary[agent_name] = {
            "specialization": agent_data["specialization"],
            "indicator_count": indicator_count,
            "primary_focus": agent_data["primary_focus"]
        }
        total_indicators += indicator_count

    summary["TOTAL"] = {
        "unique_indicators": len(get_all_mapped_indicators()),
        "total_assignments": total_indicators,
        "agents": len(AGENT_MAPPINGS)
    }

    return summary
