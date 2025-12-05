"""
Opportunity Radar Configuration for AUJ Platform

This module provides configuration settings for the intelligent pair selection
system that scans all trading pairs and identifies the best opportunities.

✨ CROWN JEWEL COMPONENT: OpportunityRadar
"""

from typing import Dict, List


# ============================================================================
# OPPORTUNITY RADAR CONFIGURATION
# ============================================================================

OPPORTUNITY_RADAR_CONFIG = {
    # Enable/disable intelligent selection (set False to use round-robin)
    "enabled": True,
    
    # Pair scope: "all" (25 pairs), "majors" (7), "active" (12 most liquid)
    "pair_scope": "active",
    
    # Number of top pairs to deep analyze with 10 agents
    "top_n": 3,
    
    # Minimum opportunity score to consider (0-100)
    "min_score_threshold": 45.0,
    
    # Maximum correlation with existing positions (0-1)
    "max_correlation": 0.7,
    
    # Quick scan timeout per pair (seconds)
    "quick_scan_timeout": 10,
    
    # Score weights for opportunity calculation
    "score_weights": {
        "trend_clarity": 0.25,      # ADX-based trend strength
        "momentum_strength": 0.20,  # RSI-based momentum
        "entry_quality": 0.20,      # Distance from MA, pullback quality
        "regime_suitability": 0.20, # How well current regime fits strategy
        "volatility_fit": 0.15      # ATR in optimal range
    },
    
    # Grade thresholds (opportunity score ranges)
    "grade_thresholds": {
        "A_PLUS": 85,  # Exceptional opportunity
        "A": 75,       # Excellent
        "B": 60,       # Good
        "C": 45,       # Average (minimum tradeable)
        "D": 0         # Poor (skip)
    }
}


# ============================================================================
# TRADING PAIRS FOR RADAR SCANNING
# ============================================================================

# Major pairs (7) - most liquid
MAJOR_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
    'AUDUSD', 'USDCAD', 'NZDUSD'
]

# Active pairs (12) - majors + most active crosses + metals
ACTIVE_PAIRS = [
    # Majors
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
    'AUDUSD', 'USDCAD', 'NZDUSD',
    # Yen crosses
    'EURJPY', 'GBPJPY', 'AUDJPY',
    # Metals
    'XAUUSD', 'XAGUSD'
]

# All pairs (25) - full portfolio from trading_pairs.py
ALL_PAIRS = [
    # Majors
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
    'AUDUSD', 'USDCAD', 'NZDUSD',
    # Yen crosses
    'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY', 'NZDJPY',
    # Euro crosses
    'EURGBP', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD',
    # Sterling crosses
    'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD',
    # Commodities
    'XAUUSD', 'XAGUSD', 'WTIUSD'
]


def get_radar_pairs(scope: str = None) -> List[str]:
    """
    Get trading pairs for OpportunityRadar based on scope.
    
    Args:
        scope: "all" (25), "majors" (7), "active" (12)
        
    Returns:
        List of trading pair symbols
    """
    scope = scope or OPPORTUNITY_RADAR_CONFIG.get("pair_scope", "active")
    
    if scope == "all":
        return ALL_PAIRS.copy()
    elif scope == "majors":
        return MAJOR_PAIRS.copy()
    elif scope == "active":
        return ACTIVE_PAIRS.copy()
    else:
        return MAJOR_PAIRS.copy()


def get_radar_config() -> Dict:
    """Get OpportunityRadar configuration."""
    return OPPORTUNITY_RADAR_CONFIG.copy()


def is_radar_enabled() -> bool:
    """Check if OpportunityRadar is enabled."""
    return OPPORTUNITY_RADAR_CONFIG.get("enabled", True)


def get_top_n() -> int:
    """Get number of top pairs to deep analyze."""
    return OPPORTUNITY_RADAR_CONFIG.get("top_n", 3)


def get_score_weights() -> Dict[str, float]:
    """Get opportunity score weights."""
    return OPPORTUNITY_RADAR_CONFIG.get("score_weights", {}).copy()
