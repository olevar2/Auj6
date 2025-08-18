"""
Indicators Package - Other Category Module
========================================

This module contains miscellaneous advanced indicators that don't fit into standard categories.
All indicators implement sophisticated algorithms and advanced signal processing techniques.
"""

from .biorhythm_market_synth_indicator import BiorhythmMarketSynthIndicator
from .custom_ai_composite_indicator import CustomAICompositeIndicator
from .hidden_divergence_detector_indicator import HiddenDivergenceDetectorIndicator
from .momentum_divergence_scanner_indicator import MomentumDivergenceScannerIndicator
from .news_article_indicator import NewsArticleIndicator
from .pattern_signal_indicator import PatternSignalIndicator
from .price_volume_divergence_indicator import PriceVolumeDivergenceIndicator
from .self_similarity_detector_indicator import SelfSimilarityDetectorIndicator

__all__ = [
    "BiorhythmMarketSynthIndicator",
    "CustomAICompositeIndicator", 
    "HiddenDivergenceDetectorIndicator",
    "MomentumDivergenceScannerIndicator",
    "NewsArticleIndicator",
    "PatternSignalIndicator",
    "PriceVolumeDivergenceIndicator",
    "SelfSimilarityDetectorIndicator"
]
