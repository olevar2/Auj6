"""
News Sentiment Impact Indicator
Analyzes news sentiment and its correlation with economic events to generate trading signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
import logging
import re
from collections import defaultdict

from ....core.data_contracts import EconomicEvent
from ..base.base_indicator import EconomicIndicator

logger = logging.getLogger(__name__)

@dataclass
class SentimentSignal:
    """News sentiment trading signal"""
    currency: str
    sentiment_score: float          # -1.0 to 1.0
    sentiment_strength: float       # 0.0 to 1.0
    signal_direction: str           # BUY, SELL, NEUTRAL
    confidence: float               # 0.0 to 1.0
    source_count: int              # Number of news sources
    economic_correlation: float     # Correlation with economic events

class NewsSentimentImpactIndicator(EconomicIndicator):
    """
    Advanced indicator that analyzes news sentiment in correlation with economic events
    to generate high-confidence trading signals based on market psychology and fundamentals.
    """
    
    def __init__(self,
                 sentiment_threshold: float = 0.3,
                 correlation_threshold: float = 0.6,
                 news_window_hours: int = 24):
        """
        Initialize News Sentiment Impact Indicator
        
        Args:
            sentiment_threshold: Minimum sentiment strength for signals
            correlation_threshold: Minimum correlation with economic events
            news_window_hours: Hours of news data to analyze
        """
        super().__init__()
        self.sentiment_threshold = sentiment_threshold
        self.correlation_threshold = correlation_threshold
        self.news_window_hours = news_window_hours
        
        # Financial sentiment lexicon (simplified version)
        self.sentiment_lexicon = {
            # Positive economic terms
            'growth': 0.8, 'rally': 0.9, 'surge': 0.9, 'gains': 0.7,
            'strong': 0.6, 'robust': 0.7, 'bullish': 0.9, 'optimistic': 0.6,
            'recovery': 0.7, 'expansion': 0.6, 'improve': 0.5, 'rise': 0.5,
            'increase': 0.4, 'positive': 0.5, 'upbeat': 0.6, 'confident': 0.5,
            
            # Negative economic terms
            'decline': -0.6, 'fall': -0.5, 'drop': -0.6, 'crash': -0.9,
            'plunge': -0.8, 'bearish': -0.9, 'pessimistic': -0.6, 'weak': -0.5,
            'recession': -0.9, 'crisis': -0.8, 'concern': -0.4, 'worry': -0.5,
            'risk': -0.4, 'fear': -0.6, 'uncertainty': -0.5, 'volatile': -0.3,
            'negative': -0.5, 'disappointing': -0.6, 'poor': -0.5,
            
            # Neutral but context-dependent
            'stable': 0.1, 'steady': 0.1, 'unchanged': 0.0, 'mixed': 0.0
        }
        
        # Currency-specific sentiment modifiers
        self.currency_sentiment_modifiers = {
            'USD': {'federal_reserve': 0.8, 'fed': 0.8, 'dollar_strength': 0.7},
            'EUR': {'ecb': 0.8, 'eurozone': 0.6, 'european_union': 0.5},
            'GBP': {'boe': 0.8, 'brexit': -0.3, 'uk_economy': 0.6},
            'JPY': {'boj': 0.8, 'yen_intervention': 0.9, 'japan_economy': 0.6},
            'AUD': {'rba': 0.8, 'commodity_prices': 0.7, 'china_trade': 0.6},
            'CAD': {'boc': 0.8, 'oil_prices': 0.8, 'commodity_demand': 0.6}
        }
        
        # Event impact correlation patterns
        self.event_sentiment_correlations = {
            'NFP': {'employment': 0.9, 'jobs': 0.9, 'unemployment': -0.8},
            'CPI': {'inflation': 0.8, 'prices': 0.6, 'cost_of_living': 0.7},
            'GDP': {'economic_growth': 0.9, 'gdp': 0.9, 'economy': 0.6},
            'Interest Rate': {'interest_rates': 0.9, 'monetary_policy': 0.8}
        }
    
    async def calculate(self, data: pd.DataFrame,
                       economic_events: List[EconomicEvent] = None,
                       news_data: List[Dict] = None,
                       pair: str = None) -> Dict:
        """
        Analyze news sentiment impact in correlation with economic events
        
        Args:
            data: Price data DataFrame
            economic_events: List of economic events
            news_data: List of news articles/headlines
            pair: Currency pair
            
        Returns:
            Dictionary containing sentiment analysis and signals
        """
        try:
            if not news_data:
                news_data = await self._fetch_synthetic_news_data(pair)
            
            if not economic_events:
                economic_events = []
            
            current_time = datetime.now()
            
            # Filter recent news and relevant events
            recent_news = self._filter_recent_news(news_data, current_time)
            relevant_events = self._filter_relevant_events(economic_events, current_time, pair)
            
            # Analyze sentiment for each currency
            sentiment_analysis = await self._analyze_sentiment_by_currency(
                recent_news, pair
            )
            
            # Correlate sentiment with economic events
            event_correlation = self._correlate_sentiment_with_events(
                sentiment_analysis, relevant_events
            )
            
            # Generate trading signals
            signals = self._generate_sentiment_signals(
                sentiment_analysis, event_correlation, pair
            )
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(
                sentiment_analysis, event_correlation, len(recent_news)
            )
            
            return {
                'signals': signals,
                'sentiment_analysis': sentiment_analysis,
                'event_correlation': event_correlation,
                'confidence_metrics': confidence_metrics,
                'news_count': len(recent_news),
                'relevant_events': len(relevant_events),
                'calculation_time': current_time,
                'pair': pair
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._generate_error_result(str(e))
    
    async def _fetch_synthetic_news_data(self, pair: str) -> List[Dict]:
        """Generate synthetic news data for demonstration"""
        current_time = datetime.now()
        
        synthetic_news = [
            {
                'headline': 'Federal Reserve signals cautious approach to interest rates',
                'content': 'The Federal Reserve indicated a cautious stance on future rate decisions...',
                'timestamp': current_time - timedelta(hours=2),
                'source': 'Reuters',
                'currencies': ['USD']
            },
            {
                'headline': 'European Central Bank maintains dovish outlook',
                'content': 'ECB officials expressed concerns about economic growth...',
                'timestamp': current_time - timedelta(hours=4),
                'source': 'Bloomberg',
                'currencies': ['EUR']
            },
            {
                'headline': 'Strong jobs data boosts dollar sentiment',
                'content': 'Better than expected employment figures support USD strength...',
                'timestamp': current_time - timedelta(hours=6),
                'source': 'Financial Times',
                'currencies': ['USD']
            },
            {
                'headline': 'Oil prices surge on supply concerns',
                'content': 'Rising oil prices support commodity currencies...',
                'timestamp': current_time - timedelta(hours=8),
                'source': 'CNBC',
                'currencies': ['CAD', 'AUD']
            }
        ]
        
        return synthetic_news
    
    def _filter_recent_news(self, news_data: List[Dict], current_time: datetime) -> List[Dict]:
        """Filter news within the analysis window"""
        cutoff_time = current_time - timedelta(hours=self.news_window_hours)
        
        return [
            news for news in news_data
            if news.get('timestamp', current_time) >= cutoff_time
        ]
    
    def _filter_relevant_events(self, events: List[EconomicEvent],
                               current_time: datetime, pair: str) -> List[EconomicEvent]:
        """Filter events relevant to the analysis"""
        if not pair or len(pair) < 6:
            return events
        
        base_currency = pair[:3]
        quote_currency = pair[3:6]
        relevant_currencies = {base_currency, quote_currency}
        
        return [
            event for event in events
            if (event.currency in relevant_currencies and
                abs((event.time - current_time).total_seconds()) <= 48 * 3600)  # 48 hours window
        ]
    
    async def _analyze_sentiment_by_currency(self, news_data: List[Dict],
                                           pair: str) -> Dict:
        """Analyze sentiment for each currency"""
        currency_sentiments = defaultdict(lambda: {
            'sentiment_score': 0.0,
            'sentiment_strength': 0.0,
            'positive_count': 0,
            'negative_count': 0,
            'news_items': []
        })
        
        for news_item in news_data:
            currencies = news_item.get('currencies', [])
            if not currencies and pair:
                # Try to infer currencies from content
                currencies = self._infer_currencies_from_content(
                    news_item.get('headline', '') + ' ' + news_item.get('content', ''),
                    pair
                )
            
            # Analyze sentiment of this news item
            item_sentiment = self._analyze_text_sentiment(
                news_item.get('headline', '') + ' ' + news_item.get('content', '')
            )
            
            for currency in currencies:
                if currency in currency_sentiments or len(currency_sentiments) < 10:
                    sentiment_data = currency_sentiments[currency]
                    
                    # Apply currency-specific modifiers
                    modified_sentiment = self._apply_currency_modifiers(
                        item_sentiment, currency, news_item
                    )
                    
                    # Update cumulative sentiment
                    current_count = len(sentiment_data['news_items'])
                    sentiment_data['sentiment_score'] = (
                        sentiment_data['sentiment_score'] * current_count + modified_sentiment
                    ) / (current_count + 1)
                    
                    # Update strength (absolute value)
                    sentiment_data['sentiment_strength'] = max(
                        sentiment_data['sentiment_strength'],
                        abs(modified_sentiment)
                    )
                    
                    # Update counts
                    if modified_sentiment > 0.1:
                        sentiment_data['positive_count'] += 1
                    elif modified_sentiment < -0.1:
                        sentiment_data['negative_count'] += 1
                    
                    # Store news item
                    sentiment_data['news_items'].append({
                        'headline': news_item.get('headline', ''),
                        'sentiment': modified_sentiment,
                        'timestamp': news_item.get('timestamp'),
                        'source': news_item.get('source', 'Unknown')
                    })
        
        return dict(currency_sentiments)
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using lexicon-based approach"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        sentiment_scores = []
        for word in words:
            if word in self.sentiment_lexicon:
                sentiment_scores.append(self.sentiment_lexicon[word])
        
        if not sentiment_scores:
            return 0.0
        
        # Calculate weighted sentiment
        avg_sentiment = np.mean(sentiment_scores)
        
        # Apply intensity modifiers
        if 'very' in text_lower or 'extremely' in text_lower:
            avg_sentiment *= 1.3
        elif 'slightly' in text_lower or 'somewhat' in text_lower:
            avg_sentiment *= 0.7
        
        # Clip to valid range
        return max(-1.0, min(1.0, avg_sentiment))
    
    def _infer_currencies_from_content(self, content: str, pair: str) -> List[str]:
        """Infer relevant currencies from content"""
        currencies = []
        content_lower = content.lower()
        
        # Check for explicit currency mentions
        currency_keywords = {
            'USD': ['dollar', 'usd', 'fed', 'federal reserve', 'united states'],
            'EUR': ['euro', 'eur', 'ecb', 'european central bank', 'eurozone'],
            'GBP': ['pound', 'gbp', 'sterling', 'boe', 'bank of england'],
            'JPY': ['yen', 'jpy', 'boj', 'bank of japan'],
            'AUD': ['aussie', 'aud', 'rba', 'reserve bank of australia'],
            'CAD': ['cad', 'loonie', 'boc', 'bank of canada'],
            'CHF': ['franc', 'chf', 'snb', 'swiss national bank'],
            'NZD': ['kiwi', 'nzd', 'rbnz', 'reserve bank of new zealand']
        }
        
        for currency, keywords in currency_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                currencies.append(currency)
        
        # If no currencies found, assume pair currencies
        if not currencies and pair and len(pair) >= 6:
            currencies = [pair[:3], pair[3:6]]
        
        return currencies
    
    def _apply_currency_modifiers(self, base_sentiment: float,
                                 currency: str, news_item: Dict) -> float:
        """Apply currency-specific sentiment modifiers"""
        if currency not in self.currency_sentiment_modifiers:
            return base_sentiment
        
        modifiers = self.currency_sentiment_modifiers[currency]
        content = (news_item.get('headline', '') + ' ' + 
                  news_item.get('content', '')).lower()
        
        modifier_factor = 1.0
        for keyword, factor in modifiers.items():
            if keyword.replace('_', ' ') in content:
                modifier_factor *= (1.0 + factor * 0.2)  # 20% impact
        
        return base_sentiment * modifier_factor
    
    def _correlate_sentiment_with_events(self, sentiment_analysis: Dict,
                                       events: List[EconomicEvent]) -> Dict:
        """Correlate sentiment with economic events"""
        correlations = {}
        
        for event in events:
            event_correlations = {
                'event_name': event.name,
                'currency': event.currency,
                'time': event.time,
                'impact_level': event.impact_level,
                'sentiment_correlation': 0.0,
                'sentiment_alignment': 'NEUTRAL'
            }
            
            # Get sentiment for event currency
            currency_sentiment = sentiment_analysis.get(event.currency, {})
            sentiment_score = currency_sentiment.get('sentiment_score', 0.0)
            
            # Calculate correlation based on event type
            correlation_keywords = self.event_sentiment_correlations.get(
                event.name, {'general': 0.5}
            )
            
            max_correlation = 0.0
            for keyword, correlation_strength in correlation_keywords.items():
                # Simple correlation calculation
                if abs(sentiment_score) > 0.1:  # Significant sentiment
                    correlation = correlation_strength * abs(sentiment_score)
                    max_correlation = max(max_correlation, correlation)
            
            event_correlations['sentiment_correlation'] = max_correlation
            
            # Determine alignment
            if sentiment_score > 0.2 and event.impact_level in ['HIGH', 'CRITICAL']:
                event_correlations['sentiment_alignment'] = 'BULLISH'
            elif sentiment_score < -0.2 and event.impact_level in ['HIGH', 'CRITICAL']:
                event_correlations['sentiment_alignment'] = 'BEARISH'
            else:
                event_correlations['sentiment_alignment'] = 'NEUTRAL'
            
            correlations[f"{event.name}_{event.currency}"] = event_correlations
        
        return correlations
    
    def _generate_sentiment_signals(self, sentiment_analysis: Dict,
                                  event_correlation: Dict,
                                  pair: str) -> List[SentimentSignal]:
        """Generate trading signals based on sentiment analysis"""
        signals = []
        
        if not pair or len(pair) < 6:
            return signals
        
        base_currency = pair[:3]
        quote_currency = pair[3:6]
        
        # Analyze sentiment for both currencies
        base_sentiment = sentiment_analysis.get(base_currency, {})
        quote_sentiment = sentiment_analysis.get(quote_currency, {})
        
        base_score = base_sentiment.get('sentiment_score', 0.0)
        quote_score = quote_sentiment.get('sentiment_score', 0.0)
        
        # Calculate relative sentiment
        relative_sentiment = base_score - quote_score
        sentiment_strength = abs(relative_sentiment)
        
        if sentiment_strength >= self.sentiment_threshold:
            # Calculate correlation with events
            avg_correlation = 0.0
            correlation_count = 0
            for correlation_data in event_correlation.values():
                if correlation_data['currency'] in [base_currency, quote_currency]:
                    avg_correlation += correlation_data['sentiment_correlation']
                    correlation_count += 1
            
            if correlation_count > 0:
                avg_correlation /= correlation_count
            
            # Generate signal only if correlation is sufficient
            if avg_correlation >= self.correlation_threshold:
                signal_direction = 'BUY' if relative_sentiment > 0 else 'SELL'
                
                # Calculate confidence
                confidence = min(
                    sentiment_strength * 0.7 + avg_correlation * 0.3,
                    1.0
                )
                
                # Count news sources
                source_count = (
                    len(base_sentiment.get('news_items', [])) +
                    len(quote_sentiment.get('news_items', []))
                )
                
                signal = SentimentSignal(
                    currency=base_currency,
                    sentiment_score=relative_sentiment,
                    sentiment_strength=sentiment_strength,
                    signal_direction=signal_direction,
                    confidence=confidence,
                    source_count=source_count,
                    economic_correlation=avg_correlation
                )
                
                signals.append(signal)
        
        return signals
    
    def _calculate_confidence_metrics(self, sentiment_analysis: Dict,
                                    event_correlation: Dict,
                                    news_count: int) -> Dict:
        """Calculate confidence metrics for the analysis"""
        metrics = {
            'overall_confidence': 0.0,
            'sentiment_reliability': 0.0,
            'event_correlation_strength': 0.0,
            'data_sufficiency': 0.0
        }
        
        # Calculate sentiment reliability
        if sentiment_analysis:
            total_news_items = sum(
                len(data.get('news_items', []))
                for data in sentiment_analysis.values()
            )
            metrics['sentiment_reliability'] = min(total_news_items / 10.0, 1.0)
        
        # Calculate event correlation strength
        if event_correlation:
            correlations = [
                data['sentiment_correlation']
                for data in event_correlation.values()
            ]
            metrics['event_correlation_strength'] = np.mean(correlations) if correlations else 0.0
        
        # Calculate data sufficiency
        metrics['data_sufficiency'] = min(news_count / 5.0, 1.0)
        
        # Calculate overall confidence
        metrics['overall_confidence'] = (
            metrics['sentiment_reliability'] * 0.4 +
            metrics['event_correlation_strength'] * 0.4 +
            metrics['data_sufficiency'] * 0.2
        )
        
        return metrics
    
    def _generate_error_result(self, error_message: str) -> Dict:
        """Generate error result"""
        return {
            'signals': [],
            'sentiment_analysis': {},
            'event_correlation': {},
            'confidence_metrics': {
                'overall_confidence': 0.0,
                'sentiment_reliability': 0.0,
                'event_correlation_strength': 0.0,
                'data_sufficiency': 0.0
            },
            'news_count': 0,
            'relevant_events': 0,
            'calculation_time': datetime.now(),
            'error': error_message
        }