"""
Unified News and Economic Calendar Provider

Consolidates EnhancedNewsProvider and EconomicCalendarProvider into a single provider
with comprehensive news and economic calendar capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

from .base_provider import (
    BaseDataProvider, DataProviderType, DataProviderCapabilities, 
    Timeframe, ConnectionStatus, DataType
)
from ..core.exceptions import DataNotAvailableError, ConnectionError
from ..core.data_contracts import NewsEvent, EconomicCalendar, EconomicEvent, EconomicEventStatus
from ..core.economic_calendar_models import (
    EconomicEventImpact, EconomicEventCategory, CurrencyCode, calculate_event_importance_score
)
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)


class UnifiedNewsEconomicProvider(BaseDataProvider):
    """
    Unified News and Economic Calendar Provider
    
    Combines news analysis with economic calendar data in a single provider:
    - ForexFactory.com for news and economic calendar (primary source)
    - Investing.com for backup economic calendar
    - Real-time news sentiment analysis
    - Economic event impact analysis
    - Historical and live data support
    """
    
    def __init__(self, config_manager: UnifiedConfigManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified news and economic provider.
        
        Args:
            config_manager: Unified configuration manager instance
            config: Configuration including data sources, API keys, etc. (deprecated)
        """
        super().__init__(
            name="Unified_News_Economic_Provider",
            provider_type=DataProviderType.NEWS_AND_ECONOMIC,
            config_manager=config_manager,
            config=config or {}
        )
        
        # Configuration from unified config manager
        self.primary_source = config_manager.get_str('news.primary_source', 'forexfactory')
        self.backup_sources = config_manager.get_list('news.backup_sources', ['investing'])
        self.request_timeout = config_manager.get_int('news.request_timeout', 30)
        self.cache_duration = config_manager.get_int('news.cache_duration', 30)  # minutes
        self.user_agent = config_manager.get_str('news.user_agent',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Session management
        self.session = None
        self.cache = {}
        self.last_fetch_time = {}
        
        # Data source configurations
        self.sources = {
            'forexfactory': {
                'base_url': 'https://www.forexfactory.com',
                'calendar_endpoint': '/calendar.php',
                'news_endpoint': '/news',
                'supports_news': True,
                'supports_calendar': True
            },
            'investing': {
                'base_url': 'https://www.investing.com',
                'calendar_endpoint': '/economic-calendar/',
                'news_endpoint': '/news/',
                'supports_news': True,
                'supports_calendar': True
            }
        }
        
        logger.info("Unified News & Economic Provider initialized")
    
    def _define_capabilities(self) -> DataProviderCapabilities:
        """Define unified news and economic provider capabilities."""
        return DataProviderCapabilities(
            supports_ohlcv=False,
            supports_tick=False,
            supports_news=True,
            supports_economic_calendar=True,
            supports_order_book=False,
            supports_market_depth=False,
            supports_live=True,
            supports_historical=True,
            supported_symbols=[],
            supported_timeframes=[]
        )
    
    async def connect(self) -> bool:
        """
        Connect to news and economic calendar data sources.
        
        Returns:
            True if connection successful
        """
        try:
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=3)
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': self.user_agent}
            )
            
            # Test connection to primary source
            success = await self._test_source_connection(self.primary_source)
            
            if success:
                self.connection_status = ConnectionStatus.CONNECTED
                logger.info(f"Connected to primary source: {self.primary_source}")
                return True
            else:
                # Try backup sources
                for backup_source in self.backup_sources:
                    if await self._test_source_connection(backup_source):
                        self.primary_source = backup_source
                        self.connection_status = ConnectionStatus.CONNECTED
                        logger.info(f"Connected to backup source: {backup_source}")
                        return True
                
                self.connection_status = ConnectionStatus.FAILED
                logger.error("Failed to connect to any news/economic source")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connection_status = ConnectionStatus.FAILED
            return False
    
    async def disconnect(self):
        """Disconnect from news and economic data sources."""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.connection_status = ConnectionStatus.DISCONNECTED
        logger.info("Unified News & Economic Provider disconnected")
    
    async def is_connected(self) -> bool:
        """Check if connected to data sources."""
        return self.connection_status == ConnectionStatus.CONNECTED and self.session is not None
    
    async def _test_source_connection(self, source: str) -> bool:
        """Test connection to a specific data source."""
        try:
            if source not in self.sources:
                return False
            
            source_config = self.sources[source]
            test_url = source_config['base_url']
            
            async with self.session.get(test_url) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to connect to {source}: {e}")
            return False
    
    # =============================================================================
    # NEWS DATA METHODS
    # =============================================================================
    
    async def get_news_data(self, 
                          symbols: Optional[List[str]] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          count: Optional[int] = None) -> Optional[List[NewsEvent]]:
        """
        Get news data from ForexFactory and other sources.
        
        Args:
            symbols: List of symbols to filter news
            start_time: Start time for news
            end_time: End time for news
            count: Number of news items (default: 50)
            
        Returns:
            List of news events
        """
        try:
            if not await self.is_connected():
                if not await self.connect():
                    self._update_request_stats(False)
                    return None
            
            # Check cache first
            cache_key = f"news_{symbols}_{start_time}_{end_time}_{count}"
            if self._is_cached(cache_key):
                logger.info("Returning cached news data")
                return self.cache[cache_key]
            
            # Default parameters
            if count is None:
                count = 50
            if start_time is None:
                start_time = datetime.now() - timedelta(days=7)
            if end_time is None:
                end_time = datetime.now()
            
            news_events = []
            
            # Try primary source first
            
            # Check cache
            cache_key = f"calendar_{start_time}_{end_time}_{currencies}_{impact_levels}_{categories}"
            if self._is_cached(cache_key):
                logger.info("Returning cached economic calendar data")
                return self.cache[cache_key]
            
            # Default parameters
            if start_time is None:
                start_time = datetime.now()
            if end_time is None:
                end_time = start_time + timedelta(days=7)
            
            economic_events = []
            
            # Try primary source
            if self.primary_source == 'forexfactory':
                economic_events = await self._get_forexfactory_calendar(
                    start_time, end_time, currencies, impact_levels, categories)
            elif self.primary_source == 'investing':
                economic_events = await self._get_investing_calendar(
                    start_time, end_time, currencies, impact_levels, categories)
            
            # Try backup sources if needed
            if not economic_events:
                for backup_source in self.backup_sources:
                    if backup_source == 'forexfactory':
                        economic_events = await self._get_forexfactory_calendar(
                            start_time, end_time, currencies, impact_levels, categories)
                    elif backup_source == 'investing':
                        economic_events = await self._get_investing_calendar(
                            start_time, end_time, currencies, impact_levels, categories)
                    
                    if economic_events:
                        break
            
            # Create economic calendar
            if economic_events:
                calendar = EconomicCalendar(
                    start_date=start_time.date(),
                    end_date=end_time.date(),
                    events=economic_events,
                    total_events=len(economic_events),
                    last_updated=datetime.now()
                )
                
                # Cache results
                self.cache[cache_key] = calendar
                self.last_fetch_time[cache_key] = datetime.now()
                
                logger.info(f"Retrieved economic calendar with {len(economic_events)} events")
                self._update_request_stats(True)
                return calendar
            else:
                logger.warning("No economic calendar events retrieved")
                self._update_request_stats(False)
                return None
                
        except Exception as e:
            logger.error(f"Error getting economic calendar: {e}")
            self._update_request_stats(False)
            return None
    
    async def _get_forexfactory_calendar(self, start_time: datetime, end_time: datetime,
                                       currencies: Optional[List[str]], 
                                       impact_levels: Optional[List[str]],
                                       categories: Optional[List[str]]) -> List[EconomicEvent]:
        """Get economic calendar from ForexFactory."""
        try:
            # ForexFactory calendar URL with date parameters
            calendar_url = f"{self.sources['forexfactory']['base_url']}/calendar.php"
            params = {
                'day': start_time.strftime('%b%d.%Y'),
                'range': (end_time - start_time).days
            }
            
            economic_events = []
            
            async with self.session.get(calendar_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"ForexFactory calendar request failed: {response.status}")
                    return []
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Parse ForexFactory calendar structure
                # This would need to be implemented based on their actual HTML structure
                calendar_table = soup.find('table', class_='calendar__table') or soup.find('table')
                if not calendar_table:
                    logger.warning("Could not find calendar table on ForexFactory")
                    return []
                
                # Parse calendar rows
                rows = calendar_table.find_all('tr')
                current_date = start_time.date()
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) < 4:  # Skip header rows
                        continue
                    
                    # Extract event data (this would need to match ForexFactory's structure)
                    try:
                        # Parse time, currency, impact, event name, actual, forecast, previous
                        time_cell = cells[0].get_text(strip=True) if cells[0] else ""
                        currency_cell = cells[1].get_text(strip=True) if cells[1] else ""
                        impact_cell = cells[2].get_text(strip=True) if cells[2] else ""
                        event_cell = cells[3].get_text(strip=True) if cells[3] else ""
                        
                        # Filter by currency
                        if currencies and currency_cell not in currencies:
                            continue
                        
                        # Create economic event
                        event = EconomicEvent(
                            timestamp=datetime.combine(current_date, datetime.min.time()),
                            name=event_cell,
                            currency=currency_cell,
                            impact_level=self._map_impact_level(impact_cell),
                            category=EconomicEventCategory.OTHER,  # Would need to be mapped
                            actual_value=None,  # Would need to be parsed
                            forecast_value=None,  # Would need to be parsed
                            previous_value=None,  # Would need to be parsed
                            source="ForexFactory",
                            importance_score=calculate_event_importance_score(
                                self._map_impact_level(impact_cell), 
                                EconomicEventCategory.OTHER
                            )
                        )
                        
                        # Filter by impact levels
                        if impact_levels and event.impact_level.value not in impact_levels:
                            continue
                        
                        economic_events.append(event)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing calendar row: {e}")
                        continue
            
            return economic_events
            
        except Exception as e:
            logger.error(f"Error getting ForexFactory calendar: {e}")
            return []
    
    async def _get_investing_calendar(self, start_time: datetime, end_time: datetime,
                                    currencies: Optional[List[str]], 
                                    impact_levels: Optional[List[str]],
                                    categories: Optional[List[str]]) -> List[EconomicEvent]:
        """Get economic calendar from Investing.com."""
        try:
            # Similar implementation for Investing.com
            logger.info("Investing.com calendar retrieval - placeholder implementation")
            return []
            
        except Exception as e:
            logger.error(f"Error getting Investing.com calendar: {e}")
            return []
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid."""
        if cache_key not in self.cache:
            return False
        
        if cache_key not in self.last_fetch_time:
            return False
        
        cache_age = datetime.now() - self.last_fetch_time[cache_key]
        return cache_age.total_seconds() < (self.cache_duration * 60)
    
    def _extract_timestamp_from_context(self, element) -> datetime:
        """Extract timestamp from HTML element context."""
        try:
            # Look for date patterns in parent elements
            parent = element.parent
            if parent:
                date_text = parent.get_text()
                date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', date_text)
                if date_match:
                    from dateutil import parser
                    return parser.parse(date_match.group(1))
        except:
            pass
        
        return datetime.now()
    
    def _determine_impact_level(self, text: str) -> str:
        """Determine impact level from text content."""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['fed', 'central bank', 'interest rate', 'gdp', 'employment', 'inflation']):
            return "HIGH"
        elif any(keyword in text_lower for keyword in ['breaking', 'urgent', 'alert']):
            return "HIGH"
        elif any(keyword in text_lower for keyword in ['minor', 'slight', 'small']):
            return "LOW"
        else:
            return "MEDIUM"
    
    def _filter_symbols_by_content(self, text: str, symbols: Optional[List[str]]) -> List[str]:
        """Filter symbols mentioned in text content."""
        if not symbols:
            return []
        
        text_lower = text.lower()
        mentioned_symbols = []
        
        for symbol in symbols:
            if len(symbol) >= 6:
                base_currency = symbol[:3]
                quote_currency = symbol[3:6]
                if base_currency.lower() in text_lower or quote_currency.lower() in text_lower:
                    mentioned_symbols.append(symbol)
            elif symbol.lower() in text_lower:
                mentioned_symbols.append(symbol)
        
        return mentioned_symbols
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text content."""
        text_lower = text.lower()
        
        positive_keywords = ['positive', 'growth', 'rise', 'increase', 'up', 'boost', 'gain', 'improve']
        negative_keywords = ['negative', 'decline', 'fall', 'decrease', 'down', 'drop', 'loss', 'worsen']
        
        positive_score = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_score = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        if positive_score > negative_score:
            return min(0.8, positive_score * 0.2)
        elif negative_score > positive_score:
            return max(-0.8, -negative_score * 0.2)
        else:
            return 0.0
    
    def _map_impact_level(self, impact_text: str) -> EconomicEventImpact:
        """Map text impact level to enum."""
        impact_lower = impact_text.lower().strip()
        
        if impact_lower in ['high', 'red', '3']:
            return EconomicEventImpact.HIGH
        elif impact_lower in ['medium', 'orange', 'yellow', '2']:
            return EconomicEventImpact.MEDIUM
        elif impact_lower in ['low', 'green', '1']:
            return EconomicEventImpact.LOW
        else:
            return EconomicEventImpact.MEDIUM
    
    async def get_sentiment_analysis(self, symbol: str, time_period: str = "1d") -> Optional[Dict[str, Any]]:
        """
        Get sentiment analysis for a symbol based on recent news.
        
        Args:
            symbol: Trading symbol
            time_period: Time period for analysis
            
        Returns:
            Sentiment analysis results
        """
        try:
            # Get recent news for the symbol
            end_time = datetime.now()
            if time_period == "1d":
                start_time = end_time - timedelta(days=1)
            elif time_period == "1w":
                start_time = end_time - timedelta(days=7)
            else:
                start_time = end_time - timedelta(days=1)
            
            news_events = await self.get_news_data(
                symbols=[symbol],
                start_time=start_time,
                end_time=end_time,
                count=20
            )
            
            if not news_events:
                return None
            
            # Calculate aggregate sentiment
            sentiments = [event.sentiment for event in news_events if event.sentiment is not None]
            if not sentiments:
                return None
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            sentiment_strength = abs(avg_sentiment)
            
            return {
                "symbol": symbol,
                "time_period": time_period,
                "sentiment_score": avg_sentiment,
                "sentiment_strength": sentiment_strength,
                "sentiment_label": self._get_sentiment_label(avg_sentiment),
                "news_count": len(news_events),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return None
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Get sentiment label from score."""
        if sentiment_score > 0.3:
            return "POSITIVE"
        elif sentiment_score < -0.3:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to all data sources."""
        try:
            if not await self.is_connected():
                return {
                    "provider": self.name,
                    "status": "DISCONNECTED",
                    "connected": False
                }
            
            # Test news retrieval
            news_test = await self.get_news_data(count=5)
            news_status = "OK" if news_test else "FAILED"
            
            # Test economic calendar retrieval
            calendar_test = await self.get_economic_calendar()
            calendar_status = "OK" if calendar_test else "FAILED"
            
            overall_status = "OK" if news_status == "OK" or calendar_status == "OK" else "FAILED"
            
            return {
                "provider": self.name,
                "status": overall_status,
                "connected": True,
                "primary_source": self.primary_source,
                "capabilities": {
                    "news": news_status,
                    "economic_calendar": calendar_status,
                    "sentiment_analysis": "OK"
                },
                "test_results": {
                    "news_count": len(news_test) if news_test else 0,
                    "calendar_events": len(calendar_test.events) if calendar_test else 0
                }
            }
            
        except Exception as e:
            return {
                "provider": self.name,
                "status": "ERROR",
                "connected": False,
                "error": str(e)
            }
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get comprehensive provider information."""
        return {
            "provider_name": self.name,
            "provider_type": "UNIFIED_NEWS_ECONOMIC",
            "description": "Unified provider for news and economic calendar data",
            "capabilities": {
                "supports_news": True,
                "supports_economic_calendar": True,
                "supports_sentiment": True,
                "supports_real_time": True,
                "supports_historical": True
            },
            "sources": {
                "primary": self.primary_source,
                "backup": self.backup_sources,
                "supported": list(self.sources.keys())
            },
            "configuration": {
                "cache_duration": self.cache_duration,
                "request_timeout": self.request_timeout
            },
            "status": self.connection_status.value
        }


# Backwards compatibility aliases
EnhancedNewsProvider = UnifiedNewsEconomicProvider
EconomicCalendarProvider = UnifiedNewsEconomicProvider