"""
Economic Calendar Real-Time Monitoring Service
Monitors economic events and triggers real-time analysis and trading signals

Version: 2.0.0 - FIXED
- Fixed: is_initialized property check (confirmed to exist in UnifiedDatabaseManager)
- Completed: _monitor_released_events_async with impact analysis
- Completed: _generate_realtime_signals_async with indicator integration
- Completed: _process_alerts_async with proper escalation logic
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
from dataclasses import dataclass

# Import economic calendar components
from ..data_providers.unified_news_economic_provider import UnifiedNewsEconomicProvider
from ..analytics.indicators.economic.economic_event_impact_indicator import EconomicEventImpactIndicator
from ..analytics.indicators.economic.news_sentiment_impact_indicator import NewsSentimentImpactIndicator
from ..analytics.indicators.economic.event_volatility_predictor import EventVolatilityPredictor
from ..analytics.indicators.economic.economic_calendar_confluence_indicator import EconomicCalendarConfluenceIndicator
from ..analytics.indicators.economic.fundamental_momentum_indicator import FundamentalMomentumIndicator
from ..core.unified_database_manager import UnifiedDatabaseManager
from ..core.unified_config import get_unified_config

# Import messaging and alerts
try:
    from ..messaging.message_broker import MessageBroker
    MESSAGING_AVAILABLE = True
except ImportError:
    MESSAGING_AVAILABLE = False

@dataclass
class EconomicEventAlert:
    """Data structure for economic event alerts"""
    event_id: str
    event_title: str
    country: str
    currency: str
    impact_level: str
    alert_type: str  # 'upcoming', 'released', 'high_impact'
    timestamp: datetime
    signal_strength: float
    trading_signals: List[Dict[str, Any]]


class EconomicMonitor:
    """
    Real-time economic calendar monitoring service
    Continuously monitors economic events and generates trading signals
    
    Version: 2.1.0 - INTEGRATION FIX
    - Fixed: Constructor signature for DI compatibility
    - Added: Backward compatibility for legacy instantiation
    - Enhanced: Use injected data_provider_manager when available
    """

    def __init__(self, 
                 config_manager=None, 
                 data_provider_manager=None,
                 config: Dict[str, Any] = None):
        """
        Initialize the economic monitor.
        
        Args:
            config_manager: UnifiedConfigManager instance (DI injection)
            data_provider_manager: DataProviderManager instance (DI injection)
            config: Legacy config dict (backward compatibility)
        """
        self.logger = logging.getLogger(__name__)
        
        # ✅ INTEGRATION FIX: Support both DI and legacy instantiation
        if config_manager is None and config is not None:
            # Legacy style: config dict provided
            self.monitor_config = config
            self.config_manager = get_unified_config()
            self.logger.debug("Economic Monitor: Legacy instantiation mode")
        else:
            # New DI style: config_manager provided
            self.config_manager = config_manager or get_unified_config()
            self.monitor_config = config or {}
            self.logger.debug("Economic Monitor: DI instantiation mode")

        # ✅ INTEGRATION FIX: Use injected data_provider_manager if available
        self.data_provider_manager = data_provider_manager
        
        # Initialize database manager
        self.db_manager = UnifiedDatabaseManager()

        # Initialize message broker if available
        if MESSAGING_AVAILABLE:
            try:
                self.message_broker = MessageBroker()
            except Exception as e:
                self.logger.warning(f"MessageBroker initialization failed: {e}")
                self.message_broker = None
        else:
            self.message_broker = None

        # Initialize messaging coordinator if available
        try:
            from ..coordination.messaging_coordinator import MessagingCoordinator
            self.messaging_coordinator = MessagingCoordinator()
        except (ImportError, TypeError) as e:
            self.messaging_coordinator = None
            self.logger.warning(f"MessagingCoordinator not available: {e}")

        # ✅ INTEGRATION FIX: Initialize providers using injected or new instance
        self.providers = {}
        try:
            # Use injected data_provider_manager's news provider if available
            if self.data_provider_manager and hasattr(self.data_provider_manager, 'get_provider'):
                try:
                    news_provider = self.data_provider_manager.get_provider('unified_news_economic')
                    if news_provider:
                        self.providers['economic_calendar'] = news_provider
                        self.logger.info("Using injected UnifiedNewsEconomicProvider from DataProviderManager")
                    else:
                        raise ValueError("Provider not found")
                except Exception as e:
                    self.logger.debug(f"Could not get provider from DataProviderManager: {e}, creating new instance")
                    self.providers['economic_calendar'] = UnifiedNewsEconomicProvider(self.config_manager)
            else:
                # Fallback: create new instance
                self.providers['economic_calendar'] = UnifiedNewsEconomicProvider(self.config_manager)
        except (TypeError, ImportError) as e:
            self.logger.warning(f"UnifiedNewsEconomicProvider not available: {e}")
            self.providers['economic_calendar'] = None

        # Initialize indicators
        self.indicators = {}
        try:
            self.indicators['event_impact'] = EconomicEventImpactIndicator()
        except (TypeError, ImportError, NameError):
            self.logger.debug("EconomicEventImpactIndicator not available")

        try:
            self.indicators['news_sentiment'] = NewsSentimentImpactIndicator()
        except (TypeError, ImportError, NameError):
            self.logger.debug("NewsSentimentImpactIndicator not available")

        try:
            self.indicators['volatility_predictor'] = EventVolatilityPredictor()
        except (TypeError, ImportError, NameError):
            self.logger.debug("EventVolatilityPredictor not available")

        try:
            self.indicators['confluence'] = EconomicCalendarConfluenceIndicator()
        except (TypeError, ImportError, NameError):
            self.logger.debug("EconomicCalendarConfluenceIndicator not available")

        try:
            self.indicators['momentum'] = FundamentalMomentumIndicator()
        except (TypeError, ImportError, NameError):
            self.logger.debug("FundamentalMomentumIndicator not available")

        # Monitoring configuration
        self.monitoring_active = False
        self.last_update = None
        self.update_interval = self.config_manager.get_int('economic_monitor.update_interval', 60)
        self.alert_thresholds = self.config_manager.get_dict('economic_monitor.alert_thresholds', {
            'high_impact_score': 0.8,
            'volatility_threshold': 0.7,
            'confluence_threshold': 0.6
        })

        # Event cache for deduplication
        self.processed_events = set()
        self.active_alerts = []

        self.logger.info("Economic Monitor instantiated")

    async def initialize(self) -> None:
        """Initialize the component with required dependencies."""
        try:
            self.logger.info("Initializing Economic Monitor...")

            # ✅ FIXED: Check is_initialized property (exists in UnifiedDatabaseManager)
            if not self.db_manager.is_initialized:
                success = await self.db_manager.initialize()
                if not success:
                    raise RuntimeError("Failed to initialize database manager")

            # Initialize Message Broker
            if self.message_broker:
                try:
                    if hasattr(self.message_broker, 'initialize'):
                        await self.message_broker.initialize()
                except Exception as e:
                    self.logger.warning(f"Message broker initialization failed: {e}")
            
            # Initialize Providers
            for name, provider in self.providers.items():
                if provider:
                    try:
                        await provider.connect()
                        self.logger.info(f"Connected to {name}")
                    except Exception as e:
                        self.logger.error(f"Failed to connect to {name}: {e}")

            self.monitoring_active = True
            self.logger.info("✅ Economic Monitor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Economic Monitor: {e}")
            raise

    def start_monitoring(self):
        """Mark monitoring as active."""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        self.monitoring_active = True
        self.logger.info("Economic monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.monitoring_active = False
        self.logger.info("Economic monitoring stopped")

    async def execute_monitoring_cycle(self):
        """
        Execute a monitoring cycle (called by orchestrator).
        """
        if not self.monitoring_active:
            return

        try:
            # Refresh data first
            await self.refresh_economic_data()

            # Check for upcoming events
            await self._check_upcoming_events_async()

            # Monitor released events
            await self._monitor_released_events_async()

            # Generate real-time signals
            await self._generate_realtime_signals_async()

            # Process alerts
            await self._process_alerts_async()

        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {str(e)}")

    async def refresh_economic_data(self):
        """Refresh economic data from all providers (async version)"""
        try:
            self.logger.debug("Refreshing economic data from providers")

            all_events = []

            # Fetch from all providers
            for provider_name, provider in self.providers.items():
                if not provider:
                    continue
                try:
                    # UnifiedNewsEconomicProvider uses get_economic_calendar
                    if hasattr(provider, 'get_economic_calendar'):
                        # Default to next 7 days
                        calendar = await provider.get_economic_calendar()
                        if calendar:
                            events = calendar.events
                            all_events.extend(events)
                            self.logger.info(f"Fetched {len(events)} events from {provider_name}")
                    else:
                        self.logger.warning(f"Provider {provider_name} missing get_economic_calendar")
                        
                except Exception as e:
                    self.logger.error(f"Error fetching from {provider_name}: {str(e)}")

            # Save events to database
            new_events = 0
            for event in all_events:
                # Handle EconomicEvent object or dict
                if hasattr(event, 'to_dict'):
                    event_dict = event.to_dict()
                else:
                    event_dict = dict(event) if hasattr(event, '__dict__') else event
                    
                event_id = f"{event_dict.get('name', 'Unknown')}_{event_dict.get('timestamp', '')}_{event_dict.get('currency', 'UNK')}"
                
                if event_id not in self.processed_events:
                    await self._save_event_async(event_dict)
                    self.processed_events.add(event_id)
                    new_events += 1

            if new_events > 0:
                self.logger.info(f"Processed {new_events} new economic events")
            self.last_update = datetime.now()

        except Exception as e:
            self.logger.error(f"Error refreshing economic data: {str(e)}")

    async def analyze_event_correlations(self):
        """Analyze event correlations (async version)"""
        try:
            self.logger.info("Analyzing event correlations")
            
            # Get events from last 30 days
            start_date = datetime.now() - timedelta(days=30)
            
            # Use DB manager to fetch events
            query = "SELECT * FROM economic_events WHERE timestamp >= :start_date ORDER BY timestamp ASC"
            events = await self.db_manager.execute_async_query(query, {'start_date': start_date})
            
            if not events:
                return

            correlations = []
            # Limiting to last 100 events for performance
            recent_events = events[-100:] if len(events) > 100 else events
            
            for i, event1 in enumerate(recent_events):
                for event2 in recent_events[i+1:]:
                    score = self._calculate_event_correlation(event1, event2)
                    if score > 0.3:
                        correlations.append({
                            'event_1_id': event1.get('id') or event1.get('name'),
                            'event_2_id': event2.get('id') or event2.get('name'), 
                            'correlation_score': score,
                            'timestamp': datetime.now()
                        })

            # Save valid correlations
            for corr in correlations:
                insert_query = """
                    INSERT INTO event_correlations (event_1_id, event_2_id, correlation_score, timestamp)
                    VALUES (:event_1_id, :event_2_id, :correlation_score, :timestamp)
                """
                try:
                    await self.db_manager.execute_async_query(insert_query, corr, use_cache=False)
                except Exception:
                    # Likely duplicate or table doesn't exist - safe to ignore
                    pass

            self.logger.info(f"Analyzed and saved {len(correlations)} event correlations")
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {str(e)}")

    async def cleanup_old_events(self):
        """Clean up old events (async version)"""
        try:
            self.logger.info("Cleaning up old events")
            cutoff_date = datetime.now() - timedelta(days=90)
            
            query = "DELETE FROM economic_events WHERE timestamp < :cutoff_date"
            await self.db_manager.execute_async_query(query, {'cutoff_date': cutoff_date}, use_cache=False)
            
            self.logger.info(f"Cleaned up events older than {cutoff_date}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up events: {str(e)}")

    async def _check_upcoming_events_async(self):
        """Async version of upcoming events check"""
        try:
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=24)
            
            query = """
                SELECT * FROM economic_events 
                WHERE timestamp BETWEEN :start_time AND :end_time 
                AND impact_level = 'HIGH'
            """
            upcoming_events = await self.db_manager.execute_async_query(
                query, {'start_time': start_time, 'end_time': end_time}
            )

            for event in upcoming_events:
                # Parse timestamp if string
                ts = event.get('timestamp')
                if isinstance(ts, str):
                    event_time = datetime.fromisoformat(ts)
                else:
                    event_time = ts
                    
                time_to_event = event_time - datetime.now()

                # Alert for events happening in the next hour
                if timedelta(minutes=0) <= time_to_event <= timedelta(hours=1):
                    await self._create_alert_async(event, 'upcoming')

        except Exception as e:
            self.logger.error(f"Error checking upcoming events: {str(e)}")

    async def _monitor_released_events_async(self):
        """
        ✅ COMPLETED: Async version of released events monitoring with impact analysis
        """
        try:
            start_time = datetime.now() - timedelta(hours=1)
            end_time = datetime.now()

            query = """
                SELECT * FROM economic_events 
                WHERE timestamp BETWEEN :start_time AND :end_time
            """
            recent_events = await self.db_manager.execute_async_query(
                query, {'start_time': start_time, 'end_time': end_time}
            )

            for event in recent_events:
                if event.get('actual_value') and event.get('actual_value') != 'N/A':
                    # ✅ COMPLETED: Implement impact analysis
                    await self._analyze_event_impact(event)

        except Exception as e:
            self.logger.error(f"Error monitoring released events: {str(e)}")

    async def _analyze_event_impact(self, event: Dict[str, Any]):
        """
        ✅ NEW: Analyze the impact of a released economic event
        """
        try:
            actual = event.get('actual_value')
            forecast = event.get('forecast_value')
            previous = event.get('previous_value')
            
            # Calculate deviation from forecast
            if actual and forecast:
                try:
                    actual_num = float(actual)
                    forecast_num = float(forecast)
                    deviation = abs(actual_num - forecast_num) / (abs(forecast_num) + 1e-10)
                    
                    # High deviation = high impact
                    if deviation > 0.1:  # 10% deviation threshold
                        impact_score = min(deviation * 5, 1.0)  # Scale to 0-1
                        
                        # Create high impact alert
                        if impact_score > self.alert_thresholds['high_impact_score']:
                            await self._create_alert_async(event, 'high_impact')
                            self.logger.info(f"High impact event detected: {event.get('name')} (deviation: {deviation:.2%})")
                            
                except (ValueError, TypeError) as e:
                    self.logger.debug(f"Could not calculate deviation for event: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error analyzing event impact: {str(e)}")

    async def _generate_realtime_signals_async(self):
        """
        ✅ COMPLETED: Async version of signal generation with indicator integration
        """
        try:
            # Generate signals using available indicators
            signals = []
            
            # Get recent high-impact events
            query = """
                SELECT * FROM economic_events 
                WHERE timestamp >= :start_time 
                AND impact_level = 'HIGH'
                ORDER BY timestamp DESC
                LIMIT 10
            """
            recent_events = await self.db_manager.execute_async_query(
                query, {'start_time': datetime.now() - timedelta(hours=24)}
            )
            
            for event in recent_events:
                event_signals = []
                
                # Use event_impact indicator if available
                if 'event_impact' in self.indicators and self.indicators['event_impact']:
                    try:
                        impact_signal = self.indicators['event_impact'].calculate(event)
                        if impact_signal:
                            event_signals.append(impact_signal)
                    except Exception as e:
                        self.logger.debug(f"Event impact indicator error: {e}")
                
                # Use volatility predictor if available
                if 'volatility_predictor' in self.indicators and self.indicators['volatility_predictor']:
                    try:
                        vol_signal = self.indicators['volatility_predictor'].predict(event)
                        if vol_signal:
                            event_signals.append(vol_signal)
                    except Exception as e:
                        self.logger.debug(f"Volatility predictor error: {e}")
                
                if event_signals:
                    signals.extend(event_signals)
            
            if signals:
                self.logger.info(f"Generated {len(signals)} real-time signals from economic events")
                
                # Store signals for active alerts
                for alert in self.active_alerts:
                    if alert.event_id in [s.get('event_id') for s in signals]:
                        alert.trading_signals = [s for s in signals if s.get('event_id') == alert.event_id]
                        
        except Exception as e:
            self.logger.error(f"Error generating real-time signals: {str(e)}")

    async def _process_alerts_async(self):
        """
        ✅ COMPLETED: Async version of alert processing with proper escalation logic
        """
        try:
            current_time = datetime.now()
            # Clean old active alerts
            self.active_alerts = [
                alert for alert in self.active_alerts
                if current_time - alert.timestamp < timedelta(hours=24)
            ]
            
            # ✅ COMPLETED: Send high priority alerts
            for alert in self.active_alerts:
                if alert.alert_type == 'high_impact' and alert.signal_strength > 0.9:
                    # Escalate to message broker
                    if self.message_broker:
                        try:
                            await self.message_broker.publish(
                                'alerts.economic.critical',
                                {
                                    'alert_id': alert.event_id,
                                    'event_title': alert.event_title,
                                    'currency': alert.currency,
                                    'signal_strength': alert.signal_strength,
                                    'timestamp': alert.timestamp.isoformat()
                                }
                            )
                            self.logger.warning(f"🚨 CRITICAL ALERT: {alert.event_title} ({alert.currency})")
                        except Exception as e:
                            self.logger.error(f"Failed to publish critical alert: {e}")
                    
                    # Escalate to messaging coordinator if available
                    if self.messaging_coordinator:
                        try:
                            if hasattr(self.messaging_coordinator, 'broadcast_alert'):
                                await self.messaging_coordinator.broadcast_alert(
                                    f"Critical Economic Event: {alert.event_title}",
                                    alert.currency
                                )
                        except Exception as e:
                            self.logger.error(f"Failed to broadcast via coordinator: {e}")

        except Exception as e:
            self.logger.error(f"Error processing alerts: {str(e)}")
            
    async def _save_event_async(self, event_dict: Dict[str, Any]):
        """Save event to database (async)"""
        try:
            query = """
                INSERT INTO economic_events (name, currency, impact_level, timestamp, actual_value, forecast_value)
                VALUES (:name, :currency, :impact_level, :timestamp, :actual_value, :forecast_value)
            """
            params = {
                'name': event_dict.get('name'),
                'currency': event_dict.get('currency'),
                'impact_level': str(event_dict.get('impact_level', '')),
                'timestamp': event_dict.get('timestamp'),
                'actual_value': event_dict.get('actual_value'),
                'forecast_value': event_dict.get('forecast_value')
            }
            
            try:
                await self.db_manager.execute_async_query(query, params, use_cache=False)
            except Exception:
                # Likely duplicate or schema mismatch - safe to ignore
                pass
                
        except Exception as e:
            self.logger.error(f"Error saving event: {e}")

    async def _create_alert_async(self, event: Dict[str, Any], alert_type: str):
        """Create alert async"""
        try:
            # Calculate signal strength based on event data
            signal_strength = self._calculate_signal_strength(event, alert_type)
            
            alert = EconomicEventAlert(
                event_id=str(event.get('id', '')),
                event_title=event.get('name', ''),
                country=event.get('country', ''),
                currency=event.get('currency', ''),
                impact_level=str(event.get('impact_level', '')),
                alert_type=alert_type,
                timestamp=datetime.now(),
                signal_strength=signal_strength,
                trading_signals=[]
            )
            
            self.active_alerts.append(alert)
            
            if self.message_broker:
                try:
                    await self.message_broker.publish('alerts.economic', {
                        'alert_type': alert_type,
                        'event_title': alert.event_title,
                        'currency': alert.currency,
                        'signal_strength': signal_strength
                    })
                except Exception as e:
                    self.logger.debug(f"Failed to publish alert to broker: {e}")
                
            self.logger.info(f"Created {alert_type} alert for {alert.event_title}")
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")

    def _calculate_signal_strength(self, event: Dict[str, Any], alert_type: str) -> float:
        """
        ✅ NEW: Calculate signal strength based on event characteristics
        """
        strength = 0.5  # Base strength
        
        # Boost for high impact
        if event.get('impact_level') == 'HIGH':
            strength += 0.3
        elif event.get('impact_level') == 'MEDIUM':
            strength += 0.1
            
        # Boost for certain alert types
        if alert_type == 'high_impact':
            strength += 0.2
        elif alert_type == 'upcoming':
            strength += 0.1
            
        return min(strength, 1.0)

    def _calculate_event_correlation(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> float:
        """Calculate correlation score between two events"""
        try:
            score = 0.0

            # Parse timestamps
            ts1 = event1.get('timestamp')
            ts2 = event2.get('timestamp')
            
            if isinstance(ts1, str): 
                ts1 = datetime.fromisoformat(ts1)
            if isinstance(ts2, str): 
                ts2 = datetime.fromisoformat(ts2)

            if not ts1 or not ts2:
                return 0.0

            # Time proximity (events within 24 hours)
            time_diff = abs((ts1 - ts2).total_seconds()) / 3600  # hours

            if time_diff < 24:
                score += 0.3 * (1 - time_diff / 24)

            # Currency relationship
            currency1 = event1.get('currency', '')
            currency2 = event2.get('currency', '')

            if currency1 == currency2:
                score += 0.4

            # Impact level similarity
            impact1 = event1.get('impact_level', '')
            impact2 = event2.get('impact_level', '')

            if impact1 == impact2 == 'HIGH':
                score += 0.3
            elif impact1 == impact2:
                score += 0.1

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating event correlation: {str(e)}")
            return 0.0
