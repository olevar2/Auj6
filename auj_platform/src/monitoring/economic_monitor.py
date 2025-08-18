"""
Economic Calendar Real-Time Monitoring Service
Monitors economic events and triggers real-time analysis and trading signals
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
    print("[WARNING] 'schedule' module not available - Economic Monitor scheduling will be disabled")
from dataclasses import dataclass

# Import economic calendar components
from ..data_providers.unified_news_economic_provider import UnifiedNewsEconomicProvider
from ..analytics.indicators.economic.economic_event_impact_indicator import EconomicEventImpactIndicator
from ..analytics.indicators.economic.news_sentiment_impact_indicator import NewsSentimentImpactIndicator
from ..analytics.indicators.economic.event_volatility_predictor import EventVolatilityPredictor
from ..analytics.indicators.economic.economic_calendar_confluence_indicator import EconomicCalendarConfluenceIndicator
from ..analytics.indicators.economic.fundamental_momentum_indicator import FundamentalMomentumIndicator
from ..core.unified_database_manager import UnifiedDatabaseManager

# Import messaging and alerts
try:
    from ..messaging.message_broker import MessageBroker
    # Note: real_time_coordinator module doesn't exist yet, using message_broker only
    MESSAGING_AVAILABLE = True
except ImportError:
    MESSAGING_AVAILABLE = False
    print("[WARNING] Messaging components not available - Economic Monitor will run without messaging")


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
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the economic monitor"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize config manager
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()

        # Initialize database manager
        self.db_manager = UnifiedDatabaseManager()

        # Initialize message broker if available
        if MESSAGING_AVAILABLE:
            self.message_broker = MessageBroker()
        else:
            self.message_broker = None

        # Initialize messaging coordinator if available
        try:
            from ..coordination.messaging_coordinator import MessagingCoordinator
            self.messaging_coordinator = MessagingCoordinator()
        except (ImportError, TypeError) as e:
            self.messaging_coordinator = None
            self.logger.warning(f"MessagingCoordinator not available - using basic messaging only: {e}")

        # Initialize providers
        self.providers = {}
        try:
            self.providers['economic_calendar'] = UnifiedNewsEconomicProvider()
        except (TypeError, ImportError) as e:
            self.logger.warning(f"UnifiedNewsEconomicProvider not available: {e}")
            self.providers['economic_calendar'] = None

        # Initialize indicators
        self.indicators = {}
        try:
            self.indicators['event_impact'] = EconomicEventImpactIndicator()
        except (TypeError, ImportError) as e:
            self.logger.warning(f"EconomicEventImpactIndicator not available: {e}")

        try:
            self.indicators['news_sentiment'] = NewsSentimentImpactIndicator()
        except (TypeError, ImportError) as e:
            self.logger.warning(f"NewsSentimentImpactIndicator not available: {e}")

        try:
            self.indicators['volatility_predictor'] = EventVolatilityPredictor()
        except (TypeError, ImportError) as e:
            self.logger.warning(f"EventVolatilityPredictor not available: {e}")

        try:
            self.indicators['confluence'] = EconomicCalendarConfluenceIndicator()
        except (TypeError, ImportError) as e:
            self.logger.warning(f"EconomicCalendarConfluenceIndicator not available: {e}")

        try:
            self.indicators['momentum'] = FundamentalMomentumIndicator()
        except (TypeError, ImportError) as e:
            self.logger.warning(f"FundamentalMomentumIndicator not available: {e}")

        # Monitoring configuration
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_update = None
        self.update_interval = self.config_manager.get_int('update_interval', 60)  # seconds
        self.alert_thresholds = self.config_manager.get_dict('alert_thresholds', {
            'high_impact_score': 0.8,
            'volatility_threshold': 0.7,
            'confluence_threshold': 0.6
        })

        # Event cache for deduplication
        self.processed_events = set()
        self.active_alerts = []

        self.logger.info("Economic Monitor initialized successfully")


    async def initialize(self) -> None:
        """Initialize the component with required dependencies."""
        pass  # TODO: Implement initialization logic

    def start_monitoring(self):
        """Initialize the monitoring service (no longer runs independent loop)"""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return

        self.monitoring_active = True
        self.logger.info("Economic monitoring initialized - will be called by orchestrator")

    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.monitoring_active = False
        self.logger.info("Economic monitoring stopped")

    async def execute_monitoring_cycle(self):
        """
        Execute a monitoring cycle (called by orchestrator).

        This method replaces the old threading loop and should be called
        periodically by the DailyFeedbackLoop.
        """
        if not self.monitoring_active:
            return

        try:
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
            self.logger.info("Refreshing economic data from providers")

            all_events = []

            # Fetch from all providers
            for provider_name, provider in self.providers.items():
                try:
                    # Convert sync calls to async if needed
                    events = await self._get_events_from_provider_async(provider)
                    all_events.extend(events)
                    self.logger.info(f"Fetched {len(events)} events from {provider_name}")
                except Exception as e:
                    self.logger.error(f"Error fetching from {provider_name}: {str(e)}")

            # Save events to database
            new_events = 0
            for event in all_events:
                event_id = f"{event.get('title', '')}_{event.get('date_time', '')}_{event.get('country', '')}"
                if event_id not in self.processed_events:
                    await self._save_event_async(event)
                    self.processed_events.add(event_id)
                    new_events += 1

            self.logger.info(f"Processed {new_events} new economic events")

        except Exception as e:
            self.logger.error(f"Error refreshing economic data: {str(e)}")

    async def analyze_event_correlations(self):
        """Analyze event correlations (async version)"""
        try:
            self.logger.info("Analyzing event correlations")
            # Implementation would go here
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {str(e)}")

    async def cleanup_old_events(self):
        """Clean up old events (async version)"""
        try:
            self.logger.info("Cleaning up old events")
            # Implementation would go here
        except Exception as e:
            self.logger.error(f"Error cleaning up events: {str(e)}")

    async def _check_upcoming_events_async(self):
        """Async version of upcoming events check"""
        try:
            # Get events in the next 24 hours
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=24)

            # Use async database call if available, otherwise wrap sync call
            upcoming_events = await asyncio.get_event_loop().run_in_executor(
                None,
                self.db_manager.get_economic_events,
                start_time,
                end_time,
                'High'
            )

            for event in upcoming_events:
                event_time = datetime.fromisoformat(event.get('date_time', ''))
                time_to_event = event_time - datetime.now()

                # Alert for events happening in the next hour
                if timedelta(minutes=30) <= time_to_event <= timedelta(hours=1):
                    await self._create_alert_async(event, 'upcoming')

        except Exception as e:
            self.logger.error(f"Error checking upcoming events: {str(e)}")

    async def _monitor_released_events_async(self):
        """Async version of released events monitoring"""
        try:
            # Get events released in the last hour
            start_time = datetime.now() - timedelta(hours=1)
            end_time = datetime.now()

            # Use async database call if available, otherwise wrap sync call
            recent_events = await asyncio.get_event_loop().run_in_executor(
                None,
                self.db_manager.get_economic_events,
                start_time,
                end_time
            )

            for event in recent_events:
                # Check if event has actual value (indicating it was released)
                if event.get('actual_value') and event.get('actual_value') != 'N/A':
                    await self._analyze_event_impact_async(event)

        except Exception as e:
            self.logger.error(f"Error monitoring released events: {str(e)}")

    async def _generate_realtime_signals_async(self):
        """Async version of signal generation"""
        try:
            # Get recent events and current market conditions
            recent_events = await asyncio.get_event_loop().run_in_executor(
                None,
                self.db_manager.get_economic_events,
                datetime.now() - timedelta(hours=6),
                datetime.now() + timedelta(hours=6)
            )

            # Major currency pairs to analyze
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']

            for pair in major_pairs:
                try:
                    # Calculate confluence and momentum asynchronously
                    confluence_task = asyncio.create_task(
                        self._calculate_indicator_async('confluence', {
                            'economic_events': recent_events,
                            'current_time': datetime.now(),
                            'currency_pair': pair
                        })
                    )

                    momentum_task = asyncio.create_task(
                        self._calculate_indicator_async('momentum', {
                            'economic_events': recent_events,
                            'current_time': datetime.now(),
                            'currency_pair': pair
                        })
                    )

                    # Wait for both calculations
                    confluence_data, momentum_data = await asyncio.gather(
                        confluence_task, momentum_task
                    )

                    # Generate trading signals if confluence is high
                    if confluence_data.get('confluence_score', 0) > self.alert_thresholds['confluence_threshold']:
                        signal = {
                            'currency_pair': pair,
                            'confluence_score': confluence_data.get('confluence_score'),
                            'momentum_score': momentum_data.get('momentum_score'),
                            'signal_strength': (confluence_data.get('confluence_score', 0) +
                                              momentum_data.get('momentum_score', 0)) / 2,
                            'timestamp': datetime.now().isoformat(),
                            'signal_type': 'economic_confluence'
                        }

                        # Send signal to message broker
                        await self._send_signal_async(signal)

                except Exception as pair_error:
                    self.logger.error(f"Error processing {pair}: {str(pair_error)}")

        except Exception as e:
            self.logger.error(f"Error generating real-time signals: {str(e)}")

    async def _process_alerts_async(self):
        """Async version of alert processing"""
        try:
            # Remove old alerts (older than 24 hours)
            current_time = datetime.now()
            self.active_alerts = [
                alert for alert in self.active_alerts
                if current_time - alert.timestamp < timedelta(hours=24)
            ]

            # Process any pending alert escalations
            for alert in self.active_alerts:
                if alert.alert_type == 'high_impact' and alert.signal_strength > 0.9:
                    # Send high priority alert
                    await self._send_priority_alert_async(alert)

        except Exception as e:
            self.logger.error(f"Error processing alerts: {str(e)}")

    async def _get_events_from_provider_async(self, provider):
        """Get events from provider (async wrapper)"""
        try:
            # Check if provider has async methods
            if hasattr(provider, 'get_economic_events_async'):
                return await provider.get_economic_events_async()
            elif hasattr(provider, 'get_economic_events'):
                # Wrap sync call in executor
                return await asyncio.get_event_loop().run_in_executor(
                    None, provider.get_economic_events
                )
            else:
                self.logger.warning(f"Provider {provider.__class__.__name__} has no economic events method")
                return []
        except Exception as e:
            self.logger.error(f"Error getting events from provider: {str(e)}")
            return []

    async def _save_event_async(self, event):
        """Save event to database (async version)"""
        try:
            # Use async database save if available, otherwise wrap sync call
            if hasattr(self.db_manager, 'save_economic_event_async'):
                await self.db_manager.save_economic_event_async(event)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.db_manager.save_economic_event, event
                )
            self.logger.debug(f"Saved economic event: {event.get('title', 'Unknown')}")
        except Exception as e:
            self.logger.error(f"Error saving event to database: {str(e)}")

    def _refresh_economic_data(self):
        """Refresh economic data from all providers"""
        try:
            self.logger.info("Refreshing economic data from providers")

            all_events = []

            # Fetch from all providers
            for provider_name, provider in self.providers.items():
                try:
                    events = provider.get_economic_events()
                    all_events.extend(events)
                    self.logger.info(f"Fetched {len(events)} events from {provider_name}")
                except Exception as e:
                    self.logger.error(f"Error fetching from {provider_name}: {str(e)}")

            # Save events to database
            new_events = 0
            for event in all_events:
                event_id = f"{event.get('title', '')}_{event.get('date_time', '')}_{event.get('country', '')}"
                if event_id not in self.processed_events:
                    self.db_manager.save_economic_event(event)
                    self.processed_events.add(event_id)
                    new_events += 1

            self.logger.info(f"Saved {new_events} new economic events")
            self.last_update = datetime.now()

        except Exception as e:
            self.logger.error(f"Error refreshing economic data: {str(e)}")

    def _check_upcoming_events(self):
        """Check for upcoming high-impact events"""
        try:
            # Get events in the next 24 hours
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=24)

            upcoming_events = self.db_manager.get_economic_events(
                start_date=start_time,
                end_date=end_time,
                impact_level='High'
            )

            for event in upcoming_events:
                event_time = datetime.fromisoformat(event.get('date_time', ''))
                time_to_event = event_time - datetime.now()

                # Alert for events happening in the next hour
                if timedelta(minutes=30) <= time_to_event <= timedelta(hours=1):
                    self._create_alert(event, 'upcoming')

        except Exception as e:
            self.logger.error(f"Error checking upcoming events: {str(e)}")

    def _monitor_released_events(self):
        """Monitor recently released economic events"""
        try:
            # Get events released in the last hour
            start_time = datetime.now() - timedelta(hours=1)
            end_time = datetime.now()

            recent_events = self.db_manager.get_economic_events(
                start_date=start_time,
                end_date=end_time
            )

            for event in recent_events:
                # Check if event has actual value (indicating it was released)
                if event.get('actual_value') and event.get('actual_value') != 'N/A':
                    self._analyze_event_impact(event)

        except Exception as e:
            self.logger.error(f"Error monitoring released events: {str(e)}")

    def _analyze_event_impact(self, event: Dict[str, Any]):
        """Analyze the impact of a released economic event"""
        try:
            # Calculate event impact using indicators
            impact_scores = {}

            for indicator_name, indicator in self.indicators.items():
                try:
                    result = indicator.calculate({
                        'economic_events': [event],
                        'current_time': datetime.now(),
                        'currency_pair': 'EURUSD'  # Default pair, should be configurable
                    })
                    if result:
                        impact_scores[indicator_name] = result.get('signal_strength', 0)
                except Exception as e:
                    self.logger.error(f"Error calculating {indicator_name}: {str(e)}")

            # Determine overall impact
            overall_impact = sum(impact_scores.values()) / len(impact_scores) if impact_scores else 0

            # Save impact analysis
            impact_data = {
                'economic_event_id': event.get('id'),
                'impact_score': overall_impact,
                'indicator_scores': impact_scores,
                'timestamp': datetime.now().isoformat(),
                'currency_pair': event.get('currency', 'USD')
            }

            self.db_manager.save_economic_event_impact(impact_data)

            # Create alert if high impact
            if overall_impact > self.alert_thresholds['high_impact_score']:
                self._create_alert(event, 'high_impact', overall_impact)

        except Exception as e:
            self.logger.error(f"Error analyzing event impact: {str(e)}")

    def _generate_realtime_signals(self):
        """Generate real-time trading signals based on economic events"""
        try:
            # Get recent events and current market conditions
            recent_events = self.db_manager.get_economic_events(
                start_date=datetime.now() - timedelta(hours=6),
                end_date=datetime.now() + timedelta(hours=6)
            )

            # Major currency pairs to analyze
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']

            for pair in major_pairs:
                try:
                    # Calculate confluence and momentum
                    confluence_data = self.indicators['confluence'].calculate({
                        'economic_events': recent_events,
                        'current_time': datetime.now(),
                        'currency_pair': pair
                    })

                    momentum_data = self.indicators['momentum'].calculate({
                        'economic_events': recent_events,
                        'current_time': datetime.now(),
                        'currency_pair': pair
                    })

                    volatility_data = self.indicators['volatility_predictor'].calculate({
                        'economic_events': recent_events,
                        'current_time': datetime.now(),
                        'currency_pair': pair
                    })

                    # Generate signal if conditions are met
                    if (confluence_data and confluence_data.get('signal_strength', 0) > self.alert_thresholds['confluence_threshold'] and
                        momentum_data and abs(momentum_data.get('signal_strength', 0) - 0.5) > 0.2):

                        signal_type = 'BUY' if momentum_data.get('signal_strength', 0) > 0.5 else 'SELL'

                        signal_data = {
                            'currency_pair': pair,
                            'signal_type': signal_type,
                            'signal_strength': confluence_data.get('signal_strength', 0),
                            'confidence_score': momentum_data.get('signal_strength', 0),
                            'expected_volatility': volatility_data.get('signal_strength', 0) if volatility_data else 0,
                            'source': 'economic_monitor',
                            'timestamp': datetime.now().isoformat(),
                            'related_events': [e.get('id') for e in recent_events if e.get('currency') in pair]
                        }

                        # Save signal
                        self.db_manager.save_economic_trading_signal(signal_data)

                        # Send real-time notification
                        self._send_signal_notification(signal_data)

                except Exception as e:
                    self.logger.error(f"Error generating signals for {pair}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error generating real-time signals: {str(e)}")

    def _create_alert(self, event: Dict[str, Any], alert_type: str, signal_strength: float = 0):
        """Create an economic event alert"""
        try:
            alert = EconomicEventAlert(
                event_id=event.get('id', ''),
                event_title=event.get('title', ''),
                country=event.get('country', ''),
                currency=event.get('currency', ''),
                impact_level=event.get('impact_level', ''),
                alert_type=alert_type,
                timestamp=datetime.now(),
                signal_strength=signal_strength,
                trading_signals=[]
            )

            self.active_alerts.append(alert)

            # Send alert through message broker
            alert_message = {
                'type': 'economic_alert',
                'data': {
                    'event_title': alert.event_title,
                    'country': alert.country,
                    'currency': alert.currency,
                    'impact_level': alert.impact_level,
                    'alert_type': alert_type,
                    'signal_strength': signal_strength
                }
            }

            if self.message_broker:
                self.message_broker.publish('alerts.economic', alert_message)
            self.logger.info(f"Created {alert_type} alert for {alert.event_title}")

        except Exception as e:
            self.logger.error(f"Error creating alert: {str(e)}")

    def _send_signal_notification(self, signal_data: Dict[str, Any]):
        """Send trading signal notification"""
        try:
            notification = {
                'type': 'trading_signal',
                'data': signal_data
            }

            if self.message_broker:
                self.message_broker.publish('signals.economic', notification)
            
            if self.messaging_coordinator:
                self.messaging_coordinator.broadcast_signal(signal_data)

            self.logger.info(f"Sent trading signal: {signal_data['signal_type']} {signal_data['currency_pair']}")

        except Exception as e:
            self.logger.error(f"Error sending signal notification: {str(e)}")

    def _process_alerts(self):
        """Process and cleanup active alerts"""
        try:
            # Remove old alerts (older than 24 hours)
            current_time = datetime.now()
            self.active_alerts = [
                alert for alert in self.active_alerts
                if current_time - alert.timestamp < timedelta(hours=24)
            ]

        except Exception as e:
            self.logger.error(f"Error processing alerts: {str(e)}")

    def _analyze_event_correlations(self):
        """Analyze correlations between economic events"""
        try:
            # Get events from the last 30 days
            start_date = datetime.now() - timedelta(days=30)
            events = self.db_manager.get_economic_events(start_date=start_date)

            # Analyze correlations between events
            correlations = []

            for i, event1 in enumerate(events):
                for event2 in events[i+1:]:
                    # Calculate correlation based on timing, impact, and currency
                    correlation_score = self._calculate_event_correlation(event1, event2)

                    if correlation_score > 0.3:  # Threshold for significant correlation
                        correlation_data = {
                            'event_1_id': event1.get('id'),
                            'event_2_id': event2.get('id'),
                            'event_1_title': event1.get('title'),
                            'event_2_title': event2.get('title'),
                            'correlation_score': correlation_score,
                            'correlation_type': 'temporal_impact',
                            'timestamp': datetime.now().isoformat()
                        }
                        correlations.append(correlation_data)

            # Save correlations
            for correlation in correlations:
                self.db_manager.save_economic_event_correlation(correlation)

            self.logger.info(f"Analyzed and saved {len(correlations)} event correlations")

        except Exception as e:
            self.logger.error(f"Error analyzing event correlations: {str(e)}")

    def _calculate_event_correlation(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> float:
        """Calculate correlation score between two events"""
        try:
            score = 0.0

            # Time proximity (events within 24 hours)
            time1 = datetime.fromisoformat(event1.get('date_time', ''))
            time2 = datetime.fromisoformat(event2.get('date_time', ''))
            time_diff = abs(time1 - time2).total_seconds() / 3600  # hours

            if time_diff < 24:
                score += 0.3 * (1 - time_diff / 24)

            # Currency relationship
            currency1 = event1.get('currency', '')
            currency2 = event2.get('currency', '')

            if currency1 == currency2:
                score += 0.4
            elif currency1 in ['USD', 'EUR', 'GBP', 'JPY'] and currency2 in ['USD', 'EUR', 'GBP', 'JPY']:
                score += 0.2

            # Impact level similarity
            impact1 = event1.get('impact_level', '')
            impact2 = event2.get('impact_level', '')

            if impact1 == impact2 == 'High':
                score += 0.3
            elif impact1 == impact2:
                score += 0.1

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating event correlation: {str(e)}")
            return 0.0

    def _cleanup_old_events(self):
        """Clean up old events and performance data"""
        try:
            # Remove events older than 90 days
            cutoff_date = datetime.now() - timedelta(days=90)

            # This would typically be implemented in the database manager
            self.logger.info("Cleaning up old economic events")

        except Exception as e:
            self.logger.error(f"Error cleaning up old events: {str(e)}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'active': self.monitoring_active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'active_alerts': len(self.active_alerts),
            'processed_events': len(self.processed_events),
            'update_interval': self.update_interval,
            'providers_active': len(self.providers),
            'indicators_active': len(self.indicators)
        }

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_alerts = [
            {
                'event_title': alert.event_title,
                'country': alert.country,
                'currency': alert.currency,
                'impact_level': alert.impact_level,
                'alert_type': alert.alert_type,
                'timestamp': alert.timestamp.isoformat(),
                'signal_strength': alert.signal_strength
            }
            for alert in self.active_alerts
            if alert.timestamp > cutoff_time
        ]

        return recent_alerts


# Service factory function
def create_economic_monitor(config: Dict[str, Any] = None) -> EconomicMonitor:
    """Create and configure an economic monitor instance"""
    return EconomicMonitor(config)


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Economic Calendar Monitor')
    parser.add_argument('--start', action='store_true', help='Start monitoring')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--test', action='store_true', help='Run test monitoring cycle')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    monitor = create_economic_monitor()

    if args.start:
        print("Starting economic monitor...")
        monitor.start_monitoring()
        try:
            while True:
                time.sleep(10)
                status = monitor.get_monitoring_status()
                print(f"Status: {status}")
        except KeyboardInterrupt:
            print("Stopping monitor...")
            monitor.stop_monitoring()

    elif args.status:
        status = monitor.get_monitoring_status()
        print(f"Monitor Status: {status}")

    elif args.test:
        print("Running test monitoring cycle...")
        monitor._refresh_economic_data()
        monitor._check_upcoming_events()
        monitor._generate_realtime_signals()
        print("Test completed")
    async def _create_alert_async(self, event: Dict[str, Any], alert_type: str):
        """Create an alert for an economic event (async version)"""
        try:
            alert = EconomicEventAlert(
                event_id=event.get('id', ''),
                event_title=event.get('title', ''),
                country=event.get('country', ''),
                currency=event.get('currency', ''),
                impact_level=event.get('impact', ''),
                alert_type=alert_type,
                timestamp=datetime.now(),
                signal_strength=float(event.get('impact_score', 0.5)),
                trading_signals=[]
            )

            self.active_alerts.append(alert)

            # Send alert to message broker
            await self._send_alert_async(alert)

        except Exception as e:
            self.logger.error(f"Error creating alert: {str(e)}")

    async def _analyze_event_impact_async(self, event: Dict[str, Any]):
        """Analyze the impact of a released economic event (async version)"""
        try:
            # Calculate impact using indicators
            impact_data = await self._calculate_indicator_async('event_impact', {
                'event': event,
                'current_time': datetime.now()
            })

            # If impact is significant, create alert
            if impact_data.get('impact_score', 0) > self.alert_thresholds['high_impact_score']:
                await self._create_alert_async(event, 'high_impact')

        except Exception as e:
            self.logger.error(f"Error analyzing event impact: {str(e)}")

    async def _calculate_indicator_async(self, indicator_name: str, data: Dict[str, Any]):
        """Calculate indicator value asynchronously"""
        try:
            indicator = self.indicators.get(indicator_name)
            if not indicator:
                return {}

            # Run indicator calculation in executor if it's CPU intensive
            return await asyncio.get_event_loop().run_in_executor(
                None, indicator.calculate, data
            )
        except Exception as e:
            self.logger.error(f"Error calculating {indicator_name}: {str(e)}")
            return {}

    async def _send_signal_async(self, signal: Dict[str, Any]):
        """Send trading signal via message broker (async)"""
        try:
            if hasattr(self.message_broker, 'send_message_async'):
                await self.message_broker.send_message_async('economic_signals', signal)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.message_broker.send_message, 'economic_signals', signal
                )
        except Exception as e:
            self.logger.error(f"Error sending signal: {str(e)}")

    async def _send_alert_async(self, alert: EconomicEventAlert):
        """Send alert via message broker (async)"""
        try:
            alert_data = {
                'type': 'economic_alert',
                'event_id': alert.event_id,
                'title': alert.event_title,
                'country': alert.country,
                'currency': alert.currency,
                'alert_type': alert.alert_type,
                'signal_strength': alert.signal_strength,
                'timestamp': alert.timestamp.isoformat()
            }

            if hasattr(self.message_broker, 'send_message_async'):
                await self.message_broker.send_message_async('economic_alerts', alert_data)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.message_broker.send_message, 'economic_alerts', alert_data
                )
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")

    async def _send_priority_alert_async(self, alert: EconomicEventAlert):
        """Send high priority alert (async)"""
        try:
            priority_data = {
                'type': 'priority_economic_alert',
                'event_id': alert.event_id,
                'title': alert.event_title,
                'signal_strength': alert.signal_strength,
                'timestamp': alert.timestamp.isoformat(),
                'urgent': True
            }

            if self.messaging_coordinator and hasattr(self.messaging_coordinator, 'send_priority_message_async'):
                await self.messaging_coordinator.send_priority_message_async(priority_data)
            elif self.messaging_coordinator:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.messaging_coordinator.send_priority_message, priority_data
                )
        except Exception as e:
            self.logger.error(f"Error sending priority alert: {str(e)}")
