"""
MetaApi Provider for AUJ Platform - Linux Compatible

Complete MetaApi integration for Linux deployment.
Replaces MT5 direct connection with cloud-based MetaApi service.

This provider handles all trading operations, market data retrieval,
and real-time streaming through MetaApi's REST API and WebSocket.
"""

import asyncio
import aiohttp
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from decimal import Decimal
import logging

# Import base classes and data contracts
from .base_provider import (
    BaseDataProvider, 
    DataProviderType, 
    DataProviderCapabilities, 
    ConnectionStatus,
    DataType,
    Timeframe,
    DataProviderError,
    DataNotAvailableError,
    ConnectionError as ProviderConnectionError
)

try:
    from ..core.data_contracts import OHLCVData, TickData, NewsEvent
except ImportError:
    # Fallback data contracts
    OHLCVData = dict
    TickData = dict
    NewsEvent = dict

logger = logging.getLogger(__name__)


class MetaApiProvider(BaseDataProvider):
    """
    MetaApi cloud-based provider for Linux deployment.
    
    Provides complete MT5 functionality through MetaApi REST API and WebSocket.
    Supports all trading operations, market data, and account management.
    
    Features:
    - Cross-platform compatibility (Linux, Windows, macOS)
    - Real-time data streaming via WebSocket
    - Complete MT5 API functionality
    - Automatic reconnection and error handling
    - Comprehensive trading operations
    """

    def __init__(self, config_manager=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MetaApi Provider.
        
        Args:
            config_manager: Unified configuration manager
            config: Provider configuration dictionary (fallback)
        """
        super().__init__(
            name="MetaApi_Provider",
            provider_type=DataProviderType.MT5,
            config_manager=config_manager,
            config=config or {}
        )
        
        # MetaApi Configuration
        self.api_token = self._get_config('api_token')
        self.account_id = self._get_config('account_id')
        self.region = self._get_config('region', 'london')
        self.timeout = self._get_config('timeout', 30)
        self.retry_attempts = self._get_config('retry_attempts', 3)
        self.websocket_enabled = self._get_config('websocket_enabled', True)
        
        # API URLs
        self.provisioning_url = "https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai"
        self.client_url = "https://mt-client-api-v1.agiliumtrade.agiliumtrade.ai"
        self.streaming_url = "wss://mt-client-api-v1.agiliumtrade.agiliumtrade.ai"
        
        # Session and connection management
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self.connection_id: Optional[str] = None
        self._websocket_task: Optional[asyncio.Task] = None
        
        # Account and trading info
        self.account_info: Optional[Dict[str, Any]] = None
        self.symbols: List[str] = []
        self.symbol_info: Dict[str, Dict[str, Any]] = {}
        
        # Real-time data management
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.tick_subscribers: Dict[str, List[Callable]] = {}
        self.price_subscribers: Dict[str, List[Callable]] = {}
        
        # Request tracking
        self.request_id = 0
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # Connection monitoring
        self._last_heartbeat = datetime.now()
        self._heartbeat_interval = 30  # seconds
        self._reconnect_delay = 5  # seconds
        
        logger.info("MetaApi Provider initialized for Linux deployment")

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value from config manager or fallback config."""
        if self.config_manager and hasattr(self.config_manager, 'get_str'):
            if key in ['api_token', 'account_id', 'region']:
                return self.config_manager.get_str(f'metaapi.{key}', default)
            elif key in ['timeout', 'retry_attempts']:
                return self.config_manager.get_int(f'metaapi.{key}', default)
            elif key in ['websocket_enabled']:
                return self.config_manager.get_bool(f'metaapi.{key}', default)
        
        # Fallback to environment variables or config dict
        import os
        env_mapping = {
            'api_token': 'AUJ_METAAPI_TOKEN',
            'account_id': 'AUJ_METAAPI_ACCOUNT_ID',
            'region': 'AUJ_METAAPI_REGION',
            'timeout': 'AUJ_METAAPI_TIMEOUT'
        }
        
        if key in env_mapping:
            env_value = os.getenv(env_mapping[key])
            if env_value:
                if key in ['timeout', 'retry_attempts']:
                    return int(env_value)
                elif key in ['websocket_enabled']:
                    return env_value.lower() in ['true', '1', 'yes']
                return env_value
        
        return self.config.get('metaapi', {}).get(key, default)

    def _define_capabilities(self) -> DataProviderCapabilities:
        """Define MetaApi provider capabilities."""
        return DataProviderCapabilities(
            supports_ohlcv=True,
            supports_tick=True,
            supports_news=False,  # Not available through MetaApi
            supports_economic_calendar=False,  # Not available through MetaApi
            supports_order_book=True,
            supports_market_depth=True,
            supports_live=True,
            supports_historical=True,
            supported_symbols=getattr(self, 'symbols', []),  # Use empty list if symbols not loaded yet
            supported_timeframes=[
                Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30,
                Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1, Timeframe.MN1
            ]
        )

    async def connect(self) -> bool:
        """Connect to MetaApi service with comprehensive error handling."""
        try:
            self.connection_status = ConnectionStatus.CONNECTING
            logger.info("Connecting to MetaApi service...")

            # Validate configuration
            if not all([self.api_token, self.account_id]):
                raise ProviderConnectionError("MetaApi API token and account ID are required")

            # Create HTTP session with proper headers
            headers = {
                'auth-token': self.api_token,
                'Content-Type': 'application/json',
                'User-Agent': 'AUJ-Platform/1.0 (Linux; MetaApi)'
            }
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=100)
            )

            # Test connection and get account info
            account_info = await self._get_account_info()
            if not account_info.get('success', False):
                raise ProviderConnectionError("Failed to get account information")

            self.account_info = account_info
            logger.info(f"Account connected: {account_info.get('login', 'Unknown')}")
            
            # Get available symbols
            await self._update_symbols()
            logger.info(f"Loaded {len(self.symbols)} trading symbols")
            
            # Initialize WebSocket connection for real-time data
            if self.websocket_enabled:
                await self._connect_websocket()
            
            # Update capabilities with actual symbols
            self.capabilities = self._define_capabilities()
            
            self.connection_status = ConnectionStatus.CONNECTED
            self.successful_requests += 1
            logger.info(f"✅ MetaApi connected successfully. Account: {self.account_id}")
            return True

        except Exception as e:
            logger.error(f"❌ MetaApi connection failed: {e}")
            self.connection_status = ConnectionStatus.FAILED
            self.last_error = str(e)
            await self._cleanup_connection()
            return False

    async def disconnect(self):
        """Disconnect from MetaApi service."""
        try:
            self.connection_status = ConnectionStatus.DISCONNECTED
            await self._cleanup_connection()
            logger.info("✅ MetaApi disconnected successfully")
        except Exception as e:
            logger.error(f"❌ Error during MetaApi disconnect: {e}")

    async def _cleanup_connection(self):
        """Clean up connection resources."""
        # Stop WebSocket message handling
        if self._websocket_task and not self._websocket_task.done():
            self._websocket_task.cancel()
            try:
                await self._websocket_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None

        # Close HTTP session
        if self.session and not self.session.closed:
            try:
                await self.session.close()
            except Exception:
                pass
            self.session = None

        self.connection_id = None

    async def is_connected(self) -> bool:
        """Check if connected to MetaApi."""
        if self.connection_status != ConnectionStatus.CONNECTED:
            return False
            
        if not self.session or self.session.closed:
            return False
            
        # Perform a simple health check
        try:
            async with self.session.get(
                f"{self.client_url}/users/current/accounts/{self.account_id}/accountInformation",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except:
            return False

    async def _get_account_info(self) -> Dict[str, Any]:
        """Get account information from MetaApi."""
        try:
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            async with self.session.get(
                f"{self.client_url}/users/current/accounts/{self.account_id}/accountInformation"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "login": data.get("login"),
                        "balance": float(data.get("balance", 0.0)),
                        "equity": float(data.get("equity", 0.0)),
                        "margin": float(data.get("margin", 0.0)),
                        "free_margin": float(data.get("freeMargin", 0.0)),
                        "leverage": int(data.get("leverage", 1)),
                        "currency": data.get("currency", "USD"),
                        "server": data.get("server", ""),
                        "company": data.get("company", "")
                    }
                else:
                    error_data = await response.json()
                    return {
                        "success": False,
                        "error": error_data.get("message", "Failed to get account info")
                    }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {"success": False, "error": str(e)}

    async def _update_symbols(self):
        """Update available symbols from MetaApi."""
        try:
            self.request_count += 1
            async with self.session.get(
                f"{self.client_url}/users/current/accounts/{self.account_id}/symbols"
            ) as response:
                if response.status == 200:
                    symbols_data = await response.json()
                    self.symbols = [symbol["symbol"] for symbol in symbols_data]
                    
                    # Store symbol specifications
                    for symbol_data in symbols_data:
                        symbol = symbol_data["symbol"]
                        self.symbol_info[symbol] = {
                            "digits": symbol_data.get("digits", 5),
                            "point": symbol_data.get("point", 0.00001),
                            "lot_size": symbol_data.get("contractSize", 100000),
                            "min_lot": symbol_data.get("minVolume", 0.01),
                            "max_lot": symbol_data.get("maxVolume", 100.0),
                            "lot_step": symbol_data.get("volumeStep", 0.01),
                            "currency_base": symbol_data.get("currencyBase", ""),
                            "currency_profit": symbol_data.get("currencyProfit", ""),
                            "swap_long": symbol_data.get("swapLong", 0.0),
                            "swap_short": symbol_data.get("swapShort", 0.0),
                            "spread": symbol_data.get("spread", 0),
                            "trade_allowed": symbol_data.get("tradeAllowed", True)
                        }
                    
                    logger.info(f"Updated {len(self.symbols)} symbols from MetaApi")
                    self.successful_requests += 1
                else:
                    logger.warning(f"Failed to get symbols from MetaApi: {response.status}")
        except Exception as e:
            logger.error(f"Error updating symbols: {e}")
            self.last_error = str(e)

    async def _connect_websocket(self):
        """Connect to MetaApi WebSocket for real-time data."""
        try:
            ws_url = f"{self.streaming_url}/users/current/accounts/{self.account_id}/subscribe"
            headers = {'auth-token': self.api_token}
            
            self.websocket = await self.session.ws_connect(
                ws_url, 
                headers=headers,
                heartbeat=self._heartbeat_interval
            )
            
            # Start message handling task
            self._websocket_task = asyncio.create_task(self._handle_websocket_messages())
            
            logger.info("✅ MetaApi WebSocket connected")
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            self.websocket = None

    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages with error recovery."""
        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._process_websocket_message(data)
                        self._last_heartbeat = datetime.now()
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in WebSocket message: {msg.data}")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.websocket.exception()}")
                    break
                    
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    logger.warning("WebSocket connection closed")
                    break
                    
        except asyncio.CancelledError:
            logger.info("WebSocket task cancelled")
        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")
        finally:
            # Attempt reconnection if needed
            if self.connection_status == ConnectionStatus.CONNECTED:
                logger.info("Attempting WebSocket reconnection...")
                await asyncio.sleep(self._reconnect_delay)
                await self._connect_websocket()

    async def _process_websocket_message(self, data: Dict[str, Any]):
        """Process WebSocket message based on type."""
        msg_type = data.get("type", "")
        
        if msg_type == "prices":
            await self._handle_price_update(data)
        elif msg_type == "tick":
            await self._handle_tick_update(data)
        elif msg_type == "accountInformation":
            await self._handle_account_update(data)
        elif msg_type == "positions":
            await self._handle_positions_update(data)
        elif msg_type == "orders":
            await self._handle_orders_update(data)
        elif msg_type == "synchronization":
            await self._handle_synchronization(data)

    async def _handle_price_update(self, data: Dict[str, Any]):
        """Handle price update from WebSocket."""
        prices = data.get("prices", [])
        for price_data in prices:
            symbol = price_data.get("symbol")
            if symbol:
                self.price_cache[symbol] = {
                    "bid": float(price_data.get("bid", 0.0)),
                    "ask": float(price_data.get("ask", 0.0)),
                    "last": float(price_data.get("last", 0.0)),
                    "volume": float(price_data.get("volume", 0.0)),
                    "time": datetime.now(),
                    "spread": float(price_data.get("spread", 0.0))
                }
                
                # Notify price subscribers
                if symbol in self.price_subscribers:
                    for callback in self.price_subscribers[symbol]:
                        try:
                            await callback(self.price_cache[symbol])
                        except Exception as e:
                            logger.error(f"Error in price callback: {e}")

    async def _handle_tick_update(self, data: Dict[str, Any]):
        """Handle tick update from WebSocket."""
        symbol = data.get("symbol")
        if symbol and symbol in self.tick_subscribers:
            tick_data = {
                "symbol": symbol,
                "bid": float(data.get("bid", 0.0)),
                "ask": float(data.get("ask", 0.0)),
                "last": float(data.get("last", 0.0)),
                "volume": float(data.get("volume", 0.0)),
                "time": datetime.fromisoformat(data.get("time", "").replace("Z", "+00:00")),
                "spread": float(data.get("spread", 0.0))
            }
            
            # Notify subscribers
            for callback in self.tick_subscribers[symbol]:
                try:
                    await callback(tick_data)
                except Exception as e:
                    logger.error(f"Error in tick callback: {e}")

    async def _handle_account_update(self, data: Dict[str, Any]):
        """Handle account information update."""
        if "accountInformation" in data:
            account_data = data["accountInformation"]
            self.account_info.update({
                "balance": float(account_data.get("balance", 0.0)),
                "equity": float(account_data.get("equity", 0.0)),
                "margin": float(account_data.get("margin", 0.0)),
                "free_margin": float(account_data.get("freeMargin", 0.0))
            })
            logger.debug("Account information updated")

    async def _handle_positions_update(self, data: Dict[str, Any]):
        """Handle positions update."""
        positions = data.get("positions", [])
        logger.debug(f"Positions update: {len(positions)} positions")

    async def _handle_orders_update(self, data: Dict[str, Any]):
        """Handle orders update."""
        orders = data.get("orders", [])
        logger.debug(f"Orders update: {len(orders)} orders")

    async def _handle_synchronization(self, data: Dict[str, Any]):
        """Handle synchronization status."""
        sync_status = data.get("status", "")
        logger.debug(f"Synchronization status: {sync_status}")

    # Market Data Methods
    async def get_ohlcv_data(self,
                           symbol: str,
                           timeframe: Timeframe,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Get OHLCV data from MetaApi with retry logic."""
        if not await self.is_connected():
            raise ProviderConnectionError("Not connected to MetaApi")
            
        try:
            # Map timeframe to MetaApi format
            tf_mapping = {
                Timeframe.M1: "1m",
                Timeframe.M5: "5m", 
                Timeframe.M15: "15m",
                Timeframe.M30: "30m",
                Timeframe.H1: "1h",
                Timeframe.H4: "4h",
                Timeframe.D1: "1d",
                Timeframe.W1: "1w",
                Timeframe.MN1: "1M"
            }
            
            metaapi_timeframe = tf_mapping.get(timeframe, "1h")
            
            # Build request parameters
            params = {
                "symbol": symbol,
                "timeframe": metaapi_timeframe
            }
            
            if start_time:
                params["startTime"] = start_time.isoformat()
            if end_time:
                params["endTime"] = end_time.isoformat()
            if count:
                params["limit"] = min(count, 1000)  # API limit

            self.request_count += 1
            self.last_request_time = datetime.now()

            async with self.session.get(
                f"{self.client_url}/users/current/accounts/{self.account_id}/historical-market-data/candles",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data:
                        # Convert to DataFrame
                        df = pd.DataFrame(data)
                        
                        # Ensure proper column naming and types
                        df['time'] = pd.to_datetime(df['time'])
                        df.set_index('time', inplace=True)
                        
                        # Rename columns to standard format
                        column_mapping = {
                            'brokerTime': 'time',
                            'open': 'open',
                            'high': 'high', 
                            'low': 'low',
                            'close': 'close',
                            'tickVolume': 'volume',
                            'realVolume': 'real_volume'
                        }
                        
                        for old_col, new_col in column_mapping.items():
                            if old_col in df.columns:
                                df.rename(columns={old_col: new_col}, inplace=True)
                        
                        # Ensure required columns exist
                        required_cols = ['open', 'high', 'low', 'close', 'volume']
                        if all(col in df.columns for col in required_cols):
                            # Convert to proper numeric types
                            for col in required_cols:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            self.successful_requests += 1
                            logger.debug(f"✅ Retrieved {len(df)} candles for {symbol} {timeframe}")
                            return df[required_cols]  # Return only required columns
                        else:
                            logger.warning(f"Missing required columns in OHLCV data for {symbol}")
                            return None
                    else:
                        logger.warning(f"No OHLCV data returned for {symbol}")
                        return pd.DataFrame()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get OHLCV data: {response.status} - {error_text}")
                    self.last_error = f"API Error {response.status}: {error_text}"
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {symbol}: {e}")
            self.last_error = str(e)
            return None

    async def get_tick_data(self,
                          symbol: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Get tick data from MetaApi."""
        if not await self.is_connected():
            raise ProviderConnectionError("Not connected to MetaApi")
            
        try:
            params = {"symbol": symbol}
            
            if start_time:
                params["startTime"] = start_time.isoformat()
            if end_time:
                params["endTime"] = end_time.isoformat()
            if count:
                params["limit"] = min(count, 1000)  # API limit

            self.request_count += 1
            self.last_request_time = datetime.now()

            async with self.session.get(
                f"{self.client_url}/users/current/accounts/{self.account_id}/historical-market-data/ticks",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data:
                        df = pd.DataFrame(data)
                        df['time'] = pd.to_datetime(df['time'])
                        df.set_index('time', inplace=True)
                        
                        # Ensure numeric columns
                        numeric_cols = ['bid', 'ask', 'last', 'volume']
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        self.successful_requests += 1
                        logger.debug(f"✅ Retrieved {len(df)} ticks for {symbol}")
                        return df
                    else:
                        return pd.DataFrame()
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get tick data: {response.status} - {error_text}")
                    self.last_error = f"API Error {response.status}: {error_text}"
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting tick data for {symbol}: {e}")
            self.last_error = str(e)
            return None

    async def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price for symbol with caching."""
        try:
            # Check cache first (valid for 5 seconds)
            if symbol in self.price_cache:
                cached_price = self.price_cache[symbol]
                if (datetime.now() - cached_price["time"]).seconds < 5:
                    return cached_price

            # Fetch from API if cache miss or expired
            self.request_count += 1
            async with self.session.get(
                f"{self.client_url}/users/current/accounts/{self.account_id}/symbols/{symbol}/current-price"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    price_data = {
                        "bid": float(data.get("bid", 0.0)),
                        "ask": float(data.get("ask", 0.0)),
                        "last": float(data.get("last", 0.0)),
                        "volume": float(data.get("volume", 0.0)),
                        "spread": float(data.get("spread", 0.0)),
                        "time": datetime.now()
                    }
                    
                    # Update cache
                    self.price_cache[symbol] = price_data
                    self.successful_requests += 1
                    return price_data
                else:
                    logger.error(f"Failed to get current price: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None    # Trading Operations
    async def place_order(self,
                         symbol: str,
                         order_type: str,
                         volume: float,
                         price: Optional[float] = None,
                         sl: Optional[float] = None,
                         tp: Optional[float] = None,
                         comment: str = "",
                         magic: int = 0) -> Dict[str, Any]:
        """Place trading order via MetaApi."""
        if not await self.is_connected():
            raise ProviderConnectionError("Not connected to MetaApi")
            
        try:
            # Map order types to MetaApi format
            action_type = self._map_order_type(order_type)
            
            order_data = {
                "actionType": action_type,
                "symbol": symbol,
                "volume": volume,
                "comment": comment,
                "clientId": str(magic)
            }
            
            if price is not None:
                order_data["openPrice"] = price
            if sl is not None:
                order_data["stopLoss"] = sl
            if tp is not None:
                order_data["takeProfit"] = tp

            self.request_count += 1
            self.last_request_time = datetime.now()

            async with self.session.post(
                f"{self.client_url}/users/current/accounts/{self.account_id}/trade",
                json=order_data
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    self.successful_requests += 1
                    return {
                        "success": True,
                        "order_id": result.get("orderId"),
                        "deal_id": result.get("positionId"),
                        "volume": result.get("volume", volume),
                        "price": result.get("price", price),
                        "message": "Order placed successfully"
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("message", "Unknown error"),
                        "error_code": result.get("error", "API_ERROR")
                    }
                    
        except Exception as e:
            logger.error(f"Error placing order via MetaApi: {e}")
            self.last_error = str(e)
            return {"success": False, "error": str(e)}

    def _map_order_type(self, order_type: str) -> str:
        """Map internal order types to MetaApi format."""
        mapping = {
            "BUY": "ORDER_TYPE_BUY",
            "SELL": "ORDER_TYPE_SELL",
            "BUY_LIMIT": "ORDER_TYPE_BUY_LIMIT",
            "SELL_LIMIT": "ORDER_TYPE_SELL_LIMIT",
            "BUY_STOP": "ORDER_TYPE_BUY_STOP",
            "SELL_STOP": "ORDER_TYPE_SELL_STOP",
            "BUY_STOP_LIMIT": "ORDER_TYPE_BUY_STOP_LIMIT",
            "SELL_STOP_LIMIT": "ORDER_TYPE_SELL_STOP_LIMIT"
        }
        return mapping.get(order_type.upper(), "ORDER_TYPE_BUY")

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions via MetaApi."""
        if not await self.is_connected():
            raise ProviderConnectionError("Not connected to MetaApi")
            
        try:
            self.request_count += 1
            async with self.session.get(
                f"{self.client_url}/users/current/accounts/{self.account_id}/positions"
            ) as response:
                if response.status == 200:
                    positions = await response.json()
                    if symbol:
                        positions = [pos for pos in positions if pos.get("symbol") == symbol]
                    
                    self.successful_requests += 1
                    return positions
                else:
                    logger.error(f"Failed to get positions: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            self.last_error = str(e)
            return []

    async def get_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get pending orders via MetaApi."""
        if not await self.is_connected():
            raise ProviderConnectionError("Not connected to MetaApi")
            
        try:
            self.request_count += 1
            async with self.session.get(
                f"{self.client_url}/users/current/accounts/{self.account_id}/orders"
            ) as response:
                if response.status == 200:
                    orders = await response.json()
                    if symbol:
                        orders = [order for order in orders if order.get("symbol") == symbol]
                    
                    self.successful_requests += 1
                    return orders
                else:
                    logger.error(f"Failed to get orders: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            self.last_error = str(e)
            return []

    # Symbol and Account Information
    async def get_symbols(self) -> List[str]:
        """Get available symbols."""
        if not self.symbols:
            await self._update_symbols()
        return self.symbols

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol specification."""
        return self.symbol_info.get(symbol)

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get current account information."""
        if not await self.is_connected():
            return None
        return self.account_info

    # Subscription methods for real-time data
    async def subscribe_to_ticks(self, symbol: str, callback: Callable):
        """Subscribe to tick updates for a symbol."""
        if symbol not in self.tick_subscribers:
            self.tick_subscribers[symbol] = []
        self.tick_subscribers[symbol].append(callback)
        
        # Send subscription message via WebSocket if connected
        if self.websocket and not self.websocket.closed:
            subscription_msg = {
                "type": "subscribe",
                "symbol": symbol,
                "subscriptions": ["ticks"]
            }
            await self.websocket.send_str(json.dumps(subscription_msg))
            logger.debug(f"Subscribed to ticks for {symbol}")

    async def unsubscribe_from_ticks(self, symbol: str, callback: Callable):
        """Unsubscribe from tick updates."""
        if symbol in self.tick_subscribers:
            if callback in self.tick_subscribers[symbol]:
                self.tick_subscribers[symbol].remove(callback)
            
            # If no more subscribers, unsubscribe from WebSocket
            if not self.tick_subscribers[symbol]:
                del self.tick_subscribers[symbol]
                if self.websocket and not self.websocket.closed:
                    unsubscription_msg = {
                        "type": "unsubscribe",
                        "symbol": symbol,
                        "subscriptions": ["ticks"]
                    }
                    await self.websocket.send_str(json.dumps(unsubscription_msg))
                    logger.debug(f"Unsubscribed from ticks for {symbol}")

    async def subscribe_to_prices(self, symbol: str, callback: Callable):
        """Subscribe to price updates for a symbol."""
        if symbol not in self.price_subscribers:
            self.price_subscribers[symbol] = []
        self.price_subscribers[symbol].append(callback)
        
        # Send subscription message via WebSocket if connected
        if self.websocket and not self.websocket.closed:
            subscription_msg = {
                "type": "subscribe",
                "symbol": symbol,
                "subscriptions": ["quotes"]
            }
            await self.websocket.send_str(json.dumps(subscription_msg))
            logger.debug(f"Subscribed to prices for {symbol}")

    async def unsubscribe_from_prices(self, symbol: str, callback: Callable):
        """Unsubscribe from price updates."""
        if symbol in self.price_subscribers:
            if callback in self.price_subscribers[symbol]:
                self.price_subscribers[symbol].remove(callback)
            
            if not self.price_subscribers[symbol]:
                del self.price_subscribers[symbol]
                if self.websocket and not self.websocket.closed:
                    unsubscription_msg = {
                        "type": "unsubscribe",
                        "symbol": symbol,
                        "subscriptions": ["quotes"]
                    }
                    await self.websocket.send_str(json.dumps(unsubscription_msg))
                    logger.debug(f"Unsubscribed from prices for {symbol}")

    # Health and monitoring
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on MetaApi connection."""
        try:
            is_connected = await self.is_connected()
            
            health_data = {
                "provider_name": self.name,
                "provider_type": self.provider_type.value,
                "connection_status": self.connection_status.value,
                "is_connected": is_connected,
                "last_error": self.last_error,
                "account_id": self.account_id,
                "symbols_count": len(self.symbols),
                "websocket_connected": self.websocket is not None and not self.websocket.closed,
                "request_count": self.request_count,
                "successful_requests": self.successful_requests,
                "success_rate": (self.successful_requests / max(self.request_count, 1)) * 100,
                "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
                "region": self.region,
                "websocket_enabled": self.websocket_enabled
            }
            
            if is_connected:
                # Test with a simple API call
                account_info = await self._get_account_info()
                health_data["test_call_success"] = account_info.get('success', False)
                if account_info.get('success'):
                    health_data["account_balance"] = account_info.get('balance')
                    health_data["account_currency"] = account_info.get('currency')
            else:
                health_data["test_call_success"] = False

            return health_data
            
        except Exception as e:
            return {
                "provider_name": self.name,
                "provider_type": self.provider_type.value,
                "connection_status": ConnectionStatus.ERROR.value,
                "is_connected": False,
                "error": str(e),
                "test_call_success": False
            }

    def get_provider_info(self) -> Dict[str, Any]:
        """Get comprehensive provider information."""
        return {
            "provider_name": self.name,
            "provider_type": "METAAPI",
            "description": "MetaApi cloud-based MT5 provider for Linux deployment",
            "version": "1.0.0",
            "capabilities": {
                "supports_ohlcv": True,
                "supports_tick": True,
                "supports_trading": True,
                "supports_positions": True,
                "supports_account_info": True,
                "supports_symbols": True,
                "supports_websocket": True,
                "supports_real_time": True,
                "supports_historical": True,
                "cross_platform": True,
                "linux_compatible": True
            },
            "configuration": {
                "region": self.region,
                "account_id": self.account_id,
                "timeout": self.timeout,
                "retry_attempts": self.retry_attempts,
                "websocket_enabled": self.websocket_enabled
            },
            "status": {
                "connection_status": self.connection_status.value,
                "symbols_available": len(self.symbols),
                "request_count": self.request_count,
                "successful_requests": self.successful_requests,
                "last_error": self.last_error
            },
            "endpoints": {
                "provisioning_url": self.provisioning_url,
                "client_url": self.client_url,
                "streaming_url": self.streaming_url
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get provider usage statistics."""
        return {
            "requests": {
                "total": self.request_count,
                "successful": self.successful_requests,
                "failed": self.request_count - self.successful_requests,
                "success_rate": (self.successful_requests / max(self.request_count, 1)) * 100
            },
            "symbols": {
                "total_available": len(self.symbols),
                "cached_prices": len(self.price_cache),
                "tick_subscriptions": len(self.tick_subscribers),
                "price_subscriptions": len(self.price_subscribers)
            },
            "connection": {
                "status": self.connection_status.value,
                "last_request": self.last_request_time.isoformat() if self.last_request_time else None,
                "websocket_active": self.websocket is not None and not self.websocket.closed,
                "last_heartbeat": self._last_heartbeat.isoformat()
            }
        }


# Factory function for easy provider creation
async def create_metaapi_provider(config: Dict[str, Any]) -> MetaApiProvider:
    """
    Create and initialize MetaApi provider.
    
    Args:
        config: Configuration dictionary containing MetaApi settings
        
    Returns:
        Initialized MetaApiProvider instance
        
    Raises:
        ProviderConnectionError: If connection fails
    """
    provider = MetaApiProvider(config=config)
    if await provider.connect():
        return provider
    else:
        raise ProviderConnectionError("Failed to connect to MetaApi")


# Testing and validation functions
async def test_metaapi_provider():
    """Test MetaApi provider functionality."""
    import os
    
    # Load configuration from environment variables
    test_config = {
        "metaapi": {
            "api_token": os.getenv("AUJ_METAAPI_TOKEN"),
            "account_id": os.getenv("AUJ_METAAPI_ACCOUNT_ID"),
            "region": os.getenv("AUJ_METAAPI_REGION", "london"),
            "timeout": int(os.getenv("AUJ_METAAPI_TIMEOUT", "30"))
        }
    }
    
    if not all([test_config["metaapi"]["api_token"], test_config["metaapi"]["account_id"]]):
        print("❌ MetaApi credentials not found in environment variables")
        print("   Set AUJ_METAAPI_TOKEN and AUJ_METAAPI_ACCOUNT_ID")
        return False
    
    try:
        print("🚀 Testing MetaApi Provider...")
        provider = await create_metaapi_provider(test_config)
        
        # Test account info
        account_info = await provider.get_account_info()
        print(f"✅ Account Info: Balance={account_info.get('balance')}, Currency={account_info.get('currency')}")
        
        # Test symbols
        symbols = await provider.get_symbols()
        print(f"✅ Available symbols: {len(symbols)}")
        
        # Test OHLCV data for a common symbol
        if symbols:
            test_symbol = "EURUSD" if "EURUSD" in symbols else symbols[0]
            ohlcv = await provider.get_ohlcv_data(
                test_symbol, 
                Timeframe.H1, 
                count=10
            )
            if ohlcv is not None:
                print(f"✅ OHLCV data for {test_symbol}: {len(ohlcv)} candles")
            else:
                print(f"⚠️ No OHLCV data for {test_symbol}")
        
        # Test current price
        if symbols:
            price = await provider.get_current_price(test_symbol)
            if price:
                print(f"✅ Current price for {test_symbol}: Bid={price['bid']}, Ask={price['ask']}")
        
        # Health check
        health = await provider.health_check()
        print(f"✅ Health check: Connected={health['is_connected']}, Success Rate={health['success_rate']:.1f}%")
        
        await provider.disconnect()
        print("✅ MetaApi provider test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_metaapi_provider())