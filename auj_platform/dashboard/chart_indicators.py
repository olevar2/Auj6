# chart_indicators.py
"""
Advanced Technical Indicators Library for AUJ Platform
Supports AI-Enhanced indicators with multiple categories
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Some advanced indicators will use fallback implementations.")

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AdvancedIndicators:
    """Advanced Technical Indicators with AI Enhancement"""
    
    def __init__(self):
        self.indicator_categories = {
            'Fractals': [
                'Williams Fractals', 'Fractal Dimension', 'Fractal Adaptive MA',
                'Fractal Energy', 'AI Fractal Pattern', 'Multi-Timeframe Fractals'
            ],
            'Gann': [
                'Gann Fan', 'Gann Square', 'Gann Angles', 'Gann HiLo',
                'AI Gann Levels', 'Dynamic Gann Grid'
            ],
            'Fibonacci': [
                'Fibonacci Retracements', 'Fibonacci Extensions', 'Fibonacci Arcs',
                'Fibonacci Time Zones', 'AI Fibonacci Clusters', 'Auto Fibonacci'
            ],
            'Elliott Wave': [
                'Elliott Wave Count', 'Wave Pattern Recognition', 'Impulse Waves',
                'Corrective Waves', 'AI Elliott Prediction', 'Wave Degree Analysis'
            ],
            'Pivot Points': [
                'Classic Pivots', 'Fibonacci Pivots', 'Camarilla Pivots',
                'Woodie Pivots', 'AI Dynamic Pivots', 'Multi-Timeframe Pivots'
            ],
            'AI-Enhanced': [
                'LSTM Price Predictor', 'Neural Harmonic Resonance', 'Chaos Geometry',
                'Quantum Oscillator', 'AI Sentiment Analysis', 'Deep Learning RSI'
            ],
            'Volume': [
                'Volume Profile', 'VWAP Bands', 'Accumulation/Distribution',
                'Money Flow Index', 'AI Volume Prediction', 'Smart Money Index'
            ],
            'Momentum': [
                'RSI Multi-Timeframe', 'Stochastic Fast/Slow', 'MACD Histogram',
                'CCI', 'Williams %R', 'AI Momentum Fusion'
            ],
            'Trend': [
                'Adaptive MA', 'Hull MA', 'T3 MA', 'Zero Lag MA',
                'AI Trend Detector', 'Dynamic Support/Resistance'
            ],
            'Volatility': [
                'Bollinger Bands', 'Keltner Channels', 'ATR Bands',
                'Donchian Channels', 'AI Volatility Forecast', 'VIX Correlation'
            ],
            'Oscillators': [
                'Awesome Oscillator', 'DeMarker', 'Force Index',
                'Ultimate Oscillator', 'AI Oscillator Fusion', 'Sentiment Oscillator'
            ],
            'Pattern Recognition': [
                'Candlestick Patterns', 'Chart Patterns', 'Harmonic Patterns',
                'AI Pattern Scanner', 'Pattern Completion Predictor', 'Reversal Patterns'
            ]
        }
    
    def get_all_indicators(self) -> Dict[str, List[str]]:
        """Return all available indicator categories"""
        return self.indicator_categories
    
    def calculate_fractals(self, data: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
        """Calculate Williams Fractals"""
        high = data['High'].values
        low = data['Low'].values
        
        fractals_high = np.zeros(len(high))
        fractals_low = np.zeros(len(low))
        
        for i in range(periods, len(high) - periods):
            # High fractal
            if all(high[i] >= high[i-j] for j in range(1, periods+1)) and \
               all(high[i] >= high[i+j] for j in range(1, periods+1)):
                fractals_high[i] = high[i]
            
            # Low fractal
            if all(low[i] <= low[i-j] for j in range(1, periods+1)) and \
               all(low[i] <= low[i+j] for j in range(1, periods+1)):
                fractals_low[i] = low[i]
        
        return pd.DataFrame({
            'fractal_high': fractals_high,
            'fractal_low': fractals_low
        })
    
    def calculate_fibonacci_levels(self, data: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        recent_data = data.tail(lookback)
        high_price = recent_data['High'].max()
        low_price = recent_data['Low'].min()
        diff = high_price - low_price
        
        levels = {
            '0.0%': high_price,
            '23.6%': high_price - 0.236 * diff,
            '38.2%': high_price - 0.382 * diff,
            '50.0%': high_price - 0.5 * diff,
            '61.8%': high_price - 0.618 * diff,
            '78.6%': high_price - 0.786 * diff,
            '100.0%': low_price
        }
        return levels
    
    def calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Classic Pivot Points"""
        last_bar = data.iloc[-2]  # Previous day
        high, low, close = last_bar['High'], last_bar['Low'], last_bar['Close']
        
        pivot = (high + low + close) / 3
        
        levels = {
            'PP': pivot,
            'R1': 2 * pivot - low,
            'R2': pivot + (high - low),
            'R3': high + 2 * (pivot - low),
            'S1': 2 * pivot - high,
            'S2': pivot - (high - low),
            'S3': low - 2 * (high - pivot)
        }
        return levels
    
    def calculate_ai_enhanced_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """AI-Enhanced RSI with adaptive periods"""
        close = data['Close']
        
        if TALIB_AVAILABLE:
            # Use TA-Lib for precise RSI calculation
            rsi = talib.RSI(close.values, timeperiod=period)
        else:
            # Fallback RSI implementation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        # AI Enhancement: Adaptive smoothing based on volatility
        volatility = data['High'].rolling(20).std() / data['Close'].rolling(20).mean()
        adaptive_factor = np.clip(volatility * 10, 0.1, 2.0)
        
        # Apply adaptive smoothing
        ai_rsi = pd.Series(rsi).ewm(alpha=adaptive_factor).mean()
        
        return ai_rsi
    
    def calculate_volume_profile(self, data: pd.DataFrame, bins: int = 50) -> Dict[str, Any]:
        """Calculate Volume Profile"""
        price_range = data['High'].max() - data['Low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        for i in range(bins):
            price_level = data['Low'].min() + i * bin_size
            volume_at_level = 0
            
            for _, row in data.iterrows():
                if price_level >= row['Low'] and price_level <= row['High']:
                    volume_at_level += row['Volume']
            
            volume_profile[price_level] = volume_at_level
        
        return volume_profile


class ChartRenderer:
    """Advanced Chart Rendering with Multiple Indicators"""
    
    def __init__(self):
        self.indicators = AdvancedIndicators()
    
    def create_advanced_chart(self, data: pd.DataFrame, selected_indicators: List[str], 
                            pair: str = "EUR/USD") -> go.Figure:
        """Create advanced candlestick chart with selected indicators"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=['Price Chart', 'Volume', 'Oscillators'],
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=pair,
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add selected indicators
        for indicator in selected_indicators:
            self._add_indicator_to_chart(fig, data, indicator)
        
        # Volume chart
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Layout configuration
        fig.update_layout(
            title=f'{pair} - Advanced Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
    
    def _add_indicator_to_chart(self, fig: go.Figure, data: pd.DataFrame, indicator: str):
        """Add specific indicator to chart"""
        
        if 'Fractal' in indicator:
            fractals = self.indicators.calculate_fractals(data)
            
            # Add fractal highs
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=fractals['fractal_high'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name='Fractal High'
                ),
                row=1, col=1
            )
            
            # Add fractal lows
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=fractals['fractal_low'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name='Fractal Low'
                ),
                row=1, col=1
            )
        
        elif 'Fibonacci' in indicator:
            fib_levels = self.indicators.calculate_fibonacci_levels(data)
            
            for level, price in fib_levels.items():
                fig.add_hline(
                    y=price,
                    line_dash="dash",
                    line_color="gold",
                    annotation_text=f"Fib {level}: {price:.5f}",
                    row=1, col=1
                )
        
        elif 'Pivot' in indicator:
            pivot_levels = self.indicators.calculate_pivot_points(data)
            
            colors = {
                'PP': 'white',
                'R1': 'red', 'R2': 'red', 'R3': 'red',
                'S1': 'green', 'S2': 'green', 'S3': 'green'
            }
            
            for level, price in pivot_levels.items():
                fig.add_hline(
                    y=price,
                    line_dash="dot",
                    line_color=colors.get(level, 'gray'),
                    annotation_text=f"{level}: {price:.5f}",
                    row=1, col=1
                )
        
        elif 'RSI' in indicator and 'AI' in indicator:
            ai_rsi = self.indicators.calculate_ai_enhanced_rsi(data)
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ai_rsi,
                    name='AI-Enhanced RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        elif 'Bollinger' in indicator:
            # Calculate Bollinger Bands
            close = data['Close']
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=upper_band,
                    name='BB Upper', line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=sma,
                    name='BB Middle', line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=lower_band,
                    name='BB Lower', line=dict(color='blue', width=1),
                    fill='tonexty', fillcolor='rgba(0,100,80,0.1)'
                ),
                row=1, col=1
            )


def generate_sample_data(pair: str = "EUR/USD", days: int = 100) -> pd.DataFrame:
    """Generate realistic sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Base price for different pairs
    base_prices = {
        "EUR/USD": 1.0850,
        "GBP/USD": 1.2650,
        "USD/JPY": 149.50,
        "XAU/USD": 2050.00,
        "GBP/JPY": 189.25
    }
    
    base_price = base_prices.get(pair, 1.0000)
    
    # Generate price movements
    returns = np.random.normal(0.0001, 0.01, days)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i in range(days):
        open_price = prices[i]
        close_price = prices[i + 1]
        
        # Generate high/low with some randomness
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
        
        # Generate volume
        volume = np.random.randint(50000, 500000)
        
        data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2025-01-01', periods=days, freq='D')
    
    return df
