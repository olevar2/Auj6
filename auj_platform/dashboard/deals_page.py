"""
AUJ Platform Deals Page
====================

Dedicated page for displaying deal quality grades, status, and performance data.
Shows A+ to F graded deals with clean, organized interface.

Author: AUJ Platform Development Team
Created: June 27, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any


def get_grade_color(grade: str) -> str:
    """Return color code for deal grade badges"""
    grade_colors = {
        'A+': '#28a745',  # Dark green
        'A': '#40c460',   # Green
        'A-': '#6bcf7f',  # Light green
        'B+': '#ffc107', # Amber
        'B': '#fd7e14',   # Orange
        'B-': '#dc3545', # Red-orange
        'C': '#dc3545',   # Red
        'F': '#6c757d'    # Gray
    }
    return grade_colors.get(grade, '#6c757d')


def get_status_color(status: str) -> str:
    """Return color code for deal status"""
    status_colors = {
        'OPEN': '#17a2b8',      # Blue
        'CLOSED_WIN': '#28a745', # Green
        'CLOSED_LOSS': '#dc3545', # Red
        'PENDING': '#ffc107',    # Yellow
        'CANCELLED': '#6c757d'   # Gray
    }
    return status_colors.get(status, '#6c757d')


def create_grade_badge(grade: str) -> str:
    """Create HTML badge for deal grade"""
    color = get_grade_color(grade)
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.8em;
        margin: 2px;
    ">{grade}</span>
    """


def create_status_badge(status: str) -> str:
    """Create HTML badge for deal status"""
    color = get_status_color(status)
    status_text = status.replace('_', ' ')
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 8px;
        border-radius: 8px;
        font-weight: normal;
        font-size: 0.7em;
        margin: 2px;
    ">{status_text}</span>
    """


def generate_sample_deals() -> List[Dict]:
    """Generate sample deal data with proper grading"""
    
    strategies = [
        ('golden_alignment', 'A+', 'Perfect H4/H1 trend alignment'),
        ('momentum_continuation', 'A', 'Strong trending momentum'),
        ('volatility_breakout', 'A-', 'Bollinger squeeze breakout'),
        ('reversal_correction', 'B+', 'H4 trend + H1 retracement'),
        ('range_trading', 'B', 'Stable ranging market'),
        ('counter_trend_reversal', 'B', 'High stress reversal'),
        ('correlation_arbitrage', 'B-', 'Pair correlation play'),
    ]
    
    pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'EUR/GBP', 'GBP/JPY', 'XAU/USD']
    statuses = ['OPEN', 'CLOSED_WIN', 'CLOSED_LOSS', 'PENDING']
    
    deals = []
    for i in range(20):
        strategy_data = strategies[np.random.randint(0, len(strategies))]
        strategy, grade, description = strategy_data
        pair = np.random.choice(pairs)
        status = np.random.choice(statuses)
        
        # Generate realistic data based on grade
        grade_multipliers = {'A+': 1.2, 'A': 1.1, 'A-': 1.0, 'B+': 0.9, 'B': 0.8, 'B-': 0.7, 'C': 0.6, 'F': 0.4}
        multiplier = grade_multipliers.get(grade, 0.8)
        
        confidence = round(np.random.uniform(0.6, 0.95) * multiplier, 2)
        
        if status in ['CLOSED_WIN', 'CLOSED_LOSS']:
            pnl = round(np.random.uniform(-300, 500) * multiplier, 2)
            if status == 'CLOSED_LOSS':
                pnl = -abs(pnl)
            duration_hours = round(np.random.uniform(2, 48), 1)
        else:
            pnl = None
            duration_hours = None
            
        entry_time = datetime.now() - timedelta(hours=np.random.uniform(1, 72))
        
        deals.append({
            'id': f'DEAL-{1000+i:04d}',
            'pair': pair,
            'strategy': strategy,
            'grade': grade,
            'description': description,
            'status': status,
            'confidence': confidence,
            'entry_time': entry_time,
            'entry_price': round(np.random.uniform(1.0500, 1.2000), 5),
            'current_price': round(np.random.uniform(1.0500, 1.2000), 5),
            'position_size': round(np.random.uniform(0.1, 2.0), 2),
            'pnl': pnl,
            'duration_hours': duration_hours,
            'risk_reward': round(np.random.uniform(1.5, 3.5), 1),
            'stop_loss': round(np.random.uniform(1.0450, 1.0550), 5),
            'take_profit': round(np.random.uniform(1.1500, 1.2500), 5)
        })
    
    return deals


def render_deals_overview(deals: List[Dict]):
    """Render deals overview metrics"""
    
    # Calculate metrics
    total_deals = len(deals)
    open_deals = len([d for d in deals if d['status'] == 'OPEN'])
    closed_deals = len([d for d in deals if d['status'].startswith('CLOSED')])
    
    win_deals = len([d for d in deals if d['status'] == 'CLOSED_WIN'])
    win_rate = (win_deals / closed_deals * 100) if closed_deals > 0 else 0
    
    total_pnl = sum([d['pnl'] for d in deals if d['pnl'] is not None])
    
    # Grade distribution
    grade_counts = {}
    for deal in deals:
        grade = deal['grade']
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Deals", total_deals)
        st.metric("Open Positions", open_deals)
    
    with col2:
        st.metric("Closed Deals", closed_deals)
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col3:
        st.metric("Total P&L", f"${total_pnl:.2f}")
        avg_pnl = total_pnl / closed_deals if closed_deals > 0 else 0
        st.metric("Avg P&L", f"${avg_pnl:.2f}")
    
    with col4:
        st.write("**Grade Distribution**")
        grade_html = ""
        for grade in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C', 'F']:
            count = grade_counts.get(grade, 0)
            if count > 0:
                grade_html += f"{create_grade_badge(grade)} {count} "
        st.markdown(grade_html, unsafe_allow_html=True)


def render_deals_table(deals: List[Dict]):
    """Render main deals table with filtering"""
    
    st.subheader("📋 Deal Management")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ['All'] + list(set([d['status'] for d in deals]))
        )
    
    with col2:
        grade_filter = st.selectbox(
            "Filter by Grade", 
            ['All'] + sorted(list(set([d['grade'] for d in deals])), reverse=True)
        )
    
    with col3:
        pair_filter = st.selectbox(
            "Filter by Pair",
            ['All'] + sorted(list(set([d['pair'] for d in deals])))
        )
    
    # Apply filters
    filtered_deals = deals
    if status_filter != 'All':
        filtered_deals = [d for d in filtered_deals if d['status'] == status_filter]
    if grade_filter != 'All':
        filtered_deals = [d for d in filtered_deals if d['grade'] == grade_filter]
    if pair_filter != 'All':
        filtered_deals = [d for d in filtered_deals if d['pair'] == pair_filter]
    
    # Create display table
    if filtered_deals:
        display_data = []
        for deal in filtered_deals:
            
            # Format entry time
            time_str = deal['entry_time'].strftime("%m/%d %H:%M")
            
            # Format P&L
            if deal['pnl'] is not None:
                pnl_str = f"${deal['pnl']:.2f}"
                pnl_color = "🟢" if deal['pnl'] > 0 else "🔴"
                pnl_display = f"{pnl_color} {pnl_str}"
            else:
                unrealized = (deal['current_price'] - deal['entry_price']) * deal['position_size'] * 10000
                pnl_display = f"📊 ${unrealized:.2f}"
            
            # Format duration
            if deal['duration_hours']:
                duration_str = f"{deal['duration_hours']:.1f}h"
            else:
                hours_open = (datetime.now() - deal['entry_time']).total_seconds() / 3600
                duration_str = f"{hours_open:.1f}h"
            
            display_data.append({
                'ID': deal['id'],
                'Pair': deal['pair'],
                'Grade': deal['grade'],
                'Status': deal['status'],
                'Strategy': deal['strategy'].replace('_', ' ').title(),
                'Confidence': f"{deal['confidence']:.0%}",
                'Entry Time': time_str,
                'Entry Price': f"{deal['entry_price']:.5f}",
                'Position Size': f"{deal['position_size']:.2f}",
                'P&L': pnl_display,
                'Duration': duration_str,
                'R:R': f"{deal['risk_reward']:.1f}:1"
            })
        
        df = pd.DataFrame(display_data)
        
        # Custom styling for table
        def style_grade(val):
            color = get_grade_color(val)
            return f'background-color: {color}; color: white; font-weight: bold'
        
        def style_status(val):
            color = get_status_color(val)
            return f'background-color: {color}; color: white'
        
        # Apply styling
        styled_df = df.style.applymap(style_grade, subset=['Grade']) \
                           .applymap(style_status, subset=['Status'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.caption(f"Showing {len(filtered_deals)} of {len(deals)} deals")
    else:
        st.info("No deals match the selected filters.")


def render_deals_charts(deals: List[Dict]):
    """Render deal performance charts"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Grade Performance")
        
        # Calculate P&L by grade
        grade_performance = {}
        for deal in deals:
            if deal['pnl'] is not None:
                grade = deal['grade']
                if grade not in grade_performance:
                    grade_performance[grade] = []
                grade_performance[grade].append(deal['pnl'])
        
        # Create bar chart
        grades = []
        avg_pnls = []
        colors = []
        
        for grade in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C', 'F']:
            if grade in grade_performance:
                grades.append(grade)
                avg_pnl = np.mean(grade_performance[grade])
                avg_pnls.append(avg_pnl)
                colors.append(get_grade_color(grade))
        
        if grades:
            fig = go.Figure(data=[
                go.Bar(x=grades, y=avg_pnls, marker_color=colors, text=[f"${pnl:.0f}" for pnl in avg_pnls], textposition="auto")
            ])
            fig.update_layout(
                title="Average P&L by Deal Grade",
                xaxis_title="Deal Grade",
                yaxis_title="Average P&L ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Deal Duration Analysis")
        
        # Duration by grade for closed deals
        closed_deals = [d for d in deals if d['duration_hours'] is not None]
        
        if closed_deals:
            duration_data = []
            for deal in closed_deals:
                duration_data.append({
                    'Grade': deal['grade'],
                    'Duration (hours)': deal['duration_hours'],
                    'P&L': deal['pnl']
                })
            
            df_duration = pd.DataFrame(duration_data)
            
            # Create scatter plot
            fig = px.scatter(
                df_duration, 
                x='Duration (hours)', 
                y='P&L',
                color='Grade',
                color_discrete_map={grade: get_grade_color(grade) for grade in df_duration['Grade'].unique()},
                title="P&L vs Duration by Grade",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def deals_page():
    """Main deals page function"""
    
    st.header("💎 Deal Quality Management")
    st.markdown("---")
    
    # Generate sample data (in real implementation, this would fetch from API)
    deals = generate_sample_deals()
    
    # Overview metrics
    render_deals_overview(deals)
    
    st.markdown("---")
    
    # Main deals table
    render_deals_table(deals)
    
    st.markdown("---")
    
    # Performance charts
    render_deals_charts(deals)
    
    # Deal grading legend
    with st.expander("📖 Deal Grading System"):
        st.markdown("""
        **AUJ Platform Deal Quality Grades:**
        
        🏆 **A+ - Golden Alignment**: Perfect H4/H1 trend alignment, highest confidence
        
        🥇 **A - Premium Quality**: Strong momentum continuation, high confidence  
        
        🥈 **A- - High Quality**: Volatility breakouts, good risk/reward
        
        🥉 **B+ - Good Quality**: Reversal corrections, solid setups
        
        📊 **B - Standard Quality**: Range trading, counter-trend reversals
        
        ⚠️ **B- - Below Average**: Marginal setups, higher risk
        
        📉 **C - Poor Quality**: Low confidence, unfavorable conditions
        
        ❌ **F - Rejected**: No strategy fits, avoid trading
        """)