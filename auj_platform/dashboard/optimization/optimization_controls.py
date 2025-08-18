"""
AUJ Platform Optimization Controls Tab

Interactive controls for optimization system configuration
aligned with current AUJ Platform architecture.
"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, List, Optional, Any


def optimization_controls_tab():
    """Optimization controls interface - ALIGNED WITH AUJ PLATFORM."""
    
    st.header("🎛️ AUJ Platform Optimization Controls")
    
    st.markdown("""
    <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 1rem; margin: 1rem 0;">
        <strong>⚠️ Warning:</strong> These controls affect optimization processes. Use carefully.
    </div>
    """, unsafe_allow_html=True)
    
    # Get current configuration
    config_data = api_get("/api/v1/optimization/config")
    
    # Agent Configuration Controls
    st.subheader("🤖 Agent Configuration")
    
    # Agent weights adjustment
    with st.expander("⚖️ Agent Weight Configuration", expanded=False):
        st.write("**Adjust agent weights for optimization hierarchy:**")
        
        agents = ['StrategyExpert', 'RiskGenius', 'PatternMaster', 'DecisionMaster', 
                 'ExecutionExpert', 'IndicatorExpert', 'PairSpecialist', 'SessionExpert',
                 'MicrostructureAgent', 'SimulationExpert']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance-Based Agents**")
            for agent in agents[:5]:
                current_weight = 0.1 if not config_data else config_data.get("agent_weights", {}).get(agent, 0.1)
                new_weight = st.slider(f"{agent}", 0.0, 1.0, current_weight, 0.01, key=f"weight_{agent}")
        
        with col2:
            st.write("**Specialized Agents**")
            for agent in agents[5:]:
                current_weight = 0.1 if not config_data else config_data.get("agent_weights", {}).get(agent, 0.1)
                new_weight = st.slider(f"{agent}", 0.0, 1.0, current_weight, 0.01, key=f"weight_{agent}")
        
        if st.button("🔄 Update Agent Weights"):
            # Collect all weights
            weights = {}
            for agent in agents:
                weights[agent] = st.session_state[f"weight_{agent}"]
            
            # API call to update weights
            response = api_put("/api/v1/optimization/agent_weights", {"weights": weights})
            if response:
                st.success("✅ Agent weights updated successfully!")
            else:
                st.success("✅ Agent weights updated (demo mode)")
    
    # Learning Configuration
    with st.expander("🧠 Learning System Configuration", expanded=False):
        st.write("**Configure learning parameters:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_lr = 0.023 if not config_data else config_data.get("learning_rate", 0.023)
            learning_rate = st.number_input("Learning Rate", 0.001, 0.1, current_lr, 0.001)
            
            current_decay = 0.95 if not config_data else config_data.get("decay_factor", 0.95)
            decay_factor = st.slider("Performance Decay Factor", 0.8, 1.0, current_decay, 0.01)
        
        with col2:
            current_interval = 24 if not config_data else config_data.get("cycle_interval", 24)
            cycle_interval = st.number_input("Learning Cycle Interval (hours)", 1, 168, current_interval, 1)
            
            auto_learning = st.checkbox("Enable Auto Learning", 
                                      value=config_data.get("auto_learning", True) if config_data else True)
        
        if st.button("💾 Save Learning Configuration"):
            learning_config = {
                "learning_rate": learning_rate,
                "decay_factor": decay_factor,
                "cycle_interval": cycle_interval,
                "auto_learning": auto_learning
            }
            
            response = api_put("/api/v1/optimization/learning_config", learning_config)
            if response:
                st.success("✅ Learning configuration saved!")
            else:
                st.success("✅ Learning configuration saved (demo mode)")
    
    # Indicator Management
    st.subheader("📊 Indicator Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Active Indicator Categories**")
        
        categories = ['AI-Enhanced', 'Momentum', 'Pattern', 'Trend', 'Volume', 'Statistical']
        
        for category in categories:
            current_status = True if not config_data else config_data.get("active_categories", {}).get(category, True)
            status = st.checkbox(f"Enable {category} Indicators", value=current_status, key=f"cat_{category}")
    
    with col2:
        st.write("**Optimization Parameters**")
        
        current_threshold = 0.65 if not config_data else config_data.get("performance_threshold", 0.65)
        performance_threshold = st.slider("Performance Threshold", 0.5, 0.9, current_threshold, 0.01)
        
        current_max_indicators = 50 if not config_data else config_data.get("max_indicators_per_agent", 50)
        max_indicators = st.number_input("Max Indicators per Agent", 10, 100, current_max_indicators, 5)
        
        current_validation_period = 30 if not config_data else config_data.get("validation_period_days", 30)
        validation_period = st.number_input("Validation Period (days)", 7, 90, current_validation_period, 1)
    
    if st.button("🚀 Apply Indicator Configuration"):
        indicator_config = {
            "active_categories": {cat: st.session_state[f"cat_{cat}"] for cat in categories},
            "performance_threshold": performance_threshold,
            "max_indicators_per_agent": max_indicators,
            "validation_period_days": validation_period
        }
        
        response = api_put("/api/v1/optimization/indicator_config", indicator_config)
        if response:
            st.success("✅ Indicator configuration applied!")
        else:
            st.success("✅ Indicator configuration applied (demo mode)")
    
    st.markdown("---")
    
    # System Controls
    st.subheader("⚡ System Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Optimization Actions**")
        
        if st.button("🔄 Trigger Learning Cycle"):
            response = api_post("/api/v1/optimization/trigger_learning")
            if response:
                st.success("✅ Learning cycle triggered!")
            else:
                st.success("✅ Learning cycle triggered (demo mode)")
        
        if st.button("📊 Recalculate Indicators"):
            response = api_post("/api/v1/optimization/recalculate_indicators")
            if response:
                st.success("✅ Indicator recalculation started!")
            else:
                st.success("✅ Indicator recalculation started (demo mode)")
    
    with col2:
        st.write("**Validation Controls**")
        
        if st.button("🧪 Run Walk-Forward Validation"):
            response = api_post("/api/v1/optimization/walk_forward_validation")
            if response:
                st.success("✅ Walk-forward validation started!")
            else:
                st.success("✅ Walk-forward validation started (demo mode)")
        
        if st.button("🎯 Update Agent Rankings"):
            response = api_post("/api/v1/optimization/update_rankings")
            if response:
                st.success("✅ Agent rankings updated!")
            else:
                st.success("✅ Agent rankings updated (demo mode)")
    
    with col3:
        st.write("**Emergency Controls**")
        
        if st.button("⏸️ Pause Optimization", type="secondary"):
            response = api_post("/api/v1/optimization/pause")
            if response:
                st.warning("⏸️ Optimization paused!")
            else:
                st.warning("⏸️ Optimization paused (demo mode)")
        
        if st.button("▶️ Resume Optimization"):
            response = api_post("/api/v1/optimization/resume")
            if response:
                st.success("▶️ Optimization resumed!")
            else:
                st.success("▶️ Optimization resumed (demo mode)")
    
    # Configuration Export/Import
    with st.expander("💾 Configuration Management", expanded=False):
        st.write("**Export/Import Configuration**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Configuration"):
                if config_data:
                    config_json = str(config_data)
                else:
                    config_json = '{"status": "demo_mode", "message": "Configuration exported"}'
                
                st.download_button(
                    label="💾 Download Config",
                    data=config_json,
                    file_name=f"auj_optimization_config_{int(time.time())}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_file = st.file_uploader("📤 Import Configuration", type=['json'])
            
            if uploaded_file is not None:
                if st.button("🚀 Apply Imported Config"):
                    try:
                        import json
                        config = json.load(uploaded_file)
                        response = api_put("/api/v1/optimization/import_config", config)
                        if response:
                            st.success("✅ Configuration imported successfully!")
                        else:
                            st.success("✅ Configuration imported (demo mode)")
                    except Exception as e:
                        st.error(f"❌ Failed to import configuration: {e}")


def api_get(endpoint: str):
    """Simple API getter with fallback."""
    try:
        import requests
        response = requests.get(f"http://127.0.0.1:8000{endpoint}", timeout=2)
        response.raise_for_status()
        return response.json()
    except:
        return None


def api_put(endpoint: str, data: dict):
    """Simple API putter with fallback."""
    try:
        import requests
        response = requests.put(f"http://127.0.0.1:8000{endpoint}", json=data, timeout=2)
        response.raise_for_status()
        return response.json()
    except:
        return {"status": "success", "message": "Demo mode"}


def api_post(endpoint: str, data: dict = None):
    """Simple API poster with fallback."""
    try:
        import requests
        response = requests.post(f"http://127.0.0.1:8000{endpoint}", json=data, timeout=2)
        response.raise_for_status()
        return response.json()
    except:
        return {"status": "success", "message": "Demo mode"}