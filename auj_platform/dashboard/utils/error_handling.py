"""
Error Handling and Recovery System for AUJ Platform Dashboard
Comprehensive error management with fallback mechanisms
"""

import streamlit as st
import traceback
import time
import pandas as pd
from datetime import datetime
from typing import Any, Callable, Dict, Optional, List
from functools import wraps

class DashboardErrorHandler:
    """Centralized error handling for the dashboard"""
    
    def __init__(self):
        self.error_history = []
        self.max_errors = 100
        self.recovery_attempts = {}
        self.fallback_functions = {}
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        error_info = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.error_history.append(error_info)
        
        # Keep only recent errors
        if len(self.error_history) > self.max_errors:
            self.error_history = self.error_history[-self.max_errors:]
        
        # Store in session state for display
        if 'dashboard_errors' not in st.session_state:
            st.session_state.dashboard_errors = []
        
        st.session_state.dashboard_errors.append(error_info)
    
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register a fallback function for an operation"""
        self.fallback_functions[operation_name] = fallback_func