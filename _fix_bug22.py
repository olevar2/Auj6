#!/usr/bin/env python3
"""Script to fix Bug #22 in system_health_monitor.py"""

import re

# Read the original file
with open(r'E:\AUG6\auj_platform\src\monitoring\system_health_monitor.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and replace _check_database_health (lines 513-539)
new_db_function = '''    async def _check_database_health(self):
        """Check database connection health with REAL database queries."""
        try:
            # ✅ FIX Bug #22: Real database connectivity check
            if not hasattr(self, 'database') or not self.database:
                try:
                    from ..core.containers import get_database_manager
                    self.database = get_database_manager()
                except Exception as import_error:
                    self.logger.warning(f"⚠️ Database manager not available: {import_error}")
                    self._component_status['database'] = MonitorStatus.UNKNOWN
                    return
            
            start_time = time.time()
            
            # ✅ REAL DATABASE CHECK
            try:
                result = await self.database.execute_query(
                    "SELECT 1 AS health_check",
                    use_cache=False
                )
                
                if not result or not result.get('success', False):
                    raise Exception("Database query failed")
                
            except Exception as db_error:
                response_time = time.time() - start_time
                self.record_metric('database_response_time', response_time)
                raise Exception(f"Database connectivity check failed: {db_error}")
            
            response_time = time.time() - start_time
            self.record_metric('database_response_time', response_time)
            
            if hasattr(self.database, 'get_connection_pool_statu s'):
                try:
                    pool_status = await self.database.get_connection_pool_status()
                    self.record_metric('database_pool_active', pool_status.get('active', 0))
                    self.record_metric('database_pool_idle', pool_status.get('idle', 0))
                except Exception:
                    pass
            
            self._component_status['database'] = MonitorStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"❌ Database health check failed: {e}")
            self._component_status['database'] = MonitorStatus.CRITICAL
            await self._create_alert(
                'database',
                AlertSeverity.CRITICAL,
                f"Database health check failed: {e}"
            )
'''

# Replace lines 513-539
lines[512:539] = [new_db_function + '\n']

# Find and replace _check_broker_connections (lines 541-566, now shifted)
new_broker_function = '''    async def _check_broker_connections(self):
        """Check broker connection status with REAL broker API calls."""
        try:
            # ✅ FIX Bug #22: Real broker connectivity check
            if not hasattr(self, 'broker_manager') or not self.broker_manager:
                try:
                    from ..brokers.broker_manager import get_broker_manager
                    self.broker_manager = get_broker_manager()
                except Exception as import_error:
                    self.logger.warning(f"⚠️ Broker manager not available: {import_error}")
                    self._component_status['broker_connections'] = MonitorStatus.UNKNOWN
                    return
            
            start_time = time.time()
            
            # ✅ REAL BROKER CHECK
            try:
                account_info = await self.broker_manager.get_account_info()
                
                if not account_info or 'error' in account_info:
                    raise Exception("Failed to get account info")
                
                is_connected = account_info.get('connected', False)
                if not is_connected:
                    raise Exception("Broker not connected")
                    
            except Exception as broker_error:
                response_time = time.time() - start_time
                self.record_metric('broker_response_time', response_time)
                raise Exception(f"Broker connectivity check failed: {broker_error}")
            
            response_time = time.time() - start_time
            self.record_metric('broker_response_time', response_time)
            
            trading_session = account_info.get('trading_allowed', False)
            self.record_metric('trading_session_active', 1 if trading_session else 0)
            
            if 'balance' in account_info:
                self.record_metric('broker_account_balance', account_info['balance'])
            
            self._component_status['broker_connections'] = MonitorStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"❌ Broker connection check failed: {e}")
            self._component_status['broker_connections'] = MonitorStatus.CRITICAL
            await self._create_alert(
                'broker_connections',
                AlertSeverity.CRITICAL,
                f"Broker connection check failed: {e}"
            )
'''

# Find the new position (around line 514 after first replacement)
# Replace lines 514-540 (shifted after first replacement)
lines[513:540] = [new_broker_function + '\n']

# Write to NEW file
with open(r'E:\AUG6\auj_platform\src\monitoring\system_health_monitor_NEW.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Created system_health_monitor_NEW.py with Bug #22 fixes")
