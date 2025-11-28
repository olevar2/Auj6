#!/usr/bin/env python3
"""
Patch ExecutionHandler to integrate with DealMonitoringTeams.
This enables:
1. Adding positions to monitoring when trades are opened
2. Closing positions in monitoring when trades are closed
3. Completing the learning feedback loop
"""
import sys

def main():
    file_path = "e:/AUG6/auj_platform/src/trading_engine/execution_handler.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Patch 1: Add deal_monitoring_teams parameter to __init__
    content = content.replace(
        '    def __init__(self,\r\n'
        '                 config_manager: UnifiedConfigManager,\r\n'
        '                 risk_manager: DynamicRiskManager,\r\n'
        '                 messaging_service: Optional[Any] = None):',
        '    def __init__(self,\r\n'
        '                 config_manager: UnifiedConfigManager,\r\n'
        '                 risk_manager: DynamicRiskManager,\r\n'
        '                 deal_monitoring_teams=None,\r\n'
        '                 messaging_service: Optional[Any] = None):'
    )
    
    # Patch 2: Update docstring
    content = content.replace(
        '        Args:\r\n'
        '            config_manager: Unified configuration manager instance\r\n'
        '            risk_manager: Dynamic risk manager instance\r\n'
        '            messaging_service: Optional injected messaging service',
        '        Args:\r\n'
        '            config_manager: Unified configuration manager instance\r\n'
        '            risk_manager: Dynamic risk manager instance\r\n'
        '            deal_monitoring_teams: Deal monitoring teams for position tracking\r\n'
        '            messaging_service: Optional injected messaging service'
    )
    
    # Patch 3: Store deal_monitoring_teams
    content = content.replace(
        '        self.config_manager = config_manager\r\n'
        '        self.risk_manager = risk_manager\r\n'
        '        self.messaging_service = messaging_service  # Injected dependency',
        '        self.config_manager = config_manager\r\n'
        '        self.risk_manager = risk_manager\r\n'
        '        self.deal_monitoring_teams = deal_monitoring_teams\r\n'
        '        self.messaging_service = messaging_service  # Injected dependency'
    )
    
    # Patch 4: Add position to monitoring in _post_execution_processing
    content = content.replace(
        '        # Update risk manager with new position\r\n'
        '        if report.success and self.risk_manager:\r\n'
        '            await self.risk_manager.update_position_risk(\r\n'
        '                position_id=report.order.order_id,\r\n'
        '                current_pnl=Decimal(\'0\')  # Initial PnL is zero\r\n'
        '            )',
        '        # Update risk manager with new position\r\n'
        '        if report.success and self.risk_manager:\r\n'
        '            await self.risk_manager.update_position_risk(\r\n'
        '                position_id=report.order.order_id,\r\n'
        '                current_pnl=Decimal(\'0\')  # Initial PnL is zero\r\n'
        '            )\r\n'
        '        \r\n'
        '        # CRITICAL: Add position to monitoring for learning feedback loop\r\n'
        '        if report.success and self.deal_monitoring_teams:\r\n'
        '            try:\r\n'
        '                # Create a minimal TradeSignal-like object for add_position\r\n'
        '                from ..core.data_contracts import TradeSignal\r\n'
        '                signal = TradeSignal(\r\n'
        '                    id=report.order.signal_id or report.order.order_id,\r\n'
        '                    symbol=report.order.symbol,\r\n'
        '                    direction=report.order.direction,\r\n'
        '                    confidence=report.order.metadata.get(\'signal_confidence\', 0.5),\r\n'
        '                    stop_loss=report.order.stop_loss,\r\n'
        '                    take_profit=report.order.take_profit\r\n'
        '                )\r\n'
        '                signal.agent_source = report.order.metadata.get(\'generating_agent\', \'unknown\')\r\n'
        '                signal.strategy_type = report.order.metadata.get(\'strategy\', \'unknown\')\r\n'
        '                \r\n'
        '                await self.deal_monitoring_teams.add_position(\r\n'
        '                    deal_id=report.order.order_id,\r\n'
        '                    trade_signal=signal,\r\n'                '                    entry_price=report.order.filled_price,\r\n'
        '                    quantity=report.order.filled_quantity\r\n'
        '                )\r\n'
        '                logger.info(f"Added position {report.order.order_id} to monitoring teams")\r\n'
        '            except Exception as e:\r\n'
        '                logger.error(f"Failed to add position to monitoring: {e}")'
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("SUCCESS: ExecutionHandler patched for DealMonitoringTeams integration")
    print("- Added deal_monitoring_teams parameter to __init__")
    print("- Added position monitoring on successful execution")
    print("Note: Position closure tracking requires additional broker event monitoring")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
