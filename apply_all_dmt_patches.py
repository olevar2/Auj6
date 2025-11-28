#!/usr/bin/env python3
"""Apply all Critical patches to deal_monitoring_teams.py"""
import sys

def main():
    file_path = "e:/AUG6/auj_platform/src/trading_engine/deal_monitoring_teams.py"
    
    #  Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Patch 1: hierarchy_manager param
    content = content.replace(
        '                 risk_manager=None,\r\n'  
        '                 alert_callback: Optional[Callable] = None):',
        '                 risk_manager=None,\r\n'
        '                 hierarchy_manager=None,\r\n'
        '                 alert_callback: Optional[Callable] = None):'
    )
    
    content = content.replace(
        '            risk_manager: Dynamic risk manager instance\r\n'
        '            alert_callback: Optional callback for alert notifications',
        '            risk_manager: Dynamic risk manager instance\r\n'
        '            hierarchy_manager: Hierarchy manager for learning feedback loop\r\n'
        '            alert_callback: Optional callback for alert notifications'
    )
    
    content = content.replace(
        '        self.performance_tracker = performance_tracker\r\n'
        '        self.risk_manager = risk_manager\r\n'
        '        self.alert_callback = alert_callback\r\n',
        '        self.performance_tracker = performance_tracker\r\n'
        '        self.risk_manager = risk_manager\r\n'
        '        self.hierarchy_manager = hierarchy_manager\r\n'
        '        self.alert_callback = alert_callback\r\n'
    )
    
    # Patch 2: Add shutdown method
    content = content.replace(
        '        logger.info("Deal Monitoring Teams initialization completed successfully")\r\n        \r\n    def _validate_team_configurations',
        '        logger.info("Deal Monitoring Teams initialization completed successfully")\r\n        \r\n    async def shutdown(self) -> None:\r\n        """Shutdown the deal monitoring system cleanly."""\r\n        logger.info("Shutting down Deal Monitoring Teams...")\r\n        await self.stop_monitoring()\r\n        if self.active_positions:\r\n            logger.warning(f"{len(self.active_positions)} positions still active during shutdown")\r\n        logger.info("Deal Monitoring Teams shutdown completed")\r\n        \r\n    def _validate_team_configurations'
    )
    
    # Patch 3: Add learning loop to close_position()
    content = content.replace(
        '        # Move to history\r\n        self.position_history.append(position)\r\n        del self.active_positions[deal_id]\r\n\r\n        # Record with performance tracker',
        '        # Move to history\r\n        self.position_history.append(position)\r\n        del self.active_positions[deal_id]\r\n        \r\n        # CRITICAL: Record trade result to HierarchyManager for learning\r\n        if self.hierarchy_manager and position.agent_source:\r\n            try:\r\n                from ..core.data_contracts import GradedDeal\r\n                graded_deal = GradedDeal(\r\n                    deal_id=position.deal_id,\r\n                    symbol=position.symbol,\r\n                    direction=position.direction,\r\n                    entry_price=position.entry_price,\r\n                    exit_price=position.current_price,\r\n                    quantity=position.quantity,\r\n                    pnl=final_pnl,\r\n                    pnl_percentage=position.pnl_percentage,\r\n                    entry_time=position.entry_time,\r\n                    exit_time=close_time,\r\n                    holding_duration=position.duration_hours,\r\n                    grade=position.grade,\r\n                    agent_source=position.agent_source,\r\n                    strategy_type=position.strategy_type\r\n                )\r\n                self.hierarchy_manager.record_trade_result(\r\n                    agent_name=position.agent_source,\r\n                    trade=graded_deal,\r\n                    is_out_of_sample=True\r\n                )\r\n                logger.info(f"Recorded trade result for agent {position.agent_source}: PnL={final_pnl:.2f}")\r\n            except Exception as e:\r\n                logger.error(f"Failed to record trade result: {e}")\r\n\r\n        # Record with performance tracker'
    )
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("SUCCESS: All critical patches applied!")
    print("- Added hierarchy_manager parameter")
    print("- Added shutdown method")
    print("- record_trade_result integration in close_position")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
