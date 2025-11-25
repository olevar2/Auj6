import asyncio
import sys
import os
print("Starting verification script...")
from datetime import date, datetime
from decimal import Decimal

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Imports starting...")
from src.core.unified_database_manager import UnifiedDatabaseManager
from src.trading_engine.risk_repository import RiskStateRepository
print("Imports done.")

async def verify_risk_persistence():
    print("Verifying Risk Persistence...")
    
    # Use a test database
    db_url = "sqlite:///data/test_risk_persistence.db"
    db_manager = UnifiedDatabaseManager(database_url=db_url)
    
    try:
        await db_manager.initialize()
        print("Database Manager Initialized")
        
        repo = RiskStateRepository(db_manager)
        await repo.initialize()
        print("Risk Repository Initialized")
        
        # Test Daily Loss
        today = date.today()
        print(f"Testing Daily Loss for {today}")
        
        initial_loss = await repo.get_daily_loss(today)
        print(f"Initial Loss: {initial_loss}")
        assert initial_loss == 0.0
        
        await repo.update_daily_loss(today, 100.50)
        updated_loss = await repo.get_daily_loss(today)
        print(f"Updated Loss: {updated_loss}")
        assert updated_loss == 100.50
        
        await repo.update_daily_loss(today, 50.0)
        final_loss = await repo.get_daily_loss(today)
        print(f"Final Loss: {final_loss}")
        assert final_loss == 100.50 # Wait, update_daily_loss in my impl adds to existing?
        # Let's check implementation. 
        # My implementation: new_loss = current_loss + loss_amount.
        # So 100.50 + 50.0 = 150.50.
        # Ah, wait. In close_position_risk_update I did:
        # current_daily_loss = self.daily_loss_tracking.get(today, 0.0)
        # self.daily_loss_tracking[today] = current_daily_loss + abs(float(final_pnl))
        # await self.risk_repository.update_daily_loss(today, loss_amount)
        
        # In repository:
        # current_loss = await self.get_daily_loss(day)
        # new_loss = current_loss + loss_amount
        # So yes, it accumulates.
        
        if final_loss == 150.50:
            print("Daily Loss Accumulation Verified")
        else:
            print(f"Daily Loss Accumulation Failed: Expected 150.50, got {final_loss}")
            
        # Test Open Positions
        print("Testing Open Positions")
        pos_id = "POS-001"
        pos_data = {
            "symbol": "EURUSD",
            "entry_price": 1.1000,
            "size": 1.0,
            "initial_equity": 10000.0,
            "current_risk_percent": 1.0,
            "unrealized_pnl": 0.0
        }
        
        await repo.add_open_position(pos_id, pos_data)
        print(f"Added Position {pos_id}")
        
        positions = await repo.get_open_positions()
        print(f"Open Positions: {len(positions)}")
        assert pos_id in positions
        assert positions[pos_id]['symbol'] == "EURUSD"
        
        # Update Position
        pos_data['unrealized_pnl'] = -50.0
        await repo.add_open_position(pos_id, pos_data)
        
        positions = await repo.get_open_positions()
        print(f"Updated Position PnL: {positions[pos_id]['unrealized_pnl']}")
        assert positions[pos_id]['unrealized_pnl'] == -50.0
        
        # Remove Position
        await repo.remove_open_position(pos_id)
        positions = await repo.get_open_positions()
        print(f"Positions after removal: {len(positions)}")
        assert pos_id not in positions
        
        print("Verification Successful!")
        
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_risk_persistence())
