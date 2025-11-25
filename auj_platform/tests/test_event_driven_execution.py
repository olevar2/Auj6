import asyncio
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

async def test_event_driven_fill():
    """Test event-driven order fill detection"""
    print("Testing Event-Driven Execution Handler...")
    print("=" * 60)
    
    # Verify MetaApiProvider has order subscription methods
    from data_providers.metaapi_provider import MetaApiProvider
    
    provider_attrs = dir(MetaApiProvider)
    has_subscribe = 'subscribe_to_order_updates' in provider_attrs
    has_unsubscribe = 'unsubscribe_from_order_updates' in provider_attrs
    has_subscribers_dict = 'order_subscribers' in provider_attrs  # Check if __init__ adds it
    
    print(f"✓ MetaApiProvider.subscribe_to_order_updates: {has_subscribe}")
    print(f"✓ MetaApiProvider.unsubscribe_from_order_updates: {has_unsubscribe}")
    print()
    
    # Verify MetaApiBroker forwards the methods
    from broker_interfaces.metaapi_broker import MetaApiBroker
    
    broker_attrs = dir(MetaApiBroker)
    broker_has_subscribe = 'subscribe_to_order_updates' in broker_attrs
    broker_has_unsubscribe = 'unsubscribe_from_order_updates' in broker_attrs
    
    print(f"✓ MetaApiBroker.subscribe_to_order_updates: {broker_has_subscribe}")
    print(f"✓ MetaApiBroker.unsubscribe_from_order_updates: {broker_has_unsubscribe}")
    print()
    
    # Test event callback
    test_passed = False
    received_order_id = None
    
    async def test_callback(order_data):
        nonlocal test_passed, received_order_id
        received_order_id = order_data.get('id')
        test_passed = True
    
    # Create a mock provider scenario
    config = {'metaapi': {}}
    provider = MetaApiProvider(config=config)
    
    # Test subscription
    order_id = "TEST_ORDER_123"
    await provider.subscribe_to_order_updates(order_id, test_callback)
    
    print(f"✓ Subscribed to order {order_id}")
    
    # Simulate order update via _handle_orders_update
    mock_data = {
        'orders': [
            {'id': order_id, 'status': 'FILLED', 'volume': 0.01}
        ]
    }
    
    await provider._handle_orders_update(mock_data)
    
    # Wait a moment for callback execution
    await asyncio.sleep(0.1)
    
    print(f"✓ Received callback: {test_passed}")
    print(f"✓ Order ID matched: {received_order_id == order_id}")
    print()
    
    # Cleanup
    await provider.unsubscribe_from_order_updates(order_id, test_callback)
    print(f"✓ Unsubscribed from order {order_id}")
    print()
    
    # Final verdict
    print("=" * 60)
    if all([has_subscribe, has_unsubscribe, broker_has_subscribe, broker_has_unsubscribe, test_passed]):
        print("✅ ALL TESTS PASSED - Event-driven system is functional!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Review implementation")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_event_driven_fill())
    sys.exit(0 if result else 1)
