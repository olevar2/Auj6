#!/usr/bin/env python3
"""
Script to update dependent files to use MetaApiProvider instead of MT5
"""

import os
import re
from datetime import datetime

def backup_file(file_path):
    """Create a backup of the file"""
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(file_path, 'r', encoding='utf-8') as src:
        with open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
    print(f"✅ Backup created: {backup_path}")

def update_real_order_book_provider():
    """Update real_order_book_provider.py"""
    file_path = "e:/AUJ/auj_platform/src/data_providers/real_order_book_provider.py"
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace imports
    content = re.sub(
        r'from \.unified_mt5_provider import UnifiedMT5Provider',
        'from .metaapi_provider import MetaApiProvider',
        content
    )
    
    # Replace class instantiation
    content = re.sub(
        r'self\.mt5_provider = UnifiedMT5Provider\(',
        'self.metaapi_provider = MetaApiProvider(',
        content
    )
    
    # Replace all mt5_provider usages
    content = re.sub(r'self\.mt5_provider', 'self.metaapi_provider', content)
    content = re.sub(r'mt5_provider', 'metaapi_provider', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated real_order_book_provider.py")

def update_real_market_depth_provider():
    """Update real_market_depth_provider.py"""
    file_path = "e:/AUJ/auj_platform/src/data_providers/real_market_depth_provider.py"
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace imports
    content = re.sub(
        r'from \.unified_mt5_provider import UnifiedMT5Provider',
        'from .metaapi_provider import MetaApiProvider',
        content
    )
    
    # Replace class instantiation and usage
    content = re.sub(
        r'self\.mt5_provider = UnifiedMT5Provider\(',
        'self.metaapi_provider = MetaApiProvider(',
        content
    )
    
    content = re.sub(r'self\.mt5_provider', 'self.metaapi_provider', content)
    content = re.sub(r'mt5_provider', 'metaapi_provider', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated real_market_depth_provider.py")

def update_execution_handler():
    """Update execution_handler.py"""
    file_path = "e:/AUJ/auj_platform/src/trading_engine/execution_handler.py"
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace imports
    content = re.sub(
        r'from \.\.data_providers\.unified_mt5_provider import UnifiedMT5Provider',
        'from ..data_providers.metaapi_provider import MetaApiProvider',
        content
    )
    
    # Replace provider creation
    content = re.sub(
        r'mt5_provider = UnifiedMT5Provider\(',
        'metaapi_provider = MetaApiProvider(',
        content
    )
    
    # Replace provider usage
    content = re.sub(r'mt5_provider', 'metaapi_provider', content)
    content = re.sub(r'"MT5"\] = mt5_provider', '"MetaApi"] = metaapi_provider', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated execution_handler.py")

def update_containers():
    """Update containers.py"""
    file_path = "e:/AUJ/auj_platform/src/core/containers.py"
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace MT5 provider registration with MetaApi
    content = re.sub(
        r'from \.\.data_providers\.unified_mt5_provider import UnifiedMT5Provider',
        'from ..data_providers.metaapi_provider import MetaApiProvider',
        content
    )
    
    content = re.sub(r'mt5_provider = provider', 'metaapi_provider = provider', content)
    content = re.sub(r'UnifiedMT5Provider', 'MetaApiProvider', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated containers.py")

def update_data_providers_init():
    """Update __init__.py in data_providers"""
    file_path = "e:/AUJ/auj_platform/src/data_providers/__init__.py"
    backup_file(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove MT5 imports and add MetaApi
    content = re.sub(
        r'from \.unified_mt5_provider import UnifiedMT5Provider',
        '# Removed MT5Provider - using MetaApiProvider instead',
        content
    )
    
    content = re.sub(
        r'from \.tick_data_provider import MT5TickDataProvider',
        '# Removed MT5TickDataProvider - using MetaApiProvider instead',
        content
    )
    
    content = re.sub(
        r'from \.ohlcv_provider import MT5OHLCVProvider',
        '# Removed MT5OHLCVProvider - using MetaApiProvider instead',
        content
    )
    
    # Add MetaApiProvider import if not present
    if 'from .metaapi_provider import MetaApiProvider' not in content:
        content = content.replace(
            'from .base_provider import BaseDataProvider',
            'from .base_provider import BaseDataProvider\nfrom .metaapi_provider import MetaApiProvider'
        )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated __init__.py")

def main():
    """Execute all updates"""
    print("🔄 Starting dependency updates...")
    
    try:
        update_real_order_book_provider()
        update_real_market_depth_provider()
        update_execution_handler()
        update_containers()
        update_data_providers_init()
        
        print("\n✅ All dependency updates completed successfully!")
        print("🔄 Next: Run tests to verify changes")
        
    except Exception as e:
        print(f"❌ Error during updates: {e}")
        print("💡 Check backup files to restore if needed")

if __name__ == "__main__":
    main()