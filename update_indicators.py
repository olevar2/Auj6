#!/usr/bin/env python3
"""
Script to update indicator_data_requirements.py to support MetaApi
"""

import re

def update_indicator_requirements():
    """Update indicator_data_requirements.py file"""
    
    file_path = "e:/AUJ/auj_platform/config/indicator_data_requirements.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace "MT5Provider" with "MetaApiProvider", "MT5Provider"
    # This makes MetaApiProvider primary and MT5Provider as fallback
    updated_content = re.sub(
        r'available_providers=\["MT5Provider"',
        r'available_providers=["MetaApiProvider", "MT5Provider"',
        content
    )
    
    # Replace "MT5OHLCVProvider" with "MetaApiProvider", "MT5OHLCVProvider"  
    updated_content = re.sub(
        r'available_providers=\["MT5OHLCVProvider"',
        r'available_providers=["MetaApiProvider", "MT5OHLCVProvider"',
        updated_content
    )
    
    # Replace "MT5TickDataProvider" with "MetaApiProvider", "MT5TickDataProvider"
    updated_content = re.sub(
        r'available_providers=\["MT5TickDataProvider"',
        r'available_providers=["MetaApiProvider", "MT5TickDataProvider"',
        updated_content
    )
    
    # Save the updated file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("Updated indicator_data_requirements.py successfully")
    print("MetaApiProvider is now primary provider for all indicators")

if __name__ == "__main__":
    update_indicator_requirements()