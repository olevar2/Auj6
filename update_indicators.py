#!/usr/bin/env python3
"""
Script لتحديث indicator_data_requirements.py ليدعم MetaApi
"""

import re

def update_indicator_requirements():
    """تحديث ملف indicator_data_requirements.py"""
    
    file_path = "e:/AUJ/auj_platform/config/indicator_data_requirements.py"
    
    # قراءة الملف
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # استبدال "MT5Provider" بـ "MetaApiProvider", "MT5Provider"
    # هذا يجعل MetaApiProvider أولاً ثم MT5Provider كـ fallback
    updated_content = re.sub(
        r'available_providers=\["MT5Provider"',
        r'available_providers=["MetaApiProvider", "MT5Provider"',
        content
    )
    
    # استبدال "MT5OHLCVProvider" بـ "MetaApiProvider", "MT5OHLCVProvider"  
    updated_content = re.sub(
        r'available_providers=\["MT5OHLCVProvider"',
        r'available_providers=["MetaApiProvider", "MT5OHLCVProvider"',
        updated_content
    )
    
    # استبدال "MT5TickDataProvider" بـ "MetaApiProvider", "MT5TickDataProvider"
    updated_content = re.sub(
        r'available_providers=\["MT5TickDataProvider"',
        r'available_providers=["MetaApiProvider", "MT5TickDataProvider"',
        updated_content
    )
    
    # حفظ الملف المحدث
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("Updated indicator_data_requirements.py successfully")
    print("MetaApiProvider is now primary provider for all indicators")

if __name__ == "__main__":
    update_indicator_requirements()