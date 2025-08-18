#!/usr/bin/env python3
"""
Script لحذف الملفات غير الضرورية بعد تكامل MetaApi
"""

import os
import shutil
from datetime import datetime

def create_backup_dir():
    """إنشاء مجلد backup"""
    backup_dir = f"e:/AUJ/backups/cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir

def safe_remove_file(file_path, backup_dir):
    """حذف ملف مع إنشاء نسخة احتياطية"""
    if os.path.exists(file_path):
        # إنشاء نسخة احتياطية
        rel_path = os.path.relpath(file_path, "e:/AUJ")
        backup_file_path = os.path.join(backup_dir, rel_path)
        os.makedirs(os.path.dirname(backup_file_path), exist_ok=True)
        shutil.copy2(file_path, backup_file_path)
        
        # حذف الملف الأصلي
        os.remove(file_path)
        print(f"✅ Removed: {file_path} (backed up)")
        return True
    else:
        print(f"⚠️ File not found: {file_path}")
        return False

def remove_mt5_core_files(backup_dir):
    """حذف الملفات الأساسية لـ MT5"""
    files_to_remove = [
        "e:/AUJ/auj_platform/src/data_providers/unified_mt5_provider.py",
        "e:/AUJ/auj_platform/src/data_providers/ohlcv_provider.py",
        "e:/AUJ/auj_platform/src/data_providers/tick_data_provider.py",
        "e:/AUJ/auj_platform/src/broker_interfaces/mt5_broker.py"
    ]
    
    print("🗑️ Removing MT5 core files...")
    removed_count = 0
    
    for file_path in files_to_remove:
        if safe_remove_file(file_path, backup_dir):
            removed_count += 1
    
    print(f"✅ Removed {removed_count} core MT5 files")
    return removed_count

def clean_cache_files():
    """تنظيف ملفات cache و build"""
    cache_patterns = [
        "e:/AUJ/.mypy_cache/3.13/auj_platform/src/broker_interfaces/mt5_broker.*",
        "e:/AUJ/.mypy_cache/3.13/auj_platform/src/data_providers/unified_mt5_provider.*",
        "e:/AUJ/build/lib/auj_platform/src/broker_interfaces/mt5_broker.py",
        "e:/AUJ/build/lib/auj_platform/src/data_providers/unified_mt5_provider.py",
        "e:/AUJ/auj_platform/src/data_providers/__pycache__/unified_mt5_provider.cpython-313.pyc",
        "e:/AUJ/auj_platform/src/broker_interfaces/__pycache__/mt5_broker.cpython-313.pyc"
    ]
    
    print("🧹 Cleaning cache files...")
    cleaned_count = 0
    
    import glob
    for pattern in cache_patterns:
        for file_path in glob.glob(pattern):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"✅ Cleaned: {file_path}")
                    cleaned_count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"✅ Cleaned dir: {file_path}")
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️ Could not clean {file_path}: {e}")
    
    print(f"✅ Cleaned {cleaned_count} cache files")
    return cleaned_count

def main():
    """تنفيذ التنظيف"""
    print("🧹 Starting safe cleanup process...")
    
    # إنشاء backup directory
    backup_dir = create_backup_dir()
    print(f"📁 Backup directory: {backup_dir}")
    
    try:
        # حذف الملفات الأساسية
        core_removed = remove_mt5_core_files(backup_dir)
        
        # تنظيف cache
        cache_cleaned = clean_cache_files()
        
        print(f"\n✅ Cleanup completed successfully!")
        print(f"📊 Summary:")
        print(f"   - Core MT5 files removed: {core_removed}")
        print(f"   - Cache files cleaned: {cache_cleaned}")
        print(f"   - Backup location: {backup_dir}")
        
        print(f"\n🔄 Next steps:")
        print(f"   1. Run tests: py -c 'from auj_platform.src.data_providers import *'")
        print(f"   2. Test indicators: py auj_platform/src/tests/test_new_indicators_integration.py")
        print(f"   3. Full platform test")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        print(f"💡 Backup available at: {backup_dir}")

if __name__ == "__main__":
    main()