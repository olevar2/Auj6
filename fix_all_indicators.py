import os

INDICATORS_DIR = r"e:\AUG6\auj_platform\src\indicator_engine\indicators"

def fix_imports_recursively():
    count = 0
    for root, dirs, files in os.walk(INDICATORS_DIR):
        for filename in files:
            if not filename.endswith(".py"):
                continue
                
            filepath = os.path.join(root, filename)
            
            # Skip base directory itself to avoid messing up standard_indicator.py
            if "indicators\\base" in filepath and filename in ["standard_indicator.py", "base_indicator.py"]:
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            original_content = content
            
            # Calculate relative path depth to 'src'
            # root is like ...\indicators\volume
            # depth from indicators root
            rel_path = os.path.relpath(root, INDICATORS_DIR)
            depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
            
            # Construct correct relative path to base
            # If in volume (depth 1), base is ..base
            # If in trend (depth 1), base is ..base
            base_prefix = "." * (depth + 1) + "base"
            
            # Construct correct relative path to core (src/core)
            # If in volume (depth 1), src is .... (4 dots? No)
            # indicators/volume -> indicators -> indicator_engine -> src
            # volume is at src/indicator_engine/indicators/volume
            # src is 3 levels up from volume?
            # volume -> indicators -> indicator_engine -> src
            # So "...." is correct for src?
            # Let's verify:
            # from . import x (same dir)
            # from .. import x (parent)
            # from ... import x (grandparent)
            # from .... import x (great-grandparent)
            
            # src/core/data_contracts.py
            # We want: from ....core.data_contracts import SignalType
            
            # Fix 1: StandardIndicatorInterface
            # Pattern: from ..indicator_engine.indicators.base.standard_indicator
            # This pattern is absolute-ish relative to something?
            # It seems they used a mix.
            
            # Regex might be better but let's try string replacement first for common patterns
            
            # Replace old base import
            content = content.replace(
                "from ..indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface",
                f"from {base_prefix}.standard_indicator import StandardIndicatorInterface"
            )
            
            # Fix 2: SignalType - The big one
            # Patterns found:
            # from ...core.signal_type import SignalType
            # from ..core.signal_type import SignalType
            # from auj_platform.src.core.signal_type import SignalType
            
            # We want to point to src.core.data_contracts
            # If we use absolute import it's safer:
            # from auj_platform.src.core.data_contracts import SignalType
            # But relative is preferred in this codebase?
            # Let's use relative "....core.data_contracts" if depth is 1 (volume, trend, etc)
            
            core_import = "from ....core.data_contracts import SignalType"
            
            content = content.replace("from ....core.signal_type import SignalType", core_import)
            content = content.replace("from ...core.signal_type import SignalType", core_import)
            content = content.replace("from ..core.signal_type import SignalType", core_import)
            content = content.replace("from ..indicator_engine.core.signal_type import SignalType", core_import)
            content = content.replace("from ....core.signal_types import SignalType", core_import)
            content = content.replace("from ...core.signal_types import SignalType", core_import)

            # Fix indicator_base -> base_indicator
            content = content.replace("from ...base.indicator_base import BaseIndicator", f"from {base_prefix}.base_indicator import BaseIndicator")
            content = content.replace("from ..base.indicator_base import BaseIndicator", f"from {base_prefix}.base_indicator import BaseIndicator")

            # Fix exceptions
            # Pattern: from ..indicator_engine.core.exceptions import ...
            # Target: from ....core.exceptions import ...
            content = content.replace("from ..indicator_engine.core.exceptions", "from ....core.exceptions")
            content = content.replace("from ...indicator_engine.core.exceptions", "from ....core.exceptions")
            
            # Also fix BaseIndicator if it's used
            # Pattern: from ..base.base_indicator import BaseIndicator
            # This is actually correct IF base_indicator.py exists in base/
            # I created it, so this import should be valid now.
            
            if content != original_content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Fixed {filename}")
                count += 1
                
    print(f"Total files fixed: {count}")

if __name__ == "__main__":
    fix_imports_recursively()
