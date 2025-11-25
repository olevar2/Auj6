import sys
from pathlib import Path
import re

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from registry.agent_indicator_mapping import AGENT_MAPPINGS
except ImportError:
    # Fallback if import fails (e.g. if run from wrong dir)
    sys.path.insert(0, str(project_root / "src"))
    from registry.agent_indicator_mapping import AGENT_MAPPINGS

def get_active_indicators():
    indicators = set()
    for agent, data in AGENT_MAPPINGS.items():
        indicators.update(data["assigned_indicators"])
    return indicators

def prune_requirements_file():
    active_indicators = get_active_indicators()
    print(f"Found {len(active_indicators)} active indicators.")
    
    req_file = project_root / "config" / "indicator_data_requirements.py"
    
    with open(req_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # We need to parse the file content to find keys in the INDICATOR_DATA_REQUIREMENTS dict
    # This is a bit complex with regex, but we can try to identify blocks
    
    # Strategy: 
    # 1. Read the file line by line
    # 2. Identify the start of the dictionary
    # 3. Identify keys like "indicator_name": IndicatorDataRequirement(...)
    # 4. Keep only if indicator_name is in active_indicators
    
    new_lines = []
    in_dict = False
    skip_block = False
    current_indicator = None
    
    lines = content.split('\n')
    
    # Keep imports and class definitions
    dict_start_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("INDICATOR_DATA_REQUIREMENTS: Dict[str, IndicatorDataRequirement] = {"):
            dict_start_line = i
            break
        new_lines.append(line)
        
    new_lines.append(lines[dict_start_line]) # Add the dict start line
    
    # Process the dictionary body
    i = dict_start_line + 1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check for end of dict
        if stripped == "}":
            new_lines.append(line)
            i += 1
            continue
            
        # Check for new entry
        # Format: "indicator_name": IndicatorDataRequirement(
        match = re.match(r'\s*"([^"]+)": IndicatorDataRequirement\(', line)
        if match:
            indicator_name = match.group(1)
            if indicator_name in active_indicators:
                skip_block = False
                new_lines.append(line)
            else:
                skip_block = True
                print(f"Pruning requirement for: {indicator_name}")
        elif skip_block:
            # We are inside a block we want to skip
            # Check if block ends (usually with ), or ),\n)
            if stripped.endswith("),") or stripped.endswith(")") or (stripped == "" and i+1 < len(lines) and "IndicatorDataRequirement" in lines[i+1]):
                 # This logic is tricky for multi-line calls. 
                 # Safer: just skip until we see the closing parenthesis of the call
                 pass
        else:
            # Not skipping, just add the line
            new_lines.append(line)
            
        i += 1
        
    # Re-write file
    # This regex approach is risky. Let's try a safer AST-based or just simpler block removal if possible.
    # Actually, since the file structure is very regular, we can count parens.
    
    # Alternative: Read the whole file, identify the dict, and reconstruct it.
    # But that's hard to format.
    
    # Let's try a simpler approach:
    # We will read the file, identify the keys, and if a key is not in active, we comment it out or remove it.
    
    pass

def prune_requirements_simple():
    active_indicators = get_active_indicators()
    req_file = project_root / "config" / "indicator_data_requirements.py"
    
    with open(req_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    new_lines = []
    skip_mode = False
    paren_count = 0
    
    for line in lines:
        # Check for start of an entry
        match = re.search(r'"([^"]+)": IndicatorDataRequirement\(', line)
        if match:
            indicator_name = match.group(1)
            if indicator_name not in active_indicators:
                skip_mode = True
                paren_count = line.count('(') - line.count(')')
                # print(f"Removing {indicator_name}...")
                continue
            else:
                skip_mode = False
        
        if skip_mode:
            paren_count += line.count('(') - line.count(')')
            if paren_count <= 0:
                skip_mode = False
            continue
            
        new_lines.append(line)
        
    with open(req_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
        
    print("Pruning complete.")

if __name__ == "__main__":
    prune_requirements_simple()
