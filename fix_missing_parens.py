import os
import re

def get_indent(line):
    return len(line) - len(line.lstrip())

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    modified = False
    
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        
        # Skip comments/empty lines for logic, but keep them in new_lines
        # We need to look ahead to the next CODE line
        if line.strip().startswith('#') or not line.strip():
            i += 1
            continue
            
        # Check if this line ends with , or (
        # Ignore trailing comments
        code_part = line.split('#')[0].strip()
        if code_part.endswith(',') or code_part.endswith('('):
            current_indent = get_indent(line)
            
            # Find next code line
            j = i + 1
            next_code_line = None
            next_code_indent = 0
            while j < len(lines):
                l = lines[j]
                if l.strip() and not l.strip().startswith('#'):
                    next_code_line = l
                    next_code_indent = get_indent(l)
                    break
                j += 1
            
            if next_code_line:
                # Check if next line is dedented relative to current line
                # If current line ends with (, next line should be indented > current
                # If current line ends with ,, next line should be indented >= current (usually)
                # If next line is dedented, we might be missing a closing paren
                
                # Case 1: Ends with (
                if code_part.endswith('('):
                    if next_code_indent <= current_indent:
                        # Missing )
                        # Insert ) at current_indent
                        indent_str = ' ' * current_indent
                        new_lines.append(f"{indent_str})\n")
                        modified = True
                        print(f"Fixed missing ) after ( at line {i+1}")

                # Case 2: Ends with ,
                elif code_part.endswith(','):
                    # If next line is dedented, we likely finished the block but forgot )
                    # Expected indent for args is usually current_indent.
                    # If next line is LESS indented, we are missing )
                    if next_code_indent < current_indent:
                        # Check if next line starts with ) or ] or }
                        if not next_code_line.strip().startswith((')', ']', '}')):
                            # Insert ) at next_code_indent (or current_indent - 4?)
                            # Usually the closing paren aligns with the start of the call.
                            # The args are indented.
                            # So closing paren should be at current_indent - 4 (assuming 4 space indent)
                            # But safer to use next_code_indent if it matches the outer block.
                            
                            # Let's try to match indentation of the line that started the call?
                            # Too hard to find backwards.
                            # Let's use next_code_indent?
                            # If next_code_indent is 0, and current is 16.
                            # We probably want ) at 12?
                            # But if next line is `return`, indent 12.
                            # Then ) at 12 is correct.
                            
                            # What if multiple levels?
                            # e.g. indent 16 -> indent 4.
                            # We need )))?
                            # This script only inserts one. Run multiple times?
                            
                            indent_str = ' ' * next_code_indent # Or max(next_code_indent, current_indent - 4)?
                            # If we use next_code_indent, it aligns with the next statement.
                            # Which is syntactically valid (usually).
                            
                            new_lines.append(f"{indent_str})\n")
                            modified = True
                            print(f"Fixed missing ) after , at line {i+1}")
        
        i += 1

    if modified:
        print(f"Fixed {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

def process_directory(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                fix_file(os.path.join(root, file))

if __name__ == '__main__':
    target_dir = r'e:\AUG6\auj_platform\src\indicator_engine\indicators'
    process_directory(target_dir)
