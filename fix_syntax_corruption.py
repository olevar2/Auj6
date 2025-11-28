import os
import re

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = list(lines)
    modified = False
    
    # Pass 1: Fix explicit patterns
    for i in range(len(new_lines)):
        line = new_lines[i]
        
        # Fix Def End: def ...(args,:) -> def ...(args,
        if re.search(r'def\s+.*,:\)\s*$', line):
            new_lines[i] = line.replace(',:)', ',')
            modified = True
            
        # Fix Dict Start: ({) -> ({
        if line.strip().endswith('({)'):
            new_lines[i] = line.replace('({)', '({')
            modified = True

    # Pass 2: Fix corrupted closers and their openers
    for i in range(len(new_lines)):
        line = new_lines[i]
        
        # Detect corrupted paren close: (            )
        # Regex: Start of line, (, spaces, ), End of line
        paren_match = re.match(r'^(\(\s+)\)$', line)
        if paren_match:
            # Remove leading (
            # The spaces inside are the indentation
            indentation = line[1:].find(')') # Length of spaces
            # Actually, line[1:] is "            )\n"
            # So just removing the first char fixes the line
            new_lines[i] = line[1:]
            modified = True
            
            # Now search backwards for the opener ()
            # We look for a line ending in () with indentation <= current indentation
            # Actually, usually same indentation.
            for j in range(i-1, -1, -1):
                prev_line = new_lines[j]
                if prev_line.strip().endswith('()'):
                    # Check indentation
                    prev_indent = len(prev_line) - len(prev_line.lstrip())
                    # The corrupted line `(            )` has 0 indentation physically, but logically `            ` spaces.
                    # The spaces count is len(line) - 2 (minus ( and ) and newline)
                    # Let's just look for the nearest ()
                    # And verify it's not a valid empty call like func()
                    # If it's valid, it wouldn't be followed by indented block ending in )
                    
                    # Heuristic: Just fix it.
                    # Replace () with (
                    pre, sep, post = prev_line.rpartition('()')
                    new_lines[j] = pre + '(' + post
                    modified = True
                    break
                if prev_line.strip() == '' or prev_line.strip().startswith('#'):
                    continue
                # If we hit a block end or something else, maybe stop?
                # But the block could be long.
                
        # Detect corrupted bracket close: [        ]
        bracket_match = re.match(r'^(\[\s+\])$', line.strip())
        if bracket_match:
            # Remove leading [
            new_lines[i] = line[1:]
            modified = True
            
            # Search backwards for opener []
            for j in range(i-1, -1, -1):
                prev_line = new_lines[j]
                if prev_line.strip().endswith('[]'):
                    pre, sep, post = prev_line.rpartition('[]')
                    new_lines[j] = pre + '[' + post
                    modified = True
                    break

        # Detect corrupted def continuation: (                 arg: type):
        # Or (                 arg: type) -> ret:
        # Regex: Start with (, spaces, word, colon
        if re.match(r'^\(\s+\w+\s*:', line):
            new_lines[i] = line[1:]
            modified = True
            
        # Detect corrupted def continuation ending with ):
        def_cont_match = re.match(r'^(\(\s+.*?\):)\s*$', line)
        if def_cont_match:
            new_lines[i] = line[1:]
            modified = True
            
            # But what if it's (a + b)? Then ) is valid.
            # But (a +) is invalid.
            # So if ) follows an operator, it's definitely wrong (unless inside string/comment).
            new_lines[i] = line.replace(')', '') # Replace last )?
            # Be careful not to replace inside string.
            # Assuming code structure.
            # Use rpartition
            pre, sep, post = line.rpartition(')')
            if pre.strip().endswith(('+', '-', '*', '/')):
                 new_lines[i] = pre + post
                 modified = True


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
