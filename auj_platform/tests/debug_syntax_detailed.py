
import sys

def check_syntax_detailed(filename):
    print(f"Checking {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        in_string = False
        string_start_line = -1
        
        # Simple state machine to track triple quotes
        # Note: This is a simplified parser and might be fooled by escaped quotes or quotes in comments
        # but it should be good enough for this specific issue.
        
        total_triple_quotes = 0
        
        for i, line in enumerate(lines):
            # Check for """
            # We need to handle multiple triple quotes on the same line
            line_content = line
            while '"""' in line_content:
                idx = line_content.find('"""')
                total_triple_quotes += 1
                
                if not in_string:
                    in_string = True
                    string_start_line = i + 1
                    print(f"Line {i+1}: String STARTED")
                else:
                    in_string = False
                    print(f"Line {i+1}: String ENDED (Started at {string_start_line})")
                    string_start_line = -1
                
                # Move past this quote
                line_content = line_content[idx+3:]
                
        print(f"Total triple quotes found: {total_triple_quotes}")
        if in_string:
            print(f"ERROR: Unterminated string starting at line {string_start_line}")
        else:
            print("Triple quotes seem balanced.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_syntax_detailed(r"e:\AUG6\auj_platform\src\trading_engine\deal_monitoring_teams.py")
