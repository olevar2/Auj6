
import sys

def check_syntax(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for triple quotes
        count = content.count('"""')
        print(f"Triple double quotes count: {count}")
        
        if count % 2 != 0:
            print("ODD NUMBER OF TRIPLE DOUBLE QUOTES!")
            # Find locations
            import re
            matches = [m.start() for m in re.finditer('"""', content)]
            for i, pos in enumerate(matches):
                line_num = content[:pos].count('\n') + 1
                print(f"Match {i+1} at line {line_num}")
                
        # Check for triple single quotes
        count_single = content.count("'''")
        print(f"Triple single quotes count: {count_single}")
        
        if count_single % 2 != 0:
            print("ODD NUMBER OF TRIPLE SINGLE QUOTES!")
            
        # Try to compile
        compile(content, filename, 'exec')
        print("Syntax OK")
        
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        print(f"Line: {e.lineno}")
        print(f"Offset: {e.offset}")
        print(f"Text: {e.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_syntax(r"e:\AUG6\auj_platform\src\trading_engine\deal_monitoring_teams.py")
