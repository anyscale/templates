#!/usr/bin/env python3
"""Fix trailing parenthesis syntax errors in read_* calls."""

import re
from pathlib import Path

def fix_trailing_paren_syntax(filepath):
    """Fix syntax like ', num_cpus=0.025)' to proper format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Pattern 1: Fix ', num_cpus=X)' at end of read calls
    # This should be ',\n    num_cpus=X\n)'
    pattern1 = r'(\("s3://[^"]+")(\s*, num_cpus=0\.\d+\))'
    
    def replace_func(match):
        path_part = match.group(1)
        param_part = match.group(2).strip()
        # Extract the num_cpus value
        num_cpus_match = re.search(r'num_cpus=(0\.\d+)', param_part)
        if num_cpus_match:
            num_cpus = num_cpus_match.group(1)
            return f'{path_part},\n    num_cpus={num_cpus}\n)'
        return match.group(0)
    
    content = re.sub(pattern1, replace_func, content)
    
    # Pattern 2: Fix standalone ', num_cpus=X)' on separate lines
    content = re.sub(r'^, num_cpus=(0\.\d+)\)', r',\n    num_cpus=\1\n)', content, flags=re.MULTILINE)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix syntax errors in all templates."""
    templates_dir = Path('/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates')
    
    md_files = []
    for pattern in ['*/README.md', '*/01-*.md', '*/02-*.md']:
        md_files.extend(templates_dir.glob(pattern))
    
    md_files = [f for f in md_files if 'BACKUP' not in f.name and 'TEMP' not in f.name]
    
    print(f"Fixing trailing parenthesis syntax errors in {len(md_files)} files...\n")
    
    fixed_count = 0
    for md_file in sorted(md_files):
        if fix_trailing_paren_syntax(md_file):
            print(f"  ✓ Fixed: {md_file.parent.name}/{md_file.name}")
            fixed_count += 1
    
    print(f"\nCompleted: {fixed_count}/{len(md_files)} files modified")

if __name__ == '__main__':
    main()

