#!/usr/bin/env python3
"""Fix all template issues found in validation."""

import re
from pathlib import Path

def fix_template_issues(filepath):
    """Fix common template issues."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix 1: Remove import comment headers
        if re.match(r'^\s*# standard library imports?\s*$', line, re.IGNORECASE):
            modified = True
            i += 1
            continue
        if re.match(r'^\s*# third-party imports?\s*$', line, re.IGNORECASE):
            modified = True
            i += 1
            continue
        if re.match(r'^\s*# local imports?\s*$', line, re.IGNORECASE):
            modified = True
            i += 1
            continue
        
        # Fix 2: Capitalize code comments (but preserve special cases)
        if re.match(r'^(\s*)# ([a-z])', line):
            # Skip URLs, special phrases
            if not any(x in line for x in ['http://', 'https://', '# e.g', '# i.e', '# etc', '.com', 'ray.', 'num_cpus']):
                original_line = line
                # Capitalize first letter after '# '
                match = re.match(r'^(\s*# )([a-z])(.*)', line)
                if match:
                    line = match.group(1) + match.group(2).upper() + match.group(3)
                    if line != original_line:
                        modified = True
        
        new_lines.append(line)
        i += 1
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    return modified

def main():
    """Fix issues in all templates."""
    templates_dir = Path('/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates')
    
    md_files = []
    for pattern in ['*/README.md', '*/01-*.md', '*/02-*.md']:
        md_files.extend(templates_dir.glob(pattern))
    
    md_files = [f for f in md_files if 'BACKUP' not in f.name and 'TEMP' not in f.name]
    
    print(f"Fixing template issues in {len(md_files)} files...\n")
    
    fixed_count = 0
    for md_file in sorted(md_files):
        if fix_template_issues(md_file):
            print(f"  ✓ {md_file.parent.name}/{md_file.name}")
            fixed_count += 1
    
    print(f"\nCompleted: {fixed_count}/{len(md_files)} files modified")

if __name__ == '__main__':
    main()

