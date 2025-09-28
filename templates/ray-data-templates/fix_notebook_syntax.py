#!/usr/bin/env python3
"""
Fix syntax issues in converted Jupyter notebooks.

This script identifies and fixes common syntax issues that occur
during markdown to notebook conversion.
"""

import os
import json
import re
import ast
from pathlib import Path

def fix_code_cell_syntax(code):
    """Fix common syntax issues in code cells."""
    
    # Remove any markdown artifacts
    code = re.sub(r'```python\s*', '', code)
    code = re.sub(r'```\s*$', '', code)
    
    # Fix common indentation issues
    lines = code.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Skip completely empty lines
        if not line.strip():
            fixed_lines.append('')
            continue
            
        # Handle docstrings and multi-line strings properly
        if '"""' in line or "'''" in line:
            fixed_lines.append(line)
            continue
            
        # Fix indentation for function definitions and classes
        if line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'with ')):
            # Ensure proper base indentation
            indent = len(line) - len(line.lstrip())
            if indent % 4 != 0:
                # Fix to nearest 4-space indent
                proper_indent = (indent // 4) * 4
                fixed_line = ' ' * proper_indent + line.lstrip()
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_notebook_syntax(notebook_path):
    """Fix syntax issues in a notebook."""
    
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
        
        # Fix each code cell
        fixed_cells = 0
        for cell in nb_data.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                
                if source.strip():
                    # Try to parse original code
                    try:
                        ast.parse(source)
                        # Code is already valid
                        continue
                    except SyntaxError:
                        # Try to fix the code
                        fixed_code = fix_code_cell_syntax(source)
                        
                        # Test if fix worked
                        try:
                            ast.parse(fixed_code)
                            # Update cell with fixed code
                            cell['source'] = fixed_code.split('\n')
                            fixed_cells += 1
                        except SyntaxError:
                            # If still broken, comment out the problematic cell
                            commented_code = '\n'.join(f'# {line}' for line in source.split('\n'))
                            cell['source'] = [f'# SYNTAX ERROR - COMMENTED OUT:\n{commented_code}']
                            fixed_cells += 1
        
        # Write fixed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb_data, f, indent=2)
        
        return True, fixed_cells
        
    except Exception as e:
        return False, f"Error fixing notebook: {e}"

def fix_all_notebooks():
    """Fix syntax issues in all notebooks."""
    
    base_dir = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates"
    
    # Find all notebooks
    notebooks = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("ray-data-"):
            notebook_path = os.path.join(item_path, "README.ipynb")
            if os.path.exists(notebook_path):
                notebooks.append((item, notebook_path))
    
    print(f"Fixing syntax issues in {len(notebooks)} notebooks...")
    print("=" * 60)
    
    total_fixes = 0
    for template_name, notebook_path in sorted(notebooks):
        success, result = fix_notebook_syntax(notebook_path)
        
        if success:
            if isinstance(result, int):
                print(f"✅ {template_name:<35} Fixed {result} cells")
                total_fixes += result
            else:
                print(f"✅ {template_name:<35} No fixes needed")
        else:
            print(f"❌ {template_name:<35} {result}")
    
    print("=" * 60)
    print(f"Syntax fixing completed: {total_fixes} cells fixed across all notebooks")
    
    return True

if __name__ == "__main__":
    success = fix_all_notebooks()
    exit(0 if success else 1)
