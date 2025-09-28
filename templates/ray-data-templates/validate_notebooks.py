#!/usr/bin/env python3
"""
Validate Ray Data template notebooks without executing them.

This script checks notebook structure, cell types, and basic syntax
without requiring full execution (avoiding pandas/numpy issues).
"""

import os
import json
from pathlib import Path
import ast

def validate_python_syntax(code):
    """Validate Python code syntax without executing."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def validate_notebook(notebook_path):
    """Validate a single notebook structure and syntax."""
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
        
        # Check basic notebook structure
        if 'cells' not in nb_data:
            return False, "No cells found in notebook"
        
        cells = nb_data['cells']
        code_cells = 0
        markdown_cells = 0
        syntax_errors = []
        
        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type', 'unknown')
            
            if cell_type == 'code':
                code_cells += 1
                # Check Python syntax
                source = ''.join(cell.get('source', []))
                if source.strip():
                    valid, error = validate_python_syntax(source)
                    if not valid:
                        syntax_errors.append(f"Cell {i+1}: {error}")
                        
            elif cell_type == 'markdown':
                markdown_cells += 1
        
        # Validation results
        total_cells = len(cells)
        
        if syntax_errors:
            return False, f"Syntax errors found: {'; '.join(syntax_errors)}"
        
        if code_cells == 0:
            return False, "No code cells found"
        
        if markdown_cells == 0:
            return False, "No markdown cells found"
        
        return True, f"Valid notebook: {total_cells} cells ({code_cells} code, {markdown_cells} markdown)"
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"

def validate_all_notebooks():
    """Validate all Ray Data template notebooks."""
    
    base_dir = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates"
    
    # Find all notebooks
    notebooks = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("ray-data-"):
            notebook_path = os.path.join(item_path, "README.ipynb")
            if os.path.exists(notebook_path):
                notebooks.append((item, notebook_path))
    
    print(f"Found {len(notebooks)} Ray Data template notebooks to validate")
    print("=" * 80)
    
    valid_count = 0
    for template_name, notebook_path in sorted(notebooks):
        valid, message = validate_notebook(notebook_path)
        
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"{status:<12} {template_name:<35} {message}")
        
        if valid:
            valid_count += 1
    
    print("=" * 80)
    print(f"Validation completed: {valid_count}/{len(notebooks)} notebooks are valid")
    
    # Additional checks
    print("\nNotebook Structure Analysis:")
    print("-" * 40)
    
    for template_name, notebook_path in sorted(notebooks):
        try:
            with open(notebook_path, 'r') as f:
                nb_data = json.load(f)
            
            cells = nb_data.get('cells', [])
            code_cells = sum(1 for cell in cells if cell.get('cell_type') == 'code')
            markdown_cells = sum(1 for cell in cells if cell.get('cell_type') == 'markdown')
            
            print(f"{template_name:<35} Code: {code_cells:<3} Markdown: {markdown_cells:<3} Total: {len(cells)}")
            
        except Exception as e:
            print(f"{template_name:<35} ERROR: {e}")
    
    return valid_count == len(notebooks)

if __name__ == "__main__":
    success = validate_all_notebooks()
    exit(0 if success else 1)
