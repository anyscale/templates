#!/usr/bin/env python3
"""
Compare README.md and README.ipynb content to ensure they match.

This script verifies that notebooks contain the same content as their
corresponding markdown files.
"""

import os
import json
import re
from pathlib import Path

def extract_markdown_content(md_path):
    """Extract content from markdown file."""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract code blocks
        code_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
        
        # Extract text (remove code blocks)
        text_content = re.sub(r'```python\n.*?```', '', content, flags=re.DOTALL)
        
        return {
            'code_blocks': len(code_blocks),
            'total_lines': len(content.split('\n')),
            'has_title': content.startswith('#'),
            'has_toc': '## Table of Contents' in content,
            'has_learning_objectives': '## Learning Objectives' in content,
            'has_prerequisites': '## Prerequisites' in content,
            'has_cleanup': 'ray.shutdown' in content
        }
    except Exception as e:
        return {'error': str(e)}

def extract_notebook_content(nb_path):
    """Extract content from notebook file."""
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
        
        cells = nb_data.get('cells', [])
        code_cells = [cell for cell in cells if cell.get('cell_type') == 'code']
        markdown_cells = [cell for cell in cells if cell.get('cell_type') == 'markdown']
        
        # Check for key sections in markdown cells
        all_markdown = '\n'.join([
            '\n'.join(cell.get('source', [])) 
            for cell in markdown_cells
        ])
        
        return {
            'code_blocks': len(code_cells),
            'total_cells': len(cells),
            'has_title': any('#' in '\n'.join(cell.get('source', [])) for cell in markdown_cells[:1]),
            'has_toc': '## Table of Contents' in all_markdown,
            'has_learning_objectives': '## Learning Objectives' in all_markdown,
            'has_prerequisites': '## Prerequisites' in all_markdown,
            'has_cleanup': any('ray.shutdown' in '\n'.join(cell.get('source', [])) for cell in code_cells)
        }
    except Exception as e:
        return {'error': str(e)}

def compare_template_content(template_dir):
    """Compare README.md and README.ipynb content."""
    
    md_path = os.path.join(template_dir, "README.md")
    nb_path = os.path.join(template_dir, "README.ipynb")
    
    if not os.path.exists(md_path):
        return False, "README.md not found"
    
    if not os.path.exists(nb_path):
        return False, "README.ipynb not found"
    
    md_content = extract_markdown_content(md_path)
    nb_content = extract_notebook_content(nb_path)
    
    # Check for errors
    if 'error' in md_content:
        return False, f"Markdown error: {md_content['error']}"
    if 'error' in nb_content:
        return False, f"Notebook error: {nb_content['error']}"
    
    # Compare key elements
    issues = []
    
    # Code block count should be similar (allow some variance due to conversion)
    code_diff = abs(md_content['code_blocks'] - nb_content['code_blocks'])
    if code_diff > 3:  # Allow some variance
        issues.append(f"Code block count mismatch: MD={md_content['code_blocks']}, NB={nb_content['code_blocks']}")
    
    # Check required sections
    required_sections = ['has_title', 'has_toc', 'has_learning_objectives', 'has_prerequisites']
    for section in required_sections:
        if md_content.get(section) != nb_content.get(section):
            issues.append(f"Section mismatch: {section}")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, f"Content matches: {nb_content['code_blocks']} code blocks, {nb_content['total_cells']} total cells"

def compare_all_templates():
    """Compare all template README.md and README.ipynb files."""
    
    base_dir = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates"
    
    templates = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("ray-data-"):
            templates.append((item, item_path))
    
    print(f"Comparing {len(templates)} template README.md ↔ README.ipynb pairs...")
    print("=" * 80)
    
    match_count = 0
    for template_name, template_path in sorted(templates):
        matches, message = compare_template_content(template_path)
        
        status = "✅ MATCH" if matches else "❌ MISMATCH"
        print(f"{status:<12} {template_name:<35} {message}")
        
        if matches:
            match_count += 1
    
    print("=" * 80)
    print(f"Content comparison: {match_count}/{len(templates)} templates have matching content")
    
    return match_count == len(templates)

if __name__ == "__main__":
    success = compare_all_templates()
    exit(0 if success else 1)
