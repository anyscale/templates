#!/usr/bin/env python3
"""
Fix synchronization issues between README.md and README.ipynb files.

This script identifies and fixes content mismatches to ensure perfect
synchronization between markdown and notebook formats.
"""

import os
import json
import re
from pathlib import Path

def fix_notebook_from_markdown(template_path: str, template_name: str) -> bool:
    """Re-synchronize notebook with markdown content."""
    
    md_path = os.path.join(template_path, "README.md")
    nb_path = os.path.join(template_path, "README.ipynb")
    
    if not os.path.exists(md_path):
        return False
    
    try:
        # Read markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Create new notebook from markdown
        import nbformat as nbf
        
        nb = nbf.v4.new_notebook()
        
        # Split content by code blocks
        sections = re.split(r'```python\n(.*?)```', md_content, flags=re.DOTALL)
        
        for i, section in enumerate(sections):
            if i % 2 == 0:  # Text section
                if section.strip():
                    # Clean up the markdown
                    clean_text = section.strip()
                    clean_text = re.sub(r'```\s*$', '', clean_text)
                    nb.cells.append(nbf.v4.new_markdown_cell(clean_text))
            else:  # Code section
                if section.strip():
                    # Clean and fix code
                    clean_code = section.strip()
                    clean_code = clean_code.replace('ray.init()', 'ray.init(ignore_reinit_error=True)')
                    clean_code = re.sub(r'```\s*$', '', clean_code)
                    
                    # Only add if there's actual code content
                    if clean_code.strip() and not clean_code.strip().startswith('```'):
                        nb.cells.append(nbf.v4.new_code_cell(clean_code))
        
        # Write synchronized notebook
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        
        return True
        
    except Exception as e:
        print(f"Error synchronizing {template_name}: {e}")
        return False

def fix_all_synchronization_issues():
    """Fix synchronization issues in all templates."""
    
    base_dir = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates"
    
    templates = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("ray-data-"):
            templates.append((item, item_path))
    
    print(f"Re-synchronizing {len(templates)} templates...")
    print("=" * 60)
    
    success_count = 0
    for template_name, template_path in sorted(templates):
        if fix_notebook_from_markdown(template_path, template_name):
            print(f"✅ {template_name:<35} Synchronized")
            success_count += 1
        else:
            print(f"❌ {template_name:<35} Failed")
    
    print("=" * 60)
    print(f"Synchronization completed: {success_count}/{len(templates)} templates")
    
    return success_count == len(templates)

if __name__ == "__main__":
    success = fix_all_synchronization_issues()
    exit(0 if success else 1)
