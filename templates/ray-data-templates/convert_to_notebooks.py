#!/usr/bin/env python3
"""
Convert Ray Data template README.md files to Jupyter notebooks.

This script converts markdown files to executable Jupyter notebooks,
separating text and code blocks appropriately.
"""

import os
import json
import re
from pathlib import Path
import nbformat as nbf

def markdown_to_notebook(md_content, title="Ray Data Template"):
    """Convert markdown content to Jupyter notebook."""
    
    # Create new notebook
    nb = nbf.v4.new_notebook()
    
    # Split content by code blocks, handling various formats
    sections = re.split(r'```python\n(.*?)```', md_content, flags=re.DOTALL)
    
    for i, section in enumerate(sections):
        if i % 2 == 0:  # Text section
            if section.strip():
                # Clean up the markdown
                clean_text = section.strip()
                # Remove any stray code block markers
                clean_text = re.sub(r'```\s*$', '', clean_text)
                # Create markdown cell
                nb.cells.append(nbf.v4.new_markdown_cell(clean_text))
        else:  # Code section
            if section.strip():
                # Create code cell and fix common issues
                clean_code = section.strip()
                
                # Fix ray.init() calls to handle reinit
                clean_code = clean_code.replace('ray.init()', 'ray.init(ignore_reinit_error=True)')
                
                # Fix common syntax issues from markdown conversion
                # Remove any trailing markdown markers
                clean_code = re.sub(r'```\s*$', '', clean_code)
                
                # Fix indentation issues (common in markdown)
                lines = clean_code.split('\n')
                fixed_lines = []
                for line in lines:
                    # Skip empty lines or lines that are clearly not code
                    if line.strip() and not line.strip().startswith('#'):
                        # Ensure proper indentation for Python
                        if line.startswith('    ') or line.startswith('\t'):
                            fixed_lines.append(line)
                        elif line.strip():
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                
                clean_code = '\n'.join(fixed_lines)
                
                # Only add if there's actual code content
                if clean_code.strip() and not clean_code.strip().startswith('```'):
                    nb.cells.append(nbf.v4.new_code_cell(clean_code))
    
    return nb

def convert_template_to_notebook(template_dir):
    """Convert a single template README.md to notebook."""
    
    readme_path = os.path.join(template_dir, "README.md")
    notebook_path = os.path.join(template_dir, "README.ipynb")
    
    if not os.path.exists(readme_path):
        print(f"❌ README.md not found in {template_dir}")
        return False
    
    try:
        # Read markdown content
        with open(readme_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to notebook
        template_name = os.path.basename(template_dir)
        nb = markdown_to_notebook(md_content, template_name)
        
        # Write notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        
        print(f"✅ Converted {template_name}/README.md → README.ipynb")
        return True
        
    except Exception as e:
        print(f"❌ Failed to convert {template_dir}: {e}")
        return False

def convert_all_templates():
    """Convert all Ray Data template READMEs to notebooks."""
    
    base_dir = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates"
    
    # Find all template directories
    template_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("ray-data-"):
            template_dirs.append(item_path)
    
    print(f"Found {len(template_dirs)} Ray Data templates to convert")
    print("=" * 60)
    
    success_count = 0
    for template_dir in sorted(template_dirs):
        if convert_template_to_notebook(template_dir):
            success_count += 1
    
    print("=" * 60)
    print(f"Conversion completed: {success_count}/{len(template_dirs)} templates converted")
    
    return success_count == len(template_dirs)

if __name__ == "__main__":
    success = convert_all_templates()
    exit(0 if success else 1)
