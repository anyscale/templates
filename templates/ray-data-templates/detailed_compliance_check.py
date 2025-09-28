#!/usr/bin/env python3
"""
Detailed rule compliance checker that shows exactly what violations are found.
"""

import os
import json
import re

def check_professional_terminology_detailed(content, is_notebook=False):
    """Check for specific professional terminology issues."""
    
    if is_notebook:
        try:
            nb_data = json.loads(content) if isinstance(content, str) else content
            all_text = '\n'.join([
                '\n'.join(cell.get('source', [])) 
                for cell in nb_data.get('cells', [])
            ])
        except:
            return {}
    else:
        all_text = content
    
    # Check for specific unprofessional terms
    violations = []
    
    unprofessional_terms = [
        'superpowers', 'awesome', 'amazing', 'incredible', 'fantastic', 
        'magic', 'killer', 'badass', 'sick', 'insane', 'crazy good',
        'mind-blowing', 'game-changer', 'revolutionary', 'cutting-edge'
    ]
    
    for term in unprofessional_terms:
        if term.lower() in all_text.lower():
            # Find specific occurrences
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            matches = pattern.findall(all_text)
            if matches:
                violations.append(f"'{term}' found {len(matches)} times")
    
    return violations

def check_specific_template_issues(template_name, template_path):
    """Check specific issues for a template."""
    
    md_path = os.path.join(template_path, "README.md")
    nb_path = os.path.join(template_path, "README.ipynb")
    
    results = {'template_name': template_name, 'issues': []}
    
    # Check markdown
    if os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        md_violations = check_professional_terminology_detailed(md_content, False)
        if md_violations:
            results['issues'].extend([f"MD: {v}" for v in md_violations])
    
    # Check notebook
    if os.path.exists(nb_path):
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb_content = json.load(f)
        
        nb_violations = check_professional_terminology_detailed(nb_content, True)
        if nb_violations:
            results['issues'].extend([f"NB: {v}" for v in nb_violations])
    
    return results

def detailed_compliance_check():
    """Run detailed compliance check showing specific violations."""
    
    base_dir = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates"
    
    templates = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("ray-data-"):
            templates.append((item, item_path))
    
    print("Detailed Rule Compliance Check")
    print("=" * 60)
    
    for template_name, template_path in sorted(templates):
        results = check_specific_template_issues(template_name, template_path)
        
        if results['issues']:
            print(f"❌ {template_name}")
            for issue in results['issues']:
                print(f"   {issue}")
        else:
            print(f"✅ {template_name}")
    
    print("=" * 60)

if __name__ == "__main__":
    detailed_compliance_check()
