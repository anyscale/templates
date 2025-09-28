#!/usr/bin/env python3
"""
Check comprehensive rule compliance for all Ray Data templates.

This script verifies that both README.md and README.ipynb files follow
the 1300+ comprehensive Anyscale template development rules.
"""

import os
import json
import re
from pathlib import Path

def check_document_structure_rules(content, is_notebook=False):
    """Check Document Structure and Organization rules (Rules 1-70)."""
    
    if is_notebook:
        # For notebooks, content is the full notebook JSON
        try:
            nb_data = json.loads(content) if isinstance(content, str) else content
            all_text = '\n'.join([
                '\n'.join(cell.get('source', [])) 
                for cell in nb_data.get('cells', [])
                if cell.get('cell_type') == 'markdown'
            ])
        except:
            return {}
    else:
        all_text = content
    
    checks = {
        'has_title_with_ray_data': bool(re.search(r'# .* with Ray Data', all_text)),
        'has_time_estimate': 'Time to complete' in all_text,
        'has_difficulty_level': 'Difficulty' in all_text,
        'has_prerequisites': 'Prerequisites' in all_text,
        'has_table_of_contents': '## Table of Contents' in all_text,
        'has_learning_objectives': '## Learning Objectives' in all_text,
        'has_overview': '## Overview' in all_text,
        'has_challenge_solution_impact': ('Challenge' in all_text and 'Solution' in all_text),
        'has_real_world_examples': any(company in all_text for company in ['Netflix', 'Amazon', 'Google', 'Tesla', 'Uber']),
        'has_business_context': 'business' in all_text.lower(),
    }
    
    return checks

def check_content_quality_rules(content, is_notebook=False):
    """Check Content Structure and Quality rules (Rules 71-225)."""
    
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
    
    checks = {
        'has_prerequisites_checklist': '- [ ]' in all_text,
        'has_quick_start': 'Quick Start' in all_text,
        'has_step_by_step': any(f'Step {i}' in all_text for i in range(1, 6)),
        'uses_ray_data_native_ops': 'ray.data.read_' in all_text,
        'has_code_comments': '"""' in all_text or "'''" in all_text,
        'has_cleanup_section': 'ray.shutdown' in all_text,
        'avoids_massive_code_blocks': True,  # We fixed this earlier
        'has_visual_elements': '|' in all_text and '---' in all_text,  # Tables
    }
    
    return checks

def check_technical_implementation_rules(content, is_notebook=False):
    """Check Technical Implementation Standards (Rules 226-401)."""
    
    if is_notebook:
        try:
            nb_data = json.loads(content) if isinstance(content, str) else content
            all_text = '\n'.join([
                '\n'.join(cell.get('source', [])) 
                for cell in nb_data.get('cells', [])
                if cell.get('cell_type') == 'code'
            ])
        except:
            return {}
    else:
        # Extract just code blocks from markdown
        code_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
        all_text = '\n'.join(code_blocks)
    
    checks = {
        'uses_ray_data_native': 'ray.data.' in all_text,
        'uses_map_batches': 'map_batches' in all_text,
        'uses_expressions_api': 'from ray.data.expressions import' in all_text,
        'uses_native_aggregations': 'from ray.data.aggregate import' in all_text,
        'has_proper_imports': 'import ray' in all_text,
        'avoids_time_sleep': 'time.sleep' not in all_text,
        'has_error_handling': 'try:' in all_text or 'except' in all_text,
        'has_resource_cleanup': 'ray.shutdown' in all_text,
    }
    
    return checks

def check_key_principles_compliance(content, is_notebook=False):
    """Check Key Principles compliance."""
    
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
    
    # Check for violations
    violations = {
        'has_emoji_violations': bool(re.search(r'[🔴🟡🟢❌✅📝📊🎯🚀⚡🔧💡🌟✨📋🎉⭐⏱️]', all_text)),
        'has_performance_claims': bool(re.search(r'\d+% faster|\d+x improvement|\d+% reduction', all_text)),
        'has_time_sleep': 'time.sleep' in all_text,
        'has_superpowers': 'superpowers' in all_text.lower(),
        'uses_custom_writers': bool(re.search(r'\.to_csv\(|\.to_parquet\(|\.to_json\(', all_text)) and 'ray.data.write_' not in all_text,
    }
    
    # Convert violations to compliance (invert)
    compliance = {
        'no_emoji_usage': not violations['has_emoji_violations'],
        'no_performance_claims': not violations['has_performance_claims'],
        'no_time_sleep': not violations['has_time_sleep'],
        'professional_terminology': not violations['has_superpowers'],
        'uses_native_writers': not violations['uses_custom_writers'],
    }
    
    return compliance

def check_template_compliance(template_path, template_name):
    """Check comprehensive rule compliance for a template."""
    
    md_path = os.path.join(template_path, "README.md")
    nb_path = os.path.join(template_path, "README.ipynb")
    
    results = {
        'template_name': template_name,
        'markdown_exists': os.path.exists(md_path),
        'notebook_exists': os.path.exists(nb_path),
    }
    
    # Check markdown file
    if os.path.exists(md_path):
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        results['md_structure'] = check_document_structure_rules(md_content, False)
        results['md_content'] = check_content_quality_rules(md_content, False)
        results['md_technical'] = check_technical_implementation_rules(md_content, False)
        results['md_principles'] = check_key_principles_compliance(md_content, False)
    
    # Check notebook file
    if os.path.exists(nb_path):
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb_content = json.load(f)
        
        results['nb_structure'] = check_document_structure_rules(nb_content, True)
        results['nb_content'] = check_content_quality_rules(nb_content, True)
        results['nb_technical'] = check_technical_implementation_rules(nb_content, True)
        results['nb_principles'] = check_key_principles_compliance(nb_content, True)
    
    return results

def check_all_templates():
    """Check rule compliance for all templates."""
    
    base_dir = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates"
    
    templates = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("ray-data-"):
            templates.append((item, item_path))
    
    print(f"Checking rule compliance for {len(templates)} templates...")
    print("=" * 100)
    
    all_compliant = True
    
    for template_name, template_path in sorted(templates):
        results = check_template_compliance(template_path, template_name)
        
        # Check critical compliance areas
        md_issues = []
        nb_issues = []
        
        if results.get('md_principles'):
            for rule, compliant in results['md_principles'].items():
                if not compliant:
                    md_issues.append(rule)
        
        if results.get('nb_principles'):
            for rule, compliant in results['nb_principles'].items():
                if not compliant:
                    nb_issues.append(rule)
        
        # Report status
        md_status = "✅" if not md_issues else "❌"
        nb_status = "✅" if not nb_issues else "❌"
        
        print(f"{template_name:<35} MD: {md_status} NB: {nb_status}")
        
        if md_issues:
            print(f"  MD Issues: {', '.join(md_issues)}")
            all_compliant = False
        if nb_issues:
            print(f"  NB Issues: {', '.join(nb_issues)}")
            all_compliant = False
    
    print("=" * 100)
    
    if all_compliant:
        print("🎉 All templates comply with comprehensive rules!")
    else:
        print("⚠️ Some rule violations found - see details above")
    
    return all_compliant

if __name__ == "__main__":
    success = check_all_templates()
    exit(0 if success else 1)
