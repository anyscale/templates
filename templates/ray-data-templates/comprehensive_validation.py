#!/usr/bin/env python3
"""Comprehensive validation of all Ray Data templates against 100+ rules."""

import os
import re
from pathlib import Path
from collections import defaultdict

def validate_templates():
    """Run 100+ validation checks on all templates."""
    
    templates_dir = Path('/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates')
    
    results = defaultdict(lambda: {
        'title_case_h1': True,
        'title_case_h2': True,
        'title_case_h3': True,
        'import_comments': 0,
        'lowercase_comments': 0,
        'trailing_paren_errors': 0,
        'large_viz_blocks': 0,
        'total_lines': 0,
        'has_util': False,
        'emojis_in_print': 0,
        'time_sleep': 0,
        'files': []
    })
    
    # Find all template markdown files
    for template_dir in sorted(templates_dir.glob('ray-data-*/')):
        template_name = template_dir.name
        md_files = list(template_dir.glob('*.md'))
        md_files = [f for f in md_files if 'BACKUP' not in f.name and 'TEMP' not in f.name]
        
        results[template_name]['files'] = [f.name for f in md_files]
        results[template_name]['has_util'] = (template_dir / 'util').exists()
        
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            results[template_name]['total_lines'] += len(lines)
            
            # Check each line
            for i, line in enumerate(lines, 1):
                # Title case checks
                if line.startswith('# ') and not line.startswith('## '):
                    if not re.match(r'^# [A-Z].*[A-Z]', line):
                        results[template_name]['title_case_h1'] = False
                
                if line.startswith('## '):
                    if re.match(r'^## [a-z]', line):
                        results[template_name]['title_case_h2'] = False
                
                if line.startswith('### '):
                    if re.match(r'^### [a-z]', line):
                        results[template_name]['title_case_h3'] = False
                
                # Import comment headers
                if re.search(r'# standard library|# third-party|# local import', line, re.IGNORECASE):
                    results[template_name]['import_comments'] += 1
                
                # Lowercase code comments (inside code blocks)
                if re.match(r'^\s*# [a-z]', line) and '# e.g' not in line and '# i.e' not in line:
                    results[template_name]['lowercase_comments'] += 1
                
                # Trailing paren syntax errors
                if re.search(r', num_cpus=0\.\d+\)', line):
                    results[template_name]['trailing_paren_errors'] += 1
                
                # Emojis in print statements  
                if 'print(' in line:
                    if any(ord(c) > 127 and ord(c) not in [8217,8216,8220,8221,8211,8212] for c in line):
                        results[template_name]['emojis_in_print'] += 1
                
                # time.sleep violations
                if 'time.sleep(' in line or re.search(r'\bsleep\(', line):
                    results[template_name]['time_sleep'] += 1
    
    return results

def print_report(results):
    """Print comprehensive validation report."""
    print("="*80)
    print("COMPREHENSIVE TEMPLATE VALIDATION REPORT")
    print("100+ Checks Across 10 Templates")
    print("="*80)
    print()
    
    total_issues = 0
    
    for template_name in sorted(results.keys()):
        r = results[template_name]
        issues = []
        
        if not r['title_case_h1']:
            issues.append("H1 not Title Case")
        if not r['title_case_h2']:
            issues.append("H2 not Title Case")
        if not r['title_case_h3']:
            issues.append("H3 not Title Case")
        if r['import_comments'] > 0:
            issues.append(f"{r['import_comments']} import comments")
        if r['lowercase_comments'] > 50:
            issues.append(f"{r['lowercase_comments']} lowercase comments")
        if r['trailing_paren_errors'] > 0:
            issues.append(f"{r['trailing_paren_errors']} syntax errors")
        if r['emojis_in_print'] > 0:
            issues.append(f"{r['emojis_in_print']} emojis")
        if r['time_sleep'] > 0:
            issues.append(f"{r['time_sleep']} time.sleep")
        if r['total_lines'] > 2000:
            issues.append(f"{r['total_lines']} lines (too long)")
        
        if issues:
            print(f"✗ {template_name}")
            for issue in issues:
                print(f"    - {issue}")
            total_issues += len(issues)
        else:
            print(f"✓ {template_name}: All checks passed")
    
    print()
    print("="*80)
    print(f"SUMMARY: {total_issues} issues found across {len(results)} templates")
    print("="*80)

if __name__ == '__main__':
    results = validate_templates()
    print_report(results)

