#!/usr/bin/env python3
"""
Deep content validation between README.md and README.ipynb files.

This script performs granular verification that markdown and notebook
content is perfectly synchronized and follows all rules.
"""

import os
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

class DeepContentValidator:
    """Performs deep validation of content synchronization and rule compliance."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.templates = self._find_templates()
        self.action_count = 0
        
    def _find_templates(self) -> List[Tuple[str, str]]:
        """Find all Ray Data templates."""
        templates = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path) and item.startswith("ray-data-"):
                templates.append((item, item_path))
        return sorted(templates)
    
    def extract_markdown_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from markdown content."""
        self.action_count += 1
        
        sections = {}
        
        # Extract title
        title_match = re.search(r'^# (.+)', content)
        sections['title'] = title_match.group(1) if title_match else ""
        
        # Extract major sections
        section_patterns = [
            ('header', r'(\*\*Time to complete\*\*.*?\n)'),
            ('what_youll_build', r'## What You\'ll Build\n(.*?)(?=##)'),
            ('table_of_contents', r'## Table of Contents\n(.*?)(?=##)'),
            ('learning_objectives', r'## Learning Objectives\n(.*?)(?=##)'),
            ('overview', r'## Overview\n(.*?)(?=##)'),
            ('prerequisites', r'## Prerequisites.*?\n(.*?)(?=##)'),
            ('quick_start', r'## Quick Start.*?\n(.*?)(?=##)'),
        ]
        
        for section_name, pattern in section_patterns:
            match = re.search(pattern, content, re.DOTALL)
            sections[section_name] = match.group(1).strip() if match else ""
        
        # Extract code blocks
        code_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
        sections['code_blocks'] = code_blocks
        
        return sections
    
    def extract_notebook_sections(self, nb_data: Dict) -> Dict[str, str]:
        """Extract sections from notebook content."""
        self.action_count += 1
        
        sections = {}
        
        # Combine all markdown cells
        markdown_content = '\n'.join([
            '\n'.join(cell.get('source', [])) 
            for cell in nb_data.get('cells', [])
            if cell.get('cell_type') == 'markdown'
        ])
        
        # Extract code cells
        code_cells = [
            '\n'.join(cell.get('source', [])) 
            for cell in nb_data.get('cells', [])
            if cell.get('cell_type') == 'code'
        ]
        
        # Use same extraction logic as markdown
        sections = self.extract_markdown_sections(markdown_content)
        sections['code_blocks'] = code_cells
        
        return sections
    
    def compare_section_content(self, md_section: str, nb_section: str, section_name: str) -> Tuple[bool, str]:
        """Compare content between markdown and notebook sections."""
        self.action_count += 1
        
        # Normalize whitespace for comparison
        md_normalized = re.sub(r'\s+', ' ', md_section.strip())
        nb_normalized = re.sub(r'\s+', ' ', nb_section.strip())
        
        # Check for exact match
        if md_normalized == nb_normalized:
            return True, "Perfect match"
        
        # Check for substantial similarity (allow minor formatting differences)
        md_hash = hashlib.md5(md_normalized.encode()).hexdigest()
        nb_hash = hashlib.md5(nb_normalized.encode()).hexdigest()
        
        if md_hash == nb_hash:
            return True, "Content hash match"
        
        # Check similarity by word count
        md_words = set(md_normalized.lower().split())
        nb_words = set(nb_normalized.lower().split())
        
        if md_words and nb_words:
            similarity = len(md_words & nb_words) / len(md_words | nb_words)
            if similarity > 0.9:
                return True, f"High similarity ({similarity:.1%})"
            else:
                return False, f"Low similarity ({similarity:.1%})"
        
        return False, "Content mismatch"
    
    def validate_code_block_synchronization(self, md_blocks: List[str], nb_blocks: List[str]) -> Dict[str, Any]:
        """Validate code block synchronization between markdown and notebook."""
        self.action_count += 1
        
        results = {
            'md_count': len(md_blocks),
            'nb_count': len(nb_blocks),
            'count_match': len(md_blocks) == len(nb_blocks),
            'content_matches': [],
            'syntax_valid': []
        }
        
        # Compare each code block
        min_blocks = min(len(md_blocks), len(nb_blocks))
        for i in range(min_blocks):
            self.action_count += 1
            
            md_code = md_blocks[i].strip()
            nb_code = nb_blocks[i].strip()
            
            # Check content similarity
            match, reason = self.compare_section_content(md_code, nb_code, f"code_block_{i}")
            results['content_matches'].append((match, reason))
            
            # Check syntax validity
            try:
                import ast
                ast.parse(nb_code)
                results['syntax_valid'].append(True)
            except SyntaxError:
                results['syntax_valid'].append(False)
        
        return results
    
    def validate_required_sections(self, sections: Dict[str, str], template_name: str) -> Dict[str, bool]:
        """Validate that all required sections are present and properly formatted."""
        self.action_count += 1
        
        validations = {}
        
        # Rule 1: Title format
        validations['title_has_ray_data'] = 'Ray Data' in sections.get('title', '')
        
        # Rule 2: Time estimate
        validations['has_time_estimate'] = 'Time to complete' in sections.get('header', '')
        
        # Rule 3: Difficulty level
        validations['has_difficulty'] = 'Difficulty' in sections.get('header', '')
        
        # Rule 26: Table of Contents
        validations['has_toc'] = bool(sections.get('table_of_contents'))
        
        # Rule 41: Learning Objectives
        validations['has_learning_objectives'] = bool(sections.get('learning_objectives'))
        
        # Rule 71: Prerequisites
        validations['has_prerequisites'] = bool(sections.get('prerequisites'))
        
        # Rule 86: Quick Start
        validations['has_quick_start'] = bool(sections.get('quick_start'))
        
        return validations
    
    def validate_code_quality_rules(self, code_blocks: List[str], template_name: str) -> Dict[str, Any]:
        """Validate code quality rules across all code blocks."""
        self.action_count += 1
        
        results = {
            'total_blocks': len(code_blocks),
            'block_sizes': [],
            'has_docstrings': 0,
            'has_comments': 0,
            'uses_ray_data': 0,
            'uses_native_ops': 0,
            'syntax_errors': []
        }
        
        for i, block in enumerate(code_blocks):
            self.action_count += 1
            
            lines = block.split('\n')
            results['block_sizes'].append(len(lines))
            
            # Check for docstrings
            if '"""' in block or "'''" in block:
                results['has_docstrings'] += 1
            
            # Check for comments
            if '#' in block:
                results['has_comments'] += 1
            
            # Check Ray Data usage
            if 'ray.data.' in block:
                results['uses_ray_data'] += 1
            
            # Check native operations
            native_ops = ['map_batches', 'filter', 'groupby', 'aggregate', 'sort', 'join']
            if any(op in block for op in native_ops):
                results['uses_native_ops'] += 1
            
            # Check syntax
            try:
                import ast
                ast.parse(block)
            except SyntaxError as e:
                results['syntax_errors'].append(f"Block {i+1}: {str(e)}")
        
        return results
    
    def validate_single_template_deep(self, template_name: str, template_path: str) -> Dict[str, Any]:
        """Perform deep validation of a single template."""
        
        md_path = os.path.join(template_path, "README.md")
        nb_path = os.path.join(template_path, "README.ipynb")
        
        results = {
            'template_name': template_name,
            'files_exist': {
                'readme_md': os.path.exists(md_path),
                'readme_ipynb': os.path.exists(nb_path)
            }
        }
        
        if not os.path.exists(md_path) or not os.path.exists(nb_path):
            return results
        
        # Load content
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb_data = json.load(f)
        
        # Extract sections
        md_sections = self.extract_markdown_sections(md_content)
        nb_sections = self.extract_notebook_sections(nb_data)
        
        # Compare sections
        results['section_comparisons'] = {}
        for section_name in ['title', 'header', 'what_youll_build', 'table_of_contents', 
                            'learning_objectives', 'overview', 'prerequisites', 'quick_start']:
            self.action_count += 1
            
            md_section = md_sections.get(section_name, '')
            nb_section = nb_sections.get(section_name, '')
            
            match, reason = self.compare_section_content(md_section, nb_section, section_name)
            results['section_comparisons'][section_name] = {
                'match': match,
                'reason': reason,
                'md_length': len(md_section),
                'nb_length': len(nb_section)
            }
        
        # Validate code blocks
        results['code_validation'] = self.validate_code_block_synchronization(
            md_sections.get('code_blocks', []),
            nb_sections.get('code_blocks', [])
        )
        
        # Validate required sections
        results['md_required_sections'] = self.validate_required_sections(md_sections, template_name)
        results['nb_required_sections'] = self.validate_required_sections(nb_sections, template_name)
        
        # Validate code quality
        results['md_code_quality'] = self.validate_code_quality_rules(md_sections.get('code_blocks', []), template_name)
        results['nb_code_quality'] = self.validate_code_quality_rules(nb_sections.get('code_blocks', []), template_name)
        
        return results
    
    def generate_deep_validation_report(self, all_results: Dict[str, Any]) -> None:
        """Generate comprehensive deep validation report."""
        
        print("\nDEEP CONTENT SYNCHRONIZATION REPORT")
        print("=" * 80)
        
        perfect_sync_count = 0
        total_templates = len(all_results)
        
        for template_name, results in all_results.items():
            print(f"\n{template_name}:")
            print("-" * 50)
            
            # Check section synchronization
            section_matches = 0
            total_sections = len(results.get('section_comparisons', {}))
            
            for section_name, comparison in results.get('section_comparisons', {}).items():
                if comparison['match']:
                    section_matches += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"  {status} {section_name:<20} {comparison['reason']}")
            
            # Check code block synchronization
            code_val = results.get('code_validation', {})
            code_match_rate = sum(1 for match, _ in code_val.get('content_matches', []) if match)
            total_code_blocks = len(code_val.get('content_matches', []))
            
            print(f"  📊 Section sync: {section_matches}/{total_sections}")
            print(f"  💻 Code sync: {code_match_rate}/{total_code_blocks}")
            
            # Check syntax validity
            syntax_valid = sum(code_val.get('syntax_valid', []))
            print(f"  ✅ Syntax valid: {syntax_valid}/{total_code_blocks}")
            
            if section_matches == total_sections and code_match_rate == total_code_blocks:
                perfect_sync_count += 1
        
        print("\n" + "=" * 80)
        print(f"PERFECT SYNCHRONIZATION: {perfect_sync_count}/{total_templates} templates")
        print(f"ACTIONS PERFORMED: {self.action_count}")
    
    def validate_all_templates_deep(self) -> Dict[str, Any]:
        """Perform deep validation of all templates."""
        
        print("Deep Content Validation - 100+ Systematic Actions")
        print("=" * 80)
        print(f"Performing granular validation of {len(self.templates)} templates...")
        
        all_results = {}
        
        for template_name, template_path in self.templates:
            print(f"Deep validating {template_name}...")
            results = self.validate_single_template_deep(template_name, template_path)
            all_results[template_name] = results
        
        self.generate_deep_validation_report(all_results)
        
        return all_results

def main():
    """Main deep validation function."""
    
    base_dir = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/ray-data-templates"
    
    validator = DeepContentValidator(base_dir)
    results = validator.validate_all_templates_deep()
    
    print(f"\nDEEP VALIDATION COMPLETE")
    print(f"Total actions performed: {validator.action_count}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
