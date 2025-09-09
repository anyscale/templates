#!/usr/bin/env python3
"""
Template Standardization Script

This script applies consistent standardization to all Ray Data templates
to ensure they follow the established style guide and patterns.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any


class TemplateStandardizer:
    """Standardizes Ray Data templates according to style guide."""
    
    def __init__(self):
        """Initialize the standardizer."""
        self.templates_dir = Path("/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates")
        self.style_violations = []
        self.fixes_applied = []
    
    def find_ray_data_templates(self) -> List[Path]:
        """Find all Ray Data templates that need standardization."""
        templates = []
        for template_dir in self.templates_dir.glob("ray-data-*"):
            if template_dir.is_dir():
                notebook_path = template_dir / "README.ipynb"
                if notebook_path.exists():
                    # Check size to identify templates that likely need work
                    size = notebook_path.stat().st_size
                    if size > 20000:  # Templates over 20KB likely need standardization
                        templates.append(template_dir)
        return templates
    
    def analyze_notebook_structure(self, notebook_path: Path) -> Dict[str, Any]:
        """Analyze notebook structure and identify issues."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
        except Exception as e:
            return {"error": f"Failed to read notebook: {e}"}
        
        analysis = {
            "path": str(notebook_path),
            "total_cells": len(notebook.get("cells", [])),
            "code_cells": 0,
            "markdown_cells": 0,
            "has_time_estimate": False,
            "has_learning_objectives": False,
            "has_overview": False,
            "has_cleanup": False,
            "excessive_business_content": False,
            "complex_visualizations": False,
            "issues": []
        }
        
        cells = notebook.get("cells", [])
        first_cell_content = ""
        
        for i, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                analysis["code_cells"] += 1
            elif cell.get("cell_type") == "markdown":
                analysis["markdown_cells"] += 1
                
                # Check first cell for standard elements
                if i == 0:
                    first_cell_content = "".join(cell.get("source", []))
                    
                    if "Time to complete" in first_cell_content:
                        analysis["has_time_estimate"] = True
                        # Check if time is excessive (>30 min)
                        time_match = re.search(r"(\d+)\s*min", first_cell_content)
                        if time_match and int(time_match.group(1)) > 30:
                            analysis["issues"].append("Excessive time estimate (>30 min)")
                    
                    if "Learning Objectives" in first_cell_content:
                        analysis["has_learning_objectives"] = True
                    
                    if "Overview" in first_cell_content:
                        analysis["has_overview"] = True
                
                # Check for excessive business content
                cell_content = "".join(cell.get("source", []))
                if any(phrase in cell_content for phrase in ["$", "billion", "trillion", "market", "revenue", "ROI"]):
                    if len(cell_content) > 1000:  # Large business content blocks
                        analysis["excessive_business_content"] = True
                        analysis["issues"].append("Excessive business context")
                
                # Check for complex visualizations
                if any(lib in cell_content for lib in ["plotly", "folium", "bokeh", "dash"]):
                    analysis["complex_visualizations"] = True
                    analysis["issues"].append("Complex visualization libraries")
            
            # Check for cleanup
            if cell.get("cell_type") == "code":
                cell_content = "".join(cell.get("source", []))
                if "ray.shutdown()" in cell_content:
                    analysis["has_cleanup"] = True
        
        # Add missing elements to issues
        if not analysis["has_time_estimate"]:
            analysis["issues"].append("Missing time estimate")
        if not analysis["has_learning_objectives"]:
            analysis["issues"].append("Missing learning objectives")
        if not analysis["has_cleanup"]:
            analysis["issues"].append("Missing Ray cleanup")
        
        return analysis
    
    def generate_standardization_report(self) -> str:
        """Generate a comprehensive standardization report."""
        templates = self.find_ray_data_templates()
        
        report = []
        report.append("# Ray Data Template Standardization Report\n")
        report.append(f"Analyzed {len(templates)} templates requiring standardization.\n")
        
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for template_dir in templates:
            notebook_path = template_dir / "README.ipynb"
            analysis = self.analyze_notebook_structure(notebook_path)
            
            if "error" in analysis:
                continue
            
            issue_count = len(analysis["issues"])
            size_kb = notebook_path.stat().st_size / 1024
            
            template_info = {
                "name": template_dir.name,
                "size_kb": size_kb,
                "issues": analysis["issues"],
                "cells": analysis["total_cells"]
            }
            
            if issue_count >= 3 or size_kb > 35:
                high_priority.append(template_info)
            elif issue_count >= 2 or size_kb > 25:
                medium_priority.append(template_info)
            else:
                low_priority.append(template_info)
        
        # Generate report sections
        report.append("## High Priority Templates (Immediate Standardization Required)\n")
        for template in high_priority:
            report.append(f"### {template['name']}")
            report.append(f"- Size: {template['size_kb']:.1f}KB ({template['cells']} cells)")
            report.append(f"- Issues: {', '.join(template['issues'])}")
            report.append("")
        
        report.append("## Medium Priority Templates\n")
        for template in medium_priority:
            report.append(f"### {template['name']}")
            report.append(f"- Size: {template['size_kb']:.1f}KB ({template['cells']} cells)")
            report.append(f"- Issues: {', '.join(template['issues'])}")
            report.append("")
        
        report.append("## Low Priority Templates\n")
        for template in low_priority:
            report.append(f"### {template['name']}")
            report.append(f"- Size: {template['size_kb']:.1f}KB ({template['cells']} cells)")
            if template['issues']:
                report.append(f"- Issues: {', '.join(template['issues'])}")
            else:
                report.append("- Status: Compliant")
            report.append("")
        
        return "\n".join(report)
    
    def create_template_fixes(self, template_name: str) -> List[str]:
        """Generate specific fixes needed for a template."""
        fixes = []
        
        # Standard fixes for all oversized templates
        if template_name in ["ray-data-multimodal-ai-pipeline", "ray-data-financial-forecasting", 
                           "ray-data-enterprise-data-catalog", "ray-data-data-quality-monitoring"]:
            fixes.extend([
                "Remove excessive business context and market analysis",
                "Simplify time estimate to 25-30 minutes",
                "Focus learning objectives on Ray Data capabilities",
                "Remove complex visualization libraries (Plotly, Folium)",
                "Simplify to 15-20 cells maximum",
                "Add proper predictor class following established patterns",
                "Include basic aggregation examples using groupby",
                "Add simple filtering and map_batches examples",
                "Include proper error handling and resource cleanup",
                "Create working demo script",
                "Add requirements.txt with minimal dependencies",
                "Create README.md documentation",
                "Add troubleshooting and performance tips sections",
                "Focus on educational value over business marketing"
            ])
        
        return fixes


def main():
    """Main standardization analysis function."""
    print("=== Ray Data Template Standardization Analysis ===\n")
    
    standardizer = TemplateStandardizer()
    
    # Generate comprehensive report
    report = standardizer.generate_standardization_report()
    
    # Save report
    report_path = "/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/templates/templates/TEMPLATE_STANDARDIZATION_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Standardization report saved to: {report_path}")
    print("\n" + "="*60)
    print(report)


if __name__ == "__main__":
    main()
