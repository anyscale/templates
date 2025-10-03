"""Visualization utilities for unstructured data ingestion templates."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def visualize_document_processing():
    """Create concise document processing analytics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Document types processed
    doc_types = ['Financial', 'Legal', 'Regulatory', 'Client', 'Research', 'General']
    counts = [145, 89, 67, 123, 98, 201]
    colors = ['green', 'blue', 'purple', 'orange', 'red', 'gray']
    
    bars = axes[0].bar(doc_types, counts, color=colors, alpha=0.7)
    axes[0].set_title('Documents by Type', fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    str(cnt), ha='center', fontsize=9, fontweight='bold')
    
    # 2. Quality distribution
    quality_levels = ['High', 'Medium', 'Low']
    quality_counts = [482, 168, 73]
    colors_qual = ['darkgreen', 'orange', 'red']
    
    wedges, texts, autotexts = axes[1].pie(quality_counts, labels=quality_levels,
                                           colors=colors_qual, autopct='%1.1f%%',
                                           startangle=90)
    axes[1].set_title('Document Quality Distribution', fontweight='bold')
    
    # 3. Processing pipeline stages
    stages = ['Discovered', 'Extracted', 'Chunked', 'Analyzed', 'Warehouse\nReady']
    stage_counts = [723, 695, 682, 671, 643]
    
    axes[2].plot(stages, stage_counts, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[2].fill_between(range(len(stages)), stage_counts, alpha=0.3)
    axes[2].set_title('Pipeline Stage Funnel', fontweight='bold')
    axes[2].set_ylabel('Document Count')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    for i, (stage, cnt) in enumerate(zip(stages, stage_counts)):
        axes[2].text(i, cnt + 10, str(cnt), ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('document_processing.png', dpi=150, bbox_inches='tight')
    print("Document processing visualization saved")
    
    return fig


def create_document_type_sunburst(doc_df):
    """Create interactive sunburst chart for document hierarchy."""
    import plotly.express as px
    
    fig = px.sunburst(
        doc_df,
        path=['business_category', 'document_type', 'quality_rating'],
        title='Document Hierarchy and Quality Distribution',
        height=600
    )
    
    return fig


def create_processing_funnel(doc_df):
    """Create processing funnel visualization."""
    import plotly.graph_objects as go
    
    stages = ['Discovered', 'Quality Passed', 'Text Extracted', 'LLM Analyzed', 'Warehouse Ready']
    counts = [1000, 850, 780, 720, 680]
    
    fig = go.Figure(go.Funnel(
        y=stages,
        x=counts,
        textinfo="value+percent initial",
        marker={"color": ["lightblue", "lightgreen", "lightyellow", "lightcoral", "darkgreen"]}
    ))
    
    fig.update_layout(
        title='Document Processing Funnel',
        height=500
    )
    
    return fig

