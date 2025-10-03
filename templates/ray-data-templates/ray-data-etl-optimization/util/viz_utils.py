"""Visualization utilities for ETL optimization templates."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def visualize_etl_performance():
    """Visualize ETL pipeline performance metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Processing throughput
    stages = ['Extract', 'Transform', 'Load']
    throughput = [15000, 12000, 18000]  # Records per second
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    bars = axes[0].bar(stages, throughput, color=colors, alpha=0.7)
    axes[0].set_title('ETL Stage Throughput', fontweight='bold')
    axes[0].set_ylabel('Records/Second')
    
    for bar, val in zip(bars, throughput):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{val:,}', ha='center', fontweight='bold')
    
    # 2. Memory usage
    operations = ['Read', 'Transform', 'Aggregate', 'Write']
    memory = [2.5, 4.2, 3.8, 1.9]  # GB
    
    axes[1].barh(operations, memory, color='orange', alpha=0.7)
    axes[1].set_title('Memory Usage by Operation', fontweight='bold')
    axes[1].set_xlabel('Memory (GB)')
    
    # 3. Optimization impact
    configs = ['Baseline', 'Optimized\nBatch Size', 'Optimized\nConcurrency', 'Fully\nOptimized']
    speedup = [1.0, 2.3, 3.1, 4.2]
    
    axes[2].plot(configs, speedup, 'o-', linewidth=2, markersize=10, color='green')
    axes[2].set_title('Optimization Impact', fontweight='bold')
    axes[2].set_ylabel('Speedup Factor')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('etl_performance.png', dpi=150, bbox_inches='tight')
    print("ETL performance visualization saved")
    
    return fig


def create_interactive_etl_pipeline(etl_results):
    """Create interactive ETL pipeline visualization with plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    
    df = pd.DataFrame(etl_results) if isinstance(etl_results, list) else etl_results
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Data Volume by Stage', 'Processing Time', 
                       'Resource Utilization', 'Data Quality Score'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # Stage volumes
    stages = ['Extract', 'Transform', 'Filter', 'Aggregate', 'Load']
    volumes = [1500000, 1450000, 980000, 125000, 125000]
    fig.add_trace(
        go.Bar(x=stages, y=volumes, marker_color='lightblue'),
        row=1, col=1
    )
    
    # Processing time
    fig.add_trace(
        go.Scatter(x=stages, y=[12, 45, 8, 15, 10], mode='lines+markers',
                  line=dict(color='green', width=3)),
        row=1, col=2
    )
    
    # Resource utilization
    resources = ['CPU', 'Memory', 'I/O']
    utilization = [78, 65, 82]
    fig.add_trace(
        go.Bar(x=resources, y=utilization, marker_color=['blue', 'orange', 'purple']),
        row=2, col=1
    )
    
    # Quality score indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=92,
            title={'text': "ETL Quality Score"},
            delta={'reference': 85},
            gauge={'axis': {'range': [None, 100]},
                  'bar': {'color': "darkgreen"},
                  'steps': [
                      {'range': [0, 70], 'color': "lightcoral"},
                      {'range': [70, 85], 'color': "lightyellow"},
                      {'range': [85, 100], 'color': "lightgreen"}]},
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=False,
                     title_text="ETL Pipeline Performance Dashboard")
    
    return fig


def create_data_lineage_diagram():
    """Create interactive data lineage visualization."""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Source DB", "Raw Data Lake", "ETL Process", 
                   "Validated Data", "Data Warehouse", "BI Tools"],
            color=["blue", "lightblue", "green", "lightgreen", "orange", "red"]
        ),
        link=dict(
            source=[0, 1, 2, 2, 3, 4],
            target=[1, 2, 3, 4, 4, 5],
            value=[1500, 1500, 1200, 1450, 1200, 1200]
        )
    )])
    
    fig.update_layout(title_text="ETL Data Lineage Flow", height=500, font_size=12)
    return fig


def create_tpch_schema_diagram():
    """Create TPC-H schema relationship visualization."""
    import plotly.graph_objects as go
    
    # Define TPC-H tables and relationships
    fig = go.Figure()
    
    # Add nodes for each table
    tables = {
        'CUSTOMER': (1, 3),
        'ORDERS': (2, 3),
        'LINEITEM': (3, 3),
        'PART': (3, 2),
        'SUPPLIER': (2, 1),
        'PARTSUPP': (3, 1),
        'NATION': (1, 1),
        'REGION': (0, 1)
    }
    
    for table, (x, y) in tables.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=40, color='lightblue'),
            text=[table],
            textposition="middle center",
            name=table
        ))
    
    fig.update_layout(
        title='TPC-H Schema Relationships',
        height=600,
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    
    return fig

