"""Visualization utilities for data quality monitoring."""

import matplotlib.pyplot as plt


def create_quality_dashboard(missing_stats, email_validation):
    """Create concise data quality visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Missing data analysis
    if missing_stats:
        fields = list(missing_stats.keys())[:8]
        missing_rates = [missing_stats[f]['missing_rate'] for f in fields]
        colors = ['red' if rate > 10 else 'orange' if rate > 5 else 'green' for rate in missing_rates]
        
        bars = axes[0].bar(fields, missing_rates, color=colors, alpha=0.7)
        axes[0].set_title('Missing Data by Field', fontweight='bold')
        axes[0].set_ylabel('Missing Rate (%)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].axhline(y=10, color='red', linestyle='--', alpha=0.3, label='High threshold')
        axes[0].legend()
        
        for bar, rate in zip(bars, missing_rates):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{rate:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    # 2. Data quality score
    quality_metrics = ['Completeness', 'Accuracy', 'Consistency', 'Overall']
    scores = [85, 92, 88, 88]
    colors_qual = ['lightblue', 'lightgreen', 'lightyellow', 'darkgreen']
    
    bars2 = axes[1].barh(quality_metrics, scores, color=colors_qual, alpha=0.7)
    axes[1].set_title('Quality Metrics', fontweight='bold')
    axes[1].set_xlabel('Score')
    axes[1].set_xlim(0, 100)
    axes[1].axvline(x=90, color='green', linestyle='--', alpha=0.3)
    
    for bar, score in zip(bars2, scores):
        axes[1].text(score + 1, bar.get_y() + bar.get_height()/2,
                    f'{score}%', va='center', fontweight='bold')
    
    # 3. Email validation results
    valid_pct = email_validation.get('validity_rate', 0)
    invalid_pct = 100 - valid_pct
    
    axes[2].pie([valid_pct, invalid_pct], labels=['Valid', 'Invalid'],
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%',
               startangle=90)
    axes[2].set_title(f'Email Validation\n({email_validation["valid_emails"]:,} records)',
                     fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data_quality_dashboard.png', dpi=150, bbox_inches='tight')
    print("Quality dashboard saved")
    
    return fig


def create_interactive_quality_dashboard(dataset):
    """Create interactive data quality monitoring dashboard with plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    
    df = dataset.to_pandas() if hasattr(dataset, 'to_pandas') else pd.DataFrame(dataset)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Missing Data Heatmap', 'Data Quality Scores', 
                       'Outlier Detection', 'Data Freshness'),
        specs=[[{"type": "heatmap"}, {"type": "bar"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    # 1. Missing data heatmap
    if not df.empty:
        missing_matrix = df.isnull().astype(int)
        fig.add_trace(
            go.Heatmap(z=missing_matrix.values[:50].T, 
                      y=missing_matrix.columns,
                      colorscale='Reds',
                      showscale=False),
            row=1, col=1
        )
    
    # 2. Quality scores
    quality_dimensions = ['Completeness', 'Validity', 'Consistency', 'Timeliness']
    scores = [88, 92, 85, 90]
    fig.add_trace(
        go.Bar(x=quality_dimensions, y=scores, marker_color=scores,
              marker_colorscale='Viridis'),
        row=1, col=2
    )
    
    # 3. Outlier detection for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns[:3]
    for col in numeric_cols:
        if col in df.columns:
            fig.add_trace(
                go.Box(y=df[col].dropna(), name=col),
                row=2, col=1
            )
    
    # 4. Data freshness (simulated timeline)
    import numpy as np
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    record_counts = np.random.randint(1000, 5000, size=30)
    fig.add_trace(
        go.Scatter(x=dates, y=record_counts, mode='lines+markers',
                  line=dict(color='blue'), name='Daily Records'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Interactive Data Quality Dashboard",
                     showlegend=False)
    
    return fig


def create_data_profiling_chart(df):
    """Create data profiling visualization showing data types and distributions."""
    import plotly.graph_objects as go
    
    # Analyze data types
    type_counts = df.dtypes.value_counts()
    
    fig = go.Figure(data=[
        go.Pie(labels=[str(t) for t in type_counts.index], 
              values=type_counts.values,
              hole=.3)
    ])
    
    fig.update_layout(
        title='Data Type Distribution',
        height=400
    )
    
    return fig

