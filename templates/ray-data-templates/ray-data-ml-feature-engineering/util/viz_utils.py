"""Visualization utilities for ML feature engineering templates."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def visualize_feature_engineering():
    """Create concise feature engineering analytics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Feature importance scores
    features = ['Family_Size', 'Fare_pp', 'Title_Score', 'Age_Group', 'Deck', 'Embarked']
    importance = [0.85, 0.72, 0.68, 0.55, 0.48, 0.35]
    colors_imp = ['darkgreen' if i > 0.7 else 'green' if i > 0.5 else 'orange' for i in importance]
    
    bars = axes[0].barh(features, importance, color=colors_imp, alpha=0.7)
    axes[0].set_title('Feature Importance Scores', fontweight='bold')
    axes[0].set_xlabel('Importance Score')
    axes[0].set_xlim(0, 1.0)
    axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    
    for bar, imp in zip(bars, importance):
        axes[0].text(imp + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{imp:.2f}', va='center', fontweight='bold')
    
    # 2. Feature type distribution
    feature_types = ['Numerical', 'Categorical', 'Temporal', 'Interaction', 'Derived']
    type_counts = [12, 15, 8, 22, 18]
    colors_type = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum']
    
    wedges, texts, autotexts = axes[1].pie(type_counts, labels=feature_types,
                                           colors=colors_type, autopct='%1.0f%%',
                                           startangle=90)
    axes[1].set_title('Feature Type Distribution', fontweight='bold')
    
    # 3. Feature engineering pipeline
    stages = ['Raw\nData', 'Basic\nFeatures', 'Derived\nFeatures', 'Selected\nFeatures', 'Final\nSet']
    stage_counts = [12, 28, 75, 35, 25]
    
    axes[2].plot(stages, stage_counts, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[2].fill_between(range(len(stages)), stage_counts, alpha=0.3)
    axes[2].set_title('Feature Pipeline Progression', fontweight='bold')
    axes[2].set_ylabel('Feature Count')
    axes[2].grid(True, alpha=0.3)
    
    for i, (stage, cnt) in enumerate(zip(stages, stage_counts)):
        axes[2].text(i, cnt + 2, str(cnt), ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_engineering.png', dpi=150, bbox_inches='tight')
    print("Feature engineering visualization saved")
    
    return fig


def create_feature_correlation_heatmap(feature_df):
    """Create interactive feature correlation heatmap."""
    import plotly.express as px
    
    # Calculate correlation matrix for numeric features
    numeric_features = feature_df.select_dtypes(include=['number'])
    corr_matrix = numeric_features.corr()
    
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdBu',
                    aspect='auto')
    
    fig.update_layout(height=600)
    return fig


def create_feature_importance_chart(importance_scores):
    """Create interactive feature importance chart."""
    import plotly.graph_objects as go
    
    # Sort by importance
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    features, scores = zip(*sorted_features[:15])
    
    fig = go.Figure(data=[
        go.Bar(x=list(scores), y=list(features), orientation='h',
              marker_color=list(scores), marker_colorscale='Viridis')
    ])
    
    fig.update_layout(
        title='Top 15 Feature Importance Scores',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=500
    )
    
    return fig


def create_feature_distribution_dashboard(feature_df):
    """Create comprehensive feature distribution dashboard."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Select numeric features
    numeric_cols = feature_df.select_dtypes(include=['number']).columns[:6]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'{col} Distribution' for col in numeric_cols]
    )
    
    for idx, col in enumerate(numeric_cols):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1
        
        fig.add_trace(
            go.Histogram(x=feature_df[col], nbinsx=20,
                        marker_color='lightblue', name=col),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=600, showlegend=False,
                     title_text="Feature Distribution Analysis")
    
    return fig

