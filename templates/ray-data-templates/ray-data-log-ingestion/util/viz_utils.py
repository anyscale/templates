"""Visualization utilities for log analytics templates."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_log_analytics_dashboard(log_df):
    """Create interactive log analytics dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Log Level Distribution', 'Logs Over Time', 
                       'Top Error Sources', 'Response Time Distribution'),
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # Log level distribution
    if 'level' in log_df.columns:
        level_counts = log_df['level'].value_counts()
        fig.add_trace(
            go.Pie(labels=level_counts.index, values=level_counts.values,
                  name="Log Level"),
            row=1, col=1
        )
    
    # Logs over time
    if 'timestamp' in log_df.columns:
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
        logs_over_time = log_df.groupby(log_df['timestamp'].dt.hour).size().reset_index()
        fig.add_trace(
            go.Scatter(x=logs_over_time['timestamp'], y=logs_over_time[0],
                      mode='lines+markers', name="Log Volume"),
            row=1, col=2
        )
    
    # Top error sources
    if 'source' in log_df.columns:
        error_logs = log_df[log_df['level'] == 'ERROR']
        if len(error_logs) > 0:
            source_counts = error_logs['source'].value_counts().head(10).reset_index()
            fig.add_trace(
                go.Bar(x=source_counts['source'], y=source_counts['count'],
                      marker_color='red', name="Errors"),
                row=2, col=1
            )
    
    # Response time distribution
    if 'response_time' in log_df.columns:
        fig.add_trace(
            go.Histogram(x=log_df['response_time'], nbinsx=30,
                        marker_color='orange', name="Response Time"),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False,
                     title_text="Log Analytics Dashboard")
    
    return fig


def create_security_dashboard(security_df):
    """Create security monitoring dashboard."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Threat Distribution', 'Attack Sources', 'Severity Levels'),
        specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # Threat type distribution
    if 'threat_type' in security_df.columns:
        fig.add_trace(
            go.Pie(labels=security_df['threat_type'].value_counts().index,
                  values=security_df['threat_type'].value_counts().values,
                  name="Threats"),
            row=1, col=1
        )
    
    # Attack sources
    if 'source_ip' in security_df.columns:
        source_counts = security_df['source_ip'].value_counts().head(10).reset_index()
        fig.add_trace(
            go.Bar(x=source_counts['source_ip'], y=source_counts['count'],
                  marker_color='red', name="Sources"),
            row=1, col=2
        )
    
    # Severity levels
    if 'severity' in security_df.columns:
        severity_counts = security_df['severity'].value_counts().reset_index()
        fig.add_trace(
            go.Bar(x=severity_counts['severity'], y=severity_counts['count'],
                  marker_color=['green', 'yellow', 'orange', 'red'],
                  name="Severity"),
            row=1, col=3
        )
    
    fig.update_layout(height=500, title_text="Security Monitoring Dashboard")
    
    return fig


def create_interactive_log_dashboard(log_data):
    """Create comprehensive interactive log analytics dashboard."""
    import pandas as pd
    
    df = log_data.to_pandas() if hasattr(log_data, 'to_pandas') else pd.DataFrame(log_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Log Level Distribution', 'Error Trends', 'Top Error Sources', 'Response Times'),
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # Log levels
    if 'level' in df.columns:
        level_counts = df['level'].value_counts()
        fig.add_trace(go.Pie(labels=level_counts.index, values=level_counts.values), row=1, col=1)
    
    # Error trends over time
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        errors_over_time = df[df['level'] == 'ERROR'].groupby(df['timestamp'].dt.hour).size()
        fig.add_trace(go.Scatter(x=errors_over_time.index, y=errors_over_time.values, mode='lines+markers'), row=1, col=2)
    
    # Top error sources
    if 'source' in df.columns:
        error_sources = df[df['level'] == 'ERROR']['source'].value_counts().head(10)
        fig.add_trace(go.Bar(x=error_sources.index, y=error_sources.values, marker_color='red'), row=2, col=1)
    
    # Response time distribution
    if 'response_time' in df.columns:
        fig.add_trace(go.Histogram(x=df['response_time'], nbinsx=30, marker_color='orange'), row=2, col=2)
    
    fig.update_layout(height=800, title_text="Log Analytics Dashboard", showlegend=False)
    return fig


def create_network_security_visualization():
    """Create network security monitoring visualization."""
    import plotly.graph_objects as go
    import numpy as np
    
    # Sample security data for demonstration
    threat_types = ['SQL Injection', 'XSS', 'DDoS', 'Brute Force', 'Malware']
    counts = [145, 89, 234, 167, 56]
    
    fig = go.Figure(data=[
        go.Bar(x=threat_types, y=counts, marker_color='red', opacity=0.7)
    ])
    
    fig.update_layout(
        title='Network Security Threats Detected',
        xaxis_title='Threat Type',
        yaxis_title='Count',
        height=500
    )
    
    return fig

