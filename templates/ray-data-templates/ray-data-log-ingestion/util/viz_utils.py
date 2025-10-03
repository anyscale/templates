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


def create_traffic_heatmap(log_df):
    """Create hourly traffic heatmap visualization."""
    import plotly.express as px
    import pandas as pd
    
    # Create hourly heatmap
    if 'timestamp' in log_df.columns:
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
        log_df['hour'] = log_df['timestamp'].dt.hour
        log_df['day'] = log_df['timestamp'].dt.day_name()
        
        heatmap_data = log_df.groupby(['day', 'hour']).size().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='day', columns='hour', values=0)
        
        fig = px.imshow(heatmap_pivot,
                       labels=dict(x="Hour of Day", y="Day of Week", color="Request Count"),
                       title="Traffic Heatmap (Requests by Hour and Day)",
                       color_continuous_scale='Blues')
        
        fig.update_layout(height=500)
        return fig
    return None


def create_error_rate_timeline(log_df):
    """Create error rate timeline visualization."""
    import plotly.graph_objects as go
    import pandas as pd
    
    if 'timestamp' in log_df.columns and 'level' in log_df.columns:
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
        
        # Calculate hourly error rates
        hourly_total = log_df.groupby(log_df['timestamp'].dt.hour).size()
        hourly_errors = log_df[log_df['level'] == 'ERROR'].groupby(
            log_df[log_df['level'] == 'ERROR']['timestamp'].dt.hour
        ).size()
        
        error_rate = (hourly_errors / hourly_total * 100).fillna(0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=error_rate.index,
            y=error_rate.values,
            mode='lines+markers',
            line=dict(color='red', width=2),
            fill='tozeroy',
            name='Error Rate'
        ))
        
        fig.update_layout(
            title='Hourly Error Rate Trends',
            xaxis_title='Hour of Day',
            yaxis_title='Error Rate (%)',
            height=400
        )
        
        return fig
    return None


def create_service_health_dashboard(log_df):
    """Create service health monitoring dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Service Availability', 'Response Time Percentiles', 'Error Distribution'),
        specs=[[{"type": "indicator"}, {"type": "bar"}, {"type": "pie"}]]
    )
    
    # Service availability
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=99.7,
            title={'text': "Availability %"},
            gauge={'axis': {'range': [95, 100]},
                  'bar': {'color': "green"},
                  'steps': [
                      {'range': [95, 97], 'color': "lightcoral"},
                      {'range': [97, 99], 'color': "lightyellow"},
                      {'range': [99, 100], 'color': "lightgreen"}]},
        ),
        row=1, col=1
    )
    
    # Response time percentiles
    percentiles = ['P50', 'P90', 'P95', 'P99']
    times = [120, 450, 780, 1200]  # milliseconds
    fig.add_trace(
        go.Bar(x=percentiles, y=times, marker_color='skyblue'),
        row=1, col=2
    )
    
    # Error distribution
    error_types = ['4xx Client', '5xx Server', 'Timeout', 'Other']
    error_counts = [234, 89, 45, 32]
    fig.add_trace(
        go.Pie(labels=error_types, values=error_counts),
        row=1, col=3
    )
    
    fig.update_layout(height=500, showlegend=False,
                     title_text="Service Health Monitoring")
    
    return fig


def visualize_log_operations():
    """Create concise log analytics visualization."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Log volume by source
    sources = ['Apache', 'App Logs', 'Security', 'System']
    volumes = [2500, 4200, 850, 1800]  # thousands of log entries
    colors = ['blue', 'green', 'red', 'orange']
    
    bars = axes[0].bar(sources, volumes, color=colors, alpha=0.7)
    axes[0].set_title('Log Volume by Source', fontweight='bold')
    axes[0].set_ylabel('Entries (thousands)')
    for bar, vol in zip(bars, volumes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{vol}K', ha='center', fontweight='bold')
    
    # 2. Error rates
    hours = list(range(24))
    error_rate = 2 + np.sin(np.array(hours) * np.pi / 12) + np.random.rand(24) * 0.5
    
    axes[1].plot(hours, error_rate, 'o-', linewidth=2, markersize=4, color='red')
    axes[1].fill_between(hours, error_rate, alpha=0.3, color='red')
    axes[1].set_title('Hourly Error Rate', fontweight='bold')
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Error Rate (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 23)
    
    # 3. Security event breakdown
    event_types = ['Failed\nLogin', 'Unauthorized\nAccess', 'Suspicious\nIP', 'Policy\nViolation']
    counts = [145, 89, 67, 43]
    colors_sec = ['darkred', 'orange', 'yellow', 'red']
    
    bars3 = axes[2].bar(event_types, counts, color=colors_sec, alpha=0.7)
    axes[2].set_title('Security Events', fontweight='bold')
    axes[2].set_ylabel('Count')
    for bar, cnt in zip(bars3, counts):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(cnt), ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('log_operations.png', dpi=150, bbox_inches='tight')
    print("Log operations visualization saved")
    
    return fig

