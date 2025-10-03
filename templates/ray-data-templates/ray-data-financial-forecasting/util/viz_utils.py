"""Visualization utilities for financial forecasting templates."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_candlestick_chart(df, symbol):
    """Create interactive candlestick chart with technical indicators."""
    symbol_data = df[df['symbol'] == symbol].sort_values('date')
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=symbol_data['date'],
            open=symbol_data['open'],
            high=symbol_data['high'],
            low=symbol_data['low'],
            close=symbol_data['close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'sma_20' in symbol_data.columns:
        fig.add_trace(
            go.Scatter(x=symbol_data['date'], y=symbol_data['sma_20'],
                      name='SMA 20', line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(x=symbol_data['date'], y=symbol_data['volume'],
               name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} - Price and Volume',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig


def create_portfolio_dashboard(portfolio_df):
    """Create interactive portfolio analysis dashboard."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Portfolio Allocation', 'Risk vs Return', 'Performance Trends'),
        specs=[[{"type": "pie"}, {"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Portfolio allocation
    fig.add_trace(
        go.Pie(labels=portfolio_df['symbol'], values=portfolio_df['allocation'],
               name="Allocation"),
        row=1, col=1
    )
    
    # Risk vs return scatter
    fig.add_trace(
        go.Scatter(x=portfolio_df['risk'], y=portfolio_df['return'],
                  mode='markers+text', text=portfolio_df['symbol'],
                  marker=dict(size=15), name="Stocks"),
        row=1, col=2
    )
    
    # Performance comparison
    fig.add_trace(
        go.Bar(x=portfolio_df['symbol'], y=portfolio_df['return'],
               name="Returns", marker_color='green'),
        row=1, col=3
    )
    
    fig.update_layout(height=500, showlegend=True,
                     title_text="Portfolio Analysis Dashboard")
    
    return fig


def create_financial_summary_dashboard(stock_df, news_df):
    """Create simple financial data summary visualizations."""
    import plotly.express as px
    
    # Stock price trends
    fig1 = px.line(stock_df, x='date', y='close', color='symbol',
                   title='Stock Price Trends')
    
    # Trading volume
    volume_avg = stock_df.groupby('symbol')['volume'].mean().reset_index()
    fig2 = px.bar(volume_avg, x='symbol', y='volume',
                  title='Average Trading Volume')
    
    # News sentiment
    sentiment_counts = news_df['sentiment'].value_counts().reset_index()
    fig3 = px.pie(sentiment_counts, names='sentiment', values='count',
                  title='News Sentiment Distribution')
    
    return fig1, fig2, fig3


def create_portfolio_performance_dashboard(dataset):
    """Create comprehensive portfolio performance dashboard."""
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import pandas as pd
    
    # Convert to pandas
    df = dataset.to_pandas()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio Allocation', 'Risk vs Return', 'Performance Over Time', 'Sharpe Ratios'),
        specs=[[{"type": "pie"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Portfolio allocation
    if 'symbol' in df.columns and 'allocation' in df.columns:
        fig.add_trace(
            go.Pie(labels=df['symbol'], values=df['allocation'], name="Allocation"),
            row=1, col=1
        )
    
    # Risk vs return scatter
    if 'risk' in df.columns and 'return' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['risk'], y=df['return'], mode='markers+text',
                      text=df['symbol'], marker=dict(size=15)),
            row=1, col=2
        )
    
    # Performance over time
    if 'date' in df.columns and 'cumulative_return' in df.columns:
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            fig.add_trace(
                go.Scatter(x=symbol_data['date'], y=symbol_data['cumulative_return'],
                          mode='lines', name=symbol),
                row=2, col=1
            )
    
    # Sharpe ratios
    if 'sharpe_ratio' in df.columns:
        fig.add_trace(
            go.Bar(x=df['symbol'], y=df['sharpe_ratio'], marker_color='green'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True,
                     title_text="Portfolio Performance Dashboard")
    
    return fig


def create_volatility_surface(stock_df):
    """Create 3D volatility surface visualization."""
    import plotly.graph_objects as go
    
    # Calculate rolling volatility
    symbols = stock_df['symbol'].unique()
    dates = stock_df['date'].unique()
    
    fig = go.Figure(data=[go.Surface(
        z=stock_df.pivot_table(values='volatility', index='date', columns='symbol').values,
        x=symbols,
        y=dates,
        colorscale='Viridis'
    )])
    
    fig.update_layout(
        title='Volatility Surface Across Stocks',
        scene=dict(
            xaxis_title='Symbol',
            yaxis_title='Date',
            zaxis_title='Volatility'
        ),
        height=700
    )
    
    return fig


def create_correlation_heatmap(returns_df):
    """Create correlation heatmap for portfolio analysis."""
    import plotly.express as px
    
    # Calculate correlation matrix
    corr_matrix = returns_df.pivot(columns='symbol', values='return').corr()
    
    fig = px.imshow(corr_matrix,
                    title='Stock Returns Correlation Matrix',
                    color_continuous_scale='RdBu',
                    aspect='auto')
    
    fig.update_layout(height=600)
    return fig

