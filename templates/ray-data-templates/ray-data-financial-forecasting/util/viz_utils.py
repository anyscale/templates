"""Visualization utilities for financial forecasting templates."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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


def create_comprehensive_portfolio_dashboard(dataset):
    """Create comprehensive portfolio performance dashboard with detailed metrics."""
    print("Creating portfolio performance dashboard...")
    
    financial_df = dataset.to_pandas()
    
    # Calculate portfolio metrics
    portfolio_data = []
    symbols = financial_df['symbol'].unique()
    
    for symbol in symbols:
        symbol_data = financial_df[financial_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        
        if len(symbol_data) > 1:
            # Calculate returns
            symbol_data['daily_return'] = symbol_data['close'].pct_change()
            
            # Calculate cumulative returns
            symbol_data['cumulative_return'] = (1 + symbol_data['daily_return']).cumprod()
            
            # Portfolio metrics
            total_return = (symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0] - 1) * 100
            volatility = symbol_data['daily_return'].std() * np.sqrt(252) * 100
            sharpe_ratio = (symbol_data['daily_return'].mean() * 252) / (symbol_data['daily_return'].std() * np.sqrt(252)) if symbol_data['daily_return'].std() > 0 else 0
            max_drawdown = ((symbol_data['close'] / symbol_data['close'].cummax()) - 1).min() * 100
            
            portfolio_data.append({
                'symbol': symbol,
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'current_price': symbol_data['close'].iloc[-1],
                'data': symbol_data
            })
    
    # Create dashboard with multiple subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Cumulative Returns', 'Risk-Return Scatter', 'Price Correlation Heatmap', 
                       'Volatility Comparison', 'Drawdown Analysis', 'Performance Metrics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Cumulative returns comparison
    colors = ['blue', 'red', 'green', 'orange']
    for i, portfolio in enumerate(portfolio_data):
        symbol_data = portfolio['data']
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(symbol_data['date']),
            y=portfolio['data']['cumulative_return'],
            mode='lines',
            name=f"{portfolio['symbol']} Returns",
            line=dict(color=colors[i % len(colors)], width=2)
        ), row=1, col=1)
    
    # 2. Risk-Return scatter plot
    fig.add_trace(go.Scatter(
        x=[p['volatility'] for p in portfolio_data],
        y=[p['total_return'] for p in portfolio_data],
        mode='markers+text',
        text=[p['symbol'] for p in portfolio_data],
        textposition="top center",
        marker=dict(
            size=15,
            color=[p['sharpe_ratio'] for p in portfolio_data],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio", x=0.45)
        ),
        name="Risk-Return"
    ), row=1, col=2)
    
    # 3. Correlation heatmap
    if len(portfolio_data) > 1:
        price_data = {}
        for portfolio in portfolio_data:
            symbol_data = portfolio['data']
            price_data[portfolio['symbol']] = symbol_data.set_index('date')['close']
        
        price_df = pd.DataFrame(price_data).dropna()
        correlation_matrix = price_df.corr()
        
        fig.add_trace(go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            showscale=True
        ), row=2, col=1)
    
    # 4. Volatility comparison
    fig.add_trace(go.Bar(
        x=[p['symbol'] for p in portfolio_data],
        y=[p['volatility'] for p in portfolio_data],
        name="Volatility (%)",
        marker_color='lightcoral'
    ), row=2, col=2)
    
    # 5. Drawdown analysis
    for i, portfolio in enumerate(portfolio_data):
        symbol_data = portfolio['data']
        drawdown = ((symbol_data['close'] / symbol_data['close'].cummax()) - 1) * 100
        
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(symbol_data['date']),
            y=drawdown,
            mode='lines',
            name=f"{portfolio['symbol']} Drawdown",
            line=dict(color=colors[i % len(colors)]),
            fill='tonexty' if i == 0 else None,
            fillcolor=f'rgba({colors[i % len(colors)]}, 0.3)' if i == 0 else None
        ), row=3, col=1)
    
    # 6. Performance metrics table (as bar chart)
    metrics = ['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
    
    for i, metric in enumerate(metrics):
        if metric == 'Total Return (%)':
            values = [p['total_return'] for p in portfolio_data]
        elif metric == 'Volatility (%)':
            values = [p['volatility'] for p in portfolio_data]
        elif metric == 'Sharpe Ratio':
            values = [p['sharpe_ratio'] for p in portfolio_data]
        else:  # Max Drawdown
            values = [p['max_drawdown'] for p in portfolio_data]
        
        fig.add_trace(go.Bar(
            x=[p['symbol'] for p in portfolio_data],
            y=values,
            name=metric,
            offsetgroup=i,
            width=0.2
        ), row=3, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="Portfolio Performance Dashboard",
        height=1200,
        showlegend=True
    )
    
    # Update axis titles
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
    fig.update_yaxes(title_text="Total Return (%)", row=1, col=2)
    fig.update_xaxes(title_text="Symbol", row=2, col=2)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=2)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    fig.update_xaxes(title_text="Symbol", row=3, col=2)
    fig.update_yaxes(title_text="Value", row=3, col=2)
    
    # Save dashboard
    fig.write_html("portfolio_performance_dashboard.html")
    print("Portfolio performance dashboard saved as 'portfolio_performance_dashboard.html'")
    
    return fig


def create_improved_financial_analytics(dataset):
    """Create improved financial analytics visualizations with matplotlib."""
    print("Creating improved financial analytics...")
    
    financial_df = dataset.to_pandas()
    
    # Set style for matplotlib plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create comprehensive analytics dashboard
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Advanced Financial Analytics Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Returns distribution
    ax1 = axes[0, 0]
    for symbol in financial_df['symbol'].unique():
        symbol_data = financial_df[financial_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        if len(symbol_data) > 1:
            returns = symbol_data['close'].pct_change().dropna()
            ax1.hist(returns, alpha=0.7, bins=30, label=symbol, density=True)
    
    ax1.set_title('Daily Returns Distribution', fontweight='bold')
    ax1.set_xlabel('Daily Return')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Volume analysis
    ax2 = axes[0, 1]
    volume_data = financial_df.groupby('symbol')['volume'].mean().sort_values(ascending=False)
    bars = ax2.bar(volume_data.index, volume_data.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('Average Trading Volume by Symbol', fontweight='bold')
    ax2.set_ylabel('Average Volume')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height/1000000):.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 3. Price volatility heatmap
    ax3 = axes[0, 2]
    volatility_data = []
    symbols = financial_df['symbol'].unique()
    
    for symbol in symbols:
        symbol_data = financial_df[financial_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        if len(symbol_data) > 1:
            returns = symbol_data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().dropna()
            volatility_data.append(volatility.values)
    
    if volatility_data:
        # Pad arrays to same length
        max_len = max(len(arr) for arr in volatility_data)
        padded_data = []
        for arr in volatility_data:
            padded = np.full(max_len, np.nan)
            padded[:len(arr)] = arr
            padded_data.append(padded)
        
        volatility_matrix = np.array(padded_data)
        im = ax3.imshow(volatility_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_title('Rolling Volatility Heatmap', fontweight='bold')
        ax3.set_ylabel('Symbol')
        ax3.set_xlabel('Time Period')
        ax3.set_yticks(range(len(symbols)))
        ax3.set_yticklabels(symbols)
        plt.colorbar(im, ax=ax3, label='Volatility')
    
    # 4. Technical indicators comparison
    ax4 = axes[1, 0]
    if 'rsi' in financial_df.columns:
        for symbol in symbols:
            symbol_data = financial_df[financial_df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            if 'rsi' in symbol_data.columns:
                rsi_values = symbol_data['rsi'].dropna()
                if len(rsi_values) > 0:
                    ax4.plot(range(len(rsi_values)), rsi_values, label=symbol, alpha=0.8)
        
        ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        ax4.set_title('RSI Technical Indicator', fontweight='bold')
        ax4.set_ylabel('RSI')
        ax4.set_xlabel('Time Period')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Price momentum analysis
    ax5 = axes[1, 1]
    momentum_data = []
    for symbol in symbols:
        symbol_data = financial_df[financial_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        if len(symbol_data) > 20:
            momentum = symbol_data['close'].pct_change(20).dropna()
            if len(momentum) > 0:
                momentum_data.append({
                    'symbol': symbol,
                    'momentum': momentum.iloc[-1] * 100  # Latest 20-day momentum
                })
    
    if momentum_data:
        momentum_df = pd.DataFrame(momentum_data)
        colors = ['green' if x > 0 else 'red' for x in momentum_df['momentum']]
        bars = ax5.bar(momentum_df['symbol'], momentum_df['momentum'], color=colors, alpha=0.7)
        ax5.set_title('20-Day Price Momentum', fontweight='bold')
        ax5.set_ylabel('Momentum (%)')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -2),
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold')
    
    # 6. Correlation network
    ax6 = axes[1, 2]
    if len(symbols) > 1:
        price_data = {}
        for symbol in symbols:
            symbol_data = financial_df[financial_df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            price_data[symbol] = symbol_data.set_index('date')['close']
        
        price_df = pd.DataFrame(price_data).dropna()
        if len(price_df) > 1:
            correlation_matrix = price_df.corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax6, cbar_kws={"shrink": .8})
            ax6.set_title('Price Correlation Matrix', fontweight='bold')
    
    # 7. Risk metrics comparison
    ax7 = axes[2, 0]
    risk_metrics = []
    for symbol in symbols:
        symbol_data = financial_df[financial_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        if len(symbol_data) > 1:
            returns = symbol_data['close'].pct_change().dropna()
            if len(returns) > 0:
                var_95 = np.percentile(returns, 5) * 100
                risk_metrics.append({'symbol': symbol, 'VaR_95': var_95})
    
    if risk_metrics:
        risk_df = pd.DataFrame(risk_metrics)
        bars = ax7.bar(risk_df['symbol'], risk_df['VaR_95'], color='red', alpha=0.7)
        ax7.set_title('Value at Risk (95%)', fontweight='bold')
        ax7.set_ylabel('VaR (%)')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height - 0.1,
                    f'{height:.2f}%', ha='center', va='top', fontweight='bold', color='white')
    
    # 8. Trading volume patterns
    ax8 = axes[2, 1]
    for symbol in symbols:
        symbol_data = financial_df[financial_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        if len(symbol_data) > 0:
            # Create volume moving average
            if len(symbol_data) > 20:
                volume_ma = symbol_data['volume'].rolling(window=20).mean()
                ax8.plot(range(len(volume_ma)), volume_ma, label=f'{symbol} Volume MA', alpha=0.8)
    
    ax8.set_title('Volume Moving Average Trends', fontweight='bold')
    ax8.set_ylabel('Volume (MA20)')
    ax8.set_xlabel('Time Period')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Performance summary
    ax9 = axes[2, 2]
    performance_metrics = []
    for symbol in symbols:
        symbol_data = financial_df[financial_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        if len(symbol_data) > 1:
            total_return = (symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0] - 1) * 100
            performance_metrics.append({'symbol': symbol, 'return': total_return})
    
    if performance_metrics:
        perf_df = pd.DataFrame(performance_metrics)
        colors = ['green' if x > 0 else 'red' for x in perf_df['return']]
        bars = ax9.bar(perf_df['symbol'], perf_df['return'], color=colors, alpha=0.7)
        ax9.set_title('Total Return Performance', fontweight='bold')
        ax9.set_ylabel('Total Return (%)')
        ax9.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -2),
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('improved_financial_analytics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Advanced financial analytics saved as 'improved_financial_analytics.png'")

