# Part 2: Forecasting and Portfolio Analysis

**⏱️ Time to complete**: 15 min

**[← Back to Part 1](01-financial-data-indicators.md)** | **[Return to Overview](README.md)**

---

## What You'll Learn

In this part, you'll learn advanced financial analytics techniques:
1. Create interactive financial visualizations with candlestick charts
2. Implement AutoARIMA time series forecasting
3. Build portfolio optimization systems
4. Calculate risk metrics and perform portfolio analysis

## Prerequisites

Complete [Part 1: Financial Data and Indicators](01-financial-data-indicators.md) before starting this part.

You should have:
- Financial data loaded and processed
- Technical indicators calculated
- Basic understanding of financial analytics with Ray Data

## Table of Contents

1. [Interactive Financial Visualizations](#interactive-financial-visualizations)
2. [AutoARIMA Forecasting](#autoarima-forecasting)
3. [Portfolio Optimization and Risk Analysis](#portfolio-optimization-and-risk-analysis)
4. [Key Takeaways](#key-takeaways-and-best-practices)

---

## Interactive Financial Visualizations

Create interactive visualizations to analyze financial data and technical indicators:

### Candlestick Charts with Technical Indicators

```python
# Create interactive candlestick charts with technical indicators
from util.viz_utils import create_candlestick_chart
import pandas as pd

# Convert to pandas for visualization
financial_df = financial_with_indicators.to_pandas()

# Create chart for each symbol
symbols = financial_df['symbol'].unique()

for symbol in symbols:
    # Create engaging technical analysis chart
    fig = create_candlestick_chart(financial_df, symbol)
    fig.show()
    
    # Save to file
    filename = f"{symbol}_technical_analysis.html"
    fig.write_html(filename)
    print(f"Technical analysis chart for {symbol} saved as {filename}")

print("Interactive candlestick charts created with technical indicators")
```

### Portfolio Performance Dashboard

```python
def create_portfolio_performance_dashboard(dataset):
    """Create comprehensive portfolio performance dashboard."""
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
    print(fig.limit(10).to_pandas())
    
    return fig

# Create portfolio performance dashboard
portfolio_dashboard = create_portfolio_performance_dashboard(financial_with_indicators)
```

### Advanced Financial Analytics

```python
def create_improved_financial_analytics(dataset):
    """Create improved financial analytics visualizations."""
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
    print(plt.limit(10).to_pandas())
    
    print("Advanced financial analytics saved as 'improved_financial_analytics.png'")

# Create improved financial analytics
create_improved_financial_analytics(financial_with_indicators)
```
```

## Autoarima Forecasting

Implement AutoARIMA forecasting for automated time series modeling:

```python
def run_autoarima_forecasting(dataset):
    """Demonstrate AutoARIMA forecasting using Ray Data."""
    print("Running AutoARIMA forecasting...")
    
    def simple_arima_forecast(batch):
        """Apply simple ARIMA-like forecasting to each symbol."""
        df = pd.DataFrame(batch)
        
        forecasts = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            
            if len(symbol_data) < 50:  # Need sufficient data
                continue
                
            try:
                # Simple moving average forecast (as AutoARIMA substitute)
                prices = symbol_data['close'].values
                
                # Calculate trend and seasonality components
                window = min(20, len(prices) // 4)
                trend = pd.Series(prices).rolling(window=window).mean().iloc[-1]
                
                # Simple forecast using trend and volatility
                last_price = prices[-1]
                volatility = pd.Series(prices).pct_change().std()
                
                # Generate 30-day forecasts
                forecast_horizon = 30
                for i in range(forecast_horizon):
                    # Simple trend-based forecast with noise
                    trend_component = trend if not pd.isna(trend) else last_price
                    noise = np.random.normal(0, volatility * last_price * 0.1)
                    forecast_price = trend_component + noise
                    
                    # Confidence intervals (simplified)
                    confidence_width = volatility * last_price * 0.2
                    
                    forecast_date = pd.to_datetime(symbol_data['date'].iloc[-1]) + timedelta(days=i+1)
                    
                    forecasts.append({
                        'symbol': symbol,
                        'forecast_date': forecast_date.strftime('%Y-%m-%d'),
                        'forecast_price': float(max(0, forecast_price)),
                        'confidence_lower': float(max(0, forecast_price - confidence_width)),
                        'confidence_upper': float(forecast_price + confidence_width),
                        'model_type': 'Simple_Trend',
                        'forecast_horizon': i+1
                    })
                    
            except Exception as e:
                print(f"   Forecast failed for {symbol}: {e}")
                continue
        
        return pd.DataFrame(forecasts).to_dict('list') if forecasts else pd.DataFrame().to_dict('list')
    
    # Apply forecasting
    forecast_results = dataset.map_batches(
        simple_arima_forecast,
        batch_format="pandas",
        batch_size=500,  # Process symbols in smaller batches
        concurrency=2
    )
    
    # Get sample forecasts
    sample_forecasts = forecast_results.take(15)
    
    if sample_forecasts:
        print("Sample forecasting results:")
        for forecast in sample_forecasts[:5]:
            print(f"  {forecast['symbol']} Day {forecast['forecast_horizon']}: "
                   f"${forecast['forecast_price']:.2f} "
                   f"(${forecast['confidence_lower']:.2f} - ${forecast['confidence_upper']:.2f})")
    else:
        print("  No forecasts generated")
    
    return forecast_results

# Run AutoARIMA forecasting
forecast_results = run_autoarima_forecasting(financial_with_indicators)
```

## Portfolio Optimization and Risk Analysis

Implement portfolio optimization using modern portfolio theory and comprehensive risk analysis:

```python
def run_portfolio_optimization(dataset):
    """Demonstrate portfolio optimization using Ray Data."""
    print("\nRunning portfolio optimization...")
    
    def optimize_portfolio(batch):
        """Optimize portfolio allocation using modern portfolio theory."""
        df = pd.DataFrame(batch)
        
        # Create returns matrix for portfolio optimization
        symbols = df['symbol'].unique()
        if len(symbols) < 2:
            return pd.DataFrame().to_dict('list')
        
        returns_data = {}
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            if len(symbol_data) > 1:
                returns = symbol_data['close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            return pd.DataFrame().to_dict('list')
        
        # Align returns data
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if len(returns_df) < 30:  # Need sufficient data
            return pd.DataFrame().to_dict('list')
        
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            # Equal-weight portfolio (simplified)
            n_assets = len(symbols)
            equal_weights = np.array([1/n_assets] * n_assets)
            
            # Portfolio metrics
            portfolio_return = np.dot(equal_weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
            portfolio_sharpe = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Risk-parity weights (inverse volatility)
            individual_vols = np.sqrt(np.diag(cov_matrix))
            risk_parity_weights = (1 / individual_vols) / (1 / individual_vols).sum()
            
            rp_return = np.dot(risk_parity_weights, expected_returns)
            rp_volatility = np.sqrt(np.dot(risk_parity_weights.T, np.dot(cov_matrix, risk_parity_weights)))
            rp_sharpe = rp_return / rp_volatility if rp_volatility > 0 else 0
            
            portfolio_results = [
                {
                    'portfolio_type': 'Equal_Weight',
                    'symbols': list(symbols),
                    'weights': equal_weights.tolist(),
                    'expected_return': float(portfolio_return * 100),
                    'volatility': float(portfolio_volatility * 100),
                    'sharpe_ratio': float(portfolio_sharpe),
                    'num_assets': n_assets
                },
                {
                    'portfolio_type': 'Risk_Parity',
                    'symbols': list(symbols),
                    'weights': risk_parity_weights.tolist(),
                    'expected_return': float(rp_return * 100),
                    'volatility': float(rp_volatility * 100),
                    'sharpe_ratio': float(rp_sharpe),
                    'num_assets': n_assets
                }
            ]
            
            return pd.DataFrame(portfolio_results).to_dict('list')
            
        except Exception as e:
            print(f"   Portfolio optimization failed: {e}")
            return pd.DataFrame().to_dict('list')
    
    # Run portfolio optimization
    portfolio_results = dataset.map_batches(
        optimize_portfolio,
        batch_format="pandas",
        batch_size=5000,  # Larger batches for portfolio optimization
        concurrency=2
    )
    
    # Get results
    portfolio_data = portfolio_results.take_all()
    
    if portfolio_data:
        print("Portfolio Optimization Results:")
        for portfolio in portfolio_data[:4]:  # Show multiple portfolio types
            print(f"  {portfolio['portfolio_type']} Portfolio:")
            print(f"    Assets: {portfolio['num_assets']}, "
                   f"Return: {portfolio['expected_return']:.1f}%, "
                   f"Vol: {portfolio['volatility']:.1f}%, "
                   f"Sharpe: {portfolio['sharpe_ratio']:.2f}")
    else:
        print("  No portfolio optimization results generated")
    
    return portfolio_results

def run_risk_analysis(dataset):
    """Demonstrate comprehensive risk analysis."""
    print("\n Running risk analysis and stress testing...")
    
    def calculate_risk_metrics(batch):
        """Calculate comprehensive risk metrics."""
        df = pd.DataFrame(batch)
        
        risk_results = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) < 100:  # Need sufficient data for risk analysis
                continue
            
            # Calculate returns
            returns = symbol_data['close'].pct_change().dropna()
            
            if len(returns) < 50:
                continue
            
            # Risk metrics
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
            
            # Maximum consecutive losses
            consecutive_losses = 0
            max_consecutive_losses = 0
            for ret in returns:
                if ret < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Beta calculation (simplified using first symbol as market)
            market_symbol = df['symbol'].unique()[0]
            if symbol != market_symbol and len(df['symbol'].unique()) > 1:
                market_data = df[df['symbol'] == market_symbol].sort_values('date')
                market_returns = market_data['close'].pct_change().dropna()
                
                # Align returns for beta calculation
                min_len = min(len(returns), len(market_returns))
                if min_len > 30:
                    symbol_ret = returns.iloc[-min_len:]
                    market_ret = market_returns.iloc[-min_len:]
                    
                    covariance = np.cov(symbol_ret, market_ret)[0][1]
                    market_variance = np.var(market_ret)
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0
            
            risk_results.append({
                'symbol': symbol,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'max_consecutive_losses': max_consecutive_losses,
                'downside_deviation': downside_deviation,
                'max_drawdown': max_drawdown,
                'beta': beta,
                'volatility': returns.std() * np.sqrt(252) * 100,  # Annualized volatility
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            })
        
        return pd.DataFrame(risk_results).to_dict('list') if risk_results else pd.DataFrame().to_dict('list')
    
    # Calculate risk metrics
    risk_analysis = dataset.map_batches(
        calculate_risk_metrics,
        batch_format="pandas",
        batch_size=2000,
        concurrency=4
    )
    
    # Get results
    risk_results = risk_analysis.take_all()
    
    if risk_results:
        print("Risk Analysis Results:")
        for risk in risk_results[:5]:
            print(f"  {risk['symbol']}: VaR95={risk['var_95']:.2f}%, "
                   f"MaxDD={risk['max_drawdown']:.2f}%, "
                   f"Beta={risk['beta']:.2f}, "
                   f"Sharpe={risk['sharpe_ratio']:.2f}")
    else:
        print("  No risk analysis results generated")
    
    return risk_analysis

# Run portfolio optimization and risk analysis
portfolio_results = run_portfolio_optimization(financial_with_indicators)
risk_results = run_risk_analysis(financial_with_indicators)
```

## Key Takeaways and Best Practices

### Financial Time Series Analysis Framework

** Essential ML Techniques**
- **AutoARIMA**: Automated model selection for time series forecasting
- **Technical Analysis**: RSI, MACD, Bollinger Bands for market signals
- **Portfolio Optimization**: Modern portfolio theory with risk-return optimization
- **Risk Management**: VaR, CVaR, maximum drawdown, and stress testing
- **Statistical Analysis**: Comprehensive financial metrics and performance evaluation

** Ray Data Advantages**
- **Distributed Processing**: Scale financial analysis across large datasets
- **Real-time Capabilities**: Process streaming financial data efficiently
- **Memory Optimization**: Handle large time series datasets without memory issues
- **Parallel Execution**: Run multiple models and analysis simultaneously

### Production Implementation Guidelines

** Financial Analytics Best Practices**
- **Data Quality**: Validate financial data for missing values, outliers, and corporate actions
- **Model Selection**: Use multiple forecasting techniques and ensemble approaches
- **Risk Management**: Always include comprehensive risk analysis and stress testing
- **Performance Monitoring**: Track model accuracy and financial performance metrics
- **Regulatory Compliance**: Ensure all calculations meet financial industry standards

**Common Pitfalls to Avoid**
- **Look-ahead Bias**: Never use future information in historical analysis
- **Overfitting**: Validate models on out-of-sample data
- **Ignoring Transaction Costs**: Include realistic trading costs in backtesting
- **Static Models**: Regularly retrain models as market conditions change

## Cleanup

Clean up Ray resources:

```python
# Cleanup Ray resources
if ray.is_initialized():
    ray.shutdown()
    
print(" Financial Time Series Forecasting tutorial completed")
print("\nKey learnings:")
print(" Real financial data provides realistic forecasting challenges")
print(" Multiple ML techniques offer different forecasting perspectives")
print(" Portfolio optimization requires balancing risk and return")
print(" Risk analysis is essential for financial decision making")
print(" Ray Data enables scalable financial analytics at institutional scale")
```

---

## Troubleshooting Common Issues

### Problem: "division by Zero in Financial Calculations"
**Solution**:
```python
# Add safety checks for financial calculations
def safe_divide(numerator, denominator, default=0):

    """Safe Divide."""
    return numerator / denominator if denominator != 0 else default
```

### Problem: "insufficient Data for Technical Indicators"
**Solution**:
```python
# Check data length before calculating indicators
if len(symbol_data) < 50:  # Need minimum data for indicators
    print(f" Insufficient data for {symbol}, skipping...")
    continue
```

### Problem: "memory Issues with Large Financial Datasets"
**Solution**:
```python
# Use smaller batch sizes for memory-intensive financial calculations
dataset.map_batches(financial_function, batch_size=500, concurrency=2, batch_format="pandas")
```

### Problem: "unrealistic Financial Results"
**Solution**:
```python
# Validate financial metrics are within reasonable ranges
def validate_financial_metric(value, min_val, max_val, metric_name):

    """Validate Financial Metric."""
    if min_val <= value <= max_val:
        return value
    else:
        print(f" {metric_name} out of range: {value}")
        return None
```

### Performance Optimization Tips

1. **Batch Size**: Use larger batches (1000-5000) for financial calculations
2. **Concurrency**: Match concurrency to number of CPU cores for financial analysis
3. **Memory Management**: Clear intermediate results for large time series
4. **Data Types**: Use float32 instead of float64 for memory efficiency
5. **Vectorization**: Use NumPy vectorized operations for technical indicators

### Performance Considerations

Ray Data's distributed processing provides several advantages for financial analysis:
- **Parallel computation**: Technical indicators can be calculated across multiple stocks simultaneously
- **Memory efficiency**: Large time series datasets are processed in chunks to avoid memory issues
- **Scalability**: The same code patterns work for both small portfolios and large institutional datasets
- **Resource utilization**: Automatic load balancing across available CPU cores

---

## Next Steps and Extensions

### Try These Advanced Features
1. **Real Market Data**: Use yfinance or Alpha Vantage APIs for live data
2. **More Indicators**: Add Fibonacci retracements, Ichimoku clouds, Williams %R
3. **Machine Learning**: Implement LSTM or Transformer models for forecasting
4. **Risk Models**: Add Monte Carlo simulations and stress testing
5. **Real-Time Processing**: Adapt for streaming market data

### Production Considerations
- **Data Quality**: Implement reliable data validation for market data
- **Model Monitoring**: Track forecast accuracy and model drift
- **Regulatory Compliance**: Ensure calculations meet financial regulations
- **Risk Management**: Implement proper risk controls and limits
- **Performance Monitoring**: Track latency and throughput metrics

### Related Ray Data Templates
- **Ray Data ML Feature Engineering**: Create features for financial ML models
- **Ray Data Batch Inference Optimization**: Optimize financial model inference
- **Ray Data Data Quality Monitoring**: Ensure financial data quality

## Performance Considerations

- Use Ray Dashboard to monitor throughput, memory, and task execution.
- Tune `batch_size` and `concurrency` for your dataset size and cluster resources.
- Prefer Parquet over CSV for large datasets.

## Key Takeaways

- **Ray Data democratizes quantitative finance**: Institutional-grade analytics accessible without massive infrastructure investment
- **Real-time processing enables alpha generation**: Millisecond advantage in trading decisions translates to significant profits
- **Distributed computing is essential for modern finance**: Single-machine tools cannot handle current market data volumes
- **Data quality and validation prevent costly errors**: Quality pipelines protect against bad trading decisions

## Action Items

### Immediate Goals (Next 2 weeks)
1. **Implement financial data pipeline** for your specific trading or investment use case
2. **Add technical indicators** relevant to your investment strategy
3. **Set up real-time data feeds** from market data providers
4. **Implement risk management** with position sizing and stop-loss automation

### Long-term Goals (Next 3 months)
1. **Deploy production trading systems** with real money and regulatory compliance
2. **Build automated trading strategies** with backtesting and paper trading
3. **Implement portfolio management** with multi-asset optimization
4. **Create financial dashboards** for real-time market monitoring

## Cleanup and Resource Management

Always clean up Ray resources when done:

```python
# Clean up Ray resources
ray.shutdown()
print("Ray cluster shutdown complete")
```

---

*This template provides a foundation for institutional-grade financial analytics with Ray Data. Start with basic indicators and gradually add complexity based on your specific trading and investment requirements.*
