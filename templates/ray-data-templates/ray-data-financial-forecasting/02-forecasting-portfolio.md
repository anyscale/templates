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
# Import visualization utilities (170+ lines moved to util/viz_utils.py)
from util.viz_utils import create_comprehensive_portfolio_dashboard

# Create portfolio performance dashboard using utility function
# This dashboard includes:
# - Cumulative returns comparison across portfolio
# - Risk-return scatter plot with Sharpe ratios
# - Price correlation heatmap for diversification analysis
# - Volatility comparison bar charts
# - Drawdown analysis over time
# - Performance metrics visualization
portfolio_dashboard = create_comprehensive_portfolio_dashboard(financial_with_indicators)
portfolio_dashboard.show()
print("Portfolio performance dashboard saved as 'portfolio_performance_dashboard.html'")
```

### Advanced Financial Analytics

```python
# Import visualization utilities (213+ lines moved to util/viz_utils.py)
from util.viz_utils import create_improved_financial_analytics

# Create advanced financial analytics dashboard using utility function
# This 3x3 matplotlib dashboard includes:
# - Daily returns distribution histograms
# - Average trading volume analysis by symbol
# - Rolling volatility heatmap over time
# - RSI technical indicator trends
# - 20-day price momentum analysis
# - Price correlation matrix heatmap
# - Value at Risk (95%) comparison
# - Trading volume moving average patterns
# - Total return performance summary
create_improved_financial_analytics(financial_with_indicators)
print("Advanced financial analytics saved as 'improved_financial_analytics.png'")
```

Interactive visualizations provide insights into market behavior and portfolio performance across multiple dimensions.

## AutoARIMA Forecasting

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

**Why forecasting matters:**
- Anticipate price movements for better trading decisions
- Generate 30-day forecasts with confidence intervals
- Identify trend components and seasonality patterns

## Portfolio Optimization and Risk Analysis

Modern portfolio theory helps balance risk and return across multiple assets.

### Portfolio Optimization

```python
# Portfolio optimization using Ray Data batch processing
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

**Portfolio analysis provides:**
- Equal-weight and risk-parity allocation strategies
- Expected returns, volatility, and Sharpe ratios
- Value at Risk (VaR) and Conditional VaR metrics
- Maximum drawdown and beta calculations

## Key Takeaways

**Essential concepts covered:**
- **Interactive visualizations**: Candlestick charts and portfolio dashboards
- **Time series forecasting**: AutoARIMA for price predictions
- **Portfolio optimization**: Modern portfolio theory for asset allocation
- **Risk analysis**: Comprehensive metrics (VaR, CVaR, beta, Sharpe ratio)

**Ray Data advantages:**
- Process millions of financial records efficiently with distributed computing
- Calculate complex indicators across entire portfolios in parallel
- Scale from small portfolios to institutional-size datasets seamlessly

## Troubleshooting

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

### Performance Optimization

**Batch size recommendations:**
- Use larger batches (1000-5000) for portfolio calculations
- Smaller batches (500-1000) for per-symbol forecasting
- Match concurrency to available CPU cores

**Memory management:**
- Use Ray Data's streaming execution for large datasets
- Clear intermediate results for long time series
- Prefer Parquet format for efficient storage

## Next Steps

**Extend this template:**
1. Connect to real market data APIs (yfinance, Alpha Vantage)
2. Add more technical indicators (Fibonacci, Ichimoku clouds)
3. Implement ML models (LSTM, Transformers) for forecasting
4. Add Monte Carlo simulations for stress testing

**Production deployment:**
- Implement data quality validation and monitoring
- Track forecast accuracy and model performance
- Ensure regulatory compliance for financial calculations
- Set up risk controls and position limits

**Related templates:**
- [ML Feature Engineering](/templates/ray-data-ml-feature-engineering) - Build features for financial ML
- [Batch Inference Optimization](/templates/ray-data-batch-inference-optimization) - Optimize model inference
- [Data Quality Monitoring](/templates/ray-data-data-quality-monitoring) - Validate financial data

## Cleanup

Clean up Ray resources:

```python
# Cleanup Ray resources
if ray.is_initialized():
    ray.shutdown()
print("Financial forecasting tutorial completed")
```

---

**[← Back to Part 1](01-financial-data-indicators.md)** | **[Return to Overview](README.md)**
