# Option chain pricing with Ray

**⏱️ Time to complete**: 30 min

- Option chain pricing explodes into many independent calculations (symbol x contract x shock), which is a natural fit for task parallelism.
- Ray Core lets us keep Python functions while distributing work across CPUs with `@ray.remote`.
- Ray's dynamic execution model (from the Ray paper) is built for irregular workloads where task counts change at runtime.


#### Dependencies

- `ray`, `pandas`, and the local `util.py` helpers are enough to move from serial loops to distributed execution.
- `ray.init()` lets the same notebook run on a laptop or scale out to an Anyscale cluster.
- Resource hints (for example `@ray.remote(num_cpus=1)`) keep throughput predictable and cost-aware as concurrency grows.



```python
!pip install QuantLib==1.42 yfinance==1.2.2
```


```python
import os
import pandas as pd
import ray

from util import get_iv, get_npv, get_options_chain, save_csv, get_symbols_stat_print as SYMBOL_STATS_PRINT, FuncTimer as ft

os.environ["RAY_DEDUP_LOGS"] = "0"
```

#### Serial implementation

- This baseline is useful for correctness checks and side-by-side timing.
- Limitation: symbols are processed one at a time, so runtime grows roughly linearly with universe size.
- Ray Core value: we keep the pricing logic and parallelize with minimal structural changes.



```python
def price_option_chain(
    symbol: str,
    iv_shocks = [0.05, 0.10],    # Shock percents to apply
    price_shocks = [0.05, 0.10], # Shock percents to apply
) -> str:
    """
    Price an options chain for a given symbol with various shocks to implied volatility and underlying price.
    
    Arguments:
    - symbol : The stock ticker symbol.
    - iv_shocks : List of shocks to apply to implied volatility.
    - price_shocks List of shocks to apply to the underlying stock price.
    
    Returns the path to the CSV file containing the results.
    
    """
    total_t = ft()

    # Fetch options chain data
    df = pd.DataFrame(get_options_chain(symbol)['options'])

    # Calculate base implied volatility
    df["implied_volatility"] = df.apply(lambda row: get_iv(row), axis=1)

    # Apply shocks and calculate NPVs
    for i, (iv_shock, price_shock) in enumerate(zip(iv_shocks, price_shocks), start=1):
        # Shocked implied volatility
        df[f"s{i}_implied_volatility"] = df["implied_volatility"] + iv_shock
        
        # Shocked underlying price
        df[f"s{i}_underlying_price"] = df["underlying_price"] * (1 - price_shock)
        
        # Calculate NPV under shocked scenario
        df[f"s{i}_npv"] = df.apply(
            lambda row: get_npv(
                row, row[f"s{i}_underlying_price"], row[f"s{i}_implied_volatility"]
            ), axis=1
        )

    # Save results to CSV
    new_file_path = save_csv(df, symbol)

    total_t.e(SYMBOL_STATS_PRINT(symbol, df))
    return new_file_path
```


```python
skip_non_ray = True

if skip_non_ray:
    print(
        """
        Stats for   AAPL:  1843 options, calc'd IV for all shocks in 137.022463 sec
        Stats for   AMZN:  1520 options, calc'd IV for all shocks in 112.304598 sec
        Total time for all symbols: 249.327518 sec
        """
    )
else:
    ## the following takes ~5 minutes to run for all symbols
    symbol_list = ["AAPL", "AMZN"] #, "TSLA", "NVDA", "MSFT", "META", "GOOGL"]

    all_symbols_t = ft()

    for symbol in symbol_list:
        price_option_chain(symbol)

    all_symbols_t.e("Total time for all symbols: ")
```

#### Parallel implementation with Ray

- One Ray task per symbol gives immediate horizontal scale across cores and nodes.
- Best practice: submit all tasks first, then call `ray.get(futures)` once to preserve parallelism.
- For uneven symbol workloads, `ray.wait(...)` helps process completed work early and keep workers busy.
- Operational benefit: unfinished tasks can be rescheduled if a worker fails, reducing rerun risk for long pricing jobs.



```python
@ray.remote
def parallel_price_option_chain(
    symbol: str,
    iv_shocks = [0.05, 0.10],    # Shock percents to apply
    price_shocks = [0.05, 0.10], # Shock percents to apply
) -> str:
    """
    Price an options chain for a given symbol with various shocks to implied volatility and underlying price.
    
    Arguments:
    - symbol : The stock ticker symbol.
    - iv_shocks : List of shocks to apply to implied volatility.
    - price_shocks List of shocks to apply to the underlying stock price.
    
    Returns the path to the CSV file containing the results.
    
    """
    total_t = ft()

    # Fetch options chain data
    df = pd.DataFrame(get_options_chain(symbol)['options'])

    # Calculate base implied volatility
    df["implied_volatility"] = df.apply(lambda row: get_iv(row), axis=1)

    # Apply shocks and calculate NPVs
    for i, (iv_shock, price_shock) in enumerate(zip(iv_shocks, price_shocks), start=1):
        # Shocked implied volatility
        df[f"s{i}_implied_volatility"] = df["implied_volatility"] + iv_shock
        
        # Shocked underlying price
        df[f"s{i}_underlying_price"] = df["underlying_price"] * (1 - price_shock)
        
        # Calculate NPV under shocked scenario
        df[f"s{i}_npv"] = df.apply(
            lambda row: get_npv(
                row, row[f"s{i}_underlying_price"], row[f"s{i}_implied_volatility"]
            ), axis=1
        )

    # Save results to CSV
    new_file_path = save_csv(df, symbol)

    total_t.e(SYMBOL_STATS_PRINT(symbol, df))
    return new_file_path
```


```python
symbol_list = ["AAPL", "AMZN"] #, "TSLA", "NVDA", "MSFT", "META", "GOOGL"]
all_symbols_t = ft()

futures = [parallel_price_option_chain.remote(symbol) for symbol in symbol_list]
results = ray.get(futures) # ray.wait(...)

all_symbols_t.e("Total time for all symbols: ")
# this will run for ~2-3 minutes. let's look at the Anyscale observability (e.g. metrics) tabs in the meantime
```

### Observability

While that's running, let's look at the Anyscale observability suite under the Metrics and Ray Dashboard tabs. 

#### Additional parallelization

- This cell demonstrates **nested parallelism**: a parent Ray task (`more_parallel_price_option_chain`) dynamically spawns child Ray tasks (`compute_stats`) per shock scenario.
- Ray Core supports dynamic task graphs, so fan-out can adapt at runtime to symbol complexity, shock count, or market regime.
- Cost/perf guardrail: keep nested tasks coarse enough to amortize scheduler overhead; batch tiny computations together.

**Example of [Nested Parallelism](https://docs.ray.io/en/latest/ray-core/patterns/nested-tasks.html)**

<img src=https://docs.ray.io/en/latest/_images/tree-of-tasks.svg>

- **Quant examples: intraday desk-level stress + VaR**
  - Parent task `run_desk_risk(desk)` fans out to symbol-level pricing tasks.
  - Each symbol task dynamically fans out to shock tasks (spot shocks, vol surface bumps, rate shifts).
  - Child tasks compute scenario PnL/Greeks; parent aggregates P95/P99 VaR, expected shortfall, and top contributors.
  - Only symbols breaching risk thresholds spawn deeper Monte Carlo path tasks, concentrating compute where it matters most.

- **Other finance use cases for nested tasks**
  - XVA/CVA: counterparty -> netting set -> scenario/path simulation.
  - Calibration: model family -> maturity bucket -> strike-surface slice.
  - Backtesting: strategy -> day -> parameter grid, with fast portfolio-level aggregation.


```python
@ray.remote(num_cpus=1)
def more_parallel_price_option_chain(
    symbol,
    iv_shocks = [0.05, 0.10],    # Shock percents to apply
    price_shocks = [0.05, 0.10], # Shock percents to apply
):
    """
    Price an options chain for a given symbol with various shocks to implied volatility and underlying price.
    
    Arguments:
    - symbol : The stock ticker symbol.
    - iv_shocks : List of shocks to apply to implied volatility.
    - price_shocks List of shocks to apply to the underlying stock price.
    
    Returns the path to the CSV file containing the results.
    
    """
    total_t = ft()

    # Fetch options chain data
    df = pd.DataFrame(get_options_chain(symbol)['options'])
 
    # Calculate base implied volatility
    df["implied_volatility"] = df.apply(lambda row: get_iv(row), axis=1)

    @ray.remote
    def compute_stats(df: pd.DataFrame, iv_shock: float, price_shock, idx: int, shock_num: int):
        iv_col = f"s{idx}_implied_volatility"
        up_col = f"s{idx}_underlying_price"
        npv_col = f"s{idx}_npv"
        df[iv_col] = df["implied_volatility"] + iv_shock
        df[up_col] = df["underlying_price"] * (1-price_shock)
        df[npv_col] = df.apply(lambda row: get_npv(row, row[up_col], row[iv_col]), axis=1)
        return df[[iv_col, up_col, npv_col]]

    shock_num = len(iv_shocks)
    
    futures = [
        compute_stats.remote(df, iv_shocks[idx], price_shocks[idx], idx+1, shock_num)
        for idx in range(shock_num)
    ]

    scenario_cols = ray.get(futures)
    df = pd.concat([df] + scenario_cols, axis=1)

    # Save results to CSV
    new_file_path = save_csv(df, symbol)

    total_t.e(SYMBOL_STATS_PRINT(symbol, df))
    return new_file_path
```


```python
symbol_list = ["AAPL", "AMZN"] #, "TSLA", "NVDA", "MSFT", "META", "GOOGL"]

all_symbols_t = ft()
futures = [more_parallel_price_option_chain.remote(symbol) for symbol in symbol_list]
results = ray.get(futures)

all_symbols_t.e("Total time for all symbols: ")

print(
    """
    Total time for all symbols: 43.070964 sec
    Stats for   AMZN:  1520 options, calc'd IV for all  shocks in 35.734421 sec
    Stats for   AAPL:  1843 options, calc'd IV for all  shocks in 43.060711 sec
    """
)
```
