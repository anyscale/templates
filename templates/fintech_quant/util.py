
import logging
import math
import re
import time
from datetime import datetime

import os
import tempfile
import pandas as pd
import numpy as np
import QuantLib as ql
import yfinance as yf

log = logging.getLogger(__name__)

# What get_iv() and get_npv() return when QuantLib can't price a contract.
# NaN rather than 0.0, because 0.0 passes for a real vol or NPV; see get_iv().
PRICING_FAILED = float("nan")

# iv_source values for contracts get_iv() skips rather than solves.
NO_QUOTE = "no_quote"        # no usable bid/ask mid or last_price
BELOW_BOUND = "below_bound"  # price below the no-arbitrage lower bound
SKIPPED = (NO_QUOTE, BELOW_BOUND)


def _pricing_columns(df):
    """The implied_volatility column plus every scenario NPV column (s1_npv, s2_npv, ...)."""
    return [c for c in df.columns
            if c == "implied_volatility" or re.fullmatch(r"s\d+_npv", str(c))]


def _priced(df, columns=None):
    if columns is None:
        columns = _pricing_columns(df)
    if not columns or any(c not in df for c in columns):
        return pd.Series(False, index=df.index)
    return df[columns].notna().all(axis=1)


def count_priced(df, columns=None):
    """Count the rows with a value in every one of `columns`: the N in "N of M".

    By default that is implied_volatility and every scenario NPV, so a contract
    counts as priced only if all of them solved. A contract whose base IV
    solved can still fail a scenario; see get_npv().
    """
    return int(_priced(df, columns).sum())


def count_skipped(df):
    """Count the contracts get_iv() skipped for a bad quote (iv_source in SKIPPED)."""
    if "iv_source" not in df:
        return 0
    return int(df["iv_source"].isin(SKIPPED).sum())


def count_from_last(df):
    """Count the priced contracts whose IV came from last_price, not a bid/ask mid."""
    if "iv_source" not in df:
        return 0
    return int((_priced(df) & (df["iv_source"] == "last")).sum())


def _stats_label(symbol, priced, skipped, total, from_last):
    # One label for the serial and Ray paths. The caller appends the seconds.
    # Whatever is neither priced nor skipped failed in QuantLib.
    failed = total - priced - skipped
    unpriced = [f"{skipped} skipped"] if skipped else []
    if failed:
        unpriced.append(f"{failed} FAILED, see warnings")
    notes = [f"{from_last} from last_price"] if from_last else []
    if unpriced:
        notes.append(", ".join(unpriced))
    suffix = f" ({'; '.join(notes)})" if notes else ""
    return f"Stats for {symbol:>6}: {priced:>5} of {total:>5} options priced{suffix} in "


# Common, long print string, pulled out of notebook
def get_symbols_stat_print(symbol, df):
    # "Priced" means the IV and every scenario NPV solved; see count_priced().
    return _stats_label(symbol, count_priced(df), count_skipped(df), len(df), count_from_last(df))


def pricing_summary(symbol, df, path, seconds):
    """What a Ray pricing task returns: the CSV path plus its pricing counts.

    The driver prints them with print_pricing_summary(). A print() inside the
    task would reach the notebook through Ray's log forwarding, which can
    deliver it late or not at all (on CI, the AAPL line never arrived).
    """
    return {
        "symbol": symbol,
        "path": path,
        "priced": count_priced(df),
        "skipped": count_skipped(df),
        "from_last": count_from_last(df),
        "total": len(df),
        "seconds": seconds,
    }


def print_pricing_summary(summaries):
    """Print one "N of M options priced" line per pricing_summary(), on the driver."""
    for s in summaries:
        label = _stats_label(s['symbol'], s['priced'], s['skipped'], s['total'], s['from_last'])
        print(f"{label}{s['seconds']:.6f} sec")


def _as_float(x):
    return float("nan") if x is None else float(x)


def _iv_price(option):
    """The price get_iv() solves from, and its iv_source: "mid", "last" or NO_QUOTE."""
    bid, ask, last = (_as_float(option.get(k)) for k in ("bid", "ask", "last_price"))
    if bid > 0 and ask > 0 and ask >= bid:  # all False for NaN
        return (bid + ask) / 2, "mid"
    if last > 0:
        return last, "last"
    return PRICING_FAILED, NO_QUOTE


def _iv_result(price, source, implied_volatility):
    # One row of get_iv() output; df.apply(get_iv, axis=1) makes these columns.
    return pd.Series({
        "iv_price": price,
        "iv_source": source,
        "implied_volatility": implied_volatility,
    })


def get_iv(option):
    """
    Implied vol of an American option, solved from its bid/ask mid.

    Returns a Series, which df.apply(get_iv, axis=1) turns into three columns:
      iv_price            the option price the vol is solved from
      iv_source           where iv_price came from:
        "mid"             (bid + ask) / 2, when bid > 0, ask > 0 and ask >= bid
        "last"            last_price, when there is no such two-sided quote
                          (typically a zero bid on a far-OTM or illiquid strike)
        NO_QUOTE          skipped: no usable mid or last_price
        BELOW_BOUND       skipped: iv_price is below the no-arbitrage lower
                          bound, so no vol reproduces it
      implied_volatility  the vol; NaN if skipped, or if QuantLib can't solve
                          it (logged, and counted as FAILED)
    """
    risk_free_rate = 0.0425

    volatility = 0.001
    option_price, source = _iv_price(option)
    if source == NO_QUOTE:
        return _iv_result(option_price, source, PRICING_FAILED)
    dividend_yield = float(option['dividend_yield'])
    strike_price = float(option['strike'])
    spot_price = float(option['underlying_price'])
    days_to_maturity = (datetime.strptime(option['expiration'], '%Y-%m-%d') - datetime.now()).days
    option_type = ql.Option.Call if option['type'] == 'call' else ql.Option.Put

    calendar = ql.NullCalendar()
    day_count = ql.Actual360()
    today = ql.Date().todaysDate()

    ql.Settings.instance().evaluationDate = today
    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, risk_free_rate, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, dividend_yield, day_count)
    )
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))

    expiration_date = today + ql.Period(days_to_maturity, ql.Days)

    # No-arbitrage lower bound for an American option under the model's r and
    # q: the larger of immediate exercise and the European bound. No vol
    # reproduces a price below it. That is a data problem, typically a stale
    # trade or quote on a deep-ITM contract, not a solver failure: skip it.
    disc_r = risk_free_ts.discount(expiration_date)
    disc_q = dividend_ts.discount(expiration_date)
    if option_type == ql.Option.Call:
        lower_bound = max(spot_price - strike_price, spot_price * disc_q - strike_price * disc_r, 0.0)
    else:
        lower_bound = max(strike_price - spot_price, strike_price * disc_r - spot_price * disc_q, 0.0)
    if option_price < lower_bound:
        return _iv_result(option_price, BELOW_BOUND, PRICING_FAILED)

    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.AmericanExercise(today, expiration_date)
    american_option = ql.VanillaOption(payoff, exercise)

    volatility_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, volatility, day_count)
    )

    bsm_process = ql.BlackScholesMertonProcess(
        spot_handle, dividend_ts, risk_free_ts, volatility_handle
    )
    engine = ql.BinomialVanillaEngine(bsm_process, "crr", 1000)
    american_option.setPricingEngine(engine)

    try:
        implied_volatility = american_option.impliedVolatility(
            option_price, bsm_process, 1e-4, 1000, 1e-8, 4.0
        )
        return _iv_result(option_price, source, float(implied_volatility))
    except RuntimeError as exc:
        # QuantLib raises RuntimeError for solver and pricing errors, such as
        # "root not bracketed" when the price implies a vol outside the
        # [1e-8, 4.0] search range. With sub-bound prices skipped above, that
        # leaves a price above the 400%-vol value, or one within the solver's
        # grid error of the bound. Catch only RuntimeError. A bare except would
        # also swallow Ctrl-C and real bugs.
        #
        # Return NaN rather than 0.0. A zero passes for a real vol in the CSV
        # and in any aggregate; NaN marks the contract as failed, and
        # count_priced() counts it. pandas sum() and mean() skip NaN by
        # default, so check the priced count before aggregating.
        log.warning(
            "implied volatility failed for %s (%s strike=%s exp=%s %s price=%s): %s",
            option.get("contractSymbol", "<unknown contract>"),
            option.get("type"),
            option.get("strike"),
            option.get("expiration"),
            source,
            option_price,
            exc,
        )
        return _iv_result(option_price, source, PRICING_FAILED)

def get_npv(option, underlying_price, implied_volatility):
    """
    NPV of an American option at the given spot and vol.

    The scenarios pass a shocked spot and the contract's base implied vol plus
    a vol shock. That is sticky-strike: each strike keeps its own vol when spot
    moves, so nothing is re-solved at the shocked spot.

    Returns NaN (PRICING_FAILED) if the engine fails, so a failed valuation
    stays distinguishable from a zero-value position.
    """
    risk_free_rate = 0.0425

    volatility = float(implied_volatility)

    # A NaN vol means get_iv() skipped or failed this contract, and it is
    # already counted. Skip the binomial solve so a failure isn't logged twice.
    if not math.isfinite(volatility):
        return PRICING_FAILED

    spot_price = underlying_price
    dividend_yield = option['dividend_yield']
    strike_price = option['strike']
    days_to_maturity = (datetime.strptime(option['expiration'], '%Y-%m-%d') - datetime.now()).days
    option_type = ql.Option.Call if option['type'] == 'call' else ql.Option.Put

    calendar = ql.NullCalendar()
    day_count = ql.Actual360()
    today = ql.Date().todaysDate()

    ql.Settings.instance().evaluationDate = today
    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, risk_free_rate, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, dividend_yield, day_count)
    )
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))

    expiration_date = today + ql.Period(days_to_maturity, ql.Days)
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.AmericanExercise(today, expiration_date)
    american_option = ql.VanillaOption(payoff, exercise)

    volatility_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, volatility, day_count)
    )

    bsm_process = ql.BlackScholesMertonProcess(
        spot_handle, dividend_ts, risk_free_ts, volatility_handle
    )
    engine = ql.BinomialVanillaEngine(bsm_process, "crr", 1000)
    american_option.setPricingEngine(engine)

    try:
        return american_option.NPV()
    except RuntimeError as exc:
        # Same handling as get_iv(): QuantLib engine errors (such as the
        # binomial tree's "negative probability") become NaN, and
        # count_priced() checks every sN_npv column for them.
        log.warning(
            "NPV failed for %s (%s strike=%s exp=%s spot=%s vol=%s): %s",
            option.get("contractSymbol", "<unknown contract>"),
            option.get("type"),
            option.get("strike"),
            option.get("expiration"),
            spot_price,
            volatility,
            exc,
        )
        return PRICING_FAILED

_yf_cache_isolated = False

def isolate_yf_cache():
    """Give this process its own yfinance cache directory.

    yfinance keeps cookies and timezones in one on-disk SQLite file. These calls run
    concurrently across Ray workers, and sharing that file makes them collide
    ("UNIQUE constraint failed: _cookieschema.strategy"), which fails the fetch.
    """
    global _yf_cache_isolated
    if _yf_cache_isolated:
        return
    setter = getattr(yf, "set_tz_cache_location", None)
    if setter:  # not present on every yfinance release
        setter(os.path.join(tempfile.gettempdir(), f"yf-cache-{os.getpid()}"))
    _yf_cache_isolated = True

def get_options_chain(symbol):
    """
    Get options chain data for a stock

    Always returns a dict with an 'options' list. On failure that list is empty and
    'error' says why -- callers can read `chain['options']` unconditionally, and a
    fetch failure shows up as its own message rather than a KeyError on a worker.
    """
    isolate_yf_cache()
    try:
        # Get ticker data
        ticker = yf.Ticker(symbol)

        # Get current info
        ticker_info = ticker.info
        current_price = ticker_info['currentPrice']
        dividend_yield = ticker_info['trailingAnnualDividendYield']

        # Get options data
        options_chain = {
            'symbol': symbol,
            'current_price': current_price,
            'options': [],
        }

        # Get options expirations
        try:
            expirations = ticker.options
            if not expirations:
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'options': [],
                    'error': 'No options data available'
                }

            # filter out near term options
            expirations = [exp for exp in expirations if
                                       (datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days > 30]
            # Sort expirations by date (nearest first)
            filtered_expirations = sorted(expirations, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))


            # Process each expiration date
            for expiration in filtered_expirations:
                # Get option chain for this expiration
                opt = ticker.option_chain(expiration)

                # Process calls if requested
                if not opt.calls.empty:
                    opt_calls = opt.calls

                    # Convert to list of dictionaries
                    for _, row in opt_calls.iterrows():
                        call_data = {
                            'contractSymbol': row['contractSymbol'],
                            'type': 'call',
                            'dividend_yield': dividend_yield,
                            'strike': float(row['strike']),
                            'underlying_price': current_price,
                            'expiration': expiration,
                            'last_price': float(row['lastPrice']) if 'lastPrice' in row else None,
                            'bid': float(row['bid']) if 'bid' in row else None,
                            'ask': float(row['ask']) if 'ask' in row else None,
                            'volume': int(row['volume']) if 'volume' in row and not np.isnan(row['volume']) else 0
                        }

                        options_chain['options'].append(call_data)

                    opt_puts = opt.puts

                    # Convert to list of dictionaries
                    for _, row in opt_puts.iterrows():
                        put_data = {
                            'contractSymbol': row['contractSymbol'],
                            'type': 'put',
                            'dividend_yield': dividend_yield,
                            'strike': float(row['strike']),
                            'underlying_price': current_price,
                            'expiration': expiration,
                            'last_price': float(row['lastPrice']) if 'lastPrice' in row else None,
                            'bid': float(row['bid']) if 'bid' in row else None,
                            'ask': float(row['ask']) if 'ask' in row else None,
                            'volume': int(row['volume']) if 'volume' in row and not np.isnan(row['volume']) else 0
                        }

                        options_chain['options'].append(put_data)
        except Exception as e:
            print(f"Error getting options data: {str(e)}")
            return {
                'symbol': symbol,
                'current_price': current_price,
                'options': [],
                'error': f'Error retrieving options data: {str(e)}'
            }

        return options_chain

    except Exception as e:
        print(f"Error in get_options_chain: {str(e)}")
        return {'symbol': symbol, 'options': [], 'error': str(e)}

# NOTE:
#   Saving to csv isn't completely necessary for the demo, but
#   it is included to demonstrate that it is very quick, in case someone asks.
#   Can use it as opportunity to discuss shared storage
def save_csv(data: pd.DataFrame, symbol: str, dir="quantlib"):
    path = f'/mnt/shared_storage/{dir}'
    # makes the directory, if it already exists it does nothing except suppresses FileExistsError
    os.makedirs(path, exist_ok=True)
    # e.g., /mnt/shared_storage/quantlib/AAPL-output.csv
    new_file_path = f'{path}/{symbol}-output.csv'
    data.to_csv(new_file_path, index=False)
    return new_file_path

class FuncTimer:
    def __init__(self,
                 s=True # start time when instantiated
                ):
        self.start_time = None
        if s:
            self.s()

    def s(self):
        self.start_time = time.perf_counter()

    def elapsed(self):
        """Seconds since s(), without printing or resetting."""
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        return time.perf_counter() - self.start_time

    def e(self, label="Elapsed time"):
        duration = self.elapsed()
        print(f"{label}{duration:.6f} sec")
        self.start_time = None  # reset for reuse
