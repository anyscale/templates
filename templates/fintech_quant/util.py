
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

# A failed pricing is reported as NaN, never as a number. See the comment on the
# `except RuntimeError` in get_iv() for why 0.0 was the wrong sentinel.
PRICING_FAILED = float("nan")


def _pricing_columns(df):
    """The implied_volatility column plus every scenario NPV column (s1_npv, s2_npv, ...)."""
    return [c for c in df.columns
            if c == "implied_volatility" or re.fullmatch(r"s\d+_npv", str(c))]


def count_priced(df, columns=None):
    """How many rows priced in every one of `columns`.

    get_iv()/get_npv() return NaN when QuantLib can't solve, so a row that
    failed is countable rather than invisible. Use this to report "N of M".

    By default a row counts only if its implied volatility *and* every
    scenario NPV came back. Checking the IV alone is not enough: get_npv()
    re-solves an implied vol at the shocked spot, so a contract whose base IV
    priced can still fail a scenario (see the comment in get_npv()).
    """
    if columns is None:
        columns = _pricing_columns(df)
    if not columns or any(c not in df for c in columns):
        return 0
    return int(df[columns].notna().all(axis=1).sum())


def _stats_label(symbol, priced, total):
    # Report priced-vs-total, not just total. A run that prices 900 of 1843
    # contracts is not the same result as one that prices all 1843, and the
    # summary line is the only place a user would notice the difference.
    failed = total - priced
    suffix = f" ({failed} FAILED to price -- see the per-contract warnings)" if failed else ""
    return (
        f"Stats for {symbol:>6}: {priced:>5} of {total:>5} options priced{suffix}, "
        "calc'd IV for all  shocks in "
    )


# Common, long print string, pulled out of notebook
def get_symbols_stat_print(symbol, df):
    # "Priced" means the IV and every scenario NPV -- see count_priced().
    return _stats_label(symbol, count_priced(df), len(df))


def pricing_summary(symbol, df, path, seconds):
    """What a Ray pricing task returns: the CSV path plus its priced/total counts.

    The counts travel back in the return value so the driver can print them.
    A print() inside a task reaches the notebook only through Ray's worker log
    forwarding, which can drop lines: on CI runs the AAPL summary never arrived.
    """
    return {
        "symbol": symbol,
        "path": path,
        "priced": count_priced(df),
        "total": len(df),
        "seconds": seconds,
    }


def print_pricing_summary(summaries):
    """Print one "N of M options priced" line per pricing_summary(), on the driver."""
    for s in summaries:
        print(f"{_stats_label(s['symbol'], s['priced'], s['total'])}{s['seconds']:.6f} sec")

def get_iv(option):
    """
    Get implied volatility for a given option

    Returns NaN -- never 0.0 -- if QuantLib cannot solve for the vol, so a
    failed calculation stays distinguishable from a genuine zero.
    """
    risk_free_rate = 0.0425

    volatility = 0.001
    option_price = option['last_price']
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
        return float(implied_volatility)
    except RuntimeError as exc:
        # QuantLib's SWIG bindings surface every pricing/solver error as
        # RuntimeError -- e.g. "root not bracketed" when the quoted price
        # implies a vol outside the [1e-8, 4.0] search bounds. Catch that, and
        # only that.
        #
        # This was `except: return 0.0`, which is wrong twice over:
        #
        #   1. A bare `except` also swallows KeyboardInterrupt and SystemExit
        #      (so the job ignores Ctrl-C and shutdown) and hides real bugs --
        #      a TypeError from a None `last_price` -- as if they were market
        #      data the model couldn't fit. None is what get_options_chain()
        #      stores when yfinance's chain has no `lastPrice` column, so a
        #      renamed upstream column arrives here as that TypeError, and
        #      used to price every contract at 0.0. (A KeyError never reached
        #      this handler: the option[...] lookups are above the `try`.)
        #   2. 0.0 is a legal-looking volatility. A contract that failed to
        #      price became indistinguishable from one that priced at zero, so
        #      the run emitted a full-looking result set partly made of
        #      failures and still reported success. Nothing downstream -- the
        #      CSV, a mean, a risk number -- could tell the difference.
        #
        # NaN cannot be mistaken for a price, it stays visible through
        # downstream arithmetic, and count_priced() turns it into a number the
        # run can report.
        log.warning(
            "implied volatility failed for %s (%s strike=%s exp=%s last_price=%s): %s",
            option.get("contractSymbol", "<unknown contract>"),
            option.get("type"),
            option.get("strike"),
            option.get("expiration"),
            option_price,
            exc,
        )
        return PRICING_FAILED

def get_npv(option, underlying_price, implied_volatility):
    """
    Get NPV for a given option

    Returns NaN -- never 0.0 -- if the contract cannot be priced, so a failed
    valuation is not silently aggregated as a zero-value position.
    """
    risk_free_rate = 0.0425

    volatility = float(implied_volatility)

    # A non-finite vol means get_iv() already failed on this contract and
    # already logged it. Propagate that failure rather than spend a 1000-step
    # binomial solve to fail again and log the same contract twice.
    if not math.isfinite(volatility):
        return PRICING_FAILED

    spot_price = underlying_price
    option_price = option['last_price']
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
        implied_volatility = american_option.impliedVolatility(
            option_price, bsm_process, 1e-4, 1000, 1e-8, 4.0
        )
        return american_option.NPV()
    except RuntimeError as exc:
        # Same reasoning as get_iv(): QuantLib raises RuntimeError for solver
        # and engine failures ("root not bracketed", "negative probability"),
        # and a bare `except: return 0.0` turned an unpriced contract into a
        # zero-valued one -- a number that flows into a scenario NPV total
        # without ever looking wrong. NaN can't be mistaken for a valuation.
        #
        # The solve that fails here is usually the impliedVolatility() call
        # above, not NPV(): it re-solves last_price at the *shocked* spot, and
        # a quote the model can't reach at that spot -- typically a put whose
        # intrinsic value there exceeds its quote -- has no root to find. So a
        # contract whose base IV priced can still fail a scenario, which is
        # why count_priced() checks every sN_npv column.
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
