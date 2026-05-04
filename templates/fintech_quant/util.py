
import time
from datetime import datetime

import os
import pandas as pd
import numpy as np
import QuantLib as ql
import yfinance as yf

# Common, long print string, pulled out of notebook
def get_symbols_stat_print(symbol, df):
    return f"Stats for {symbol:>6}: {len(df):>5} options, calc'd IV for all  shocks in "

def get_iv(option):
    """
    Get implied volatility for a given option
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
    except:
        return 0.0

def get_npv(option, underlying_price, implied_volatility):
    """
    Get NPV for a given option
    """
    risk_free_rate = 0.0425

    volatility = float(implied_volatility)
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
    except:
        return 0.0

def get_options_chain(symbol):
    """
    Get options chain data for a stock
    """
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
                'error': f'Error retrieving options data: {str(e)}'
            }

        return options_chain

    except Exception as e:
        print(f"Error in get_options_chain: {str(e)}")
        return {'error': str(e)}

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

    def e(self, label="Elapsed time"):
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        duration = time.perf_counter() - self.start_time
        print(f"{label}{duration:.6f} sec")
        self.start_time = None  # reset for reuse
