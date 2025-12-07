import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# --- Market / sector tickers ---
DEFAULT_TICKERS = ["SBUX", "SPY", "XLY", "VIX", "MCD"]

def get_weekly_prices(tickers=None, start="2018-01-01", end=None):
    """Download weekly adjusted close prices for given tickers."""
    if tickers is None:
        tickers = DEFAULT_TICKERS

    df = yf.download(tickers, start=start, end=end, interval="1wk", auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers)
    df.columns = tickers
    return df

def save_prices(df, raw_dir=RAW_DIR):
    """Save individual ticker CSVs in raw dir."""
    for ticker in df.columns:
        path = os.path.join(raw_dir, f"{ticker}_weekly.csv")
        df[[ticker]].to_csv(path)
        print(f"Saved {path}")
    return [os.path.join(raw_dir, f"{ticker}_weekly.csv") for ticker in df.columns]

# --- Macro data via FRED ---
def get_fred_series(series_ids, api_key, start="2018-01-01"):
    """
    Fetch weekly macroeconomic series from FRED.
    
    Args:
        series_ids: dict {name: fred_id}
        api_key: FRED API key
        start: start date
    Returns:
        dict of DataFrames {name: df}
    """
    fred = Fred(api_key=api_key)
    series_data = {}
    for name, fred_id in series_ids.items():
        df = fred.get_series(fred_id, observation_start=start)
        df = df.to_frame(name=name)
        df.index = pd.to_datetime(df.index)
        df = df.resample("W-MON").last()  # align to weekly
        series_data[name] = df
        path = os.path.join(RAW_DIR, f"{name}_weekly.csv")
        df.to_csv(path)
        print(f"Saved {path}")
    return series_data

# --- Google Trends from Monthly Data Download ---
def gt_monthly_to_weekly(filename: str):
    """
    Convert a monthly Google Trends CSV to weekly frequency.
    The last day of each month is forward-filled to the weeks until the next month.
    
    Args:
        filename: CSV file in RAW_DIR with columns 'Month' and interest values.
    """
    # Load CSV
    df = pd.read_csv(os.path.join(RAW_DIR, filename), skiprows=2)
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.rename(columns={"Month": "Date"})
    df = df.set_index('Date')
    
    # Ensure the interest column is named consistently
    interest_col = df.columns[0]
    df = df.rename(columns={interest_col: 'gt_interest'})
    
    # Resample to weekly, using forward fill to propagate monthly value to all weeks in that month
    df_weekly = df.resample("W-MON").ffill()
    
    # Optionally, you can normalize or scale here if needed
    path = os.path.join(RAW_DIR, f"{filename[:-4]}_weekly.csv")
    df_weekly.to_csv(path)
    print(f"Saved {path}")
