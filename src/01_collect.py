import os
import yfinance as yf
import pandas as pd

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

tickers = ["SBUX", "SPY"]
start = "2018-01-01"
end = None  # today

def get_weekly_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, interval="1wk", auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers)
    df.columns = tickers
    return df

prices = get_weekly_prices(tickers, start, end)

for ticker in tickers:
    path = os.path.join(RAW_DIR, f"{ticker}_weekly_prices.csv")
    prices[[ticker]].to_csv(path)
    print(f"Saved {path}")
