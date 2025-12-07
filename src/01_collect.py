import os
from sbux_model.collect import get_weekly_prices, save_prices, get_fred_series, gt_monthly_to_weekly
from dotenv import load_dotenv

load_dotenv()


# ---------------------------
# Market / sector prices
# ---------------------------
tickers = ["SBUX", "SPY", "XLY", "^VIX", "MCD"]
prices_df = get_weekly_prices(tickers)
save_prices(prices_df)

# ---------------------------
# Macro data via FRED
# ---------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY")  # set in .env

macro_series = {
    "10Y_treasury": "DGS10",
    "2Y_treasury": "DGS2",
    "fed_funds_rate": "FEDFUNDS",
    "CPI": "CPIAUCSL"
}

if FRED_API_KEY:
    get_fred_series(macro_series, FRED_API_KEY)
else:
    print("FRED_API_KEY not set. Macro series not downloaded.")

# ---------------------------
# Google Trends
# ---------------------------
monthly_gt_filename = "gt_starbucks_2018_2025_monthly.csv"
gt_monthly_to_weekly(monthly_gt_filename)
