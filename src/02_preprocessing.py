import os
import json
import pandas as pd
from sbux_model.io import save_table
from sbux_model.data import unify_gt_csvs

# Load config
CONFIG_PATH = "src/config/preprocessing_config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Directories
RAW_DIR = "data/raw"
OUTPUT_DIR = f"data/{config['stage_name']}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Unify Google Trends ---
gt_files = config["raw_files"]["gt"]
gt_df = unify_gt_csvs(gt_files)

# --- Load raw stock data ---
sbux_df = pd.read_csv(config["raw_files"]["SBUX"], parse_dates=["Date"], index_col="Date")
spy_df = pd.read_csv(config["raw_files"]["SPY"], parse_dates=["Date"], index_col="Date")

# Align weekly dates if needed
sbux_df = sbux_df.resample("W-MON").last()
spy_df = spy_df.resample("W-MON").last()
gt_df = gt_df.resample("W-MON").mean()

# Merge into table
preprocessed_df = sbux_df.join(spy_df, how="inner")
preprocessed_df = preprocessed_df.join(gt_df, how="inner")
preprocessed_df.dropna(inplace=True)

# Save
output_path = save_table(preprocessed_df, config["stage_name"], config.get("output"))
print(f"Saved preprocessed table → {output_path}")
