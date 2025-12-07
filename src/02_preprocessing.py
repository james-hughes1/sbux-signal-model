import os
import json
import pandas as pd
from sbux_model.io import save_table
from sbux_model import preprocessing as pp

# Load config
CONFIG_PATH = "src/config/preprocessing_config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

OUTPUT_DIR = f"data/{config['stage_name']}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

preprocessed_dfs = []

for key, raw_info in config["raw_files"].items():
    path = raw_info["filename"] if isinstance(raw_info, dict) else raw_info
    print(path)
    df = pd.read_csv(path, parse_dates=True)

    # Apply preprocessing function if specified
    if isinstance(raw_info, dict) and "preprocessing" in raw_info:
        func_name = raw_info["preprocessing"]
        func = getattr(pp, func_name)
        df = func(df)
    else:
        # Default: resample last weekly
        df = pp.resample_weekly_last(df)

    preprocessed_dfs.append(df)

# Merge all tables on Date
preprocessed_df = pd.concat(preprocessed_dfs, axis=1)
preprocessed_df = preprocessed_df.loc[~preprocessed_df.index.duplicated(keep='last')]
preprocessed_df.dropna(inplace=True)

# Save
output_path = save_table(preprocessed_df, config["stage_name"], config.get("output"))
print(f"Saved preprocessed table → {output_path}")
