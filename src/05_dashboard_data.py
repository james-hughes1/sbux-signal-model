import os
import json
import pandas as pd
from sbux_model.io import read_table, save_table

CONFIG_PATH = "src/config/dashboard_config.json"

# Load JSON config
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

stage_name = config["stage_name"]            # e.g. "dashboard"
model_stage = config["model_stage"]          # e.g. "model"
preproc_stage = config["preproc_stage"]      # e.g. "preprocessing"

# Columns from preprocessing CSV to include
preproc_cols = config.get("preproc_columns", [])

# --- Read latest tables ---
model_df = read_table(stage_name=model_stage, config=config.get("input_model"))
preproc_df = read_table(stage_name=preproc_stage, config=config.get("input_preproc"))

# Select columns from preprocessing
if preproc_cols:
    preproc_df = preproc_df[preproc_cols]
else:
    preproc_df = preproc_df.copy()  # keep all if empty

# Remove any columns that overlap with model_df
preproc_df = preproc_df.loc[:, ~preproc_df.columns.isin(model_df.columns)]

# Merge on index (Date)
dashboard_df = model_df.join(preproc_df, how="left")

# Save dashboard-ready CSV
output_path = save_table(dashboard_df, stage_name, config.get("output"))
print(f"Saved dashboard table → {output_path}")
