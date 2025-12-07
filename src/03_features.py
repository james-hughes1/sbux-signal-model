import os
import json
import pandas as pd
from sbux_model.io import read_table, save_table

CONFIG_PATH = "src/config/features_config.json"

# Load JSON config
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

stage_name = config["stage_name"]              # e.g. "features"
input_stage = config["input_stage"]            # e.g. "preprocessing"
feature_defs = config["features"]              # dict of feature specs

# Load input table (latest unless filename override)
df = read_table(stage_name=input_stage, config=config.get("input"))

# ----------------------------------------------------
# 1. Compute RETURNS and TARGET (Expected Excess Return)
# ----------------------------------------------------

# Forward 1-period SBUX return
df["sbux_ret_fwd_1"] = df["SBUX"].pct_change().shift(-1)

# Forward 1-period SPY return
df["spy_ret_fwd_1"] = df["SPY"].pct_change().shift(-1)

# Expected excess return target:
#   (Future SBUX return) – (Future SPY market return)
df["excess_ret_fwd_1"] = df["sbux_ret_fwd_1"] - df["spy_ret_fwd_1"]


# ----------------------------------------------------
# 2. Feature Engineering from JSON specs
# ----------------------------------------------------
for feat_name, feat_cfg in feature_defs.items():
    ftype = feat_cfg["type"]
    col = feat_cfg["column"]

    if ftype == "lag":
        df[f"{col}_lag_{feat_cfg['lag']}"] = df[col].shift(feat_cfg["lag"])

    elif ftype == "rolling_mean":
        window = feat_cfg["window"]
        df[f"{col}_rm_{window}"] = df[col].rolling(window).mean()

    elif ftype == "zscore":
        window = feat_cfg["window"]
        mean = df[col].rolling(window).mean()
        std = df[col].rolling(window).std()
        df[f"{col}_z_{window}"] = (df[col] - mean) / std

    elif ftype == "momentum":
        window = feat_cfg["window"]
        # Proper momentum: percent change over window
        df[f"{col}_mom_{window}"] = df[col].pct_change(window)

    else:
        raise ValueError(f"Unknown feature type: {ftype}")


# ----------------------------------------------------
# 3. Drop NA from rolling/lag + final rows with no target
# ----------------------------------------------------
df.dropna(inplace=True)

# ----------------------------------------------------
# 4. Save output
# ----------------------------------------------------
output_path = save_table(df, stage_name, config.get("output"))
print(f"Saved features table → {output_path}")
