import pandas as pd

def compute_forward_returns(df, col, periods=1):
    """Compute forward return for a column"""
    return df[col].pct_change(periods=periods).shift(-periods)

def compute_excess_return(df, asset_col="SBUX", benchmark_col="SPY"):
    """Expected excess return target: future asset return minus future benchmark return"""
    df[f"{asset_col}_ret_fwd_1"] = compute_forward_returns(df, asset_col)
    df[f"{benchmark_col}_ret_fwd_1"] = compute_forward_returns(df, benchmark_col)
    df["excess_ret_fwd_1"] = df[f"{asset_col}_ret_fwd_1"] - df[f"{benchmark_col}_ret_fwd_1"]
    return df

def apply_feature(df, feat_cfg):
    """Apply a single feature transformation"""
    col = feat_cfg["column"]
    ftype = feat_cfg["type"]

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
        df[f"{col}_mom_{window}"] = df[col].pct_change(window)

    else:
        raise ValueError(f"Unknown feature type: {ftype}")
