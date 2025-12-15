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

    elif ftype == "diff":
        lag = feat_cfg.get("lag", 1)
        df[f"{col}_diff_{lag}"] = df[col].diff(lag)

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

    elif ftype == "lagged_alpha":
        df = add_lagged_alpha(
            df,
            alpha_col=feat_cfg.get("alpha_col", "alpha"),
            lags=feat_cfg.get("lags", [1]),
            mas=feat_cfg.get("mas", [])
        )
    
    elif ftype == "latest_pct_change":
        df = latest_pct_change(
            df,
            col,
            eps=feat_cfg.get("epsilon", 1e-8)
        )

    else:
        raise ValueError(f"Unknown feature type: {ftype}")


def compute_timevarying_beta(df, asset_col="SBUX", benchmark_col="SPY", window=52):
    """
    Compute rolling beta between asset and benchmark.
    Uses 52-week (1-year) rolling regression by default.
    """
    asset_ret = df[asset_col].pct_change()
    bench_ret = df[benchmark_col].pct_change()

    # Covariance & variance rolling windows
    cov = asset_ret.rolling(window).cov(bench_ret)
    var = bench_ret.rolling(window).var()

    df["beta_roll"] = cov / var

    return df


def compute_residual_alpha(df, asset_col="SBUX", benchmark_col="SPY", window=52):
    """
    Compute idiosyncratic alpha_t = r_asset - beta_t * r_bench
    and forward alpha (target).
    """

    # Rolling beta first
    df = compute_timevarying_beta(df, asset_col, benchmark_col, window)

    # Returns
    df["asset_ret"] = df[asset_col].pct_change()
    df["bench_ret"] = df[benchmark_col].pct_change()

    # Residual alpha_t
    df["alpha"] = df["asset_ret"] - df["beta_roll"] * df["bench_ret"]

    # Forward alpha target
    df["alpha_fwd_1"] = df["alpha"].shift(-1)

    return df


def add_lagged_alpha(df, alpha_col="alpha", lags=[1, 2, 4, 8], mas=[4, 12]):
    """
    Adds lagged residual alphas and moving-average alpha features.
    """
    for L in lags:
        df[f"{alpha_col}_lag{L}"] = df[alpha_col].shift(L)

    for M in mas:
        df[f"{alpha_col}_ma{M}"] = df[alpha_col].rolling(M).mean()

    return df


def latest_pct_change(df, column, eps=1e-8):
    """
    Latest non-zero percentage change in `column`, forward-filled.
    
    Assumes the column is forward-filled to the target frequency.
    
    Example:
        CPI, Fed Funds, macro step series.
    """
    pct = df[column].pct_change()

    # Keep only actual changes
    pct = pct.where(pct.abs() > eps)

    # Carry last change forward
    df[f"{column}_latest_pct_change"] = pct.ffill()

    return df
