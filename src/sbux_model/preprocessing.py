import pandas as pd
import os

def resample_weekly_last(df):
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.resample("W-MON").last()
    return df

def resample_weekly_mean(df):
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.resample("W-MON").mean()
    return df

def resample_weekly_ffill(df):
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].ffill()

    return df

def impute_low_freq_ffill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute low-frequency series after resampling to higher frequency (weekly).

    Args:
        df (pd.DataFrame): DataFrame indexed by datetime.
    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df_imputed = df.copy()
    
    for col in df_imputed.columns:
        if df_imputed[col].isna().any():
            df_imputed[col] = df_imputed[col].ffill()
    return df_imputed

