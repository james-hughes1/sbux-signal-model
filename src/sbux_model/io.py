import os
import pandas as pd
from datetime import datetime

def save_table(df: pd.DataFrame, stage_name: str, config: dict = None):
    """
    Save a DataFrame to a stage folder with a timestamp, unless overridden by config filename.

    Args:
        df (pd.DataFrame): Table to save
        stage_dir (str): Directory for this stage
        stage_name (str): Name of the stage, used as default filename
        config (dict, optional): If contains 'filename', use it instead of timestamped default
    """
    stage_dir = "data/" + stage_name

    os.makedirs(stage_dir, exist_ok=True)
    if config and config["filename"]:
        filename = config["filename"]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stage_name}_{timestamp}.csv"
    path = os.path.join(stage_dir, filename)
    df.to_csv(path, index=True)
    return path


def read_table(stage_name: str, config: dict = None):
    """
    Read the latest file in a stage folder, unless overridden by config filename.

    Args:
        stage_dir (str): Directory for this stage
        stage_name (str): Name of the stage
        config (dict, optional): If contains 'filename', use it instead
    """
    stage_dir = "data/" + stage_name

    if config and config["filename"]:
        filename = config["filename"]
        path = os.path.join(stage_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")
        return pd.read_csv(path, parse_dates=True, index_col=0)

    # Find latest file in folder matching stage_name prefix
    all_files = [f for f in os.listdir(stage_dir) if f.startswith(stage_name) and f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No files found in {stage_dir} starting with {stage_name}")
    latest_file = max(all_files)  # timestamped files sort lexicographically
    path = os.path.join(stage_dir, latest_file)
    return pd.read_csv(path, parse_dates=True, index_col=0)
