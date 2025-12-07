import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path

def unify_gt_csvs(files):
    """
    Merge multiple Google Trends CSVs into a single series, scaling overlaps
    so that all earlier data aligns with later reference data.
    
    Args:
        files (list): List of CSV file paths, ordered from newest to oldest.
        output_file (str): Path to save the unified CSV.
    """
    # --- Read the latest file first as reference ---
    ref = pd.read_csv(files[0], skiprows=2)
    ref['Week'] = pd.to_datetime(ref['Week'])
    ref = ref.rename(columns={"Week": "Date"})
    ref = ref.set_index('Date')
    ref.rename(columns={ref.columns[0]: 'gt_interest'}, inplace=True)
    
    unified = ref.copy()
    
    # --- Iterate over remaining files in order ---
    for file in files[1:]:
        df = pd.read_csv(file, skiprows=2)
        df['Week'] = pd.to_datetime(df['Week'])
        df = df.rename(columns={"Week": "Date"})
        df = df.set_index('Date')
        df.rename(columns={df.columns[0]: 'gt_interest'}, inplace=True)
        
        # Find overlapping dates
        overlap_dates = df.index.intersection(unified.index)
        if len(overlap_dates) == 0:
            scale_factor = 1.0  # no overlap, no scaling
        else:
            # Average of overlapping values
            df_overlap_avg = df.loc[overlap_dates, 'gt_interest'].mean()
            ref_overlap_avg = unified.loc[overlap_dates, 'gt_interest'].mean()
            scale_factor = ref_overlap_avg / df_overlap_avg
        
        # Scale the earlier dataset
        df['gt_interest'] = df['gt_interest'] * scale_factor
        
        # Keep only dates not already in unified
        df_to_add = df.loc[~df.index.isin(unified.index)]
        
        # Prepend earlier data
        unified = pd.concat([df_to_add, unified]).sort_index()
    
    # --- Optional: rescale 0-1 for modelling ---
    unified['gt_scaled'] = (unified['gt_interest'] - unified['gt_interest'].min()) / \
                           (unified['gt_interest'].max() - unified['gt_interest'].min())
    
    return unified
