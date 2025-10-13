# Goal: Convert raw WISDM readings into a feature table compatible with UCI HAR.

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats, signal
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load the raw WISDM dataset

def load_wisdm_data(raw_folder_path):

    print(f"Loading WISDM data from {raw_folder_path}...")

    # Ensure folder path exists
    raw_folder_path = Path(raw_folder_path)
    if not raw_folder_path.exists() or not raw_folder_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {raw_folder_path}")

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Expected column names for WISDM accelerometer data
    columns = ['user_id', 'activity', 'timestamp', 'x', 'y', 'z']

    # Define subfolders to process
    subfolders = [
        'phone/accel',
        'phone/gyro',
        'watch/accel',
        'watch/gyro'
    ]

    

# Iterate through each subfolder
    for subfolder in subfolders:
        folder_path = raw_folder_path / subfolder
        if not folder_path.exists():
            print(f"Warning: Subfolder {folder_path} does not exist. Skipping.")
            continue

        # Iterate through all .txt files in the subfolder
        for file_path in folder_path.glob('*.txt'):
            print(f"Processing file: {file_path.name} in {subfolder}")
            try:
                # Read the semicolon-separated text file
                df = pd.read_csv(file_path, sep=',', header=None, names=columns)

                # Remove any extra spaces in column names
                df.columns = df.columns.str.strip()

                # Remove trailing semicolon from the 'z' column, if present
                if df['z'].dtype == object:
                    df['z'] = df['z'].str.replace(';', '').astype(float)

                # Add a 'source' column to indicate data origin
                df['source'] = subfolder.replace('/', '_')  # e.g., 'phone_accel'

                # Append to list of DataFrames
                dataframes.append(df)
                print(f" Loaded {len(df)} rows from {file_path.name}")

            except Exception as e:
                print(f"Error reading {file_path.name} in {subfolder}: {e}")

    # Combine all DataFrames
    if not dataframes:
        raise ValueError("No valid data files were loaded.")
    
    combined_df = pd.concat(dataframes, ignore_index=True)

    print(f"\n Total rows loaded: {len(combined_df)}")
    print(f" Columns: {list(combined_df.columns)}")
    print(f" Sample data:\n{combined_df.head()}\n")
    
    return combined_df



if __name__ == "__main__":
    raw_folder_path = r"C:\Users\tomin\OneDrive\Machine Learning\Extension_research\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\wisdm-dataset\raw"
    

    try:
        # Load WISDM data
        wisdm_df = load_wisdm_data(raw_folder_path)

        # Additional checks
        print(f" Unique sources: {wisdm_df['source'].unique()}")
        print(f" Unique activities: {wisdm_df['activity'].unique()}")
        print(f" Unique user IDs: {wisdm_df['user_id'].unique()}")
        print(f" Data types:\n{wisdm_df.dtypes}")

    except Exception as e:
        print(f"Test failed: {e}")
