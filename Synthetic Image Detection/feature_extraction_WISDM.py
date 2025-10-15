# Goal: Convert raw WISDM readings into a feature table compatible with UCI HAR.

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats, signal
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# STEP 1: LOAD WISDM RAW CSV
# ============================================================================
def load_wisdm_data(raw_folder_path):

    print("Step 1: Loading WISDM data...")

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


# ============================================================================
# STEP 2: SEGMENT INTO WINDOWS
# ============================================================================
def segment_data_into_windows(df, window_size=128, overlap=0.5):
    print("Step 2: Segmenting data into windows...")

    # Calculate how many samples to skip between windows
    step_size = int(window_size * (1 - overlap))

    windows = []

    # Group data by user to keep their data separate
    for user_id, user_group in df.groupby('user_id'):
        for activity, activity_group in user_group.groupby('activity'):

            # Extract only the sensor axes (X, Y, Z columns)
            activity_group = activity_group.reset_index(drop=True)

            # Extract only the sensor axes (X, Y, Z columns)
            sensor_data = activity_group[['x', 'y', 'z']].values

            # Slide a window across the data
            for start_idx in range(0, len(sensor_data) - window_size + 1, step_size):

                # Get the window 
                window = sensor_data[start_idx:start_idx + window_size]

                # Store the window with its label and user ID
                windows.append({
                    'window': window,
                    'activity': activity,
                    'user_id': user_id
                })


    print(f"Created {len(windows)} windows")
    print(f"Window size: {window_size} samples")
    print(f"Step size: {step_size} samples ({overlap*100}% overlap)\n")
    
    return windows

            
# ============================================================================
# STEP 3: COMPUTE STATISTICAL FEATURES FOR EACH WINDOW
# ============================================================================
def compute_features_for_window(windows):

    print("Step 3: Computing features for a sample window...")

    features = {}

    # Extract individual axes
    x = windows[:, 0]
    y = windows[:, 1]
    z = windows[:, 2]

    # ---- Basic Statistics per axis ----
    for axis_name, data in (('X', x), ('Y', y), ('Z', z)):
        features[f'{axis_name}_mean'] = np.mean(data)
        features[f'{axis_name}_std'] = np.std(data)
        features[f'{axis_name}_min'] = np.min(data)
        features[f'{axis_name}_max'] = np.max(data)
        features[f'{axis_name}_median'] = np.median(data)

        # ---- Energy ----
        # Energy = sum of squared values
        # High energy = lots of movement
        features[f'{axis_name}_energy'] = np.sum(data ** 2)
        # ---- Entropy ----
        # Entropy = measure of randomness
        hist, _ = np.histogram(data, bins=10) 
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features[f'{axis_name}_entropy'] = entropy
        # ---- Interquartile Range (IQR) ----
        # IQR = Q3 - Q1 (middle 50% spread)
        # Shows variability without being affected by extremes
        features[f'{axis_name}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)

    # ---- Signal Magnitude Area (SMA) ----
    sma = np.mean(np.sqrt(x**2 + y**2 + z**2))
    features['signal_magnitude_area'] = sma

    # ---- Pairwise Correlations ----
    features['corr_xy'] = np.corrcoef(x, y)[0, 1]
    features['corr_xz'] = np.corrcoef(x, z)[0, 1]
    features['corr_yz'] = np.corrcoef(y, z)[0, 1]

    # ---- Peak Frequency (using FFT) ----
    fft_x = np.abs(np.fft.fft(x))
    fft_y = np.abs(np.fft.fft(y))
    fft_z = np.abs(np.fft.fft(z))

    # Find the frequency with maximum power (skip DC component at index 0)
    features['peak_freq_x'] = np.argmax(fft_x[1:]) + 1
    features['peak_freq_y'] = np.argmax(fft_y[1:]) + 1
    features['peak_freq_z'] = np.argmax(fft_z[1:]) + 1

    # ---- Skewness and Kurtosis ----
    features['skew_x'] = stats.skew(x)
    features['skew_y'] = stats.skew(y)
    features['skew_z'] = stats.skew(z)
    
    features['kurt_x'] = stats.kurtosis(x)
    features['kurt_y'] = stats.kurtosis(y)
    features['kurt_z'] = stats.kurtosis(z)

    return features


# ============================================================================
# STEP 4: EXTRACT FEATURES FOR ALL WINDOWS
# ============================================================================
def extract_all_features(windows):

    print("Step 4: Computing features for all windows...")

    all_features = []

    for i, window_dict in enumerate(windows):
        # Extract features from this window
        window_features = compute_features_for_window(window_dict['window'])

        # Add the activity label and user ID
        window_features['activity'] = window_dict['activity']
        window_features['user_id'] = window_dict['user_id']

        all_features.append(window_features)

        # Print progress every 100 windows
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} / {len(windows)} windows")

    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(all_features)

    print(f"Created feature DataFrame: {features_df.shape[0]} rows X {features_df.shape[1]} columns")
    print(f"Activities: {features_df['activity'].unique()}")
    print(f"Sample feature row:\n{features_df.iloc[0]}\n")

    return features_df


# ============================================================================
# STEP 5: PREPARE FOR UCI HAR FORMAT (Optional but recommended)
# ============================================================================
def prepare_uci_har_format(features_df):
    print("Step 5: Preparing UCI HAR format...")

    # Create mapping from activity names to numeric labels
    activity_mapping = {
        activity: idx + 1 
        for idx, activity in enumerate(sorted(features_df['activity'].unique()))
    }

     # Reverse mapping for reference
    reverse_mapping = {v: k for k, v in activity_mapping.items()}
    
    print(f"Activity mapping:")
    for activity, label in sorted(activity_mapping.items(), key=lambda x: x[1]):
        print(f"    {label} = {activity}")

    # Extract X (features) - all columns except activity and user
    X = features_df.drop(columns=['activity', 'user_id'])
    
    # Extract y (labels) - convert activity names to numbers
    y = features_df['activity'].map(activity_mapping)

    subjects = features_df['user_id']
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}\n")
    
    return X, y, subjects, activity_mapping, reverse_mapping



# ============================================================================
# STEP 6: SAVE TO CSV
# ============================================================================
def save_to_csv(features_df, X, y, subjects, activity_mapping, reverse_mapping, output_dir= r"C:\Users\tomin\source\repos\Synthetic Image Detection\Synthetic Image Detection"):

    print("Step 6: Saving to CSV files...")

    os.makedirs(output_dir, exist_ok=True)

    # Save full feature dataframe (with activity and user columns)
    features_csv_path = os.path.join(output_dir, "wisdm_features_with_metadata.csv")
    features_df.to_csv(features_csv_path, index=False)
    print(f"Saved: {features_csv_path}")

    # UCI-like: space-separated, no header/index
    X.to_csv(os.path.join(output_dir, "X.txt"), sep=' ', index=False, header=False)
    y.to_csv(os.path.join(output_dir, "y.txt"), sep=' ', index=False, header=False)
    subjects.to_csv(os.path.join(output_dir, "subject.txt"), sep=' ', index=False, header=False)
    print(f"Saved UCI files: X.txt, y.txt, subject.txt")

    # Activity labels
    mapping_path = os.path.join(output_dir, "activity_labels.txt")
    with open(mapping_path, 'w') as f:
        for label, activity in sorted(reverse_mapping.items(), key=lambda x: x[0]):
            f.write(f"{label} {activity}\n")
    print(f"Saved: {mapping_path}\n")
    
    print("Conversion complete!")
    

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main(raw_folder_path = r"C:\Users\tomin\OneDrive\Machine Learning\Extension_research\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\wisdm-dataset\raw",
         subset_size=1000):

    print("=" * 70)
    print("WISDM to UCI HAR Feature Extraction Pipeline")
    print("=" * 70 + "\n")
    
    try:
        # Step 1-2
        wisdm_df = load_wisdm_data(raw_folder_path)
        windows = segment_data_into_windows(wisdm_df, window_size=128, overlap=0.5)  
        
        if not windows:
            raise ValueError("No windows created—check data segments.")
        
        # --- SUBSET FOR TESTING ---
        if subset_size is not None and subset_size > 0:
            original_count = len(windows)
            windows = windows[:subset_size]
            print(f"Testing on subset: {len(windows)} windows (out of {original_count} total)\n")
        
        # Step 3-4
        features_df = extract_all_features(windows)
        
        # Step 5
        X, y, subjects, activity_mapping, reverse_mapping = prepare_uci_har_format(features_df)
        
        # Step 6
        save_to_csv(features_df, X, y, subjects, activity_mapping, reverse_mapping)

        # Quick validation on subset
        print("Subset CSV Validation:")
        print(f"  - No NaNs in X: {not X.isna().any().any()}")
        print(f"  - Activity distribution: {features_df['activity'].value_counts()}")
        print(f"  - Sample features mean (X_mean): {X['X_mean'].mean():.4f} (plausible ~0-10 for accel)\n")
        
        print("=" * 70)
        print("Summary:")
        print(f"  Total windows: {len(features_df)}")
        print(f"  Features per window: {X.shape[1]}")
        print(f"  Activities: {len(activity_mapping)}")
        print("=" * 70)
        
        return features_df, X, y, activity_mapping  
    
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    


if __name__ == "__main__":

    raw_folder = r"C:\Users\tomin\OneDrive\Machine Learning\Extension_research\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\wisdm-dataset\raw"
    features_df, X, y, activity_mapping = main(raw_folder, subset_size=1000)

    # try:
    #     # Step 1: Load WISDM data
    #     wisdm_df = load_wisdm_data(raw_folder_path)

    #     # Step 2: Segment into windows
    #     windows = segment_data_into_windows(wisdm_df, window_size=128, overlap=0.5)      

    #     # Step 3: Test compute_features_for_window on real data (first window)
    #     if windows:
    #         sample_window_array = np.array(windows[0]['window'])
    #         features = compute_features_for_window(sample_window_array)

    #         print("Step 3: Features computed for first real window:")
    #         print(features)
    #         print(f"Total features extracted: {len(features)}\n")

    #         # Basic checks
    #         assert 'X_mean' in features, "Missing features!"
    #         assert not np.isnan(features['X_mean']), "NaN in features!"
    #         print("Feature computation test passed!\n")

    #     # Step 4: Test extract_all_features on real data (subset for speed, then full)
    #     if windows:
    #         # Quick subtest test (first 500 windows)
    #         test_windows = windows[:500]
    #         print(f"Testing Step 4 on subset of {len(test_windows)} windows...\n")
    #         features_df_subset = extract_all_features(test_windows)

    #         # Validations for subset
    #         assert features_df_subset.shape[0] == len(test_windows), "Subset row mismatch!"
    #         assert 'activity' in features_df_subset.columns, "Missing metadata!"
    #         print("Step 4 subset test passed!")
    #         features_df_subset.to_csv('features_subset.csv', index=False)
    #         print("Saved features_subset.csv\n")

    #         # Full run (uncomment to run on all windows)
    #         # print("Running Step 4 on FULL windows...")
    #         # features_df = extract_all_features(windows)
    #         # 
    #         # # Save UCI HAR compatible files
    #         # feature_cols = [col for col in features_df.columns if col not in ['activity', 'user_id']]
    #         # features_df[feature_cols].to_csv('X_train.txt', sep=' ', index=False, header=False)
    #         # # Encode activity to numeric labels (UCI style)
    #         # activity_labels = features_df['activity'].astype('category').cat.codes + 1
    #         # activity_labels.to_csv('y_train.txt', index=False, header=False)
    #         # features_df['user_id'].to_csv('subject_train.txt', index=False, header=False)
    #         # print("Saved full UCI HAR files: X_train.txt, y_train.txt, subject_train.txt")


    #     # Additional checks for testing
    #     if windows:
    #         print("Sample window:")
    #         print(f"User ID: {windows[0]['user_id']}")
    #         print(f"Activity: {windows[0]['activity']}")
    #         print(f"Window shape: {windows[0]['window'].shape}")
    #         print(f"First 5 rows of the window:\n{windows[0]['window'][:5]}\n")
    #     else:
    #         print("No windows created. Check data.")
    #     # Additional checks
    #     # print(f" Unique sources: {wisdm_df['source'].unique()}")
    #     # print(f" Unique activities: {wisdm_df['activity'].unique()}")
    #     # print(f" Unique user IDs: {wisdm_df['user_id'].unique()}")
    #     # print(f" Data types:\n{wisdm_df.dtypes}")

    # except Exception as e:
    #     print(f"Test failed: {e}")
