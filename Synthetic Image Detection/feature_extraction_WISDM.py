# Goal: Convert raw WISDM readings into a feature table compatible with UCI HAR.

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats, signal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# ============================================================================
# UCI HAR ACTIVITY MAPPING
# ============================================================================

#  Assumptions:
#    WISDM    → UCI HAR
# B (jogging) → WALKING_UPSTAIRS (both involve leg movement with elevation change)
# C (stairs) → WALKING_DOWNSTAIRS (arbitrary direction choice)
# F (typing) → LAYING (both are sedentary/stationary)
# Map WISDM activities to UCI HAR's 6 activities
UCI_ACTIVITY_MAPPING = {
    'A': 'WALKING',           # Walking
    'B': 'WALKING_UPSTAIRS',  # Stairs up
    'C': 'WALKING_DOWNSTAIRS',# Stairs down
    'D': 'SITTING',           # Sitting
    'E': 'STANDING',          # Standing
    'F': 'LAYING',            # Lying down    
}


# ============================================================================
# STEP 1: LOAD WISDM RAW CSV  (PHONE ACCELEROMETER ONLY)
# ============================================================================
def load_wisdm_data(raw_folder_path, phone_only=True, accel_only=True):

    print("Step 1: Loading WISDM data (phone accelerometer only)...")

    # Ensure folder path exists
    raw_folder_path = Path(raw_folder_path)
    if not raw_folder_path.exists() or not raw_folder_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {raw_folder_path}")

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Expected column names for WISDM accelerometer data
    columns = ['user_id', 'activity', 'timestamp', 'x', 'y', 'z']

    # Phone accelerometer only (primary analysis)
    if phone_only and accel_only:
        subfolders = ['phone/accel']
        print("  Loading: Phone accelerometer only (UCI-like setup)")
    else:
        subfolders = ['phone/accel', 'phone/gyro', 'watch/accel', 'watch/gyro']
        print("  Loading: All sensors")

    

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

    #Filter to UCI-mappable activities only
    original_count = len(combined_df)
    combined_df = combined_df[combined_df['activity'].isin(UCI_ACTIVITY_MAPPING.keys())]
    filtered_count = len(combined_df)

    print(f"\n  Total rows loaded: {original_count:,}")
    print(f"  Rows after UCI activity filtering: {filtered_count:,}")
    print(f"  Activities kept: {sorted(combined_df['activity'].unique())}")
    print(f"  UCI mapping: {UCI_ACTIVITY_MAPPING}\n")
    
    return combined_df


# ============================================================================
# STEP 2: SEGMENT INTO WINDOWS
# ============================================================================
def segment_data_into_windows(df, window_size=128, overlap=0.5):
    print("Step 2: Segmenting data into windows...")
    print(f"  Window size: {window_size} samples @ 20Hz = {window_size/20:.1f} seconds")

    # Calculate how many samples to skip between windows
    step_size = int(window_size * (1 - overlap))

    windows = []

    # Group data by user to keep their data separate
    for user_id, user_group in df.groupby('user_id'):
        for activity, activity_group in user_group.groupby('activity'):

            # Extract only the sensor axes (X, Y, Z columns)
            activity_group = activity_group.sort_values('timestamp').reset_index(drop=True)

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
                    'activity_uci': UCI_ACTIVITY_MAPPING[activity],
                    'user_id': user_id
                })


    print(f"  Created {len(windows):,} windows")
    print(f"  Step size: {step_size} samples ({overlap*100:.0f}% overlap)\n")    
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
        features[f'{axis_name}_mad'] = np.mean(np.abs(data - np.mean(data)))
        features[f'{axis_name}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
        features[f'{axis_name}_energy'] = np.sum(data ** 2) / len(data)
        features[f'{axis_name}_skew'] = stats.skew(data)
        features[f'{axis_name}_kurt'] = stats.kurtosis(data)

       
       
       
    # === MAGNITUDE-BASED FEATURES (7 features) ===
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    features['magnitude_mean'] = np.mean(magnitude)
    features['magnitude_std'] = np.std(magnitude)
    features['magnitude_max'] = np.max(magnitude)
    features['magnitude_min'] = np.min(magnitude)
    features['sma'] = np.mean(np.abs(x) + np.abs(y) + np.abs(z))  # Signal Magnitude Area
    features['magnitude_iqr'] = np.percentile(magnitude, 75) - np.percentile(magnitude, 25)
    features['magnitude_mad'] = np.mean(np.abs(magnitude - np.mean(magnitude)))
    
    # === CORRELATION FEATURES (3 features) ===
    features['corr_xy'] = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    features['corr_xz'] = np.corrcoef(x, z)[0, 1] if len(x) > 1 else 0
    features['corr_yz'] = np.corrcoef(y, z)[0, 1] if len(x) > 1 else 0
    
    # === FREQUENCY DOMAIN FEATURES (Per Axis: 3 axes × 7 = 21 features) ===
    for axis_name, data in [('X', x), ('Y', y), ('Z', z)]:
        # FFT and Power Spectral Density
        fft = np.fft.fft(data)
        fft_abs = np.abs(fft[:len(fft)//2])  # Only positive frequencies
        psd = (fft_abs ** 2) / len(fft_abs)
        
        # Peak frequency
        if len(fft_abs) > 1:
            peak_idx = np.argmax(psd[1:]) + 1
            features[f'{axis_name}_peak_freq'] = peak_idx
            features[f'{axis_name}_peak_power'] = psd[peak_idx]
        else:
            features[f'{axis_name}_peak_freq'] = 0
            features[f'{axis_name}_peak_power'] = 0
        
        # Spectral energy in frequency bands
        n = len(psd)
        if n >= 6:
            features[f'{axis_name}_energy_band1'] = np.sum(psd[1:n//3])      # Low freq
            features[f'{axis_name}_energy_band2'] = np.sum(psd[n//3:2*n//3]) # Mid freq
            features[f'{axis_name}_energy_band3'] = np.sum(psd[2*n//3:])     # High freq
        else:
            features[f'{axis_name}_energy_band1'] = 0
            features[f'{axis_name}_energy_band2'] = 0
            features[f'{axis_name}_energy_band3'] = 0
        
        # Spectral entropy
        psd_norm = psd / (np.sum(psd) + 1e-10)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        features[f'{axis_name}_spectral_entropy'] = spectral_entropy

    # === ENTROPY FEATURES (Per Axis: 3 features) ===
    for axis_name, data in [('X', x), ('Y', y), ('Z', z)]:
        hist, _ = np.histogram(data, bins=10)
        hist_norm = hist / (np.sum(hist) + 1e-10)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        features[f'{axis_name}_entropy'] = entropy

    # Total: 30 + 7 + 3 + 21 + 3 = 64 features (expandable to ~70-80)
    
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
        window_features['activity_uci'] = window_dict['activity_uci']
        window_features['user_id'] = window_dict['user_id']

        all_features.append(window_features)

        # Print progress every 500 windows
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} / {len(windows)} windows")

    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(all_features)

    print(f"\n  Feature DataFrame: {features_df.shape[0]:,} rows x {features_df.shape[1]} columns")
    print(f"  Feature count (excluding metadata): {features_df.shape[1] - 3}")
    print(f"  UCI activities: {sorted(features_df['activity_uci'].unique())}\n")
    
    return features_df


# ============================================================================
# STEP 5: NORMALIZE FEATURES TO [-1, 1] (CRITICAL FOR UCI COMPATIBILITY)
# ============================================================================
def normalize_features(features_df):
    print("Step 5: Normalizing features to [-1, 1] range...")
    
    metadata_cols = ['activity', 'activity_uci', 'user_id']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_features = scaler.fit_transform(features_df[feature_cols])
    
    normalized_df = pd.DataFrame(normalized_features, columns=feature_cols)
    normalized_df['activity'] = features_df['activity'].values
    normalized_df['activity_uci'] = features_df['activity_uci'].values
    normalized_df['user_id'] = features_df['user_id'].values
    
    print(f"  Features normalized to [-1, 1]")
    print(f"  Sample ranges:")
    print(f"    X_mean: [{normalized_df['X_mean'].min():.4f}, {normalized_df['X_mean'].max():.4f}]")
    print(f"    magnitude_mean: [{normalized_df['magnitude_mean'].min():.4f}, {normalized_df['magnitude_mean'].max():.4f}]\n")
    
    return normalized_df, scaler


# ============================================================================
# STEP 6: PREPARE UCI HAR FORMAT
# ============================================================================
def prepare_uci_har_format(features_df):
    print("Step 5: Preparing UCI HAR format...")

    # Map UCI activity names to numeric labels (1-6, matching UCI HAR)
    uci_label_mapping = {
        'WALKING': 1,
        'WALKING_UPSTAIRS': 2,
        'WALKING_DOWNSTAIRS': 3,
        'SITTING': 4,
        'STANDING': 5,
        'LAYING': 6
    }

    print("  UCI HAR Activity Labels:")
    for activity, label in sorted(uci_label_mapping.items(), key=lambda x: x[1]):
        count = (features_df['activity_uci'] == activity).sum()
        print(f"    {label}: {activity} ({count:,} samples)")

    # Extract features only
    feature_cols = [col for col in features_df.columns 
                   if col not in ['activity', 'activity_uci', 'user_id']]
    X = features_df[feature_cols].reset_index(drop=True)

    # Map to UCI numeric labels
    y = features_df['activity_uci'].map(uci_label_mapping).reset_index(drop=True)
    subjects = features_df['user_id'].reset_index(drop=True)
    
    print(f"\n  X shape: {X.shape} (samples X features)")
    print(f"  y shape: {y.shape}")
    print(f"  Unique subjects: {subjects.nunique()}\n")
    
    return X, y, subjects, uci_label_mapping



# ============================================================================
# STEP 7: STATISTICAL VALIDATION (Distribution Comparison)
# ============================================================================
def statistical_validation(features_df, output_dir):
    print("Step 7: Statistical validation...")
    
    feature_cols = [col for col in features_df.columns 
                   if col not in ['activity', 'activity_uci', 'user_id']]
    
    stats_summary = features_df[feature_cols].describe()
    
    print("\n  Feature Statistics (sample):")
    print(stats_summary[['X_mean', 'Y_mean', 'Z_mean', 'magnitude_mean']].to_string())
    
    # Save statistics
    stats_path = os.path.join(output_dir, "feature_statistics.csv")
    stats_summary.to_csv(stats_path)
    print(f"\n   Saved feature statistics: {stats_path}\n")
    
    return stats_summary


# ============================================================================
# STEP 8: SAVE TO UCI HAR FORMAT
# ============================================================================
def save_to_uci_format(features_df, X, y, subjects, uci_label_mapping, output_dir):
    print("Step 8: Saving to UCI HAR format...")

    os.makedirs(output_dir, exist_ok=True)

    # Full feature DataFrame with metadata
    features_csv = os.path.join(output_dir, "wisdm_features_normalized.csv")
    features_df.to_csv(features_csv, index=False)
    print(f"   Saved: wisdm_features_normalized.csv")

    # UCI format files (space-separated, no headers)
    X.to_csv(os.path.join(output_dir, "X_wisdm.txt"), sep=' ', index=False, header=False)
    print(f"   Saved: X_wisdm.txt ({X.shape[0]:,} samples x {X.shape[1]} features)")

    y.to_csv(os.path.join(output_dir, "y_wisdm.txt"), index=False, header=False)
    print(f"   Saved: y_wisdm.txt ({len(y):,} labels)")

    subjects.to_csv(os.path.join(output_dir, "subject_wisdm.txt"), index=False, header=False)
    print(f"   Saved: subject_wisdm.txt ({subjects.nunique()} unique subjects)")
    
    # Activity labels (UCI format)
    labels_file = os.path.join(output_dir, "activity_labels.txt")
    with open(labels_file, 'w') as f:
        for activity, label in sorted(uci_label_mapping.items(), key=lambda x: x[1]):
            f.write(f"{label} {activity}\n")
    print(f"   Saved: activity_labels.txt")

    # Feature names
    features_file = os.path.join(output_dir, "feature_names.txt")
    with open(features_file, 'w') as f:
        for i, col in enumerate(X.columns, 1):
            f.write(f"{i} {col}\n")
    print(f"   Saved: feature_names.txt ({len(X.columns)} features)\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main(raw_folder_path, output_dir, subset_size=None):

    print("=" * 80)
    print("WISDM to UCI HAR Feature Extraction Pipeline")
    print("Focus: Phone accelerometer only, 6 UCI-mappable activities, ~70 features")
    print("=" * 80 + "\n")
    
    try:
        # Step 1-2
        wisdm_df = load_wisdm_data(raw_folder_path, phone_only=True, accel_only=True)
        windows = segment_data_into_windows(wisdm_df, window_size=128, overlap=0.5)  
        
        if not windows:
            raise ValueError("No windows created—check data segments.")
        
        # Optional subset for testing
        if subset_size and subset_size > 0:
            print(f"Using subset: {subset_size:,} windows (testing mode)\n")
            windows = windows[:subset_size]
        
        # Step 3-4: Extract and normalize features
        features_df = extract_all_features(windows)
        features_df_norm, scaler = normalize_features(features_df)
        
        # Step 5: UCI format
        X, y, subjects, uci_label_mapping = prepare_uci_har_format(features_df_norm)
        
        # Step 6: Statistical validation
        stats_summary = statistical_validation(features_df_norm, output_dir)
        
        # Step 7: Save
        save_to_uci_format(features_df_norm, X, y, subjects, uci_label_mapping, output_dir)

        # Final validation
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f" Total samples: {len(features_df_norm):,}")
        print(f" Features per sample: {X.shape[1]}")
        print(f" Activities (UCI format): {len(uci_label_mapping)}")
        print(f" Unique subjects: {subjects.nunique()}")
        print(f" Feature range: [-1, 1] (normalized)")
        print(f" No missing values: {not X.isna().any().any()}")
        print("\nActivity distribution:")
        for activity, label in sorted(uci_label_mapping.items(), key=lambda x: x[1]):
            count = (y == label).sum()
            pct = 100 * count / len(y)
            print(f"  {label}. {activity:20s}: {count:5,} ({pct:5.1f}%)")
        print("=" * 80)
        
        return features_df_norm, X, y, uci_label_mapping
    
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    


if __name__ == "__main__":

    raw_folder = r"C:\Users\tomin\OneDrive\Machine Learning\Extension_research\wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset\wisdm-dataset\wisdm-dataset\raw"
    output_folder = r"C:\Users\tomin\source\repos\Synthetic Image Detection\Synthetic Image Detection"
    
    # Run full pipeline (remove subset_size for complete dataset)
    features_df, X, y, activity_mapping = main(raw_folder, output_folder, subset_size=1000)
   
   
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
