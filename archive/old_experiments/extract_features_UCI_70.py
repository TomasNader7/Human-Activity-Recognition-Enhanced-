# extract_features_UCI_70.py
import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

# ================================================
# Feature recipe (same as WISDM)
# ================================================
def compute_features_for_window(win_xyz, fs=50.0):
    """
    win_xyz: np.ndarray shape (128, 3) -> columns [x, y, z]
    returns: dict of ~64 features (exactly the same names/order as your WISDM extractor)
    """
    x = win_xyz[:, 0]
    y = win_xyz[:, 1]
    z = win_xyz[:, 2]

    feats = {}

    # ---- Basic Statistics per axis ----
    for axis_name, data in (('X', x), ('Y', y), ('Z', z)):
        feats[f'{axis_name}_mean']   = float(np.mean(data))
        feats[f'{axis_name}_std']    = float(np.std(data))
        feats[f'{axis_name}_min']    = float(np.min(data))
        feats[f'{axis_name}_max']    = float(np.max(data))
        feats[f'{axis_name}_median'] = float(np.median(data))
        feats[f'{axis_name}_mad']    = float(np.mean(np.abs(data - np.mean(data))))
        feats[f'{axis_name}_iqr']    = float(np.percentile(data, 75) - np.percentile(data, 25))
        feats[f'{axis_name}_energy'] = float(np.sum(data ** 2) / len(data))
        feats[f'{axis_name}_skew']   = float(stats.skew(data))
        feats[f'{axis_name}_kurt']   = float(stats.kurtosis(data))

    # === MAGNITUDE-BASED FEATURES (7 features) ===
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    feats['magnitude_mean'] = float(np.mean(magnitude))
    feats['magnitude_std']  = float(np.std(magnitude))
    feats['magnitude_max']  = float(np.max(magnitude))
    feats['magnitude_min']  = float(np.min(magnitude))
    feats['sma']            = float(np.mean(np.abs(x) + np.abs(y) + np.abs(z)))  # Signal Magnitude Area
    feats['magnitude_iqr']  = float(np.percentile(magnitude, 75) - np.percentile(magnitude, 25))
    feats['magnitude_mad']  = float(np.mean(np.abs(magnitude - np.mean(magnitude))))

    # === CORRELATION FEATURES (3 features) ===
    feats['corr_xy'] = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    feats['corr_xz'] = float(np.corrcoef(x, z)[0, 1]) if np.std(x) > 0 and np.std(z) > 0 else 0.0
    feats['corr_yz'] = float(np.corrcoef(y, z)[0, 1]) if np.std(y) > 0 and np.std(z) > 0 else 0.0

    # === FREQUENCY DOMAIN FEATURES (Per Axis: 3 axes × 7 = 21 features) ===
    for axis_name, data in [('X', x), ('Y', y), ('Z', z)]:
        fft = np.fft.fft(data)
        fft_abs = np.abs(fft[:len(fft)//2])           # positive freqs
        psd = (fft_abs ** 2) / max(len(fft_abs), 1)   # simple PSD proxy

        # Peak frequency (index; matches your WISDM implementation)
        if len(fft_abs) > 1:
            peak_idx = int(np.argmax(psd[1:]) + 1)
            feats[f'{axis_name}_peak_freq']  = float(peak_idx)
            feats[f'{axis_name}_peak_power'] = float(psd[peak_idx])
        else:
            feats[f'{axis_name}_peak_freq']  = 0.0
            feats[f'{axis_name}_peak_power'] = 0.0

        # Spectral energy in 3 coarse bands
        n = len(psd)
        if n >= 6:
            feats[f'{axis_name}_energy_band1'] = float(np.sum(psd[1:n//3]))
            feats[f'{axis_name}_energy_band2'] = float(np.sum(psd[n//3:2*n//3]))
            feats[f'{axis_name}_energy_band3'] = float(np.sum(psd[2*n//3:]))
        else:
            feats[f'{axis_name}_energy_band1'] = 0.0
            feats[f'{axis_name}_energy_band2'] = 0.0
            feats[f'{axis_name}_energy_band3'] = 0.0

        # Spectral entropy
        psd_norm = psd / (np.sum(psd) + 1e-10)
        feats[f'{axis_name}_spectral_entropy'] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))

    # === HISTOGRAM ENTROPY (Per Axis: 3 features) ===
    for axis_name, data in [('X', x), ('Y', y), ('Z', z)]:
        hist, _ = np.histogram(data, bins=10)
        hist_norm = hist / (np.sum(hist) + 1e-10)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        feats[f'{axis_name}_entropy'] = float(entropy)

    # Total ≈ 64 features (same as your WISDM file)
    return feats

# ================================================
# Data loading & writer
# ================================================
def _load_axis(path): 
    if not os.path.exists(path): raise FileNotFoundError(path)
    return np.loadtxt(path, dtype=np.float32)

def build_uci_features(uci_dir: str, split: str, names_file: str|None):
    sigdir = os.path.join(uci_dir, split, "Inertial Signals")
    # Use body acceleration to mirror your phone-accel WISDM choice
    ax = _load_axis(os.path.join(sigdir, f"body_acc_x_{split}.txt"))
    ay = _load_axis(os.path.join(sigdir, f"body_acc_y_{split}.txt"))
    az = _load_axis(os.path.join(sigdir, f"body_acc_z_{split}.txt"))

    feats = []
    for i in range(ax.shape[0]):
        win = np.stack([ax[i], ay[i], az[i]], axis=1)
        feats.append(compute_features_for_window(win, fs=50.0))
    X = pd.DataFrame(feats, dtype=np.float32)

    # === NEW: enforce same 61 columns (names & order) ===
    if names_file:
        wisdm_names = [line.strip().split(" ", 1)[1] for line in open(names_file, "r")]
        # Select the intersection in case of tiny naming mismatches
        missing = [c for c in wisdm_names if c not in X.columns]
        if missing:
            print("WARNING - these WISDM columns not found in UCI features:", missing)
        keep = [c for c in wisdm_names if c in X.columns]
        X = X.reindex(columns=keep)
        assert X.shape[1] == len(keep), "Column reindexing failed"
        print(f"Aligned UCI to WISDM: {len(keep)} features")

    y = pd.read_csv(os.path.join(uci_dir, split, f"y_{split}.txt"), header=None)[0].astype(int)
    subj = pd.read_csv(os.path.join(uci_dir, split, f"subject_{split}.txt"), header=None)[0].astype(int)

    out = os.path.join(uci_dir, f"uci61_{split}")
    os.makedirs(out, exist_ok=True)

    # Save exactly the columns we used
    with open(os.path.join(out, "feature_names_61.txt"), "w") as f:
        for i, c in enumerate(X.columns, 1):
            f.write(f"{i} {c}\n")

    X.to_csv(os.path.join(out, "X_uci61.txt"), sep=" ", index=False, header=False)
    y.to_csv(os.path.join(out, "y.txt"), index=False, header=False)
    subj.to_csv(os.path.join(out, "subject.txt"), index=False, header=False)
    print(f"[OK] Saved {out} | X: {X.shape}")

# ================================================
# CLI
# ================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uci_dir", required=True)
    ap.add_argument("--split", choices=["train","test"], default="train")
    ap.add_argument("--names_file", required=True, help="Path to WISDM feature_names_61.txt")
    args = ap.parse_args()
    build_uci_features(args.uci_dir, args.split, args.names_file)

if __name__ == "__main__":
    main()
