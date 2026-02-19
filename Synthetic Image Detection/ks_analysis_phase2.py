import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# ======================
# CONFIG 
# ======================
PHASE_TAG = "phase2"
RESULTS_DIR = os.path.join("Filtered_datasets_and_KS_results", f"3class_{PHASE_TAG}")
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS = {1: "WALKING", 2: "SITTING", 3: "STANDING"}

# ======================
# LOAD FEATURES 
# ======================
X_hapt = np.loadtxt(r"C:\Users\tomin\source\repos\Synthetic Image Detection\Synthetic Image Detection\hapt_3class_output_phase2\X_hapt.txt")
X_wisdm = np.loadtxt(r"C:\Users\tomin\source\repos\Synthetic Image Detection\Filtered_datasets_and_KS_results\3class_wisdm_phase2\X_filtered.txt")

# ======================
# LOAD LABELS (NEW)
# ======================
y_hapt = np.loadtxt(r"C:\Users\tomin\source\repos\Synthetic Image Detection\Synthetic Image Detection\hapt_3class_output_phase2\y_hapt.txt").astype(int)
y_wisdm = np.loadtxt(r"C:\Users\tomin\source\repos\Synthetic Image Detection\Filtered_datasets_and_KS_results\3class_wisdm_phase2\y_filtered.txt").astype(int)

# If your WISDM y_filtered is NOT aligned to HAPT semantics, uncomment and apply:
remap = {1: 2, 2: 3, 3: 1}
y_wisdm = np.vectorize(remap.get)(y_wisdm)

# Safety checks
assert X_hapt.shape[0] == y_hapt.shape[0], "HAPT X/y mismatch"
assert X_wisdm.shape[0] == y_wisdm.shape[0], "WISDM X/y mismatch"
assert X_hapt.shape[1] == X_wisdm.shape[1], "Feature count mismatch (alignment broken)"

n_features = X_hapt.shape[1]

# =========================
# PER-ACTIVITY KS TESTS 
# =========================
per_activity_rows = []
activities = [1, 2, 3]

for act in activities:
    Xh = X_hapt[y_hapt == act]
    Xw = X_wisdm[y_wisdm == act]

    if len(Xh) == 0 or len(Xw) == 0:
        continue

    for i in range(n_features):
        stat, p = ks_2samp(Xh[:, i], Xw[:, i])
        per_activity_rows.append([LABELS[act], i, stat, p, int(p < 0.05)])

per_activity_arr = np.array(per_activity_rows, dtype=object)

# Save per-activity CSV (matches your naming convention)
per_activity_csv = os.path.join(RESULTS_DIR, "ks_test_per_activity_3class_phase2.csv")
with open(per_activity_csv, "w") as f:
    f.write("activity,feature_idx,ks_statistic,p_value,significant\n")
    for row in per_activity_rows:
        f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

print("Saved:", per_activity_csv)

# ===============================
# PER-ACTIVITY SUMMARY + PLOT 
# ===============================
# Compute mean KS and % significant per activity
activity_summary = []
for act in activities:
    act_name = LABELS[act]
    rows = [r for r in per_activity_rows if r[0] == act_name]
    if not rows:
        continue
    ks_vals = np.array([r[2] for r in rows], dtype=float)
    sig_vals = np.array([r[4] for r in rows], dtype=int)
    activity_summary.append((act_name, ks_vals.mean(), 100.0 * sig_vals.mean()))

# Sort by mean KS desc (like your plots)
activity_summary.sort(key=lambda x: x[1], reverse=True)

names = [x[0] for x in activity_summary]
mean_ks = [x[1] for x in activity_summary]
sig_pct = [x[2] for x in activity_summary]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: mean KS by activity
axes[0].barh(names, mean_ks)
axes[0].set_xlabel("Mean K-S Statistic")
axes[0].set_title("Domain Shift by Activity (3CLASS)")
axes[0].invert_yaxis()

# Right: % significant by activity
axes[1].barh(names, sig_pct)
axes[1].set_xlabel("% Features with Significant Shift (p < 0.05)")
axes[1].set_title("Statistical Significance by Activity")
axes[1].invert_yaxis()

plt.tight_layout()
out_plot = os.path.join(RESULTS_DIR, "ks_per_activity_3class_phase2.png")
plt.savefig(out_plot, dpi=300)
plt.close()

print("Saved:", out_plot)
