import os
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, List

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================
LABELS_3CLASS = ["WALKING", "SITTING", "STANDING"]
REMAPPING_WISDM = {1: 2, 2: 3, 3: 1}  # WISDM: 1=SITTING,2=STANDING,3=WALKING -> desired: 1=WALKING,2=SITTING,3=STANDING

HAPT_X_PATH = r"C:\Users\tomin\source\repos\Synthetic Image Detection\Synthetic Image Detection\hapt_3class_output_phase2\X_hapt.txt"
HAPT_Y_PATH = r"C:\Users\tomin\source\repos\Synthetic Image Detection\Synthetic Image Detection\hapt_3class_output_phase2\y_hapt.txt"

WISDM_X_PATH = r"C:\Users\tomin\source\repos\Synthetic Image Detection\Filtered_datasets_and_KS_results\3class_wisdm_phase2\X_filtered.txt"
WISDM_Y_PATH = r"C:\Users\tomin\source\repos\Synthetic Image Detection\Filtered_datasets_and_KS_results\3class_wisdm_phase2\y_filtered.txt"


PHASE3_DIR = os.path.join("results", "phase3")
os.makedirs(PHASE3_DIR, exist_ok=True)


# =========================
# MODEL (Base learners)
# =========================
def get_base_learners(random_state: int = 42):
    return [
        ("sgd", SGDClassifier(loss='log_loss', max_iter=2000, tol=1e-3, random_state=random_state)),
        ("random_forest", RandomForestClassifier(n_estimators=20, max_depth=5, random_state=random_state)),
        ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=random_state, class_weight="balanced")),
        ("xgboost", XGBClassifier(n_estimators=20, learning_rate=0.1, random_state=random_state, eval_metric="mlogloss")),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=random_state)),
    ]


def get_meta_learner(random_state: int = 42):
    return LogisticRegression(
        solver="saga",
        max_iter=2000,
        tol=1e-4,
        class_weight="balanced",
        random_state=random_state
    )


# =========================
# DATA LOAD
# =========================
def load_hapt_wisdm_3class() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_hapt = np.loadtxt(HAPT_X_PATH)
    y_hapt = np.loadtxt(HAPT_Y_PATH).astype(int)

    X_wisdm = np.loadtxt(WISDM_X_PATH)
    y_wisdm = np.loadtxt(WISDM_Y_PATH).astype(int)

    # Remap WISDM semantics
    y_wisdm = np.vectorize(REMAPPING_WISDM.get)(y_wisdm)
    
    # Convert to 0-indexed for XGBoost compatibility
    y_hapt = y_hapt - 1   # [1,2,3] -> [0,1,2]
    y_wisdm = y_wisdm - 1 # [1,2,3] -> [0,1,2]

    return X_hapt, y_hapt, X_wisdm, y_wisdm


# =========================
# NORMALIZATION OPTIONS
# =========================
def normalize_minmax_source_only(X_src, X_tgt):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_src_n = scaler.fit_transform(X_src)
    X_tgt_n = scaler.transform(X_tgt)
    return X_src_n, X_tgt_n


def normalize_zscore_per_dataset(X_src, X_tgt):
    # Fit separate scalers (unsupervised target normalization)
    s_src = StandardScaler()
    s_tgt = StandardScaler()
    X_src_n = s_src.fit_transform(X_src)
    X_tgt_n = s_tgt.fit_transform(X_tgt)
    return X_src_n, X_tgt_n


# ==================================================
# STACKING (custom, to allow "meta-only retrain")
# ==================================================
"""
The flow: 
    Show data to 5 experts
    Collect their probability guesses
    Give those guesses to the manager
    Manager makes final call
"""

@dataclass
class FrozenBaseStacking:
    base_learners: List[Tuple[str, object]]
    meta_learner: object

    def fit_base(self, X, y):
        self.fitted_base_ = []
        for name, model in self.base_learners:
            m = clone(model)
            m.fit(X, y)
            self.fitted_base_.append((name, m))
        return self

    def base_proba_features(self, X) -> np.ndarray:
        # concatenate probabilities from each base learner
        feats = []
        for _, m in self.fitted_base_:
            p = m.predict_proba(X)  # shape (n, n_classes) => Ask: "How confident are you?"
            feats.append(p)
        return np.hstack(feats)  # shape (n, n_classes * n_models) => Combine all opinions

    def fit_meta(self, X_meta, y_meta):
        self.fitted_meta_ = clone(self.meta_learner)
        self.fitted_meta_.fit(X_meta, y_meta)
        return self

    def predict(self, X):
        Z = self.base_proba_features(X)
        return self.fitted_meta_.predict(Z)

    def evaluate(self, X, y) -> Dict:
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, target_names=LABELS_3CLASS, digits=4)
        return {"acc": acc, "cm": cm, "report": report}


# =========================
# SAVING HELPERS
# =========================
def save_confusion_matrix(cm, out_png, title):
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS_3CLASS, yticklabels=LABELS_3CLASS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_metrics(out_dir, name, train_acc, test_acc, report, cm):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Train accuracy: {train_acc:.4f}\n")
        f.write(f"Test accuracy : {test_acc:.4f}\n")

    with open(os.path.join(out_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(report)

    np.savetxt(os.path.join(out_dir, f"{name}_confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
    save_confusion_matrix(cm, os.path.join(out_dir, f"{name}_confusion_matrix.png"),
                          title=f"{name}: Confusion Matrix (HAPT → WISDM)")


def save_degradation_plot(out_dir, name, train_acc, test_acc):
    plt.figure(figsize=(6, 5))
    plt.bar(["Source (HAPT)", "Target (WISDM)"], [train_acc, test_acc])
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(f"{name}: Performance Degradation")
    for i, v in enumerate([train_acc, test_acc]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_performance_degradation.png"), dpi=300)
    plt.close()


# =========================
# EXPERIMENTS
# =========================
def exp_feature_distribution_alignment(X_hapt, y_hapt, X_wisdm, y_wisdm):
    """
    Compare:
      A) MinMax scaler fit on HAPT only (your current baseline behavior)
      B) Z-score per dataset (unsupervised target normalization)
    """
    out_dir = os.path.join(PHASE3_DIR, "A_distribution_alignment")
    os.makedirs(out_dir, exist_ok=True)

    base = get_base_learners()
    meta = get_meta_learner()

    # --- A) MinMax source-only ---
    Xs, Xt = normalize_minmax_source_only(X_hapt, X_wisdm)

    modelA = FrozenBaseStacking(base, meta).fit_base(Xs, y_hapt)
    Zs = modelA.base_proba_features(Xs)
    modelA.fit_meta(Zs, y_hapt)

    train_res = modelA.evaluate(Xs, y_hapt)
    test_res = modelA.evaluate(Xt, y_wisdm)

    save_metrics(out_dir, "minmax_source_only",
                 train_res["acc"], test_res["acc"],
                 test_res["report"], test_res["cm"])
    save_degradation_plot(out_dir, "minmax_source_only", train_res["acc"], test_res["acc"])

    # --- B) Z-score per dataset ---
    Xs2, Xt2 = normalize_zscore_per_dataset(X_hapt, X_wisdm)

    modelB = FrozenBaseStacking(base, meta).fit_base(Xs2, y_hapt)
    Zs2 = modelB.base_proba_features(Xs2)
    modelB.fit_meta(Zs2, y_hapt)

    train_res2 = modelB.evaluate(Xs2, y_hapt)
    test_res2 = modelB.evaluate(Xt2, y_wisdm)

    save_metrics(out_dir, "zscore_per_dataset",
                 train_res2["acc"], test_res2["acc"],
                 test_res2["report"], test_res2["cm"])
    save_degradation_plot(out_dir, "zscore_per_dataset", train_res2["acc"], test_res2["acc"])


def exp_meta_learner_adaptation(X_hapt, y_hapt, X_wisdm, y_wisdm, wisdm_frac=0.10, seed=42):
    """
    Freeze base learners trained on HAPT.
    Retrain ONLY meta-learner using a small labeled subset of WISDM.
    """
    out_dir = os.path.join(PHASE3_DIR, "B_meta_learner_adaptation")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    n = len(y_wisdm)
    idx = np.arange(n)
    rng.shuffle(idx)

    k = int(np.floor(wisdm_frac * n))
    idx_ft = idx[:k]
    idx_eval = idx[k:]  # optional (or evaluate on full WISDM)

    X_ft = X_wisdm[idx_ft]
    y_ft = y_wisdm[idx_ft]

    base = get_base_learners()
    meta = get_meta_learner()

    # Keep normalization consistent with your baseline first (MinMax source-only)
    Xs, Xt = normalize_minmax_source_only(X_hapt, X_wisdm)
    X_ft_n = Xt[idx_ft]

    model = FrozenBaseStacking(base, meta).fit_base(Xs, y_hapt)

    # Train meta on HAPT (baseline meta)
    model.fit_meta(model.base_proba_features(Xs), y_hapt)
    base_train = model.evaluate(Xs, y_hapt)
    base_test = model.evaluate(Xt, y_wisdm)

    save_metrics(out_dir, f"baseline_meta_before_adapt_frac{int(wisdm_frac*100)}",
                 base_train["acc"], base_test["acc"],
                 base_test["report"], base_test["cm"])
    save_degradation_plot(out_dir, f"baseline_meta_before_adapt_frac{int(wisdm_frac*100)}",
                          base_train["acc"], base_test["acc"])

    # Now retrain ONLY meta using WISDM subset (base frozen)
    model.fit_meta(model.base_proba_features(X_ft_n), y_ft)

    after_train = model.evaluate(Xs, y_hapt)      # optional: see if source drops
    after_test = model.evaluate(Xt, y_wisdm)

    save_metrics(out_dir, f"meta_only_adapt_frac{int(wisdm_frac*100)}",
                 after_train["acc"], after_test["acc"],
                 after_test["report"], after_test["cm"])
    save_degradation_plot(out_dir, f"meta_only_adapt_frac{int(wisdm_frac*100)}",
                          after_train["acc"], after_test["acc"])


def exp_fine_tuning_learning_curve(X_hapt, y_hapt, X_wisdm, y_wisdm, fracs=(0.05, 0.10, 0.25, 0.50), seed=42):
    """
    Practical sklearn 'fine-tuning':
      retrain the whole ensemble on (HAPT + labeled subset of WISDM).
    This is the simplest consistent way with non-incremental learners.
    """
    out_dir = os.path.join(PHASE3_DIR, "C_finetune_learning_curve")
    os.makedirs(out_dir, exist_ok=True)

    # normalize baseline way first (minmax source-only)
    Xs, Xt = normalize_minmax_source_only(X_hapt, X_wisdm)

    rng = np.random.default_rng(seed)
    n = len(y_wisdm)
    idx = np.arange(n)
    rng.shuffle(idx)

    accs = []

    for frac in fracs:
        k = int(np.floor(frac * n))
        idx_ft = idx[:k]

        X_ft = Xt[idx_ft]
        y_ft = y_wisdm[idx_ft]

        # "Fine-tune" by retraining on HAPT + subset WISDM
        X_mix = np.vstack([Xs, X_ft])
        y_mix = np.concatenate([y_hapt, y_ft])

        model = FrozenBaseStacking(get_base_learners(), get_meta_learner())
        model.fit_base(X_mix, y_mix)
        model.fit_meta(model.base_proba_features(X_mix), y_mix)

        train_res = model.evaluate(X_mix, y_mix)
        test_res = model.evaluate(Xt, y_wisdm)

        name = f"finetune_frac{int(frac*100)}"
        save_metrics(out_dir, name,
                     train_res["acc"], test_res["acc"],
                     test_res["report"], test_res["cm"])
        save_degradation_plot(out_dir, name, train_res["acc"], test_res["acc"])

        accs.append((frac, test_res["acc"]))

    # learning curve plot
    fr = [f*100 for f, _ in accs]
    ac = [a for _, a in accs]
    plt.figure(figsize=(7, 5))
    plt.plot(fr, ac, marker="o")
    plt.ylim(0, 1.0)
    plt.xlabel("WISDM labeled subset used for fine-tuning (%)")
    plt.ylabel("WISDM Accuracy")
    plt.title("Fine-tuning Learning Curve (HAPT → WISDM)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve.png"), dpi=300)
    plt.close()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    X_hapt, y_hapt, X_wisdm, y_wisdm = load_hapt_wisdm_3class()

    print("HAPT:", X_hapt.shape, np.unique(y_hapt, return_counts=True))
    print("WISDM:", X_wisdm.shape, np.unique(y_wisdm, return_counts=True))

    # A) distribution alignment
    exp_feature_distribution_alignment(X_hapt, y_hapt, X_wisdm, y_wisdm)

    # B) meta-learner adaptation (start with 10% as professor suggested)
    exp_meta_learner_adaptation(X_hapt, y_hapt, X_wisdm, y_wisdm, wisdm_frac=0.10)

    # C) fine-tuning curve
    exp_fine_tuning_learning_curve(X_hapt, y_hapt, X_wisdm, y_wisdm, fracs=(0.05, 0.10, 0.25, 0.50))
