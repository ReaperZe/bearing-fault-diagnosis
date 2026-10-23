"""
Predictive Maintenance: Bearing Fault Diagnosis on the CWRU Bearing Dataset
---------------------------------------------------------------------------
Pipeline:
  1. Download CWRU 12 kHz drive-end .mat files
  2. Split BEFORE windowing (no data leakage) -> train / val / test
  3. Window signals, extract time-domain + FFT features
  4. Train Random Forest + XGBoost, tune on VALIDATION set
  5. Evaluate ONCE on the held-out TEST set

Install:  pip install numpy scipy pandas scikit-learn matplotlib joblib xgboost

Dataset page (manual download if auto-download fails):
  https://engineering.case.edu/bearingdatacenter/download-data-file
  -> put the .mat files (97.mat, 105.mat, ...) into ./data/cwru/

Split modes (SPLIT_MODE):
  "segment": every recording is cut in time order -> first 70% train,
             next 15% val, last 15% test (a gap is left between them).
             All fault classes appear in all splits.
  "file"   : whole recordings are assigned to train/val/test.
             Stricter test (model must generalise to unseen recordings /
             loads), but with only 4 Normal files the numbers are noisier.
"""

import os
import urllib.request
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed - only Random Forest will be trained.")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_DIR = "data/cwru"
SEED = 42
FS = 12_000            # sampling rate (Hz) of the 12k drive-end data
WIN = 2048             # window length (samples)
STRIDE = 1024          # 50% overlap (only applied INSIDE a split)
SPLIT_MODE = "segment"  # "segment" or "file"
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # test = remainder (0.15)
N_FFT_BANDS = 32

# CWRU 12k drive-end file numbers: {class: [file ids]}  (loads 0-3 hp, 4 each)
# Fault diameters: 0.007", 0.014", 0.021"; outer race fault at 6 o'clock.
FILES = {
    "Normal":     [97, 98, 99, 100],
    "InnerRace":  [105, 106, 107, 108, 169, 170, 171, 172, 209, 210, 211, 212],
    "Ball":       [118, 119, 120, 121, 185, 186, 187, 188, 222, 223, 224, 225],
    "OuterRace":  [130, 131, 132, 133, 197, 198, 199, 200, 234, 235, 236, 237],
}
CLASSES = list(FILES.keys())

URL_TEMPLATES = [
    "https://engineering.case.edu/sites/default/files/{id}.mat",
    "https://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/{id}.mat",
]


# ----------------------------------------------------------------------------
# 1. Data download / loading
# ----------------------------------------------------------------------------
def download_file(fid: int) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{fid}.mat")
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path
    for tpl in URL_TEMPLATES:
        url = tpl.format(id=fid)
        try:
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, path)
            if os.path.getsize(path) > 10_000:
                return path
        except Exception as e:  # noqa: BLE001
            print(f"    failed: {e}")
    raise FileNotFoundError(
        f"Could not download {fid}.mat. Download it manually from "
        f"https://engineering.case.edu/bearingdatacenter/download-data-file "
        f"and place it in {DATA_DIR}/"
    )


def load_signal(fid: int) -> np.ndarray:
    mat = loadmat(download_file(fid))
    key = next(k for k in mat if k.endswith("_DE_time"))  # drive-end accel.
    return mat[key].ravel().astype(np.float64)


# ----------------------------------------------------------------------------
# 2. Splitting (done on raw signals, BEFORE windowing)
# ----------------------------------------------------------------------------
def make_splits(rng: np.random.Generator):
    """Return {split: list of (signal_array, label_idx, file_id)}."""
    splits = {"train": [], "val": [], "test": []}

    for label, (cls, ids) in enumerate(FILES.items()):
        if SPLIT_MODE == "file":
            ids = list(rng.permutation(ids))
            n_val = max(1, round(len(ids) * VAL_FRAC))
            n_test = max(1, round(len(ids) * (1 - TRAIN_FRAC - VAL_FRAC)))
            assign = {
                "test": ids[:n_test],
                "val": ids[n_test:n_test + n_val],
                "train": ids[n_test + n_val:],
            }
            for split, fids in assign.items():
                for fid in fids:
                    splits[split].append((load_signal(fid), label, fid))
        else:  # "segment": chronological cut of every recording
            for fid in ids:
                sig = load_signal(fid)
                n = len(sig)
                a = int(n * TRAIN_FRAC)
                b = int(n * (TRAIN_FRAC + VAL_FRAC))
                # leave a WIN-sized gap so no window straddles two splits
                splits["train"].append((sig[:a - WIN], label, fid))
                splits["val"].append((sig[a:b - WIN], label, fid))
                splits["test"].append((sig[b:], label, fid))
    return splits


# ----------------------------------------------------------------------------
# 3. Windowing + feature extraction
# ----------------------------------------------------------------------------
def windows(sig: np.ndarray):
    for s in range(0, len(sig) - WIN + 1, STRIDE):
        yield sig[s:s + WIN]


def extract_features(w: np.ndarray) -> np.ndarray:
    w = w - w.mean()
    rms = np.sqrt(np.mean(w ** 2))
    peak = np.max(np.abs(w))
    feats = [
        rms, peak, np.ptp(w), np.std(w),
        kurtosis(w), skew(w),
        peak / (rms + 1e-12),                      # crest factor
        rms / (np.mean(np.abs(w)) + 1e-12),        # shape factor
        peak / (np.mean(np.sqrt(np.abs(w))) ** 2 + 1e-12),  # clearance factor
    ]
    # --- frequency domain (FFT) ---
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) / len(w)
    bands = np.array_split(spec, N_FFT_BANDS)       # equal-width bands 0..FS/2
    band_energy = [np.sum(b ** 2) for b in bands]
    freqs = np.fft.rfftfreq(len(w), 1 / FS)
    p = spec ** 2 / (np.sum(spec ** 2) + 1e-12)
    centroid = np.sum(freqs * p)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * p))
    feats += band_energy + [centroid, spread, freqs[np.argmax(spec)]]
    return np.array(feats)


FEATURE_NAMES = (
    ["rms", "peak", "p2p", "std", "kurtosis", "skew", "crest", "shape", "clearance"]
    + [f"fft_band_{i}" for i in range(N_FFT_BANDS)]
    + ["spec_centroid", "spec_spread", "dominant_freq"]
)


def build_xy(items):
    X, y, g = [], [], []
    for sig, label, fid in items:
        for w in windows(sig):
            X.append(extract_features(w))
            y.append(label)
            g.append(fid)
    return np.array(X), np.array(y), np.array(g)


# ----------------------------------------------------------------------------
# 4. Models
# ----------------------------------------------------------------------------
def candidate_models():
    models = {}
    for n in (100, 300):
        for d in (None, 12):
            models[f"RF(n={n},depth={d})"] = RandomForestClassifier(
                n_estimators=n, max_depth=d, class_weight="balanced",
                n_jobs=-1, random_state=SEED)
    if HAS_XGB:
        for d in (4, 6):
            for lr in (0.05, 0.1):
                models[f"XGB(depth={d},lr={lr})"] = XGBClassifier(
                    n_estimators=300, max_depth=d, learning_rate=lr,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="mlogloss", n_jobs=-1, random_state=SEED)
    return models


def main():
    rng = np.random.default_rng(SEED)
    print(f"Loading data (split mode = {SPLIT_MODE}) ...")
    splits = make_splits(rng)

    print("Extracting features ...")
    X_tr, y_tr, _ = build_xy(splits["train"])
    X_va, y_va, _ = build_xy(splits["val"])
    X_te, y_te, _ = build_xy(splits["test"])
    print(f"  train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")
    for name, y in (("train", y_tr), ("val", y_va), ("test", y_te)):
        counts = {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))}
        print(f"  {name} class counts: {counts}")

    # Scaler fitted on TRAIN only
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = (scaler.transform(a) for a in (X_tr, X_va, X_te))

    # ---- model selection on the VALIDATION set ----
    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidate_models().items():
        model.fit(X_tr, y_tr)
        f1 = f1_score(y_va, model.predict(X_va), average="macro")
        print(f"  {name:28s} val macro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1
    print(f"\nBest on validation: {best_name} (macro-F1 {best_f1:.4f})")

    # ---- final evaluation on the TEST set (used once) ----
    pred = best_model.predict(X_te)
    print("\n=== TEST SET RESULTS ===")
    print(f"Accuracy : {accuracy_score(y_te, pred):.4f}")
    print(f"Macro F1 : {f1_score(y_te, pred, average='macro'):.4f}\n")
    print(classification_report(y_te, pred, target_names=CLASSES, digits=4))

    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, display_labels=CLASSES, cmap="Blues", xticks_rotation=30)
    plt.title(f"Test confusion matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    # Feature importance (tree models)
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        top = np.argsort(imp)[::-1][:15]
        plt.figure(figsize=(7, 5))
        plt.barh([FEATURE_NAMES[i] for i in top][::-1], imp[top][::-1])
        plt.title("Top 15 features")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)

    # Example FFT plot per class
    plt.figure(figsize=(9, 5))
    for label, cls in enumerate(CLASSES):
        sig = next(s for s, l, _ in splits["train"] if l == label)
        w = sig[:WIN] - sig[:WIN].mean()
        plt.plot(np.fft.rfftfreq(WIN, 1 / FS),
                 np.abs(np.fft.rfft(w)) / WIN, label=cls, alpha=0.8)
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude"); plt.legend()
    plt.title("FFT of one window per class"); plt.tight_layout()
    plt.savefig("fft_examples.png", dpi=150)

    joblib.dump({"model": best_model, "scaler": scaler, "classes": CLASSES},
                "bearing_fault_model.joblib")
    print("Saved: bearing_fault_model.joblib, confusion_matrix.png, "
          "feature_importance.png, fft_examples.png")


if __name__ == "__main__":
    main()
