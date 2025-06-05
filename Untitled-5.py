# %%
"""
Label Noise Detection with Cleanlab – Jupyter Edition
====================================================

This notebook identifies potentially mislabeled entries in a binary fraud vs.
licit dataset using **Cleanlab**’s `find_label_issues` function. It then
re-trains a RandomForest on the cleaned subset to compare F1 performance.

Steps:
1. Install required packages (`cleanlab`, `tqdm`, `matplotlib`).
2. Load dataset from `DATA_PATH`; ensure fraud=1, licit=0 mapping.
3. Train baseline RandomForest and compute F1 on a hold-out set.
4. Perform 5‑fold CV `predict_proba` on full data; flag low-confidence labels
   via Cleanlab.
5. Remove suspect rows, retrain RF, and report Δ F1.
6. Display top 10 suspect indices and plot noise vs clean counts.
7. Perform 3D PCA on full feature set and visualize suspect vs clean samples.

> **Note:** Adjust `DATA_PATH`, `CV_FOLDS`, and `RANDOM_STATE` as needed.
"""

# %%
%pip install -q cleanlab tqdm matplotlib scikit-learn plotly

# %%
# -----------------------------------------------------------------------------
# 1. Imports & configuration
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

# Configuration ---------------------------------------------------------------
BASE_DIR = Path("../../../data/dataset_with_label")
USE_CASE  = "chainabuse"
FILE_NAME = "chainabuse_fraud_elliptic_licit_processed.csv"
DATA_PATH = BASE_DIR / USE_CASE / FILE_NAME
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Logging ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cleanlab-pipeline")

# Verify dataset path exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")


# %%
# -----------------------------------------------------------------------------
# 2. Load dataset and map labels
# -----------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
if "class" not in df.columns:
    raise KeyError("Column 'class' not found in dataset")

le = LabelEncoder()
le.fit(df["class"])  # Expecting labels {"fraud","licit"}

# Ensure fraud=1, licit=0 ------------------------------------------------------
if set(le.classes_) != {"fraud", "licit"}:
    raise ValueError("Expected labels 'fraud' and 'licit' only")

fraud_is_index1 = list(le.classes_).index("fraud") == 1
labels = le.transform(df["class"])  # 0/1 but check order
if not fraud_is_index1:
    labels = 1 - labels  # Flip so fraud=1, licit=0
log.info("Label encoding: %s -> [0,1] if fraud at index1? %s", list(le.classes_), fraud_is_index1)
log.info("Class counts: %s", dict(pd.Series(labels).value_counts()))

df["y"] = labels
X = df.drop(columns=["class", "y"]).values
y = df["y"].values

# %%
# -----------------------------------------------------------------------------
# 3. Train/test split & baseline RandomForest
# -----------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

clf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight="balanced", random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
baseline_f1 = f1_score(y_test, y_pred)
log.info("Baseline F1 on hold‑out set: %.3f", baseline_f1)

# %%
# -----------------------------------------------------------------------------
# 4. Cleanlab: detect potential label issues via cross‑validated predict_proba
# -----------------------------------------------------------------------------

log.info("Running %s‑fold CV predict_proba on full dataset…", CV_FOLDS)
probs_full = cross_val_predict(clf, X, y, cv=CV_FOLDS, method="predict_proba", n_jobs=-1)

noise_indices = find_label_issues(labels=y, pred_probs=probs_full, return_indices_ranked_by="self_confidence")
noise_rate = len(noise_indices) / len(y)
log.info("Detected %s suspect labels (%.2f%% of data)", len(noise_indices), 100 * noise_rate)

# %%
# -----------------------------------------------------------------------------
# 5. Remove suspect rows and retrain RandomForest on cleaned subset
# -----------------------------------------------------------------------------

mask_clean = np.ones(len(y), dtype=bool)
mask_clean[noise_indices] = False

X_clean = X[mask_clean]
y_clean = y[mask_clean]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clean, y_clean, test_size=TEST_SIZE, stratify=y_clean, random_state=RANDOM_STATE
)

clf_clean = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight="balanced", random_state=RANDOM_STATE)
clf_clean.fit(Xc_train, yc_train)

yc_pred = clf_clean.predict(Xc_test)
clean_f1 = f1_score(yc_test, yc_pred)
log.info("F1 after removing suspect labels: %.3f (Δ = %.3f)", clean_f1, clean_f1 - baseline_f1)

# %%
# -----------------------------------------------------------------------------
# 6. Display top 10 suspect indices and plot noise vs clean counts
# -----------------------------------------------------------------------------

print("Top 10 indices flagged as potential label errors:")
print(noise_indices[:10])

# Plot count of suspect vs clean labels
total = len(y)
num_suspect = len(noise_indices)
num_clean = total - num_suspect
plt.figure(figsize=(6,4))
plt.bar(["Clean","Suspect"], [num_clean, num_suspect], color=["#4CAF50", "#F44336"])
plt.title("Label Noise: Clean vs Suspect Samples")
plt.ylabel("Count")
plt.show()

# %%
# -----------------------------------------------------------------------------
# 7. Interactive 3D PCA visualization with Plotly
# -----------------------------------------------------------------------------

import plotly.express as px

# Fit PCA to full feature set
pca = PCA(n_components=3, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

# Create DataFrame for plotting
df_plot = pd.DataFrame(
    X_pca, columns=["PC1", "PC2", "PC3"],
)
# Add label type: Clean vs Suspect
labels_type = np.where(mask_suspect, "Suspect", "Clean")
df_plot["LabelType"] = labels_type

# Interactive 3D scatter
fig = px.scatter_3d(
    df_plot,
    x="PC1", y="PC2", z="PC3",
    color="LabelType",
    color_discrete_map={"Clean": "#4CAF50", "Suspect": "#F44336"},
    title="Interactive 3D PCA: Clean vs Suspect Samples",
    opacity=0.7,
    width=800,
    height=600
)
fig.update_layout(scene=dict(
    xaxis_title="PC1",
    yaxis_title="PC2",
    zaxis_title="PC3"
))
fig.show()


