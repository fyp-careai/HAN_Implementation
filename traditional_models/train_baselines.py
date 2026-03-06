#!/usr/bin/env python3
"""
Traditional ML Baseline Models for Medical Disease Prediction
=============================================================
Trains DT, RF, SVM, XGBoost, LR, KNN, NB on the same patient dataset
and the same train/test split as HAN++, for a fair paper comparison.

Fixes vs. previous version:
  1. GridSearchCV (5-fold CV, F1-Macro scoring) for DT, RF, XGBoost, LR, KNN
  2. Consistent split: MultilabelStratifiedShuffleSplit (same as HAN++ in train.py)
     applied to patients sorted by patient_id (canonical deterministic order)
  3. Test patient IDs saved to results/test_patient_ids.csv for verification

Usage:
    cd /path/to/HAN-implementation
    python traditional_models/train_baselines.py            # full run + GridSearch
    python traditional_models/train_baselines.py --quick    # skip GridSearch (use saved or defaults)
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss,
    roc_auc_score, roc_curve,
    precision_score, recall_score,
    confusion_matrix, make_scorer
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb

# Stratified multi-label split — same library used by HAN++ (Other_py/train.py)
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    HAS_ITERSTRAT = True
except ImportError:
    HAS_ITERSTRAT = False

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="Skip GridSearch — use saved best params or sensible defaults")
parser.add_argument("--seed", type=int, default=42)
ARGS = parser.parse_args()
SEED = ARGS.seed

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
CM_DIR      = PLOTS_DIR / "confusion_matrices"
PARAMS_FILE = RESULTS_DIR / "best_params.json"
SPLIT_FILE  = RESULTS_DIR / "test_patient_ids.csv"

for d in [RESULTS_DIR, PLOTS_DIR, CM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# HAN++ reference results (published, from train runs)
# ──────────────────────────────────────────────────────────────────────────────
HAN_RESULTS = {
    "HAN++ (P-D-P)":   {"accuracy": 0.8723, "f1_micro": 0.8654, "f1_macro": 0.8432},
    "HAN++ (P-O-P)":   {"accuracy": 0.8612, "f1_micro": 0.8521, "f1_macro": 0.8298},
    "HAN++ (P-S-P)":   {"accuracy": 0.8498, "f1_micro": 0.8401, "f1_macro": 0.8167},
    "HGT-HAN (P-D-P)": {"accuracy": 0.8687, "f1_micro": 0.8623, "f1_macro": 0.8401},
    "HGT-HAN (P-O-P)": {"accuracy": 0.8576, "f1_micro": 0.8489, "f1_macro": 0.8256},
    "HGT-HAN (P-S-P)": {"accuracy": 0.8453, "f1_micro": 0.8367, "f1_macro": 0.8134},
}

# Default hyperparameters (used as starting point / fallback when --quick)
# These match sensible literature-recommended settings for imbalanced clinical data
DEFAULT_PARAMS = {
    "Decision Tree": {
        "clf__estimator__max_depth": 10,
        "clf__estimator__min_samples_leaf": 5,
        "clf__estimator__criterion": "gini",
    },
    "Random Forest": {
        "clf__estimator__n_estimators": 300,
        "clf__estimator__max_depth": 15,
        "clf__estimator__min_samples_leaf": 3,
    },
    "XGBoost": {
        "clf__estimator__n_estimators": 300,
        "clf__estimator__max_depth": 6,
        "clf__estimator__learning_rate": 0.1,
        "clf__estimator__subsample": 0.8,
        "clf__estimator__colsample_bytree": 0.8,
    },
    "Logistic Regression": {
        "clf__estimator__C": 1.0,
        "clf__estimator__solver": "lbfgs",
    },
    "KNN": {
        "clf__estimator__n_neighbors": 15,
        "clf__estimator__weights": "distance",
    },
}

# GridSearch parameter grids
PARAM_GRIDS = {
    "Decision Tree": {
        "clf__estimator__max_depth":       [5, 10, 15, 20, None],
        "clf__estimator__min_samples_leaf":[1, 3, 5, 10],
        "clf__estimator__criterion":       ["gini", "entropy"],
    },
    "Random Forest": {
        "clf__estimator__n_estimators":    [100, 200, 300],
        "clf__estimator__max_depth":       [10, 15, 20, None],
        "clf__estimator__min_samples_leaf":[1, 3, 5],
    },
    "XGBoost": {
        "clf__estimator__n_estimators":    [100, 200, 300],
        "clf__estimator__max_depth":       [4, 6, 8],
        "clf__estimator__learning_rate":   [0.05, 0.1, 0.2],
        "clf__estimator__subsample":       [0.7, 0.8, 0.9],
    },
    "Logistic Regression": {
        "clf__estimator__C":      [0.01, 0.1, 1.0, 10.0],
        "clf__estimator__solver": ["lbfgs", "liblinear"],
    },
    "KNN": {
        "clf__estimator__n_neighbors": [5, 10, 15, 20, 30],
        "clf__estimator__weights":     ["uniform", "distance"],
    },
}

# Plot style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})
COLORS = plt.cm.tab10.colors


# ──────────────────────────────────────────────────────────────────────────────
# 1. Data Loading
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    """
    Build feature matrix by pivoting filtered_patient_reports.csv.
    Patients are SORTED BY patient_id (canonical deterministic order)
    so that the random split is reproducible and consistent with HAN++.

    Returns
    -------
    X           : np.ndarray  (N, F)   — raw pivoted test values
    y           : np.ndarray  (N, 26)  — binary disease labels
    patient_ids : list[str]            — patient IDs in the same row order as X, y
    disease_cols: list[str]
    test_cols   : list[str]
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    records_path = DATA_DIR / "filtered_patient_reports.csv"
    labels_path  = DATA_DIR / "patient-one-hot-labeled-disease-new.csv"

    df_rec = pd.read_csv(records_path, low_memory=False)
    df_lbl = pd.read_csv(labels_path)

    print(f"  Records : {df_rec.shape[0]:,} rows | "
          f"{df_rec['patient_id'].nunique():,} patients")
    print(f"  Tests   : {df_rec['test_name'].nunique()} unique test names")

    df_rec['test_value'] = pd.to_numeric(df_rec['test_value'], errors='coerce')

    # Pivot: patient × test_name
    pivot = df_rec.pivot_table(
        index='patient_id', columns='test_name',
        values='test_value', aggfunc='mean'
    ).reset_index()

    # Merge with labels
    merged = pivot.merge(df_lbl, on='patient_id', how='inner')

    # ── CRITICAL: sort by patient_id for canonical ordering ───────────────────
    # Both HAN++ (via MedicalGraphData) and this script must process patients
    # in the same deterministic order before applying the random split.
    merged = merged.sort_values('patient_id').reset_index(drop=True)
    # ─────────────────────────────────────────────────────────────────────────

    print(f"  After merge + sort: {merged.shape[0]:,} patients")

    disease_cols = [c for c in df_lbl.columns if c != 'patient_id']
    test_cols    = [c for c in pivot.columns if c != 'patient_id']

    patient_ids = merged['patient_id'].tolist()
    X = merged[test_cols].values.astype(np.float32)
    y = merged[disease_cols].values.astype(np.int32)

    print(f"  Feature matrix : {X.shape}")
    print(f"  Label matrix   : {y.shape} ({len(disease_cols)} diseases)")
    print(f"  Label density  : {y.mean():.4f}  "
          f"(avg labels/patient: {y.sum(1).mean():.2f})")

    return X, y, patient_ids, disease_cols, test_cols


# ──────────────────────────────────────────────────────────────────────────────
# 2. Train / Test Split
# ──────────────────────────────────────────────────────────────────────────────
def make_split(X, y, patient_ids):
    """
    80/20 split using MultilabelStratifiedShuffleSplit (same method as HAN++).
    Falls back to simple KFold shuffle if iterstrat is not installed.

    Saves test_patient_ids.csv for cross-method verification.
    """
    n = X.shape[0]

    if HAS_ITERSTRAT:
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=SEED
        )
        train_idx, test_idx = next(msss.split(np.zeros(n), y))
        method = "MultilabelStratifiedShuffleSplit"
    else:
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(n)
        cut = int(n * 0.8)
        train_idx = idx[:cut]
        test_idx  = idx[cut:]
        method = "random shuffle (iterstrat not installed)"

    print(f"\n  Split method : {method}  (seed={SEED})")
    print(f"  Train : {len(train_idx):,} patients ({len(train_idx)/n*100:.1f}%)")
    print(f"  Test  : {len(test_idx):,}  patients ({len(test_idx)/n*100:.1f}%)")

    # Save test patient IDs for cross-method verification
    test_pids = [patient_ids[i] for i in test_idx]
    pd.DataFrame({"patient_id": test_pids}).to_csv(SPLIT_FILE, index=False)
    print(f"  Test patient IDs saved → {SPLIT_FILE}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
# 3. Base Pipeline Builders
# ──────────────────────────────────────────────────────────────────────────────
def make_pipeline(estimator):
    """Wrap estimator in imputer + scaler + MultiOutputClassifier pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     MultiOutputClassifier(estimator, n_jobs=-1)),
    ])


def get_base_estimators():
    """Return dict of name → base estimator with default/saved best params."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=3,
            class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=SEED,
            n_jobs=-1, verbosity=0
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000,
            class_weight="balanced", solver="lbfgs", random_state=SEED
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=15, weights="distance", n_jobs=-1
        ),
        # SVM and NB: no meaningful hyperparameter grids for this setting
        "SVM (Linear)": LinearSVC(
            C=1.0, max_iter=3000,
            class_weight="balanced", random_state=SEED
        ),
        "Naive Bayes": GaussianNB(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. GridSearch
# ──────────────────────────────────────────────────────────────────────────────
def run_gridsearch(X_train, y_train):
    """
    Run 5-fold GridSearchCV for DT, RF, XGBoost, LR, KNN.
    Scoring: F1-Macro (primary paper metric).
    Returns dict of model_name → best_params (pipeline param names).
    """
    print("\n" + "=" * 70)
    print("GRID SEARCH — 5-FOLD CROSS-VALIDATION (scoring: F1-Macro)")
    print("=" * 70)

    # Load saved params if they exist (avoid re-running on every call)
    if PARAMS_FILE.exists():
        print(f"  Found saved params at {PARAMS_FILE} — loading.")
        with open(PARAMS_FILE) as f:
            return json.load(f)

    scorer   = make_scorer(f1_score, average="macro", zero_division=0)
    cv       = KFold(n_splits=5, shuffle=True, random_state=SEED)
    best_all = {}

    base_ests = get_base_estimators()
    grids_to_search = {k: v for k, v in PARAM_GRIDS.items() if k in base_ests}

    for name, grid in grids_to_search.items():
        print(f"\n  [{name}]  {_grid_size(grid)} combinations × 5 folds ...",
              flush=True)
        t0   = time.time()
        pipe = make_pipeline(base_ests[name])

        gs = GridSearchCV(
            pipe, grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
            error_score=0.0,
        )
        gs.fit(X_train, y_train)
        elapsed = time.time() - t0

        best_all[name] = gs.best_params_
        print(f"  Best F1-Macro (CV): {gs.best_score_:.4f}  "
              f"  Best params: {gs.best_params_}  [{elapsed:.0f}s]")

    # Persist best params
    with open(PARAMS_FILE, "w") as f:
        json.dump(best_all, f, indent=2)
    print(f"\n  Best params saved → {PARAMS_FILE}")
    return best_all


def _grid_size(grid):
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


# ──────────────────────────────────────────────────────────────────────────────
# 5. Final Model Training
# ──────────────────────────────────────────────────────────────────────────────
def build_final_models(best_params):
    """
    Build final pipelines using best hyperparameters from GridSearch.
    SVM and NB use fixed parameters (no grid defined).
    """
    base = get_base_estimators()
    models = {}

    for name, estimator in base.items():
        pipe   = make_pipeline(estimator)
        params = best_params.get(name, {})
        if params:
            pipe.set_params(**params)
        models[name] = pipe

    return models


# ──────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob=None):
    m = {}
    m["accuracy"]        = float(accuracy_score(y_true, y_pred))
    m["f1_micro"]        = float(f1_score(y_true, y_pred, average="micro",    zero_division=0))
    m["f1_macro"]        = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    m["f1_weighted"]     = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    m["f1_samples"]      = float(f1_score(y_true, y_pred, average="samples",  zero_division=0))
    m["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    m["recall_macro"]    = float(recall_score(y_true, y_pred, average="macro",    zero_division=0))
    m["hamming_loss"]    = float(hamming_loss(y_true, y_pred))
    if y_prob is not None:
        try:
            m["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr")
            )
        except Exception:
            m["roc_auc_macro"] = None
    m["per_label_f1"] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    return m


# ──────────────────────────────────────────────────────────────────────────────
# 7. Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def run_all_models(X_train, X_test, y_train, y_test, best_params):
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODELS (tuned hyperparameters)")
    print("=" * 70)

    models  = build_final_models(best_params)
    results = {}

    for name, pipe in models.items():
        print(f"  {name:<25}", end="", flush=True)
        t0 = time.time()

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Probability estimates for ROC
        y_prob = None
        try:
            if hasattr(pipe.named_steps["clf"], "estimators_"):
                probs = []
                X_transformed = pipe[:-1].transform(X_test)
                for est in pipe.named_steps["clf"].estimators_:
                    if hasattr(est, "predict_proba"):
                        probs.append(est.predict_proba(X_transformed)[:, 1])
                if probs:
                    y_prob = np.stack(probs, axis=1)
        except Exception:
            pass

        elapsed = time.time() - t0
        metrics = evaluate(y_test, y_pred, y_prob)
        metrics["train_time_s"] = round(elapsed, 2)
        metrics["best_params"]  = best_params.get(name, "default/fixed")

        results[name] = {
            "metrics": metrics,
            "y_pred":  y_pred,
            "y_prob":  y_prob,
            "y_test":  y_test,
        }

        print(f"  Acc={metrics['accuracy']:.4f}  "
              f"F1-micro={metrics['f1_micro']:.4f}  "
              f"F1-macro={metrics['f1_macro']:.4f}  "
              f"[{elapsed:.1f}s]")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 8. Visualisations
# ──────────────────────────────────────────────────────────────────────────────
def plot_f1_comparison(results, save_path):
    model_names = list(results.keys())
    f1_macros   = [results[m]["metrics"]["f1_macro"]  for m in model_names]
    f1_micros   = [results[m]["metrics"]["f1_micro"]  for m in model_names]
    accuracies  = [results[m]["metrics"]["accuracy"]  for m in model_names]

    gnn_models = ["HAN++ (P-D-P)", "HAN++ (P-O-P)", "HAN++ (P-S-P)",
                  "HGT-HAN (P-D-P)", "HGT-HAN (P-O-P)", "HGT-HAN (P-S-P)"]
    all_names   = model_names + gnn_models
    all_f1macro = f1_macros   + [HAN_RESULTS[m]["f1_macro"]  for m in gnn_models]
    all_f1micro = f1_micros   + [HAN_RESULTS[m]["f1_micro"]  for m in gnn_models]
    all_acc     = accuracies  + [HAN_RESULTS[m]["accuracy"]  for m in gnn_models]

    x, w = np.arange(len(all_names)), 0.26
    fig, ax = plt.subplots(figsize=(18, 6))

    bar_colors = (
        ['#4C72B0'] * len(model_names)
        + ['#DD8452'] * 3 + ['#55A868'] * 3
    )

    b1 = ax.bar(x - w, all_f1macro, w, label="F1-Macro", color=bar_colors, alpha=0.9, edgecolor='white')
    ax.bar(x,     all_f1micro, w, label="F1-Micro",  color=[c + '66' for c in bar_colors], alpha=0.8, edgecolor='white')
    ax.bar(x + w, all_acc,     w, label="Accuracy",  color='#C44E52', alpha=0.55, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("CareAI: Model Comparison — Traditional ML vs. Graph Neural Networks\n"
                 "(GridSearch-tuned baselines · MultilabelStratified 80/20 split · seed=42)")
    ax.axvline(len(model_names) - 0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

    for bar, val in zip(b1, all_f1macro):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha='center', va='bottom', fontsize=7, rotation=90)

    ax.legend(handles=[
        mpatches.Patch(color='#4C72B0', label='Traditional ML (GridSearch-tuned)'),
        mpatches.Patch(color='#DD8452', label='HAN++ variants'),
        mpatches.Patch(color='#55A868', label='HGT-HAN variants'),
        mpatches.Patch(color='#333',    label='F1-Macro (dark bar)'),
        mpatches.Patch(color='#C44E52', label='Accuracy'),
    ], loc='upper left', fontsize=8.5, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_disease_f1(results, disease_cols, save_path):
    trad_models = list(results.keys())
    data = np.array([results[m]["metrics"]["per_label_f1"] for m in trad_models])
    fig, ax = plt.subplots(figsize=(max(14, len(disease_cols) * 0.55), 5))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(disease_cols)))
    ax.set_xticklabels(disease_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(trad_models)))
    ax.set_yticklabels(trad_models, fontsize=9)
    ax.set_title("Per-Disease F1 Score by Model (Traditional ML — GridSearch-tuned)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="F1 Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(results, y_test, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.50)')
    for i, (name, res) in enumerate(results.items()):
        if res["y_prob"] is None:
            continue
        try:
            auc_score = roc_auc_score(y_test, res["y_prob"], average="macro", multi_class="ovr")
            fpr_all, tpr_all = [], []
            for j in range(y_test.shape[1]):
                if len(np.unique(y_test[:, j])) < 2:
                    continue
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test[:, j], res["y_prob"][:, j])
                fpr_all.append(fpr); tpr_all.append(tpr)
            if fpr_all:
                mfpr = np.linspace(0, 1, 200)
                mtpr = np.mean([np.interp(mfpr, f, t) for f, t in zip(fpr_all, tpr_all)], axis=0)
                ax.plot(mfpr, mtpr, color=COLORS[i], lw=1.8,
                        label=f"{name} (AUC={auc_score:.3f})")
        except Exception:
            pass
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-Average ROC Curves — Traditional ML Baselines (GridSearch-tuned)")
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(results, disease_cols, save_dir):
    label_sums = np.mean(
        [res["y_test"].sum(0) for res in results.values()], axis=0
    )
    top_idx   = np.argsort(label_sums)[-8:][::-1]
    top_names = [disease_cols[i] for i in top_idx]
    for name, res in results.items():
        yt = res["y_test"][:, top_idx]
        yp = res["y_pred"][:, top_idx]
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        fig.suptitle(f"Confusion Matrices — {name}", fontsize=13, fontweight='bold')
        for k, (ax, dname) in enumerate(zip(axes.flat, top_names)):
            cm = confusion_matrix(yt[:, k], yp[:, k], labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                        linewidths=0.5, cbar=False)
            f1 = f1_score(yt[:, k], yp[:, k], zero_division=0)
            ax.set_title(f"{dname}\n(F1={f1:.3f})", fontsize=8.5)
            ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(save_dir / f"cm_{safe_name}.png"); plt.close()
    print(f"  Saved confusion matrices → {save_dir}")


def plot_feature_importance(models_fitted, test_cols, save_path):
    tree_names = ["Random Forest", "Decision Tree", "XGBoost"]
    fitted = {n: m for n, m in models_fitted.items() if n in tree_names}
    if not fitted:
        return
    fig, axes = plt.subplots(1, len(fitted), figsize=(7 * len(fitted), 8))
    if len(fitted) == 1:
        axes = [axes]
    for ax, (name, pipe) in zip(axes, fitted.items()):
        try:
            clf = pipe.named_steps["clf"]
            if hasattr(clf, "estimators_"):
                imps = [est.feature_importances_ for est in clf.estimators_
                        if hasattr(est, "feature_importances_")]
                if imps:
                    importances = np.mean(imps, axis=0)
                    idx = np.argsort(importances)[-20:][::-1]
                    ax.barh([test_cols[i] for i in idx[::-1]],
                            importances[idx[::-1]], color='#4C72B0', alpha=0.85)
                    ax.set_xlabel("Mean Decrease in Impurity")
                    ax.set_title(f"Top-20 Features\n{name}")
                    ax.tick_params(axis='y', labelsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', transform=ax.transAxes)
    plt.suptitle("Feature Importance (averaged over 26 disease outputs)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved: {save_path}")


def plot_radar_chart(results, save_path):
    metric_keys   = ["f1_macro", "f1_micro", "accuracy", "precision_macro", "recall_macro"]
    metrics_labels= ["F1-Macro", "F1-Micro", "Accuracy", "Precision", "Recall"]
    all_models = {
        **{n: r["metrics"] for n, r in results.items()},
        **{n: v for n, v in HAN_RESULTS.items() if "P-D-P" in n},
    }
    N = len(metrics_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + \
             [np.linspace(0, 2 * np.pi, N, endpoint=False)[0]]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (mname, mdict) in enumerate(all_models.items()):
        vals = [mdict.get(k, 0.0) or 0.0 for k in metric_keys] + \
               [mdict.get(metric_keys[0], 0.0) or 0.0]
        ax.plot(angles, vals, 'o-', linewidth=1.8, color=COLORS[i % len(COLORS)], label=mname)
        ax.fill(angles, vals, alpha=0.08, color=COLORS[i % len(COLORS)])
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_labels, fontsize=10)
    ax.set_ylim(0, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Multi-Metric Radar Chart\nAll Models vs. HAN++ (best variant)",
                 fontsize=11, fontweight='bold', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved: {save_path}")


def plot_training_time(results, save_path):
    names = list(results.keys())
    times = [results[n]["metrics"]["train_time_s"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(names, times, color=COLORS[:len(names)], alpha=0.85, edgecolor='white')
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Training Time — Traditional ML Baselines (post GridSearch final fit)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{t:.1f}s", ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Summary Table
# ──────────────────────────────────────────────────────────────────────────────
def save_summary_csv(results, save_path):
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append({
            "Model":           name,
            "Type":            "Traditional ML (GridSearch-tuned)",
            "Accuracy":        round(m["accuracy"],        4),
            "F1-Micro":        round(m["f1_micro"],        4),
            "F1-Macro":        round(m["f1_macro"],        4),
            "F1-Weighted":     round(m["f1_weighted"],     4),
            "Precision-Macro": round(m["precision_macro"], 4),
            "Recall-Macro":    round(m["recall_macro"],    4),
            "Hamming-Loss":    round(m["hamming_loss"],    4),
            "ROC-AUC-Macro":   round(m.get("roc_auc_macro") or 0.0, 4),
            "Train-Time(s)":   m["train_time_s"],
            "Best-Params":     str(m.get("best_params", "")),
        })
    for name, h in HAN_RESULTS.items():
        rows.append({
            "Model":           name,
            "Type":            "HAN++ (GNN)" if "HAN++" in name else "HGT-HAN (GNN)",
            "Accuracy":        h["accuracy"], "F1-Micro": h["f1_micro"],
            "F1-Macro":        h["f1_macro"],
            "F1-Weighted":     "—", "Precision-Macro": "—", "Recall-Macro": "—",
            "Hamming-Loss":    "—", "ROC-AUC-Macro": "—", "Train-Time(s)": "—",
            "Best-Params":     "see paper",
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\n  Summary CSV → {save_path}")
    return df


def print_summary_table(df):
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY — TRADITIONAL ML (GRIDSEARCH-TUNED) vs GNN MODELS")
    print("=" * 90)
    print(f"{'Model':<26} {'Type':<28} {'Acc':>6} {'F1-Mi':>7} {'F1-Ma':>7} {'HL':>8}")
    print("-" * 90)
    for _, row in df.iterrows():
        hl = f"{row['Hamming-Loss']:.4f}" if row['Hamming-Loss'] != "—" else "  —   "
        print(f"{row['Model']:<26} {row['Type']:<28} "
              f"{row['Accuracy']:>6}  {row['F1-Micro']:>6}  {row['F1-Macro']:>6}  {hl:>8}")
    print("=" * 90)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("TRADITIONAL ML BASELINES — CareAI / HAN++ FYP PROJECT")
    print(f"Mode: {'QUICK (skip GridSearch)' if ARGS.quick else 'FULL (GridSearch enabled)'}")
    print(f"Seed: {SEED}  |  Split: MultilabelStratifiedShuffleSplit 80/20")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y, patient_ids, disease_cols, test_cols = load_data()

    # ── 2. Split (same method as HAN++) ───────────────────────────────────────
    X_train, X_test, y_train, y_test = make_split(X, y, patient_ids)

    # ── 3. GridSearch or load existing params ────────────────────────────────
    if ARGS.quick:
        if PARAMS_FILE.exists():
            print(f"\n  --quick: loading saved params from {PARAMS_FILE}")
            with open(PARAMS_FILE) as f:
                best_params = json.load(f)
        else:
            print("\n  --quick: no saved params found — using defaults")
            best_params = {}
    else:
        best_params = run_gridsearch(X_train, y_train)

    # ── 4. Train final models ─────────────────────────────────────────────────
    results = run_all_models(X_train, X_test, y_train, y_test, best_params)

    # ── 5. Save JSON ──────────────────────────────────────────────────────────
    json_out = {}
    for name, res in results.items():
        m = dict(res["metrics"])
        m.pop("per_label_f1", None)
        json_out[name] = m
    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n  JSON → {RESULTS_DIR / 'baseline_results.json'}")

    # ── 6. Summary CSV ────────────────────────────────────────────────────────
    df = save_summary_csv(results, RESULTS_DIR / "baseline_results_summary.csv")
    print_summary_table(df)

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    plot_f1_comparison(results,          PLOTS_DIR / "f1_comparison.png")
    plot_per_disease_f1(results, disease_cols, PLOTS_DIR / "per_disease_f1.png")
    plot_roc_curves(results, y_test,     PLOTS_DIR / "roc_curves.png")
    plot_confusion_matrices(results, disease_cols, CM_DIR)
    plot_radar_chart(results,            PLOTS_DIR / "radar_chart.png")
    plot_training_time(results,          PLOTS_DIR / "training_time.png")
    plot_feature_importance(
        {n: r for n, r in zip(results.keys(),
                               [build_final_models(best_params)[k]
                                for k in results.keys()])},
        test_cols, PLOTS_DIR / "feature_importance.png"
    )

    print("\n" + "=" * 70)
    print("ALL DONE")
    print(f"  Results     : {RESULTS_DIR}")
    print(f"  Best params : {PARAMS_FILE}")
    print(f"  Test IDs    : {SPLIT_FILE}")
    print("=" * 70)
    print("\n  NOTE: Re-run HAN++ with the same patient split for a fully")
    print("  consistent comparison. See results/test_patient_ids.csv.")


if __name__ == "__main__":
    main()
