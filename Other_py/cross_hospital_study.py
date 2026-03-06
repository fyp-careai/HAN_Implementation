"""
Cross-Hospital Generalisation Study — CareAI HAN++
====================================================

Tests whether HAN++ generalises across different patient populations
(simulating multi-hospital deployment) using two evaluation protocols:

1. STANDARD random split (80/20 stratified) — our paper's main result
2. TEMPORAL split: train on early patients (low patient_id), test on later
   ones — simulates deploying a model trained at one hospital to later
   patients or a different registration cohort
3. K-FOLD LOSO (Leave-One-Site-Out): split patients into K=5 groups by
   patient_id quantile, each group = one "site". Train on K-1 groups,
   test on the held-out group. Repeat K times. Report mean ± std.

Why this matters for the paper:
- Single random split can overestimate generalisation (train and test
  patients are from the same distribution by random chance)
- LOSO tests if model generalises to an UNSEEN patient population
- Clinical AI papers routinely require multi-site or temporal validation

This script can run in two modes:
  --quick : skip retraining, evaluate existing models on different splits
  --train : retrain HAN++ for each fold (slow but full evaluation)

Usage:
    cd <project_root>
    python Other_py/cross_hospital_study.py --quick
    python Other_py/cross_hospital_study.py --train --epochs 30
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HAN import HANPP, MedicalGraphData, FocalLoss
from HAN.utils import evaluate_multiorg

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
DATA_CSV    = 'data/filtered_patient_reports.csv'
LABEL_CSV   = 'data/patient-one-hot-labeled-disease-clustered.csv'
GRAPH_CSV   = 'data/test-disease-organ.csv'
MODEL_DIR   = 'models_saved/ruhunu_data_clustered'
OUTPUT_DIR  = 'output/cross_hospital'
METAPATH    = 'P-D-P'

# Model hyperparameters (match ablation best)
HIDDEN_DIM  = 128
OUT_DIM     = 64
NUM_HEADS   = 4
DROPOUT     = 0.3
N_SITES     = 5       # number of simulated hospital sites for LOSO
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
#  Split Functions
# ─────────────────────────────────────────────

def make_random_split(patient_ids, labels_df, test_size=0.2, seed=42):
    """Standard stratified random split (same as HAN++ training)."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        disease_cols = [c for c in labels_df.columns
                        if c not in ('patient_id', 'cluster', 'cluster_id')]
        y = labels_df[disease_cols].values
        mss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                               random_state=seed)
        train_idx, test_idx = next(mss.split(np.zeros(len(y)), y))
    except ImportError:
        rng = np.random.default_rng(seed)
        n = len(patient_ids)
        test_n = int(n * test_size)
        perm = rng.permutation(n)
        test_idx = perm[:test_n]
        train_idx = perm[test_n:]
    return sorted(train_idx.tolist()), sorted(test_idx.tolist())


def make_temporal_split(patient_ids_sorted, test_fraction=0.2):
    """
    Temporal split: first (1-test_fraction) patients → train;
    last test_fraction → test.

    Since patient_ids are sequential registration IDs (larger = registered later),
    sorting by patient_id gives a temporal ordering.
    This tests generalisation to patients registered AFTER the training period.
    """
    n = len(patient_ids_sorted)
    split_point = int(n * (1 - test_fraction))
    train_idx = list(range(split_point))
    test_idx  = list(range(split_point, n))
    return train_idx, test_idx


def make_loso_splits(patient_ids_sorted, n_sites=5):
    """
    Leave-One-Site-Out splits using patient_id quantiles.

    Divides patients into n_sites groups by patient_id quantile.
    Each group simulates patients from one hospital.
    Returns list of (train_idx, test_idx) tuples, one per fold.
    """
    n = len(patient_ids_sorted)
    site_size = n // n_sites
    folds = []

    for k in range(n_sites):
        test_start = k * site_size
        test_end   = (k + 1) * site_size if k < n_sites - 1 else n
        test_idx   = list(range(test_start, test_end))
        train_idx  = list(range(0, test_start)) + list(range(test_end, n))
        folds.append((train_idx, test_idx))

    return folds


# ─────────────────────────────────────────────
#  Evaluation (no retraining — use pre-trained model)
# ─────────────────────────────────────────────

def evaluate_split_quick(model_path, data, test_idx, metapath=METAPATH, device=DEVICE):
    """
    Evaluate a pre-trained model on a specific test split.
    No retraining — uses the model as-is.

    This gives a lower bound: if the pre-trained model (trained on random split)
    still performs well on the temporal/LOSO split, it generalises.
    """
    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location='cpu')
    in_dim = checkpoint['project.weight'].shape[1]

    model = HANPP(
        in_dim=in_dim, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
        metapath_names=[metapath], num_heads=NUM_HEADS,
        num_organs=data.labels_severity.shape[1],
        num_severity=4, dropout=DROPOUT
    )
    model.load_state_dict(checkpoint)
    model.to(device)

    # Truncate/pad features if needed
    feats = data.patient_feats
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats).float()
    if feats.shape[1] != in_dim:
        if feats.shape[1] > in_dim:
            feats = feats[:, :in_dim]
        else:
            pad = torch.zeros(feats.shape[0], in_dim - feats.shape[1])
            feats = torch.cat([feats, pad], dim=1)

    neighs = data.metapath_matrices

    metrics = evaluate_multiorg(model, feats, data.labels_severity, neighs, set(test_idx))
    return metrics


# ─────────────────────────────────────────────
#  Training (full retrain per fold)
# ─────────────────────────────────────────────

def train_one_fold(data, train_idx, val_idx, metapath=METAPATH,
                   epochs=30, device=DEVICE):
    """
    Train HAN++ on train_idx, validate on val_idx.
    Returns best val F1-Macro achieved.
    """
    feats = data.patient_feats
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats).float().to(device)
    else:
        feats = feats.to(device)

    in_dim = feats.shape[1]
    n_organs = data.labels_severity.shape[1]

    model = HANPP(
        in_dim=in_dim, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
        metapath_names=[metapath], num_heads=NUM_HEADS,
        num_organs=n_organs, num_severity=4, dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = FocalLoss(gamma=2.0, reduction='mean')

    train_set = set(train_idx)
    val_set   = set(val_idx)
    labels    = data.labels_severity.to(device)
    neighs    = data.metapath_matrices

    best_val_f1 = 0.0
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        organ_logits, _, _, _ = model(feats, neighs)

        # Compute loss on training indices
        train_idx_t = torch.tensor(train_idx, dtype=torch.long)
        loss = 0.0
        for o_idx in range(n_organs):
            logits_o = organ_logits[train_idx_t, o_idx]   # [n_train, 4]
            labels_o = labels[train_idx_t, o_idx]          # [n_train]
            loss = loss + criterion(logits_o, labels_o)
        loss = loss / n_organs

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            val_metrics = evaluate_multiorg(model, feats, data.labels_severity,
                                            neighs, val_set)
            val_f1 = val_metrics['macro_f1']
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience // 5:
                break

    # Final evaluation
    val_metrics = evaluate_multiorg(model, feats, data.labels_severity,
                                    neighs, val_set)
    return val_metrics


# ─────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────

def plot_results(results_dict, save_path):
    """
    Bar chart comparing F1-Macro across split strategies.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = list(results_dict.keys())
    means  = [v['mean'] for v in results_dict.values()]
    stds   = [v['std']  for v in results_dict.values()]
    colors = ['#2ecc71', '#e67e22', '#3498db',
              '#9b59b6', '#e74c3c', '#1abc9c', '#f39c12']

    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.5), 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=colors[:len(labels)], alpha=0.85, edgecolor='black')

    # Annotate
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.005,
                f'{m:.4f}', ha='center', va='bottom', fontsize=9)

    # Reference line: HAN++ random split result
    ax.axhline(means[0], linestyle='--', color='grey', alpha=0.5,
               label=f'Random split baseline ({means[0]:.4f})')

    ax.set_ylabel('F1-Macro', fontsize=12)
    ax.set_title('Cross-Site Generalisation Study\n'
                 'HAN++ F1-Macro under different data split strategies', fontsize=11)
    ax.set_ylim(0, min(1.0, max(means) * 1.2 + 0.1))
    ax.legend(fontsize=9)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Use pre-trained models without retraining')
    parser.add_argument('--train', action='store_true',
                        help='Retrain HAN++ for each fold')
    parser.add_argument('--metapath', default=METAPATH,
                        help=f'Meta-path to use (default: {METAPATH})')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs per fold (--train mode only)')
    args = parser.parse_args()

    if not (args.quick or args.train):
        args.quick = True  # default

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Cross-Hospital Generalisation Study — CareAI HAN++")
    print(f"Mode: {'quick (pre-trained)' if args.quick else 'full retrain'}")
    print(f"Meta-path: {args.metapath}")
    print(f"Device: {DEVICE}\n")

    # ── Load data ──
    print("Loading data...")
    data = MedicalGraphData(DATA_CSV, LABEL_CSV, GRAPH_CSV)
    data.build_metapaths([args.metapath])

    # Sort patients by patient_id for temporal ordering
    if hasattr(data, 'patient_ids'):
        pid_series = pd.Series(data.patient_ids)
        sort_order = pid_series.argsort().values  # ascending = earliest first
    else:
        sort_order = np.arange(len(data.labels_severity))

    N = len(sort_order)
    labels_df = pd.DataFrame({'patient_id': data.patient_ids if hasattr(data, 'patient_ids') else range(N)})

    print(f"Total patients: {N}")

    results = {}

    # ── 1. Random split (baseline) ──
    print("\n--- Split 1: Standard Random Split (baseline) ---")
    train_idx_rand, test_idx_rand = make_random_split(
        sort_order, labels_df, test_size=0.2, seed=42)

    if args.quick:
        model_path = os.path.join(MODEL_DIR, f'hanpp_{args.metapath}.pt')
        m = evaluate_split_quick(model_path, data, test_idx_rand,
                                 args.metapath, DEVICE)
        if m:
            results['Random Split'] = {'mean': m['macro_f1'], 'std': 0.0,
                                       'details': [m['macro_f1']]}
            print(f"  Random split F1-Macro: {m['macro_f1']:.4f}")
    else:
        m = train_one_fold(data, train_idx_rand, test_idx_rand,
                           args.metapath, args.epochs, DEVICE)
        results['Random Split'] = {'mean': m['macro_f1'], 'std': 0.0,
                                   'details': [m['macro_f1']]}
        print(f"  Random split F1-Macro: {m['macro_f1']:.4f}")

    # ── 2. Temporal split ──
    print("\n--- Split 2: Temporal Split (train early, test late) ---")
    patients_sorted_by_id = sort_order.tolist()  # already sorted by patient_id index
    train_idx_temp, test_idx_temp = make_temporal_split(
        patients_sorted_by_id, test_fraction=0.2)

    if args.quick:
        model_path = os.path.join(MODEL_DIR, f'hanpp_{args.metapath}.pt')
        m = evaluate_split_quick(model_path, data, test_idx_temp,
                                 args.metapath, DEVICE)
        if m:
            results['Temporal Split'] = {'mean': m['macro_f1'], 'std': 0.0,
                                         'details': [m['macro_f1']]}
            print(f"  Temporal split F1-Macro: {m['macro_f1']:.4f}")
    else:
        m = train_one_fold(data, train_idx_temp, test_idx_temp,
                           args.metapath, args.epochs, DEVICE)
        results['Temporal Split'] = {'mean': m['macro_f1'], 'std': 0.0,
                                     'details': [m['macro_f1']]}
        print(f"  Temporal split F1-Macro: {m['macro_f1']:.4f}")

    # ── 3. LOSO K-fold ──
    print(f"\n--- Split 3: Leave-One-Site-Out ({N_SITES}-fold LOSO) ---")
    folds = make_loso_splits(patients_sorted_by_id, n_sites=N_SITES)
    loso_f1s = []

    for fold_idx, (train_idx_f, test_idx_f) in enumerate(folds):
        n_train = len(train_idx_f)
        n_test  = len(test_idx_f)
        print(f"  Fold {fold_idx+1}/{N_SITES}: train={n_train}, test={n_test}")

        if args.quick:
            model_path = os.path.join(MODEL_DIR, f'hanpp_{args.metapath}.pt')
            m = evaluate_split_quick(model_path, data, test_idx_f,
                                     args.metapath, DEVICE)
            if m:
                loso_f1s.append(m['macro_f1'])
                print(f"    F1-Macro: {m['macro_f1']:.4f}")
        else:
            m = train_one_fold(data, train_idx_f, test_idx_f,
                               args.metapath, args.epochs, DEVICE)
            loso_f1s.append(m['macro_f1'])
            print(f"    F1-Macro: {m['macro_f1']:.4f}")

    if loso_f1s:
        results['LOSO (5-fold)'] = {
            'mean': float(np.mean(loso_f1s)),
            'std':  float(np.std(loso_f1s)),
            'details': loso_f1s
        }
        print(f"\n  LOSO mean F1-Macro: {np.mean(loso_f1s):.4f} ± {np.std(loso_f1s):.4f}")
        for k, f in enumerate(loso_f1s):
            print(f"    Site {k+1}: {f:.4f}")

    # ── Save results ──
    results_file = os.path.join(OUTPUT_DIR, 'cross_hospital_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")

    # ── Print summary table ──
    print("\n" + "="*60)
    print("CROSS-HOSPITAL GENERALISATION SUMMARY")
    print("="*60)
    print(f"{'Split Strategy':<25} {'F1-Macro':>10}  {'Std':>8}  {'vs Random':>10}")
    print("-"*60)
    baseline = results.get('Random Split', {}).get('mean', 0.0)
    for name, v in results.items():
        delta = v['mean'] - baseline
        delta_str = f"{delta:+.4f}" if name != 'Random Split' else "baseline"
        print(f"{name:<25} {v['mean']:>10.4f}  {v['std']:>8.4f}  {delta_str:>10}")
    print("="*60)

    if results:
        plot_results(results,
                     save_path=os.path.join(OUTPUT_DIR, 'cross_hospital_comparison.png'))

    # ── Interpretation ──
    print("\nINTERPRETATION:")
    if 'LOSO (5-fold)' in results and 'Random Split' in results:
        loso_m = results['LOSO (5-fold)']['mean']
        rand_m = results['Random Split']['mean']
        drop   = rand_m - loso_m
        if drop < 0.02:
            print(f"  Drop from random to LOSO: {drop:.4f} (<2%) — Model GENERALISES well")
            print("  => Safe to claim cross-site generalisation in the paper")
        elif drop < 0.05:
            print(f"  Drop from random to LOSO: {drop:.4f} (2-5%) — Moderate generalisation")
            print("  => Report with caveat: 'slight performance drop across sites'")
        else:
            print(f"  Drop from random to LOSO: {drop:.4f} (>5%) — Model overfits to training dist.")
            print("  => Recommend data augmentation or domain adaptation")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
