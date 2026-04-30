#!/usr/bin/env python3
"""
train.py — HANPP_Disease GPU Training Script
==============================================

Trains the HANPP_Disease model from scratch with:
  - Stratified train/val/test split (70/15/15)
  - Full-graph training on CUDA GPU
  - FocalLoss for class-imbalanced multi-label disease prediction
  - Validation every epoch with F1 metrics
  - Early stopping on validation loss
  - ReduceLROnPlateau learning rate scheduling
  - Post-training disease & organ prototype generation
  - Training metrics plotting

Usage:
    python train.py                          # Train with defaults
    python train.py --epochs 200 --lr 0.002  # Override hyperparams

After training, update inference_v6.py MODEL_PATH / PROTOTYPE_PATH to
point at the new checkpoint.
"""

import os
import sys
import json
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# ── HAN package ──────────────────────────────────────────────────────────────
from HAN.data import MedicalGraphData
from HAN.model import HANPP_Disease
from HAN.utils import (
    FocalLoss,
    neighbors_to_padded_tensors,
    plot_training_metrics,
)
from HAN.inductive import (
    build_disease_prototypes,
    build_organ_prototypes,
    compare_inference_modes,
)
from HAN.validation_metrics import (
    plot_training_metrics_enhanced,
)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    # Data
    'records_path':     'data/HAN_data/merged_coop_ruhunu_patient_data.csv',
    'symptom_path':     'data/HAN_data/test_reference_full_v2.csv',
    'output_dir':       'models_saved/retrained',

    # Data splitting
    'train_ratio':      0.70,
    'val_ratio':        0.15,
    'test_ratio':       0.15,
    'seed':             42,

    # Graph construction
    'symptom_freq_threshold': 0.08,
    'prune_per_patient':      300,
    'nnz_threshold':          80_000_000,
    'metapaths':              ['P-D-P', 'P-O-P'],

    # Model architecture
    'hidden_dim':       256,
    'out_dim':          128,
    'num_heads':        4,
    'dropout':          0.4,

    # Training
    'epochs':           150,
    'lr':               0.001,
    'weight_decay':     1e-4,
    'focal_gamma':      2.0,
    'patience':         20,       # Early stopping patience
    'lr_patience':      10,       # ReduceLROnPlateau patience
    'lr_factor':        0.5,      # LR reduction factor
    'min_lr':           1e-6,

    # Evaluation
    'eval_every':       1,        # Validate every N epochs
    'mc_samples':       50,       # MC-Dropout samples for benchmark
}


def parse_args():
    """Parse command-line arguments, falling back to DEFAULTS."""
    parser = argparse.ArgumentParser(description='Train HANPP_Disease model')

    for key, val in DEFAULTS.items():
        if isinstance(val, list):
            parser.add_argument(f'--{key}', nargs='+', default=val)
        elif isinstance(val, bool):
            parser.add_argument(f'--{key}', action='store_true', default=val)
        else:
            parser.add_argument(f'--{key}', type=type(val), default=val)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Stratified multi-label split
# ─────────────────────────────────────────────────────────────────────────────

def stratified_multilabel_split(labels_np, train_ratio, val_ratio, seed):
    """
    Split patient indices into train/val/test with stratification.

    For multi-label data, we stratify on the most frequent disease per patient
    to ensure each split has proportional disease representation.
    """
    N = labels_np.shape[0]
    all_idx = np.arange(N)

    # Create a single stratification target: most prevalent disease per patient,
    # or -1 for patients with no disease label.
    strat_target = np.full(N, -1, dtype=int)
    for i in range(N):
        positives = np.where(labels_np[i] == 1)[0]
        if len(positives) > 0:
            strat_target[i] = positives[0]  # first positive disease

    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=strat_target,
    )

    # Second split: val vs test
    temp_strat = strat_target[temp_idx]
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_strat,
    )

    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_disease(model, feats_t, labels_t, idx, disease_order, device):
    """
    Evaluate HANPP_Disease on a subset of patients.

    Returns dict with loss, macro_f1, micro_f1, per_disease_f1.
    """
    model.eval()
    empty_nbr = {name: {} for name in model.metapath_names}

    logits, z, beta = model(feats_t, empty_nbr)

    # Loss on subset
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits[idx], labels_t[idx]).item()

    # Predictions
    probs = torch.sigmoid(logits[idx]).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    y_true = labels_t[idx].cpu().numpy().astype(int)

    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, preds, average='micro', zero_division=0)

    per_disease = {}
    for j, disease in enumerate(disease_order):
        per_disease[disease] = f1_score(y_true[:, j], preds[:, j], zero_division=0)

    # Beta (semantic attention weights) — average across patients in subset
    beta_mean = beta[idx].mean(dim=0).cpu().numpy()

    return {
        'loss':         loss,
        'macro_f1':     macro_f1,
        'micro_f1':     micro_f1,
        'per_disease':  per_disease,
        'beta_mean':    beta_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optimal threshold search (per-disease)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def find_optimal_thresholds(model, feats_t, labels_t, idx, disease_order, device):
    """
    Search for the per-disease threshold that maximises F1 on the given subset.
    """
    model.eval()
    empty_nbr = {name: {} for name in model.metapath_names}
    logits, _, _ = model(feats_t, empty_nbr)
    probs = torch.sigmoid(logits[idx]).cpu().numpy()
    y_true = labels_t[idx].cpu().numpy().astype(int)

    thresholds = {}
    for j, disease in enumerate(disease_order):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            preds_j = (probs[:, j] > t).astype(int)
            f1 = f1_score(y_true[:, j], preds_j, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[disease] = round(float(best_t), 2)

    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  HANPP_Disease Training Script")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"{'='*70}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load Data ─────────────────────────────────────────────────────────
    print("[1/7] Loading data and building graph ...")
    data_loader = MedicalGraphData(
        path_records=args.records_path,
        path_symptom=args.symptom_path,
        symptom_freq_threshold=args.symptom_freq_threshold,
        prune_per_patient=args.prune_per_patient,
        nnz_threshold=args.nnz_threshold,
        seed=args.seed,
    )
    data_loader.load_data()
    data_loader.build_labels_and_features()
    data_loader.build_adjacency_matrices()
    neighbor_dicts = data_loader.build_metapaths(args.metapaths)

    # Extract arrays
    feats_np    = data_loader.patient_feats        # [P, in_dim]
    labels_np   = data_loader.patient_disease      # [P, D]
    disease_order = data_loader.diseases
    organ_map   = {i: name for i, name in enumerate(data_loader.organs)}

    P, in_dim = feats_np.shape
    D         = labels_np.shape[1]

    print(f"\n  Patients:  {P:,}")
    print(f"  Features:  {in_dim}  (S={data_loader.S} symptoms + O={data_loader.O} organs)")
    print(f"  Diseases:  {D}  {disease_order}")

    # Disease prevalence
    print(f"\n  Disease prevalence:")
    for j, d in enumerate(disease_order):
        count = int(labels_np[:, j].sum())
        pct   = count / P * 100
        print(f"    {d:<35} {count:>8,}  ({pct:.1f}%)")

    # ── 2. Split Data ────────────────────────────────────────────────────────
    print(f"\n[2/7] Splitting data (train={args.train_ratio:.0%} / val={args.val_ratio:.0%} / test={1-args.train_ratio-args.val_ratio:.0%}) ...")

    train_idx, val_idx, test_idx = stratified_multilabel_split(
        labels_np, args.train_ratio, args.val_ratio, args.seed,
    )
    print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}  |  Test: {len(test_idx):,}")

    # Verify disease distribution per split
    for name, idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        counts = labels_np[idx].sum(axis=0)
        print(f"  {name} disease counts: {dict(zip(disease_order, counts.astype(int).tolist()))}")

    # ── 3. Prepare Tensors ───────────────────────────────────────────────────
    print(f"\n[3/7] Moving tensors to {device} ...")
    feats_t  = torch.from_numpy(feats_np.astype(np.float32)).to(device)
    labels_t = torch.from_numpy(labels_np.astype(np.float32)).to(device)

    # Vectorize neighbor tensors for GPU-accelerated attention
    max_nbr = args.prune_per_patient
    vectorized_nbr = {}
    for mp_name, nbr_dict in neighbor_dicts.items():
        idx_t, mask_t = neighbors_to_padded_tensors(nbr_dict, P, max_nbr, device=device)
        vectorized_nbr[mp_name] = (idx_t, mask_t)
        print(f"  Meta-path {mp_name}: padded to [{P}, {max_nbr}] on {device}")

    # ── 4. Initialize Model ──────────────────────────────────────────────────
    print(f"\n[4/7] Initializing HANPP_Disease ...")
    model = HANPP_Disease(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        metapath_names=args.metapaths,
        num_heads=args.num_heads,
        num_diseases=D,
        dropout=args.dropout,
    ).to(device)

    # Pre-set vectorized neighbors for fast training
    model.set_vectorized_neighbors(vectorized_nbr)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: in={in_dim}, hidden={args.hidden_dim}, out={args.out_dim}")
    print(f"  Heads: {args.num_heads}, Dropout: {args.dropout}")
    print(f"  Parameters: {total_params:,} total, {train_params:,} trainable")

    # ── 5. Training Setup ────────────────────────────────────────────────────
    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr, verbose=True,
    )

    print(f"\n  Loss:      FocalLoss (gamma={args.focal_gamma})")
    print(f"  Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"  Scheduler: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor})")
    print(f"  Early stopping patience: {args.patience}")

    # ── 6. Training Loop ─────────────────────────────────────────────────────
    print(f"\n[5/7] Training for up to {args.epochs} epochs ...\n")

    best_val_loss  = float('inf')
    best_val_f1    = 0.0
    best_epoch     = 0
    patience_count = 0

    # History for plotting
    hist = {
        'train_loss': [], 'val_loss': [],
        'val_macro_f1': [], 'val_micro_f1': [], 'val_mean_f1': [],
        'train_acc': [], 'val_acc': [],
        'lr': [],
    }

    best_model_path = os.path.join(args.output_dir, 'hanpp_disease_best.pt')

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        logits, z, beta = model(feats_t, neighbor_dicts)

        # Compute loss ONLY on training patients
        train_loss = criterion(logits[train_idx], labels_t[train_idx])
        train_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_loss_val = train_loss.item()
        hist['train_loss'].append(train_loss_val)
        hist['lr'].append(optimizer.param_groups[0]['lr'])

        # ── Validate ─────────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == 1:
            val_metrics = evaluate_disease(model, feats_t, labels_t, val_idx, disease_order, device)
            val_loss = val_metrics['loss']
            val_macro_f1 = val_metrics['macro_f1']
            val_micro_f1 = val_metrics['micro_f1']
            val_mean_f1 = (val_macro_f1 + val_micro_f1) / 2

            hist['val_loss'].append(val_loss)
            hist['val_macro_f1'].append(val_macro_f1)
            hist['val_micro_f1'].append(val_micro_f1)
            hist['val_mean_f1'].append(val_mean_f1)

            # LR scheduler step
            scheduler.step(val_loss)

            # Early stopping check
            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_f1 = val_macro_f1
                best_epoch = epoch
                patience_count = 0
                torch.save(model.state_dict(), best_model_path)
                improved = " ✅ saved"
            else:
                patience_count += 1

            elapsed = time.time() - epoch_start
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:>4}/{args.epochs}  "
                  f"train_loss={train_loss_val:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"F1(macro={val_macro_f1:.4f} micro={val_micro_f1:.4f})  "
                  f"lr={lr_now:.1e}  "
                  f"[{elapsed:.1f}s]{improved}")

            # Print per-disease F1 every 10 epochs
            if epoch % 10 == 0:
                beta_str = ", ".join(f"{args.metapaths[i]}={val_metrics['beta_mean'][i]:.3f}"
                                     for i in range(len(args.metapaths)))
                print(f"         β weights: [{beta_str}]")
                for d, f1 in val_metrics['per_disease'].items():
                    print(f"           {d:<30} F1={f1:.4f}")

            # Early stopping
            if patience_count >= args.patience:
                print(f"\n  ⚠ Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break
        else:
            elapsed = time.time() - epoch_start
            print(f"  Epoch {epoch:>4}/{args.epochs}  train_loss={train_loss_val:.4f}  [{elapsed:.1f}s]")

    total_time = time.time() - t_start
    print(f"\n  Training complete in {total_time/60:.1f} minutes")
    print(f"  Best epoch: {best_epoch}  (val_loss={best_val_loss:.4f}, macro_f1={best_val_f1:.4f})")

    # ── 7. Post-Training ─────────────────────────────────────────────────────
    print(f"\n[6/7] Post-training: evaluation, prototypes, plots ...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # ── Test set evaluation ──────────────────────────────────────────────────
    print(f"\n  === Test Set Evaluation ===")
    test_metrics = evaluate_disease(model, feats_t, labels_t, test_idx, disease_order, device)
    print(f"  Test Loss:     {test_metrics['loss']:.4f}")
    print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Test Micro-F1: {test_metrics['micro_f1']:.4f}")
    print(f"\n  Per-disease F1 (Test):")
    for d, f1 in test_metrics['per_disease'].items():
        print(f"    {d:<35} F1={f1:.4f}")

    # ── Optimal thresholds ───────────────────────────────────────────────────
    print(f"\n  Finding optimal per-disease thresholds on validation set ...")
    opt_thresholds = find_optimal_thresholds(model, feats_t, labels_t, val_idx, disease_order, device)
    print(f"  Thresholds: {opt_thresholds}")

    # Save thresholds
    thresh_path = os.path.join(args.output_dir, 'opt_thresholds.json')
    with open(thresh_path, 'w') as f:
        json.dump(opt_thresholds, f, indent=2)
    print(f"  Thresholds saved → {thresh_path}")

    # ── Build disease prototypes ─────────────────────────────────────────────
    print(f"\n  Building disease prototypes ...")
    proto_path = os.path.join(args.output_dir, 'prototypes.pkl')
    build_disease_prototypes(
        model=model,
        feats_t=feats_t,
        labels_np=labels_np,
        disease_order=disease_order,
        device=device,
        save_path=proto_path,
    )

    # ── Build organ prototypes ───────────────────────────────────────────────
    print(f"\n  Building organ prototypes ...")
    organ_proto_path = os.path.join(args.output_dir, 'organ_prototypes.pkl')
    build_organ_prototypes(
        model=model,
        feats_t=feats_t,
        patient_organ_score=data_loader.patient_organ_score,
        organ_map=organ_map,
        device=device,
        save_path=organ_proto_path,
    )

    # ── Inference mode benchmark ─────────────────────────────────────────────
    print(f"\n  Running inference mode benchmark (transductive vs prototype vs MLP) ...")

    # Need to load prototypes for benchmark
    with open(proto_path, 'rb') as f:
        proto_data = pickle.load(f)

    # Disable pre-set vectorized neighbors for the comparison (it uses its own mini-graphs)
    model.set_vectorized_neighbors({mp: (None, None) for mp in args.metapaths})
    for layer in model.node_atts:
        layer.neighbor_idx = None
        layer.neighbor_mask = None

    benchmark = compare_inference_modes(
        model=model,
        feats_np=feats_np,
        labels_np=labels_np,
        test_indices=test_idx,
        nbr_dicts_full=neighbor_dicts,
        prototypes=proto_data['prototypes'],
        z_train=proto_data['patient_embeddings'],
        disease_order=disease_order,
        opt_thresholds=opt_thresholds,
        device=device,
        n_mc_samples=args.mc_samples,
        n_patients=min(10, len(test_idx)),
        seed=args.seed,
    )

    print(f"\n  Inference Mode Comparison:")
    print(f"  {'Mode':<20} {'Macro-F1':>10} {'Micro-F1':>10}")
    print(f"  {'-'*42}")
    for mode in ['transductive', 'prototype', 'mlp_only']:
        print(f"  {mode:<20} {benchmark[mode]['f1_macro']:>10.4f} {benchmark[mode]['f1_micro']:>10.4f}")

    # ── Save training history ────────────────────────────────────────────────
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(hist, f, indent=2)
    print(f"\n  Training history saved → {history_path}")

    # ── Plot training metrics ────────────────────────────────────────────────
    print(f"\n  Generating training plots ...")
    plot_path = os.path.join(args.output_dir, 'training_plots.png')
    try:
        plot_training_metrics(
            train_losses=hist['train_loss'],
            val_losses=hist['val_loss'],
            val_hist=hist['val_mean_f1'],
            val_micro_f1=hist['val_micro_f1'],
            val_macro_f1=hist['val_macro_f1'],
            model_name='HANPP_Disease',
            meta_path='+'.join(args.metapaths),
            save_path=plot_path,
        )
    except Exception as e:
        print(f"  [Warning] Plot generation failed: {e}")

    # ── Save hyperparameters ─────────────────────────────────────────────────
    hparams = {k: v for k, v in vars(args).items()}
    hparams['in_dim'] = in_dim
    hparams['num_diseases'] = D
    hparams['num_patients'] = P
    hparams['disease_order'] = disease_order
    hparams['best_epoch'] = best_epoch
    hparams['best_val_loss'] = best_val_loss
    hparams['best_val_f1'] = best_val_f1
    hparams['test_macro_f1'] = test_metrics['macro_f1']
    hparams['test_micro_f1'] = test_metrics['micro_f1']
    hparams['total_params'] = total_params
    hparams['training_time_min'] = round(total_time / 60, 1)
    hparams['benchmark'] = benchmark

    hparams_path = os.path.join(args.output_dir, 'hparams.json')
    with open(hparams_path, 'w') as f:
        json.dump(hparams, f, indent=2, default=str)

    # ── Final Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Best model:       {best_model_path}")
    print(f"  Prototypes:       {proto_path}")
    print(f"  Organ prototypes: {organ_proto_path}")
    print(f"  Thresholds:       {thresh_path}")
    print(f"  History:          {history_path}")
    print(f"  Hyperparams:      {hparams_path}")
    print(f"  Plots:            {plot_path}")
    print(f"\n  Test Macro-F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  Test Micro-F1:    {test_metrics['micro_f1']:.4f}")
    print(f"  Training time:    {total_time/60:.1f} minutes")
    print(f"\n  To use this model, update inference_v6.py:")
    print(f"    MODEL_PATH     = '{best_model_path}'")
    print(f"    PROTOTYPE_PATH = '{proto_path}'")
    print(f"    ORGAN_PROTOTYPE_PATH = '{organ_proto_path}'")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    args = parse_args()
    train(args)
