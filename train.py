#!/usr/bin/env python3
"""
train.py — HANPP_LinkPredict GPU Training Script (Contrastive Link Prediction)
================================================================================

Trains the HANPP_LinkPredict model with InfoNCE contrastive loss.
Patients and diseases are embedded in a shared vector space, and link scores
are computed via cosine similarity scaled by a learnable temperature τ.

Usage:
    python train.py
    python train.py --epochs 200 --lr 0.002 --hidden_dim 512
"""

import os, sys, json, time, argparse, pickle, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from HAN.data import MedicalGraphData
from HAN.model import HANPP_LinkPredict
from HAN.losses import InfoNCELinkLoss
from HAN.utils import neighbors_to_padded_tensors, plot_training_metrics
from HAN.inductive import build_disease_prototypes, build_organ_prototypes, compare_inference_modes

# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    'records_path':     'data/HAN_data/merged_coop_ruhunu_patient_data.csv',
    'symptom_path':     'data/HAN_data/test_reference_full_v2.csv',
    'output_dir':       'models_saved/retrained',
    'train_ratio': 0.70, 'val_ratio': 0.15, 'seed': 42,
    'symptom_freq_threshold': 0.08, 'prune_per_patient': 300,
    'nnz_threshold': 80_000_000,
    'metapaths': ['P-D-P', 'P-O-P'],
    'hidden_dim': 256, 'out_dim': 128, 'num_heads': 4, 'dropout': 0.4,
    'init_temperature': 0.07,
    'epochs': 150, 'lr': 0.001, 'weight_decay': 1e-4,
    'hard_neg_weight': 1.0,
    'patience': 20, 'lr_patience': 10, 'lr_factor': 0.5, 'min_lr': 1e-6,
    'eval_every': 1, 'mc_samples': 50,
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train HANPP_LinkPredict')
    for key, val in DEFAULTS.items():
        if isinstance(val, list):
            parser.add_argument(f'--{key}', nargs='+', default=val)
        else:
            parser.add_argument(f'--{key}', type=type(val), default=val)
    return parser.parse_args()


def stratified_split(labels_np, train_ratio, val_ratio, seed):
    N = labels_np.shape[0]
    all_idx = np.arange(N)
    strat = np.full(N, -1, dtype=int)
    for i in range(N):
        pos = np.where(labels_np[i] == 1)[0]
        if len(pos) > 0: strat[i] = pos[0]
    test_ratio = 1.0 - train_ratio - val_ratio
    train_idx, temp_idx = train_test_split(all_idx, test_size=(val_ratio + test_ratio),
                                            random_state=seed, stratify=strat)
    temp_strat = strat[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_ratio/(val_ratio+test_ratio),
                                          random_state=seed, stratify=temp_strat)
    return train_idx, val_idx, test_idx


@torch.no_grad()
def evaluate(model, feats_t, labels_t, idx, disease_order, criterion, device):
    model.eval()
    empty_nbr = {name: {} for name in model.metapath_names}
    scores, z, beta = model(feats_t, empty_nbr)
    loss = criterion(scores[idx], labels_t[idx]).item()
    probs = torch.sigmoid(scores[idx]).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    y_true = labels_t[idx].cpu().numpy().astype(int)
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, preds, average='micro', zero_division=0)
    per_disease = {d: f1_score(y_true[:,j], preds[:,j], zero_division=0)
                   for j, d in enumerate(disease_order)}
    beta_mean = beta[idx].mean(dim=0).cpu().numpy()
    return {'loss': loss, 'macro_f1': macro_f1, 'micro_f1': micro_f1,
            'per_disease': per_disease, 'beta_mean': beta_mean}


@torch.no_grad()
def find_optimal_thresholds(model, feats_t, labels_t, idx, disease_order, device):
    model.eval()
    empty_nbr = {name: {} for name in model.metapath_names}
    scores, _, _ = model(feats_t, empty_nbr)
    probs = torch.sigmoid(scores[idx]).cpu().numpy()
    y_true = labels_t[idx].cpu().numpy().astype(int)
    thresholds = {}
    for j, d in enumerate(disease_order):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(y_true[:,j], (probs[:,j] > t).astype(int), zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, t
        thresholds[d] = round(float(best_t), 2)
    return thresholds


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  HANPP_LinkPredict Training (Contrastive Link Prediction)")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print(f"{'='*70}\n")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load Data ─────────────────────────────────────────────────────────
    print("[1/7] Loading data and building graph ...")
    dl = MedicalGraphData(path_records=args.records_path, path_symptom=args.symptom_path,
                          symptom_freq_threshold=args.symptom_freq_threshold,
                          prune_per_patient=args.prune_per_patient,
                          nnz_threshold=args.nnz_threshold, seed=args.seed)
    dl.load_data(); dl.build_labels_and_features(); dl.build_adjacency_matrices()
    neighbor_dicts = dl.build_metapaths(args.metapaths)

    feats_np, labels_np = dl.patient_feats, dl.patient_disease
    disease_order = dl.diseases
    organ_map = {i: name for i, name in enumerate(dl.organs)}
    P, in_dim = feats_np.shape
    D = labels_np.shape[1]
    print(f"  Patients: {P:,}  Features: {in_dim}  Diseases: {D}  {disease_order}")

    # ── 2. Split ─────────────────────────────────────────────────────────────
    print(f"\n[2/7] Stratified split ({args.train_ratio:.0%}/{args.val_ratio:.0%}/{1-args.train_ratio-args.val_ratio:.0%}) ...")
    train_idx, val_idx, test_idx = stratified_split(labels_np, args.train_ratio, args.val_ratio, args.seed)
    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    # ── 3. Tensors to GPU ────────────────────────────────────────────────────
    print(f"\n[3/7] Moving tensors to {device} ...")
    feats_t  = torch.from_numpy(feats_np.astype(np.float32)).to(device)
    labels_t = torch.from_numpy(labels_np.astype(np.float32)).to(device)

    vectorized_nbr = {}
    for mp, nbr_dict in neighbor_dicts.items():
        idx_t, mask_t = neighbors_to_padded_tensors(nbr_dict, P, args.prune_per_patient, device=device)
        vectorized_nbr[mp] = (idx_t, mask_t)
        print(f"  {mp}: padded [{P}, {args.prune_per_patient}]")

    # ── 4. Model ─────────────────────────────────────────────────────────────
    print(f"\n[4/7] Initializing HANPP_LinkPredict ...")
    model = HANPP_LinkPredict(
        in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
        metapath_names=args.metapaths, num_heads=args.num_heads,
        num_diseases=D, dropout=args.dropout, init_temperature=args.init_temperature,
    ).to(device)
    model.set_vectorized_neighbors(vectorized_nbr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: in={in_dim}, hidden={args.hidden_dim}, out={args.out_dim}")
    print(f"  Disease embeddings: [{D}, {args.out_dim}]")
    print(f"  Init temperature τ = {args.init_temperature}")
    print(f"  Parameters: {total_params:,}")

    # ── 5. Training setup ────────────────────────────────────────────────────
    criterion = InfoNCELinkLoss(hard_neg_weight=args.hard_neg_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.min_lr, verbose=True)

    print(f"\n  Loss:      InfoNCE (hard_neg_weight={args.hard_neg_weight})")
    print(f"  Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"  Early stopping: patience={args.patience}")

    # ── 6. Training loop ─────────────────────────────────────────────────────
    print(f"\n[5/7] Training for up to {args.epochs} epochs ...\n")

    best_val_loss, best_val_f1, best_epoch, patience_count = float('inf'), 0.0, 0, 0
    hist = {'train_loss': [], 'val_loss': [], 'val_macro_f1': [], 'val_micro_f1': [],
            'val_mean_f1': [], 'lr': [], 'temperature': []}
    best_model_path = os.path.join(args.output_dir, 'hanpp_linkpredict_best.pt')
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        scores, z, beta = model(feats_t, neighbor_dicts)
        train_loss = criterion(scores[train_idx], labels_t[train_idx])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        tl = train_loss.item()
        hist['train_loss'].append(tl)
        hist['lr'].append(optimizer.param_groups[0]['lr'])
        hist['temperature'].append(float(model.temperature.detach().cpu()))

        if epoch % args.eval_every == 0 or epoch == 1:
            vm = evaluate(model, feats_t, labels_t, val_idx, disease_order, criterion, device)
            vl, vf1_ma, vf1_mi = vm['loss'], vm['macro_f1'], vm['micro_f1']
            vf1_mean = (vf1_ma + vf1_mi) / 2
            hist['val_loss'].append(vl)
            hist['val_macro_f1'].append(vf1_ma)
            hist['val_micro_f1'].append(vf1_mi)
            hist['val_mean_f1'].append(vf1_mean)
            scheduler.step(vl)

            improved = ""
            if vl < best_val_loss:
                best_val_loss, best_val_f1, best_epoch, patience_count = vl, vf1_ma, epoch, 0
                torch.save(model.state_dict(), best_model_path)
                improved = " ✅ saved"
            else:
                patience_count += 1

            tau = float(model.temperature.detach().cpu())
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:>4}/{args.epochs}  "
                  f"train={tl:.4f}  val={vl:.4f}  "
                  f"F1(ma={vf1_ma:.4f} mi={vf1_mi:.4f})  "
                  f"τ={tau:.4f}  lr={lr_now:.1e}{improved}")

            if epoch % 10 == 0:
                for d, f1 in vm['per_disease'].items():
                    print(f"           {d:<30} F1={f1:.4f}")

            if patience_count >= args.patience:
                print(f"\n  ⚠ Early stopping at epoch {epoch}")
                break
        else:
            print(f"  Epoch {epoch:>4}/{args.epochs}  train={tl:.4f}  [{time.time()-t0:.1f}s]")

    total_time = time.time() - t_start
    print(f"\n  Training complete in {total_time/60:.1f} min")
    print(f"  Best epoch: {best_epoch}  (val_loss={best_val_loss:.4f}, F1={best_val_f1:.4f})")

    # ── 7. Post-training ─────────────────────────────────────────────────────
    print(f"\n[6/7] Post-training evaluation & prototype generation ...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Test evaluation
    test_m = evaluate(model, feats_t, labels_t, test_idx, disease_order, criterion, device)
    print(f"\n  === Test Set ===")
    print(f"  Loss: {test_m['loss']:.4f}  Macro-F1: {test_m['macro_f1']:.4f}  Micro-F1: {test_m['micro_f1']:.4f}")
    for d, f1 in test_m['per_disease'].items():
        print(f"    {d:<35} F1={f1:.4f}")

    # Optimal thresholds
    opt_thresholds = find_optimal_thresholds(model, feats_t, labels_t, val_idx, disease_order, device)
    thresh_path = os.path.join(args.output_dir, 'opt_thresholds.json')
    with open(thresh_path, 'w') as f: json.dump(opt_thresholds, f, indent=2)
    print(f"\n  Thresholds: {opt_thresholds}")

    # Disease embeddings analysis
    d_emb = model.get_disease_embeddings()  # [D, out_dim]
    sim_matrix = (d_emb @ d_emb.T).cpu().numpy()
    print(f"\n  Disease Embedding Similarity Matrix:")
    print(f"  {'':>20}", "  ".join(f"{d[:6]:>6}" for d in disease_order))
    for i, d in enumerate(disease_order):
        row = "  ".join(f"{sim_matrix[i,j]:>6.3f}" for j in range(D))
        print(f"  {d:>20}  {row}")

    # Save disease embeddings
    emb_path = os.path.join(args.output_dir, 'disease_embeddings.pkl')
    with open(emb_path, 'wb') as f:
        pickle.dump({'embeddings': d_emb.cpu().numpy(), 'disease_order': disease_order}, f)

    # Build prototypes
    proto_path = os.path.join(args.output_dir, 'prototypes.pkl')
    build_disease_prototypes(model=model, feats_t=feats_t, labels_np=labels_np,
                             disease_order=disease_order, device=device, save_path=proto_path)

    organ_proto_path = os.path.join(args.output_dir, 'organ_prototypes.pkl')
    build_organ_prototypes(model=model, feats_t=feats_t,
                           patient_organ_score=dl.patient_organ_score,
                           organ_map=organ_map, device=device, save_path=organ_proto_path)

    # Save history & hparams
    hist_path = os.path.join(args.output_dir, 'training_history.json')
    with open(hist_path, 'w') as f: json.dump(hist, f, indent=2)

    hparams = {k: v for k, v in vars(args).items()}
    hparams.update({'in_dim': in_dim, 'num_diseases': D, 'num_patients': P,
                    'disease_order': disease_order, 'best_epoch': best_epoch,
                    'best_val_loss': best_val_loss, 'best_val_f1': best_val_f1,
                    'test_macro_f1': test_m['macro_f1'], 'test_micro_f1': test_m['micro_f1'],
                    'final_temperature': float(model.temperature.detach().cpu()),
                    'total_params': total_params, 'training_time_min': round(total_time/60, 1)})
    with open(os.path.join(args.output_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=2, default=str)

    # Plot
    try:
        plot_training_metrics(hist['train_loss'], hist['val_loss'], hist['val_mean_f1'],
                              hist['val_micro_f1'], hist['val_macro_f1'],
                              'HANPP_LinkPredict', '+'.join(args.metapaths),
                              os.path.join(args.output_dir, 'training_plots.png'))
    except Exception as e:
        print(f"  [Warning] Plot failed: {e}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE — Contrastive Link Prediction")
    print(f"{'='*70}")
    print(f"  Model:            {best_model_path}")
    print(f"  Prototypes:       {proto_path}")
    print(f"  Organ prototypes: {organ_proto_path}")
    print(f"  Disease embeds:   {emb_path}")
    print(f"  Learned τ:        {float(model.temperature.detach().cpu()):.4f}")
    print(f"  Test Macro-F1:    {test_m['macro_f1']:.4f}")
    print(f"  Test Micro-F1:    {test_m['micro_f1']:.4f}")
    print(f"\n  Update inference_v6.py:")
    print(f"    MODEL_PATH           = '{best_model_path}'")
    print(f"    PROTOTYPE_PATH       = '{proto_path}'")
    print(f"    ORGAN_PROTOTYPE_PATH = '{organ_proto_path}'")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    train(parse_args())
