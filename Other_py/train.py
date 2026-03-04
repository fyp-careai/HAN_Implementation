"""
HAN Training Script - Vectorized Version
Train HAN++ and HGT-HAN models on medical data with meta-path experiments.

Requirements: numpy, pandas, scipy, torch, scikit-learn, matplotlib, iterative-stratification
"""

import os
import time
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from HAN import (
    MedicalGraphData,
    HANPP,
    HGT_HAN,
    compute_loss_multiorg,
    evaluate_multiorg,
    plot_training_metrics,
    neighbors_to_padded_tensors
)

# ============== Configuration ==============

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Data paths
PATH_RECORDS = "/Users/charlie/Documents/Coding/VS Code/Language_python/FYP/DATA_FYP/patient_reports.csv"
PATH_SYMPTOM = "/Users/charlie/Documents/Coding/VS Code/Language_python/FYP/DATA_FYP/test,Organ.csv"

# Meta-paths to evaluate
META_PATHS = ["P-O-P", "P-D-P", "P-S-P", "P-S-O-P", "P-O-D-P"]

# Models to run
RUN_HANPP = True   # Version B
RUN_HGTHAN = True  # Version C

# Training hyperparameters
EPOCHS_B = 40
EPOCHS_C = 40
LR = 1e-3
PATIENCE = 12
BATCH_TRAINING = False  # Full-batch training

# Loss configuration
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 1.0

# Data filtering thresholds
SYMPTOM_FREQ_THRESHOLD = 0.08
PRUNE_PER_PATIENT = 300
NNZ_THRESHOLD = 80_000_000

# Output directory
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Output directory: {OUT_DIR}")

# ============== Load and Process Data ==============

print("\n" + "="*80)
print("LOADING AND PROCESSING DATA")
print("="*80)

data_loader = MedicalGraphData(
    path_records=PATH_RECORDS,
    path_symptom=PATH_SYMPTOM,
    symptom_freq_threshold=SYMPTOM_FREQ_THRESHOLD,
    prune_per_patient=PRUNE_PER_PATIENT,
    nnz_threshold=NNZ_THRESHOLD,
    seed=SEED
)

# Load data
data_loader.load_data()

# Build labels and features
data_loader.build_labels_and_features()

# Build adjacency matrices
data_loader.build_adjacency_matrices()

# Build meta-paths
available_metapaths = [mp for mp in META_PATHS]
patient_metapath_neighbors = data_loader.build_metapaths(available_metapaths)

if not patient_metapath_neighbors:
    raise RuntimeError("No candidate meta-paths available after pruning. Check thresholds.")

print("\nAvailable meta-paths:", list(patient_metapath_neighbors.keys()))

# Get tensors
tensors = data_loader.get_tensors(device=DEVICE)
patient_feats = tensors['patient_feats']
labels_organ_severity = tensors['labels_organ_severity']

# Compute class weights
organ_class_weights = data_loader.compute_class_weights(device=DEVICE)

print(f"Patient features shape: {patient_feats.shape}")
print(f"Organ severity labels shape: {labels_organ_severity.shape}")

# ============== Train/Val Split ==============

print("\n" + "="*80)
print("CREATING TRAIN/VAL SPLIT")
print("="*80)

P = data_loader.P
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx_list, val_idx_list = next(msss.split(np.zeros(P), data_loader.patient_disease))
    train_idx = set(train_idx_list.tolist())
    val_idx = set(val_idx_list.tolist())
    print("Using iterative stratified split")
except Exception as e:
    print("iterative-stratification not available or failed:", e)
    indices = list(range(P))
    random.shuffle(indices)
    train_n = int(0.8 * P)
    train_idx = set(indices[:train_n])
    val_idx = set(indices[train_n:])
    print("Using random split")

print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

# ============== Meta-Path Experiments ==============

results = []
meta_paths_to_try = [mp for mp in META_PATHS if mp in patient_metapath_neighbors]

print("\n" + "="*80)
print("STARTING META-PATH EXPERIMENTS")
print("="*80)
print(f"Will evaluate meta-paths: {meta_paths_to_try}")

for mp in meta_paths_to_try:
    print("\n" + "="*80)
    print(f"Experiment: meta-path = [{mp}]")
    print("="*80)
    
    # Build neighbor dict containing only chosen meta-path
    neighs = {mp: patient_metapath_neighbors[mp]}
    metapath_list = [mp]
    
    # Pre-compute vectorized neighbor tensors
    print("Pre-computing vectorized neighbor tensors...")
    neighbor_tensors = {}
    for name, neigh_dict in neighs.items():
        idx, mask = neighbors_to_padded_tensors(neigh_dict, P, PRUNE_PER_PATIENT, DEVICE)
        neighbor_tensors[name] = (idx, mask)
    print(f"✔ Vectorized neighbors prepared for {len(neighbor_tensors)} meta-paths")
    
    # ============== HAN++ (Version B) ==============
    
    if RUN_HANPP:
        print(f"\n{'='*60}")
        print(f"Training HAN++ on meta-path {mp} (VECTORIZED)")
        print('='*60)
        
        model_b = HANPP(
            in_dim=patient_feats.shape[1],
            hidden_dim=128,
            out_dim=128,
            metapath_names=metapath_list,
            num_heads=4,
            num_organs=data_loader.O,
            num_severity=4,
            dropout=0.4
        ).to(DEVICE)
        
        # Set vectorized neighbors
        model_b.set_vectorized_neighbors(neighbor_tensors)
        
        optimizer = torch.optim.AdamW(model_b.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
        
        best_val = 0.0
        patience_ctr = 0
        best_path = os.path.join(OUT_DIR, f"hanpp_{mp}.pt")
        
        train_losses = []
        val_losses = []
        val_hist = []
        val_micro_f1 = []
        val_macro_f1 = []
        
        for epoch in range(1, EPOCHS_B + 1):
            model_b.train()
            optimizer.zero_grad()
            
            organ_logits, organ_scores, z, beta = model_b(patient_feats, neighs)
            loss = compute_loss_multiorg(
                organ_logits, labels_organ_severity, train_idx, organ_class_weights,
                use_focal=USE_FOCAL_LOSS, focal_gamma=FOCAL_GAMMA
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if epoch % 2 == 0 or epoch == 1:
                model_b.eval()
                with torch.no_grad():
                    val_organ_logits, _, _, _ = model_b(patient_feats, neighs)
                    val_loss = compute_loss_multiorg(
                        val_organ_logits, labels_organ_severity, val_idx, organ_class_weights,
                        use_focal=USE_FOCAL_LOSS, focal_gamma=FOCAL_GAMMA
                    )
                    val_losses.append(val_loss.item())
                
                val_metrics = evaluate_multiorg(model_b, patient_feats, labels_organ_severity, neighs, val_idx)
                val_f1 = val_metrics['mean_organ_f1']
                val_hist.append(val_f1)
                val_micro_f1.append(val_metrics['micro_f1'])
                val_macro_f1.append(val_metrics['macro_f1'])
                
                print(f"Epoch {epoch:3d}/{EPOCHS_B} | train_loss={loss.item():.4f} | "
                      f"val_loss={val_loss.item():.4f} | val_mean_f1={val_f1:.4f} | "
                      f"micro_f1={val_metrics['micro_f1']:.4f} | valid_organs={val_metrics['num_valid_organs']}")
                
                if val_f1 > best_val + 1e-5:
                    best_val = val_f1
                    torch.save(model_b.state_dict(), best_path)
                    patience_ctr = 0
                    print(f"  → New best saved: {best_val:.4f}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= PATIENCE:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                scheduler.step(val_f1)
        
        # Load best model and final evaluation
        if os.path.exists(best_path):
            model_b.load_state_dict(torch.load(best_path))
        
        val_metrics = evaluate_multiorg(model_b, patient_feats, labels_organ_severity, neighs, val_idx)
        
        print("\n" + "="*60)
        print("HAN++ Final Results")
        print("="*60)
        print(f"Mean Organ F1:    {val_metrics['mean_organ_f1']:.4f}")
        print(f"Micro F1:         {val_metrics['micro_f1']:.4f}")
        print(f"Macro F1:         {val_metrics['macro_f1']:.4f}")
        print(f"Final Train Loss: {train_losses[-1]:.6f}")
        print(f"Final Val Loss:   {val_losses[-1]:.6f}")
        print(f"Best Val F1:      {best_val:.6f}")
        print("="*60)
        
        # Plot training metrics
        plot_path = os.path.join(OUT_DIR, f"training_plots_hanpp_{mp}.png")
        plot_training_metrics(
            train_losses, val_losses, val_hist, val_micro_f1, val_macro_f1,
            "HAN++", mp, plot_path
        )
        
        # Store results
        results.append({
            'model': 'HAN++',
            'meta_path': mp,
            'mean_organ_f1': val_metrics['mean_organ_f1'],
            'micro_f1': val_metrics['micro_f1'],
            'macro_f1': val_metrics['macro_f1'],
            'per_organ_f1': val_metrics['per_organ_f1'],
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1]),
            'best_val_f1': float(best_val),
            'total_epochs': len(train_losses),
            'beta': val_metrics['beta'].tolist()
        })
    
    # ============== HGT-HAN (Version C) ==============
    
    if RUN_HGTHAN:
        print(f"\n{'='*60}")
        print(f"Training HGT-HAN on meta-path {mp} (VECTORIZED)")
        print('='*60)
        
        model_c = HGT_HAN(
            in_dim=patient_feats.shape[1],
            hidden_dim=128,
            out_dim=128,
            metapath_names=metapath_list,
            num_heads=4,
            num_organs=data_loader.O,
            num_severity=4,
            dropout=0.4
        ).to(DEVICE)
        
        # Set vectorized neighbors
        model_c.set_vectorized_neighbors(neighbor_tensors)
        
        optimizer = torch.optim.AdamW(model_c.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
        
        best_val = 0.0
        patience_ctr = 0
        best_path = os.path.join(OUT_DIR, f"hgthan_{mp}.pt")
        
        train_losses = []
        val_losses = []
        val_hist = []
        val_micro_f1 = []
        val_macro_f1 = []
        
        for epoch in range(1, EPOCHS_C + 1):
            model_c.train()
            optimizer.zero_grad()
            
            organ_logits, organ_scores, z, beta = model_c(patient_feats, neighs)
            loss = compute_loss_multiorg(
                organ_logits, labels_organ_severity, train_idx, organ_class_weights,
                use_focal=USE_FOCAL_LOSS, focal_gamma=FOCAL_GAMMA
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_c.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if epoch % 2 == 0 or epoch == 1:
                model_c.eval()
                with torch.no_grad():
                    val_organ_logits, _, _, _ = model_c(patient_feats, neighs)
                    val_loss = compute_loss_multiorg(
                        val_organ_logits, labels_organ_severity, val_idx, organ_class_weights,
                        use_focal=USE_FOCAL_LOSS, focal_gamma=FOCAL_GAMMA
                    )
                    val_losses.append(val_loss.item())
                
                val_metrics = evaluate_multiorg(model_c, patient_feats, labels_organ_severity, neighs, val_idx)
                val_f1 = val_metrics['mean_organ_f1']
                val_hist.append(val_f1)
                val_micro_f1.append(val_metrics['micro_f1'])
                val_macro_f1.append(val_metrics['macro_f1'])
                
                print(f"Epoch {epoch:3d}/{EPOCHS_C} | train_loss={loss.item():.4f} | "
                      f"val_loss={val_loss.item():.4f} | val_mean_f1={val_f1:.4f} | "
                      f"micro_f1={val_metrics['micro_f1']:.4f} | valid_organs={val_metrics['num_valid_organs']}")
                
                if val_f1 > best_val + 1e-5:
                    best_val = val_f1
                    torch.save(model_c.state_dict(), best_path)
                    patience_ctr = 0
                    print(f"  → New best saved: {best_val:.4f}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= PATIENCE:
                        print(f"Early stopping at epoch {epoch}")
                        break
                
                scheduler.step(val_f1)
        
        # Load best model and final evaluation
        if os.path.exists(best_path):
            model_c.load_state_dict(torch.load(best_path))
        
        val_metrics = evaluate_multiorg(model_c, patient_feats, labels_organ_severity, neighs, val_idx)
        
        print("\n" + "="*60)
        print("HGT-HAN Final Results")
        print("="*60)
        print(f"Mean Organ F1:    {val_metrics['mean_organ_f1']:.4f}")
        print(f"Micro F1:         {val_metrics['micro_f1']:.4f}")
        print(f"Macro F1:         {val_metrics['macro_f1']:.4f}")
        print(f"Final Train Loss: {train_losses[-1]:.6f}")
        print(f"Final Val Loss:   {val_losses[-1]:.6f}")
        print(f"Best Val F1:      {best_val:.6f}")
        print("="*60)
        
        # Plot training metrics
        plot_path = os.path.join(OUT_DIR, f"training_plots_hgthan_{mp}.png")
        plot_training_metrics(
            train_losses, val_losses, val_hist, val_micro_f1, val_macro_f1,
            "HGT-HAN", mp, plot_path
        )
        
        # Store results
        results.append({
            'model': 'HGT-HAN',
            'meta_path': mp,
            'mean_organ_f1': val_metrics['mean_organ_f1'],
            'micro_f1': val_metrics['micro_f1'],
            'macro_f1': val_metrics['macro_f1'],
            'per_organ_f1': val_metrics['per_organ_f1'],
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1]),
            'best_val_f1': float(best_val),
            'total_epochs': len(train_losses),
            'beta': val_metrics['beta'].tolist()
        })

# ============== Save Summary ==============

print("\n" + "="*90)
print("FINAL COMPARISON OF ALL MODELS AND META-PATHS")
print("="*90)

df_res = pd.DataFrame(results)
summary_csv = os.path.join(OUT_DIR, "results_summary.csv")
df_res.to_csv(summary_csv, index=False)

print(f"{'Model':<12} | {'Meta-path':<12} | {'Mean F1':<10} | {'Micro F1':<10} | {'Train Loss':<11} | {'Val Loss':<11}")
print("-" * 90)
for res in results:
    print(f"{res['model']:<12} | {res['meta_path']:<12} | {res['mean_organ_f1']:<10.4f} | "
          f"{res['micro_f1']:<10.4f} | {res.get('final_train_loss', 0):<11.6f} | "
          f"{res.get('final_val_loss', 0):<11.6f}")

if results:
    best_result = max(results, key=lambda x: x['mean_organ_f1'])
    print("\n" + "="*90)
    print("BEST MODEL RESULTS")
    print("="*90)
    print(f"Model:             {best_result['model']}")
    print(f"Meta-path:         {best_result['meta_path']}")
    print(f"Mean Organ F1:     {best_result['mean_organ_f1']:.6f}")
    print(f"Micro F1:          {best_result['micro_f1']:.6f}")
    print(f"Macro F1:          {best_result['macro_f1']:.6f}")
    print(f"Final Train Loss:  {best_result.get('final_train_loss', 0):.6f}")
    print(f"Final Val Loss:    {best_result.get('final_val_loss', 0):.6f}")
    print(f"Best Val F1:       {best_result.get('best_val_f1', 0):.6f}")
    print(f"Total Epochs:      {best_result.get('total_epochs', 0)}")
    print("="*90)

print(f"\n✔ All results saved to {OUT_DIR}/")
print("  - results_summary.csv")
print("  - training_plots_<model>_<metapath>.png")
print("  - hanpp_<metapath>.pt / hgthan_<metapath>.pt (model checkpoints)")
print("\nDataFrame Summary:")
print(df_res[['model', 'meta_path', 'mean_organ_f1', 'micro_f1', 'macro_f1']])

print("\n" + "="*90)
print("TRAINING COMPLETE!")
print("="*90)
