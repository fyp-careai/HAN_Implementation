#!/usr/bin/env python3
"""
Complete Training Script with Accuracy Tracking and Enhanced Visualization

This script provides a complete, proper training setup for the HAN medical 
prediction model with:
- Automatic label generation from test values
- Accuracy tracking alongside F1 scores
- Enhanced 6-subplot visualization
- Best model saving with early stopping
- Comprehensive results logging

Usage:
    python train_complete.py
"""

import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Import HAN components with enhanced validation metrics
from HAN import (
    MedicalGraphData,
    AttentionLayer,
    HANModel,
    SubgraphSampler,
    evaluate_model,
    compute_accuracy,              # For accuracy computation
    plot_training_metrics_enhanced  # For enhanced 6-subplot visualization
)


def verify_data_files(path_records, path_symptom):
    """Verify that all required data files exist."""
    print("\n" + "="*80)
    print("VERIFYING DATA FILES")
    print("="*80)
    
    files_to_check = {
        "Patient Records (Features)": path_records,
        "Symptom/Disease Map": path_symptom,
        "Label File (pre-computed)": "data/patient-one-hot-labeled-disease-new.csv"
    }
    
    all_required_exist = True
    for name, path in files_to_check.items():
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        required = "(optional)" if "Label File" in name else "(required)"
        status = "✅" if exists else "❌"
        
        print(f"{status} {name:30} {required:12}")
        print(f"   Path: {abs_path}")
        
        if "required" in required and not exists:
            all_required_exist = False
    
    print()
    if all_required_exist:
        print("✅ All required data files found!")
        print("📝 Labels will be computed automatically from patient test records")
        print("   by comparing test values against normal ranges.")
    else:
        print("❌ Some required files are missing!")
        raise FileNotFoundError("Required data files missing!")
    
    return all_required_exist


def train_model_with_accuracy(
    model, optimizer, criterion, 
    features, labels, adj_dict,
    train_idx, val_idx,
    epochs=40, patience=10, device='cpu', out_dir='output'
):
    """
    Training loop with accuracy tracking for enhanced visualization.
    
    Returns:
        dict: Contains all metrics including train/val losses, F1 scores, and accuracies
    """
    print("\n" + "="*80)
    print("STARTING TRAINING WITH ACCURACY TRACKING")
    print("="*80)
    
    # Move data to device
    features = features.to(device)
    labels = labels.to(device)
    model = model.to(device)
    
    # Initialize metric tracking
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    train_accuracies = []  # Track training accuracy
    val_accuracies = []    # Track validation accuracy
    
    val_micro_f1 = []
    val_macro_f1 = []
    
    best_val_f1 = 0.0
    best_val_acc = 0.0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # === TRAINING PHASE ===
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features, adj_dict)
        train_outputs = outputs[train_idx]
        train_labels_batch = labels[train_idx]
        
        # Compute loss
        loss = criterion(train_outputs, train_labels_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record training metrics
        train_losses.append(loss.item())
        
        # Compute training F1 score and accuracy
        with torch.no_grad():
            train_preds = (torch.sigmoid(train_outputs) > 0.5).float()
            train_f1 = f1_score(
                train_labels_batch.cpu().numpy(),
                train_preds.cpu().numpy(),
                average='samples',
                zero_division=0
            )
            train_f1_scores.append(train_f1)
            
            # Compute training accuracy
            train_acc_dict = compute_accuracy(
                train_labels_batch.cpu().numpy(),
                train_preds.cpu().numpy()
            )
            train_accuracies.append(train_acc_dict['overall_accuracy'])
        
        # === VALIDATION PHASE ===
        model.eval()
        with torch.no_grad():
            val_outputs = outputs[val_idx]
            val_labels_batch = labels[val_idx]
            
            # Compute validation loss
            val_loss = criterion(val_outputs, val_labels_batch)
            val_losses.append(val_loss.item())
            
            # Compute validation predictions
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            
            # Compute validation F1 scores
            val_f1 = f1_score(
                val_labels_batch.cpu().numpy(),
                val_preds.cpu().numpy(),
                average='samples',
                zero_division=0
            )
            val_f1_scores.append(val_f1)
            
            val_micro = f1_score(
                val_labels_batch.cpu().numpy(),
                val_preds.cpu().numpy(),
                average='micro',
                zero_division=0
            )
            val_micro_f1.append(val_micro)
            
            val_macro = f1_score(
                val_labels_batch.cpu().numpy(),
                val_preds.cpu().numpy(),
                average='macro',
                zero_division=0
            )
            val_macro_f1.append(val_macro)
            
            # Compute validation accuracy
            val_acc_dict = compute_accuracy(
                val_labels_batch.cpu().numpy(),
                val_preds.cpu().numpy()
            )
            val_accuracies.append(val_acc_dict['overall_accuracy'])
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {loss.item():.4f} → {val_loss.item():.4f} | "
                  f"F1: {train_f1:.4f} → {val_f1:.4f} | "
                  f"Acc: {train_acc_dict['overall_accuracy']:.4f} → {val_acc_dict['overall_accuracy']:.4f}")
        
        # Track best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc_dict['overall_accuracy']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time:.1f}s!")
    print(f"   Best validation F1: {best_val_f1:.4f}")
    print(f"   Best validation Accuracy: {best_val_acc:.4f}")
    
    # Return all metrics
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'val_micro_f1': val_micro_f1,
        'val_macro_f1': val_macro_f1,
        'best_val_f1': best_val_f1,
        'best_val_acc': best_val_acc,
        'total_epochs': len(train_losses),
        'training_time': training_time
    }


def main():
    """Main training pipeline."""
    
    # ========== CONFIGURATION ==========
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training hyperparameters
    SEED = 42
    BATCH_SIZE = 128
    EPOCHS = 40
    LR = 1e-3
    PATIENCE = 10
    
    # Data paths - Using relative paths for local data folder
    PATH_RECORDS = "data/filtered_patient_reports.csv"
    PATH_SYMPTOM = "data/test-disease-organ.csv"
    
    # Data preprocessing parameters
    SYMPTOM_FREQ_THRESHOLD = 5
    PRUNE_PER_PATIENT = True
    NNZ_THRESHOLD = 3
    
    # Output directory
    OUT_DIR = "output"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    print("="*80)
    print("HAN MEDICAL PREDICTION - COMPLETE TRAINING SETUP")
    print("="*80)
    print(f"✓ Device: {DEVICE}")
    print(f"✓ Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    print(f"✓ Learning rate: {LR}")
    print(f"✓ Early stopping patience: {PATIENCE}")
    
    # ========== VERIFY DATA FILES ==========
    verify_data_files(PATH_RECORDS, PATH_SYMPTOM)
    
    # ========== LOAD AND PROCESS DATA ==========
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
    
    # Load and process (automatically generates labels!)
    data_loader.load_data()
    data_loader.build_labels_and_features()
    data_loader.build_adjacency_matrices()
    
    print(f"✅ Data loaded successfully!")
    print(f"   Patients: {len(data_loader.patient_list)}")
    print(f"   Symptoms: {len(data_loader.symptom_list)}")
    print(f"   Labels shape: {data_loader.labels.shape}")
    print(f"   Features shape: {data_loader.features.shape}")
    
    # ========== INITIALIZE MODEL ==========
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    INPUT_DIM = data_loader.features.shape[1]
    HIDDEN_DIM = 64
    OUTPUT_DIM = data_loader.labels.shape[1]
    NUM_HEADS = 8
    DROPOUT = 0.3
    
    model = HANModel(
        in_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=OUTPUT_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"✓ Model initialized: {INPUT_DIM} → {HIDDEN_DIM} → {OUTPUT_DIM}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Attention heads: {NUM_HEADS}")
    print(f"   Dropout: {DROPOUT}")
    
    # Get train/val split
    train_idx = data_loader.train_idx
    val_idx = data_loader.val_idx
    
    print(f"✓ Training set: {len(train_idx)} patients ({len(train_idx)/len(data_loader.patient_list)*100:.1f}%)")
    print(f"✓ Validation set: {len(val_idx)} patients ({len(val_idx)/len(data_loader.patient_list)*100:.1f}%)")
    
    # ========== TRAIN MODEL ==========
    results = train_model_with_accuracy(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        features=data_loader.features,
        labels=data_loader.labels,
        adj_dict=data_loader.adj_dict,
        train_idx=train_idx,
        val_idx=val_idx,
        epochs=EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        out_dir=OUT_DIR
    )
    
    # ========== GENERATE VISUALIZATIONS ==========
    print("\n" + "="*80)
    print("GENERATING ENHANCED VISUALIZATIONS")
    print("="*80)
    
    plot_path = os.path.join(OUT_DIR, "training_metrics_enhanced.png")
    
    plot_training_metrics_enhanced(
        train_losses=results['train_losses'],
        val_losses=results['val_losses'],
        train_f1=results['train_f1_scores'],
        val_f1=results['val_f1_scores'],
        val_micro_f1=results['val_micro_f1'],
        val_macro_f1=results['val_macro_f1'],
        train_accuracies=results['train_accuracies'],
        val_accuracies=results['val_accuracies'],
        model_name="HAN Medical Predictor",
        figsize=(18, 12),
        save_path=plot_path
    )
    
    print(f"✅ Enhanced visualization saved to: {plot_path}")
    print("   Plot contains 6 subplots:")
    print("   1. Training & Validation Loss")
    print("   2. F1 Score (Samples Average)")
    print("   3. Micro & Macro F1 Scores")
    print("   4. Overall Accuracy")
    print("   5. F1 vs Accuracy Comparison")
    print("   6. Final Metrics Summary Table")
    
    # ========== SAVE RESULTS ==========
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print(f"{'Metric':<25} {'Training':<15} {'Validation':<15}")
    print("-" * 55)
    print(f"{'Loss':<25} {results['train_losses'][-1]:<15.4f} {results['val_losses'][-1]:<15.4f}")
    print(f"{'F1 Score (Samples)':<25} {results['train_f1_scores'][-1]:<15.4f} {results['val_f1_scores'][-1]:<15.4f}")
    print(f"{'Accuracy':<25} {results['train_accuracies'][-1]:<15.4f} {results['val_accuracies'][-1]:<15.4f}")
    print(f"{'Micro F1':<25} {'-':<15} {results['val_micro_f1'][-1]:<15.4f}")
    print(f"{'Macro F1':<25} {'-':<15} {results['val_macro_f1'][-1]:<15.4f}")
    print()
    print(f"{'Best Validation F1':<25} {results['best_val_f1']:<15.4f}")
    print(f"{'Best Validation Accuracy':<25} {results['best_val_acc']:<15.4f}")
    print(f"{'Total Epochs':<25} {results['total_epochs']:<15}")
    print(f"{'Training Time':<25} {results['training_time']:<15.1f}s")
    
    # Save results to JSON
    results_json = {
        'best_val_f1': float(results['best_val_f1']),
        'best_val_acc': float(results['best_val_acc']),
        'final_train_loss': float(results['train_losses'][-1]),
        'final_val_loss': float(results['val_losses'][-1]),
        'final_train_f1': float(results['train_f1_scores'][-1]),
        'final_val_f1': float(results['val_f1_scores'][-1]),
        'final_train_acc': float(results['train_accuracies'][-1]),
        'final_val_acc': float(results['val_accuracies'][-1]),
        'final_val_micro_f1': float(results['val_micro_f1'][-1]),
        'final_val_macro_f1': float(results['val_macro_f1'][-1]),
        'total_epochs': results['total_epochs'],
        'training_time_seconds': float(results['training_time']),
        'device': str(DEVICE),
        'config': {
            'seed': SEED,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LR,
            'patience': PATIENCE,
            'hidden_dim': HIDDEN_DIM,
            'num_heads': NUM_HEADS,
            'dropout': DROPOUT
        }
    }
    
    json_path = os.path.join(OUT_DIR, 'training_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to: {json_path}")
    print(f"✅ Best model saved to: {os.path.join(OUT_DIR, 'best_model.pth')}")
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)
    

if __name__ == "__main__":
    main()
