"""
Enhanced Validation Metrics for HAN Medical Prediction
Adds accuracy computation and enhanced plotting capabilities.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


def compute_accuracy(model, patient_feats, labels_severity, neighbor_dicts, idx_set):
    """
    Compute accuracy for multi-organ classification.
    
    IMPORTANT: This function REQUIRES ground truth labels for validation!
    Without labels, we cannot measure model performance.
    
    Args:
        model: HAN model
        patient_feats: patient feature tensor [P, D]
        labels_severity: ground truth severity labels [P, O] (from patient-one-hot-labeled-disease-new.csv)
        neighbor_dicts: dictionary of meta-path neighbors
        idx_set: set of indices to evaluate on
    
    Returns:
        dict: {
            'overall_accuracy': float - accuracy across all organs and patients
            'per_organ_accuracy': list - accuracy for each organ
            'mean_organ_accuracy': float - mean accuracy across organs
        }
    """
    model.eval()
    with torch.no_grad():
        organ_logits, _, _, _ = model(patient_feats, neighbor_dicts)
        preds = torch.argmax(organ_logits, dim=2).cpu().numpy()  # (P, O)
        y_true = labels_severity.cpu().numpy()
    
    # Create mask for evaluation indices
    mask = np.zeros(y_true.shape[0], dtype=bool)
    mask[list(idx_set)] = True
    
    # Overall accuracy (all organs, all patients in idx_set)
    correct = (preds[mask] == y_true[mask]).sum()
    total = preds[mask].size
    overall_acc = correct / total if total > 0 else 0.0
    
    # Per-organ accuracy
    per_organ_acc = []
    for o_idx in range(y_true.shape[1]):
        y_true_o = y_true[mask, o_idx]
        preds_o = preds[mask, o_idx]
        organ_acc = accuracy_score(y_true_o, preds_o)
        per_organ_acc.append(organ_acc)
    
    return {
        'overall_accuracy': float(overall_acc),
        'per_organ_accuracy': per_organ_acc,
        'mean_organ_accuracy': float(np.mean(per_organ_acc)) if per_organ_acc else 0.0
    }


def plot_training_metrics_enhanced(train_losses, val_losses, val_hist, val_micro_f1, val_macro_f1,
                                   train_acc, val_acc, model_name, meta_path, save_path):
    """
    Create comprehensive training plots including accuracy curves.
    
    Creates 6 subplots:
    1. Training & Validation Loss
    2. Validation Mean Organ F1
    3. Micro & Macro F1 Scores
    4. Training & Validation Accuracy (NEW!)
    5. F1 vs Accuracy Comparison (NEW!)
    6. Summary Statistics Table (NEW!)
    
    Args:
        train_losses: list of training losses per epoch
        val_losses: list of validation losses
        val_hist: list of mean organ F1 scores
        val_micro_f1: list of micro F1 scores
        val_macro_f1: list of macro F1 scores
        train_acc: list of training accuracies (NEW!)
        val_acc: list of validation accuracies (NEW!)
        model_name: name of the model (e.g., "HAN++", "HGT-HAN")
        meta_path: meta-path used (e.g., "P-O-P")
        save_path: path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_name} Training Metrics - Meta-path: {meta_path}', 
                 fontsize=16, fontweight='bold')
    
    # ==================== Plot 1: Training & Validation Loss ====================
    ax1 = axes[0, 0]
    epochs_train = range(1, len(train_losses) + 1)
    ax1.plot(epochs_train, train_losses, label='Train Loss', marker='o', 
             linewidth=2, markersize=4, color='steelblue')
    
    if val_losses:
        # Val loss is recorded every 2 epochs (or as specified)
        epochs_val = [1] + list(range(2, len(val_losses)*2 + 1, 2))
        epochs_val = epochs_val[:len(val_losses)]
        ax1.plot(epochs_val, val_losses, label='Val Loss', marker='s', 
                 linewidth=2, markersize=4, color='coral')
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # ==================== Plot 2: F1 Scores (Mean Organ) ====================
    ax2 = axes[0, 1]
    if val_hist:
        epochs_val = [1] + list(range(2, len(val_hist)*2 + 1, 2))
        epochs_val = epochs_val[:len(val_hist)]
        ax2.plot(epochs_val, val_hist, label='Val Mean Organ F1', marker='o', 
                 linewidth=2, markersize=4, color='green')
        ax2.axhline(y=max(val_hist), color='red', linestyle='--', linewidth=1, 
                    alpha=0.7, label=f'Best: {max(val_hist):.4f}')
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Mean Organ F1', fontsize=11)
    ax2.set_title('Validation Mean Organ F1 Score', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # ==================== Plot 3: Micro & Macro F1 ====================
    ax3 = axes[0, 2]
    if val_micro_f1:
        epochs_val = [1] + list(range(2, len(val_micro_f1)*2 + 1, 2))
        epochs_val = epochs_val[:len(val_micro_f1)]
        ax3.plot(epochs_val, val_micro_f1, label='Micro F1', marker='o', 
                 linewidth=2, markersize=4, color='purple')
    
    if val_macro_f1:
        epochs_val = [1] + list(range(2, len(val_macro_f1)*2 + 1, 2))
        epochs_val = epochs_val[:len(val_macro_f1)]
        ax3.plot(epochs_val, val_macro_f1, label='Macro F1', marker='s', 
                 linewidth=2, markersize=4, color='orange')
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('F1 Score', fontsize=11)
    ax3.set_title('Micro & Macro F1 Scores', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # ==================== Plot 4: Training & Validation Accuracy (NEW!) ====================
    ax4 = axes[1, 0]
    if train_acc:
        epochs_train = range(1, len(train_acc) + 1)
        ax4.plot(epochs_train, train_acc, label='Train Accuracy', marker='o', 
                 linewidth=2, markersize=4, color='steelblue')
    
    if val_acc:
        epochs_val = [1] + list(range(2, len(val_acc)*2 + 1, 2))
        epochs_val = epochs_val[:len(val_acc)]
        ax4.plot(epochs_val, val_acc, label='Val Accuracy', marker='s', 
                 linewidth=2, markersize=4, color='coral')
        ax4.axhline(y=max(val_acc), color='red', linestyle='--', linewidth=1, 
                    alpha=0.7, label=f'Best Val: {max(val_acc):.4f}')
    
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Accuracy', fontsize=11)
    ax4.set_title('Training & Validation Accuracy ⭐', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # ==================== Plot 5: F1 vs Accuracy Comparison (NEW!) ====================
    ax5 = axes[1, 1]
    if val_hist and val_acc:
        epochs_val = [1] + list(range(2, min(len(val_hist), len(val_acc))*2 + 1, 2))
        epochs_val = epochs_val[:min(len(val_hist), len(val_acc))]
        ax5.plot(epochs_val, val_hist[:len(epochs_val)], label='F1 Score', marker='o', 
                 linewidth=2, markersize=4, color='green')
        ax5.plot(epochs_val, val_acc[:len(epochs_val)], label='Accuracy', marker='s', 
                 linewidth=2, markersize=4, color='coral')
    
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Score', fontsize=11)
    ax5.set_title('Validation: F1 vs Accuracy ⭐', fontsize=12, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # ==================== Plot 6: Summary Statistics Table (NEW!) ====================
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_data = []
    if train_losses:
        summary_data.append(['Final Train Loss', f'{train_losses[-1]:.4f}'])
    if val_losses:
        summary_data.append(['Final Val Loss', f'{val_losses[-1]:.4f}'])
    if val_hist:
        summary_data.append(['Best Val F1', f'{max(val_hist):.4f}'])
        summary_data.append(['Final Val F1', f'{val_hist[-1]:.4f}'])
    if val_acc:
        summary_data.append(['Best Val Acc ⭐', f'{max(val_acc):.4f}'])
        summary_data.append(['Final Val Acc ⭐', f'{val_acc[-1]:.4f}'])
    if val_micro_f1:
        summary_data.append(['Final Micro F1', f'{val_micro_f1[-1]:.4f}'])
    if val_macro_f1:
        summary_data.append(['Final Macro F1', f'{val_macro_f1[-1]:.4f}'])
    
    if summary_data:
        table = ax6.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                         cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_data) + 1):
            if i % 2 == 0:
                table[(i, 0)].set_facecolor('#f0f0f0')
                table[(i, 1)].set_facecolor('#f0f0f0')
    
    ax6.set_title('Training Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Enhanced plot saved to {save_path}")


# ==================== Quick Usage Example ====================
"""
USAGE IN TRAINING LOOP:

# Import the new functions
from HAN.validation_metrics import compute_accuracy, plot_training_metrics_enhanced

# Initialize accuracy tracking lists
train_accuracies = []
val_accuracies = []

# In your training loop:
for epoch in range(1, EPOCHS + 1):
    # ... train model ...
    
    # Compute training accuracy
    train_acc_dict = compute_accuracy(
        model, patient_feats, labels_organ_severity, 
        neighbor_dicts, train_idx
    )
    train_accuracies.append(train_acc_dict['overall_accuracy'])
    
    # During validation:
    if epoch % 2 == 0:
        # ... compute val loss and F1 ...
        
        # Compute validation accuracy
        val_acc_dict = compute_accuracy(
            model, patient_feats, labels_organ_severity,
            neighbor_dicts, val_idx
        )
        val_accuracies.append(val_acc_dict['overall_accuracy'])
        
        print(f"Epoch {epoch} | val_f1={val_f1:.4f} | val_acc={val_acc_dict['overall_accuracy']:.4f}")

# After training, create enhanced plots
plot_training_metrics_enhanced(
    train_losses, val_losses, val_hist, val_micro_f1, val_macro_f1,
    train_accuracies, val_accuracies,  # NEW!
    "HAN++", "P-O-P", "output/training_plots.png"
)
"""
