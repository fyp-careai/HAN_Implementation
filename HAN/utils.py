"""
HAN Utils Module
Contains utility functions for HAN model training and evaluation.
"""

import os
import random
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


# ---------------- Helper Functions ----------------

def try_float(x):
    """Convert value to float, return None if conversion fails."""
    try:
        return float(x)
    except:
        return None


def parse_normal_range(row):
    """
    Parse normal range from row data.
    Returns (low, high) tuple or (None, None) if invalid.
    
    Handles multiple column name formats:
    - 'Min'/'Max' (lowercase: min/max)
    - 'Normal Range' (single string)
    - 'Under'/'Over'
    """
    # Try Min/Max columns first (most common format)
    low = try_float(row.get('Min', None))
    high = try_float(row.get('Max', None))
    if low is not None and high is not None and high >= low:
        return low, high
    
    # Try lowercase min/max
    low = try_float(row.get('min', None))
    high = try_float(row.get('max', None))
    if low is not None and high is not None and high >= low:
        return low, high
    
    # Try parsing from 'Normal Range' string
    rng = row.get('Normal Range', None)
    if pd.notna(rng):
        s = str(rng)
        m = re.findall(r"[-+]?\d*\.?\d+|\d+", s)
        if len(m) >= 2:
            low = try_float(m[0])
            high = try_float(m[1])
            if low is not None and high is not None and high >= low:
                return low, high
    
    # Try Under/Over columns
    low = try_float(row.get('Under', None))
    high = try_float(row.get('Over', None))
    if low is not None and high is not None and high >= low:
        return low, high
    
    return None, None


# ---------------- Focal Loss ----------------

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits (before sigmoid/softmax)
            targets: ground truth labels (one-hot or class indices)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# ---------------- Neighbor Processing Functions ----------------

def csr_to_neighbors_prune(M, max_per_node=300, seed=42):
    """
    Convert CSR matrix to neighbor dictionary with pruning.
    
    Args:
        M: scipy.sparse.csr_matrix - adjacency matrix
        max_per_node: maximum neighbors to keep per node
        seed: random seed for reproducibility
    
    Returns:
        dict: {node_id: [neighbor_ids]}
    """
    N = M.shape[0]
    neigh = {i: [] for i in range(N)}
    
    if M.nnz == 0:
        return neigh
    
    Mcoo = M.tocoo()
    rows = Mcoo.row
    cols = Mcoo.col
    
    grouped = defaultdict(list)
    for r, c in zip(rows, cols):
        grouped[int(r)].append(int(c))
    
    for i in range(N):
        lst = grouped.get(i, [])
        if max_per_node and len(lst) > max_per_node:
            random.seed(seed + i)
            lst = random.sample(lst, max_per_node)
        neigh[i] = lst
    
    return neigh


def neighbors_to_padded_tensors(neighbor_dict, N, max_neighbors=300, device='cpu'):
    """
    Convert neighbor dictionary to padded tensors for vectorized attention.
    
    Args:
        neighbor_dict: dict of {node_id: [neighbor_ids]}
        N: total number of nodes
        max_neighbors: maximum neighbors per node
        device: torch device
    
    Returns:
        tuple: (neighbor_idx, neighbor_mask)
            - neighbor_idx: [N, max_neighbors] tensor of neighbor indices
            - neighbor_mask: [N, max_neighbors] tensor of validity mask
    """
    idx = np.zeros((N, max_neighbors), dtype=np.int64)
    mask = np.zeros((N, max_neighbors), dtype=np.float32)
    
    for i in range(N):
        neighbors = neighbor_dict.get(i, [])
        if len(neighbors) == 0:
            neighbors = [i]  # self-loop if no neighbors
        
        n_valid = min(len(neighbors), max_neighbors)
        idx[i, :n_valid] = neighbors[:n_valid]
        mask[i, :n_valid] = 1.0
    
    return torch.tensor(idx, device=device), torch.tensor(mask, device=device)


# ---------------- Loss Computation ----------------

def compute_loss_multiorg(organ_logits, labels_severity, idx_set, organ_class_weights, 
                          use_focal=True, focal_gamma=2.0):
    """
    Compute multi-organ classification loss.
    
    Args:
        organ_logits: [N, num_organs, num_classes] predicted logits
        labels_severity: [N, num_organs] ground truth severity levels
        idx_set: set of indices to compute loss on (train/val split)
        organ_class_weights: list of class weight tensors per organ
        use_focal: whether to use focal loss
        focal_gamma: gamma parameter for focal loss
    
    Returns:
        torch.Tensor: computed loss
    """
    mask = torch.zeros(labels_severity.shape[0], dtype=torch.bool, device=labels_severity.device)
    mask[list(idx_set)] = True
    
    total_loss = torch.tensor(0.0, device=organ_logits.device, requires_grad=True)
    valid_organs = 0
    
    for o_idx in range(labels_severity.shape[1]):
        logits_o = organ_logits[mask, o_idx, :]  # (N_mask, num_classes)
        targets_o = labels_severity[mask, o_idx]  # (N_mask,)
        
        if logits_o.shape[0] == 0:
            continue
        
        unique_classes = torch.unique(targets_o)
        if len(unique_classes) < 2:
            continue
        
        valid_organs += 1
        
        if use_focal:
            # Focal loss with class weights
            ce = F.cross_entropy(logits_o, targets_o, weight=organ_class_weights[o_idx], reduction='none')
            probs = F.softmax(logits_o, dim=1)
            pt = probs.gather(1, targets_o.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt) ** focal_gamma
            loss_o = (focal_weight * ce).mean()
        else:
            loss_o = F.cross_entropy(logits_o, targets_o, weight=organ_class_weights[o_idx])
        
        total_loss = total_loss + loss_o
    
    if valid_organs > 0:
        return total_loss / valid_organs
    else:
        # Return a small loss tensor instead of 0.0 to maintain gradient flow
        return torch.tensor(0.0, device=organ_logits.device, requires_grad=True)


# ---------------- Evaluation Functions ----------------

@torch.no_grad()
def evaluate_multiorg(model, patient_feats, labels_severity, neighbor_dicts, idx_set):
    """
    Evaluate multi-organ classification model.
    
    Args:
        model: HAN model
        patient_feats: patient feature tensor
        labels_severity: ground truth severity labels
        neighbor_dicts: dictionary of meta-path neighbors
        idx_set: set of indices to evaluate on
    
    Returns:
        dict: evaluation metrics including F1 scores
    """
    model.eval()
    organ_logits, organ_scores, z, beta = model(patient_feats, neighbor_dicts)
    
    preds = torch.argmax(organ_logits, dim=2).cpu().numpy()  # (P, O)
    y_true = labels_severity.cpu().numpy()
    
    mask = np.zeros(y_true.shape[0], dtype=bool)
    mask[list(idx_set)] = True
    
    organ_f1s = []
    valid_organ_indices = []
    
    for o_idx in range(y_true.shape[1]):
        y_true_o = y_true[mask, o_idx]
        preds_o = preds[mask, o_idx]
        
        unique_classes = np.unique(y_true_o)
        if len(unique_classes) < 2:
            organ_f1s.append(0.0)
            continue
        
        valid_organ_indices.append(o_idx)
        f1 = f1_score(y_true_o, preds_o, average='macro', zero_division=0)
        organ_f1s.append(f1)
    
    if len(valid_organ_indices) > 0:
        y_true_valid = y_true[mask][:, valid_organ_indices].flatten()
        preds_valid = preds[mask][:, valid_organ_indices].flatten()
        
        metrics = {}
        metrics['macro_f1'] = f1_score(y_true_valid, preds_valid, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true_valid, preds_valid, average='micro', zero_division=0)
        metrics['per_organ_f1'] = organ_f1s
        
        valid_f1s = [organ_f1s[i] for i in valid_organ_indices]
        metrics['mean_organ_f1'] = float(np.mean(valid_f1s)) if valid_f1s else 0.0
        metrics['num_valid_organs'] = len(valid_organ_indices)
    else:
        metrics = {
            'macro_f1': 0.0,
            'micro_f1': 0.0,
            'per_organ_f1': organ_f1s,
            'mean_organ_f1': 0.0,
            'num_valid_organs': 0
        }
    
    # beta is [N, K] for HANPP (patient-conditioned) or [K] for legacy models.
    # Store the mean across patients so logging is consistent either way.
    beta_np = beta.cpu().numpy()
    metrics['beta'] = beta_np.mean(axis=0) if beta_np.ndim == 2 else beta_np
    metrics['beta_per_patient'] = beta_np  # full [N, K] for interpretability analysis
    return metrics


# ---------------- Plotting Functions ----------------

def plot_training_metrics(train_losses, val_losses, val_hist, val_micro_f1, val_macro_f1, 
                          model_name, meta_path, save_path):
    """
    Create comprehensive training plots.
    
    Args:
        train_losses: list of training losses
        val_losses: list of validation losses
        val_hist: list of validation mean F1 scores
        val_micro_f1: list of validation micro F1 scores
        val_macro_f1: list of validation macro F1 scores
        model_name: name of the model
        meta_path: meta-path being used
        save_path: path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_name} Training Metrics - {meta_path}', fontsize=16, fontweight='bold')
    
    # Compute evaluation epochs (every 2 epochs starting from 1)
    eval_epochs = [2*i+1 if i==0 else 2*i for i in range(len(val_losses))]
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(train_losses)+1), train_losses, 'b-', linewidth=2, alpha=0.7, label='Train Loss')
    ax1.plot(eval_epochs, val_losses, 'r-', linewidth=2, marker='o', markersize=4, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Mean F1 Score
    ax2 = axes[0, 1]
    ax2.plot(eval_epochs, val_hist, 'g-', linewidth=2, marker='s', markersize=4, label='Val Mean F1')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title('Validation Mean F1 Score', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation Micro-F1
    ax3 = axes[0, 2]
    ax3.plot(eval_epochs, val_micro_f1, 'c-', linewidth=2, marker='d', markersize=4, label='Val Micro-F1')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Micro-F1 Score', fontsize=11)
    ax3.set_title('Validation Micro-F1 Score', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Macro-F1
    ax4 = axes[1, 0]
    ax4.plot(eval_epochs, val_macro_f1, 'm-', linewidth=2, marker='^', markersize=4, label='Val Macro-F1')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Macro-F1 Score', fontsize=11)
    ax4.set_title('Validation Macro-F1 Score', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: All F1 Scores Comparison
    ax5 = axes[1, 1]
    ax5.plot(eval_epochs, val_hist, 'g-', linewidth=2, marker='o', markersize=3, label='Mean F1', alpha=0.7)
    ax5.plot(eval_epochs, val_micro_f1, 'c-', linewidth=2, marker='s', markersize=3, label='Micro-F1', alpha=0.7)
    ax5.plot(eval_epochs, val_macro_f1, 'm-', linewidth=2, marker='^', markersize=3, label='Macro-F1', alpha=0.7)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('F1 Score', fontsize=11)
    ax5.set_title('F1 Scores Comparison', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Loss vs F1 (dual axis)
    ax6 = axes[1, 2]
    ax6_twin = ax6.twinx()
    ln1 = ax6.plot(eval_epochs, val_losses, 'r-', linewidth=2, marker='o', markersize=3, label='Val Loss')
    ln2 = ax6_twin.plot(eval_epochs, val_hist, 'g-', linewidth=2, marker='s', markersize=3, label='Val Mean F1')
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Loss', fontsize=11, color='r')
    ax6_twin.set_ylabel('F1 Score', fontsize=11, color='g')
    ax6.set_title('Loss vs F1 Score', fontsize=12, fontweight='bold')
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax6.legend(lns, labs, fontsize=9, loc='center right')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {save_path}")
