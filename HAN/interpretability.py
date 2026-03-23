"""
Attention Interpretability for CareAI HAN++
=============================================

Extracts and visualises two levels of attention:

1. SEMANTIC-LEVEL (beta): Per-patient weights over meta-paths [N, K]
   - Which meta-path (P-D-P, P-O-P, P-S-P) did each patient prefer?
   - Enabled by our patient-conditioned semantic attention (Option A)
   - Clinical use: explain WHY disease X was predicted
     (e.g. "60% weight on P-D-P → this patient's prediction is driven
      by sharing diseases with similar patients")

2. NODE-LEVEL (alpha): Per-patient weights over neighbours [N, max_neigh]
   - Which specific patients are most influential for this prediction?
   - High alpha_ij means "patient j strongly influenced patient i"
   - Clinical use: find the most similar patients in the training set
     and show their confirmed diagnoses as evidence

This module supports:
- extract_semantic_attention(): get beta [N, K] for all patients
- extract_node_attention(): get alpha weights per meta-path
- plot_metapath_preference(): heatmap of beta across patient groups
- plot_top_neighbours(): for a query patient, show top influential peers
- generate_explanation(): human-readable explanation for one patient

Reference:
    Vaswani et al., "Attention is All You Need", NeurIPS 2017.
    Wang et al., "Heterogeneous Graph Attention Network", WWW 2019.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────
#  1. Attention Extraction
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_semantic_attention(model, features, neighbor_dicts, device=None):
    """
    Extract per-patient semantic (meta-path level) attention weights.

    With our patient-conditioned semantic attention, beta is [N, K],
    giving each patient their own meta-path preference weights.

    Args:
        model:          trained HANPP model
        features:       patient features [N, in_dim]
        neighbor_dicts: dict {metapath_name: neighbor_dict}
        device:         torch device

    Returns:
        beta: np.ndarray [N, K]  — per-patient meta-path weights (sum to 1)
        metapath_names: list of K meta-path names
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    features = features.to(device)

    _, _, _, beta = model(features, neighbor_dicts)
    beta_np = beta.cpu().numpy()  # [N, K] or [K] for legacy models

    if beta_np.ndim == 1:
        # Legacy global attention — broadcast to [N, K]
        beta_np = np.tile(beta_np, (features.shape[0], 1))

    return beta_np, list(model.metapath_names)


@torch.no_grad()
def extract_node_attention_for_patient(model, features, neighbor_dicts,
                                       patient_idx, metapath_idx=0, device=None):
    """
    Extract node-level attention weights (alpha) for one patient,
    one meta-path.

    Alpha_ij indicates how much patient i attends to neighbour j in this
    meta-path. High alpha = strong influence on prediction.

    Args:
        model:          trained HANPP model
        features:       patient features [N, in_dim]
        neighbor_dicts: dict {metapath_name: neighbor_dict}
        patient_idx:    index of the query patient
        metapath_idx:   which meta-path attention layer to inspect (0, 1, ...)
        device:         torch device

    Returns:
        neighbours:     list of neighbour patient indices
        alpha_weights:  np.ndarray of attention weights for those neighbours
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    features = features.to(device)
    N = features.size(0)

    metapath_name = model.metapath_names[metapath_idx]
    node_att = model.node_atts[metapath_idx]

    # Project features
    h = F.gelu(model.project(features))    # [N, hidden_dim]

    # Get pre-set neighbour tensors
    if node_att.neighbor_idx is not None:
        neigh_idx  = node_att.neighbor_idx[patient_idx]   # [max_neigh]
        neigh_mask = node_att.neighbor_mask[patient_idx]  # [max_neigh]

        h_proj = node_att.W(h)  # [N, out_dim]
        N_full, out_dim = h_proj.shape
        num_heads = node_att.num_heads
        head_dim = node_att.head_dim

        h_heads = h_proj.view(N_full, num_heads, head_dim)  # [N, H, D]

        self_h  = h_heads[patient_idx]  # [H, D]
        neigh_h = h_heads[neigh_idx]    # [max_neigh, H, D]

        # Compute attention for each head, then average
        el = (self_h.unsqueeze(0) * node_att.a_l.unsqueeze(0)).sum(dim=2)     # [1, H]
        er = (neigh_h * node_att.a_r.unsqueeze(0)).sum(dim=2)                  # [max_neigh, H]

        e = node_att.leaky(el + er)  # [max_neigh, H]
        e[neigh_mask == 0] = -1e9

        alpha = F.softmax(e, dim=0)    # [max_neigh, H]
        alpha_mean = alpha.mean(dim=1).cpu().numpy()  # [max_neigh] — average over heads

        valid = neigh_mask.cpu().numpy().astype(bool)
        return neigh_idx.cpu().numpy()[valid], alpha_mean[valid]

    # Fallback: loop-based (slower but always works)
    neigh_dict = neighbor_dicts.get(metapath_name, {})
    neighbours = neigh_dict.get(patient_idx, [])
    if not neighbours:
        return np.array([]), np.array([])

    h_proj = node_att.W(h)
    # Use first head only for simplicity in fallback
    hk = h_proj[:, :node_att.head_dim]
    hi = hk[patient_idx].unsqueeze(0).repeat(len(neighbours), 1)
    hj = hk[neighbours]

    el = (hi * node_att.a_l[0].unsqueeze(1).T).sum(dim=1)
    er = (hj * node_att.a_r[0].unsqueeze(1).T).sum(dim=1)
    e  = node_att.leaky(el + er)
    alpha = F.softmax(e, dim=0).cpu().numpy()

    return np.array(neighbours), alpha


# ─────────────────────────────────────────────
#  2. Visualisations
# ─────────────────────────────────────────────

def plot_metapath_preference(beta, metapath_names, patient_labels=None,
                             save_path='output/interpretability/metapath_preference.png',
                             top_n=50):
    """
    Heatmap: rows = patients (up to top_n), columns = meta-paths.
    Colour = attention weight (hotter = more weight on that meta-path).

    Args:
        beta:           [N, K] meta-path weights
        metapath_names: list of K names
        patient_labels: optional list of N strings (e.g. primary diagnosis)
        save_path:      where to save the PNG
        top_n:          max patients to show in heatmap
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    N = min(top_n, beta.shape[0])
    beta_show = beta[:N]

    fig, ax = plt.subplots(figsize=(max(6, len(metapath_names)*2), max(6, N//5)))
    im = ax.imshow(beta_show, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Attention weight')

    ax.set_xticks(range(len(metapath_names)))
    ax.set_xticklabels(metapath_names, fontsize=12)
    ax.set_xlabel('Meta-path', fontsize=12)
    ax.set_ylabel('Patient index', fontsize=12)
    ax.set_title('Patient-Conditioned Meta-Path Attention Weights (β)\n'
                 'Each row = one patient\'s learned preference over meta-paths', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metapath_distribution(beta, metapath_names,
                               save_path='output/interpretability/metapath_distribution.png'):
    """
    Violin/box plot showing the distribution of beta values across patients
    for each meta-path. Shows that different patients indeed prefer
    different meta-paths (the variance should be meaningful).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: box plot per meta-path
    ax = axes[0]
    ax.boxplot([beta[:, k] for k in range(beta.shape[1])],
               labels=metapath_names, patch_artist=True,
               boxprops=dict(facecolor='lightblue'))
    ax.set_ylabel('Attention weight β', fontsize=11)
    ax.set_title('Distribution of β across patients\n(each box = all patients for one meta-path)', fontsize=10)
    ax.axhline(1/beta.shape[1], linestyle='--', color='grey', alpha=0.5,
               label=f'Uniform = {1/beta.shape[1]:.2f}')
    ax.legend(fontsize=9)

    # Right: mean + std bar chart
    ax2 = axes[1]
    means = beta.mean(axis=0)
    stds  = beta.std(axis=0)
    x = range(len(metapath_names))
    ax2.bar(x, means, yerr=stds, color='steelblue', capsize=6, alpha=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(metapath_names, fontsize=11)
    ax2.set_ylabel('Mean attention weight', fontsize=11)
    ax2.set_title('Mean ± std of β per meta-path\n(std > 0 confirms patient-specific weighting)', fontsize=10)
    ax2.axhline(1/beta.shape[1], linestyle='--', color='grey', alpha=0.5)

    plt.suptitle('Patient-Conditioned Semantic Attention: Meta-Path Weight Analysis', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Print summary
    print("\nMeta-path attention statistics:")
    for k, name in enumerate(metapath_names):
        print(f"  {name}: mean={means[k]:.4f}, std={stds[k]:.4f}, "
              f"min={beta[:,k].min():.4f}, max={beta[:,k].max():.4f}")


def plot_top_neighbours(patient_idx, neighbours, alpha_weights,
                        patient_ids=None, top_k=10,
                        save_path='output/interpretability/top_neighbours.png'):
    """
    Bar chart: which patients most influenced the query patient's prediction?

    Args:
        patient_idx:   index of the query patient
        neighbours:    array of neighbour indices
        alpha_weights: array of attention weights for those neighbours
        patient_ids:   optional list of patient ID strings for labelling
        top_k:         how many top neighbours to show
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Sort by attention weight
    order = np.argsort(alpha_weights)[::-1][:top_k]
    top_neigh = neighbours[order]
    top_alpha = alpha_weights[order]

    # Labels
    if patient_ids is not None:
        labels = [str(patient_ids[n]) for n in top_neigh]
    else:
        labels = [f"P{n}" for n in top_neigh]

    fig, ax = plt.subplots(figsize=(max(8, top_k), 4))
    bars = ax.barh(range(len(labels)), top_alpha[::-1], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.set_xlabel('Node-level attention weight α', fontsize=11)
    pid = patient_ids[patient_idx] if patient_ids is not None else patient_idx
    ax.set_title(f'Top {top_k} Most Influential Neighbours for Patient {pid}\n'
                 f'(higher α = this patient\'s record most shaped the prediction)', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
#  3. Human-Readable Explanation
# ─────────────────────────────────────────────

def generate_explanation(patient_idx, beta, metapath_names,
                         neighbours=None, alpha_weights=None,
                         patient_ids=None, predictions=None,
                         organ_names=None, severity_names=None):
    """
    Generate a human-readable clinical explanation for one patient.

    Returns:
        list of strings (lines of the explanation)
    """
    lines = []
    pid = patient_ids[patient_idx] if patient_ids is not None else patient_idx

    lines.append(f"=== Attention-Based Explanation for Patient {pid} ===")
    lines.append("")

    # Meta-path preferences
    lines.append("WHY did the model make this prediction?")
    lines.append("The model used the following information sources (meta-paths):")
    beta_i = beta[patient_idx] if beta.ndim == 2 else beta
    for k, name in enumerate(metapath_names):
        w = beta_i[k] * 100
        bar = '#' * int(w // 5)
        desc = {
            'P-D-P': 'patients sharing similar diseases',
            'P-O-P': 'patients with similar organ involvement',
            'P-S-P': 'patients with similar lab test results',
        }.get(name, name)
        lines.append(f"  {name}: {w:5.1f}%  {bar}  ({desc})")
    lines.append("")

    dominant_k = int(np.argmax(beta_i))
    lines.append(f"=> Primary information source: {metapath_names[dominant_k]} "
                 f"({beta_i[dominant_k]*100:.1f}%)")
    lines.append("")

    # Top neighbours
    if neighbours is not None and len(neighbours) > 0 and alpha_weights is not None:
        top5_idx = np.argsort(alpha_weights)[::-1][:5]
        top5_neigh = neighbours[top5_idx]
        top5_alpha = alpha_weights[top5_idx]
        lines.append("Most similar patients (top 5 by attention weight):")
        for rank, (n, a) in enumerate(zip(top5_neigh, top5_alpha), 1):
            nid = patient_ids[n] if patient_ids is not None else n
            lines.append(f"  {rank}. Patient {nid}  (influence: {a*100:.1f}%)")
        lines.append("")

    # Predictions if provided
    if predictions is not None and organ_names is not None:
        affected = [(organ_names[o], int(predictions[patient_idx, o]))
                    for o in range(len(organ_names))
                    if o < predictions.shape[1] and predictions[patient_idx, o] > 0]
        if affected:
            lines.append("Predicted abnormalities:")
            sev_map = {0: 'NORMAL', 1: 'MILD', 2: 'MODERATE', 3: 'SEVERE'}
            for organ, sev in sorted(affected, key=lambda x: -x[1]):
                lines.append(f"  - {organ}: {sev_map.get(sev, sev)}")
        lines.append("")

    lines.append("NOTE: This explanation is generated automatically from the model's")
    lines.append("attention weights. It indicates which data sources the model relied")
    lines.append("on, not a causal medical explanation. Always consult a physician.")

    return lines


# ─────────────────────────────────────────────
#  4. Batch Analysis
# ─────────────────────────────────────────────

def run_interpretability_analysis(model, features, neighbor_dicts,
                                  patient_ids=None, predictions=None,
                                  organ_names=None, output_dir='output/interpretability',
                                  device=None, sample_patients=5):
    """
    Run full interpretability analysis: extract beta, plot distributions,
    explain a sample of patients.

    Args:
        model:           trained HANPP model
        features:        patient features [N, in_dim]
        neighbor_dicts:  meta-path neighbour dicts
        patient_ids:     list of patient ID strings (optional)
        predictions:     [N, O] predicted severities (optional)
        organ_names:     list of organ names (optional)
        output_dir:      directory for plots
        device:          torch device
        sample_patients: how many patients to generate individual explanations for
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = next(model.parameters()).device

    print("=== Attention Interpretability Analysis ===")

    # 1. Extract semantic (meta-path) attention
    beta, metapath_names = extract_semantic_attention(model, features, neighbor_dicts, device)
    print(f"Beta shape: {beta.shape}  (patients x meta-paths)")
    print(f"Meta-paths: {metapath_names}")

    # 2. Plot meta-path distributions
    plot_metapath_preference(
        beta, metapath_names,
        save_path=os.path.join(output_dir, 'metapath_heatmap.png')
    )
    plot_metapath_distribution(
        beta, metapath_names,
        save_path=os.path.join(output_dir, 'metapath_distribution.png')
    )

    # 3. Save raw beta values
    np.save(os.path.join(output_dir, 'beta_per_patient.npy'), beta)
    print(f"Saved raw beta to {output_dir}/beta_per_patient.npy")

    # 4. Patient-level explanations (sample)
    N = features.shape[0]
    sample_idxs = np.linspace(0, N-1, min(sample_patients, N), dtype=int)

    explanations_dir = os.path.join(output_dir, 'patient_explanations')
    os.makedirs(explanations_dir, exist_ok=True)

    for pidx in sample_idxs:
        # Extract node-level attention (use first meta-path)
        try:
            neigh_arr, alpha_arr = extract_node_attention_for_patient(
                model, features, neighbor_dicts, int(pidx), metapath_idx=0, device=device
            )
        except Exception:
            neigh_arr, alpha_arr = np.array([]), np.array([])

        # Plot top neighbours
        if len(neigh_arr) > 0:
            pid_str = str(patient_ids[pidx]) if patient_ids is not None else str(pidx)
            plot_top_neighbours(
                pidx, neigh_arr, alpha_arr, patient_ids=patient_ids,
                save_path=os.path.join(output_dir, f'top_neighbours_p{pidx}.png')
            )

        # Text explanation
        lines = generate_explanation(
            pidx, beta, metapath_names,
            neighbours=neigh_arr if len(neigh_arr) > 0 else None,
            alpha_weights=alpha_arr if len(alpha_arr) > 0 else None,
            patient_ids=patient_ids, predictions=predictions, organ_names=organ_names
        )
        pid_str = str(patient_ids[pidx]) if patient_ids is not None else str(pidx)
        exp_file = os.path.join(explanations_dir, f'explanation_p{pidx}.txt')
        with open(exp_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Explanation saved: {exp_file}")

    print(f"\nInterpretability analysis complete. Outputs in: {output_dir}")
    return beta
