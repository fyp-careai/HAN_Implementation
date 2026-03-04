"""
Example: Quick Start with HAN Package
Demonstrates basic usage of the HAN package for medical prediction.
"""

import torch
from HAN import (
    MedicalGraphData,
    HANPP,
    HGT_HAN,
    compute_loss_multiorg,
    evaluate_multiorg,
    neighbors_to_padded_tensors
)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_RECORDS = "data/patient_reports.csv"
PATH_SYMPTOM = "data/test_organ.csv"

# ============== Step 1: Load Data ==============
print("Step 1: Loading data...")

data_loader = MedicalGraphData(
    path_records=PATH_RECORDS,
    path_symptom=PATH_SYMPTOM,
    symptom_freq_threshold=0.08,
    prune_per_patient=300
)

data_loader.load_data()
data_loader.build_labels_and_features()
data_loader.build_adjacency_matrices()

print(f"Loaded {data_loader.P} patients, {data_loader.S} symptoms, "
      f"{data_loader.O} organs, {data_loader.D} diseases")

# ============== Step 2: Build Meta-paths ==============
print("\nStep 2: Building meta-paths...")

metapath_names = ["P-O-P", "P-D-P"]
patient_neighbors = data_loader.build_metapaths(metapath_names)

print(f"Built {len(patient_neighbors)} meta-paths: {list(patient_neighbors.keys())}")

# ============== Step 3: Prepare Tensors ==============
print("\nStep 3: Preparing tensors...")

tensors = data_loader.get_tensors(device=DEVICE)
patient_feats = tensors['patient_feats']
labels_organ_severity = tensors['labels_organ_severity']

organ_class_weights = data_loader.compute_class_weights(device=DEVICE)

print(f"Patient features: {patient_feats.shape}")
print(f"Organ labels: {labels_organ_severity.shape}")

# ============== Step 4: Create Model ==============
print("\nStep 4: Creating model...")

model = HANPP(
    in_dim=patient_feats.shape[1],
    hidden_dim=128,
    out_dim=128,
    metapath_names=metapath_names,
    num_heads=4,
    num_organs=data_loader.O,
    num_severity=4,
    dropout=0.3
).to(DEVICE)

# Pre-compute vectorized neighbors for faster training
neighbor_tensors = {}
for name, neigh_dict in patient_neighbors.items():
    idx, mask = neighbors_to_padded_tensors(
        neigh_dict, data_loader.P, max_neighbors=300, device=DEVICE
    )
    neighbor_tensors[name] = (idx, mask)

model.set_vectorized_neighbors(neighbor_tensors)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# ============== Step 5: Forward Pass Example ==============
print("\nStep 5: Running forward pass...")

model.eval()
with torch.no_grad():
    organ_logits, organ_scores, embeddings, meta_attention = model(
        patient_feats, patient_neighbors
    )

print(f"Organ logits: {organ_logits.shape}")
print(f"Organ scores: {organ_scores.shape}")
print(f"Patient embeddings: {embeddings.shape}")
print(f"Meta-path attention weights: {meta_attention}")

# ============== Step 6: Training Loop (Simplified) ==============
print("\nStep 6: Example training loop...")

# Split data (simple random split for demo)
import numpy as np
indices = list(range(data_loader.P))
np.random.shuffle(indices)
train_size = int(0.8 * data_loader.P)
train_idx = set(indices[:train_size])
val_idx = set(indices[train_size:])

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Train for a few epochs
model.train()
for epoch in range(1, 4):
    optimizer.zero_grad()
    
    organ_logits, organ_scores, z, beta = model(patient_feats, patient_neighbors)
    loss = compute_loss_multiorg(
        organ_logits, labels_organ_severity, train_idx, organ_class_weights
    )
    
    loss.backward()
    optimizer.step()
    
    # Evaluate
    metrics = evaluate_multiorg(
        model, patient_feats, labels_organ_severity, patient_neighbors, val_idx
    )
    
    print(f"Epoch {epoch}: loss={loss.item():.4f}, "
          f"val_f1={metrics['mean_organ_f1']:.4f}, "
          f"micro_f1={metrics['micro_f1']:.4f}")

print("\n" + "="*60)
print("Example complete! See train.py for full training pipeline.")
print("="*60)
