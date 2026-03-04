#!/usr/bin/env python3
"""
Disease Clustering Preprocessing Script
========================================
Run this BEFORE training to reduce 44 disease classes to ~6-8 clusters.

Usage:
    python cluster_diseases.py

Outputs:
    - data/patient-one-hot-labeled-disease-clustered.csv (reduced classes)
    - data/disease_cluster_mapping.json (original disease → cluster mapping)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = '/Users/charlie/Documents/Coding/VS Code/Language_python/FYP/New_2026/HAN-implementation/data'
INPUT_FILE = 'patient-one-hot-labeled-disease.csv'
OUTPUT_FILE = 'patient-one-hot-labeled-disease-clustered.csv'
MAPPING_FILE = 'disease_cluster_mapping.json'

# Cluster range to try (find optimal)
MIN_CLUSTERS = 4
MAX_CLUSTERS = 12


def load_data():
    """Load the one-hot encoded disease labels."""
    filepath = os.path.join(DATA_DIR, INPUT_FILE)
    df = pd.read_csv(filepath, encoding='latin1')
    print(f"Loaded {len(df)} patients with {len(df.columns) - 1} disease columns")
    return df


def analyze_class_distribution(df):
    """Analyze and print class distribution."""
    disease_cols = [c for c in df.columns if c != 'patient_id']
    
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Count positive samples per disease
    counts = df[disease_cols].sum().sort_values(ascending=False)
    
    print(f"\nTotal patients: {len(df)}")
    print(f"Total diseases: {len(disease_cols)}")
    print(f"\nTop 10 most common diseases:")
    for disease, count in counts.head(10).items():
        pct = 100 * count / len(df)
        print(f"  {disease}: {count} ({pct:.1f}%)")
    
    print(f"\nBottom 10 least common diseases:")
    for disease, count in counts.tail(10).items():
        pct = 100 * count / len(df)
        print(f"  {disease}: {count} ({pct:.1f}%)")
    
    # Imbalance ratio
    max_count = counts.max()
    min_count = counts[counts > 0].min()
    print(f"\nImbalance ratio (max/min): {max_count/min_count:.1f}x")
    
    return disease_cols, counts


def compute_disease_correlation(df, disease_cols):
    """Compute correlation matrix between diseases based on co-occurrence."""
    disease_matrix = df[disease_cols].values
    
    # Correlation matrix
    corr_matrix = np.corrcoef(disease_matrix.T)
    
    # Handle NaN values (diseases with no positive samples)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    return corr_matrix


def find_optimal_clusters(distance_matrix, min_k=4, max_k=12):
    """Find optimal number of clusters using silhouette score."""
    best_k = min_k
    best_score = -1
    scores = {}
    
    for k in range(min_k, max_k + 1):
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance_matrix)
        
        # Only compute silhouette if we have more than 1 cluster with samples
        if len(np.unique(labels)) > 1:
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
            scores[k] = score
            
            if score > best_score:
                best_score = score
                best_k = k
    
    print(f"\nSilhouette scores by cluster count:")
    for k, score in scores.items():
        marker = " ← best" if k == best_k else ""
        print(f"  k={k}: {score:.3f}{marker}")
    
    return best_k


def cluster_diseases(df, disease_cols, n_clusters=None):
    """Cluster diseases based on co-occurrence patterns."""
    print("\n" + "="*60)
    print("DISEASE CLUSTERING")
    print("="*60)
    
    # Compute correlation → distance
    corr_matrix = compute_disease_correlation(df, disease_cols)
    distance_matrix = 1 - np.abs(corr_matrix)  # Convert correlation to distance
    
    # Find optimal k if not specified
    if n_clusters is None:
        n_clusters = find_optimal_clusters(distance_matrix, MIN_CLUSTERS, MAX_CLUSTERS)
    
    print(f"\nUsing {n_clusters} clusters")
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Create mapping
    disease_to_cluster = {}
    cluster_to_diseases = {i: [] for i in range(n_clusters)}
    
    for disease, cluster_id in zip(disease_cols, cluster_labels):
        disease_to_cluster[disease] = int(cluster_id)
        cluster_to_diseases[cluster_id].append(disease)
    
    # Print cluster contents
    print(f"\nCluster assignments:")
    for cluster_id in range(n_clusters):
        diseases = cluster_to_diseases[cluster_id]
        print(f"\n  Cluster {cluster_id} ({len(diseases)} diseases):")
        for d in diseases:
            print(f"    - {d}")
    
    return disease_to_cluster, cluster_to_diseases, n_clusters


def create_clustered_labels(df, disease_cols, disease_to_cluster, n_clusters):
    """Create new label DataFrame with clustered columns."""
    # Start with patient_id
    new_df = df[['patient_id']].copy()
    
    # Create cluster columns (use OR logic - patient has cluster if ANY disease in cluster is 1)
    for cluster_id in range(n_clusters):
        cluster_diseases = [d for d, c in disease_to_cluster.items() if c == cluster_id]
        cluster_col_name = f"cluster_{cluster_id}"
        
        # OR across all diseases in this cluster
        new_df[cluster_col_name] = df[cluster_diseases].max(axis=1)
    
    # Verify
    print(f"\nClustered label shape: {new_df.shape}")
    print(f"Columns: {list(new_df.columns)}")
    
    return new_df


def save_outputs(clustered_df, disease_to_cluster, cluster_to_diseases):
    """Save the clustered CSV and mapping JSON."""
    # Save CSV
    csv_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    clustered_df.to_csv(csv_path, index=False)
    print(f"\nSaved clustered labels to: {csv_path}")
    
    # Save mapping
    mapping = {
        "disease_to_cluster": disease_to_cluster,
        "cluster_to_diseases": cluster_to_diseases
    }
    json_path = os.path.join(DATA_DIR, MAPPING_FILE)
    with open(json_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved cluster mapping to: {json_path}")


def main():
    print("="*60)
    print("DISEASE CLUSTERING PREPROCESSING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Analyze current distribution
    disease_cols, counts = analyze_class_distribution(df)
    
    # Cluster diseases
    disease_to_cluster, cluster_to_diseases, n_clusters = cluster_diseases(df, disease_cols)
    
    # Create clustered labels
    clustered_df = create_clustered_labels(df, disease_cols, disease_to_cluster, n_clusters)
    
    # Show new class distribution
    print("\n" + "="*60)
    print("NEW CLUSTERED CLASS DISTRIBUTION")
    print("="*60)
    cluster_cols = [c for c in clustered_df.columns if c != 'patient_id']
    for col in cluster_cols:
        count = clustered_df[col].sum()
        pct = 100 * count / len(clustered_df)
        print(f"  {col}: {count} ({pct:.1f}%)")
    
    # Save outputs
    save_outputs(clustered_df, disease_to_cluster, cluster_to_diseases)
    
    print("\n" + "="*60)
    print("DONE! Next steps:")
    print("="*60)
    print("1. Review the cluster mapping in disease_cluster_mapping.json")
    print("2. Optionally rename clusters to meaningful names")
    print(f"3. Update train.py to use '{OUTPUT_FILE}' instead of '{INPUT_FILE}'")
    print("4. Run training with reduced classes")


if __name__ == "__main__":
    main()
