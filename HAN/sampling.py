"""
HAN Sampling Module
Implements mini-batch and neighborhood sampling for efficient training.
Similar to pyHGT's subgraph sampling approach.
"""

import random
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import torch


class SubgraphSampler:
    """
    Mini-batch sampler with neighborhood sampling for HAN.
    
    Similar to HGT's subgraph sampling, this samples a batch of patients
    and their k-hop neighborhoods for efficient training.
    """
    
    def __init__(self, 
                 num_patients,
                 metapath_neighbors,
                 batch_size=128,
                 num_neighbors_sample=50,
                 seed=42):
        """
        Args:
            num_patients: total number of patients
            metapath_neighbors: dict of {metapath_name: neighbor_dict}
            batch_size: number of patients per batch
            num_neighbors_sample: max neighbors to sample per patient per meta-path
            seed: random seed
        """
        self.num_patients = num_patients
        self.metapath_neighbors = metapath_neighbors
        self.batch_size = batch_size
        self.num_neighbors_sample = num_neighbors_sample
        self.seed = seed
        
        self.indices = list(range(num_patients))
        random.seed(seed)
    
    def shuffle(self):
        """Shuffle patient indices for new epoch."""
        random.shuffle(self.indices)
    
    def __iter__(self):
        """Iterate over batches."""
        self.shuffle()
        for start_idx in range(0, self.num_patients, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_patients)
            batch_patients = self.indices[start_idx:end_idx]
            yield self._sample_subgraph(batch_patients)
    
    def __len__(self):
        """Number of batches."""
        return (self.num_patients + self.batch_size - 1) // self.batch_size
    
    def _sample_subgraph(self, batch_patients):
        """
        Sample subgraph for a batch of patients.
        
        Args:
            batch_patients: list of patient indices in this batch
        
        Returns:
            dict with:
                - patient_idx: original patient indices
                - local_idx: mapping from original to batch-local indices
                - sampled_neighbors: subsampled neighbor dicts per meta-path
                - batch_size: size of this batch
        """
        batch_set = set(batch_patients)
        
        # For each meta-path, sample neighbors
        sampled_neighbors = {}
        all_nodes = set(batch_patients)
        
        for mp_name, neighbor_dict in self.metapath_neighbors.items():
            sampled = {}
            
            for patient in batch_patients:
                neighbors = neighbor_dict.get(patient, [])
                
                # Sample neighbors
                if len(neighbors) > self.num_neighbors_sample:
                    neighbors = random.sample(neighbors, self.num_neighbors_sample)
                
                sampled[patient] = neighbors
                all_nodes.update(neighbors)
            
            sampled_neighbors[mp_name] = sampled
        
        # Map all nodes to local indices
        all_nodes_list = sorted(list(all_nodes))
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(all_nodes_list)}
        
        # Remap neighbor dicts to local indices
        local_neighbors = {}
        for mp_name, sampled in sampled_neighbors.items():
            local_sampled = {}
            for patient, neighbors in sampled.items():
                if patient in global_to_local:
                    local_patient = global_to_local[patient]
                    local_neighs = [global_to_local[n] for n in neighbors if n in global_to_local]
                    local_sampled[local_patient] = local_neighs
            local_neighbors[mp_name] = local_sampled
        
        return {
            'patient_idx': batch_patients,
            'all_nodes': all_nodes_list,
            'global_to_local': global_to_local,
            'local_to_global': {v: k for k, v in global_to_local.items()},
            'sampled_neighbors': local_neighbors,
            'batch_size': len(batch_patients)
        }


def create_batch_tensors(subgraph, full_features, full_labels, device='cpu'):
    """
    Create batch tensors from sampled subgraph.
    
    Args:
        subgraph: output from SubgraphSampler._sample_subgraph()
        full_features: [N, D] full feature tensor
        full_labels: [N, ...] full label tensor
        device: torch device
    
    Returns:
        dict with batch_features, batch_labels, and mapping info
    """
    all_nodes = subgraph['all_nodes']
    patient_idx = subgraph['patient_idx']
    global_to_local = subgraph['global_to_local']
    
    # Extract features and labels for all nodes in subgraph
    batch_features = full_features[all_nodes].to(device)
    batch_labels = full_labels[all_nodes].to(device)
    
    # Get local indices of the batch patients (for loss computation)
    batch_patient_local = [global_to_local[p] for p in patient_idx]
    
    return {
        'features': batch_features,
        'labels': batch_labels,
        'batch_patient_local': batch_patient_local,
        'all_nodes': all_nodes,
        'patient_idx': patient_idx,
        'global_to_local': global_to_local
    }


class NeighborSampler:
    """
    Efficient neighbor sampler for meta-path based attention.
    Samples fixed number of neighbors per node to enable batching.
    """
    
    def __init__(self, neighbor_dict, num_samples=50, seed=42):
        """
        Args:
            neighbor_dict: {node_id: [neighbor_ids]}
            num_samples: number of neighbors to sample
            seed: random seed
        """
        self.neighbor_dict = neighbor_dict
        self.num_samples = num_samples
        self.seed = seed
        self._rng = random.Random(seed)
    
    def sample(self, node_ids):
        """
        Sample neighbors for a list of nodes.
        
        Args:
            node_ids: list of node indices
        
        Returns:
            dict: {node_id: [sampled_neighbor_ids]}
        """
        sampled = {}
        for node in node_ids:
            neighbors = self.neighbor_dict.get(node, [node])  # self-loop if no neighbors
            
            if len(neighbors) > self.num_samples:
                neighbors = self._rng.sample(neighbors, self.num_samples)
            
            sampled[node] = neighbors
        
        return sampled


def adaptive_batch_size(num_patients, available_memory_gb=8, feature_dim=128):
    """
    Compute adaptive batch size based on available memory.
    
    Args:
        num_patients: total number of patients
        available_memory_gb: available GPU/CPU memory in GB
        feature_dim: feature dimension
    
    Returns:
        recommended batch size
    """
    # Rough estimate: each patient with neighbors needs ~feature_dim * 4 bytes * 100 neighbors
    bytes_per_patient = feature_dim * 4 * 100  # 4 bytes per float32, ~100 neighbors
    available_bytes = available_memory_gb * 1024**3 * 0.5  # Use 50% of available memory
    
    batch_size = int(available_bytes / bytes_per_patient)
    batch_size = max(16, min(batch_size, 512))  # Clamp between 16 and 512
    
    return batch_size


def create_sparse_batches(adjacency_matrix, batch_size=128):
    """
    Create sparse mini-batches from adjacency matrix.
    
    Args:
        adjacency_matrix: scipy.sparse matrix [N, M]
        batch_size: batch size
    
    Returns:
        list of sparse matrix batches
    """
    n_rows = adjacency_matrix.shape[0]
    batches = []
    
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        batch = adjacency_matrix[start:end]
        batches.append(batch)
    
    return batches
