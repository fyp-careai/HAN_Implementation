"""
HAN Model Architectures
Contains HAN++ and HGT-HAN hybrid models for medical predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import NodeLevelAttentionImproved, SemanticAttentionImproved, HGTLayerSingle


class HANPP(nn.Module):
    """
    HAN++ Model (Version B)
    
    Improved Hierarchical Attention Network with:
    - Multi-head node-level attention per meta-path
    - Semantic-level attention across meta-paths
    - Multi-organ severity classification
    - Organ damage score regression
    
    Args:
        in_dim: input feature dimension
        hidden_dim: hidden layer dimension
        out_dim: output embedding dimension
        metapath_names: list of meta-path names to use
        num_heads: number of attention heads
        num_organs: number of organs to predict
        num_severity: number of severity classes
        dropout: dropout rate
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, metapath_names, 
                 num_heads=4, num_organs=25, num_severity=4, dropout=0.3):
        super().__init__()
        self.metapath_names = metapath_names
        
        # Input projection
        self.project = nn.Linear(in_dim, hidden_dim)
        
        # Node-level attention for each meta-path
        self.node_atts = nn.ModuleList([
            NodeLevelAttentionImproved(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout) 
            for _ in metapath_names
        ])
        
        # Semantic-level attention to aggregate meta-paths
        self.semantic_att = SemanticAttentionImproved(hidden_dim, dropout=dropout)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
        # Organ-specific classifiers (one per organ)
        self.organ_classifiers = nn.ModuleList([
            nn.Linear(out_dim, num_severity) for _ in range(num_organs)
        ])
        
        # Organ damage regression head
        self.organ_regression = nn.Linear(out_dim, num_organs)
        
        self.dropout = nn.Dropout(dropout)
    
    def set_vectorized_neighbors(self, neighbor_tensors):
        """
        Pre-set vectorized neighbor tensors for all meta-paths.
        
        Args:
            neighbor_tensors: dict of {metapath_name: (neighbor_idx, neighbor_mask)}
        """
        for i, name in enumerate(self.metapath_names):
            if name in neighbor_tensors:
                idx, mask = neighbor_tensors[name]
                self.node_atts[i].set_neighbors(idx, mask)
    
    def forward(self, patient_feats, patient_neighbor_dicts):
        """
        Forward pass.
        
        Args:
            patient_feats: patient feature tensor [N, in_dim]
            patient_neighbor_dicts: dict of {metapath_name: neighbor_dict}
        
        Returns:
            organ_logits: [N, num_organs, num_severity] classification logits
            organ_scores: [N, num_organs] regression scores
            z: [N, out_dim] final embeddings
            beta: [num_metapaths] attention weights over meta-paths
        """
        # Project to hidden dimension
        h = F.gelu(self.project(patient_feats))
        
        # Apply node-level attention for each meta-path
        Zs = []
        for i, name in enumerate(self.metapath_names):
            neigh = patient_neighbor_dicts[name]
            Z_phi = self.node_atts[i](h, neigh)
            Zs.append(Z_phi)
        
        # Aggregate meta-paths with semantic attention
        Z_final, beta = self.semantic_att(Zs)
        
        # Final output projection
        z = F.gelu(self.out_proj(Z_final))
        
        # Organ-specific predictions
        organ_logits = [clf(self.dropout(z)) for clf in self.organ_classifiers]
        organ_logits = torch.stack(organ_logits, dim=1)  # [N, num_organs, num_severity]
        
        # Organ damage scores
        organ_scores = torch.sigmoid(self.organ_regression(z))  # [N, num_organs]
        
        return organ_logits, organ_scores, z, beta


class HGT_HAN(nn.Module):
    """
    HGT-HAN Hybrid Model (Version C)
    
    Combines HGT-style attention with HAN's hierarchical structure:
    - HGT-style multi-head attention per meta-path
    - Semantic-level attention across meta-paths  
    - Multi-organ severity classification
    - Organ damage score regression
    
    Args:
        in_dim: input feature dimension
        hidden_dim: hidden layer dimension
        out_dim: output embedding dimension
        metapath_names: list of meta-path names to use
        num_heads: number of attention heads
        num_organs: number of organs to predict
        num_severity: number of severity classes
        dropout: dropout rate
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, metapath_names,
                 num_heads=4, num_organs=25, num_severity=4, dropout=0.3):
        super().__init__()
        self.metapath_names = metapath_names
        
        # Input projection
        self.project = nn.Linear(in_dim, hidden_dim)
        
        # HGT-style attention layers for each meta-path
        self.hgt_layers = nn.ModuleList([
            HGTLayerSingle(hidden_dim, hidden_dim, nhead=num_heads, dropout=dropout)
            for _ in metapath_names
        ])
        
        # Semantic-level attention to aggregate meta-paths
        self.semantic_att = SemanticAttentionImproved(hidden_dim, dropout=dropout)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
        # Organ-specific classifiers (one per organ)
        self.organ_classifiers = nn.ModuleList([
            nn.Linear(out_dim, num_severity) for _ in range(num_organs)
        ])
        
        # Organ damage regression head
        self.organ_regression = nn.Linear(out_dim, num_organs)
        
        self.dropout = nn.Dropout(dropout)
    
    def set_vectorized_neighbors(self, neighbor_tensors):
        """
        Pre-set vectorized neighbor tensors for all meta-paths.
        
        Args:
            neighbor_tensors: dict of {metapath_name: (neighbor_idx, neighbor_mask)}
        """
        for i, name in enumerate(self.metapath_names):
            if name in neighbor_tensors:
                idx, mask = neighbor_tensors[name]
                self.hgt_layers[i].set_neighbors(idx, mask)
    
    def forward(self, patient_feats, patient_neighbor_dicts):
        """
        Forward pass.
        
        Args:
            patient_feats: patient feature tensor [N, in_dim]
            patient_neighbor_dicts: dict of {metapath_name: neighbor_dict}
        
        Returns:
            organ_logits: [N, num_organs, num_severity] classification logits
            organ_scores: [N, num_organs] regression scores
            z: [N, out_dim] final embeddings
            beta: [num_metapaths] attention weights over meta-paths
        """
        # Project to hidden dimension
        h = F.gelu(self.project(patient_feats))
        
        # Apply HGT-style attention for each meta-path
        Zs = []
        for i, name in enumerate(self.metapath_names):
            neigh = patient_neighbor_dicts[name]
            Z_phi = self.hgt_layers[i](h, neigh)
            Zs.append(Z_phi)
        
        # Aggregate meta-paths with semantic attention
        Z_final, beta = self.semantic_att(Zs)
        
        # Final output projection
        z = F.gelu(self.out_proj(Z_final))
        
        # Organ-specific predictions
        organ_logits = [clf(self.dropout(z)) for clf in self.organ_classifiers]
        organ_logits = torch.stack(organ_logits, dim=1)  # [N, num_organs, num_severity]
        
        # Organ damage scores
        organ_scores = torch.sigmoid(self.organ_regression(z))  # [N, num_organs]
        
        return organ_logits, organ_scores, z, beta
