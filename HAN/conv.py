"""
HAN Convolution/Attention Modules
Contains attention layers for Hierarchical Attention Networks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeLevelAttentionImproved(nn.Module):
    """
    Improved vectorized node-level attention with multi-head attention.
    Supports both vectorized (pre-set neighbors) and loop-based computation.
    
    Features:
    - Multi-head attention mechanism
    - Residual connections
    - Layer normalization  
    - GELU activation
    - Dropout regularization
    """
    
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        self.head_dim = out_dim // num_heads
        
        # Projection layers
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # Attention parameters (one per head)
        self.a_l = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.a_r = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_l)
        nn.init.xavier_uniform_(self.a_r)
        
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(out_dim)
        
        # Residual projection
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        
        # Pre-set neighbor tensors for vectorized computation
        self.neighbor_idx = None
        self.neighbor_mask = None
    
    def set_neighbors(self, neighbor_idx, neighbor_mask):
        """Pre-set neighbor indices and masks for vectorized computation."""
        self.neighbor_idx = neighbor_idx
        self.neighbor_mask = neighbor_mask
    
    def forward(self, h, neighbor_dict=None):
        """
        Forward pass - uses pre-set neighbors if available, falls back to dict otherwise.
        
        Args:
            h: node features [N, in_dim]
            neighbor_dict: dict mapping node_id to list of neighbor_ids (optional)
        
        Returns:
            Updated node features [N, out_dim]
        """
        N = h.size(0)
        h_proj = self.W(h)  # (N, out_dim)
        
        # Use vectorized path if neighbors are pre-set
        if self.neighbor_idx is not None and self.neighbor_mask is not None:
            return self._forward_vectorized(h, h_proj)
        else:
            # Fallback to original loop-based implementation
            return self._forward_loop(h, h_proj, neighbor_dict)
    
    def _forward_vectorized(self, h, h_proj):
        """Vectorized forward pass using pre-set neighbor tensors."""
        N = h.size(0)
        max_neighbors = self.neighbor_idx.size(1)
        
        # Reshape for multi-head attention: [N, num_heads, head_dim]
        h_heads = h_proj.view(N, self.num_heads, self.head_dim)
        
        # Gather neighbor embeddings: [N, max_neighbors, num_heads, head_dim]
        neigh_h = h_heads[self.neighbor_idx]
        
        # Expand self embeddings: [N, 1, num_heads, head_dim] -> [N, max_neighbors, num_heads, head_dim]
        self_h = h_heads.unsqueeze(1).expand(-1, max_neighbors, -1, -1)
        
        # Compute attention for each head
        # self_h: [N, max_neighbors, num_heads, head_dim]
        # a_l: [num_heads, head_dim]
        el = (self_h * self.a_l.view(1, 1, self.num_heads, self.head_dim)).sum(dim=3)  # [N, max_neighbors, num_heads]
        er = (neigh_h * self.a_r.view(1, 1, self.num_heads, self.head_dim)).sum(dim=3)  # [N, max_neighbors, num_heads]
        
        e = self.leaky(el + er)  # [N, max_neighbors, num_heads]
        
        # Mask invalid neighbors
        mask_expanded = self.neighbor_mask.unsqueeze(2).expand(-1, -1, self.num_heads)  # [N, max_neighbors, num_heads]
        e = e.masked_fill(mask_expanded == 0, -1e9)
        
        # Softmax attention weights
        alpha = F.softmax(e, dim=1)  # [N, max_neighbors, num_heads]
        alpha = self.dropout(alpha)
        
        # Weighted sum: [N, max_neighbors, num_heads, head_dim] * [N, max_neighbors, num_heads, 1]
        alpha_expanded = alpha.unsqueeze(3)  # [N, max_neighbors, num_heads, 1]
        out_heads = (alpha_expanded * neigh_h).sum(dim=1)  # [N, num_heads, head_dim]
        
        # Concatenate heads
        H_cat = out_heads.view(N, self.out_dim)  # [N, out_dim]
        
        # Residual + layernorm
        res = self.res_proj(h) if self.res_proj is not None else h_proj
        out = F.gelu(H_cat + res)
        out = self.layernorm(out)
        return out
    
    def _forward_loop(self, h, h_proj, neighbor_dict):
        """Original loop-based forward pass for backward compatibility."""
        N = h.size(0)
        heads = []
        
        for k in range(self.num_heads):
            hk = h_proj[:, k*self.head_dim:(k+1)*self.head_dim]
            out_k = []
            a_lk = self.a_l[k].unsqueeze(1)
            a_rk = self.a_r[k].unsqueeze(1)
            
            for i in range(N):
                neigh = neighbor_dict.get(i, [])
                if not neigh:
                    out_k.append(hk[i])
                    continue
                
                hi = hk[i].unsqueeze(0).repeat(len(neigh), 1)
                hj = hk[neigh]
                
                el = (hi * a_lk.t()).sum(dim=1)
                er = (hj * a_rk.t()).sum(dim=1)
                e = self.leaky(el + er)
                
                alpha = F.softmax(e, dim=0).unsqueeze(1)
                alpha = self.dropout(alpha)
                z = torch.sum(alpha * hj, dim=0)
                out_k.append(z)
            
            out_k = torch.stack(out_k, dim=0)
            heads.append(out_k)
        
        H_cat = torch.cat(heads, dim=1)
        res = self.res_proj(h) if self.res_proj is not None else h_proj
        out = F.gelu(H_cat + res)
        out = self.layernorm(out)
        return out


class SemanticAttentionImproved(nn.Module):
    """
    Semantic-level attention for aggregating meta-path specific embeddings.
    
    Computes attention weights over different meta-paths and combines them.
    """
    
    def __init__(self, hid, dropout=0.2):
        super().__init__()
        self.W = nn.Linear(hid, hid)
        self.q = nn.Parameter(torch.randn(hid))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Z_list):
        """
        Args:
            Z_list: list of tensors [N, hid], one per meta-path
        
        Returns:
            Z_final: aggregated embeddings [N, hid]
            beta: attention weights over meta-paths
        """
        weights = []
        for Z in Z_list:
            h = torch.tanh(self.W(Z))  # (N, hid)
            # Global pooling to compute scalar weight
            w = torch.mean(h @ self.q)
            weights.append(w)
        
        beta = F.softmax(torch.stack(weights), dim=0)
        Z_final = sum(beta[i] * Z_list[i] for i in range(len(Z_list)))
        Z_final = self.dropout(Z_final)
        
        return Z_final, beta


class HGTLayerSingle(nn.Module):
    """
    HGT-style multi-head attention layer.
    Vectorized implementation for efficient computation.
    
    Based on Heterogeneous Graph Transformer (HGT) architecture.
    """
    
    def __init__(self, in_dim, out_dim, nhead=4, dropout=0.2):
        super().__init__()
        self.nhead = nhead
        self.out_dim = out_dim
        assert out_dim % nhead == 0, "out_dim must be divisible by nhead"
        
        self.head_dim = out_dim // nhead
        
        # Query, Key, Value projections
        self.q_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.k_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.v_lin = nn.Linear(in_dim, out_dim, bias=False)
        
        self.fc = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(out_dim)
        
        # Residual projection
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        
        # Pre-set neighbor tensors for vectorized computation
        self.neighbor_idx = None
        self.neighbor_mask = None
    
    def set_neighbors(self, neighbor_idx, neighbor_mask):
        """Pre-set neighbor indices and masks for vectorized computation."""
        self.neighbor_idx = neighbor_idx
        self.neighbor_mask = neighbor_mask
    
    def forward(self, h, neighbor_dict=None):
        """
        Forward pass - uses pre-set neighbors if available, falls back to dict otherwise.
        
        Args:
            h: node features [N, in_dim]
            neighbor_dict: dict mapping node_id to list of neighbor_ids (optional)
        
        Returns:
            Updated node features [N, out_dim]
        """
        N = h.size(0)
        Q = self.q_lin(h)  # (N, out_dim)
        K = self.k_lin(h)
        V = self.v_lin(h)
        
        # Use vectorized path if neighbors are pre-set
        if self.neighbor_idx is not None and self.neighbor_mask is not None:
            return self._forward_vectorized(h, Q, K, V)
        else:
            return self._forward_loop(h, Q, K, V, neighbor_dict)
    
    def _forward_vectorized(self, h, Q, K, V):
        """Vectorized forward pass using pre-set neighbor tensors."""
        N = h.size(0)
        max_neighbors = self.neighbor_idx.size(1)
        
        # Reshape to multi-head: [N, nhead, head_dim]
        Q_heads = Q.view(N, self.nhead, self.head_dim)
        K_heads = K.view(N, self.nhead, self.head_dim)
        V_heads = V.view(N, self.nhead, self.head_dim)
        
        # Gather neighbor K and V: [N, max_neighbors, nhead, head_dim]
        K_neigh = K_heads[self.neighbor_idx]
        V_neigh = V_heads[self.neighbor_idx]
        
        # Expand Q for broadcasting: [N, 1, nhead, head_dim]
        Q_expanded = Q_heads.unsqueeze(1)
        
        # Compute attention scores: [N, max_neighbors, nhead]
        scores = (Q_expanded * K_neigh).sum(dim=3) / math.sqrt(self.head_dim)
        
        # Mask invalid neighbors
        mask_expanded = self.neighbor_mask.unsqueeze(2).expand(-1, -1, self.nhead)
        scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        # Softmax attention weights: [N, max_neighbors, nhead]
        alpha = F.softmax(scores, dim=1)
        
        # Weighted sum of values: [N, nhead, head_dim]
        alpha_expanded = alpha.unsqueeze(3)  # [N, max_neighbors, nhead, 1]
        out_heads = (alpha_expanded * V_neigh).sum(dim=1)  # [N, nhead, head_dim]
        
        # Concatenate heads: [N, out_dim]
        heads_out = out_heads.contiguous().view(N, self.out_dim)
        
        # Residual connection
        res = self.res_proj(h) if self.res_proj is not None else Q
        out = F.gelu(self.fc(self.dropout(heads_out)) + res)
        out = self.layernorm(out)
        return out
    
    def _forward_loop(self, h, Q, K, V, neighbor_dict):
        """Original loop-based forward pass for backward compatibility."""
        N = h.size(0)
        heads_out = []
        
        for i in range(N):
            neigh = neighbor_dict.get(i, [])
            if not neigh:
                heads_out.append(Q[i])
                continue
            
            qi = Q[i].view(self.nhead, self.head_dim)
            kj = K[neigh].view(-1, self.nhead, self.head_dim)
            vj = V[neigh].view(-1, self.nhead, self.head_dim)
            
            scores = torch.einsum('hd,nhd->hn', qi, kj) / math.sqrt(self.head_dim)
            alpha = F.softmax(scores, dim=1)
            weighted = torch.einsum('hn,nhd->hd', alpha, vj)
            weighted_cat = weighted.contiguous().view(-1)
            out_i = weighted_cat
            heads_out.append(out_i)
        
        heads_out = torch.stack(heads_out, dim=0)
        res = self.res_proj(h) if self.res_proj is not None else Q
        out = F.gelu(self.fc(self.dropout(heads_out)) + res)
        out = self.layernorm(out)
        return out
