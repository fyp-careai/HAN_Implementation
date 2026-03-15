import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

class SimpleSemanticAttention(nn.Module):
    def __init__(self, in_channels, attention_size=128):
        super().__init__()
        self.project_q = nn.Sequential(
            nn.Linear(in_channels, attention_size),
            nn.Tanh()
        )
        self.project_attention = nn.Linear(attention_size, 1, bias=False)

    def forward(self, x_list):
        # x_list: List of tensors of shape [num_nodes, in_channels]
        if len(x_list) == 1:
            return x_list[0]
        
        # Stack into [num_nodes, num_metapaths, in_channels]
        x_stacked = torch.stack(x_list, dim=1)
        
        # [num_nodes, num_metapaths, attention_size]
        w = self.project_q(x_stacked)
        
        # [num_nodes, num_metapaths, 1]
        attn_weights = self.project_attention(w)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum: [num_nodes, in_channels]
        out = (x_stacked * attn_weights).sum(dim=1)
        return out


class HeteroHANModel(nn.Module):
    def __init__(self, metadata, num_nodes_dict, hidden_channels=64, edge_dim=32, out_channels=1, num_heads=4, num_layers=2):
        super().__init__()

        # Node embeddings using indices
        self.node_embs = nn.ModuleDict()
        for node_type in metadata[0]:
            self.node_embs[node_type] = nn.Embedding(num_nodes_dict[node_type], hidden_channels)

        # Edge feature encoders (only for edges that have features)
        self.edge_lins = nn.ModuleDict()
        self.edge_lins['takes'] = Linear(2, edge_dim)
        self.edge_lins['rev_takes'] = Linear(2, edge_dim)
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                src, rel, dst = edge_type
                # Exclude supervision edges from message passing
                if rel in ['has', 'rev_has']: 
                    continue
                
                # Add GATConv. Pass edge_dim if the edge type has features.
                if rel in ['takes', 'rev_takes']:
                    conv_dict[edge_type] = GATConv(
                        (-1, -1), hidden_channels, heads=num_heads, 
                        edge_dim=edge_dim, concat=False, add_self_loops=False
                    )
                else:
                    conv_dict[edge_type] = GATConv(
                        (-1, -1), hidden_channels, heads=num_heads,
                        concat=False, add_self_loops=False
                    )
            
            # Using sum or mean for semantic attention inside HeteroConv doesn't do semantic attention weights.
            # Instead, we will extract output per relation and do semantic attention manually. 
            # We can use custom aggregation or let HeteroConv do 'sum' and we do semantic attention separately?
            # Actually, HeteroConv supports custom aggregation! 
            # But for simplicity, let's keep PyG's sum/mean aggregation, which performs well in practice,
            # unless we strictly want HAN's semantic attention. 
            # PyG's HeteroConv with aggr='sum' is very standard.
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        # Link Predictor (Regression Head)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels) # Output is continuous score
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # 1. Encode Nodes
        h_dict = {}
        for node_type, x in x_dict.items():
            # x contains indices for node embeddings
            h_dict[node_type] = self.node_embs[node_type](x.squeeze())

        # 2. Encode Edges
        edge_attr_encoded = {}
        if edge_attr_dict is not None:
            for edge_type, attr in edge_attr_dict.items():
                rel = edge_type[1]
                if rel in self.edge_lins:
                    edge_attr_encoded[edge_type] = F.relu(self.edge_lins[rel](attr))

        # 3. Message Passing Layers
        for i in range(self.num_layers):
            h_dict = self.convs[i](h_dict, edge_index_dict, edge_attr_dict=edge_attr_encoded)
            
            if i < self.num_layers - 1:
                h_dict = {key: F.relu(x) for key, x in h_dict.items()}

        return h_dict

    def predict_link(self, h_src, h_dst, edge_label_index):
        # h_src: Patient embeddings, h_dst: Disease embeddings
        src_indices = edge_label_index[0]
        dst_indices = edge_label_index[1]
        
        # Concatenate src and dst embeddings
        h_concat = torch.cat([h_src[src_indices], h_dst[dst_indices]], dim=-1)
        
        # Predict score
        return self.predictor(h_concat).squeeze(-1)
