import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.utils import dropout_edge
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score

from dataset_pyg import load_pyg_data
from models.han_pyg import HeteroHANModel
import torch_geometric.transforms as T

def train_link_prediction():
    # 1. Load Data
    data, patient_to_idx, test_to_idx, organ_to_idx, disease_to_idx, norm_stats = load_pyg_data()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split Patient-Disease edges into Train / Val / Test masks
    # data['patient', 'has', 'disease'].edge_index
    # data['patient', 'has', 'disease'].edge_label
    
    # We can use PyG's RandomLinkSplit for heterogeneous graphs!
    # However we will manually split for transparency.
    edge_index = data['patient', 'has', 'disease'].edge_index
    edge_label = data['patient', 'has', 'disease'].edge_label
    num_edges = edge_index.size(1)
    
    indices = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    val_size = int(0.1 * num_edges)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    
    edge_label_index_train = edge_index[:, train_idx]
    edge_label_train = edge_label[train_idx].squeeze(-1)
    
    edge_label_index_val = edge_index[:, val_idx]
    edge_label_val = edge_label[val_idx].squeeze(-1)
    
    edge_label_index_test = edge_index[:, test_idx]
    edge_label_test = edge_label[test_idx].squeeze(-1)
    
    # Move to device
    data = data.to(device)
    edge_label_index_train = edge_label_index_train.to(device)
    edge_label_train = edge_label_train.to(device)
    edge_label_index_val = edge_label_index_val.to(device)
    edge_label_val = edge_label_val.to(device)
    edge_label_index_test = edge_label_index_test.to(device)
    edge_label_test = edge_label_test.to(device)

    # Dictionary of attributes for Edge encoder
    edge_attr_dict = {
        ('patient', 'takes', 'labtest'): data['patient', 'takes', 'labtest'].edge_attr,
        ('labtest', 'rev_takes', 'patient'): data['labtest', 'rev_takes', 'patient'].edge_attr
    }

    # Remove the supervision edge from message passing edge_index_dict
    mp_edge_index_dict = {}
    for edge_type, e_idx in data.edge_index_dict.items():
        if edge_type[1] not in ['has', 'rev_has']:
            mp_edge_index_dict[edge_type] = e_idx

    # 2. Build Model
    num_nodes_dict = {node_type: data[node_type].num_nodes for node_type in data.node_types}
    
    model = HeteroHANModel(
        metadata=data.metadata(),
        num_nodes_dict=num_nodes_dict,
        hidden_channels=64,
        edge_dim=32,
        out_channels=1,
        num_heads=4,
        num_layers=2
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    print("Starting Training...")
    best_val_loss = float('inf')

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        h_dict = model(data.x_dict, mp_edge_index_dict, edge_attr_dict)
        
        # Predict links for training edges
        preds = model.predict_link(h_dict['patient'], h_dict['disease'], edge_label_index_train)
        
        loss = loss_fn(preds, edge_label_train)
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                h_dict_val = model(data.x_dict, mp_edge_index_dict, edge_attr_dict)
                val_preds = model.predict_link(h_dict_val['patient'], h_dict_val['disease'], edge_label_index_val)
                val_loss = loss_fn(val_preds, edge_label_val)
                
                # Check for Best Model
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(model.state_dict(), 'han_link_pred_best.pt')

            print(f"Epoch {epoch:03d} | Train MSE: {loss.item():.4f} | Val MSE: {val_loss.item():.4f}")

    # 3. Final Test Evaluation
    print("Loading best model for testing...")
    model.load_state_dict(torch.load('han_link_pred_best.pt'))
    model.eval()
    with torch.no_grad():
        h_dict_test = model(data.x_dict, mp_edge_index_dict, edge_attr_dict)
        test_preds = model.predict_link(h_dict_test['patient'], h_dict_test['disease'], edge_label_index_test)
        test_loss = loss_fn(test_preds, edge_label_test)
        mae = mean_absolute_error(edge_label_test.cpu().numpy(), test_preds.cpu().numpy())

        # Binarize labels using mean train score as threshold for classification metrics
        threshold = edge_label_train.mean().item()
        test_labels_binary = (edge_label_test > threshold).cpu().numpy().astype(int)
        test_preds_binary = (test_preds > threshold).cpu().numpy().astype(int)

        acc = accuracy_score(test_labels_binary, test_preds_binary)
        f1 = f1_score(test_labels_binary, test_preds_binary, average='macro')
    
    print(f"Final Test MSE: {test_loss.item():.4f} | Test MAE: {mae:.4f} | Test Accuracy: {acc:.4f} | Test F1: {f1:.4f}")

if __name__ == '__main__':
    train_link_prediction()
