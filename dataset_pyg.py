import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'HAN_data')
PATIENT_DATA = os.path.join(DATA_DIR, 'merged_coop_ruhunu_patient_data.csv')
TEST_DATA = os.path.join(DATA_DIR, 'unique_test_data_finalized.csv')
TEST_DISEASE_MAP = os.path.join(DATA_DIR, 'test_disease_map.csv')
PATIENT_DISEASE = os.path.join(DATA_DIR, 'patient_disease_ground_truth_long.csv')

def load_pyg_data():
    print("Loading datasets...")
    patients_df = pd.read_csv(PATIENT_DATA)
    tests_df = pd.read_csv(TEST_DATA)
    test_disease_df = pd.read_csv(TEST_DISEASE_MAP)
    patient_disease_df = pd.read_csv(PATIENT_DISEASE)

    # Process unique entities
    unique_patients = patients_df['patient_id'].dropna().unique()
    unique_tests = tests_df['test_name'].dropna().unique()

    # Extract organs
    all_organs = []
    test_to_organs = {}
    
    # Vectorized / faster tests_df processing
    for t_name, organs_str in zip(tests_df['test_name'], tests_df['organs']):
        if pd.notna(t_name) and pd.notna(organs_str):
            orgs = [org.strip() for org in organs_str.split(';')]
            all_organs.extend(orgs)
            test_to_organs[t_name] = orgs
            
    unique_organs = list(set(all_organs))
    unique_diseases = patient_disease_df['disease_name'].dropna().unique()

    print(f"Entities found: {len(unique_patients)} Patients, {len(unique_tests)} Tests, {len(unique_organs)} Organs, {len(unique_diseases)} Diseases")

    patient_to_idx = {pid: i for i, pid in enumerate(unique_patients)}
    test_to_idx = {test: i for i, test in enumerate(unique_tests)}
    organ_to_idx = {org: i for i, org in enumerate(unique_organs)}
    disease_to_idx = {dis: i for i, dis in enumerate(unique_diseases)}

    data = HeteroData()

    # 1. Add Nodes
    data['patient'].num_nodes = len(unique_patients)
    data['labtest'].num_nodes = len(unique_tests)
    data['organ'].num_nodes = len(unique_organs)
    data['disease'].num_nodes = len(unique_diseases)

    # Extract Patient Explicit Features (Age, Sex)
    pt_feats_df = patients_df.drop_duplicates(subset=['patient_id']).copy()
    gender_map = {'Male': 0.0, 'Female': 1.0, 'M': 0.0, 'F': 1.0}
    pt_feats_df['gender_num'] = pt_feats_df['sex'].map(gender_map).fillna(0.5)
    
    pt_feats_df['age'] = pd.to_numeric(pt_feats_df['age'], errors='coerce')
    age_median = pt_feats_df['age'].median()
    pt_feats_df['age'] = pt_feats_df['age'].fillna(age_median)
    
    pt_feats_df.set_index('patient_id', inplace=True)
    aligned_pt_feats = pt_feats_df.reindex(unique_patients)
    
    age_mean = aligned_pt_feats['age'].mean()
    age_std = aligned_pt_feats['age'].std() + 1e-8
    aligned_pt_feats['age_norm'] = (aligned_pt_feats['age'] - age_mean) / age_std
    
    # Extract Labtest Explicit Features (Min/Max Ref)
    test_feats_df = tests_df.drop_duplicates(subset=['test_name']).copy()
    test_feats_df['min_ref'] = pd.to_numeric(test_feats_df['lower_bound'], errors='coerce').fillna(0.0)
    test_feats_df['max_ref'] = pd.to_numeric(test_feats_df['upper_bound'], errors='coerce').fillna(0.0)
    
    test_feats_df.set_index('test_name', inplace=True)
    aligned_test_feats = test_feats_df.reindex(unique_tests)
    
    min_mean = aligned_test_feats['min_ref'].mean()
    min_std = aligned_test_feats['min_ref'].std() + 1e-8
    max_mean = aligned_test_feats['max_ref'].mean()
    max_std = aligned_test_feats['max_ref'].std() + 1e-8
    
    aligned_test_feats['min_norm'] = (aligned_test_feats['min_ref'] - min_mean) / min_std
    aligned_test_feats['max_norm'] = (aligned_test_feats['max_ref'] - max_mean) / max_std

    # Store explicit features
    data['patient'].x_feat = torch.tensor(aligned_pt_feats[['age_norm', 'gender_num']].values, dtype=torch.float)
    data['labtest'].x_feat = torch.tensor(aligned_test_feats[['min_norm', 'max_norm']].values, dtype=torch.float)

    # Store node indices to be used for PyTorch nn.Embedding lookup
    data['patient'].x = torch.arange(len(unique_patients), dtype=torch.long)
    data['labtest'].x = torch.arange(len(unique_tests), dtype=torch.long)
    data['organ'].x = torch.arange(len(unique_organs), dtype=torch.long)
    data['disease'].x = torch.arange(len(unique_diseases), dtype=torch.long)

    # 2. Patient-Labtest edges
    print("Constructing Patient-Labtest edges sequentially via vectors...")
    
    # Filter valid rows first
    valid_mask = patients_df['patient_id'].notna() & patients_df['mapped_test_name'].notna()
    patients_df_valid = patients_df[valid_mask]
    
    # Map to indices
    p_ids = patients_df_valid['patient_id'].map(patient_to_idx).values
    t_ids = patients_df_valid['mapped_test_name'].map(test_to_idx).values
    
    # Keep only mappings that were found
    valid_mapped = ~np.isnan(p_ids) & ~np.isnan(t_ids)
    p_indices = p_ids[valid_mapped].astype(np.int64)
    t_indices = t_ids[valid_mapped].astype(np.int64)
    
    # Edge Attributes
    vals = patients_df_valid['value'].values[valid_mapped]
    
    # Handle extreme outliers (e.g., -1e9 placeholders)
    vals = np.where(vals < -10000, np.nan, vals)
    vals = np.nan_to_num(vals, nan=np.nanmedian(vals))
    
    # Z-Score Normalization for values
    val_mean = np.mean(vals)
    val_std = np.std(vals) + 1e-8
    vals = (vals - val_mean) / val_std
    
    # Determine which time column to use
    t_col = np.zeros(len(p_indices))
    if 'time_period' in patients_df_valid.columns:
        t_col = patients_df_valid['time_period'].values[valid_mapped]
    elif 'time_since_test' in patients_df_valid.columns:
        t_col = patients_df_valid['time_since_test'].values[valid_mapped]
        
    # Process time values
    t_col = np.where(t_col < -10000, np.nan, t_col)
    t_col = np.nan_to_num(t_col, nan=np.nanmedian(t_col))
    time_mean = np.mean(t_col)
    time_std = np.std(t_col) + 1e-8
    time_vals = (t_col - time_mean) / time_std
        
    pt_src = torch.tensor(p_indices, dtype=torch.long)
    pt_dst = torch.tensor(t_indices, dtype=torch.long)
    pt_attr = torch.tensor(np.column_stack((vals, time_vals)), dtype=torch.float)

    data['patient', 'takes', 'labtest'].edge_index = torch.stack([pt_src, pt_dst])
    data['patient', 'takes', 'labtest'].edge_attr = pt_attr
    data['labtest', 'rev_takes', 'patient'].edge_index = torch.stack([pt_dst, pt_src])
    data['labtest', 'rev_takes', 'patient'].edge_attr = pt_attr

    # 3. Labtest-Organ edges
    print("Constructing Labtest-Organ edges...")
    to_src = []
    to_dst = []
    for t_name, orgs in test_to_organs.items():
        if t_name in test_to_idx:
            for org in orgs:
                if org in organ_to_idx:
                    to_src.append(test_to_idx[t_name])
                    to_dst.append(organ_to_idx[org])
                    
    data['labtest', 'affects', 'organ'].edge_index = torch.tensor([to_src, to_dst], dtype=torch.long)
    data['organ', 'rev_affects', 'labtest'].edge_index = torch.tensor([to_dst, to_src], dtype=torch.long)

    # 4. Organ-Disease edges
    print("Constructing Organ-Disease edges...")
    od_edges_set = set()
    for t_name, d_name in zip(test_disease_df['test_name'], test_disease_df['disease']):
        if pd.notna(t_name) and pd.notna(d_name) and d_name in disease_to_idx:
            if t_name in test_to_organs:
                for org in test_to_organs[t_name]:
                    if org in organ_to_idx:
                        od_edges_set.add((organ_to_idx[org], disease_to_idx[d_name]))

    if len(od_edges_set) > 0:
        od_src, od_dst = zip(*list(od_edges_set))
    else:
        od_src, od_dst = [], []
        
    data['organ', 'causes', 'disease'].edge_index = torch.tensor([od_src, od_dst], dtype=torch.long)
    data['disease', 'rev_causes', 'organ'].edge_index = torch.tensor([od_dst, od_src], dtype=torch.long)

    # 5. Patient-Disease edges (Ground Truth for Link Prediction)
    print("Constructing Patient-Disease edges...")
    valid_pd_mask = patient_disease_df['patient_id'].notna() & patient_disease_df['disease_name'].notna() & patient_disease_df['disease_score'].notna()
    pd_df_valid = patient_disease_df[valid_pd_mask]
    
    p_ids_pd = pd_df_valid['patient_id'].map(patient_to_idx).values
    d_ids_pd = pd_df_valid['disease_name'].map(disease_to_idx).values
    scores_pd = pd_df_valid['disease_score'].values
    
    valid_mapped_pd = ~np.isnan(p_ids_pd) & ~np.isnan(d_ids_pd)
    p_indices_pd = p_ids_pd[valid_mapped_pd].astype(np.int64)
    d_indices_pd = d_ids_pd[valid_mapped_pd].astype(np.int64)
    scores_pd = scores_pd[valid_mapped_pd].astype(np.float32)

    pd_src = torch.tensor(p_indices_pd, dtype=torch.long)
    pd_dst = torch.tensor(d_indices_pd, dtype=torch.long)
    pd_attr = torch.tensor(scores_pd, dtype=torch.float).unsqueeze(1)

    data['patient', 'has', 'disease'].edge_index = torch.stack([pd_src, pd_dst])
    data['patient', 'has', 'disease'].edge_label = pd_attr

    norm_stats = {
        'val_mean': val_mean,
        'val_std': val_std,
        'time_mean': time_mean,
        'time_std': time_std,
        'age_mean': age_mean,
        'age_std': age_std,
        'min_mean': min_mean,
        'min_std': min_std,
        'max_mean': max_mean,
        'max_std': max_std
    }

    return data, patient_to_idx, test_to_idx, organ_to_idx, disease_to_idx, norm_stats

if __name__ == '__main__':
    data, *_, norm_stats = load_pyg_data()
    print("Dataset built complete:")
    print(data)
