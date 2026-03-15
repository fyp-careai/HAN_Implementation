import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

# Define data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'HAN_data')
PATIENT_DATA = os.path.join(DATA_DIR, 'merged_coop_ruhunu_patient_data.csv')
TEST_DATA = os.path.join(DATA_DIR, 'unique_test_data_finalized.csv')
ORGAN_DISEASE = os.path.join(DATA_DIR, 'test_organ_disease.csv')
PATIENT_DISEASE = os.path.join(DATA_DIR, 'patient_disease_ground_truth_long.csv')

def load_data():
    print("Loading datasets...")
    patients_df = pd.read_csv(PATIENT_DATA)
    tests_df = pd.read_csv(TEST_DATA)
    organ_disease_df = pd.read_csv(ORGAN_DISEASE)
    patient_disease_df = pd.read_csv(PATIENT_DISEASE)
    
    # Process unique entities
    unique_patients = patients_df['patient_id'].dropna().unique()
    unique_tests = tests_df['test_name'].dropna().unique()
    
    # Extract organs (some tests map to multiple organs, split by ';')
    all_organs = []
    for organs_str in tests_df['organs'].dropna():
        all_organs.extend([org.strip() for org in organs_str.split(';')])
    unique_organs = list(set(all_organs))
    
    # Extract diseases
    unique_diseases = patient_disease_df['disease_name'].dropna().unique()

    print(f"Entities found: {len(unique_patients)} Patients, {len(unique_tests)} Tests, {len(unique_organs)} Organs, {len(unique_diseases)} Diseases")

    # Mappings from ID/Name to index (0 to N-1)
    patient_to_idx = {pid: i for i, pid in enumerate(unique_patients)}
    test_to_idx = {test: i for i, test in enumerate(unique_tests)}
    organ_to_idx = {org: i for i, org in enumerate(unique_organs)}
    disease_to_idx = {dis: i for i, dis in enumerate(unique_diseases)}
    
    # 1. Patient-Test Edges
    print("Constructing Patient-Test edges...")
    pt_edges = []
    for _, row in patients_df.iterrows():
        p_id = row['patient_id']
        t_name = row['mapped_test_name']
        if pd.notna(p_id) and pd.notna(t_name) and t_name in test_to_idx:
            pt_edges.append((patient_to_idx[p_id], test_to_idx[t_name], row['value']))
    
    # Create Patient-Test Adj Matrix
    PT_matrix = sp.dok_matrix((len(unique_patients), len(unique_tests)), dtype=np.float32)
    for p_idx, t_idx, val in pt_edges:
        PT_matrix[p_idx, t_idx] = 1.0 # Or use 'val' normalized as edge attribute later

    # 2. Test-Organ Edges
    print("Constructing Test-Organ edges...")
    to_edges = []
    for _, row in tests_df.iterrows():
        t_name = row['test_name']
        organs_str = row['organs']
        if pd.notna(t_name) and pd.notna(organs_str) and t_name in test_to_idx:
            for org in organs_str.split(';'):
                org = org.strip()
                if org in organ_to_idx:
                    to_edges.append((test_to_idx[t_name], organ_to_idx[org]))
    
    TO_matrix = sp.dok_matrix((len(unique_tests), len(unique_organs)), dtype=np.float32)
    for t_idx, o_idx in to_edges:
        TO_matrix[t_idx, o_idx] = 1.0

    # 3. Patient-Disease Link Prediction Ground Truth (and graph edges)
    # We want to predict patient-disease connections, so we separate them into train/val/test
    # Or just return them as a list of (patient_idx, disease_idx, score)
    print("Constructing Patient-Disease edges...")
    pd_edges = []
    for _, row in patient_disease_df.iterrows():
        p_id = row['patient_id']
        d_name = row['disease_name']
        score = row['disease_score']
        if pd.notna(p_id) and pd.notna(d_name) and pd.notna(score):
            if p_id in patient_to_idx and d_name in disease_to_idx:
                pd_edges.append((patient_to_idx[p_id], disease_to_idx[d_name], float(score)))

    # Compute Meta-Paths Adjacency (e.g., Patient-Test-Organ-Test-Patient, etc.)
    # For predicting Patient-Disease links, we need patient-patient meta-paths
    # E.g. P-T-P: Patients who took the same tests
    
    PT_csr = PT_matrix.tocsr()
    print("Computing P-T-P meta-path matrix...")
    PT_csr_float = PT_csr.astype(np.float32)
    PTP_matrix = PT_csr_float.dot(PT_csr_float.T)
    PTP_matrix.setdiag(0)
    PTP_matrix.eliminate_zeros()
    
    TO_csr = TO_matrix.tocsr()
    OD_edges = []
    for _, row in organ_disease_df.iterrows():
        t_id = row['test_id'] # actually organ_disease mapping has test_id, organ_id, icd_codes
        # the user's organ_disease_df actually maps test -> organ -> disease?
        # let's look at unique_test_data_finalized and test_organ_disease
        pass
    
    # We will just yield 1-hop graphs for GNN standard MP if metapath is complex
    
    return {
        'num_patients': len(unique_patients),
        'num_tests': len(unique_tests),
        'num_organs': len(unique_organs),
        'num_diseases': len(unique_diseases),
        'patient_to_idx': patient_to_idx,
        'disease_to_idx': disease_to_idx,
        'PT_matrix': PT_csr,
        'TO_matrix': TO_csr,
        'PTP_matrix': PTP_matrix,
        'patient_disease_edges': pd_edges
    }

if __name__ == '__main__':
    data = load_data()
    print("Data processing complete.")
    print(f"PTP Matrix shape: {data['PTP_matrix'].shape}, Non-zeros: {data['PTP_matrix'].nnz}")
    print(f"Total Patient-Disease edges to predict: {len(data['patient_disease_edges'])}")
