"""
HAN Data Loading and Preprocessing
Handles data loading, graph construction, and feature engineering for HAN models.
"""

import time
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from .utils import try_float, parse_normal_range, csr_to_neighbors_prune


class MedicalGraphData:
    """
    Medical heterogeneous graph data loader and processor.
    
    Constructs a heterogeneous graph with:
    - Patient nodes
    - Symptom/Test nodes
    - Organ nodes
    - Disease nodes
    
    And various meta-paths connecting them.
    """
    
    def __init__(self, 
                 path_records,
                 path_symptom,
                 symptom_freq_threshold=0.08,
                 prune_per_patient=300,
                 nnz_threshold=80_000_000,
                 seed=42):
        """
        Args:
            path_records: path to patient records CSV
            path_symptom: path to symptom metadata CSV
            symptom_freq_threshold: filter symptoms present in >X% of patients
            prune_per_patient: max neighbors per patient in meta-paths
            nnz_threshold: skip meta-paths with >X non-zero entries
            seed: random seed
        """
        self.path_records = path_records
        self.path_symptom = path_symptom
        self.symptom_freq_threshold = symptom_freq_threshold
        self.prune_per_patient = prune_per_patient
        self.nnz_threshold = nnz_threshold
        self.seed = seed
        
        # Will be populated by load_data()
        self.df_records = None
        self.df_symptom = None
        self.patient_ids = None
        self.symptoms = None
        self.organs = None
        self.diseases = None
        self.symptom_meta = {}
        
        # Node counts
        self.P = 0  # patients
        self.S = 0  # symptoms
        self.O = 0  # organs
        self.D = 0  # diseases
        
        # Mappings
        self.patient_map = {}
        self.symptom_map = {}
        self.organ_map = {}
        self.disease_map = {}
        
        # Adjacency matrices
        self.A_PS = None  # Patient-Symptom
        self.A_SP = None  # Symptom-Patient
        self.A_SO = None  # Symptom-Organ
        self.A_OS = None  # Organ-Symptom
        self.A_OD = None  # Organ-Disease
        self.A_DO = None  # Disease-Organ
        
        # Meta-path matrices
        self.M_PO = None  # Patient-Organ
        self.M_PD = None  # Patient-Disease
        self.metapath_matrices = {}
        
        # Features and labels
        self.patient_feats = None
        self.patient_disease = None
        self.patient_organ_severity = None
        self.patient_organ_score = None
        self.organ_class_weights = None
        
    def load_data(self):
        """Load CSV files and perform initial filtering."""
        print("Loading CSVs...")
        # Load without parse_dates first to handle different column name formats
        self.df_records = pd.read_csv(self.path_records, low_memory=False)
        self.df_symptom = pd.read_csv(self.path_symptom, low_memory=False)
        
        # Clean column names
        self.df_records.columns = self.df_records.columns.str.strip()
        self.df_symptom.columns = self.df_symptom.columns.str.strip()
        
        # Standardize column names to PascalCase for PATIENT RECORDS
        records_column_mapping = {
            'patient_id': 'PatientID',
            'PatientID': 'PatientID',
            'report_date': 'ReportDate',
            'ReportDate': 'ReportDate',
            'record_date': 'ReportDate',       # CareAI March dataset
            'test_name': 'TestName',
            'TestName': 'TestName',
            'mapped_test_name': 'TestName',    # CareAI March dataset
            'test_value': 'TestValue',
            'TestValue': 'TestValue',
            'value': 'TestValue',              # CareAI March dataset
            'date_of_birth': 'DateOfBirth',
            'DateOfBirth': 'DateOfBirth',
            'age_at_report': 'AgeAtReport',
            'AgeAtReport': 'AgeAtReport',
            'sex': 'Sex',
            'Sex': 'Sex',
            'is_foreign': 'IsForeign',
            'IsForeign': 'IsForeign'
        }
        
        # Standardize column names for SYMPTOM/DISEASE FILE
        # Note: CSV has duplicate columns (organ, organ, disease, disease, disease)
        # Pandas auto-renames them as: organ, organ.1, disease, disease.1, disease.2
        # We only need the first occurrence of each
        # normalize column names
        self.df_symptom.columns = [c.strip().lower() for c in self.df_symptom.columns]

        # rename to match expected schema
        rename_map = {
            "test_name": "TestName",
            "organ": "Target_Organ",
            "disease": "disease",
            "min": "Min",
            "max": "Max"
        }
        
        # Rename columns that exist in the DataFrames
        rename_records = {old: new for old, new in records_column_mapping.items() if old in self.df_records.columns}
        self.df_records.rename(columns=rename_records, inplace=True)
        
        rename_symptom = {old: new for old, new in rename_map.items() if old in self.df_symptom.columns}
        self.df_symptom.rename(columns=rename_symptom, inplace=True)
        
        # Drop duplicate organ/disease columns if they exist
        cols_to_drop = [col for col in self.df_symptom.columns if col.startswith(('organ.', 'disease.'))]
        if cols_to_drop:
            self.df_symptom.drop(columns=cols_to_drop, inplace=True)
            print(f"✓ Dropped duplicate columns: {cols_to_drop}")
        
        # Parse dates if column exists
        if 'ReportDate' in self.df_records.columns:
            self.df_records['ReportDate'] = pd.to_datetime(self.df_records['ReportDate'], errors='coerce')
        
        self.df_records['TestName'] = self.df_records['TestName'].astype(str).str.strip()
        self.df_symptom['TestName'] = self.df_symptom['TestName'].astype(str).str.strip()
        
        print(f"✓ Patient records columns: {list(self.df_records.columns)}")
        print(f"✓ Symptom metadata columns: {list(self.df_symptom.columns)}")
        
        print(f"Records rows: {len(self.df_records)}, Symptom rows: {len(self.df_symptom)}")
        
        # Filter hub symptoms
        self.patient_ids = sorted(self.df_records['PatientID'].unique().tolist())
        total_patients = len(self.patient_ids)
        
        symptom_patient_counts = self.df_records.groupby('TestName')['PatientID'].nunique()
        common_symptoms = set(
            symptom_patient_counts[symptom_patient_counts / total_patients > self.symptom_freq_threshold].index.tolist()
        )
        
        print(f"Filtering {len(common_symptoms)} symptoms present in >{self.symptom_freq_threshold*100:.1f}% patients.")
        self.df_records = self.df_records[~self.df_records['TestName'].isin(common_symptoms)].copy()
        
        # Build entity lists
        self.symptoms = sorted(self.df_records['TestName'].unique().tolist())
        self.organs = sorted([x for x in self.df_symptom['Target_Organ'].unique() if pd.notna(x)])
        self.diseases = sorted([x for x in self.df_symptom['disease'].unique() if pd.notna(x)])
        
        # Node counts
        self.P = len(self.patient_ids)
        self.S = len(self.symptoms)
        self.O = len(self.organs)
        self.D = len(self.diseases)
        
        # Build mappings
        self.patient_map = {pid: i for i, pid in enumerate(self.patient_ids)}
        self.symptom_map = {s: i for i, s in enumerate(self.symptoms)}
        self.organ_map = {o: i for i, o in enumerate(self.organs)}
        self.disease_map = {d: i for i, d in enumerate(self.diseases)}
        
        print(f"Counts -> patients:{self.P}, symptoms:{self.S}, organs:{self.O}, diseases:{self.D}")
        
        # Build symptom metadata
        self._build_symptom_meta()
    
    def _build_symptom_meta(self):
        """Build symptom metadata dictionary."""
        for _, row in self.df_symptom.iterrows():
            name = str(row['TestName']).strip()
            if name not in self.symptom_map:
                continue
            
            low, high = parse_normal_range(row)
            self.symptom_meta[name] = {
                'most_relevant_disease': row.get('disease', None),
                'organ': row.get('Target_Organ', None),
                'normal_low': low,
                'normal_high': high,
                'units': row.get('Units', None)
            }
    
    def build_labels_and_features(self):
        """Build patient labels and features from records."""
        print("Computing patient disease labels and organ damage...")
        
        # Initialize arrays
        self.patient_disease = np.zeros((self.P, self.D), dtype=np.int8)
        self.patient_organ_severity = np.zeros((self.P, self.O), dtype=np.int8)
        self.patient_organ_score = np.zeros((self.P, self.O), dtype=np.float32)
        patient_symptom_dev = np.zeros((self.P, self.S), dtype=np.float32)
        patient_symptom_count = np.zeros((self.P, self.S), dtype=np.int32)
        
        # Process each record
        for idx, row in self.df_records.iterrows():
            pid = row['PatientID']
            test = row['TestName']
            
            if pid not in self.patient_map or test not in self.symptom_map:
                continue
            
            pidx = self.patient_map[pid]
            sidx = self.symptom_map[test]
            
            v = try_float(row['TestValue'])
            if v is None:
                continue
            
            meta = self.symptom_meta.get(test, {})
            low, high = meta.get('normal_low'), meta.get('normal_high')
            
            # Compute symptom deviation
            dev = 0.0
            if low is not None and high is not None and high > low:
                mid = (low + high) / 2.0
                rng = (high - low) / 2.0
                if rng > 0:
                    dev = (v - mid) / (rng * 2)
            
            patient_symptom_dev[pidx, sidx] += dev
            patient_symptom_count[pidx, sidx] += 1
            
            # Compute organ damage
            organ = meta.get('organ')
            if organ in self.organ_map and low is not None and high is not None:
                oidx = self.organ_map[organ]
                
                if v < low:
                    deficit = (low - v) / (low if low != 0 else 1.0)
                    score = min(max(deficit, 0.0), 1.0)
                    if deficit >= 0.5:
                        sev = 3
                    elif deficit >= 0.2:
                        sev = 2
                    else:
                        sev = 1
                elif v > high:
                    excess = (v - high) / (high if high != 0 else 1.0)
                    score = min(max(excess, 0.0), 1.0)
                    if excess >= 0.5:
                        sev = 3
                    elif excess >= 0.2:
                        sev = 2
                    else:
                        sev = 1
                else:
                    score = 0.0
                    sev = 0
                
                self.patient_organ_score[pidx, oidx] = max(self.patient_organ_score[pidx, oidx], score)
                self.patient_organ_severity[pidx, oidx] = max(self.patient_organ_severity[pidx, oidx], sev)
            
            # Disease multi-label
            dname = meta.get('most_relevant_disease')
            if dname in self.disease_map:
                didx = self.disease_map[dname]
                abnormal = False
                
                if low is not None and high is not None:
                    if v < low or v > high:
                        abnormal = True
                else:
                    abnormal = True
                
                if abnormal:
                    self.patient_disease[pidx, didx] = 1
        
        # Normalize symptom deviation
        mask = patient_symptom_count > 0
        patient_symptom_dev[mask] = patient_symptom_dev[mask] / patient_symptom_count[mask]
        patient_symptom_dev = np.clip(patient_symptom_dev, -3.0, 3.0) / 3.0
        
        # Combine features
        # NOTE: patient_disease is intentionally EXCLUDED from features.
        # Including it would create circular reasoning: auto-derived disease flags
        # (computed from the same test thresholds) would trivially predict the labels.
        # patient_disease is still computed and available for graph construction
        # (e.g. disease nodes, disease_map), just not used as input features.
        self.patient_feats = np.concatenate([
            patient_symptom_dev,       # [P, S] — normalised test-value deviations
            self.patient_organ_score,  # [P, O] — per-organ damage scores [0,1]
        ], axis=1)

        print(f"Patient features shape: {self.patient_feats.shape}")
    
    def build_adjacency_matrices(self):
        """Build sparse adjacency matrices."""
        print("Building sparse adjacency (CSR) on CPU...")
        
        # Patient-Symptom
        ps_rows, ps_cols = [], []
        for _, row in self.df_records.iterrows():
            pid = row['PatientID']
            test = row['TestName']
            if pid in self.patient_map and test in self.symptom_map:
                ps_rows.append(self.patient_map[pid])
                ps_cols.append(self.symptom_map[test])
        
        self.A_PS = sp.csr_matrix(
            (np.ones(len(ps_rows), dtype=np.float32), (ps_rows, ps_cols)), 
            shape=(self.P, self.S)
        )
        self.A_SP = self.A_PS.T.tocsr()
        
        # Symptom-Organ and Organ-Disease
        so_rows, so_cols = [], []
        od_rows, od_cols = [], []
        
        for _, row in self.df_symptom.iterrows():
            sname = str(row['TestName']).strip()
            if sname not in self.symptom_map:
                continue
            
            organ = row['Target_Organ']
            disease = row['disease']
            
            if organ not in self.organ_map or disease not in self.disease_map:
                continue
            
            so_rows.append(self.symptom_map[sname])
            so_cols.append(self.organ_map[organ])
            od_rows.append(self.organ_map[organ])
            od_cols.append(self.disease_map[disease])
        
        self.A_SO = sp.csr_matrix(
            (np.ones(len(so_rows), dtype=np.float32), (so_rows, so_cols)),
            shape=(self.S, self.O)
        )
        self.A_OS = self.A_SO.T.tocsr()
        
        self.A_OD = sp.csr_matrix(
            (np.ones(len(od_rows), dtype=np.float32), (od_rows, od_cols)),
            shape=(self.O, self.D)
        )
        self.A_DO = self.A_OD.T.tocsr()
        
        print(f"Adjacency nnz: A_PS={self.A_PS.nnz}, A_SO={self.A_SO.nnz}, A_OD={self.A_OD.nnz}")
    
    def build_metapaths(self, metapath_names):
        """
        Build meta-path matrices.
        
        Args:
            metapath_names: list of meta-path names to compute
                Supported: ["P-O-P", "P-D-P", "P-S-P", "P-S-O-P", "P-O-D-P"]
        
        Returns:
            dict: {metapath_name: neighbor_dict}
        """
        print("Computing base metapaths (sparse cpu)...")
        t0 = time.time()
        
        # Build base matrices
        self.M_PO = self.A_PS.dot(self.A_SO)  # P x O
        if self.A_PS.nnz > 0 and self.A_SO.nnz > 0 and self.A_OD.nnz > 0:
            self.M_PD = self.A_PS.dot(self.A_SO).dot(self.A_OD)  # P x D
        else:
            self.M_PD = sp.csr_matrix((self.P, self.D))
        
        self.M_PO.eliminate_zeros()
        self.M_PD.eliminate_zeros()
        
        t1 = time.time()
        print(f"M_PO nnz={self.M_PO.nnz}, M_PD nnz={self.M_PD.nnz}  time={(t1-t0):.2f}s")
        
        # Build patient-patient meta-paths
        P_OP_P = self.M_PO.dot(self.M_PO.T)  # P x P via organs
        P_DP_P = sp.csr_matrix(self.patient_disease.astype(np.float32)).dot(
            sp.csr_matrix(self.patient_disease.astype(np.float32)).T
        )  # P x P via diseases
        P_SP_P = self.A_PS.dot(self.A_PS.T)  # P x P via symptoms
        
        P_OP_P.eliminate_zeros()
        P_DP_P.eliminate_zeros()
        P_SP_P.eliminate_zeros()
        
        # P-O-D-P
        if self.M_PD.shape[1] > 0:
            P_OD_P = sp.csr_matrix(self.M_PD.dot(self.M_PD.T))
        else:
            P_OD_P = sp.csr_matrix((self.P, self.P))
        
        # Store available meta-path matrices
        available_meta_matrices = {
            "P-O-P": P_OP_P,
            "P-D-P": P_DP_P,
            "P-S-P": P_SP_P,
            "P-S-O-P": P_OP_P,  # approximate with P-O-P
            "P-O-D-P": P_OD_P
        }
        
        # Build neighbor dicts for requested meta-paths
        patient_metapath_neighbors = {}
        for name in metapath_names:
            if name in available_meta_matrices:
                M = available_meta_matrices[name]
                if M.nnz > 0 and M.nnz <= self.nnz_threshold:
                    patient_metapath_neighbors[name] = csr_to_neighbors_prune(
                        M, max_per_node=self.prune_per_patient, seed=self.seed
                    )
                    print(f"Added metapath {name} with nnz {M.nnz}")
                else:
                    print(f"Skipped metapath {name} due to nnz {M.nnz}")
        
        self.metapath_matrices = patient_metapath_neighbors
        return patient_metapath_neighbors
    
    def compute_class_weights(self, device='cpu'):
        """Compute class weights for organ severity classification."""
        organ_class_weights = []
        for o_idx in range(self.O):
            counts = np.bincount(self.patient_organ_severity[:, o_idx], minlength=4)
            total = counts.sum()
            weights = total / (4 * (counts + 1e-9))
            weights = np.clip(weights, 0.1, 100.0)
            organ_class_weights.append(torch.tensor(weights, dtype=torch.float32, device=device))
        
        self.organ_class_weights = organ_class_weights
        return organ_class_weights
    
    def get_tensors(self, device='cpu'):
        """
        Convert features and labels to PyTorch tensors.
        
        Returns:
            dict with keys:
                - patient_feats
                - labels_organ_severity
                - patient_disease
                - patient_organ_score
        """
        return {
            'patient_feats': torch.from_numpy(self.patient_feats).float().to(device),
            'labels_organ_severity': torch.from_numpy(self.patient_organ_severity.astype(np.int64)).to(device),
            'patient_disease': torch.from_numpy(self.patient_disease.astype(np.float32)).to(device),
            'patient_organ_score': torch.from_numpy(self.patient_organ_score).float().to(device)
        }
