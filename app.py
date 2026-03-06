
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import HAN components
from HAN import (
    MedicalGraphData,
    HANPP_Disease,
    load_test_reference,
    recommend_all,
    format_patient_json
)


from HAN.interpretability import (
    extract_semantic_attention,
    generate_explanation
)

from HAN.utils import parse_normal_range

from HAN.mc_dropout import mc_dropout_predict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model paths
MODEL_PATH = 'models_saved/careai_march/hanpp_disease_v6_PDP_POP.pt'

# Data paths
RECORDS_PATH = 'final_processed_data/merged_coop_ruhunu_patient_data.csv'
SYMPTOM_PATH = 'final_processed_data/unique_test_data_finalized_temp.csv'
DISEASE_CLUSTER_PATH = 'data/disease_cluster_mapping.json'
TEST_REFERENCE_PATH = 'data/dataset_careai_March/processed/test_reference_full_v2.csv'

import json

with open("data/disease_cluster_mapping_2.json") as f:
    cluster_to_diseases = json.load(f)["cluster_to_diseases"]


SEVERITY_LEVELS = {
    0: {'name': 'NORMAL',   'description': 'No significant abnormalities detected'},
    1: {'name': 'MILD',     'description': 'Minor abnormalities – monitor closely'},
    2: {'name': 'MODERATE', 'description': 'Significant concern – investigation needed'},
    3: {'name': 'SEVERE',   'description': 'Critical – immediate attention required'}
}

# Medical tests reference — built from unique_test_data_finalized.csv


# ---------------------------------------------------------------------------
# Globals – populated on startup
# ---------------------------------------------------------------------------
data_loader = None
#organ_model = None          # HANPP for organ severity
disease_model = None        # HANPP_Disease for disease prediction
existing_patient_ids = []
organ_names = []
disease_names = []          # ordered list of disease cluster names
cluster_to_diseases = {}    # mapping from cluster index to disease names
test_reference = {}         # test reference for recommendations
organ_model_in_dim = None
disease_model_in_dim = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------



def load_system():

    global data_loader
    global disease_model
    global disease_names
    global existing_patient_ids
    global test_reference
    global cluster_labels

    print("Loading patient graph...")

    data_loader = MedicalGraphData(
        path_records=RECORDS_PATH,
        path_symptom=TEST_REFERENCE_PATH,
        symptom_freq_threshold=0.08,
        prune_per_patient=300,
        nnz_threshold=80_000_000,
        seed=42
    )

    data_loader.load_data()
    data_loader.build_labels_and_features()
    data_loader.build_adjacency_matrices()

    data_loader.build_metapaths(["P-D-P", "P-O-P"])
    cluster_labels = build_cluster_labels()

    existing_patient_ids = list(data_loader.patient_ids)

    print("Loading disease model...")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    in_dim = checkpoint["project.weight"].shape[1]
    hidden_dim = checkpoint["project.weight"].shape[0]
    out_dim = checkpoint["out_proj.weight"].shape[0]
    num_diseases = checkpoint["disease_classifier.weight"].shape[0]

    disease_model = HANPP_Disease(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        metapath_names=["P-D-P", "P-O-P"],
        num_heads=4,
        num_diseases=num_diseases,
        dropout=0.3
    )

    disease_model.load_state_dict(checkpoint)
    disease_model.to(DEVICE)
    disease_model.eval()

    disease_names = [f"Disease_{i}" for i in range(num_diseases)]

    if os.path.exists(TEST_REFERENCE_PATH):
        test_reference = load_test_reference(TEST_REFERENCE_PATH)

    print("System ready")


def get_feature_tensor():
    feats = data_loader.patient_feats
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    return feats.float().to(DEVICE)



def build_disease_mapping():

    mapping = {}

    for _,row in data_loader.df_symptom.iterrows():

        disease = str(row["disease"]).strip()
        organ = str(row["Target_Organ"]).strip()

        if organ not in mapping:
            mapping[organ] = set()

        mapping[organ].add(disease)

    # convert sets to lists
    mapping = {k:list(v) for k,v in mapping.items()}

    return mapping

def infer_organ_severity(tests):

    organs = {}

    for t in tests:

        rows = data_loader.df_symptom[
            data_loader.df_symptom["TestName"] == t["test_name"]
        ]

        if len(rows) == 0:
            continue

        row = rows.iloc[0]
        organ = row["Target_Organ"]

        low, high = parse_normal_range(row)
        value = float(t["test_value"])

        severity = 0

        if low and value < low:
            severity = 1
        elif high and value > high:
            severity = 2

        if organ not in organs:
            organs[organ] = severity
        else:
            organs[organ] = max(organs[organ], severity)

    result = []

    for organ, sev in organs.items():

        level = "NORMAL"

        if sev == 1:
            level = "MODERATE"

        if sev == 2:
            level = "HIGH"

        result.append({
            "organ": organ,
            "severity": level
        })

    return result

def run_uncertainty(patient_indices):

    features = torch.from_numpy(data_loader.patient_feats).float().to(DEVICE)

    neighbor_dicts = data_loader.metapath_neighbors

    preds, conf, unc, probs, _, _ = mc_dropout_predict(
        disease_model,
        features,
        neighbor_dicts,
        n_samples=30,
        device=DEVICE
    )

    results = []

    for pidx in patient_indices:

        probs_dict = {}
        unc_dict = {}

        for i, name in enumerate(disease_names):

            probs_dict[name] = float(probs[pidx].max())
            unc_dict[name] = float(unc[pidx].max())

        results.append({
            "disease_probs": probs_dict,
            "disease_uncertainties": unc_dict
        })

    return results





def build_cluster_labels():

    df = data_loader.df_symptom

    diseases = sorted(df["disease"].dropna().unique())

    cluster_map = {}

    for i, d in enumerate(diseases):
        cluster_map[i] = d

    return cluster_map

def recommend_tests(organ):

    df = data_loader.df_symptom

    rows = df[df["Target_Organ"] == organ]

    tests = rows["TestName"].unique().tolist()

    return tests[:3]


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    """Health-check / info endpoint."""
    return jsonify({
        'service': 'HAN Medical Prediction API',
        'version': '2.0.0',
        'models': {
            'organ_model': 'HANPP (P-S-P)' if organ_model else 'Not loaded',
            'disease_model': 'HANPP_Disease (P-D-P, P-O-P)' if disease_model else 'Not loaded'
        },
        'patients_loaded': len(existing_patient_ids),
        'organs_tracked': len(organ_names),
        'organ_names': organ_names,
        'diseases_tracked': len(disease_names),
        'disease_names': disease_names,
        'endpoints': {
            'POST /api/predict/existing': 'Predict organ severity + disease for existing patients',
            'POST /api/predict/new': 'Predict organ severity + disease for a new patient',
            'POST /api/recommend': 'Uncertainty-guided test recommendations',
            'GET /api/patients': 'List existing patient IDs',
            'GET /api/tests': 'List recognised medical tests',
            'GET /api/diseases': 'List detectable diseases',
        }
    })




# ──────────────────────────────────────────────────────────────────────────
# 2. PREDICT NEW PATIENT
# ──────────────────────────────────────────────────────────────────────────
@app.route("/api/predict/new", methods=["POST"])
def predict_new():

    body = request.json
    tests = body.get("tests")

    if not tests:
        return jsonify({"error": "tests array required"}), 400

    patient_id = body.get("patient_id", "NEW_PATIENT")

    # build mapping once
    organ_disease_map = build_disease_mapping()

    # -------------------------
    # Build feature vector
    # -------------------------
    feat_vec = np.zeros(data_loader.patient_feats.shape[1])

    for t in tests:

        name = t["test_name"]
        value = float(t["test_value"])

        rows = data_loader.df_symptom[
            data_loader.df_symptom["TestName"] == name
        ]

        if len(rows) == 0:
            continue

        row = rows.iloc[0]

        low, high = parse_normal_range(row)

        if low is not None and high is not None:

            mid = (low + high) / 2
            feat_vec += (value - mid) / (high - low + 1e-6)

    feat_vec = torch.tensor(feat_vec).float().unsqueeze(0)

    # -------------------------
    # Fix feature dimension
    # -------------------------
    expected_dim = disease_model.project.weight.shape[1]

    if feat_vec.shape[1] < expected_dim:
        pad = torch.zeros(1, expected_dim - feat_vec.shape[1])
        feat_vec = torch.cat([feat_vec, pad], dim=1)

    elif feat_vec.shape[1] > expected_dim:
        feat_vec = feat_vec[:, :expected_dim]

    feat_vec = feat_vec.to(DEVICE)

    neighbor_dicts = {k:{} for k in disease_model.metapath_names}

    # -------------------------
    # Run model
    # -------------------------
    with torch.no_grad():

        logits, z, beta = disease_model(feat_vec, neighbor_dicts)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # -------------------------
    # Organ analysis
    # -------------------------
    organ_analysis = infer_organ_severity(tests)

    affected_organ = None
    if len(organ_analysis) > 0:
        affected_organ = organ_analysis[0]["organ"]

    # -------------------------
    # Get top disease
    # -------------------------
    top_idx = np.argmax(probs)
    top_prob = float(probs[top_idx])

    disease_group = cluster_labels.get(top_idx, "Unknown_Disease")
    print("Model disease classes:", len(cluster_labels))
    print("Top prediction index:", top_idx)
    print("Predicted disease:", cluster_labels.get(top_idx))
    # -------------------------
    # Map diseases from organ
    # -------------------------
    possible_conditions = []

    if affected_organ:
        possible_conditions = organ_disease_map.get(affected_organ, [])

    # -------------------------
    # Recommend tests
    # -------------------------
    recommended_tests = []

    if affected_organ:

        df = data_loader.df_symptom
        rows = df[df["Target_Organ"] == affected_organ]

        recommended_tests = rows["TestName"].unique().tolist()[:3]

    # -------------------------
    # Final response
    # -------------------------
    response = {
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(),

        "organ_analysis": organ_analysis,

        "disease_predictions": [
            {
                "disease_group": disease_group,
                "possible_conditions": possible_conditions,
                "affected_organs": [affected_organ],
                "probability": round(top_prob,4)
            }
        ],

        "recommended_tests": recommended_tests
    }

    return jsonify(response)


@app.route("/api/explain", methods=["POST"])
def explain():

    body = request.json

    pid = body.get("patient_id")

    if pid not in existing_patient_ids:

        return jsonify({"error":"patient not found"}),404

    pidx = existing_patient_ids.index(pid)

    feats = get_feature_tensor()

    neighbor_dicts = data_loader.metapath_neighbors

    beta, metapaths = extract_semantic_attention(
        disease_model,
        feats,
        neighbor_dicts,
        DEVICE
    )

    beta_patient = beta[pidx]

    explanation = generate_explanation(
        patient_idx=pidx,
        beta=beta,
        metapath_names=metapaths,
        patient_ids=existing_patient_ids
    )

    weights = {}

    for i,name in enumerate(metapaths):
        weights[name] = float(beta_patient[i])

    return jsonify({
        "patient_id":pid,
        "meta_path_weights":weights,
        "explanation":explanation
    })


# ──────────────────────────────────────────────────────────────────────────
# Utility endpoints
# ──────────────────────────────────────────────────────────────────────────




@app.route("/api/tests", methods=["GET"])
def list_tests():

    tests = []

    for _,row in data_loader.df_symptom.iterrows():

        low,high = parse_normal_range(row)

        tests.append({
            "test_name":row["TestName"],
            "organ":row.get("Target_Organ",""),
            "normal_range":f"{low}-{high}" if low and high else "N/A"
        })

    return jsonify({
        "total_tests":len(tests),
        "tests":tests
    })

@app.route("/api/patients", methods=["GET"])
def get_patients():

    if data_loader is None:
        return jsonify({
            "error": "System not initialized"
        }), 500

    # patient IDs loaded by HAN graph loader
    patient_ids = list(existing_patient_ids)

    return jsonify({
        "status": "success",
        "total_patients": len(patient_ids),
        "patients": patient_ids[95000:95100]   # return first 100 to avoid huge response
    })





# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    load_system()
    app.run(host='0.0.0.0', port=5001, debug=False)
