#!/usr/bin/env python3
"""
Flask REST API for HAN Medical Prediction System
==================================================

Provides three endpoints:
  POST /api/predict/existing  — Predict organ severity for existing patients
  POST /api/predict/new       — Predict organ severity for a new patient (JSON lab data)
  POST /api/recommend         — Get confirmatory test recommendations

Startup:
  python app.py

The app loads the trained HANPP model and patient data graph on startup.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import HAN components
from HAN import MedicalGraphData, HANPP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models_saved/ruhunu_data_clustered/hanpp_P-S-P.pt'
RECORDS_PATH = 'data/filtered_patient_reports.csv'
SYMPTOM_PATH = 'data/test-disease-organ.csv'

SEVERITY_LEVELS = {
    0: {'name': 'NORMAL',   'description': 'No significant abnormalities detected'},
    1: {'name': 'MILD',     'description': 'Minor abnormalities – monitor closely'},
    2: {'name': 'MODERATE', 'description': 'Significant concern – investigation needed'},
    3: {'name': 'SEVERE',   'description': 'Critical – immediate attention required'}
}

CONFIRMATORY_TESTS = {
    'kidney': {
        'initial_tests': ['Serum_Creatinine_Result', 'eGFR Result', 'Blood Urea Result'],
        'confirmatory_tests': [
            'Urine Albumin-to-Creatinine Ratio',
            '24-hour Urine Collection',
            'Renal Ultrasound',
            'Cystatin C',
            'Kidney Biopsy (if severe)'
        ],
        'monitoring_tests': ['Serum - Potassium', 'Serum - Sodium', 'Serum Bicarbonate']
    },
    'liver': {
        'initial_tests': ['SGPT (ALT) Result', 'SGOT (AST) Result', 'Bilirubin'],
        'confirmatory_tests': [
            'Liver Function Panel (comprehensive)',
            'Gamma-GT',
            'Alkaline Phosphatase',
            'Prothrombin Time (PT/INR)',
            'Hepatitis Panel',
            'Liver Ultrasound',
            'FibroScan / Liver Elastography'
        ],
        'monitoring_tests': ['Albumin', 'Total Protein']
    },
    'cardiovascular system': {
        'initial_tests': ['Total Cholesterol', 'LDL-Cholesterol', 'HDL Cholesterol', 'Triglycerides'],
        'confirmatory_tests': [
            'Apolipoprotein B',
            'Lipoprotein(a)',
            'hs-CRP (inflammatory marker)',
            'Cardiac Troponin',
            'ECG',
            'Echocardiogram',
            'Coronary Calcium Score CT'
        ],
        'monitoring_tests': ['Blood Pressure', 'Heart Rate Variability']
    },
    'pancreas': {
        'initial_tests': ['HbA1c Result', 'Fasting Glucose'],
        'confirmatory_tests': [
            'Oral Glucose Tolerance Test (OGTT)',
            'C-Peptide',
            'Insulin Level',
            'Fructosamine',
            'Continuous Glucose Monitoring (CGM)',
            'Pancreatic Ultrasound',
            'Anti-GAD Antibodies (Type 1 screening)'
        ],
        'monitoring_tests': ['Postprandial Glucose', 'Urine Ketones']
    },
    'thyroid': {
        'initial_tests': ['TSH'],
        'confirmatory_tests': [
            'Free T4',
            'Free T3',
            'Thyroid Antibodies (TPO, TgAb)',
            'Thyroid Ultrasound',
            'Thyroid Uptake Scan',
            'Fine Needle Aspiration (if nodules)'
        ],
        'monitoring_tests': ['Reverse T3', 'Thyroglobulin']
    },
    'blood': {
        'initial_tests': ['Haemoglobin Absolute Value', 'WBC Absolute Value',
                          'Platelet Count Absolute Value', 'RBC Absolute Value'],
        'confirmatory_tests': [
            'Complete Blood Count with Differential',
            'Peripheral Blood Smear',
            'Reticulocyte Count',
            'Iron Panel (Iron, Ferritin, TIBC)',
            'Vitamin B12 and Folate',
            'Bone Marrow Biopsy (if severe)',
            'Hemoglobin Electrophoresis'
        ],
        'monitoring_tests': ['ESR', 'Haptoglobin']
    },
    'immune system': {
        'initial_tests': ['WBC Absolute Value', 'Lymphocyte Count'],
        'confirmatory_tests': [
            'Immunoglobulin Panel (IgG, IgA, IgM, IgE)',
            'Complement Levels (C3, C4)',
            'Autoimmune Panel (ANA, RF, Anti-CCP)',
            'T-cell and B-cell Subsets',
            'HIV Test',
            'CMV, EBV serologies'
        ],
        'monitoring_tests': ['Neutrophil Count', 'CD4/CD8 Ratio']
    }
}

# Medical tests reference (for new patient symptom-level analysis)
MEDICAL_TESTS = {
    'Haemoglobin Absolute Value':      {'normal': (12.5, 16),   'unit': 'g/dL',     'organ': 'blood'},
    'WBC Absolute Value':              {'normal': (4, 11),      'unit': '10^9/L',   'organ': 'immune system'},
    'Platelet Count Absolute Value':   {'normal': (150, 450),   'unit': '10^9/L',   'organ': 'blood'},
    'HbA1c Result':                    {'normal': (4, 5.6),     'unit': '%',         'organ': 'pancreas'},
    'Serum_Creatinine_Result':         {'normal': (0.65, 1.2),  'unit': 'mg/dL',    'organ': 'kidney'},
    'eGFR Result':                     {'normal': (90, 120),    'unit': 'mL/min',   'organ': 'kidney'},
    'Total Cholesterol':               {'normal': (100, 200),   'unit': 'mg/dL',    'organ': 'cardiovascular system'},
    'LDL-Cholesterol':                 {'normal': (50, 100),    'unit': 'mg/dL',    'organ': 'cardiovascular system'},
    'HDL Cholesterol':                 {'normal': (45, 80),     'unit': 'mg/dL',    'organ': 'cardiovascular system'},
    'SGPT (ALT) Result':               {'normal': (7, 56),      'unit': 'U/L',      'organ': 'liver'},
    'TSH':                             {'normal': (0.4, 4),     'unit': 'mIU/L',    'organ': 'thyroid'},
    'Serum - Potassium':               {'normal': (3.5, 5.1),   'unit': 'mmol/L',   'organ': 'kidney'},
    'Serum - Sodium':                  {'normal': (135, 145),   'unit': 'mmol/L',   'organ': 'kidney'},
    'Blood Urea Result':               {'normal': (20, 40),     'unit': 'mg/dL',    'organ': 'kidney'},
    'RBC Absolute Value':              {'normal': (4.11, 5.51), 'unit': '10^12/L',  'organ': 'blood'},
}

# ---------------------------------------------------------------------------
# Globals – populated on startup
# ---------------------------------------------------------------------------
data_loader = None
model = None
existing_patient_ids = []
organ_names = []
model_in_dim = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_system():
    """Load the patient data graph and trained model into memory."""
    global data_loader, model, existing_patient_ids, organ_names, model_in_dim

    print("=" * 70)
    print("  HAN Medical Prediction API – Initialising")
    print("=" * 70)

    # --- Data graph --------------------------------------------------------
    print(f"\n📂 Loading patient records: {RECORDS_PATH}")
    print(f"📂 Loading symptom metadata: {SYMPTOM_PATH}")

    data_loader = MedicalGraphData(
        path_records=RECORDS_PATH,
        path_symptom=SYMPTOM_PATH,
        symptom_freq_threshold=0.08,
        prune_per_patient=300,
        nnz_threshold=80_000_000,
        seed=42
    )
    data_loader.load_data()
    data_loader.build_labels_and_features()
    data_loader.build_adjacency_matrices()
    data_loader.build_metapaths(['P-S-P'])

    existing_patient_ids = list(data_loader.patient_ids)
    organ_names = list(data_loader.organs) if hasattr(data_loader, 'organs') else []

    print(f"✅ Graph ready  –  {data_loader.P} patients  |  "
          f"{data_loader.S} symptoms  |  {data_loader.O} organs")

    # --- Model -------------------------------------------------------------
    print(f"\n🔧 Loading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at {MODEL_PATH}")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    in_dim = checkpoint['project.weight'].shape[1]
    hidden_dim = checkpoint['project.weight'].shape[0]
    out_dim = checkpoint['out_proj.weight'].shape[0]
    num_organs = len(organ_names) if organ_names else 25
    model_in_dim = in_dim

    model = HANPP(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        metapath_names=['P-S-P'],
        num_heads=4,
        num_organs=num_organs,
        num_severity=4,
        dropout=0.3
    )
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    # Align feature dimension
    if data_loader.patient_feats.shape[1] != in_dim:
        feats = data_loader.patient_feats
        if feats.shape[1] < in_dim:
            pad = np.zeros((feats.shape[0], in_dim - feats.shape[1]))
            data_loader.patient_feats = np.hstack([feats, pad])
        else:
            data_loader.patient_feats = feats[:, :in_dim]

    print(f"✅ Model loaded  –  in={in_dim}  hidden={hidden_dim}  "
          f"out={out_dim}  organs={num_organs}")
    print("=" * 70)
    print("  API ready – http://localhost:5000")
    print("=" * 70)


def run_inference(patient_indices):
    """Run the HANPP model on given patient indices and return results."""
    feats = data_loader.patient_feats
    if isinstance(feats, np.ndarray):
        features = torch.from_numpy(feats).float().to(DEVICE)
    else:
        features = feats.to(DEVICE)

    neighbor_dicts = data_loader.metapath_matrices

    with torch.no_grad():
        organ_logits, organ_scores, _, _ = model(features, neighbor_dicts)
        predictions = torch.argmax(organ_logits, dim=2).cpu().numpy()
        scores = organ_scores.cpu().numpy()
        confidences = torch.softmax(organ_logits, dim=2).max(dim=2)[0].cpu().numpy()

    results = []
    for pidx in patient_indices:
        patient_id = data_loader.patient_ids[pidx]
        organs_out = []
        for oidx, oname in enumerate(organ_names):
            if oidx < predictions.shape[1]:
                sev = int(predictions[pidx, oidx])
                organs_out.append({
                    'organ': oname,
                    'severity': sev,
                    'severity_name': SEVERITY_LEVELS[sev]['name'],
                    'description': SEVERITY_LEVELS[sev]['description'],
                    'damage_score': round(float(scores[pidx, oidx]), 4),
                    'confidence': round(float(confidences[pidx, oidx]) * 100, 2)
                })
        affected = [o for o in organs_out if o['severity'] > 0]
        affected.sort(key=lambda x: (x['severity'], x['damage_score']), reverse=True)
        normal = [o for o in organs_out if o['severity'] == 0]

        results.append({
            'patient_id': patient_id,
            'overall_status': affected[0]['severity_name'] if affected else 'NORMAL',
            'affected_organs': affected,
            'normal_organs_count': len(normal),
            'total_organs_assessed': len(organs_out)
        })
    return results


def symptom_level_analysis(test_records):
    """Rule-based symptom severity (same logic as predict_psp_new_patients)."""
    symptom_results = []
    for rec in test_records:
        test_name = rec.get('test_name', '')
        test_value = float(rec.get('test_value', 0))
        if test_name in MEDICAL_TESTS:
            info = MEDICAL_TESTS[test_name]
            lo, hi = info['normal']
            if test_value < lo:
                deviation = (lo - test_value) / lo
                direction = 'LOW'
            elif test_value > hi:
                deviation = (test_value - hi) / hi
                direction = 'HIGH'
            else:
                deviation = 0
                direction = 'NORMAL'

            if deviation == 0:
                sev = 0
            elif deviation < 0.3:
                sev = 1
            elif deviation < 0.6:
                sev = 2
            else:
                sev = 3

            symptom_results.append({
                'symptom': test_name,
                'value': test_value,
                'unit': info['unit'],
                'normal_range': f"{lo}-{hi}",
                'deviation_pct': round(deviation * 100, 1),
                'direction': direction,
                'severity': sev,
                'severity_name': SEVERITY_LEVELS[sev]['name'],
                'target_organ': info['organ']
            })
    return symptom_results


def aggregate_to_organ(symptom_results):
    """Aggregate symptom severities to organ level."""
    from collections import defaultdict
    organ_syms = defaultdict(list)
    for s in symptom_results:
        organ_syms[s['target_organ']].append(s)

    organ_out = []
    for organ, syms in organ_syms.items():
        avg_sev = np.mean([s['severity'] for s in syms])
        max_sev = max(s['severity'] for s in syms)
        n_abnormal = sum(1 for s in syms if s['severity'] > 0)
        ratio = n_abnormal / len(syms)

        if max_sev == 0:
            organ_sev = 0
        elif max_sev == 1 and ratio < 0.3:
            organ_sev = 1
        elif max_sev <= 2 and ratio < 0.5:
            organ_sev = 1
        elif max_sev == 2 or (max_sev == 1 and ratio >= 0.5):
            organ_sev = 2
        else:
            organ_sev = 3

        organ_out.append({
            'organ': organ,
            'severity': organ_sev,
            'severity_name': SEVERITY_LEVELS[organ_sev]['name'],
            'description': SEVERITY_LEVELS[organ_sev]['description'],
            'avg_symptom_severity': round(avg_sev, 2),
            'max_symptom_severity': int(max_sev),
            'num_symptoms': len(syms),
            'num_abnormal': n_abnormal,
            'abnormal_ratio': round(ratio, 2),
            'damage_score': round(avg_sev * ratio, 3)
        })
    organ_out.sort(key=lambda x: x['severity'], reverse=True)
    return organ_out


def get_recommendations(affected_organs, existing_tests):
    """Generate confirmatory test recommendations."""
    recs = []
    for organ_info in affected_organs:
        organ_lower = organ_info['organ'].lower()
        severity = organ_info['severity']

        matched_system = None
        for system_key in CONFIRMATORY_TESTS:
            if system_key in organ_lower or organ_lower in system_key:
                matched_system = system_key
                break
        # extra keyword matches
        if not matched_system:
            if any(k in organ_lower for k in ['heart', 'cardiac', 'vascular']):
                matched_system = 'cardiovascular system'
            elif any(k in organ_lower for k in ['diabete', 'glucose', 'insulin']):
                matched_system = 'pancreas'
            elif any(k in organ_lower for k in ['renal', 'nephro']):
                matched_system = 'kidney'
            elif 'hepat' in organ_lower:
                matched_system = 'liver'
            elif any(k in organ_lower for k in ['hematolog', 'anemia']):
                matched_system = 'blood'

        if not matched_system:
            continue

        test_info = CONFIRMATORY_TESTS[matched_system]
        tests_done = [t for t in test_info['initial_tests'] if t in existing_tests]
        tests_missing = [t for t in test_info['initial_tests'] if t not in existing_tests]

        confirmatory = test_info['confirmatory_tests']
        monitoring = test_info['monitoring_tests']

        if severity >= 3:
            priority = 'URGENT'
            recommended = confirmatory[:4] + monitoring[:2]
        elif severity == 2:
            priority = 'HIGH'
            recommended = confirmatory[:3] + monitoring[:1]
        else:
            priority = 'ROUTINE'
            recommended = confirmatory[:2]

        recs.append({
            'organ': organ_info['organ'],
            'severity': organ_info['severity_name'],
            'priority': priority,
            'initial_tests_done': tests_done,
            'initial_tests_missing': tests_missing,
            'recommended_tests': recommended,
            'rationale': (f"To confirm {organ_info['severity_name']} "
                          f"{organ_info['organ']} dysfunction "
                          f"(Confidence: {organ_info.get('confidence', 'N/A')}%)")
        })
    return recs


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
        'version': '1.0.0',
        'model': 'HANPP with P-S-P Meta-Path',
        'patients_loaded': len(existing_patient_ids),
        'organs_tracked': len(organ_names),
        'organ_names': organ_names,
        'endpoints': {
            'POST /api/predict/existing': 'Predict organ severity for existing patients',
            'POST /api/predict/new': 'Predict organ severity for a new patient',
            'POST /api/recommend': 'Get confirmatory test recommendations'
        }
    })


# ──────────────────────────────────────────────────────────────────────────
# 1. PREDICT EXISTING PATIENTS
# ──────────────────────────────────────────────────────────────────────────
@app.route('/api/predict/existing', methods=['POST'])
def predict_existing():
    """
    Predict organ severity for existing patients in the dataset.

    JSON body:
        {
          "patient_ids": [139760, 200041, ...]   // optional, defaults to first 10
        }

    Returns prediction results per patient.
    """
    body = request.get_json(silent=True) or {}
    requested_ids = body.get('patient_ids', None)

    pid_to_idx = {pid: i for i, pid in enumerate(data_loader.patient_ids)}

    if requested_ids:
        # Validate
        valid_indices = []
        invalid_ids = []
        for pid in requested_ids:
            if pid in pid_to_idx:
                valid_indices.append(pid_to_idx[pid])
            else:
                invalid_ids.append(pid)
        if not valid_indices:
            return jsonify({
                'error': 'None of the requested patient IDs were found in the dataset.',
                'invalid_ids': invalid_ids,
                'hint': 'Use GET / to see available data, or POST without patient_ids for defaults.'
            }), 404
    else:
        # Default to first 10 patients
        valid_indices = list(range(min(10, data_loader.P)))
        invalid_ids = []

    results = run_inference(valid_indices)

    response = {
        'status': 'success',
        'model': 'HANPP (P-S-P)',
        'timestamp': datetime.now().isoformat(),
        'patients_predicted': len(results),
        'predictions': results
    }
    if invalid_ids:
        response['warnings'] = {
            'invalid_patient_ids': invalid_ids,
            'message': f'{len(invalid_ids)} patient ID(s) not found in dataset'
        }
    return jsonify(response)


# ──────────────────────────────────────────────────────────────────────────
# 2. PREDICT NEW PATIENT
# ──────────────────────────────────────────────────────────────────────────
@app.route('/api/predict/new', methods=['POST'])
def predict_new():
    """
    Predict organ severity for a brand-new patient using submitted lab tests.

    Uses symptom-level rule-based analysis (P-S-P meta-path logic) since
    the new patient is not part of the existing graph.

    JSON body:
        {
          "patient_id": "NEW_001",            // optional identifier
          "age": 45,                          // optional
          "sex": "Male",                      // optional
          "tests": [
              {"test_name": "HbA1c Result", "test_value": 7.2},
              {"test_name": "Serum_Creatinine_Result", "test_value": 1.8},
              {"test_name": "Total Cholesterol", "test_value": 260},
              ...
          ]
        }

    Returns symptom-level AND organ-level severity predictions.
    """
    body = request.get_json(silent=True)
    if not body or 'tests' not in body:
        return jsonify({
            'error': 'Missing required field: tests',
            'expected_format': {
                'tests': [
                    {'test_name': 'HbA1c Result', 'test_value': 7.2},
                    {'test_name': 'Serum_Creatinine_Result', 'test_value': 1.8}
                ]
            },
            'available_tests': list(MEDICAL_TESTS.keys())
        }), 400

    tests = body['tests']
    patient_id = body.get('patient_id', 'NEW_PATIENT')
    age = body.get('age', None)
    sex = body.get('sex', None)

    # Symptom-level analysis
    symptom_results = symptom_level_analysis(tests)

    if not symptom_results:
        return jsonify({
            'error': 'None of the submitted test names are recognised.',
            'submitted_tests': [t.get('test_name') for t in tests],
            'available_tests': list(MEDICAL_TESTS.keys())
        }), 400

    # Aggregate to organ level
    organ_results = aggregate_to_organ(symptom_results)

    affected = [o for o in organ_results if o['severity'] > 0]
    overall = affected[0]['severity_name'] if affected else 'NORMAL'

    response = {
        'status': 'success',
        'analysis_method': 'P-S-P Rule-Based Symptom Analysis',
        'timestamp': datetime.now().isoformat(),
        'patient': {
            'patient_id': patient_id,
            'age': age,
            'sex': sex
        },
        'overall_status': overall,
        'symptom_analysis': symptom_results,
        'organ_predictions': organ_results,
        'summary': {
            'total_tests_analysed': len(symptom_results),
            'tests_unrecognised': len(tests) - len(symptom_results),
            'normal_symptoms': sum(1 for s in symptom_results if s['severity'] == 0),
            'abnormal_symptoms': sum(1 for s in symptom_results if s['severity'] > 0),
            'organs_affected': len(affected),
            'organs_normal': len(organ_results) - len(affected)
        }
    }
    return jsonify(response)


# ──────────────────────────────────────────────────────────────────────────
# 3. TEST RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────
@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Get confirmatory test recommendations based on organ severity predictions.

    Can be used in two modes:

    Mode A – for an EXISTING patient (uses trained model):
        {
          "patient_id": 139760
        }

    Mode B – for a NEW patient (uses submitted test data):
        {
          "tests": [
              {"test_name": "HbA1c Result", "test_value": 7.2},
              ...
          ]
        }

    Returns recommended confirmatory tests per affected organ.
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'Request body is required.'}), 400

    # --- Mode A: existing patient ------------------------------------------
    if 'patient_id' in body and 'tests' not in body:
        pid = body['patient_id']
        pid_to_idx = {p: i for i, p in enumerate(data_loader.patient_ids)}
        if pid not in pid_to_idx:
            return jsonify({
                'error': f'Patient ID {pid} not found in dataset.',
                'hint': 'Supply "tests" array instead for a new patient.'
            }), 404

        pidx = pid_to_idx[pid]
        pred_results = run_inference([pidx])[0]
        affected = pred_results['affected_organs']

        # Gather what tests this patient already has
        patient_records = data_loader.df_records[
            data_loader.df_records['PatientID'] == pid
        ]
        existing_tests = patient_records['TestName'].unique().tolist() if len(patient_records) > 0 else []

        recs = get_recommendations(affected, existing_tests)

        return jsonify({
            'status': 'success',
            'mode': 'existing_patient',
            'patient_id': pid,
            'timestamp': datetime.now().isoformat(),
            'affected_organs': affected,
            'recommendations': recs,
            'existing_tests_count': len(existing_tests)
        })

    # --- Mode B: new patient with test data --------------------------------
    if 'tests' in body:
        tests = body['tests']
        symptom_results = symptom_level_analysis(tests)
        if not symptom_results:
            return jsonify({
                'error': 'No recognised tests found.',
                'available_tests': list(MEDICAL_TESTS.keys())
            }), 400

        organ_results = aggregate_to_organ(symptom_results)
        affected = [o for o in organ_results if o['severity'] > 0]
        existing_tests = [t.get('test_name', '') for t in tests]

        recs = get_recommendations(affected, existing_tests)

        return jsonify({
            'status': 'success',
            'mode': 'new_patient',
            'patient_id': body.get('patient_id', 'NEW_PATIENT'),
            'timestamp': datetime.now().isoformat(),
            'affected_organs': affected,
            'recommendations': recs,
            'existing_tests_count': len(existing_tests)
        })

    return jsonify({'error': 'Provide either patient_id or tests array.'}), 400


# ──────────────────────────────────────────────────────────────────────────
# Utility endpoints
# ──────────────────────────────────────────────────────────────────────────

@app.route('/api/patients', methods=['GET'])
def list_patients():
    """List available existing patient IDs (paginated)."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    per_page = min(per_page, 200)  # cap
    start = (page - 1) * per_page
    end = start + per_page
    ids_page = existing_patient_ids[start:end]
    return jsonify({
        'total_patients': len(existing_patient_ids),
        'page': page,
        'per_page': per_page,
        'patient_ids': ids_page
    })


@app.route('/api/tests', methods=['GET'])
def list_available_tests():
    """List medical tests the system recognises for new patient predictions."""
    tests_info = []
    for name, info in MEDICAL_TESTS.items():
        tests_info.append({
            'test_name': name,
            'normal_range': f"{info['normal'][0]}-{info['normal'][1]}",
            'unit': info['unit'],
            'organ': info['organ']
        })
    return jsonify({
        'total_tests': len(tests_info),
        'tests': tests_info
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    load_system()
    app.run(host='0.0.0.0', port=5001, debug=False)
