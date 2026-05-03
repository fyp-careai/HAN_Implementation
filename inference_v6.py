#!/usr/bin/env python3
"""
inference_v6.py
================
Disease LINK-PREDICTION inference for a NEW (unseen) patient using the
trained HANPP_Disease v6 model (hanpp_disease_v6_PDP_POP.pt).

What "link prediction" means here
-----------------------------------
The model outputs sigmoid(disease_logits) ∈ [0,1] for each (patient, disease)
pair — exactly the link-existence probability.  We rank all diseases by their
link score, which tells you which diseases the patient is most likely linked to.

This mirrors the original inference.py logic (which called predict_link on a
HeteroHAN), but uses the new HANPP_Disease instead.

Pipeline
---------
1. Build the knowledge graph from the CareAI March dataset.
2. Load disease / organ prototypes (prototypes_v6.pkl).
3. Encode the new patient's lab results into a feature vector.
4. Run 50 MC-Dropout forward passes with prototype-based approximate neighbours
   (inductive_predict from HAN/inductive.py) to get per-disease link scores
   AND uncertainties.
5. Rank diseases by their link score (highest = most likely link).
6. Pass uncertain / ambiguous links to recommend_all (HAN/test_recommender.py)
   to suggest missing diagnostic tests.

Usage (standalone)
------------------
    python inference_v6.py

Or import and call predict_new_patient_v6(lab_results) from Flask / any runner.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# ── HAN package ───────────────────────────────────────────────────────────────
from HAN import MedicalGraphData, HANPP_Disease
from HAN.inductive import inductive_predict
from HAN.test_recommender import load_test_reference, recommend_all, format_patient_json

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH           = 'models_saved/careai_march/hanpp_disease_v6_PDP_POP.pt'
PROTOTYPE_PATH       = 'models_saved/careai_march/prototypes_v6.pkl'
ORGAN_PROTOTYPE_PATH = 'models_saved/careai_march/organ_prototypes_v6.pkl'

RECORDS_PATH         = 'data/HAN_data/merged_coop_ruhunu_patient_data.csv'
SYMPTOM_PATH         = 'data/HAN_data/patient_disease_ground_truth_long.csv'
TEST_REFERENCE_PATH  = 'data/HAN_data/unique_test_data_finalized.csv'
TEST_INFO_PATH       = 'data/HAN_data/unique_test_data_finalized.csv'
ORGAN_PATH           = 'data/HAN_data/test_reference_full_v2.csv'

# Metapaths the v6 model was trained with (order matters — matches checkpoint)
METAPATH_NAMES = ['P-D-P', 'P-O-P']

# Inference settings
MC_SAMPLES           = 50
TOP_K_DISEASES       = 3
MAX_PER_DISEASE      = 15
UNCERTAINTY_THRESHOLD = 0.10
PROB_LOW             = 0.30
PROB_HIGH            = 0.70

# Neuro-symbolic fusion weights  (gnn_score * ALPHA + rule_score * BETA)
# Tuned via experiment_fusion_params.py grid search
ALPHA = 0.3
BETA  = 0.7
DELTA = 0.4   # Rule-score normalization steepness: norm = rule / (DELTA + rule)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_patient(lab_results: list, data_loader: MedicalGraphData) -> np.ndarray:
    """
    Convert raw lab results into the [S + O] feature vector expected by the model.

    Parameters
    ----------
    lab_results : list of {'test_name': str, 'value': float}
    data_loader : MedicalGraphData (already .load_data() + .build_labels_and_features())

    Returns
    -------
    np.ndarray  shape [in_dim]   (in_dim = S + O = 120 for this dataset)
    """
    S = data_loader.S
    O = data_loader.O

    symptom_dev   = np.zeros(S, dtype=np.float32)
    symptom_count = np.zeros(S, dtype=np.int32)
    organ_score   = np.zeros(O, dtype=np.float32)

    for res in lab_results:
        t_name = str(res.get('test_name', '')).strip()
        val    = res.get('value')

        if t_name not in data_loader.symptom_map:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue

        sidx = data_loader.symptom_map[t_name]
        meta = data_loader.symptom_meta.get(t_name, {})
        low  = meta.get('normal_low')
        high = meta.get('normal_high')

        # Deviation (same formula as MedicalGraphData.build_labels_and_features)
        dev = 0.0
        if low is not None and high is not None and high > low:
            mid = (low + high) / 2.0
            rng = (high - low) / 2.0
            if rng > 0:
                dev = (val - mid) / (rng * 2)

        symptom_dev[sidx]   += dev
        symptom_count[sidx] += 1

        # Organ damage score
        organ_name = meta.get('organ')
        if organ_name in data_loader.organ_map and low is not None and high is not None:
            oidx = data_loader.organ_map[organ_name]
            if val < low:
                deficit = (low - val) / (low if low != 0 else 1.0)
                score   = float(np.clip(deficit, 0.0, 1.0))
            elif val > high:
                excess = (val - high) / (high if high != 0 else 1.0)
                score  = float(np.clip(excess, 0.0, 1.0))
            else:
                score = 0.0
            organ_score[oidx] = max(organ_score[oidx], score)

    # Normalise (same as training)
    mask = symptom_count > 0
    symptom_dev[mask] = symptom_dev[mask] / symptom_count[mask]
    symptom_dev = np.clip(symptom_dev, -3.0, 3.0) / 3.0

    return np.concatenate([symptom_dev, organ_score])   # [S+O] = [120]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Neuro-symbolic rule helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_rule_weights():

    df = pd.read_csv('data/HAN_data/patient_disease_ground_truth_long.csv')

    rule_weights = {}

    for disease, group in df.groupby("disease_name"):

        tests = {}

        for row in group["matched_tests"].dropna():

            for t in row.split(";"):
                t = t.strip()

                if t not in tests:
                    tests[t] = 0

                tests[t] += 1

        total = sum(tests.values())

        if total == 0:
            continue

        for k in tests:
            tests[k] = tests[k] / total

        rule_weights[disease] = tests

    return rule_weights


# def extract_abnormal_features(lab_results: list,
#                               test_info_path: str = TEST_INFO_PATH) -> list:
#     """
#     Compute ratio and z_ref for each submitted test relative to reference bounds.

#     Returns
#     -------
#     list of dicts: {test, value, ratio, z_ref}
#         Only tests that have reference bounds are included.
#     """
#     tests_df = pd.read_csv(test_info_path)
#     tests_df.columns = tests_df.columns.str.strip()

#     # Build lookup {test_name: {lower_bound, upper_bound}}
#     ref = {}
#     for _, row in tests_df.iterrows():
#         name = str(row.get('test_name', '')).strip()
#         try:
#             lo = float(row.get('lower_bound', float('nan')))
#             hi = float(row.get('upper_bound', float('nan')))
#         except (TypeError, ValueError):
#             continue
#         if name and not (pd.isna(lo) and pd.isna(hi)):
#             ref[name] = {'lower': lo, 'upper': hi}

#     abnormal = []
#     for res in lab_results:
#         t_name = str(res.get('test_name', '')).strip()
#         val = res.get('value')
#         try:
#             val = float(val)
#         except (TypeError, ValueError):
#             continue

#         bounds = ref.get(t_name)
#         if bounds is None:
#             continue

#         lo, hi = bounds['lower'], bounds['upper']
#         ratio  = 1.0
#         z_ref  = 0.0

#         if not pd.isna(lo) and not pd.isna(hi) and hi > lo:
#             ref_mean = (lo + hi) / 2.0
#             ref_std  = (hi - lo) / 4.0
#             z_ref    = (val - ref_mean) / ref_std if ref_std > 0 else 0.0
#             ratio    = val / hi if hi > 0 else 1.0
#         elif not pd.isna(hi) and hi > 0:
#             ratio = val / hi

#         abnormal.append({
#             'test':  t_name,
#             'value': val,
#             'ratio': ratio,
#             'z_ref': z_ref,
#         })

#     return abnormal

def extract_abnormal_features(lab_results: list,
                              test_info_path: str = TEST_INFO_PATH) -> list:
    """
    Compute ratio and z_ref for each submitted test relative to reference bounds.

    Returns
    -------
    list of dicts: {test, value, ratio, z_ref}
        Only tests that have reference bounds are included.
    """
    tests_df = pd.read_csv(test_info_path)
    tests_df.columns = tests_df.columns.str.strip()

    # Build lookup {test_name: {lower_bound, upper_bound}}
    ref = {}
    for _, row in tests_df.iterrows():
        name = str(row.get('test_name', '')).strip()
        try:
            lo = float(row.get('lower_bound', float('nan')))
            hi = float(row.get('upper_bound', float('nan')))
        except (TypeError, ValueError):
            continue
        if name and not (pd.isna(lo) and pd.isna(hi)):
            ref[name] = {'lower': lo, 'upper': hi}

    abnormal = []
    for res in lab_results:
        t_name = str(res.get('test_name', '')).strip()
        val = res.get('value')
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue

        bounds = ref.get(t_name)
        if bounds is None:
            continue

        lo, hi = bounds['lower'], bounds['upper']
        ratio  = 1.0
        z_ref  = 0.0

        if not pd.isna(lo) and not pd.isna(hi) and hi > lo:
            ref_mean = (lo + hi) / 2.0
            ref_std  = (hi - lo) / 4.0
            z_ref    = (val - ref_mean) / ref_std if ref_std > 0 else 0.0
            ratio    = val / hi if hi > 0 else 1.0
        elif not pd.isna(hi) and hi > 0:
            ratio = val / hi

        # Only include tests that fall outside the normal reference range
        is_abnormal = False
        if not pd.isna(lo) and val < lo:
            is_abnormal = True
        if not pd.isna(hi) and val > hi:
            is_abnormal = True

        if not is_abnormal:
            continue

        abnormal.append({
            'test':  t_name,
            'value': val,
            'ratio': ratio,
            'z_ref': z_ref,
        })

    return abnormal


def compute_rule_score(abnormal_features: list, rule_weights: dict) -> dict:
    """
    Compute a rule-based disease score from abnormal lab indicators.

    For each disease, the score accumulates weight * abnormality for every
    test that is (a) in the disease rule set AND (b) present in the patient's
    submitted labs.

    abnormality = max(ratio, |z_ref| / 3)

    Returns
    -------
    {disease_name: rule_score}
    """
    lab_index = {f['test']: f for f in abnormal_features}

    scores = {}
    for disease, rules in rule_weights.items():
        score = 0.0
        for test, weight in rules.items():
            if test in lab_index:
                ratio = lab_index[test]['ratio']
                z     = abs(lab_index[test]['z_ref'])
                abnormality = max(ratio, z / 3.0)
                score += weight * abnormality
        scores[disease] = score

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> HANPP_Disease:
    """
    Reconstruct HANPP_Disease exactly from the saved checkpoint.

    Architecture inferred from checkpoint tensor shapes:
      project.weight       [256, 120]  → in_dim=120, hidden_dim=256
      out_proj.weight      [128, 256]  → out_dim=128
      disease_classifier   [9, 128]    → num_diseases=9
      node_atts count = 2             → P-D-P, P-O-P
    """
    ck = torch.load(checkpoint_path, map_location='cpu')

    in_dim       = ck['project.weight'].shape[1]        # 120
    hidden_dim   = ck['project.weight'].shape[0]        # 256
    out_dim      = ck['out_proj.weight'].shape[0]       # 128
    num_diseases = ck['disease_classifier.weight'].shape[0]  # 9
    num_atts     = sum(1 for k in ck if k.startswith('node_atts.') and k.endswith('.a_l'))
    metapaths    = METAPATH_NAMES[:num_atts]

    print(f"  Architecture: in={in_dim}, hidden={hidden_dim}, out={out_dim}, "
          f"diseases={num_diseases}, metapaths={metapaths}")

    model = HANPP_Disease(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        metapath_names=metapaths,
        num_heads=4,
        num_diseases=num_diseases,
        dropout=0.3,
    )
    model.load_state_dict(ck)
    model.to(DEVICE)
    model.eval()
    return model



# ------------------------------------------------------------
# Severity Level Determination
# ------------------------------------------------------------

def determine_severity(final_score, abnormal_features):

    max_z = 0

    for f in abnormal_features:
        max_z = max(max_z, abs(f["z_ref"]))

    # CRITICAL conditions
    if final_score >= 1.5 or max_z >= 3:
        return "CRITICAL", "c"

    # WARNING conditions
    elif final_score >= 1.0 or max_z >= 2:
        return "WARNING", "w"

    # INFO conditions
    elif final_score >= 0.5:
        return "INFO", "i"

    else:
        return None, None



# ------------------------------------------------------------
# Early Warning Message Generator with Severity
# ------------------------------------------------------------

def generate_early_warning(predictions, abnormal_features, top_k=1):

    warnings = []

    top_diseases = predictions[:top_k]

    abnormal_list = []

    for f in abnormal_features:

        if abs(f["z_ref"]) >= 2 or f["ratio"] >= 1.3:

            if f["z_ref"] > 0:
                direction = "elevated"
            else:
                direction = "decreased"

            abnormal_list.append(
                f"{f['test']} {direction}"
            )

    for d in top_diseases:

        disease_name = d["disease"]
        score = d["final_score"]

        severity, emoji = determine_severity(
            score,
            abnormal_features
        )

        if severity is None:
            continue

        msg = f"{emoji} Early Warning ({severity})\n\n"

        msg += f"High Risk of {disease_name} detected.\n\n"

        if abnormal_list:

            msg += "Abnormal Findings:\n"

            for item in abnormal_list[:5]:
                msg += f"- {item}\n"

        msg += "\nImmediate clinical evaluation recommended."

        warnings.append({
            "severity": severity,
            "message": msg
        })

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_new_patient_v6(
    lab_results: list,
    model_path: str = MODEL_PATH,
    top_n: int = None,
) -> dict:
    """
    Predict disease LINK SCORES for a new patient.

    Parameters
    ----------
    lab_results : list of dicts with at minimum:
        - 'test_name': str   (must match names in the training data)
        - 'value':     float
    model_path  : path to .pt checkpoint
    top_n       : if set, only return the top-N predicted disease links

    Returns
    -------
    dict with keys:
      'patient_id'         : str
      'disease_link_scores': list of (disease, link_score, uncertainty), sorted desc by score
      'predictions'        : {disease: link_score}  (all diseases)
      'uncertainties'      : {disease: mc_std}
      'confirmed'          : [disease]  high-confidence links
      'uncertain'          : [disease]  ambiguous — need more tests
      'ruled_out'          : [disease]  high-confidence non-links
      'recommended_tests'  : {disease: [{test_name, organ, normal_range}]}
      'method'             : 'prototype_inductive' | 'mlp_fallback'
      'neighbor_count'     : int
    """
    print("\n" + "="*70)
    print("CareAI v6 — Disease Link Prediction for New Patient")
    print("="*70)

    # ── Step 1: Knowledge graph ───────────────────────────────────────────────
    print("\n[1/5] Building knowledge graph ...")
    data_loader = MedicalGraphData(
        path_records=RECORDS_PATH,
        path_symptom=ORGAN_PATH,
        symptom_freq_threshold=0.08,
        prune_per_patient=300,
        nnz_threshold=80_000_000,
        seed=42,
    )
    data_loader.load_data()
    data_loader.build_labels_and_features()
    data_loader.build_adjacency_matrices()

    # ── Step 2: Load model ────────────────────────────────────────────────────
    print("\n[2/5] Loading model ...")
    model = load_model(model_path)

    # ── Step 3: Load prototypes ───────────────────────────────────────────────
    print("\n[3/5] Loading disease prototypes ...")
    if not os.path.exists(PROTOTYPE_PATH):
        raise FileNotFoundError(
            f"Prototype file not found: {PROTOTYPE_PATH}\n"
            "Run the training script's save-prototypes step first."
        )
    with open(PROTOTYPE_PATH, 'rb') as f:
        proto_data = pickle.load(f)

    prototypes  = proto_data['prototypes']          # {disease: tensor[out_dim]}
    z_train     = proto_data['patient_embeddings']  # np [N, out_dim]
    labels_np   = proto_data['labels_np']           # np [N, num_diseases]

    # Use the disease order that was active when the prototypes were built.
    # This guarantees alignment between prototype dict keys, labels_np columns,
    # and the model's disease_classifier outputs.
    disease_order = proto_data['disease_order']
    print(f"  Disease order from prototypes ({len(disease_order)} diseases): {disease_order}")

    organ_prototypes    = None
    patient_organ_score = None
    organ_map           = None

    if os.path.exists(ORGAN_PROTOTYPE_PATH):
        with open(ORGAN_PROTOTYPE_PATH, 'rb') as f:
            organ_data = pickle.load(f)
        organ_prototypes    = organ_data.get('organ_prototypes')
        patient_organ_score = data_loader.patient_organ_score
        organ_map           = {i: name for i, name in enumerate(data_loader.organs)}
        print("  Organ prototypes loaded (P-O-P neighbours enabled).")
    else:
        print(f"  [Warning] {ORGAN_PROTOTYPE_PATH} not found — P-O-P will use empty neighbours.")

    # ── Step 4: Encode patient ────────────────────────────────────────────────
    print("\n[4/5] Encoding patient lab results ...")
    new_feats = encode_patient(lab_results, data_loader)
    print(f"  Feature vector: {new_feats.shape}  "
          f"(matched {int(np.any(new_feats != 0))} tests to graph vocabulary)")

    # Align with model in_dim (pad / truncate if graph vocab changed)
    expected = model.project.weight.shape[1]
    if new_feats.shape[0] != expected:
        aligned = np.zeros(expected, dtype=np.float32)
        n = min(new_feats.shape[0], expected)
        aligned[:n] = new_feats[:n]
        new_feats = aligned
        print(f"  [Warning] New-patient feature dim mismatch: got {n}, expected {expected}. Padded.")

    # Also align training feature matrix to model in_dim.
    # inductive_predict does np.vstack([new_feats, training_feats]) — both must be
    # the same width. The current data loader may use a different symptom filter than
    # training, producing a different number of features.
    feats_np = data_loader.patient_feats   # [N_train, S+O]
    if feats_np.shape[1] != expected:
        aligned_train = np.zeros((feats_np.shape[0], expected), dtype=np.float32)
        n = min(feats_np.shape[1], expected)
        aligned_train[:, :n] = feats_np[:, :n]
        feats_np = aligned_train
        print(f"  [Warning] Training feature matrix padded from {n} → {expected} dims.")

    n_matched = int(np.count_nonzero(new_feats))
    if n_matched == 0:
        print("  [Warning] No submitted tests matched the graph vocabulary. "
              "Predictions will rely on prototype similarity only.")
    else:
        print(f"  {n_matched} non-zero features encoded from submitted tests.")

    opt_thresholds = {d: 0.5 for d in disease_order}

    # ── Step 5: Inductive MC-Dropout link scoring ─────────────────────────────
    print(f"\n[5/5] Running {MC_SAMPLES}-sample MC-Dropout link prediction ...")
    result = inductive_predict(
        model=model,
        new_patient_feats=new_feats,
        prototypes=prototypes,
        z_train=feats_np,
        labels_np=labels_np,
        disease_order=disease_order,
        opt_thresholds=opt_thresholds,
        device=DEVICE,
        n_mc_samples=MC_SAMPLES,
        organ_prototypes=organ_prototypes,
        patient_organ_score=patient_organ_score,
        organ_map=organ_map,
    )

    disease_probs = result['disease_probs']   # {disease: link_score ∈ [0,1]}
    disease_stds  = result['disease_stds']    # {disease: mc_std}
    method        = result['method']
    n_neighbors   = result['neighbor_count']

    # ── Neuro-symbolic fusion ─────────────────────────────────────────────────
    # Build rule weights from knowledge graph and compute per-disease rule score
    rule_weights  = build_rule_weights()
    abnormal_feats = extract_abnormal_features(lab_results, TEST_INFO_PATH)
    rule_scores   = compute_rule_score(abnormal_feats, rule_weights)

    # Print abnormal indicators
    if abnormal_feats:
        print("\nAbnormal Lab Indicators")
        print("-"*50)
        for f in abnormal_feats:
            print(f"  {f['test']:<45} value={f['value']:.2f}  "
                  f"ratio={f['ratio']:.2f}  z_ref={f['z_ref']:.2f}")

    # Fuse GNN link score with rule score
    # GNN score is already in [0, 1] (sigmoid output).
    # Rule score is in [0, ∞) — normalise it to [0, 1) using saturation:
    #     norm_rule = rule / (1 + rule)
    # This gives:  rule=0 → 0,  rule=1 → 0.5,  rule=3 → 0.75,  rule→∞ → 1
    # After normalisation:  final_score = ALPHA*gnn + BETA*norm_rule  ∈ [0, 1]
    # This is directly interpretable as a probability — no sigmoid needed.
    #
    # If rule_score == 0 (no abnormal lab evidence), zero out the final score
    # entirely — GNN-only predictions without rule-based backing are omitted.
    final_scores = {}
    for d in disease_order:
        gnn  = float(disease_probs.get(d, 0.0))
        rule = float(rule_scores.get(d, 0.0))
        if rule == 0.0:
            final_scores[d] = 0.0
        else:
            norm_rule = rule / (DELTA + rule)  # Rational normalization [0,1]
            final_scores[d] = ALPHA * gnn + BETA * norm_rule

    # ── Ranked by final (fused) score ─────────────────────────────────────────
    ranked_links = sorted(
        [(d, disease_probs[d], disease_stds[d], final_scores[d]) for d in disease_order],
        key=lambda x: x[3],   # sort by final_score
        reverse=True
    )
    if top_n:
        ranked_links = ranked_links[:top_n]

    # ── Test recommendations (based on GNN probs for threshold logic) ─────────
    patient_test_names = [str(r.get('test_name', '')).strip() for r in lab_results]
    # ORGAN_PATH (test_reference_full_v2.csv) has the required 'disease' column;
    # TEST_REFERENCE_PATH (unique_test_data_finalized.csv) does not.
    test_reference = load_test_reference(ORGAN_PATH)

    recommend_result = recommend_all(
        disease_probs=final_scores,
        disease_uncertainties=disease_stds,
        disease_order=disease_order,
        patient_existing_tests=patient_test_names,
        test_reference=test_reference,
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
        prob_threshold_low=PROB_LOW,
        prob_threshold_high=PROB_HIGH,
        opt_thresholds=opt_thresholds,
    )

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"DISEASE LINK SCORES  (final = {ALPHA}×GNN + {BETA}×rule)")
    print("="*70)
    print(f"\n{'Rank':<5} {'Disease':<35} {'GNN':>6}  {'Rule':>6}  {'Final':>7}  {'Unc':>6}  Status")
    print("-"*75)
    for rank, (disease, gnn_s, std, final_s) in enumerate(ranked_links, 1):
        rule_s = rule_scores.get(disease, 0.0)
        if disease in recommend_result['confirmed_diseases']:
            status = 'CONFIRMED'
        elif disease in recommend_result['ruled_out_diseases']:
            status = 'RULED OUT'
        else:
            status = 'UNCERTAIN'
        print(f"#{rank:<4} {disease:<35} {gnn_s:>6.3f}  {rule_s:>6.3f}  {final_s:>7.4f}  {std:>6.4f}  {status}")

    print()
    print(recommend_result['summary_report'])

    # ── Return structured result ──────────────────────────────────────────────
    patient_id = f"NEW_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    uncertain_tests = {
        d: [{'test_name': r['test_name'], 'organ': r['organ'], 'normal_range': r['normal_range']}
            for r in recs]
        for d, recs in recommend_result['uncertain_diseases'].items()
    }

    confirmed_tests = {
        d: [{'test_name': r['test_name'], 'organ': r['organ'], 'normal_range': r['normal_range']}
            for r in recs]
        for d, recs in recommend_result['confirmed_disease_tests'].items()
    }

    return {
        'patient_id':          patient_id,
        'disease_link_scores': [
            {
                'disease':     d,
                'gnn_score':   round(float(gnn_s), 4),
                'rule_score':  round(float(rule_scores.get(d, 0.0)), 4),
                'final_score': round(float(fs), 4),
                'uncertainty': round(float(u), 4),
            }
            for d, gnn_s, u, fs in ranked_links
        ],
        'predictions':              {d: round(float(p), 4) for d, p in disease_probs.items()},
        'uncertainties':            {d: round(float(s), 4) for d, s in disease_stds.items()},
        'rule_scores':              {d: round(float(v), 4) for d, v in rule_scores.items()},
        'final_scores':             {d: round(float(v), 4) for d, v in final_scores.items()},
        'confirmed':                recommend_result['confirmed_diseases'],
        'uncertain':                list(recommend_result['uncertain_diseases'].keys()),
        'ruled_out':                recommend_result['ruled_out_diseases'],
        'recommended_tests':        {**uncertain_tests, **confirmed_tests},
        'method':                   method,
        'neighbor_count':           n_neighbors,
        'fusion':                   {'alpha': ALPHA, 'beta': BETA},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Example run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Sample: Chronic Kidney Disease-like panel
    sample_lab_results = [
        {'test_name': 'Serum Creatinine',                            'value': 7.5},
        {'test_name': 'Serum Urea',                                  'value': 85.0},
        {'test_name': 'Estimated Glomerular Filtration Rate (eGFR)', 'value': 10.0},
        {'test_name': 'Serum Potassium',                             'value': 5.8},
        {'test_name': 'Hemoglobin (Hb)',                             'value': 9.0},
        {'test_name': 'Urine Albumin',                               'value': 350.0},
    ]

    print("Lab tests submitted:")
    for t in sample_lab_results:
        print(f"  {t['test_name']}: {t['value']}")

    output = predict_new_patient_v6(sample_lab_results)

    print("\n" + "="*70)
    print("STRUCTURED OUTPUT  (JSON)")
    print("="*70)
    print(json.dumps(output, indent=2))
