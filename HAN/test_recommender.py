"""
Uncertainty-Guided Adaptive Test Recommendation for CareAI
===========================================================

Novel Contribution: Given per-disease MC Dropout uncertainty scores,
recommend the most informative diagnostic tests the patient has NOT yet taken.

Priority formula:
    priority(t, d) = σ_d × (1 − coverage_d)

where σ_d is the predictive uncertainty for disease d and coverage_d is
the fraction of disease-d-specific tests already present in the patient's record.

This creates an iterative refinement loop that mirrors clinical reasoning:
"I'm uncertain about CKD — order the tests I'm missing for CKD."

Reference for paper:
    Section IV-C "Uncertainty-Guided Adaptive Test Recommendation"
"""

import pandas as pd
import numpy as np
from collections import defaultdict


# ── Public API ────────────────────────────────────────────────────────────────

def load_test_reference(path_test_reference_v2: str) -> dict:
    """
    Load test_reference_full_v2.csv and return a nested dict.

    Args:
        path_test_reference_v2: path to test_reference_full_v2.csv
            Columns: test_name, organ, disease, min, max

    Returns:
        {disease_name: [{'test_name', 'organ', 'min', 'max'}, ...]}
    """
    df = pd.read_csv(path_test_reference_v2)

    ref = defaultdict(list)
    for _, row in df.iterrows():
        disease = row.get('disease')
        if pd.isna(disease) or not isinstance(disease, str):
            continue
        ref[disease].append({
            'test_name': str(row['test_name']).strip(),
            'organ':     str(row.get('organ', '')).strip(),
            'min':       row.get('min', None),
            'max':       row.get('max', None),
        })

    return dict(ref)


def detect_missing_tests(patient_test_names: list,
                         test_reference: dict,
                         disease: str) -> list:
    """
    Return tests relevant to `disease` that the patient does NOT yet have.

    Args:
        patient_test_names: list/set of test names the patient already has
        test_reference:     output of load_test_reference()
        disease:            disease name to check

    Returns:
        list of test-info dicts (test_name, organ, min, max) that are missing
    """
    patient_set = {t.strip().lower() for t in patient_test_names}
    disease_tests = test_reference.get(disease, [])
    return [t for t in disease_tests
            if t['test_name'].lower() not in patient_set]


def recommend_tests_for_disease(disease: str,
                                uncertainty: float,
                                patient_existing_tests: list,
                                test_reference: dict,
                                max_recommendations: int = 5) -> list:
    """
    For one disease with elevated uncertainty, return a ranked list of
    tests the patient should take.

    Ranking formula:
        priority_score = uncertainty × (1 − coverage_fraction)
        where coverage_fraction = tests_present / total_tests_for_disease

    All missing tests for the disease share the same disease-level priority score.
    Within the disease they are returned in reference order (clinical ordering).

    Args:
        disease:                  disease name (e.g. 'CKD')
        uncertainty:              MC Dropout std for this disease (0–1)
        patient_existing_tests:   tests the patient already has
        test_reference:           output of load_test_reference()
        max_recommendations:      cap on returned tests

    Returns:
        list of dicts: {test_name, disease, organ, normal_range, priority}
    """
    all_disease_tests = test_reference.get(disease, [])
    if not all_disease_tests:
        return []

    total = len(all_disease_tests)
    missing = detect_missing_tests(patient_existing_tests, test_reference, disease)

    if not missing:
        return []

    coverage_fraction = (total - len(missing)) / total
    priority = float(uncertainty * (1.0 - coverage_fraction))

    recs = []
    for t in missing[:max_recommendations]:
        lo = t['min']
        hi = t['max']
        if lo is not None and hi is not None and not (pd.isna(lo) or pd.isna(hi)):
            normal_range = f"{lo}–{hi}"
        elif lo is not None and not pd.isna(lo):
            normal_range = f"≥ {lo}"
        elif hi is not None and not pd.isna(hi):
            normal_range = f"≤ {hi}"
        else:
            normal_range = "N/A"

        recs.append({
            'test_name':    t['test_name'],
            'disease':      disease,
            'organ':        t['organ'],
            'normal_range': normal_range,
            'priority':     priority,
        })

    return recs


def recommend_all(disease_probs: dict,
                  disease_uncertainties: dict,
                  disease_order: list,
                  patient_existing_tests: list,
                  test_reference: dict,
                  uncertainty_threshold: float = 0.10,
                  prob_threshold_low: float = 0.30,
                  prob_threshold_high: float = 0.70,
                  opt_thresholds: dict = None,
                  max_per_disease: int = 5) -> dict:
    """
    Main entry point for the test recommendation engine.

    For each disease where:
      - uncertainty > uncertainty_threshold (model is unsure), OR
      - probability is in the ambiguous zone (prob_threshold_low–prob_threshold_high)
    → generate ranked test recommendations.

    Args:
        disease_probs:         {disease: probability (0–1)}
        disease_uncertainties: {disease: MC Dropout std (0–1)}
        disease_order:         list of disease names (canonical order)
        patient_existing_tests: list of test names patient already has
        test_reference:        output of load_test_reference()
        uncertainty_threshold: std above which we flag as uncertain (default 0.10)
        prob_threshold_low:    lower bound of ambiguous zone (default 0.30)
        prob_threshold_high:   upper bound of ambiguous zone (default 0.70)
        opt_thresholds:        {disease: threshold} for confirmed/ruled-out decision
                               (falls back to 0.5 if None)
        max_per_disease:       max tests recommended per disease

    Returns:
        {
          'confirmed_diseases':  [str] high-confidence positive predictions
          'ruled_out_diseases':  [str] high-confidence negative predictions
          'uncertain_diseases':  {disease: [rec_dicts]}  — needs more tests
          'summary_report':      str  human-readable text
        }
    """
    thresholds = opt_thresholds or {}

    confirmed = []
    ruled_out = []
    uncertain = {}

    for disease in disease_order:
        prob = float(disease_probs.get(disease, 0.0))
        std  = float(disease_uncertainties.get(disease, 0.0))
        thr  = float(thresholds.get(disease, 0.5))

        low_conf  = std > uncertainty_threshold
        ambiguous = prob_threshold_low <= prob <= prob_threshold_high

        if low_conf or ambiguous:
            recs = recommend_tests_for_disease(
                disease, std, patient_existing_tests,
                test_reference, max_recommendations=max_per_disease
            )
            uncertain[disease] = recs
        elif prob >= thr:
            confirmed.append(disease)
        else:
            ruled_out.append(disease)

    # ── Build human-readable report ──────────────────────────────────────────
    lines = ["=" * 60, "CareAI — Test Recommendation Report", "=" * 60]

    if confirmed:
        lines.append("\nCONFIRMED (high-confidence positive):")
        for d in confirmed:
            p = disease_probs[d]
            s = disease_uncertainties[d]
            lines.append(f"  [+] {d:<30}  prob={p:.3f}  σ={s:.4f}")

    if ruled_out:
        lines.append("\nRULED OUT (high-confidence negative):")
        for d in ruled_out:
            p = disease_probs[d]
            s = disease_uncertainties[d]
            lines.append(f"  [-] {d:<30}  prob={p:.3f}  σ={s:.4f}")

    if uncertain:
        lines.append("\nUNCERTAIN — Additional Tests Recommended:")
        for d, recs in uncertain.items():
            p = disease_probs.get(d, 0.0)
            s = disease_uncertainties.get(d, 0.0)
            lines.append(f"\n  [?] {d:<30}  prob={p:.3f}  σ={s:.4f}")
            if recs:
                lines.append(f"      Recommended tests ({len(recs)}):")
                for r in recs:
                    lines.append(
                        f"        • {r['test_name']:<40}  "
                        f"organ={r['organ']:<25}  "
                        f"normal={r['normal_range']}"
                    )
            else:
                lines.append("      (no additional tests available — patient has full panel)")

    if not confirmed and not ruled_out and not uncertain:
        lines.append("\nNo predictions available.")

    lines.append("\n" + "=" * 60)

    return {
        'confirmed_diseases': confirmed,
        'ruled_out_diseases':  ruled_out,
        'uncertain_diseases':  uncertain,
        'summary_report':      "\n".join(lines),
    }


def format_patient_json(patient_id,
                        disease_probs: dict,
                        disease_uncertainties: dict,
                        recommend_result: dict,
                        beta_weights: dict = None,
                        explanation: str = "") -> dict:
    """
    Format prediction output as structured JSON for the chatbot teammate.

    Args:
        patient_id:            patient identifier
        disease_probs:         {disease: probability}
        disease_uncertainties: {disease: std}
        recommend_result:      output of recommend_all()
        beta_weights:          {metapath: weight} (optional, from interpretability)
        explanation:           text explanation (optional)

    Returns:
        dict compatible with predict_new_patient.py output contract
    """
    uncertain_tests = {
        d: [{'test_name': r['test_name'], 'organ': r['organ'],
             'normal_range': r['normal_range']}
            for r in recs]
        for d, recs in recommend_result['uncertain_diseases'].items()
    }

    return {
        'patient_id':       str(patient_id),
        'predictions':      {d: round(float(p), 4) for d, p in disease_probs.items()},
        'uncertainties':    {d: round(float(s), 4) for d, s in disease_uncertainties.items()},
        'confirmed':        recommend_result['confirmed_diseases'],
        'uncertain':        list(recommend_result['uncertain_diseases'].keys()),
        'ruled_out':        recommend_result['ruled_out_diseases'],
        'recommended_tests': uncertain_tests,
        'meta_path_weights': {k: round(float(v), 4) for k, v in (beta_weights or {}).items()},
        'explanation':       explanation,
    }
