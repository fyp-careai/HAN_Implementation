#!/usr/bin/env python3
"""
experiment_fusion_params.py
============================
Grid-search over ALPHA, BETA, and DELTA to find the best neuro-symbolic
fusion parameters for the CareAI HAN model.

Formula:
    norm_rule   = rule_score / (DELTA + rule_score)
    final_score = ALPHA * gnn_score + BETA * norm_rule

Constraint: ALPHA + BETA = 1.0 (so we only sweep ALPHA; BETA = 1 - ALPHA)

The script runs inference on a set of sample patients and evaluates how
well the final scores separate "true positive" diseases (those with high
rule evidence) from "noise" diseases (rule score ≈ 0).

Usage:
    python experiment_fusion_params.py
"""

import json
import itertools
import numpy as np
import pandas as pd
from inference_v6 import (
    predict_new_patient_v6,
    extract_abnormal_features,
    build_rule_weights,
    compute_rule_score,
    TEST_INFO_PATH,
)
import inference_v6 as inf  # so we can monkey-patch ALPHA, BETA, DELTA

# ─────────────────────────────────────────────────────────────────────────────
# 1. Define test cases (patients with known expected diseases)
# ─────────────────────────────────────────────────────────────────────────────

TEST_PATIENTS = [
    {
        "name": "CKD Patient",
        "expected_top": "Chronic Kidney Disease (CKD)",
        "labs": [
            {"test_name": "Serum Creatinine",                            "value": 7.5},
            {"test_name": "Serum Urea",                                  "value": 85.0},
            {"test_name": "Estimated Glomerular Filtration Rate (eGFR)", "value": 10.0},
            {"test_name": "Serum Potassium",                             "value": 5.8},
            {"test_name": "Hemoglobin (Hb)",                             "value": 9.0},
            {"test_name": "Urine Albumin",                               "value": 350.0},
        ],
    },
    {
        "name": "Diabetes Patient",
        "expected_top": "Diabetes_Mellitus",
        "labs": [
            {"test_name": "Fasting Blood Sugar (FBS)",  "value": 210.0},
            {"test_name": "HbA1c",                      "value": 9.5},
            {"test_name": "Random Blood Sugar (RBS)",   "value": 320.0},
        ],
    },
    {
        "name": "Liver Disease Patient",
        "expected_top": "Liver_Disease",
        "labs": [
            {"test_name": "Serum Glutamic Oxaloacetic Transaminase (SGOT/AST)", "value": 180.0},
            {"test_name": "Serum Glutamic Pyruvic Transaminase (SGPT/ALT)",     "value": 220.0},
            {"test_name": "Total Bilirubin",                                     "value": 4.5},
            {"test_name": "Alkaline Phosphatase (ALP)",                          "value": 350.0},
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parameter grid
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6]
DELTA_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

# BETA is always 1 - ALPHA (they must sum to 1 for a valid probability)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scoring function  (does NOT re-run GNN — only re-fuses cached scores)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_params(alpha, delta, gnn_probs, rule_scores, disease_order, expected_disease):
    """
    Given cached GNN probs and rule scores, compute final scores with the
    given (alpha, delta) and return evaluation metrics.
    """
    beta = 1.0 - alpha
    final = {}
    for d in disease_order:
        gnn  = float(gnn_probs.get(d, 0.0))
        rule = float(rule_scores.get(d, 0.0))
        if rule == 0.0:
            final[d] = 0.0
        else:
            norm_rule = rule / (delta + rule)
            final[d] = alpha * gnn + beta * norm_rule

    # Metrics
    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    top_disease = ranked[0][0]
    top_score   = ranked[0][1]

    # Score of the expected disease
    expected_score = final.get(expected_disease, 0.0)

    # Rank of the expected disease (1-indexed)
    expected_rank = next(
        (i + 1 for i, (d, _) in enumerate(ranked) if d == expected_disease), len(ranked)
    )

    # Gap between top score and runner-up (separation quality)
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    separation = top_score - runner_up_score

    # Count how many diseases have non-zero final score
    non_zero = sum(1 for v in final.values() if v > 0)

    return {
        "top_disease":     top_disease,
        "top_score":       round(top_score, 4),
        "expected_rank":   expected_rank,
        "expected_score":  round(expected_score, 4),
        "separation":      round(separation, 4),
        "non_zero_count":  non_zero,
        "correct_top":     top_disease == expected_disease,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Fusion Parameter Experiment:  ALPHA, BETA, DELTA Grid Search")
    print("=" * 70)

    # ── Step 1: Run GNN inference ONCE per patient (expensive) ────────────────
    print("\n[Step 1] Running GNN inference for all test patients (this is slow, only done once)...\n")

    patient_cache = []
    for patient in TEST_PATIENTS:
        print(f"  ▶ Inferring: {patient['name']} ...")
        output = predict_new_patient_v6(patient["labs"])
        gnn_probs   = output["predictions"]          # {disease: prob}
        rule_scores = output["rule_scores"]           # {disease: raw_score}
        disease_order = list(gnn_probs.keys())

        patient_cache.append({
            "name":             patient["name"],
            "expected_top":     patient["expected_top"],
            "gnn_probs":        gnn_probs,
            "rule_scores":      rule_scores,
            "disease_order":    disease_order,
        })
        print(f"    GNN probs:   { {d: round(v,3) for d,v in gnn_probs.items()} }")
        print(f"    Rule scores: { {d: round(v,3) for d,v in rule_scores.items()} }")
        print()

    # ── Step 2: Grid search (fast — just arithmetic) ──────────────────────────
    print("\n[Step 2] Running grid search over ALPHA × DELTA ...\n")

    results = []
    for alpha, delta in itertools.product(ALPHA_VALUES, DELTA_VALUES):
        beta = 1.0 - alpha
        combo_correct = 0
        combo_avg_rank = 0.0
        combo_avg_sep = 0.0
        combo_avg_top_score = 0.0

        per_patient = []
        for pc in patient_cache:
            ev = evaluate_params(
                alpha=alpha,
                delta=delta,
                gnn_probs=pc["gnn_probs"],
                rule_scores=pc["rule_scores"],
                disease_order=pc["disease_order"],
                expected_disease=pc["expected_top"],
            )
            per_patient.append(ev)
            combo_correct      += int(ev["correct_top"])
            combo_avg_rank     += ev["expected_rank"]
            combo_avg_sep      += ev["separation"]
            combo_avg_top_score += ev["top_score"]

        n = len(patient_cache)
        results.append({
            "alpha":          alpha,
            "beta":           round(beta, 2),
            "delta":          delta,
            "accuracy":       combo_correct / n,
            "avg_rank":       round(combo_avg_rank / n, 2),
            "avg_separation": round(combo_avg_sep / n, 4),
            "avg_top_score":  round(combo_avg_top_score / n, 4),
            "per_patient":    per_patient,
        })

    # ── Step 3: Print results table ───────────────────────────────────────────
    print(f"\n{'ALPHA':>6} {'BETA':>6} {'DELTA':>6} │ {'Acc':>5} {'AvgRank':>8} {'AvgSep':>8} {'AvgTop':>8}")
    print("─" * 60)

    best = None
    for r in sorted(results, key=lambda x: (-x["accuracy"], x["avg_rank"], -x["avg_separation"])):
        flag = ""
        if best is None:
            best = r
            flag = "  ◀ BEST"
        print(f"{r['alpha']:>6.2f} {r['beta']:>6.2f} {r['delta']:>6.2f} │ "
              f"{r['accuracy']:>5.0%} {r['avg_rank']:>8.2f} {r['avg_separation']:>8.4f} {r['avg_top_score']:>8.4f}{flag}")

    # ── Step 4: Print best result detail ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"BEST PARAMETERS:  ALPHA={best['alpha']},  BETA={best['beta']},  DELTA={best['delta']}")
    print(f"{'=' * 60}")
    for i, pc in enumerate(patient_cache):
        ev = best["per_patient"][i]
        status = "✅" if ev["correct_top"] else "❌"
        print(f"  {status} {pc['name']}")
        print(f"      Expected: {pc['expected_top']}")
        print(f"      Got:      {ev['top_disease']}  (score={ev['top_score']}, rank={ev['expected_rank']})")
        print(f"      Separation from runner-up: {ev['separation']}")

    # ── Step 5: Save full results to JSON ─────────────────────────────────────
    output_path = "experiment_fusion_results.json"
    # Strip per_patient detail for cleaner JSON
    summary = [{k: v for k, v in r.items() if k != "per_patient"} for r in results]
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to: {output_path}")

    print(f"\n{'─' * 60}")
    print(f"To apply the best params, update inference_v6.py:")
    print(f"  ALPHA = {best['alpha']}")
    print(f"  BETA  = {best['beta']}")
    print(f"  DELTA = {best['delta']}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
