"""
Feature Schema Utilities
========================
Saves and loads the feature schema used during HAN++ training so that
prediction-time feature vectors can be properly aligned — eliminating
the 182→203 dimension mismatch.

Usage:
    # After training:
    from HAN.feature_schema import save_schema, load_schema, align_features

    save_schema(data_loader, "models_saved/ruhunu_data_clustered/feature_schema.json")

    # At prediction time:
    schema = load_schema("models_saved/ruhunu_data_clustered/feature_schema.json")
    aligned = align_features(data_loader.patient_feats, data_loader, schema)
"""

import json
import numpy as np


def save_schema(data_loader, path: str) -> dict:
    """
    Extract and save feature schema from a fitted MedicalGraphData object.

    Feature vector layout (matches data.py:build_labels_and_features):
        [symptom_dev (S,) | organ_score (O,) | disease (D,)]

    Args:
        data_loader: fitted MedicalGraphData instance (after build_labels_and_features)
        path: filepath to save JSON schema

    Returns:
        schema dict
    """
    schema = {
        "symptoms":  data_loader.symptoms,
        "organs":    data_loader.organs,
        "diseases":  data_loader.diseases,
        "in_dim":    int(data_loader.patient_feats.shape[1]),
        "n_symptoms": len(data_loader.symptoms),
        "n_organs":   len(data_loader.organs),
        "n_diseases": len(data_loader.diseases),
    }

    with open(path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Feature schema saved: {path}")
    print(f"  Symptoms: {len(schema['symptoms'])}, "
          f"Organs: {len(schema['organs'])}, "
          f"Diseases: {len(schema['diseases'])} → in_dim={schema['in_dim']}")
    return schema


def load_schema(path: str) -> dict:
    """Load feature schema from JSON."""
    with open(path) as f:
        return json.load(f)


def align_features(patient_feats: np.ndarray, data_loader, schema: dict) -> np.ndarray:
    """
    Re-align a (P × current_dim) feature matrix to match the training schema.

    Strategy:
    1.  Split current feature vector into its three segments using the
        current data_loader dimensions.
    2.  For each segment (symptoms / organs / diseases), reindex columns
        to match the training schema order:
        - Present in both → copy value
        - In training schema but missing from current → fill 0.0
        - In current but not in training schema → drop
    3.  Concatenate the three aligned segments.

    Args:
        patient_feats: (P, current_dim) array from data_loader
        data_loader:   current MedicalGraphData (provides current entity lists)
        schema:        training schema loaded via load_schema()

    Returns:
        (P, in_dim_training) aligned feature array
    """
    P = patient_feats.shape[0]

    cur_S = len(data_loader.symptoms)
    cur_O = len(data_loader.organs)
    cur_D = len(data_loader.diseases)

    # Split current feature matrix into 3 segments
    feat_sym = patient_feats[:, :cur_S]
    feat_org = patient_feats[:, cur_S: cur_S + cur_O]
    feat_dis = patient_feats[:, cur_S + cur_O: cur_S + cur_O + cur_D]

    def _align_segment(feats, cur_names, train_names):
        """Reindex feats columns from cur_names order to train_names order."""
        name_to_cur_idx = {n: i for i, n in enumerate(cur_names)}
        out = np.zeros((P, len(train_names)), dtype=np.float32)
        for j, name in enumerate(train_names):
            if name in name_to_cur_idx:
                out[:, j] = feats[:, name_to_cur_idx[name]]
            # else: remains 0.0
        return out

    aligned_sym = _align_segment(feat_sym, data_loader.symptoms, schema["symptoms"])
    aligned_org = _align_segment(feat_org, data_loader.organs,   schema["organs"])
    aligned_dis = _align_segment(feat_dis, data_loader.diseases, schema["diseases"])

    aligned = np.concatenate([aligned_sym, aligned_org, aligned_dis], axis=1)

    expected_dim = schema["in_dim"]
    if aligned.shape[1] != expected_dim:
        raise ValueError(
            f"Alignment failed: got {aligned.shape[1]} features, "
            f"expected {expected_dim} from schema. "
            f"Check that symptom/organ/disease lists match training data."
        )

    return aligned


def print_schema_diff(data_loader, schema: dict):
    """Print a human-readable diff between training schema and current data."""
    print("\nFeature Schema Diff")
    print("=" * 60)

    def diff(name, train_list, cur_list):
        train_set = set(train_list)
        cur_set   = set(cur_list)
        missing   = sorted(train_set - cur_set)
        extra     = sorted(cur_set   - train_set)
        common    = len(train_set & cur_set)
        print(f"\n{name}:")
        print(f"  Training: {len(train_list):>4}  |  Current: {len(cur_list):>4}  "
              f"|  Common: {common:>4}")
        if missing:
            print(f"  Missing in current ({len(missing)}): {missing[:5]}"
                  + (" ..." if len(missing) > 5 else ""))
        if extra:
            print(f"  Extra in current  ({len(extra)}): {extra[:5]}"
                  + (" ..." if len(extra) > 5 else ""))

    diff("Symptoms / Tests", schema["symptoms"], data_loader.symptoms)
    diff("Organs",           schema["organs"],   data_loader.organs)
    diff("Diseases",         schema["diseases"], data_loader.diseases)
    print()
