import torch
import numpy as np
import pandas as pd
from dataset_pyg import load_pyg_data
from models.han_pyg import HeteroHANModel


# ------------------------------------------------------------
# Build rule weights automatically from dataset
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Compute rule score based on abnormal labs
# ------------------------------------------------------------

def compute_rule_score(abnormal_features, rule_weights):

    lab_values = {f["test"]: f for f in abnormal_features}

    disease_rule_scores = {}

    for disease, rules in rule_weights.items():

        score = 0

        for test, weight in rules.items():

            if test in lab_values:

                ratio = lab_values[test]["ratio"]
                z = abs(lab_values[test]["z_ref"])

                abnormality = max(ratio, z/3)

                score += weight * abnormality

        disease_rule_scores[disease] = score

    return disease_rule_scores


# ------------------------------------------------------------
# Main inference
# ------------------------------------------------------------

def predict_new_patient(lab_results, model_path="han_link_pred_best.pt"):

    data, patient_to_idx, test_to_idx, organ_to_idx, disease_to_idx, norm_stats = load_pyg_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tests_df = pd.read_csv('data/HAN_data/unique_test_data_finalized.csv')
    test_ref = tests_df.set_index('test_name')[['lower_bound','upper_bound']].to_dict('index')

    val_mean = norm_stats['val_mean']
    val_std = norm_stats['val_std']
    time_mean = norm_stats['time_mean']
    time_std = norm_stats['time_std']

    abnormal_features = []

    # ------------------------------------------------------------
    # Add new patient
    # ------------------------------------------------------------

    new_p_idx = data['patient'].num_nodes
    data['patient'].num_nodes += 1

    data['patient'].x = torch.cat([
        data['patient'].x,
        torch.tensor([new_p_idx], dtype=torch.long)
    ])

    pt_src = []
    pt_dst = []
    pt_attr = []

    # ------------------------------------------------------------
    # Process lab tests
    # ------------------------------------------------------------

    for res in lab_results:

        t_name = res.get('test_name')

        if t_name in test_to_idx:

            pt_src.append(new_p_idx)
            pt_dst.append(test_to_idx[t_name])

            val = res.get('value', 0.0)
            time_val = res.get('time_since_test', 0.0)

            val_norm = (val - val_mean) / val_std
            time_norm = (time_val - time_mean) / time_std

            pt_attr.append([float(val_norm), float(time_norm)])

            ratio = 1.0
            z_ref = 0.0

            if t_name in test_ref:

                lower = test_ref[t_name]['lower_bound']
                upper = test_ref[t_name]['upper_bound']

                if upper > lower:

                    ref_mean = (lower + upper) / 2
                    ref_std = (upper - lower) / 4

                    z_ref = (val - ref_mean) / ref_std
                    ratio = val / upper if upper > 0 else 1.0

            abnormal_features.append({
                "test": t_name,
                "value": val,
                "ratio": ratio,
                "z_ref": z_ref
            })

    if not pt_src:
        print("No valid lab tests found.")
        return

    new_pt_src = torch.tensor(pt_src, dtype=torch.long)
    new_pt_dst = torch.tensor(pt_dst, dtype=torch.long)
    new_pt_attr = torch.tensor(pt_attr, dtype=torch.float)

    # ------------------------------------------------------------
    # Append edges
    # ------------------------------------------------------------

    data['patient','takes','labtest'].edge_index = torch.cat([
        data['patient','takes','labtest'].edge_index,
        torch.stack([new_pt_src,new_pt_dst])
    ], dim=1)

    data['patient','takes','labtest'].edge_attr = torch.cat([
        data['patient','takes','labtest'].edge_attr,
        new_pt_attr
    ], dim=0)

    data['labtest','rev_takes','patient'].edge_index = torch.cat([
        data['labtest','rev_takes','patient'].edge_index,
        torch.stack([new_pt_dst,new_pt_src])
    ], dim=1)

    data['labtest','rev_takes','patient'].edge_attr = torch.cat([
        data['labtest','rev_takes','patient'].edge_attr,
        new_pt_attr
    ], dim=0)

    data = data.to(device)

    # ------------------------------------------------------------
    # Message passing graph
    # ------------------------------------------------------------

    mp_edge_index_dict = {}

    for edge_type, e_idx in data.edge_index_dict.items():

        if edge_type[1] not in ['has','rev_has']:
            mp_edge_index_dict[edge_type] = e_idx

    edge_attr_dict = {
        ('patient','takes','labtest'): data['patient','takes','labtest'].edge_attr,
        ('labtest','rev_takes','patient'): data['labtest','rev_takes','patient'].edge_attr
    }

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    num_nodes_dict = {node_type:data[node_type].num_nodes for node_type in data.node_types}

    model = HeteroHANModel(
        metadata=data.metadata(),
        num_nodes_dict=num_nodes_dict,
        hidden_channels=64,
        edge_dim=32,
        out_channels=1,
        num_heads=4,
        num_layers=2
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)

    old_patient_emb = state_dict['node_embs.patient.weight']
    new_patient_emb = old_patient_emb.mean(dim=0, keepdim=True)

    state_dict['node_embs.patient.weight'] = torch.cat([
        old_patient_emb,
        new_patient_emb
    ], dim=0)

    model.load_state_dict(state_dict)
    model.eval()

    # ------------------------------------------------------------
    # GNN prediction
    # ------------------------------------------------------------

    all_diseases = list(disease_to_idx.keys())

    d_indices = torch.tensor(list(disease_to_idx.values()), dtype=torch.long, device=device)
    p_indices = torch.full((len(all_diseases),), new_p_idx, dtype=torch.long, device=device)

    eval_edge_index = torch.stack([p_indices, d_indices])

    with torch.no_grad():

        h_dict = model(data.x_dict, mp_edge_index_dict, edge_attr_dict)
        preds = model.predict_link(h_dict['patient'], h_dict['disease'], eval_edge_index)

    preds = preds.cpu().numpy()

    # ------------------------------------------------------------
    # Disease statistics
    # ------------------------------------------------------------

    patient_disease_df = pd.read_csv('data/HAN_data/patient_disease_ground_truth_long.csv')

    valid_pd_mask = (
        patient_disease_df['patient_id'].notna() &
        patient_disease_df['disease_name'].notna() &
        patient_disease_df['disease_score'].notna()
    )

    pd_df_valid = patient_disease_df[valid_pd_mask]

    disease_stats = pd_df_valid.groupby('disease_name')['disease_score'].agg(['mean','std']).fillna(1.0).to_dict('index')

    # ------------------------------------------------------------
    # Neuro-symbolic fusion
    # ------------------------------------------------------------

    rule_weights = build_rule_weights()
    rule_scores = compute_rule_score(abnormal_features, rule_weights)

    alpha = 0.4
    beta = 0.6

    results = []

    for d_name, gnn_score in zip(all_diseases, preds):

        d_mean = disease_stats.get(d_name, {}).get('mean', 0.0)
        d_std = disease_stats.get(d_name, {}).get('std', 1.0)

        gnn_severity = (gnn_score - d_mean) / d_std

        rule_score = rule_scores.get(d_name, 0)

        final_score = alpha * gnn_severity + beta * rule_score

        results.append((d_name, gnn_score, final_score))

    results.sort(key=lambda x: x[2], reverse=True)

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    print("\nAbnormal Lab Indicators")
    print("----------------------")

    for f in abnormal_features:
        print(f"{f['test']} : {f['value']} | ratio={f['ratio']:.2f} | z_ref={f['z_ref']:.2f}")

    print("\nFinal Neuro-Symbolic Disease Ranking")
    print("-----------------------------------")

    return results, abnormal_features


def recommend_tests_from_diseases(
    data,
    predictions,
    disease_to_idx,
    organ_to_idx,
    test_to_idx,
    top_k_diseases=4
):

    test_scores = {}

    idx_to_test = {v: k for k, v in test_to_idx.items()}

    # top diseases
    top_diseases = predictions[:top_k_diseases]

    organ_disease_edges = data['organ','causes','disease'].edge_index
    labtest_organ_edges = data['labtest','affects','organ'].edge_index

    for disease, gnn_score, final_score in top_diseases:

        disease_idx = disease_to_idx[disease]

        # organs connected to disease
        organ_nodes = organ_disease_edges[0][
            organ_disease_edges[1] == disease_idx
        ]

        for organ_idx in organ_nodes:

            # tests connected to organ
            tests = labtest_organ_edges[0][
                labtest_organ_edges[1] == organ_idx
            ]

            for t in tests:

                test_name = idx_to_test[int(t)]

                if test_name not in test_scores:
                    test_scores[test_name] = 0

                # accumulate disease importance
                test_scores[test_name] += final_score

    # sort tests
    ranked_tests = sorted(
        test_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_tests[:10]

# ------------------------------------------------------------
# Example run
# ------------------------------------------------------------

# if __name__ == '__main__':

#     sample_lab_data = [
#         {'test_name':'Serum Creatinine','value':7.5,'time_since_test':1},
#         {'test_name':'Blood Urea','value':180.0,'time_since_test':1},
#         {'test_name':'Estimated Glomerular Filtration Rate (eGFR)','value':10.0,'time_since_test':1},
#         {'test_name':'Serum Potassium','value':5.8,'time_since_test':1}
#     ]

#     print("Running inference for sample new patient data:")

#     for test in sample_lab_data:
#         print(f"  - {test['test_name']}: {test['value']}")

#     predict_new_patient(sample_lab_data)