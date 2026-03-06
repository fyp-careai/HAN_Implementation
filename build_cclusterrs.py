import pandas as pd
import json

df = pd.read_csv("data/dataset_careai_March/processed/test_reference_full_v2.csv")

df.columns = [c.lower().strip() for c in df.columns]

disease_col = "disease"
organ_col = "organ"

clusters = {}

for organ, group in df.groupby(organ_col):

    diseases = sorted(group[disease_col].dropna().unique().tolist())

    clusters[organ] = diseases

print(clusters)


cluster_to_diseases = {}

for i,(organ,diseases) in enumerate(clusters.items()):
    cluster_to_diseases[str(i)] = diseases

with open("data/disease_cluster_mapping_2.json","w") as f:
    json.dump({
        "cluster_to_diseases": cluster_to_diseases
    },f,indent=2)