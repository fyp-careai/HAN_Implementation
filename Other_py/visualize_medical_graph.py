#!/usr/bin/env python3
"""
visualize_medical_graph.py
===========================
Publication-quality visualizer for the Heterogeneous Medical Graph
used in the HAN / HGT-HAN recommendation system.

Graph schema:
  Patient ─(has test)─▶ Symptom/Test ─(maps to)─▶ Organ ─(associated with)─▶ Disease
  (P)                       (S)                      (O)                          (D)

Meta-paths aggregated by HAN:
  P ──(P-S-P)──▶ P   via shared symptoms
  P ──(P-O-P)──▶ P   via shared organs
  P ──(P-D-P)──▶ P   via shared diseases

Produces FOUR publication-quality figures saved to output/:
  1. graph_schema.png          – Conceptual schema (for Methods section)
  2. sampled_subgraph.png      – Real data mini subgraph (N patients)
  3. metapath_diagram.png      – Three meta-path illustrations side-by-side
  4. graph_statistics.png      – Degree distributions & node/edge stats

Usage:
  python visualize_medical_graph.py [options]

  --records    Path to patient records CSV
  --symptom    Path to symptom-disease-organ metadata CSV
  --patients   Number of patients to sample for subgraph  (default: 12)
  --seed       Random seed                                (default: 42)
  --outdir     Output directory                           (default: ../output/graph_viz)
  --dpi        Output DPI                                 (default: 300)
  --no-schema  Skip schema figure
  --no-sub     Skip sampled subgraph figure
  --no-meta    Skip meta-path figure
  --no-stats   Skip statistics figure
"""

import os
import sys
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import matplotlib.gridspec as gridspec
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
# Default paths (absolute, matching train.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_HAN_ROOT   = os.path.dirname(_THIS_DIR)            # HAN-implementation/
_DATA_DIR   = os.path.join(_HAN_ROOT, "data")

DEFAULT_RECORDS = os.path.join(_DATA_DIR, "filtered_patient_reports.csv")
DEFAULT_SYMPTOM = os.path.join(_DATA_DIR, "test-disease-organ.csv")
DEFAULT_OUTDIR  = os.path.join(_HAN_ROOT, "output", "graph_viz")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (colour-blind friendly + publication appropriate)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "patient":  "#3A86FF",   # vivid blue
    "symptom":  "#8338EC",   # purple
    "organ":    "#FB5607",   # orange-red
    "disease":  "#06D6A0",   # teal-green
    "edge_ps":  "#8338EC",   # purple  (P→S)
    "edge_so":  "#FB5607",   # orange  (S→O)
    "edge_od":  "#06D6A0",   # teal    (O→D)
    "meta_psp": "#3A86FF",   # blue    (P-S-P)
    "meta_pop": "#FB5607",   # orange  (P-O-P)
    "meta_pdp": "#06D6A0",   # teal    (P-D-P)
    "bg":       "#FAFAFA",
}

# ─────────────────────────────────────────────────────────────────────────────
# Apply matplotlib rcparams for paper-quality output
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "DejaVu Sans",
    "axes.titlesize":       14,
    "axes.labelsize":       12,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      10,
    "figure.facecolor":     PALETTE["bg"],
    "axes.facecolor":       PALETTE["bg"],
    "savefig.facecolor":    PALETTE["bg"],
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            False,
})

warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path_records: str, path_symptom: str, freq_threshold: float = 0.08):
    """
    Minimal data loader that replicates the essential steps from
    MedicalGraphData.load_data() + build_adjacency_matrices().

    Returns a dict with:
        patients, symptoms, organs, diseases   – sorted lists
        patient_map, symptom_map, organ_map, disease_map
        A_PS, A_SO, A_OD                       – scipy CSR matrices
        df_records, df_symptom
    """
    print("  Loading CSV files …")
    df_rec = pd.read_csv(path_records, low_memory=False)
    df_sym = pd.read_csv(path_symptom, low_memory=False)

    # Normalise column names
    df_rec.columns = df_rec.columns.str.strip()
    df_sym.columns = df_sym.columns.str.strip()

    col_rec = {c: "PatientID"   for c in ("patient_id", "PatientID")   if c in df_rec.columns}
    col_rec.update({c: "TestName"   for c in ("test_name",   "TestName")   if c in df_rec.columns})
    df_rec.rename(columns=col_rec, inplace=True)

    col_sym = {c: "TestName"            for c in ("test_name",   "TestName")           if c in df_sym.columns}
    col_sym.update({c: "Target_Organ"   for c in ("organ",       "Target_Organ")       if c in df_sym.columns})
    col_sym.update({c: "Most_Relevant_Disease" for c in ("disease", "Most_Relevant_Disease") if c in df_sym.columns})
    df_sym.rename(columns=col_sym, inplace=True)

    # Drop duplicate organ/disease columns produced by pandas (organ.1 etc.)
    drop = [c for c in df_sym.columns if c.startswith(("organ.", "disease."))]
    if drop:
        df_sym.drop(columns=drop, inplace=True)

    df_rec["TestName"]  = df_rec["TestName"].astype(str).str.strip()
    df_sym["TestName"]  = df_sym["TestName"].astype(str).str.strip()

    # Filter hub symptoms (appear in > threshold fraction of patients)
    patients = sorted(df_rec["PatientID"].unique().tolist())
    n_patients = len(patients)
    sym_counts = df_rec.groupby("TestName")["PatientID"].nunique()
    hub = set(sym_counts[sym_counts / n_patients > freq_threshold].index)
    if hub:
        print(f"  Filtering {len(hub)} hub symptoms (>{freq_threshold*100:.0f}% prevalence)")
        df_rec = df_rec[~df_rec["TestName"].isin(hub)].copy()

    symptoms = sorted(df_rec["TestName"].unique().tolist())
    organs   = sorted([x for x in df_sym["Target_Organ"].unique()           if pd.notna(x)])
    diseases = sorted([x for x in df_sym["Most_Relevant_Disease"].unique()  if pd.notna(x)])

    patient_map  = {p: i for i, p in enumerate(patients)}
    symptom_map  = {s: i for i, s in enumerate(symptoms)}
    organ_map    = {o: i for i, o in enumerate(organs)}
    disease_map  = {d: i for i, d in enumerate(diseases)}

    P, S, O, D = len(patients), len(symptoms), len(organs), len(diseases)
    print(f"  Nodes → Patients:{P}  Symptoms:{S}  Organs:{O}  Diseases:{D}")

    # ── Adjacency matrices ──────────────────────────────────────────────────
    # Patient–Symptom
    ps_r, ps_c = [], []
    for _, row in df_rec.iterrows():
        pid  = row["PatientID"]
        test = row["TestName"]
        if pid in patient_map and test in symptom_map:
            ps_r.append(patient_map[pid])
            ps_c.append(symptom_map[test])

    A_PS = sp.csr_matrix(
        (np.ones(len(ps_r), dtype=np.float32), (ps_r, ps_c)), shape=(P, S)
    )

    # Symptom–Organ  &  Organ–Disease
    so_r, so_c, od_r, od_c = [], [], [], []
    for _, row in df_sym.iterrows():
        sn  = str(row["TestName"]).strip()
        org = row.get("Target_Organ")
        dis = row.get("Most_Relevant_Disease")
        if sn not in symptom_map:
            continue
        if pd.notna(org) and org in organ_map:
            so_r.append(symptom_map[sn])
            so_c.append(organ_map[org])
        if pd.notna(org) and pd.notna(dis) and org in organ_map and dis in disease_map:
            od_r.append(organ_map[org])
            od_c.append(disease_map[dis])

    A_SO = sp.csr_matrix(
        (np.ones(len(so_r), dtype=np.float32), (so_r, so_c)), shape=(S, O)
    )
    A_OD = sp.csr_matrix(
        (np.ones(len(od_r), dtype=np.float32), (od_r, od_c)), shape=(O, D)
    )

    print(f"  Edges → A_PS:{A_PS.nnz}  A_SO:{A_SO.nnz}  A_OD:{A_OD.nnz}")

    return dict(
        patients=patients, symptoms=symptoms, organs=organs, diseases=diseases,
        patient_map=patient_map, symptom_map=symptom_map,
        organ_map=organ_map,    disease_map=disease_map,
        A_PS=A_PS, A_SO=A_SO, A_OD=A_OD,
        df_records=df_rec, df_symptom=df_sym,
        P=P, S=S, O=O, D=D,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 ─ GRAPH SCHEMA  (conceptual, for the Methods section)
# ═══════════════════════════════════════════════════════════════════════════════

def figure_schema(data: dict, outpath: str, dpi: int = 300):
    """
    Draw the heterogeneous graph schema as a horizontal pipeline:
       Patient  →  Symptom/Test  →  Organ  →  Disease
    with bi-directional edges, node-type icons and counts.
    Also includes a small meta-path inset.
    """
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # ── Node boxes ───────────────────────────────────────────────────────────
    nodes = [
        dict(x=2.0,  label="Patient (P)",       count=data["P"], key="patient",  icon="👤"),
        dict(x=5.5,  label="Symptom /\nTest (S)", count=data["S"], key="symptom",  icon="🔬"),
        dict(x=9.0,  label="Organ (O)",          count=data["O"], key="organ",    icon="🫀"),
        dict(x=12.5, label="Disease (D)",        count=data["D"], key="disease",  icon="🧬"),
    ]

    node_r  = 0.85   # circle radius
    y_nodes = 4.5

    # Draw circles
    for n in nodes:
        c = Circle((n["x"], y_nodes), node_r,
                   color=PALETTE[n["key"]], zorder=3, alpha=0.92)
        ax.add_patch(c)
        # Icon / abbreviation
        ax.text(n["x"], y_nodes + 0.35, n["icon"], ha="center", va="center",
                fontsize=18, zorder=4)
        # Count inside circle
        ax.text(n["x"], y_nodes - 0.25, f"n={n['count']:,}", ha="center",
                va="center", fontsize=9, fontweight="bold", color="white", zorder=4)
        # Label below
        ax.text(n["x"], y_nodes - node_r - 0.45, n["label"],
                ha="center", va="top", fontsize=12, fontweight="bold",
                color=PALETTE[n["key"]])

    # ── Edges between nodes ──────────────────────────────────────────────────
    edge_specs = [
        (nodes[0], nodes[1], PALETTE["edge_ps"], "has test\nresult",           "A_PS"),
        (nodes[1], nodes[2], PALETTE["edge_so"], "associated\nwith organ",     "A_SO"),
        (nodes[2], nodes[3], PALETTE["edge_od"], "linked to\ndisease",         "A_OD"),
    ]
    for src, dst, col, lbl, mat_key in edge_specs:
        x0, x1 = src["x"] + node_r, dst["x"] - node_r
        xm      = (x0 + x1) / 2
        nnz     = data[mat_key].nnz if mat_key in data else 0
        # Forward arrow
        ax.annotate("", xy=(x1, y_nodes + 0.15), xytext=(x0, y_nodes + 0.15),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=2.2, mutation_scale=18),
                    zorder=2)
        # Reverse arrow
        ax.annotate("", xy=(x0, y_nodes - 0.15), xytext=(x1, y_nodes - 0.15),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=2.2, mutation_scale=18),
                    zorder=2)
        # Edge label
        ax.text(xm, y_nodes + 0.65, f"{lbl}\n({nnz:,} edges)",
                ha="center", va="bottom", fontsize=9, color=col,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col,
                          alpha=0.85, lw=1.2))

    # ── Meta-path inset (bottom strip) ───────────────────────────────────────
    meta_y = 1.5
    meta_specs = [
        dict(x=2.5,  col=PALETTE["meta_psp"], lbl="P–S–P",
             desc="Patients sharing\nthe same lab tests"),
        dict(x=7.0,  col=PALETTE["meta_pop"], lbl="P–O–P",
             desc="Patients sharing\nthe same organs affected"),
        dict(x=11.5, col=PALETTE["meta_pdp"], lbl="P–D–P",
             desc="Patients sharing\nassociated diseases"),
    ]

    ax.text(7.0, meta_y + 1.15, "Meta-paths used by HAN semantic attention",
            ha="center", va="center", fontsize=11, color="#444",
            fontweight="bold")

    for m in meta_specs:
        box = FancyBboxPatch((m["x"] - 1.5, meta_y - 0.75), 3.0, 1.35,
                             boxstyle="round,pad=0.1",
                             fc=m["col"], ec="none", alpha=0.15, zorder=1)
        ax.add_patch(box)
        ax.text(m["x"], meta_y + 0.18, m["lbl"],
                ha="center", va="center", fontsize=18, fontweight="bold",
                color=m["col"])
        ax.text(m["x"], meta_y - 0.38, m["desc"],
                ha="center", va="center", fontsize=9, color="#444")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(8.0, 6.55,
            "Heterogeneous Medical Knowledge Graph – Schema",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#222")
    ax.text(8.0, 6.15,
            f"Total nodes: {data['P']+data['S']+data['O']+data['D']:,}   "
            f"Total edges: {data['A_PS'].nnz + data['A_SO'].nnz + data['A_OD'].nnz:,}",
            ha="center", va="center", fontsize=10, color="#555")

    plt.tight_layout(pad=0.3)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Schema saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 ─ SAMPLED SUBGRAPH  (real data, small neighbourhood)
# ═══════════════════════════════════════════════════════════════════════════════

def figure_subgraph(data: dict, outpath: str,
                    n_patients: int = 12, seed: int = 42,
                    dpi: int = 300):
    """
    Pick `n_patients` real patients, find their actual symptom/organ/disease
    neighbours and draw the induced subgraph.
    """
    rng = np.random.default_rng(seed)

    # ── Sample patients ───────────────────────────────────────────────────────
    n_patients = min(n_patients, data["P"])
    sampled_p   = sorted(rng.choice(data["P"], size=n_patients, replace=False).tolist())

    # ── Build induced subgraph ────────────────────────────────────────────────
    A_PS = data["A_PS"]
    A_SO = data["A_SO"]
    A_OD = data["A_OD"]

    s_indices  = set()
    o_indices  = set()
    d_indices  = set()

    for pi in sampled_p:
        row_ps = A_PS.getrow(pi)
        s_idx  = row_ps.indices.tolist()
        s_indices.update(s_idx)
        for si in s_idx:
            row_so = A_SO.getrow(si)
            o_idx  = row_so.indices.tolist()
            o_indices.update(o_idx)
            for oi in o_idx:
                row_od = A_OD.getrow(oi)
                d_idx  = row_od.indices.tolist()
                d_indices.update(d_idx)

    s_list = sorted(s_indices)
    o_list = sorted(o_indices)
    d_list = sorted(d_indices)

    # ── Trim for readability (keep top-degree nodes) ──────────────────────────
    MAX_S, MAX_O, MAX_D = 20, 12, 15

    def top_degree(indices, mat, axis, k):
        """Return top-k by degree in `mat` along `axis`."""
        if len(indices) <= k:
            return indices
        if axis == 0:
            degs = np.asarray(mat[indices, :].sum(axis=1)).flatten()
        else:
            degs = np.asarray(mat[:, indices].sum(axis=0)).flatten()
        order = np.argsort(-degs)
        return [indices[i] for i in order[:k]]

    # Symptoms are column indices in A_PS, organs are column indices in A_SO,
    # diseases are column indices in A_OD  → use axis=1 (column-degree)
    s_list = top_degree(s_list, A_PS, axis=1, k=MAX_S)
    o_list = top_degree(o_list, A_SO, axis=1, k=MAX_O)
    d_list = top_degree(d_list, A_OD, axis=1, k=MAX_D)

    # ── Build NetworkX DiGraph ─────────────────────────────────────────────────
    G = nx.DiGraph()
    node_meta = {}   # node_id → (type, label, color)

    sym_rev  = {v: k for k, v in data["symptom_map"].items()}
    org_rev  = {v: k for k, v in data["organ_map"].items()}
    dis_rev  = {v: k for k, v in data["disease_map"].items()}
    pat_rev  = {v: k for k, v in data["patient_map"].items()}

    def short_label(name, maxlen=14):
        name = str(name)
        return name if len(name) <= maxlen else name[:maxlen - 1] + "…"

    # Add nodes
    for pi in sampled_p:
        nid = f"P{pi}"
        G.add_node(nid)
        node_meta[nid] = ("patient",  f"Pt {str(pat_rev.get(pi, pi))[:8]}", PALETTE["patient"])

    for si in s_list:
        nid = f"S{si}"
        G.add_node(nid)
        node_meta[nid] = ("symptom", short_label(sym_rev.get(si, si)), PALETTE["symptom"])

    for oi in o_list:
        nid = f"O{oi}"
        G.add_node(nid)
        node_meta[nid] = ("organ", short_label(org_rev.get(oi, oi)), PALETTE["organ"])

    for di in d_list:
        nid = f"D{di}"
        G.add_node(nid)
        node_meta[nid] = ("disease", short_label(dis_rev.get(di, di)), PALETTE["disease"])

    # Add edges
    for pi in sampled_p:
        for si in s_list:
            if A_PS[pi, si] > 0:
                G.add_edge(f"P{pi}", f"S{si}", etype="P→S")
    for si in s_list:
        for oi in o_list:
            if A_SO[si, oi] > 0:
                G.add_edge(f"S{si}", f"O{oi}", etype="S→O")
    for oi in o_list:
        for di in d_list:
            if A_OD[oi, di] > 0:
                G.add_edge(f"O{oi}", f"D{di}", etype="O→D")

    # ── Layout  ───────────────────────────────────────────────────────────────
    # Hierarchical: P (left) → S → O → D (right)
    pos = {}
    layer_nodes = [
        ([f"P{pi}" for pi in sampled_p], 0.0),
        ([f"S{si}" for si in s_list],    3.0),
        ([f"O{oi}" for oi in o_list],    6.0),
        ([f"D{di}" for di in d_list],    9.0),
    ]
    for node_list, x_val in layer_nodes:
        n = len(node_list)
        ys = np.linspace(0, 1, n) if n > 1 else [0.5]
        for node, y_val in zip(node_list, ys):
            pos[node] = (x_val, y_val)

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 11))
    ax.set_facecolor(PALETTE["bg"])
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.axis("off")

    edge_color_map = {"P→S": PALETTE["edge_ps"],
                      "S→O": PALETTE["edge_so"],
                      "O→D": PALETTE["edge_od"]}

    # Edges by type
    for etype, ecol in edge_color_map.items():
        elist = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == etype]
        if elist:
            nx.draw_networkx_edges(
                G, pos, edgelist=elist, ax=ax,
                edge_color=ecol, alpha=0.45, width=1.2,
                arrows=True, arrowsize=10,
                arrowstyle="->",
                connectionstyle="arc3,rad=0.08",
                min_source_margin=8, min_target_margin=8
            )

    # Nodes by type
    node_sizes_map = {"patient": 900, "symptom": 600, "organ": 700, "disease": 550}
    for ntype in ("patient", "symptom", "organ", "disease"):
        nodes = [n for n, m in node_meta.items() if m[0] == ntype]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes, ax=ax,
                node_color=PALETTE[ntype],
                node_size=node_sizes_map[ntype],
                alpha=0.92, linewidths=1.5, edgecolors="white"
            )

    # Labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: m[1] for n, m in node_meta.items()},
        font_size=5.5, font_color="white", font_weight="bold"
    )

    # Column headers
    for (nlist, x_val), hdr, col in zip(
        layer_nodes,
        [f"Patients (n={n_patients})",
         f"Symptoms/Tests (shown: {len(s_list)})",
         f"Organs (shown: {len(o_list)})",
         f"Diseases (shown: {len(d_list)})"],
        [PALETTE["patient"], PALETTE["symptom"], PALETTE["organ"], PALETTE["disease"]]
    ):
        ax.text(x_val, 1.07, hdr, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=col,
                transform=ax.transData)

    # Legend
    legend_items = [
        mpatches.Patch(color=PALETTE["patient"], label="Patient node"),
        mpatches.Patch(color=PALETTE["symptom"], label="Symptom/Test node"),
        mpatches.Patch(color=PALETTE["organ"],   label="Organ node"),
        mpatches.Patch(color=PALETTE["disease"], label="Disease node"),
        Line2D([0],[0], color=PALETTE["edge_ps"], lw=2, label="Patient → Symptom"),
        Line2D([0],[0], color=PALETTE["edge_so"], lw=2, label="Symptom → Organ"),
        Line2D([0],[0], color=PALETTE["edge_od"], lw=2, label="Organ → Disease"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              framealpha=0.9, fontsize=8.5, ncol=2,
              title="Node & Edge Types", title_fontsize=9)

    ax.set_title(
        f"Sampled Heterogeneous Medical Graph  "
        f"(seed={seed}, {G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
        fontsize=13, fontweight="bold", pad=12, color="#222"
    )

    plt.tight_layout(pad=0.5)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Subgraph saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 ─ META-PATH DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def figure_metapaths(data: dict, outpath: str,
                     n_patients: int = 8, seed: int = 42,
                     dpi: int = 300):
    """
    Draw three small P–P graphs (one per meta-path) side by side,
    labelled with the intermediate node type.
    """
    rng = np.random.default_rng(seed)
    n_patients = min(n_patients, data["P"])
    sampled_p  = sorted(rng.choice(data["P"], size=n_patients, replace=False).tolist())

    A_PS = data["A_PS"]
    A_SO = data["A_SO"]
    A_OD = data["A_OD"]

    # Build P-P adjacencies via each meta-path
    A_PS_sub = A_PS[sampled_p, :]
    M_PO_sub = A_PS_sub.dot(A_SO)   # P×O
    M_PD_sub = M_PO_sub.dot(A_OD)   # P×D

    PP_S = (A_PS_sub.dot(A_PS_sub.T)).toarray()
    PP_O = (M_PO_sub.dot(M_PO_sub.T)).toarray()
    PP_D = (M_PD_sub.dot(M_PD_sub.T)).toarray()

    pat_rev = {v: k for k, v in data["patient_map"].items()}

    def pp_adjacency_to_graph(PP, n):
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                if PP[i, j] > 0:
                    G.add_edge(i, j, weight=float(PP[i, j]))
        return G

    meta_info = [
        dict(label="P–S–P", subtitle="via shared Symptom/Test",
             color=PALETTE["meta_psp"], PP=PP_S, mid_icon="🔬"),
        dict(label="P–O–P", subtitle="via shared Organ",
             color=PALETTE["meta_pop"], PP=PP_O, mid_icon="🫀"),
        dict(label="P–D–P", subtitle="via shared Disease",
             color=PALETTE["meta_pdp"], PP=PP_D, mid_icon="🧬"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, m in zip(axes, meta_info):
        ax.set_facecolor(PALETTE["bg"])
        G = pp_adjacency_to_graph(m["PP"], n_patients)

        pos = nx.circular_layout(G, scale=1.0)

        # Edge widths proportional to co-occurrence count
        edges      = list(G.edges(data=True))
        edge_ws    = [e[2].get("weight", 1) for e in edges]
        max_w      = max(edge_ws) if edge_ws else 1.0
        edge_widths = [1.0 + 3.0 * (w / max_w) for w in edge_ws]

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=m["color"], alpha=0.5,
            width=edge_widths
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=PALETTE["patient"],
            node_size=400, alpha=0.92,
            linewidths=2, edgecolors="white"
        )
        labels = {i: f"P{i+1}" for i in range(n_patients)}
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels,
                                font_size=8, font_color="white",
                                font_weight="bold")
        ax.axis("off")

        # Intermediate node icon ribbon across the center
        ax.text(0, 0, m["mid_icon"], ha="center", va="center",
                fontsize=28, alpha=0.18, zorder=0,
                transform=ax.transData)

        ax.set_title(m["label"], fontsize=20, fontweight="bold",
                     color=m["color"], pad=6)
        ax.text(0.5, -0.04, m["subtitle"],
                ha="center", va="top", transform=ax.transAxes,
                fontsize=10, color="#555")
        edg_cnt = G.number_of_edges()
        ax.text(0.5, -0.10,
                f"{edg_cnt} patient-patient links ({n_patients} patients sampled)",
                ha="center", va="top", transform=ax.transAxes,
                fontsize=8.5, color="#777")

    fig.suptitle(
        "HAN Meta-paths: Patient–Patient Similarity Graphs",
        fontsize=14, fontweight="bold", color="#222", y=1.01
    )
    plt.tight_layout(pad=1.5)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Meta-path diagram saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 ─ GRAPH STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def figure_statistics(data: dict, outpath: str, dpi: int = 300):
    """
    2×3 panel figure:
      Row 1: degree distributions (P, S, O, D)  + edge-type bar chart
      Row 2: patient-organ coverage heat strip + top-N organs/diseases bar
    """
    A_PS = data["A_PS"]
    A_SO = data["A_SO"]
    A_OD = data["A_OD"]

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.48)

    # ── (0,0) Patient degree distribution ────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    p_deg = np.diff(A_PS.indptr)  # num symptoms per patient
    ax00.hist(p_deg, bins=40, color=PALETTE["patient"], alpha=0.85, edgecolor="white")
    ax00.set_xlabel("# Symptoms / Patient", fontsize=10)
    ax00.set_ylabel("# Patients", fontsize=10)
    ax00.set_title("Patient Degree\n(# lab tests)", fontweight="bold")
    ax00.axvline(p_deg.mean(), color="#333", lw=1.5, ls="--",
                 label=f"Mean = {p_deg.mean():.1f}")
    ax00.legend(fontsize=8)

    # ── (0,1) Symptom degree distribution ────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    s_deg = np.asarray(A_PS.sum(axis=0)).flatten()  # patient count per symptom
    ax01.hist(s_deg, bins=40, color=PALETTE["symptom"], alpha=0.85, edgecolor="white")
    ax01.set_xlabel("# Patients using Test", fontsize=10)
    ax01.set_ylabel("# Tests", fontsize=10)
    ax01.set_title("Symptom/Test Degree\n(# patients)", fontweight="bold")
    ax01.axvline(s_deg.mean(), color="#333", lw=1.5, ls="--",
                 label=f"Mean = {s_deg.mean():.1f}")
    ax01.legend(fontsize=8)

    # ── (0,2) Edge-type counts bar chart ─────────────────────────────────────
    ax02 = fig.add_subplot(gs[0, 2])
    edge_labels = ["Patient–\nSymptom\n(A_PS)", "Symptom–\nOrgan\n(A_SO)",
                   "Organ–\nDisease\n(A_OD)"]
    edge_counts = [A_PS.nnz, A_SO.nnz, A_OD.nnz]
    bar_colors  = [PALETTE["edge_ps"], PALETTE["edge_so"], PALETTE["edge_od"]]
    bars = ax02.bar(edge_labels, edge_counts, color=bar_colors, alpha=0.85,
                    edgecolor="white", width=0.5)
    for bar, cnt in zip(bars, edge_counts):
        ax02.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + max(edge_counts) * 0.01,
                  f"{cnt:,}", ha="center", va="bottom",
                  fontsize=10, fontweight="bold", color="#333")
    ax02.set_ylabel("# Edges", fontsize=10)
    ax02.set_title("Edge Counts by Type", fontweight="bold")

    # ── (1,0) Top-20 organs by connectivity ──────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    org_conn = np.asarray(A_SO.sum(axis=0)).flatten()   # symptoms per organ
    top20_o  = np.argsort(-org_conn)[:20]
    org_rev  = {v: k for k, v in data["organ_map"].items()}
    o_labels = [str(org_rev.get(i, i))[:16] for i in top20_o]
    ax10.barh(range(len(top20_o)), org_conn[top20_o],
              color=PALETTE["organ"], alpha=0.85, edgecolor="white")
    ax10.set_yticks(range(len(top20_o)))
    ax10.set_yticklabels(o_labels, fontsize=7.5)
    ax10.invert_yaxis()
    ax10.set_xlabel("# Associated Symptoms", fontsize=10)
    ax10.set_title("Top Organs by\nSymptom Connectivity", fontweight="bold")

    # ── (1,1) Top-20 diseases by connectivity ────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    dis_conn  = np.asarray(A_OD.sum(axis=0)).flatten()  # organs per disease
    top20_d   = np.argsort(-dis_conn)[:20]
    dis_rev   = {v: k for k, v in data["disease_map"].items()}
    d_labels  = [str(dis_rev.get(i, i))[:18] for i in top20_d]
    ax11.barh(range(len(top20_d)), dis_conn[top20_d],
              color=PALETTE["disease"], alpha=0.85, edgecolor="white")
    ax11.set_yticks(range(len(top20_d)))
    ax11.set_yticklabels(d_labels, fontsize=7.5)
    ax11.invert_yaxis()
    ax11.set_xlabel("# Associated Organs", fontsize=10)
    ax11.set_title("Top Diseases by\nOrgan Connectivity", fontweight="bold")

    # ── (1,2) Node-count summary table ───────────────────────────────────────
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.axis("off")
    rows = [
        ["Node type", "Count", "Avg degree"],
        ["Patient (P)",  f"{data['P']:,}",
         f"{np.diff(A_PS.indptr).mean():.1f}  (symptoms)"],
        ["Symptom (S)",  f"{data['S']:,}",
         f"{np.asarray(A_PS.sum(axis=0)).mean():.1f}  (patients)"],
        ["Organ (O)",    f"{data['O']:,}",
         f"{np.asarray(A_SO.sum(axis=0)).mean():.1f}  (symptoms)"],
        ["Disease (D)",  f"{data['D']:,}",
         f"{np.asarray(A_OD.sum(axis=0)).mean():.1f}  (organs)"],
        ["", "", ""],
        ["Edge type", "Count", ""],
        ["A_PS (P→S)", f"{A_PS.nnz:,}", ""],
        ["A_SO (S→O)", f"{A_SO.nnz:,}", ""],
        ["A_OD (O→D)", f"{A_OD.nnz:,}", ""],
        ["TOTAL edges", f"{A_PS.nnz+A_SO.nnz+A_OD.nnz:,}", ""],
    ]
    col_w  = [0.38, 0.28, 0.34]
    col_xs = [0.02, 0.40, 0.68]
    row_h  = 0.09
    for ri, row in enumerate(rows):
        y = 0.97 - ri * row_h
        for ci, (cell, cx) in enumerate(zip(row, col_xs)):
            fw = "bold" if ri == 0 or cell.startswith("Edge") or cell.startswith("Node") else "normal"
            fc = "#222" if ri == 0 else "#444"
            ax12.text(cx, y, cell, transform=ax12.transAxes,
                      fontsize=8.5, va="top", fontweight=fw, color=fc)

    ax12.set_title("Graph Summary", fontweight="bold")

    fig.suptitle("Heterogeneous Medical Graph – Statistics Overview",
                 fontsize=14, fontweight="bold", y=1.01, color="#222")

    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Statistics saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Publication-quality visualization of the Heterogeneous Medical Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--records",   default=DEFAULT_RECORDS,
                   help="Path to patient records CSV")
    p.add_argument("--symptom",   default=DEFAULT_SYMPTOM,
                   help="Path to symptom-disease-organ metadata CSV")
    p.add_argument("--patients",  type=int, default=12,
                   help="Patients to sample for subgraph & meta-path figures (default: 12)")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--outdir",    default=DEFAULT_OUTDIR,
                   help=f"Output directory (default: {DEFAULT_OUTDIR})")
    p.add_argument("--dpi",       type=int, default=300,
                   help="Output DPI (default: 300)")
    p.add_argument("--no-schema", action="store_true", help="Skip schema figure")
    p.add_argument("--no-sub",    action="store_true", help="Skip subgraph figure")
    p.add_argument("--no-meta",   action="store_true", help="Skip meta-path figure")
    p.add_argument("--no-stats",  action="store_true", help="Skip statistics figure")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Check input files ─────────────────────────────────────────────────────
    for path, name in [(args.records, "records CSV"), (args.symptom, "symptom CSV")]:
        if not os.path.exists(path):
            print(f"❌  {name} not found: {path}")
            print("    Use --records / --symptom flags to specify correct paths.")
            sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Heterogeneous Medical Graph Visualizer")
    print(f"{'='*60}")
    print(f"  Records : {args.records}")
    print(f"  Symptom : {args.symptom}")
    print(f"  Output  : {args.outdir}")
    print(f"  DPI     : {args.dpi}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[1/4] Loading graph data …")
    data = load_data(args.records, args.symptom)
    print()

    # ── Figure 1: Schema ──────────────────────────────────────────────────────
    if not args.no_schema:
        print("[2/4] Generating schema figure …")
        figure_schema(data,
                      outpath=os.path.join(args.outdir, "graph_schema.png"),
                      dpi=args.dpi)
    else:
        print("[2/4] Schema figure skipped.")

    # ── Figure 2: Sampled subgraph ────────────────────────────────────────────
    if not args.no_sub:
        print("[3/4] Generating sampled subgraph figure …")
        figure_subgraph(data,
                        outpath=os.path.join(args.outdir, "sampled_subgraph.png"),
                        n_patients=args.patients, seed=args.seed,
                        dpi=args.dpi)
    else:
        print("[3/4] Subgraph figure skipped.")

    # ── Figure 3: Meta-paths ──────────────────────────────────────────────────
    if not args.no_meta:
        print("[4/4a] Generating meta-path diagram …")
        figure_metapaths(data,
                         outpath=os.path.join(args.outdir, "metapath_diagram.png"),
                         n_patients=args.patients, seed=args.seed,
                         dpi=args.dpi)
    else:
        print("[4/4a] Meta-path figure skipped.")

    # ── Figure 4: Statistics ──────────────────────────────────────────────────
    if not args.no_stats:
        print("[4/4b] Generating statistics figure …")
        figure_statistics(data,
                          outpath=os.path.join(args.outdir, "graph_statistics.png"),
                          dpi=args.dpi)
    else:
        print("[4/4b] Statistics figure skipped.")

    print(f"\n🎉  All done!  Figures saved in: {args.outdir}")
    print("    graph_schema.png       ← Methods diagram (conceptual)")
    print("    sampled_subgraph.png   ← Real data mini-graph")
    print("    metapath_diagram.png   ← P–P meta-path similarity graphs")
    print("    graph_statistics.png   ← Degree distributions & counts")


if __name__ == "__main__":
    main()
