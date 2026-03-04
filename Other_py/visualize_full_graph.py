#!/usr/bin/env python3
"""
visualize_full_graph.py
========================
Draw the COMPLETE heterogeneous medical graph with every single node:
  5,766 Patients  ·  119 Symptoms/Tests  ·  19 Organs  ·  44 Diseases

Layout: strict 4-column hierarchy  P | S | O | D
  - Patient nodes sorted by degree (most-connected at centre)
  - Symptom nodes sorted by patient-degree
  - Organ / Disease nodes labelled individually
  - Edges drawn as thin LineCollections for performance

Outputs:
  full_graph.png        – full graph, all nodes + all edges
  full_graph_zoom_*.png – individual zoomed panels per edge type

Usage:
  python visualize_full_graph.py [options]

  --records  path to patient records CSV
  --symptom  path to symptom-disease-organ CSV
  --outdir   output directory  (default: ../output/graph_viz)
  --dpi      300
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REC = os.path.join(_ROOT, "data", "filtered_patient_reports.csv")
DEFAULT_SYM = os.path.join(_ROOT, "data", "test-disease-organ.csv")
DEFAULT_OUT = os.path.join(_ROOT, "output", "graph_viz")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = "#0D1117"
PAL = dict(
    patient  = "#3A86FF",   # blue
    symptom  = "#8338EC",   # purple
    organ    = "#FF6B35",   # orange-red
    disease  = "#06D6A0",   # teal-green
    edge_ps  = "#8338EC",   # P→S  purple
    edge_so  = "#FF6B35",   # S→O  orange
    edge_od  = "#06D6A0",   # O→D  teal
    bg       = DARK_BG,
)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    DARK_BG,
    "savefig.facecolor": DARK_BG,
    "text.color":        "white",
})


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_graph(path_records: str, path_symptom: str, freq_threshold: float = 0.08):
    print("  Loading CSVs …")
    df_rec = pd.read_csv(path_records, low_memory=False)
    df_sym = pd.read_csv(path_symptom,  low_memory=False)
    df_rec.columns = df_rec.columns.str.strip()
    df_sym.columns = df_sym.columns.str.strip()

    for old, new in [("patient_id","PatientID"),("test_name","TestName")]:
        if old in df_rec.columns: df_rec.rename(columns={old: new}, inplace=True)
    for old, new in [("test_name","TestName"),("organ","Target_Organ"),
                     ("disease","Most_Relevant_Disease")]:
        if old in df_sym.columns: df_sym.rename(columns={old: new}, inplace=True)

    drop = [c for c in df_sym.columns if c.startswith(("organ.","disease."))]
    if drop: df_sym.drop(columns=drop, inplace=True)

    df_rec["TestName"] = df_rec["TestName"].astype(str).str.strip()
    df_sym["TestName"] = df_sym["TestName"].astype(str).str.strip()

    patients = sorted(df_rec["PatientID"].unique().tolist())
    sc  = df_rec.groupby("TestName")["PatientID"].nunique()
    hub = set(sc[sc / len(patients) > freq_threshold].index)
    if hub:
        print(f"  Filtered {len(hub)} hub symptoms (>{freq_threshold*100:.0f}% prevalence)")
        df_rec = df_rec[~df_rec["TestName"].isin(hub)].copy()

    symptoms = sorted(df_rec["TestName"].unique().tolist())
    organs   = sorted([x for x in df_sym["Target_Organ"].unique()
                       if str(x).strip() and str(x) != "nan"])
    diseases = sorted([x for x in df_sym["Most_Relevant_Disease"].unique()
                       if str(x).strip() and str(x) != "nan"])

    P, S, O, D = len(patients), len(symptoms), len(organs), len(diseases)
    print(f"  Nodes  → Patients:{P}  Symptoms:{S}  Organs:{O}  Diseases:{D}  Total:{P+S+O+D}")

    pm = {p: i for i, p in enumerate(patients)}
    sm = {s: i for i, s in enumerate(symptoms)}
    om = {o: i for i, o in enumerate(organs)}
    dm = {d: i for i, d in enumerate(diseases)}

    # Build edge index arrays
    ps_pairs = (df_rec[df_rec["TestName"].isin(sm)]
                .groupby(["PatientID","TestName"]).size().reset_index()[["PatientID","TestName"]])
    ps_r = ps_pairs["PatientID"].map(pm).values
    ps_c = ps_pairs["TestName"].map(sm).values
    ps_valid = ~(np.isnan(ps_r.astype(float)) | np.isnan(ps_c.astype(float)))
    ps_r = ps_r[ps_valid].astype(int)
    ps_c = ps_c[ps_valid].astype(int)

    sym_set = set(symptoms)
    org_set = set(organs)
    dis_set = set(diseases)
    so_rows, so_cols = [], []
    od_rows, od_cols = [], []

    for _, row in df_sym.iterrows():
        s = str(row.get("TestName","")).strip()
        o = str(row.get("Target_Organ","")).strip()
        d = str(row.get("Most_Relevant_Disease","")).strip()
        if s in sym_set and o in org_set:
            so_rows.append(sm[s]); so_cols.append(om[o])
        if o in org_set and d in dis_set:
            od_rows.append(om[o]); od_cols.append(dm[d])

    # Deduplicate
    so_pairs = list(set(zip(so_rows, so_cols)))
    od_pairs = list(set(zip(od_rows, od_cols)))
    so_rows, so_cols = zip(*so_pairs) if so_pairs else ([],[])
    od_rows, od_cols = zip(*od_pairs) if od_pairs else ([],[])

    print(f"  Edges  → P-S:{len(ps_r)}  S-O:{len(so_rows)}  O-D:{len(od_rows)}")

    return dict(
        patients=patients, symptoms=symptoms, organs=organs, diseases=diseases,
        pm=pm, sm=sm, om=om, dm=dm,
        P=P, S=S, O=O, D=D,
        ps_r=ps_r, ps_c=ps_c,
        so_r=np.array(so_rows, dtype=int), so_c=np.array(so_cols, dtype=int),
        od_r=np.array(od_rows, dtype=int), od_c=np.array(od_cols, dtype=int),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

def build_positions(data: dict):
    """
    Assign (x, y) to every node in 4 columns.
    Within each column, nodes are ordered by degree (most-connected at centre).
    """
    P, S, O, D = data["P"], data["S"], data["O"], data["D"]
    ps_r, ps_c = data["ps_r"], data["ps_c"]
    so_r, so_c = data["so_r"], data["so_c"]
    od_r, od_c = data["od_r"], data["od_c"]

    # Degree of each node
    p_deg = np.bincount(ps_r, minlength=P)                # patient degree
    s_deg = np.bincount(ps_c, minlength=S)                # symptom degree (from patients)
    o_deg = np.bincount(so_c, minlength=O) + np.bincount(od_r, minlength=O)
    d_deg = np.bincount(od_c, minlength=D)

    def column_ys(n, deg_arr):
        """Sort by degree descending, then assign evenly-spaced y ∈ [0,1]."""
        order = np.argsort(-deg_arr)
        ys    = np.zeros(n)
        for rank, idx in enumerate(order):
            ys[idx] = rank / max(n - 1, 1)
        # Re-centre: 0 = top, 1 = bottom → convert to centred around 0
        ys = (0.5 - ys)   # high-degree nodes → positive y (top of centre)
        return ys

    X_COLS = dict(patient=0.0, symptom=3.5, organ=7.0, disease=10.5)
    Y_SCALE = dict(patient=30.0, symptom=8.0, organ=2.0, disease=4.5)

    p_ys = column_ys(P, p_deg) * Y_SCALE["patient"]
    s_ys = column_ys(S, s_deg) * Y_SCALE["symptom"]
    o_ys = column_ys(O, o_deg) * Y_SCALE["organ"]
    d_ys = column_ys(D, d_deg) * Y_SCALE["disease"]

    # Arrays of (x, y) per node type
    p_pos = np.column_stack([np.full(P, X_COLS["patient"]),  p_ys])
    s_pos = np.column_stack([np.full(S, X_COLS["symptom"]),  s_ys])
    o_pos = np.column_stack([np.full(O, X_COLS["organ"]),    o_ys])
    d_pos = np.column_stack([np.full(D, X_COLS["disease"]),  d_ys])

    return dict(p=p_pos, s=s_pos, o=o_pos, d=d_pos,
                p_deg=p_deg, s_deg=s_deg, o_deg=o_deg, d_deg=d_deg,
                x_cols=X_COLS)


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAW  (main figure)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_full(data: dict, pos: dict, outpath: str, dpi: int = 300):
    P, S, O, D = data["P"], data["S"], data["O"], data["D"]
    ps_r, ps_c = data["ps_r"], data["ps_c"]
    so_r, so_c = data["so_r"], data["so_c"]
    od_r, od_c = data["od_r"], data["od_c"]
    XC         = pos["x_cols"]

    # Figure – tall enough for patient column, wide enough for labels
    fig, ax = plt.subplots(figsize=(24, 32))
    ax.set_facecolor(DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    ax.axis("off")

    # ── Helper: build LineCollection for one edge set ─────────────────────────
    def edge_collection(src_pos, dst_pos, src_idx, dst_idx, color, alpha, lw):
        segs = [
            [src_pos[s], dst_pos[d]]
            for s, d in zip(src_idx, dst_idx)
        ]
        lc = LineCollection(segs, colors=color, alpha=alpha,
                            linewidths=lw, zorder=1)
        ax.add_collection(lc)

    # ── Draw edges (bottom layer) ─────────────────────────────────────────────
    print("  Drawing P→S edges …")
    edge_collection(pos["p"], pos["s"], ps_r, ps_c,
                    color=PAL["edge_ps"], alpha=0.06, lw=0.35)

    print("  Drawing S→O edges …")
    edge_collection(pos["s"], pos["o"], so_r, so_c,
                    color=PAL["edge_so"], alpha=0.55, lw=1.2)

    print("  Drawing O→D edges …")
    edge_collection(pos["o"], pos["d"], od_r, od_c,
                    color=PAL["edge_od"], alpha=0.65, lw=1.5)

    # ── Draw nodes ────────────────────────────────────────────────────────────
    # Patient nodes – tiny, coloured by degree
    print("  Drawing patient nodes …")
    p_deg_norm = pos["p_deg"] / max(pos["p_deg"].max(), 1)
    # Map degree → size: low-degree patients = 4 pt², high = 25 pt²
    p_sizes = 4 + 22 * p_deg_norm

    ax.scatter(pos["p"][:, 0], pos["p"][:, 1],
               s=p_sizes,
               c=PAL["patient"], alpha=0.55, linewidths=0,
               zorder=3, rasterized=True)

    # Symptom nodes – small circles sized by degree
    print("  Drawing symptom nodes …")
    s_deg_norm = pos["s_deg"] / max(pos["s_deg"].max(), 1)
    s_sizes    = 18 + 80 * s_deg_norm
    ax.scatter(pos["s"][:, 0], pos["s"][:, 1],
               s=s_sizes,
               c=PAL["symptom"], alpha=0.82,
               linewidths=0.8, edgecolors="white",
               zorder=4)

    # Organ nodes – clearly visible with labels
    print("  Drawing organ nodes …")
    o_deg_norm = pos["o_deg"] / max(pos["o_deg"].max(), 1)
    o_sizes    = 120 + 200 * o_deg_norm
    ax.scatter(pos["o"][:, 0], pos["o"][:, 1],
               s=o_sizes,
               c=PAL["organ"], alpha=0.92,
               linewidths=1.5, edgecolors="white",
               zorder=5)

    # Disease nodes – clearly visible with labels
    print("  Drawing disease nodes …")
    d_deg_norm = pos["d_deg"] / max(pos["d_deg"].max(), 1)
    d_sizes    = 100 + 160 * d_deg_norm
    ax.scatter(pos["d"][:, 0], pos["d"][:, 1],
               s=d_sizes,
               c=PAL["disease"], alpha=0.92,
               linewidths=1.5, edgecolors="white",
               zorder=5)

    # ── Symptom labels ────────────────────────────────────────────────────────
    print("  Labelling symptoms …")
    for i, sname in enumerate(data["symptoms"]):
        short = sname[:22] if len(sname) > 22 else sname
        ax.text(pos["s"][i, 0] + 0.22, pos["s"][i, 1],
                short,
                ha="left", va="center",
                fontsize=5.5, color="#CCCCCC", zorder=6)

    # ── Organ labels ──────────────────────────────────────────────────────────
    print("  Labelling organs …")
    for i, oname in enumerate(data["organs"]):
        short = oname[:22] if len(oname) > 22 else oname
        ax.text(pos["o"][i, 0] + 0.18, pos["o"][i, 1],
                short,
                ha="left", va="center",
                fontsize=8.0, fontweight="bold",
                color=PAL["organ"], zorder=6)

    # ── Disease labels ────────────────────────────────────────────────────────
    print("  Labelling diseases …")
    for i, dname in enumerate(data["diseases"]):
        short = dname[:28] if len(dname) > 28 else dname
        ax.text(pos["d"][i, 0] + 0.18, pos["d"][i, 1],
                short,
                ha="left", va="center",
                fontsize=7.5, fontweight="bold",
                color=PAL["disease"], zorder=6)

    # ── Column header labels ──────────────────────────────────────────────────
    all_ys  = np.concatenate([pos["p"][:,1], pos["s"][:,1],
                               pos["o"][:,1], pos["d"][:,1]])
    y_top   = all_ys.max()
    y_bot   = all_ys.min()

    headers = [
        (XC["patient"],  f"Patients\n(n = {P:,})",     PAL["patient"]),
        (XC["symptom"],  f"Symptoms / Tests\n(n = {S})", PAL["symptom"]),
        (XC["organ"],    f"Organs\n(n = {O})",          PAL["organ"]),
        (XC["disease"],  f"Diseases\n(n = {D})",        PAL["disease"]),
    ]
    for hx, hlbl, hcol in headers:
        ax.text(hx, y_top + 1.5, hlbl,
                ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=hcol,
                bbox=dict(fc=DARK_BG, ec=hcol, alpha=0.90,
                          boxstyle="round,pad=0.4", lw=2))
        # column guide line
        ax.plot([hx, hx], [y_top + 1.0, y_bot - 0.5],
                color=hcol, lw=0.4, alpha=0.12, ls="--", zorder=0)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_nodes = [
        mpatches.Patch(color=PAL["patient"],  label=f"Patient node  (n={P:,}, size ∝ #tests)"),
        mpatches.Patch(color=PAL["symptom"],  label=f"Symptom/Test node  (n={S}, size ∝ #patients)"),
        mpatches.Patch(color=PAL["organ"],    label=f"Organ node  (n={O})"),
        mpatches.Patch(color=PAL["disease"],  label=f"Disease node  (n={D})"),
    ]
    legend_edges = [
        Line2D([0],[0], color=PAL["edge_ps"], lw=2, label=f"Patient → Symptom  ({len(ps_r):,} edges)"),
        Line2D([0],[0], color=PAL["edge_so"], lw=2, label=f"Symptom → Organ  ({len(so_r)} edges)"),
        Line2D([0],[0], color=PAL["edge_od"], lw=2, label=f"Organ → Disease  ({len(od_r)} edges)"),
    ]
    leg = ax.legend(
        handles=legend_nodes + legend_edges,
        loc="lower left",
        fontsize=10, ncol=2,
        title=f"Complete Heterogeneous Medical Graph  |  "
              f"{P+S+O+D:,} nodes  ·  {len(ps_r)+len(so_r)+len(od_r):,} edges",
        title_fontsize=11,
        framealpha=0.88,
        facecolor="#1A1A2E",
        edgecolor="#555555",
        labelcolor="white",
    )
    leg.get_title().set_color("white")
    leg.get_title().set_fontweight("bold")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "Complete Heterogeneous Medical Knowledge Graph\n"
        f"Patients  ↔  Symptoms / Tests  →  Organs  →  Diseases",
        fontsize=16, fontweight="bold", color="white", pad=18
    )

    # Auto-fit axes
    margin_x = 2.5
    margin_y = 2.0
    ax.set_xlim(pos["p"][:,0].min() - margin_x,
                pos["d"][:,0].max() + margin_x + 3.5)   # room for disease labels
    ax.set_ylim(y_bot - margin_y, y_top + margin_y + 2.5)

    print("  Saving …")
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Full graph saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ZOOM PANELS  (S→O→D sub-graph with full labels, separate figure)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_knowledge_subgraph(data: dict, pos: dict, outpath: str, dpi: int = 300):
    """
    Separate high-detail figure showing ONLY the knowledge-graph part:
    Symptoms → Organs → Diseases with every node fully labelled.
    """
    S, O, D = data["S"], data["O"], data["D"]
    so_r, so_c = data["so_r"], data["so_c"]
    od_r, od_c = data["od_r"], data["od_c"]

    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_facecolor(DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    ax.axis("off")

    # Column X positions tighter here
    xs = dict(symptom=0.0, organ=5.5, disease=11.0)

    # Re-compute y positions spaced evenly within this figure
    def even_ys(n, span):
        return np.linspace(span / 2, -span / 2, n)

    s_ys_f = even_ys(S, 12.0)
    o_ys_f = even_ys(O, 10.0)
    d_ys_f = even_ys(D, 11.0)

    # Sort nodes by degree for readability
    s_order = np.argsort(-pos["s_deg"])
    o_order = np.argsort(-pos["o_deg"])
    d_order = np.argsort(-pos["d_deg"])

    # Reindex maps (original index → y position in sorted order)
    s_ymap = np.zeros(S)
    for rank, orig in enumerate(s_order):
        s_ymap[orig] = s_ys_f[rank]

    o_ymap = np.zeros(O)
    for rank, orig in enumerate(o_order):
        o_ymap[orig] = o_ys_f[rank]

    d_ymap = np.zeros(D)
    for rank, orig in enumerate(d_order):
        d_ymap[orig] = d_ys_f[rank]

    # Edges
    segs_so = [
        [(xs["symptom"], s_ymap[s]), (xs["organ"], o_ymap[o])]
        for s, o in zip(so_r, so_c)
    ]
    segs_od = [
        [(xs["organ"], o_ymap[o]), (xs["disease"], d_ymap[d])]
        for o, d in zip(od_r, od_c)
    ]

    ax.add_collection(LineCollection(segs_so, colors=PAL["edge_so"],
                                     alpha=0.55, linewidths=1.8, zorder=1))
    ax.add_collection(LineCollection(segs_od, colors=PAL["edge_od"],
                                     alpha=0.65, linewidths=2.0, zorder=1))

    # Symptom nodes + labels
    s_deg_n = pos["s_deg"] / max(pos["s_deg"].max(), 1)
    for i, sname in enumerate(data["symptoms"]):
        y  = s_ymap[i]
        sz = 40 + 120 * s_deg_n[i]
        ax.scatter(xs["symptom"], y, s=sz, c=PAL["symptom"],
                   alpha=0.9, linewidths=0.8, edgecolors="white", zorder=4)
        ax.text(xs["symptom"] - 0.15, y, sname[:24],
                ha="right", va="center", fontsize=6.5, color="#CCCCCC")

    # Organ nodes + labels
    o_deg_n = pos["o_deg"] / max(pos["o_deg"].max(), 1)
    for i, oname in enumerate(data["organs"]):
        y  = o_ymap[i]
        sz = 200 + 400 * o_deg_n[i]
        ax.scatter(xs["organ"], y, s=sz, c=PAL["organ"],
                   alpha=0.95, linewidths=2, edgecolors="white", zorder=5)
        ax.text(xs["organ"], y, oname[:18],
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white", zorder=6)

    # Disease nodes + labels
    d_deg_n = pos["d_deg"] / max(pos["d_deg"].max(), 1)
    for i, dname in enumerate(data["diseases"]):
        y  = d_ymap[i]
        sz = 150 + 300 * d_deg_n[i]
        ax.scatter(xs["disease"], y, s=sz, c=PAL["disease"],
                   alpha=0.95, linewidths=1.5, edgecolors="white", zorder=5)
        ax.text(xs["disease"] + 0.15, y, dname[:28],
                ha="left", va="center", fontsize=7.5,
                fontweight="bold", color=PAL["disease"], zorder=6)

    # Column headers
    for col, lbl, col_c in [
        (xs["symptom"], f"Symptoms / Tests\n(n={S})", PAL["symptom"]),
        (xs["organ"],   f"Organs\n(n={O})",           PAL["organ"]),
        (xs["disease"], f"Diseases\n(n={D})",         PAL["disease"]),
    ]:
        ax.text(col, 6.8, lbl, ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=col_c,
                bbox=dict(fc=DARK_BG, ec=col_c, boxstyle="round,pad=0.4",
                          lw=2, alpha=0.9))

    # Legend
    legend_items = [
        mpatches.Patch(color=PAL["symptom"], label=f"Symptom/Test  (n={S})"),
        mpatches.Patch(color=PAL["organ"],   label=f"Organ  (n={O})"),
        mpatches.Patch(color=PAL["disease"], label=f"Disease  (n={D})"),
        Line2D([0],[0], color=PAL["edge_so"], lw=2,
               label=f"Symptom → Organ  ({len(so_r)} connections)"),
        Line2D([0],[0], color=PAL["edge_od"], lw=2,
               label=f"Organ → Disease  ({len(od_r)} connections)"),
    ]
    leg = ax.legend(handles=legend_items, loc="lower right", fontsize=9,
                    title="Medical Knowledge Graph", title_fontsize=10,
                    facecolor="#1A1A2E", edgecolor="#555", labelcolor="white")
    leg.get_title().set_color("white")
    leg.get_title().set_fontweight("bold")

    ax.set_title(
        "Medical Knowledge Graph  —  Symptoms / Tests  →  Organs  →  Diseases\n"
        f"Node size ∝ connectivity degree",
        fontsize=14, fontweight="bold", color="white", pad=14
    )
    ax.set_xlim(-3.5, 14.5)
    ax.set_ylim(-7.5, 7.5)

    plt.tight_layout(pad=0.5)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Knowledge subgraph saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Draw the complete heterogeneous medical graph",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--records", default=DEFAULT_REC)
    p.add_argument("--symptom", default=DEFAULT_SYM)
    p.add_argument("--outdir",  default=DEFAULT_OUT)
    p.add_argument("--dpi",     type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    for path in [args.records, args.symptom]:
        if not os.path.exists(path):
            print(f"❌  Not found: {path}")
            sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Complete Heterogeneous Medical Graph Visualizer")
    print(f"{'='*60}")

    print("\n[1/3] Loading data …")
    data = load_graph(args.records, args.symptom)

    print("\n[2/3] Computing layout …")
    pos  = build_positions(data)

    print("\n[3a/3] Drawing full graph (all nodes) …")
    draw_full(data, pos,
              outpath=os.path.join(args.outdir, "full_graph.png"),
              dpi=args.dpi)

    print("\n[3b/3] Drawing knowledge subgraph (S→O→D, fully labelled) …")
    draw_knowledge_subgraph(data, pos,
              outpath=os.path.join(args.outdir, "full_graph_knowledge.png"),
              dpi=args.dpi)

    P,S,O,D = data["P"],data["S"],data["O"],data["D"]
    total_e = len(data["ps_r"]) + len(data["so_r"]) + len(data["od_r"])
    print(f"\n🎉  Done!")
    print(f"   full_graph.png           — all {P+S+O+D:,} nodes, {total_e:,} edges")
    print(f"   full_graph_knowledge.png — S→O→D knowledge graph, fully labelled")
    print(f"   Saved to: {args.outdir}")


if __name__ == "__main__":
    main()
