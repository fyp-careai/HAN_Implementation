#!/usr/bin/env python3
"""
visualize_model_graph.py
=========================
Draw HAN++ / HGT-HAN model architecture as a real NetworkX node-edge graph,
in the same visual style as sampled_subgraph.png

Every component (layer, attention head, classifier) is a node.
Data flow is shown as directed edges labelled with tensor shapes.

Columns (left → right):
  [Input]  →  [Projection]  →  [Att Heads]  →  [Meta-path Z]
           →  [Semantic Att]  →  [Out Proj]
           →  [Organ Classifiers ×19]  →  [Sev. Output nodes]
                                        →  [Regression head]

Usage:
  python visualize_model_graph.py <model.pt> [options]
  python visualize_model_graph.py models_saved/with_ruhunu_data/*.pt

  --outdir  Output directory  (default: ../output/model_graph)
  --dpi     DPI               (default: 300)
  --organs  Path to symptom CSV for real organ names  (auto-detected)
"""

import os, sys, re, argparse
import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
DEFAULT_SYMPTOM = os.path.join(_ROOT, "data", "test-disease-organ.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette – one per node category
# ─────────────────────────────────────────────────────────────────────────────
PAL = {
    "input":      "#3A86FF",   # vivid blue
    "proj":       "#7209B7",   # deep violet
    "att_head":   "#F72585",   # hot pink
    "meta_z":     "#4CC9F0",   # sky blue
    "semantic":   "#FB5607",   # orange
    "out_proj":   "#06D6A0",   # teal
    "organ_clf":  "#3A86FF",   # blue   (reused – different layer)
    "regression": "#8338EC",   # purple
    "sev_out":    "#FFD166",   # amber
    "reg_out":    "#EF476F",   # coral
    "bg":         "#0F0F1A",   # very dark navy background (publication style)
    "edge_flow":  "#AAAAAA",
    "edge_branch":"#F72585",
    "edge_merge": "#FB5607",
    "edge_clf":   "#3A86FF",
    "edge_reg":   "#8338EC",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "figure.facecolor":  PAL["bg"],
    "axes.facecolor":    PAL["bg"],
    "savefig.facecolor": PAL["bg"],
    "text.color":        "white",
})


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD + INTROSPECT
# ═══════════════════════════════════════════════════════════════════════════════

def introspect(pt_path: str, organ_csv: str) -> dict:
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]; break

    sd   = raw
    keys = list(sd.keys())

    is_hgt        = any("hgt_layers" in k for k in keys)
    att_prefix    = "hgt_layers" if is_hgt else "node_atts"
    model_type    = "HGT-HAN" if is_hgt else "HAN++"

    in_dim      = sd["project.weight"].shape[1]
    hidden_dim  = sd["project.weight"].shape[0]
    out_dim     = sd["out_proj.weight"].shape[0]
    num_organs  = sd["organ_regression.weight"].shape[0]
    num_sev     = sd["organ_classifiers.0.weight"].shape[0]

    branch_idx   = sorted({
        int(re.search(r'\.(\d+)\.', k).group(1))
        for k in keys if re.search(rf'{att_prefix}\.(\d+)\.', k)
    })
    num_metapaths = len(branch_idx)

    if is_hgt:
        nhead = 4   # default – not stored explicitly
    else:
        a_l_key = f"{att_prefix}.0.a_l"
        nhead   = sd[a_l_key].shape[0] if a_l_key in sd else 4

    head_dim = hidden_dim // nhead

    stem     = os.path.splitext(os.path.basename(pt_path))[0]
    mp_names = re.findall(r'P-[A-Z]-P', stem) or [f"MP{i}" for i in range(num_metapaths)]

    # Load organ names from CSV
    organ_names = [f"Organ {i}" for i in range(num_organs)]  # fallback
    if os.path.exists(organ_csv):
        try:
            df  = pd.read_csv(organ_csv)
            df.columns = df.columns.str.strip()
            col = next(c for c in df.columns if "organ" in c.lower())
            raw_organs = sorted([x for x in df[col].unique() if str(x).strip()])
            if len(raw_organs) == num_organs:
                organ_names = [str(o).strip() for o in raw_organs]
        except Exception:
            pass

    num_params = sum(p.numel() for p in sd.values() if isinstance(p, torch.Tensor))

    return dict(
        model_type=model_type, is_hgt=is_hgt, att_prefix=att_prefix,
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        num_metapaths=num_metapaths, num_heads=nhead, head_dim=head_dim,
        num_organs=num_organs, num_sev=num_sev,
        metapath_names=mp_names, organ_names=organ_names,
        stem=stem, num_params=num_params,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD NETWORKX GRAPH + LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(info: dict):
    """
    Build DiGraph with nodes for every component and edges for data flow.
    Returns (G, pos, node_colors, node_sizes, node_labels, edge_colors, edge_labels)
    """
    G   = nx.DiGraph()
    pos = {}
    ncolor  = {}   # node → hex color
    nsize   = {}   # node → pt²  for draw_networkx_nodes
    nlabel  = {}   # node → display string

    mp      = info["num_metapaths"]
    nhead   = info["num_heads"]
    hd      = info["head_dim"]
    D       = info["hidden_dim"]
    dout    = info["out_dim"]
    O       = info["num_organs"]
    S       = info["num_sev"]
    mpnames = info["metapath_names"]
    organs  = info["organ_names"]

    # ── Column X positions ───────────────────────────────────────────────────
    # 0:input  1:proj  2:att-heads  3:meta-z  4:semantic  5:out_proj  6:heads(clf/reg)  7:output-nodes
    X = [0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2]

    def add(nid, x, y, label, color, size):
        G.add_node(nid)
        pos[nid]    = (x, y)
        ncolor[nid] = color
        nsize[nid]  = size
        nlabel[nid] = label

    # ── Input node ───────────────────────────────────────────────────────────
    add("input", X[0], 0, f"Patient\nFeatures\n[N×{info['in_dim']}]",
        PAL["input"], 4000)

    # ── Input projection ─────────────────────────────────────────────────────
    add("proj", X[1], 0, f"Linear\n{info['in_dim']}→{D}\n+ GELU",
        PAL["proj"], 3200)
    G.add_edge("input", "proj", shape=f"[N×{info['in_dim']}]", etype="flow")

    # ── Attention heads (per meta-path per head) ──────────────────────────────
    # Layout: for each meta-path branch (row), spread nhead head-nodes vertically
    total_head_rows = mp * nhead
    head_ys = {}   # (mp_i, head_j) → y
    span    = max(total_head_rows - 1, 1) * 1.4   # total vertical span

    for mi in range(mp):
        for hi in range(nhead):
            idx  = mi * nhead + hi
            y    = span/2 - idx * span / max(total_head_rows - 1, 1)
            nid  = f"head_{mi}_{hi}"
            att_type = "Q·K/√d" if info["is_hgt"] else "aₗ·aᵣ"
            lbl  = f"{'HGT' if info['is_hgt'] else 'GAT'}\nHead {hi+1}\n[N×{hd}]"
            add(nid, X[2], y, lbl, PAL["att_head"], 2000)
            head_ys[(mi, hi)] = y
            G.add_edge("proj", nid, shape=f"[N×{D}]", etype="branch")

    # ── Meta-path embedding nodes ─────────────────────────────────────────────
    mp_y = {}
    for mi in range(mp):
        ys_for_mp = [head_ys[(mi, hi)] for hi in range(nhead)]
        y         = np.mean(ys_for_mp)
        mp_y[mi]  = y
        nid = f"meta_{mi}"
        add(nid, X[3], y,
            f"Z_{mpnames[mi]}\nConcat+Res\n[N×{D}]",
            PAL["meta_z"], 2800)
        for hi in range(nhead):
            G.add_edge(f"head_{mi}_{hi}", nid, shape=f"[N×{hd}]", etype="merge")

    # ── Semantic attention ────────────────────────────────────────────────────
    add("semantic", X[4], 0,
        f"Semantic\nAttn  β∈ℝ^{mp}\n[N×{D}]",
        PAL["semantic"], 3600)
    for mi in range(mp):
        G.add_edge(f"meta_{mi}", "semantic",
                   shape=f"Z_{mpnames[mi]}", etype="merge")

    # ── Output projection ─────────────────────────────────────────────────────
    add("out_proj", X[5], 0,
        f"Linear\n{D}→{dout}\n+ GELU",
        PAL["out_proj"], 3200)
    G.add_edge("semantic", "out_proj", shape=f"[N×{D}]", etype="flow")

    # ── Dropout (virtual node – small diamond style) ──────────────────────────
    add("dropout", X[5] + 0.75, 0,
        f"Drop\nout",
        PAL["out_proj"], 900)
    G.add_edge("out_proj", "dropout", shape="", etype="flow")

    # ── Organ classifier heads ────────────────────────────────────────────────
    clf_ys = np.linspace(O/2 * 0.9, -O/2 * 0.9, O)
    for oi, (oname, cy) in enumerate(zip(organs, clf_ys)):
        short = oname[:14] if len(oname) > 14 else oname
        nid   = f"clf_{oi}"
        add(nid, X[6], cy,
            f"{short}\nLinear→{S}",
            PAL["organ_clf"], 1800)
        G.add_edge("dropout", nid, shape=f"[N×{dout}]", etype="clf")

    # ── Regression head ───────────────────────────────────────────────────────
    reg_y = clf_ys[-1] - 1.4
    add("regression", X[6], reg_y,
        f"Damage\nRegression\nLinear→{O}",
        PAL["regression"], 2400)
    G.add_edge("dropout", "regression", shape=f"[N×{dout}]", etype="reg")

    # ── Severity output nodes ─────────────────────────────────────────────────
    sev_labels = ["Normal", "Mild", "Mod.", "Severe"]
    sev_ys     = np.linspace(O/2 * 0.9 - 0.5, -(O/2 * 0.9 - 0.5), S)
    for si, slbl in enumerate(sev_labels[:S]):
        nid = f"sev_{si}"
        add(nid, X[7], sev_ys[si],
            f"Class {si}\n{slbl}",
            PAL["sev_out"], 1400)
        # Connect each classifier to each severity output (just a subset for clarity)
        for oi in range(O):
            G.add_edge(f"clf_{oi}", nid, shape="", etype="clf_out")

    # ── Regression output node ────────────────────────────────────────────────
    add("reg_out", X[7], reg_y,
        f"Score\n[N×{O}]\n∈[0,1]",
        PAL["reg_out"], 2000)
    G.add_edge("regression", "reg_out", shape=f"[N×{O}]", etype="reg")

    return G, pos, ncolor, nsize, nlabel


# ═══════════════════════════════════════════════════════════════════════════════
# DRAW
# ═══════════════════════════════════════════════════════════════════════════════

def draw(info: dict, G, pos, ncolor, nsize, nlabel, outpath: str, dpi: int):
    O     = info["num_organs"]
    nhead = info["num_heads"]
    mp    = info["num_metapaths"]

    # Figure dimensions scale with organ count
    fig_h = max(16, O * 0.85 + 6)
    fig_w = 26
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor(PAL["bg"])
    fig.patch.set_facecolor(PAL["bg"])
    ax.axis("off")

    # ── Separate edge lists by type for styling ───────────────────────────────
    etype_styles = {
        "flow":     dict(edge_color="#9999BB", width=2.2, alpha=0.80, style="solid"),
        "branch":   dict(edge_color=PAL["edge_branch"], width=1.5, alpha=0.65, style="dashed"),
        "merge":    dict(edge_color=PAL["edge_merge"],  width=1.5, alpha=0.65, style="dashed"),
        "clf":      dict(edge_color=PAL["edge_clf"],    width=0.8, alpha=0.35, style="solid"),
        "clf_out":  dict(edge_color="#FFD166",           width=0.5, alpha=0.20, style="solid"),
        "reg":      dict(edge_color=PAL["edge_reg"],    width=1.8, alpha=0.60, style="solid"),
    }
    all_edges = nx.get_edge_attributes(G, "etype")
    for et, sty in etype_styles.items():
        elist = [(u, v) for (u, v), t in all_edges.items() if t == et]
        if not elist:
            continue
        nx.draw_networkx_edges(
            G, pos, edgelist=elist, ax=ax,
            arrows=True, arrowsize=12, arrowstyle="->",
            connectionstyle="arc3,rad=0.05",
            min_source_margin=12, min_target_margin=12,
            **sty
        )

    # ── Draw nodes by category ────────────────────────────────────────────────
    node_type_map = {
        "input":        ("input",),
        "proj":         ("proj",),
        "att_head":     [f"head_{mi}_{hi}" for mi in range(mp) for hi in range(nhead)],
        "meta_z":       [f"meta_{mi}" for mi in range(mp)],
        "semantic":     ("semantic",),
        "out_proj":     ("out_proj", "dropout"),
        "organ_clf":    [f"clf_{oi}" for oi in range(O)],
        "regression":   ("regression",),
        "sev_out":      [f"sev_{si}" for si in range(info["num_sev"])],
        "reg_out":      ("reg_out",),
    }

    for cat, nids in node_type_map.items():
        nids   = [n for n in nids if n in G.nodes]
        if not nids: continue
        col    = PAL.get(cat, "#888888")
        sizes  = [nsize[n] for n in nids]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nids, ax=ax,
            node_color=col,
            node_size=sizes,
            alpha=0.92,
            linewidths=2,
            edgecolors="white"
        )

    # ── Draw labels ───────────────────────────────────────────────────────────
    # Main labels (inside nodes)
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=nlabel,
        font_size=6.0,
        font_color="white",
        font_weight="bold",
    )

    # ── Edge shape labels (on key edges only, not the clf_out fan) ────────────
    edge_lbl = {
        (u, v): d["shape"]
        for u, v, d in G.edges(data=True)
        if d.get("shape") and d.get("etype") not in ("clf_out", "branch", "clf")
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_lbl, ax=ax,
        font_size=6.5, font_color="#BBBBBB",
        bbox=dict(fc=PAL["bg"], ec="none", alpha=0.7, pad=1.5),
        rotate=False,
    )

    # ── Column header annotations ─────────────────────────────────────────────
    # Find y_max from positions
    all_ys = [y for _, y in pos.values()]
    y_top  = max(all_ys) + 2.2

    X = [0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2]
    headers = [
        (X[0], "Input",          PAL["input"]),
        (X[1], "Projection",     PAL["proj"]),
        (X[2], f"Att Heads\n({nhead} heads × {info['head_dim']}d)", PAL["att_head"]),
        (X[3], "Meta-path\nEmbedding", PAL["meta_z"]),
        (X[4], "Semantic\nAggregation", PAL["semantic"]),
        (X[5], "Output\nProjection", PAL["out_proj"]),
        (X[6], f"Task Heads\n(×{O} organs + reg)", PAL["organ_clf"]),
        (X[7], "Predictions", PAL["sev_out"]),
    ]
    for hx, hlbl, hcol in headers:
        ax.text(hx, y_top, hlbl,
                ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=hcol,
                bbox=dict(fc=PAL["bg"], ec=hcol, alpha=0.85,
                          boxstyle="round,pad=0.35", lw=1.5))
        # Dashed column guide line (subtle)
        y_min = min(all_ys) - 1.5
        ax.plot([hx, hx], [y_top - 0.3, y_min],
                color=hcol, lw=0.5, alpha=0.12, ls="--", zorder=0)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=PAL["input"],      label="Input features"),
        mpatches.Patch(color=PAL["proj"],        label="Linear projection"),
        mpatches.Patch(color=PAL["att_head"],    label=f"{'HGT' if info['is_hgt'] else 'GAT'} attention head"),
        mpatches.Patch(color=PAL["meta_z"],      label="Meta-path embedding Z_ϕ"),
        mpatches.Patch(color=PAL["semantic"],    label="Semantic attention (β)"),
        mpatches.Patch(color=PAL["out_proj"],    label="Output projection"),
        mpatches.Patch(color=PAL["organ_clf"],   label=f"Organ classifier (×{O})"),
        mpatches.Patch(color=PAL["regression"],  label="Damage regression head"),
        mpatches.Patch(color=PAL["sev_out"],     label="Severity output (0-3)"),
        mpatches.Patch(color=PAL["reg_out"],     label="Damage score output"),
        Line2D([0],[0], color="#9999BB",          lw=2, label="Main data flow"),
        Line2D([0],[0], color=PAL["edge_branch"], lw=1.5, ls="--", label="Branch (proj→head)"),
        Line2D([0],[0], color=PAL["edge_merge"],  lw=1.5, ls="--", label="Merge (head→Z_ϕ)"),
        Line2D([0],[0], color=PAL["edge_clf"],    lw=1, label="Classifier edge"),
        Line2D([0],[0], color=PAL["edge_reg"],    lw=1.8, label="Regression edge"),
    ]
    leg = ax.legend(
        handles=legend_items, loc="lower left",
        fontsize=8, ncol=3,
        title=f"{info['model_type']} — {info['stem']}  "
              f"({info['num_params']:,} parameters)",
        title_fontsize=9.5,
        framealpha=0.85,
        facecolor="#1A1A2E",
        edgecolor="#444",
        labelcolor="white",
    )
    leg.get_title().set_color("white")
    leg.get_title().set_fontweight("bold")

    # ── Stats strip (top-right) ────────────────────────────────────────────────
    stats = (
        f"Model:       {info['model_type']}\n"
        f"Meta-path:   {', '.join(info['metapath_names'])}\n"
        f"Input dim:   {info['in_dim']}\n"
        f"Hidden dim:  {info['hidden_dim']}\n"
        f"Output dim:  {info['out_dim']}\n"
        f"Att heads:   {info['num_heads']}  (each {info['head_dim']}d)\n"
        f"Organs:      {info['num_organs']}\n"
        f"Sev classes: {info['num_sev']}\n"
        f"Parameters:  {info['num_params']:,}\n"
        f"Nodes drawn: {G.number_of_nodes()}\n"
        f"Edges drawn: {G.number_of_edges()}"
    )
    ax.text(0.995, 0.99, stats,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8, color="#CCCCCC",
            fontfamily="monospace",
            bbox=dict(fc="#1A1A2E", ec="#555", alpha=0.88,
                      boxstyle="round,pad=0.5", lw=1))

    # ── Title ────────────────────────────────────────────────────────────────
    ax.set_title(
        f"{info['model_type']} Full Computational Graph\n"
        f"{info['stem']}  ·  "
        f"{info['in_dim']}d-in  →  {info['hidden_dim']}d-hidden  →  "
        f"{info['num_organs']}×{info['num_sev']} severity + damage score",
        fontsize=13, fontweight="bold", color="white", pad=14
    )

    plt.tight_layout(pad=0.8)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Draw HAN++/HGT-HAN model as a NetworkX node-edge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("pt_files", nargs="+", help=".pt model checkpoint files")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: ../output/model_graph)")
    p.add_argument("--dpi",    type=int, default=300)
    p.add_argument("--organs", default=DEFAULT_SYMPTOM,
                   help="Path to symptom CSV for organ name labels")
    return p.parse_args()


def main():
    args = parse_args()

    for pt_path in args.pt_files:
        pt_path = os.path.abspath(pt_path)
        if not os.path.exists(pt_path):
            print(f"❌  Not found: {pt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  {os.path.basename(pt_path)}")
        print(f"{'='*60}")

        if args.outdir:
            outdir = args.outdir
        else:
            outdir = os.path.join(os.path.dirname(os.path.dirname(pt_path)),
                                  "output", "model_graph")
        os.makedirs(outdir, exist_ok=True)

        info = introspect(pt_path, args.organs)
        print(f"  {info['model_type']}  |  meta-paths: {info['metapath_names']}")
        print(f"  {info['in_dim']}→{info['hidden_dim']}→{info['out_dim']}  "
              f"|  {info['num_heads']} heads  "
              f"|  {info['num_organs']} organs  "
              f"|  {info['num_params']:,} params")

        G, pos, ncolor, nsize, nlabel = build_graph(info)
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        outpath = os.path.join(outdir, f"{info['stem']}_model_graph.png")
        draw(info, G, pos, ncolor, nsize, nlabel, outpath, args.dpi)

    print(f"\n🎉  Done!  Figures saved in: {outdir}")


if __name__ == "__main__":
    main()
