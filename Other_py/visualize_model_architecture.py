#!/usr/bin/env python3
"""
visualize_model_architecture.py
=================================
Publication-quality architecture diagram for HAN++ / HGT-HAN models.

Loads a saved .pt checkpoint, introspects the state-dict to recover
ALL hyperparameters automatically (in_dim, hidden_dim, out_dim,
num_heads, num_organs, num_metapaths, etc.) and renders a full
architecture diagram.

Produces THREE figures per model file:
  1. <stem>_architecture.png   – Full layered architecture (main paper figure)
  2. <stem>_attention.png      – Zoomed-in attention mechanism detail
  3. <stem>_dataflow.png       – Tensor shape data-flow diagram

Usage:
  python visualize_model_architecture.py <model.pt> [options]

  --outdir   Output directory (default: ../output/model_viz)
  --dpi      Output DPI      (default: 300)
  --no-att   Skip attention detail figure
  --no-flow  Skip data-flow figure
"""

import os, sys, re, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "input":        "#4361EE",   # cobalt
    "proj":         "#7209B7",   # violet
    "node_att":     "#F72585",   # hot pink  (HAN node attention)
    "hgt":          "#F72585",   # same for HGT-style
    "meta_head":    "#4CC9F0",   # sky blue  (per-metapath branch)
    "semantic":     "#FB5607",   # orange    (semantic aggregation)
    "out_proj":     "#06D6A0",   # teal      (output projection)
    "clf":          "#3A86FF",   # blue      (organ classifiers)
    "reg":          "#8338EC",   # purple    (regression head)
    "output":       "#06D6A0",   # teal
    "arrow":        "#555555",
    "bg":           "#FAFAFA",
    "box_face":     "#FFFFFF",
    "grid":         "#EEEEEE",
    "text_dark":    "#111111",
    "text_mid":     "#444444",
    "text_light":   "#777777",
    "leaky_relu":   "#FFD166",
    "softmax":      "#EF476F",
    "layernorm":    "#06D6A0",
    "gelu":         "#118AB2",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    C["bg"],
    "figure.facecolor":  C["bg"],
    "savefig.facecolor": C["bg"],
})


# ═══════════════════════════════════════════════════════════════════════════════
# INTROSPECT .pt FILE
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_introspect(pt_path: str) -> dict:
    """
    Load a .pt checkpoint and recover all hyperparameters from the state-dict
    shapes alone — no source code import required.

    Supports both:
      • torch.save(model.state_dict(), path)
      • torch.save({'model_state_dict': ..., ...}, path)
    """
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Unwrap common checkpoint wrappers
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                sd = raw[key]
                meta_extra = {k: v for k, v in raw.items()
                              if k not in ("model_state_dict", "state_dict", "model")}
                break
        else:
            sd = raw
            meta_extra = {}
    else:
        # Could be the model itself (rare)
        sd = raw.state_dict()
        meta_extra = {}

    keys = list(sd.keys())

    # ── Detect model type from key names ──────────────────────────────────────
    is_hgt = any("hgt_layers" in k for k in keys)
    model_type = "HGT-HAN" if is_hgt else "HAN++"
    att_prefix  = "hgt_layers" if is_hgt else "node_atts"

    # ── Recover dimensions ────────────────────────────────────────────────────
    in_dim     = sd["project.weight"].shape[1]
    hidden_dim = sd["project.weight"].shape[0]

    # Number of meta-path branches
    branch_indices = sorted({
        int(re.search(r'\.(\d+)\.', k).group(1))
        for k in keys if re.search(rf'{att_prefix}\.(\d+)\.', k)
    })
    num_metapaths = len(branch_indices)

    # Number of attention heads
    if is_hgt:
        q_key = f"{att_prefix}.0.q_lin.weight"
        head_key = None
        # head_dim derived from layernorm or another clue
        # Actually nhead is stored implicitly; recover from stored a_l shape or
        # just look for it in meta_extra
        nhead = int(meta_extra.get("num_heads", 4))
        # Try to recover from q_lin weight shape (out_dim == hidden_dim so not helpful)
        # Store as saved (default 4)
    else:
        a_l_key = f"{att_prefix}.0.a_l"
        if a_l_key in sd:
            nhead = sd[a_l_key].shape[0]
        else:
            nhead = int(meta_extra.get("num_heads", 4))

    out_dim     = sd["out_proj.weight"].shape[0]
    num_organs  = sd["organ_regression.weight"].shape[0]
    num_severity = sd["organ_classifiers.0.weight"].shape[0]

    # Metapath names from filename
    stem = os.path.splitext(os.path.basename(pt_path))[0]
    # e.g. hanpp_P-O-P  or  hgthan_P-D-P
    mp_match = re.findall(r'P-[A-Z]-P', stem)
    if mp_match:
        metapath_names = mp_match
    else:
        metapath_names = [f"MP{i}" for i in range(num_metapaths)]

    # Saved epoch / metrics if present
    saved_epoch  = meta_extra.get("epoch",  meta_extra.get("best_epoch", None))
    saved_metric = meta_extra.get("val_f1", meta_extra.get("best_f1",    None))

    info = dict(
        model_type=model_type,
        is_hgt=is_hgt,
        att_prefix=att_prefix,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_metapaths=num_metapaths,
        num_heads=nhead,
        head_dim=hidden_dim // nhead,
        num_organs=num_organs,
        num_severity=num_severity,
        metapath_names=metapath_names,
        stem=stem,
        saved_epoch=saved_epoch,
        saved_metric=saved_metric,
        state_dict=sd,
        num_params=sum(p.numel() for p in sd.values() if isinstance(p, torch.Tensor)),
    )
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: rounded box
# ═══════════════════════════════════════════════════════════════════════════════

def draw_box(ax, x, y, w, h, label, sublabel="", color="#4361EE",
             fontsize=10, sub_fontsize=8, alpha=0.92, zorder=3,
             text_color="white", corner_radius=0.015):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={corner_radius}",
                         fc=color, ec="none", alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    # Main label
    ty = y + (h * 0.12 if sublabel else 0)
    ax.text(x, ty, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=text_color, zorder=zorder+1)
    if sublabel:
        ax.text(x, y - h * 0.22, sublabel, ha="center", va="center",
                fontsize=sub_fontsize, color=text_color, alpha=0.88,
                zorder=zorder+1)


def arrow(ax, x0, y0, x1, y1, color=C["arrow"], lw=1.5, head=6, zorder=2,
          label="", label_fontsize=7.5, label_color="#555", rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=f"-|>,head_length=0.012,head_width=0.008",
                    color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                zorder=zorder)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.012, my, label, ha="left", va="center",
                fontsize=label_fontsize, color=label_color)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 – FULL ARCHITECTURE DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def figure_architecture(info: dict, outpath: str, dpi: int = 300):
    """
    Full left-to-right (or top-to-bottom) architecture diagram.

    Layout (Y axis = top→bottom):
      [Input Features]
           ↓
      [Input Projection  (Linear + GELU)]
           ↓  ─────── split into N meta-path branches ───────
      [NodeAtt / HGT-Layer]  ×  num_metapaths
           ↓  ─────── Semantic Attention aggregation ─────────
      [Semantic Attention  (W·tanh + softmax β)]
           ↓
      [Output Projection  (Linear + GELU)]
           ↙                         ↘
    [Organ Severity Classifiers]  [Organ Score Regression]
    Linear(out_dim, 4) × O organs  Linear(out_dim, num_organs)
    """
    mp   = info["num_metapaths"]
    nm   = info["metapath_names"]
    norg = info["num_organs"]
    nhead= info["num_heads"]
    hd   = info["head_dim"]

    fig_w = max(16, 4 * mp + 4)
    fig_h = 18
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    # ── X positions for meta-path columns ────────────────────────────────────
    x_left   = 0.10
    x_right  = 0.90
    x_center = 0.50

    mp_xs = np.linspace(x_left + 0.05, x_right - 0.05, mp)

    # ── Y positions (top→bottom, normalised 0..1) ─────────────────────────────
    Y = dict(
        input   = 0.94,
        proj    = 0.83,
        split   = 0.75,
        nodeatt = 0.63,
        merge   = 0.55,
        sematt  = 0.46,
        outproj = 0.35,
        clf     = 0.17,
        reg     = 0.17,
    )

    bw, bh = 0.18, 0.055   # default box width / height

    # ─────────────────────────────────────────────────────────
    # INPUT
    # ─────────────────────────────────────────────────────────
    draw_box(ax, x_center, Y["input"], 0.30, bh,
             "Patient Feature Vector",
             f"[N  ×  {info['in_dim']}]  (symptoms · organs · diseases)",
             color=C["input"], fontsize=10)

    # ─────────────────────────────────────────────────────────
    # INPUT PROJECTION
    # ─────────────────────────────────────────────────────────
    arrow(ax, x_center, Y["input"]-bh/2,  x_center, Y["proj"]+bh/2)
    draw_box(ax, x_center, Y["proj"], 0.38, bh,
             "Input Projection   Linear + GELU",
             f"[N × {info['in_dim']}]  →  [N × {info['hidden_dim']}]",
             color=C["proj"], fontsize=9.5)

    # ─────────────────────────────────────────────────────────
    # SPLIT ARROW + META-PATH BRANCHES
    # ─────────────────────────────────────────────────────────
    # Horizontal bar at split level
    ax.plot([mp_xs[0], mp_xs[-1]], [Y["split"], Y["split"]],
            color=C["arrow"], lw=1.4, zorder=2)
    # Vertical down from proj
    arrow(ax, x_center, Y["proj"]-bh/2,
              x_center, Y["split"],
              label=f"h  [N × {info['hidden_dim']}]")

    att_label = "HGT Attention\n(Q·K / √d + V)" if info["is_hgt"] else "Node Attention\n(Multi-Head GAT)"
    att_sub   = (f"Q,K,V: Linear({info['hidden_dim']}→{info['hidden_dim']})\n"
                 f"{nhead} heads × {hd}d") if info["is_hgt"] else \
                (f"W: Linear({info['hidden_dim']}→{info['hidden_dim']})\n"
                 f"aₗ,aᵣ: {nhead}×{hd}      Residual + LayerNorm")

    for i, (xi, mp_name) in enumerate(zip(mp_xs, nm)):
        # Vertical stub from split bar
        ax.plot([xi, xi], [Y["split"], Y["nodeatt"]+bh/2],
                color=C["arrow"], lw=1.2, zorder=2)

        # Meta-path label
        ax.text(xi, Y["split"] + 0.025, f"Meta-path\n{mp_name}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=C["meta_head"])

        # Attention block
        draw_box(ax, xi, Y["nodeatt"], 0.22, 0.09,
                 att_label, att_sub,
                 color=C["node_att"] if not info["is_hgt"] else C["hgt"],
                 fontsize=8, sub_fontsize=7)

        # Residual self-loop icon
        ax.annotate("", xy=(xi+0.115, Y["nodeatt"]+0.025),
                    xytext=(xi+0.115, Y["nodeatt"]-0.025),
                    arrowprops=dict(arrowstyle="-|>", color=C["meta_head"],
                                   lw=1.0,
                                   connectionstyle="arc3,rad=-0.6"))
        ax.text(xi+0.135, Y["nodeatt"], "res", fontsize=6,
                color=C["meta_head"], ha="left", va="center")

        # Arrow → merge bar
        ax.plot([xi, xi], [Y["nodeatt"]-bh/2 - 0.005, Y["merge"]],
                color=C["arrow"], lw=1.2, zorder=2)

    # Merge bar
    ax.plot([mp_xs[0], mp_xs[-1]], [Y["merge"], Y["merge"]],
            color=C["arrow"], lw=1.4, zorder=2)

    # Representative Z_i label
    for xi, mp_name in zip(mp_xs, nm):
        ax.text(xi, Y["merge"] - 0.022, f"Z_{mp_name}",
                ha="center", va="top", fontsize=7.5,
                color=C["text_mid"], style="italic")

    # ─────────────────────────────────────────────────────────
    # SEMANTIC ATTENTION
    # ─────────────────────────────────────────────────────────
    arrow(ax, x_center, Y["merge"],
              x_center, Y["sematt"]+bh/2)
    draw_box(ax, x_center, Y["sematt"], 0.48, 0.08,
             "Semantic Attention   (β-weighted sum)",
             f"W·tanh(Zᵢ) · q   →   softmax β   →   Σ βᵢ Zᵢ          "
             f"[N × {info['hidden_dim']}]",
             color=C["semantic"], fontsize=9.5, sub_fontsize=7.5)

    # Beta annotation
    ax.text(x_center + 0.26, Y["sematt"], "β ∈ ℝ^" + str(mp),
            ha="left", va="center", fontsize=9, color=C["semantic"],
            fontweight="bold")

    # ─────────────────────────────────────────────────────────
    # OUTPUT PROJECTION
    # ─────────────────────────────────────────────────────────
    arrow(ax, x_center, Y["sematt"]-bh/2 - 0.01,
              x_center, Y["outproj"]+bh/2,
              label=f"[N × {info['hidden_dim']}]")
    draw_box(ax, x_center, Y["outproj"], 0.38, bh,
             "Output Projection   Linear + GELU",
             f"[N × {info['hidden_dim']}]  →  [N × {info['out_dim']}]",
             color=C["out_proj"], fontsize=9.5)

    # ─────────────────────────────────────────────────────────
    # FORK: classifiers (left) + regression (right)
    # ─────────────────────────────────────────────────────────
    x_clf = 0.28
    x_reg = 0.72
    fork_y = Y["outproj"] - bh/2 - 0.02

    # Left fork line
    ax.plot([x_center, x_clf], [fork_y, fork_y],
            color=C["arrow"], lw=1.4, zorder=2)
    ax.annotate("", xy=(x_clf, Y["clf"]+bh/2+0.01), xytext=(x_clf, fork_y),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"], lw=1.4))
    # Right fork line
    ax.plot([x_center, x_reg], [fork_y, fork_y],
            color=C["arrow"], lw=1.4, zorder=2)
    ax.annotate("", xy=(x_reg, Y["reg"]+bh/2+0.01), xytext=(x_reg, fork_y),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"], lw=1.4))

    # Dropout annotation
    ax.text(x_center, fork_y - 0.012, "Dropout",
            ha="center", va="top", fontsize=8.5, color=C["text_mid"],
            style="italic")

    # ── Organ Severity Classifier block ──────────────────────────────────────
    draw_box(ax, x_clf, Y["clf"], 0.35, 0.10,
             f"Organ Severity Classifiers  ×{norg}",
             f"Linear({info['out_dim']} → {info['num_severity']})  per organ\n"
             f"Output: [N × {norg} × {info['num_severity']}]   (softmax → class 0-{info['num_severity']-1})",
             color=C["clf"], fontsize=8.5, sub_fontsize=7.5)

    # Output label below clf
    ax.text(x_clf, Y["clf"] - bh/2 - 0.06,
            f"Severity Predictions\n[N × {norg} × {info['num_severity']}]",
            ha="center", va="top", fontsize=9, color=C["clf"], fontweight="bold")
    draw_box(ax, x_clf, Y["clf"] - bh/2 - 0.11, 0.28, 0.04,
             "0 = Normal  1 = Mild  2 = Moderate  3 = Severe", "",
             color=C["clf"], alpha=0.25, fontsize=7, text_color=C["clf"])

    # ── Organ Damage Regression block ────────────────────────────────────────
    draw_box(ax, x_reg, Y["reg"], 0.30, 0.10,
             "Organ Damage Regression",
             f"Linear({info['out_dim']} → {norg})\n"
             f"Output: [N × {norg}]   (sigmoid → score ∈ [0,1])",
             color=C["reg"], fontsize=8.5, sub_fontsize=7.5)

    ax.text(x_reg, Y["reg"] - bh/2 - 0.06,
            f"Damage Scores\n[N × {norg}]",
            ha="center", va="top", fontsize=9, color=C["reg"], fontweight="bold")

    # ─────────────────────────────────────────────────────────
    # DIMENSION LEGEND  (right side)
    # ─────────────────────────────────────────────────────────
    lx, ly, ldy = 0.91, 0.88, 0.038
    ax.text(lx, ly + ldy, "Dimensions", ha="left", va="bottom",
            fontsize=9, fontweight="bold", color=C["text_dark"])
    dim_rows = [
        (f"in_dim",      f"{info['in_dim']}",     C["input"]),
        (f"hidden_dim",  f"{info['hidden_dim']}", C["proj"]),
        (f"out_dim",     f"{info['out_dim']}",    C["out_proj"]),
        (f"num_heads",   f"{nhead}",              C["node_att"]),
        (f"head_dim",    f"{hd}",                 C["meta_head"]),
        (f"num_organs",  f"{norg}",               C["clf"]),
        (f"num_severity",f"{info['num_severity']}",C["clf"]),
        (f"meta-paths",  f"{mp} ({', '.join(nm)})", C["semantic"]),
    ]
    for di, (k, v, col) in enumerate(dim_rows):
        y_row = ly - di * ldy
        rect = FancyBboxPatch((lx - 0.005, y_row - ldy*0.4),
                              0.115, ldy * 0.8,
                              boxstyle="round,pad=0.003",
                              fc=col, alpha=0.15, ec="none")
        ax.add_patch(rect)
        ax.text(lx + 0.002, y_row, k, ha="left", va="center",
                fontsize=7.5, color=col, fontweight="bold")
        ax.text(lx + 0.075, y_row, v, ha="left", va="center",
                fontsize=7.5, color=C["text_mid"])

    # ─────────────────────────────────────────────────────────
    # PARAMETER COUNT TABLE
    # ─────────────────────────────────────────────────────────
    sd  = info["state_dict"]
    param_groups = {
        "Input Projection":        ["project.weight", "project.bias"],
        "Node/HGT Attention":      [k for k in sd if info["att_prefix"] in k],
        "Semantic Attention":      [k for k in sd if "semantic_att" in k],
        "Output Projection":       ["out_proj.weight", "out_proj.bias"],
        "Organ Classifiers":       [k for k in sd if "organ_classifiers" in k],
        "Organ Regression":        ["organ_regression.weight", "organ_regression.bias"],
    }
    px, py, pdy = 0.02, 0.30, 0.030
    ax.text(px, py + pdy, "Layer Parameters", ha="left", va="bottom",
            fontsize=8.5, fontweight="bold", color=C["text_dark"])
    total = 0
    for gi, (grp_name, klist) in enumerate(param_groups.items()):
        cnt = sum(sd[k].numel() for k in klist if k in sd)
        total += cnt
        ax.text(px, py - gi*pdy, f"  {grp_name}:", ha="left", va="center",
                fontsize=7.2, color=C["text_mid"])
        ax.text(px + 0.17, py - gi*pdy, f"{cnt:,}", ha="right", va="center",
                fontsize=7.2, color=C["text_dark"], fontweight="bold")
    gi += 1
    ax.text(px, py - gi*pdy - 0.005, "  TOTAL:", ha="left", va="center",
            fontsize=8, color=C["text_dark"], fontweight="bold")
    ax.text(px + 0.17, py - gi*pdy - 0.005, f"{total:,}", ha="right", va="center",
            fontsize=8, color=C["clf"], fontweight="bold")

    # ─────────────────────────────────────────────────────────
    # TITLE
    # ─────────────────────────────────────────────────────────
    ep_str = f"  |  epoch {info['saved_epoch']}" if info["saved_epoch"] is not None else ""
    f1_str = f"  |  val F1 = {info['saved_metric']:.4f}" if info["saved_metric"] is not None else ""
    ax.set_title(
        f"{info['model_type']} Architecture — {info['stem']}{ep_str}{f1_str}\n"
        f"Total parameters: {total:,}",
        fontsize=13, fontweight="bold", color=C["text_dark"], pad=10
    )

    plt.tight_layout(pad=0.4)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Architecture diagram saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 – ATTENTION MECHANISM DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

def figure_attention_detail(info: dict, outpath: str, dpi: int = 300):
    """
    Side-by-side detail:
      Left  – Multi-head Node-Level Attention (HAN++) or HGT (HGT-HAN)
      Right – Semantic Attention (shared by both)
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.patch.set_facecolor(C["bg"])

    nhead = info["num_heads"]
    hd    = info["head_dim"]
    D     = info["hidden_dim"]
    mp    = info["num_metapaths"]

    # ── Left: Node-level attention ──────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if info["is_hgt"]:
        ax.set_title("HGT-style Multi-Head Attention\n(per meta-path branch)",
                     fontsize=12, fontweight="bold", color=C["hgt"])
        _draw_hgt_detail(ax, info)
    else:
        ax.set_title("HAN++ Multi-Head Node Attention\n(per meta-path branch)",
                     fontsize=12, fontweight="bold", color=C["node_att"])
        _draw_han_detail(ax, info)

    # ── Right: Semantic attention ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(C["bg"])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("Semantic-Level Attention\n(meta-path aggregation)",
                  fontsize=12, fontweight="bold", color=C["semantic"])
    _draw_semantic_detail(ax2, info)

    fig.suptitle(
        f"Attention Mechanism Details — {info['model_type']} / {info['stem']}",
        fontsize=13, fontweight="bold", color=C["text_dark"], y=1.01
    )
    plt.tight_layout(pad=1.5)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Attention detail saved → {outpath}")


def _draw_han_detail(ax, info):
    nhead = info["num_heads"]
    hd    = info["head_dim"]
    D     = info["hidden_dim"]

    draw_box(ax, 0.5, 0.92, 0.5, 0.06,
             f"h  [N × {D}]  (projected features)", "",
             color=C["input"], fontsize=9)

    # W projection
    arrow(ax, 0.5, 0.89, 0.5, 0.81)
    draw_box(ax, 0.5, 0.79, 0.44, 0.05,
             f"W · h   →   [N × {D}]", f"Linear({D}, {D}, bias=False)",
             color=C["proj"], fontsize=8.5, sub_fontsize=7.5)

    # Split into heads
    head_xs = np.linspace(0.15, 0.85, nhead)
    ax.plot([head_xs[0], head_xs[-1]], [0.70, 0.70],
            color=C["arrow"], lw=1.2)
    arrow(ax, 0.5, 0.765, 0.5, 0.70)
    ax.text(0.5, 0.68, f"Split into {nhead} heads  [each: N × {hd}]",
            ha="center", va="top", fontsize=7.5, color=C["text_mid"])

    for i, hx in enumerate(head_xs):
        ax.plot([hx, hx], [0.70, 0.62], color=C["arrow"], lw=1.0)

        draw_box(ax, hx, 0.58, 0.14, 0.06,
                 f"Head {i+1}", f"aₗ [{hd}],  aᵣ [{hd}]\nLeakyReLU → softmax α",
                 color=C["node_att"], fontsize=7, sub_fontsize=6)

        ax.plot([hx, hx], [0.55, 0.47], color=C["arrow"], lw=1.0)
        draw_box(ax, hx, 0.44, 0.14, 0.05,
                 f"αₕ · Wh_j", f"[N × {hd}]",
                 color=C["meta_head"], fontsize=7, sub_fontsize=6.5)

    # Concat
    ax.plot([head_xs[0], head_xs[-1]], [0.415, 0.415],
            color=C["arrow"], lw=1.2)
    arrow(ax, 0.5, 0.415, 0.5, 0.345)
    draw_box(ax, 0.5, 0.315, 0.44, 0.05,
             f"Concat heads   [N × {D}]  ({nhead} heads × {hd}d = {D}d)", "",
             color=C["proj"], fontsize=8)

    # Residual
    arrow(ax, 0.5, 0.29, 0.5, 0.215)
    draw_box(ax, 0.5, 0.19, 0.44, 0.05,
             "Residual Add + GELU", f"h_out = GELU(Z_cat + res_proj(h))",
             color=C["gelu"], fontsize=8, sub_fontsize=7.5)

    arrow(ax, 0.5, 0.165, 0.5, 0.10)
    draw_box(ax, 0.5, 0.075, 0.36, 0.05,
             f"LayerNorm   →   Z_ϕ  [N × {D}]", "",
             color=C["layernorm"], fontsize=8)


def _draw_hgt_detail(ax, info):
    nhead = info["num_heads"]
    hd    = info["head_dim"]
    D     = info["hidden_dim"]

    draw_box(ax, 0.5, 0.92, 0.5, 0.06,
             f"h  [N × {D}]  (projected features)", "",
             color=C["input"], fontsize=9)

    # Q K V projections
    arrow(ax, 0.5, 0.89, 0.5, 0.82)
    for xi, lbl, col in [(0.22, "Q = Linear(h)", C["clf"]),
                          (0.50, "K = Linear(h)", C["meta_head"]),
                          (0.78, "V = Linear(h)", C["semantic"])]:
        ax.plot([xi, xi], [0.82, 0.76], color=C["arrow"], lw=1.1)
        draw_box(ax, xi, 0.73, 0.22, 0.05, lbl, f"[N × {D}]",
                 color=col, fontsize=8, sub_fontsize=7)

    # Horizontal bar
    ax.plot([0.22, 0.78], [0.82, 0.82], color=C["arrow"], lw=1.1)

    # Multi-head scaled dot product
    arrow(ax, 0.5, 0.705, 0.5, 0.635)
    draw_box(ax, 0.5, 0.605, 0.52, 0.06,
             "Scaled Dot-Product Attention   " + f"({nhead} heads × {hd}d)",
             f"scores = (Q · Kᵀ) / √{hd}   →   softmax α   →   α · V",
             color=C["node_att"], fontsize=8.5, sub_fontsize=7.5)

    # Mask
    ax.text(0.82, 0.605, "Mask padding\n(−1e9)", ha="left", va="center",
            fontsize=7.5, color=C["text_mid"])

    arrow(ax, 0.5, 0.575, 0.5, 0.49)
    draw_box(ax, 0.5, 0.46, 0.44, 0.05,
             f"FC(Concat heads)   [N × {D}]", f"Linear({D}, {D})",
             color=C["proj"], fontsize=8)
    arrow(ax, 0.5, 0.435, 0.5, 0.36)
    draw_box(ax, 0.5, 0.33, 0.44, 0.05,
             "Residual Add + GELU", "h_out = GELU(FC_out + res_proj(h))",
             color=C["gelu"], fontsize=8, sub_fontsize=7.5)
    arrow(ax, 0.5, 0.305, 0.5, 0.24)
    draw_box(ax, 0.5, 0.21, 0.36, 0.05,
             f"LayerNorm   →   Z_ϕ  [N × {D}]", "",
             color=C["layernorm"], fontsize=8)


def _draw_semantic_detail(ax, info):
    mp  = info["num_metapaths"]
    nm  = info["metapath_names"]
    D   = info["hidden_dim"]

    mp_xs = np.linspace(0.12, 0.88, mp)

    # Input Z_i
    for xi, name in zip(mp_xs, nm):
        draw_box(ax, xi, 0.92, 0.14, 0.05,
                 f"Z_{name}", f"[N × {D}]",
                 color=C["meta_head"], fontsize=8, sub_fontsize=6.5)

    # Merge bar
    ax.plot([mp_xs[0], mp_xs[-1]], [0.84, 0.84], color=C["arrow"], lw=1.2)
    for xi in mp_xs:
        ax.plot([xi, xi], [0.895, 0.84], color=C["arrow"], lw=1.0)

    arrow(ax, 0.5, 0.84, 0.5, 0.77)

    # W·tanh
    draw_box(ax, 0.5, 0.74, 0.54, 0.055,
             f"W · tanh(Zᵢ)  →  score", f"Linear({D}, {D}) → tanh → mean pooling → w ∈ ℝ",
             color=C["semantic"], fontsize=8.5, sub_fontsize=7.5)

    # Stack scores
    arrow(ax, 0.5, 0.712, 0.5, 0.645)
    draw_box(ax, 0.5, 0.615, 0.40, 0.055,
             f"Stack  [{mp}  scalar weights]", f"wᵢ = mean( W·tanh(Zᵢ) · q )",
             color=C["semantic"], fontsize=8.5, sub_fontsize=7.5)

    arrow(ax, 0.5, 0.587, 0.5, 0.52)
    draw_box(ax, 0.5, 0.49, 0.36, 0.05,
             f"Softmax  →  β ∈ ℝ^{mp}", "Importance weight per meta-path",
             color=C["softmax"], fontsize=8.5, sub_fontsize=7.5)

    # Weighted sum
    arrow(ax, 0.5, 0.465, 0.5, 0.395)
    draw_box(ax, 0.5, 0.365, 0.44, 0.055,
             "Weighted Sum   Z_final", f"Σ βᵢ · Zᵢ   →   [N × {D}]",
             color=C["semantic"], fontsize=8.5, sub_fontsize=7.5)

    # Dropout
    arrow(ax, 0.5, 0.337, 0.5, 0.28)
    draw_box(ax, 0.5, 0.255, 0.32, 0.046,
             "Dropout", "",
             color=C["text_light"], alpha=0.6, fontsize=8.5, text_color="#333")

    arrow(ax, 0.5, 0.232, 0.5, 0.165)
    draw_box(ax, 0.5, 0.14, 0.44, 0.05,
             f"Z_final   →   Output Projection", f"[N × {D}]",
             color=C["out_proj"], fontsize=8.5, sub_fontsize=7.5)

    # Beta annotation
    for i, (xi, name) in enumerate(zip(mp_xs, nm)):
        ax.text(xi, 0.46, f"β_{name}", ha="center", va="center",
                fontsize=8, color=C["semantic"], fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 – TENSOR DATA-FLOW DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════════

def figure_dataflow(info: dict, outpath: str, dpi: int = 300):
    """
    Minimal but precise tensor-shape annotated flowchart.
    Every arrow is labelled with the exact tensor shape tensor.
    """
    mp   = info["num_metapaths"]
    nm   = info["metapath_names"]
    D    = info["hidden_dim"]
    dout = info["out_dim"]
    O    = info["num_organs"]
    S    = info["num_severity"]
    N    = "N"

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def rbox(x, y, w, h, label, color, fontsize=9):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           fc=color, ec="none", alpha=0.90, zorder=3)
        ax.add_patch(p)
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color="white", zorder=4, wrap=True)

    def arr(x0, y0, x1, y1, shape_label, col=C["arrow"]):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6),
                    zorder=2)
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.15, my, shape_label, ha="left", va="center",
                fontsize=8, color="#444",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    cx = 7.0  # center x
    bw, bh = 4.0, 0.5

    # Rows (y values for top-left of box)
    rows = [9.0, 7.8, 6.5, 5.2, 3.9, 2.6]
    labels = [
        (f"Input: Patient Features", C["input"]),
        (f"Input Projection  Linear({info['in_dim']}→{D}) + GELU", C["proj"]),
        (f"Node Attention  ×{mp} branches  [{info['model_type']}]", C["node_att"]),
        (f"Semantic Attention  (β-weighted sum over {mp} meta-paths)", C["semantic"]),
        (f"Output Projection  Linear({D}→{dout}) + GELU", C["out_proj"]),
        (f"Organ Severity Classifiers  ×{O}  +  Damage Regression", C["clf"]),
    ]
    shapes = [
        f"[{N} × {info['in_dim']}]",
        f"[{N} × {D}]",
        f"{mp} × [{N} × {D}]  (Z_ϕ per meta-path)",
        f"[{N} × {D}]  (β ∈ ℝ^{mp})",
        f"[{N} × {dout}]",
        f"[{N} × {O} × {S}]  + [{N} × {O}]",
    ]

    for i, ((lbl, col), shape) in enumerate(zip(labels, shapes)):
        rbox(cx - bw/2, rows[i], bw, bh, lbl, col)
        if i < len(rows) - 1:
            arr(cx, rows[i], cx, rows[i+1] + bh + 0.02,
                shape)

    # Meta-path branch annotation
    mp_xs_out = np.linspace(cx - 2.0, cx + 2.0, mp)
    for xi, name in zip(mp_xs_out, nm):
        ax.annotate("", xy=(xi, rows[2] + bh + 0.1),
                    xytext=(cx, rows[2] + bh),
                    arrowprops=dict(arrowstyle="-|>", color=C["meta_head"], lw=1.1))
        ax.text(xi, rows[2] + bh + 0.3, name, ha="center",
                fontsize=8.5, color=C["meta_head"], fontweight="bold")

    ax.set_title(
        f"Tensor Data-Flow — {info['model_type']}  ({info['stem']})",
        fontsize=13, fontweight="bold", color=C["text_dark"], pad=10
    )
    plt.tight_layout(pad=0.5)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Data-flow diagram saved → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Architecture visualization for HAN++ / HGT-HAN .pt models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("pt_files", nargs="+",
                   help="One or more .pt model files to visualize")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: ../output/model_viz next to each .pt)")
    p.add_argument("--dpi",    type=int, default=300)
    p.add_argument("--no-att", action="store_true", help="Skip attention detail figure")
    p.add_argument("--no-flow",action="store_true", help="Skip data-flow figure")
    return p.parse_args()


def main():
    args = parse_args()

    for pt_path in args.pt_files:
        if not os.path.exists(pt_path):
            print(f"❌  Not found: {pt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing: {os.path.basename(pt_path)}")
        print(f"{'='*60}")

        # Determine output directory
        if args.outdir:
            outdir = args.outdir
        else:
            han_root = os.path.dirname(os.path.dirname(os.path.abspath(pt_path)))
            outdir = os.path.join(han_root, "output", "model_viz")
        os.makedirs(outdir, exist_ok=True)

        # Introspect
        print("  Introspecting state-dict …")
        info = load_and_introspect(pt_path)

        print(f"  Model type  : {info['model_type']}")
        print(f"  Meta-paths  : {info['metapath_names']}")
        print(f"  in_dim      : {info['in_dim']}")
        print(f"  hidden_dim  : {info['hidden_dim']}")
        print(f"  out_dim     : {info['out_dim']}")
        print(f"  num_heads   : {info['num_heads']}  (head_dim={info['head_dim']})")
        print(f"  num_organs  : {info['num_organs']}")
        print(f"  total params: {info['num_params']:,}")

        stem    = info["stem"]
        out_pre = os.path.join(outdir, stem)

        # Fig 1 – architecture
        figure_architecture(info, f"{out_pre}_architecture.png", dpi=args.dpi)

        # Fig 2 – attention detail
        if not args.no_att:
            figure_attention_detail(info, f"{out_pre}_attention.png", dpi=args.dpi)

        # Fig 3 – data-flow
        if not args.no_flow:
            figure_dataflow(info, f"{out_pre}_dataflow.png", dpi=args.dpi)

    print(f"\n🎉  Done!  Figures saved in: {outdir}")
    print("    <stem>_architecture.png  ← Full architecture (main paper figure)")
    print("    <stem>_attention.png     ← Attention mechanism detail")
    print("    <stem>_dataflow.png      ← Tensor shape data-flow")


if __name__ == "__main__":
    main()
