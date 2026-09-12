"""
pip_pocket_geometry.py
=======================
For every frame, computes the COM-to-COM distances between 3 residue clusters
around the PIP2 binding pocket:

  Cluster 1 (N-term):  ARG60 + LYS64
  Cluster 2 (C-term):  LYS199 + LYS200
  Cluster 3 (Middle):  LYS90

Pairwise distances computed:
  d12: Cluster1 ↔ Cluster2  (N-term ↔ C-term)
  d13: Cluster1 ↔ Cluster3  (N-term ↔ LYS90)
  d23: Cluster2 ↔ Cluster3  (C-term ↔ LYS90)

Output (per subunit, in results_dir/pip_pocket_geometry/):
    per_frame.csv          - frame, subunit, d12, d13, d23
    summary.csv            - mean, median, std, min, max for each distance
    boxplot.png            - boxplot with mean + median annotated
    violin.png             - violin plot

Called from main.py:

    from analysis.pip_pocket_geometry import run_pip_pocket_geometry

    run_pip_pocket_geometry(
        u            = u,
        mapping_json = mapping_json,
        results_dir  = results_dir,
        start_frame  = start_frame,
        end_frame    = end_frame,
        channel_type = args.channel_type,
    )
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

CLUSTER_NAMES = {
    "c1": "N-term\n(ARG60+LYS64)",
    "c2": "C-term\n(LYS199+LYS200)",
    "c3": "Middle\n(LYS90)",
}
DIST_NAMES = {
    "d12": "N-term ↔ C-term",
    "d13": "N-term ↔ LYS90",
    "d23": "C-term ↔ LYS90",
}
DIST_COLORS = {
    "d12": "#E53935",  # red
    "d13": "#1E88E5",  # blue
    "d23": "#43A047",  # green
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _com(atom_groups: list) -> np.ndarray:
    """Compute center of mass (equal weights) from a list of AtomGroups."""
    positions = np.vstack([ag.positions for ag in atom_groups])
    return positions.mean(axis=0)


def _boxplot(data: dict, title: str, out_path: Path, channel_type: str):
    """
    Boxplot for d12, d13, d23.
    Shows mean (diamond) and median (line) annotated.
    """
    keys   = [k for k in ["d12", "d13", "d23"] if k in data and len(data[k]) > 0]
    values = [data[k] for k in keys]
    labels = [DIST_NAMES[k] for k in keys]
    colors = [DIST_COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 6))
    bp = ax.boxplot(values, patch_artist=True, labels=labels,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # annotate mean and median
    for i, (key, vals) in enumerate(zip(keys, values)):
        mean   = np.mean(vals)
        median = np.median(vals)
        # mean as diamond
        ax.scatter(i+1, mean, marker="D", color="white", s=60,
                   zorder=5, edgecolors="black", linewidths=1)
        # text annotations
        ax.text(i+1+0.15, mean,   f"μ={mean:.1f}",   va="center", fontsize=10, color="black")
        ax.text(i+1+0.15, median, f"m={median:.1f}", va="center", fontsize=10, color="grey")

    ax.set_ylabel("Distance (Å)", fontsize=14)
    ax.set_title(f"{title}\n{channel_type}", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _violinplot(data: dict, title: str, out_path: Path, channel_type: str):
    """Violin plot for d12, d13, d23."""
    keys   = [k for k in ["d12", "d13", "d23"] if k in data and len(data[k]) > 0]
    values = [data[k] for k in keys]
    labels = [DIST_NAMES[k] for k in keys]
    colors = [DIST_COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 6))
    parts = ax.violinplot(values, positions=range(1, len(keys)+1),
                          showmeans=True, showmedians=True)

    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("white")

    ax.set_xticks(range(1, len(keys)+1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Distance (Å)", fontsize=14)
    ax.set_title(f"{title}\n{channel_type}", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_summary(data: dict, n_frames: int) -> pd.DataFrame:
    rows = []
    for key in ["d12", "d13", "d23"]:
        if key not in data or len(data[key]) == 0:
            continue
        vals = np.array(data[key])
        rows.append({
            "distance":    key,
            "label":       DIST_NAMES[key],
            "n_frames":    n_frames,
            "mean":        round(float(vals.mean()), 3),
            "median":      round(float(np.median(vals)), 3),
            "std":         round(float(vals.std()), 3),
            "min":         round(float(vals.min()), 3),
            "max":         round(float(vals.max()), 3),
            "q25":         round(float(np.percentile(vals, 25)), 3),
            "q75":         round(float(np.percentile(vals, 75)), 3),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_pip_pocket_geometry(
    u,
    mapping_json:  str,
    results_dir,
    start_frame:   int,
    end_frame:     int,
    channel_type:  str,
):
    results_dir = Path(results_dir)
    out_dir     = results_dir / "pip_pocket_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(mapping_json) as f:
        mapping = json.load(f)

    print("\n" + "=" * 60)
    print("PIP POCKET GEOMETRY ANALYSIS")
    print(f"  Channel type : {channel_type}")
    print(f"  Frames       : {start_frame} → {end_frame}")
    print(f"  Clusters     : C1=ARG60+LYS64 | C2=LYS199+LYS200 | C3=LYS90")
    print("=" * 60)

    # ── pre-select atoms for all PIPs ────────────────────────────────────────
    pip_configs = []
    for pip_resid_str, subunit_data in mapping.items():
        pip_resid = int(pip_resid_str)
        subunit   = list(subunit_data.keys())[0]
        res_map   = subunit_data[subunit]

        c1_resids = []  # ARG60 + LYS64
        c2_resids = []  # LYS199 + LYS200
        c3_resids = []  # LYS90

        for res_name, info in res_map.items():
            raw = info["raw_resid"]
            pdb = info["pdb_number"]
            if pdb in [60, 64]:   c1_resids.append(raw)
            elif pdb in [199, 200]: c2_resids.append(raw)
            elif pdb == 90:       c3_resids.append(raw)

        if not c1_resids or not c2_resids or not c3_resids:
            print(f"  WARNING: incomplete clusters for PIP {pip_resid} subunit {subunit} — skipping")
            continue

        # single CA AtomGroup per cluster — selected once, positions update each frame
        def _ca_ag(resids):
            sel = " or ".join([f"resid {r}" for r in resids])
            return u.select_atoms(f"({sel}) and name CA")

        c1_ag = _ca_ag(c1_resids)
        c2_ag = _ca_ag(c2_resids)
        c3_ag = _ca_ag(c3_resids)

        print(f"  PIP {pip_resid} — subunit {subunit}: "
              f"C1={c1_ag.n_atoms} CA, C2={c2_ag.n_atoms} CA, C3={c3_ag.n_atoms} CA")
        pip_configs.append({
            "pip_resid": pip_resid,
            "subunit":   subunit,
            "c1":        c1_ag,
            "c2":        c2_ag,
            "c3":        c3_ag,
        })

    if not pip_configs:
        print("No valid PIP configurations found.")
        return

    # ── single trajectory traversal ──────────────────────────────────────────
    n_frames = end_frame - start_frame + 1
    all_rows = []

    for ts in tqdm(u.trajectory[start_frame:end_frame + 1],
                   total=n_frames, desc="  Trajectory", leave=True):
        frame = ts.frame
        for cfg in pip_configs:
            com1 = cfg["c1"].positions.mean(axis=0)
            com2 = cfg["c2"].positions.mean(axis=0)
            com3 = cfg["c3"].positions.mean(axis=0)

            d12 = round(float(np.linalg.norm(com1 - com2)), 3)
            d13 = round(float(np.linalg.norm(com1 - com3)), 3)
            d23 = round(float(np.linalg.norm(com2 - com3)), 3)

            all_rows.append({
                "frame":   frame,
                "pip":     cfg["pip_resid"],
                "subunit": cfg["subunit"],
                "d12_Nterm_Cterm": d12,
                "d13_Nterm_LYS90": d13,
                "d23_Cterm_LYS90": d23,
            })

    if not all_rows:
        print("No data collected.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "per_frame.csv", index=False)
    print(f"\n  Per-frame CSV → {out_dir / 'per_frame.csv'}")

    # ── per subunit outputs ───────────────────────────────────────────────────
    if channel_type == "G2":
        subunit_order = ["2a", "2b", "2c", "2d"]
    else:
        subunit_order = ["2a", "2b", "2c", "G1"]
        
    for sub in subunit_order:
        df_sub = df[df["subunit"] == sub]
        if df_sub.empty:
            continue

        sub_dir = out_dir / f"subunit_{sub}"
        sub_dir.mkdir(exist_ok=True)

        data = {
            "d12": df_sub["d12_Nterm_Cterm"].tolist(),
            "d13": df_sub["d13_Nterm_LYS90"].tolist(),
            "d23": df_sub["d23_Cterm_LYS90"].tolist(),
        }
        n = len(df_sub)

        # per-frame CSV
        df_sub.to_csv(sub_dir / "per_frame.csv", index=False)

        # summary
        summary = _make_summary(data, n)
        summary.to_csv(sub_dir / "summary.csv", index=False)

        # plots
        title = f"PIP2 pocket geometry — Subunit {sub}"
        _boxplot(data, title, sub_dir / "boxplot.png", channel_type)
        _violinplot(data, title, sub_dir / "violin.png", channel_type)

        print(f"  Subunit {sub}: mean d12={np.mean(data['d12']):.1f}Å, "
              f"d13={np.mean(data['d13']):.1f}Å, d23={np.mean(data['d23']):.1f}Å")

    print(f"\n✓ PIP pocket geometry complete → {out_dir}")