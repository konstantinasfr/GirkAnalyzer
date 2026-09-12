"""
pip_headgroup_distances.py
===========================
VERSION: 2.0
===========================
For every frame, computes minimum distances from each PIP2 P atom (P1, P4, P5)
to each residue cluster (N-term, C-term, LYS90, LYS194).

Distance definition:
  dist(P_atom -> cluster) = min distance from P atom to any heavy atom
                            of any residue in the cluster

Clusters:
  Nterm  : ARG60 + LYS64
  Cterm  : LYS199 + LYS200
  LYS90  : LYS90
  LYS194 : LYS194

12 distances per frame:
  P1->Nterm, P1->Cterm, P1->LYS90, P1->LYS194
  P4->Nterm, P4->Cterm, P4->LYS90, P4->LYS194
  P5->Nterm, P5->Cterm, P5->LYS90, P5->LYS194

Also analyses 4 gate frame subsets:
  - ASN gate frames     (end2   : ion crosses ASN gate, region 2->3)
  - ASN gate -1 frames  (end2-1 : frame just before ASN gate crossing)
  - HBC gate frames     (end3   : ion crosses HBC gate, region 3->4)
  - HBC gate -1 frames  (end3-1 : frame just before HBC gate crossing)

Output (in results_dir/pip_headgroup_distances/):
  per_frame.csv
  subunit_2a/
      per_frame.csv
      summary.csv
      histograms/
      boxplot.png
      violin.png
  asn_gate_frames/     (same structure)
  asn_gate_m1_frames/  (same structure)
  hbc_gate_frames/     (same structure)
  hbc_gate_m1_frames/  (same structure)

Called from main.py:

    from analysis.pip_headgroup_distances import run_pip_headgroup_distances

    run_pip_headgroup_distances(
        u            = u,
        mapping_json = mapping_json,
        results_dir  = results_dir,
        start_frame  = start_frame,
        end_frame    = end_frame,
        channel_type = args.channel_type,
        end2_all     = end2_all,
        end3_all     = end3_all,
    )
"""

from __future__ import annotations

import json
import operator
import warnings
from functools import reduce
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array
from tqdm import tqdm

warnings.filterwarnings("ignore")

P_ATOMS  = ["P1", "P4", "P5"]
CLUSTERS = ["Nterm", "Cterm", "LYS90", "LYS194"]

CLUSTER_LABELS = {
    "Nterm":  "N-term\n(ARG60+LYS64)",
    "Cterm":  "C-term\n(LYS199+LYS200)",
    "LYS90":  "LYS90",
    "LYS194": "LYS194",
}
P_COLORS = {"P1": "#4CAF50", "P4": "#FF9800", "P5": "#2196F3"}

DIST_COLS = [f"{p}_{c}" for p in P_ATOMS for c in CLUSTERS]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _min_dist_p_to_cluster(p_pos: np.ndarray, cluster_positions: np.ndarray) -> float:
    dists = distance_array(p_pos.reshape(1, 3), cluster_positions)[0]
    return float(dists.min())


def _make_summary(df: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    rows = []
    for col in DIST_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        p_atom, cluster = col.split("_", 1)
        rows.append({
            "distance": col,
            "p_atom":   p_atom,
            "cluster":  cluster,
            "n_frames": len(vals),
            "mean":     round(float(vals.mean()), 3),
            "median":   round(float(np.median(vals)), 3),
            "std":      round(float(vals.std()), 3),
            "min":      round(float(vals.min()), 3),
            "max":      round(float(vals.max()), 3),
            "q25":      round(float(np.percentile(vals, 25)), 3),
            "q75":      round(float(np.percentile(vals, 75)), 3),
        })
    return pd.DataFrame(rows)


def _histograms(df: pd.DataFrame, out_dir: Path, channel_type: str,
                subunit: str, label: str = "all frames"):
    hist_dir = out_dir / "histograms"
    hist_dir.mkdir(exist_ok=True)

    for col in DIST_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue
        p_atom, cluster = col.split("_", 1)
        color = P_COLORS.get(p_atom, "#9E9E9E")

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(vals, bins=80, color=color, edgecolor="none", alpha=0.85)
        ax.axvline(np.mean(vals), color="black", linestyle="--",
                   linewidth=1.5, label=f"mean={np.mean(vals):.1f}A")
        ax.axvline(np.median(vals), color="grey", linestyle=":",
                   linewidth=1.5, label=f"median={np.median(vals):.1f}A")
        ax.set_xlabel("Distance (A)", fontsize=11)
        ax.set_ylabel("Frames", fontsize=11)
        ax.set_title(f"{p_atom} -> {CLUSTER_LABELS.get(cluster, cluster)}\n"
                     f"{channel_type} | Subunit {subunit} [{label}]", fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        fig.savefig(hist_dir / f"hist_{col}.png", dpi=150)
        plt.close(fig)


def _boxplot(df: pd.DataFrame, out_path: Path, channel_type: str,
             subunit: str, n_frames: int, label: str = "all frames"):
    """4x3 grid: rows=clusters, cols=P atoms."""
    fig, axes = plt.subplots(len(CLUSTERS), len(P_ATOMS),
                             figsize=(12, 14), sharey=False)
    fig.suptitle(f"PIP2 headgroup distances -- {channel_type} | Subunit {subunit}\n"
                 f"[{label}, {n_frames} frames]", fontsize=13)

    for i, cluster in enumerate(CLUSTERS):
        for j, p_atom in enumerate(P_ATOMS):
            col = f"{p_atom}_{cluster}"
            ax  = axes[i][j]
            if col not in df.columns:
                ax.set_visible(False)
                continue
            vals  = df[col].dropna().values
            color = P_COLORS.get(p_atom, "#9E9E9E")

            bp = ax.boxplot([vals], patch_artist=True,
                            medianprops=dict(color="black", linewidth=2),
                            flierprops=dict(marker=".", markersize=1, alpha=0.2))
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)

            mean   = np.mean(vals)
            median = np.median(vals)
            ax.scatter(1, mean, marker="D", color="white", s=40,
                       zorder=5, edgecolors="black", linewidths=1)
            ax.set_title(f"{p_atom} -> {cluster}", fontsize=10)
            ax.set_ylabel("A", fontsize=9)
            ax.text(1.3, mean,   f"u={mean:.1f}",   va="center", fontsize=8)
            ax.text(1.3, median, f"m={median:.1f}", va="center", fontsize=8, color="grey")
            ax.tick_params(axis="both", labelsize=8)
            ax.set_xticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _violinplot(df: pd.DataFrame, out_path: Path, channel_type: str,
                subunit: str, n_frames: int, label: str = "all frames"):
    """4 panels (one per cluster), P1/P4/P5 side by side."""
    fig, axes = plt.subplots(1, len(CLUSTERS), figsize=(18, 5), sharey=False)
    fig.suptitle(f"PIP2 headgroup distances -- {channel_type} | Subunit {subunit}\n"
                 f"[{label}, {n_frames} frames]", fontsize=13)

    for j, cluster in enumerate(CLUSTERS):
        ax = axes[j]
        vals_list, labels, colors = [], [], []
        for p_atom in P_ATOMS:
            col = f"{p_atom}_{cluster}"
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue
            vals_list.append(vals)
            labels.append(p_atom)
            colors.append(P_COLORS.get(p_atom, "#9E9E9E"))

        if not vals_list:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(vals_list, positions=range(1, len(vals_list)+1),
                              showmeans=True, showmedians=True)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("white")

        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("Distance (A)", fontsize=12)
        ax.set_title(f"-> {CLUSTER_LABELS.get(cluster, cluster)}", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_outputs(df: pd.DataFrame, out_dir: Path, channel_type: str,
                 label: str):
    """Save all outputs for a given dataframe of frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print(f"    [{label}] 0 frames -- skipping.")
        return

    df.to_csv(out_dir / "per_frame.csv", index=False)

    subunit_order = ["2a", "2b", "2c", "G1"]
    for sub in subunit_order:
        df_sub = df[df["subunit"] == sub]
        if df_sub.empty:
            continue
        sub_dir = out_dir / f"subunit_{sub}"
        sub_dir.mkdir(exist_ok=True)
        n = len(df_sub)

        df_sub.to_csv(sub_dir / "per_frame.csv", index=False)
        summary = _make_summary(df_sub, n)
        summary.to_csv(sub_dir / "summary.csv", index=False)
        _histograms(df_sub, sub_dir, channel_type, sub, label)
        _boxplot(df_sub, sub_dir / "boxplot.png", channel_type, sub, n, label)
        _violinplot(df_sub, sub_dir / "violin.png", channel_type, sub, n, label)
        print(f"    [{label}] Subunit {sub}: {n} frames -> {sub_dir}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_pip_headgroup_distances(
    u,
    mapping_json:  str,
    results_dir,
    start_frame:   int,
    end_frame:     int,
    channel_type:  str,
    end2_all:      dict = None,   # {ion_id: [frames]} from get_end2_all_frames()
    end3_all:      dict = None,   # {ion_id: [frames]} from get_end3_all_frames()
):
    results_dir = Path(results_dir)
    out_dir     = results_dir / "pip_headgroup_distances"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(mapping_json) as f:
        mapping = json.load(f)

    # -- build gate frame sets -------------------------------------------------
    def _flatten_with_prev(d):
        crossing, prev = set(), set()
        if d:
            for frames in d.values():
                for f in frames:
                    crossing.add(int(f))
                    if int(f) - 1 >= start_frame:
                        prev.add(int(f) - 1)
        return crossing, prev

    asn_gate_frames,   asn_gate_m1_frames  = _flatten_with_prev(end2_all)
    hbc_gate_frames,   hbc_gate_m1_frames  = _flatten_with_prev(end3_all)

    gate_subsets = [
        (asn_gate_frames,    "asn_gate_frames",    "ASN gate (end2)"),
        (asn_gate_m1_frames, "asn_gate_m1_frames", "ASN gate -1 (pre-end2)"),
        (hbc_gate_frames,    "hbc_gate_frames",    "HBC gate (end3)"),
        (hbc_gate_m1_frames, "hbc_gate_m1_frames", "HBC gate -1 (pre-end3)"),
    ]

    print("\n" + "=" * 60)
    print("PIP HEADGROUP DISTANCES  (v2.0)")
    print(f"  Channel type : {channel_type}")
    print(f"  Frames       : {start_frame} -> {end_frame}")
    print(f"  P atoms      : {P_ATOMS}")
    print(f"  Clusters     : Nterm(ARG60+LYS64) | Cterm(LYS199+LYS200) | LYS90 | LYS194")
    for _, folder, lbl in gate_subsets:
        n = len(asn_gate_frames if "asn" in folder and "m1" not in folder else
                asn_gate_m1_frames if "asn" in folder else
                hbc_gate_frames if "m1" not in folder else
                hbc_gate_m1_frames)
        if n > 0:
            print(f"  {lbl}: {n} frames")
    print("=" * 60)

    # -- pre-select all atoms ONCE --------------------------------------------
    pip_configs = []
    for pip_resid_str, subunit_data in mapping.items():
        pip_resid = int(pip_resid_str)
        subunit   = list(subunit_data.keys())[0]
        res_map   = subunit_data[subunit]

        nterm_ags, cterm_ags, lys90_ags, lys194_ags = [], [], [], []
        for res_name, info in res_map.items():
            raw = info["raw_resid"]
            pdb = info["pdb_number"]
            ag  = u.select_atoms(f"resid {raw} and not name H*")
            if len(ag) == 0:
                continue
            if pdb in [60, 64]:
                nterm_ags.append(ag)
            elif pdb in [199, 200]:
                cterm_ags.append(ag)
            elif pdb == 90:
                lys90_ags.append(ag)
            elif pdb == 194:
                lys194_ags.append(ag)

        if not nterm_ags or not cterm_ags or not lys90_ags:
            print(f"  WARNING: incomplete clusters for PIP {pip_resid} -- skipping")
            continue

        nterm_ag  = reduce(operator.add, nterm_ags)
        cterm_ag  = reduce(operator.add, cterm_ags)
        lys90_ag  = reduce(operator.add, lys90_ags)
        lys194_ag = reduce(operator.add, lys194_ags) if lys194_ags else None

        p_sels = {}
        for atom_name in P_ATOMS:
            sel = u.select_atoms(f"resid {pip_resid} and name {atom_name}")
            if len(sel) > 0:
                p_sels[atom_name] = sel[0]

        print(f"  PIP {pip_resid} -- subunit {subunit}: "
              f"Nterm={nterm_ag.n_atoms}, Cterm={cterm_ag.n_atoms}, "
              f"LYS90={lys90_ag.n_atoms}, "
              f"LYS194={'yes' if lys194_ag else 'missing'}, "
              f"P atoms={list(p_sels.keys())}")

        pip_configs.append({
            "pip_resid": pip_resid,
            "subunit":   subunit,
            "nterm":     nterm_ag,
            "cterm":     cterm_ag,
            "lys90":     lys90_ag,
            "lys194":    lys194_ag,
            "p_atoms":   p_sels,
        })

    if not pip_configs:
        print("No valid PIP configurations found.")
        return

    # -- single trajectory traversal ------------------------------------------
    n_frames = end_frame - start_frame + 1
    all_rows = []

    for ts in tqdm(u.trajectory[start_frame:end_frame + 1],
                   total=n_frames, desc="  Trajectory", leave=True):
        frame = ts.frame
        for cfg in pip_configs:
            nterm_pos  = cfg["nterm"].positions
            cterm_pos  = cfg["cterm"].positions
            lys90_pos  = cfg["lys90"].positions
            lys194_pos = cfg["lys194"].positions if cfg["lys194"] is not None else None

            row = {"frame": frame, "pip": cfg["pip_resid"], "subunit": cfg["subunit"]}

            for atom_name, atom in cfg["p_atoms"].items():
                p_pos = atom.position
                row[f"{atom_name}_Nterm"]  = round(_min_dist_p_to_cluster(p_pos, nterm_pos), 3)
                row[f"{atom_name}_Cterm"]  = round(_min_dist_p_to_cluster(p_pos, cterm_pos), 3)
                row[f"{atom_name}_LYS90"]  = round(_min_dist_p_to_cluster(p_pos, lys90_pos), 3)
                row[f"{atom_name}_LYS194"] = (round(_min_dist_p_to_cluster(p_pos, lys194_pos), 3)
                                               if lys194_pos is not None else np.nan)
            all_rows.append(row)

    if not all_rows:
        print("No data collected.")
        return

    df = pd.DataFrame(all_rows)

    # -- all frames -----------------------------------------------------------
    _run_outputs(df, out_dir, channel_type, "all frames")

    # -- gate subsets ---------------------------------------------------------
    for frame_set, folder_name, human_label in gate_subsets:
        if not frame_set:
            print(f"\n  [{human_label}] no frames provided -- skipping.")
            continue
        df_gate = df[df["frame"].isin(frame_set)]
        if df_gate.empty:
            print(f"\n  [{human_label}] no frames in trajectory range -- skipping.")
            continue
        print(f"\n  [{human_label}]: {len(df_gate)} rows")
        _run_outputs(df_gate, out_dir / folder_name, channel_type, human_label)

    print(f"\n✓ PIP headgroup distances complete -> {out_dir}")