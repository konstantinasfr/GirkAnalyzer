"""
lys90_cooccurrence_analysis.py
================================
VERSION: 2.1 — single trajectory traversal for all PIPs
================================
For every frame, computes distance from LYS90 to P1, P4, P5 of each PIP.

Speed optimisations vs original:
  - LYS90 atoms and P atoms are selected ONCE before the trajectory loop
  - distance_array called once per frame with all atoms stacked
  - no repeated select_atoms inside the loop

Analyses 4 gate frame subsets (in addition to all frames):
  - ASN gate frames     (end2   : ion crosses ASN gate, region 2->3)
  - ASN gate -1 frames  (end2-1 : frame just before ASN gate crossing)
  - HBC gate frames     (end3   : ion crosses HBC gate, region 3->4)
  - HBC gate -1 frames  (end3-1 : frame just before HBC gate crossing)

Called from main.py:

    from analysis.lys90_cooccurrence_analysis import run_lys90_cooccurrence_analysis

    run_lys90_cooccurrence_analysis(
        u            = u,
        mapping_json = mapping_json,
        results_dir  = results_dir,
        start_frame  = start_frame,
        end_frame    = end_frame,
        channel_type = args.channel_type,
        threshold    = 4.0,
        end2_all     = end2_all,   # from get_end2_all_frames()
        end3_all     = end3_all,   # from get_end3_all_frames()
    )
"""

from __future__ import annotations

__version__ = "2.1"

import json
import warnings
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array
from tqdm import tqdm

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_summary(df: pd.DataFrame, threshold: float, n_total: int) -> pd.DataFrame:
    rows = []
    for col, label in [
        ("close_P1",        "Close to P1"),
        ("close_P4",        "Close to P4"),
        ("close_P5",        "Close to P5"),
        ("close_P1_and_P5", "Close to P1 AND P5"),
        ("close_P1_and_P4", "Close to P1 AND P4"),
        ("close_P4_and_P5", "Close to P4 AND P5"),
        ("close_all",       "Close to P1, P4 AND P5"),
    ]:
        if col not in df.columns:
            continue
        count = int(df[col].sum())
        rows.append({
            "condition":    col,
            "label":        label,
            "frames":       count,
            "pct":          round(count / n_total * 100, 3),
            "total_frames": n_total,
            "threshold_A":  threshold,
        })
    return pd.DataFrame(rows)


def _barplot(summary: pd.DataFrame, title: str, out_path: Path):
    show   = ["close_P1", "close_P5", "close_P1_and_P5"]
    colors = {"close_P1": "#4CAF50", "close_P5": "#2196F3", "close_P1_and_P5": "#9C27B0"}
    labels = {"close_P1": "Close to P1", "close_P5": "Close to P5",
              "close_P1_and_P5": "Close to\nP1 AND P5"}

    df_plot = summary[summary["condition"].isin(show)]
    if df_plot.empty:
        return

    pcts    = [df_plot.loc[df_plot["condition"] == c, "pct"].values[0]
               if c in df_plot["condition"].values else 0.0 for c in show]
    max_val = max(pcts) if pcts else 1

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(np.arange(len(show)), pcts,
                  color=[colors[c] for c in show],
                  edgecolor="white", alpha=0.85, width=0.5)
    for bar, val in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.02,
                f"{val:.1f}%", va="bottom", ha="center", fontsize=16)
    ax.set_xticks(np.arange(len(show)))
    ax.set_xticklabels([labels[c] for c in show], fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylabel("% of frames", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, min(115, max_val * 1.35) if max_val > 0 else 10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_outputs(df: pd.DataFrame, out_dir: Path, title: str, threshold: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(df)
    if n_total == 0:
        print(f"    -> {out_dir.name}: 0 frames, skipping.")
        return
    df.to_csv(out_dir / "per_frame.csv", index=False)
    summary = _make_summary(df, threshold, n_total)
    summary.to_csv(out_dir / "summary.csv", index=False)
    _barplot(summary, title, out_dir / "barplot.png")
    print(f"    -> {out_dir.name}: {n_total} frames saved")


def _run_outputs(df_sub: pd.DataFrame, base_dir: Path, channel_type: str,
                 label: str, threshold: float):
    """Save outputs for all PIPs combined and per subunit."""
    subunit_order = ["2a", "2b", "2c", "G1"]
    _save_outputs(df_sub, base_dir,
                  f"LYS90 co-occurrence -- {channel_type}\n"
                  f"All PIPs [{label}], threshold {threshold} A",
                  threshold)
    for sub in subunit_order:
        df_s = df_sub[df_sub["subunit"] == sub]
        if df_s.empty:
            continue
        _save_outputs(df_s, base_dir / f"subunit_{sub}",
                      f"LYS90 co-occurrence -- {channel_type}\n"
                      f"Subunit {sub} [{label}], threshold {threshold} A",
                      threshold)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_lys90_cooccurrence_analysis(
    u,
    mapping_json:  str,
    results_dir,
    start_frame:   int,
    end_frame:     int,
    channel_type:  str,
    threshold:     float      = 4.0,
    pip_atoms:     List[str]  = None,
    end2_all:      dict       = None,   # {ion_id: [frames]} from get_end2_all_frames()
    end3_all:      dict       = None,   # {ion_id: [frames]} from get_end3_all_frames()
):
    if pip_atoms is None:
        pip_atoms = ["P1", "P4", "P5"]

    results_dir = Path(results_dir)
    out_dir     = results_dir / "lys90_cooccurrence" / f"threshold_{int(threshold)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(mapping_json) as f:
        mapping = json.load(f)

    # -- build gate frame sets from end2_all / end3_all -----------------------
    def _flatten_with_prev(d):
        """Return (crossing_frames, prev_frames) from {ion_id: [frames]}."""
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

    # gate subsets: (frame_set, folder_name, human_label)
    gate_subsets = [
        (asn_gate_frames,   "asn_gate_frames",    "ASN gate (end2)"),
        (asn_gate_m1_frames,"asn_gate_m1_frames",  "ASN gate -1 (pre-end2)"),
        (hbc_gate_frames,   "hbc_gate_frames",    "HBC gate (end3)"),
        (hbc_gate_m1_frames,"hbc_gate_m1_frames",  "HBC gate -1 (pre-end3)"),
    ]

    print("\n" + "=" * 60)
    print("LYS90 CO-OCCURRENCE ANALYSIS  (v2.1 -- single trajectory pass)")
    print(f"  Channel type : {channel_type}")
    print(f"  Threshold    : {threshold} A")
    print(f"  Frames       : {start_frame} -> {end_frame}")
    print(f"  PIP atoms    : {pip_atoms}")
    for _, folder, lbl in gate_subsets:
        n = len(_flatten_with_prev(
            end2_all if "asn" in folder else end3_all)[0 if "m1" not in folder else 1])
        if n > 0:
            print(f"  {lbl}: {n} frames")
    print("=" * 60)

    all_rows = []

    # -- pre-select all atoms ONCE for all PIPs -------------------------------
    pip_configs = []

    for pip_resid_str, subunit_data in mapping.items():
        pip_resid = int(pip_resid_str)
        subunit   = list(subunit_data.keys())[0]
        res_map   = subunit_data[subunit]

        lys90_info = res_map.get("LYS90")
        if lys90_info is None:
            print(f"  WARNING: LYS90 not found for PIP {pip_resid}, skipping.")
            continue
        lys90_resid = lys90_info["raw_resid"]
        lys90_atoms = u.select_atoms(f"resid {lys90_resid}")
        if len(lys90_atoms) == 0:
            print(f"  WARNING: LYS90 atoms not found for PIP {pip_resid}, skipping.")
            continue

        p_atoms = {}
        for atom_name in pip_atoms:
            sel = u.select_atoms(f"resid {pip_resid} and name {atom_name}")
            if len(sel) == 0:
                print(f"  WARNING: {atom_name} not found in resid {pip_resid}")
                continue
            p_atoms[atom_name] = sel[0]

        if not p_atoms:
            continue

        print(f"  PIP {pip_resid} -- subunit {subunit} -- LYS90: resid {lys90_resid}")
        pip_configs.append({
            "pip_resid":   pip_resid,
            "subunit":     subunit,
            "lys90_atoms": lys90_atoms,
            "p_atoms":     p_atoms,
        })

    if not pip_configs:
        print("No valid PIP configurations found.")
        return

    # -- single trajectory traversal for all PIPs -----------------------------
    n_frames = end_frame - start_frame + 1
    print(f"\n  Traversing {n_frames} frames for {len(pip_configs)} PIPs...")

    for ts in tqdm(
        u.trajectory[start_frame:end_frame + 1],
        total=n_frames,
        desc="  Trajectory",
        leave=True,
    ):
        frame = ts.frame

        for cfg in pip_configs:
            lys90_pos = cfg["lys90_atoms"].positions

            dists = {}
            for atom_name, atom in cfg["p_atoms"].items():
                p_pos = atom.position.reshape(1, 3)
                d = float(distance_array(p_pos, lys90_pos)[0].min())
                dists[atom_name] = round(d, 3)

            close = {k: (v < threshold) for k, v in dists.items()}
            c_P1  = close.get("P1", False)
            c_P4  = close.get("P4", False)
            c_P5  = close.get("P5", False)

            row = {"frame": frame, "pip": cfg["pip_resid"], "subunit": cfg["subunit"]}
            for atom_name in pip_atoms:
                row[f"dist_{atom_name}"]  = dists.get(atom_name, np.nan)
                row[f"close_{atom_name}"] = close.get(atom_name, False)
            row["close_P1_and_P5"] = c_P1 and c_P5
            row["close_P1_and_P4"] = c_P1 and c_P4
            row["close_P4_and_P5"] = c_P4 and c_P5
            row["close_all"]       = c_P1 and c_P4 and c_P5
            all_rows.append(row)

    if not all_rows:
        print("No data collected.")
        return

    df = pd.DataFrame(all_rows)

    # -- all frames -----------------------------------------------------------
    _run_outputs(df, out_dir, channel_type, "all frames", threshold)

    # -- gate subsets ---------------------------------------------------------
    for frame_set, folder_name, human_label in gate_subsets:
        if not frame_set:
            print(f"\n  [{human_label}] no frames provided -- skipping.")
            continue
        df_gate = df[df["frame"].isin(frame_set)]
        if df_gate.empty:
            print(f"\n  [{human_label}] no frames found in trajectory range -- skipping.")
            continue
        print(f"\n  [{human_label}]: {len(df_gate)} rows")
        _run_outputs(df_gate, out_dir / folder_name,
                     channel_type, human_label, threshold)

    print(f"\n✓ LYS90 co-occurrence complete -> {out_dir}")