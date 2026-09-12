"""
pip_pose_analysis.py
=====================
For every frame of the trajectory, assigns P4 and P5 atoms of each PIP
to one of THREE neighborhoods based on MINIMUM INDIVIDUAL RESIDUE DISTANCE:
  - Nterm: ARG60 + LYS64  (G1: 52 + 50)
  - Cterm: LYS194 + LYS199 + LYS200  (G1: 183 + 188 + 189)
  - Middle (LYS90): only tracked separately  (G1: 79)

Assignment algorithm (per frame):
  1. If no residue in any neighborhood within threshold -> assign "none"
  2. Otherwise: compute min distance to each neighborhood
     (min over all residues in that neighborhood)
  3. Find overall minimum of the 6 distances (P4/P5 x 3 neighborhoods)
  4. Assign that P atom to that neighborhood first
  5. Assign remaining P atom to its closest neighborhood (by min residue dist)

Called from main.py:

    from analysis.pip_pose_analysis import run_pip_pose_analysis

    run_pip_pose_analysis(
        u            = u,
        mapping_json = "/path/to/pip_residue_mapping.json",
        results_dir  = results_dir,
        start_frame  = start_frame,
        end_frame    = end_frame,
        channel_type = args.channel_type,
        threshold    = 4.0,
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
from MDAnalysis.lib.distances import distance_array
from tqdm import tqdm

warnings.filterwarnings("ignore")

# all valid state combinations
def _all_states():
    poses = ["nterm", "cterm", "none"]
    states = {}
    for p4 in poses:
        for p5 in poses:
            key = f"p4_{p4}_p5_{p5}"
            states[key] = f"P4 {p4}\nP5 {p5}"
    return states

STATE_LABELS = _all_states()

STATE_COLORS = {
    "p4_nterm_p5_nterm": "#1565C0",  # dark blue
    "p4_nterm_p5_middle":  "#42A5F5",  # light blue
    "p4_nterm_p5_cterm":    "#4CAF50",  # green
    "p4_nterm_p5_none":    "#A5D6A7",  # light green
    "p4_middle_p5_nterm":  "#FFA726",  # orange
    "p4_middle_p5_middle":   "#FF6F00",  # dark orange
    "p4_middle_p5_cterm":     "#EF5350",  # red
    "p4_middle_p5_none":     "#FFCC80",  # light orange
    "p4_cterm_p5_nterm":    "#AB47BC",  # purple
    "p4_cterm_p5_middle":     "#CE93D8",  # light purple
    "p4_cterm_p5_cterm":       "#B71C1C",  # dark red
    "p4_cterm_p5_none":       "#EF9A9A",  # light red
    "p4_none_p5_nterm":    "#90CAF9",  # very light blue
    "p4_none_p5_middle":     "#FFECB3",  # yellow
    "p4_none_p5_cterm":       "#FFCDD2",  # very light red
    "p4_none_p5_none":       "#9E9E9E",  # grey
}


# -----------------------------------------------------------------------------
# Neighborhood PDB number mappings per subunit
# -----------------------------------------------------------------------------

def _get_neighborhood_pdb_nums(subunit: str):
    """
    Return (nterm_pdb, middle_pdb, cterm_pdb) PDB number lists for a given subunit.
    G1 uses different numbering than GIRK2 subunits (2a/2b/2c/2d).
    """
    if subunit == "G1":
        # ARG60->52, LYS64->52 (same residue in G1), LYS90->79,
        # LYS194->183, LYS199->188, LYS200->189
        nterm_pdb  = [52, 50]
        middle_pdb = [79]
        cterm_pdb  = [183, 188, 189]
    else:
        nterm_pdb  = [60, 64]
        middle_pdb = [90]
        cterm_pdb  = [194, 199, 200]
    return nterm_pdb, middle_pdb, cterm_pdb


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _min_dist_to_neighborhood(u, p_pos: np.ndarray, raw_resids: list) -> float:
    """Min distance from P atom to any atom of any residue in neighborhood."""
    p_pos_2d = p_pos.reshape(1, 3)
    min_d = np.inf
    for resid in raw_resids:
        atoms = u.select_atoms(f"resid {resid}")
        if len(atoms) == 0:
            continue
        d = float(distance_array(p_pos_2d, atoms.positions)[0].min())
        if d < min_d:
            min_d = d
    return min_d


def _assign_pose_single(p_pos, u, nterm_resids, middle_resids, cterm_resids,
                         threshold):
    """
    Assign a single P atom to nterm/middle/cterm/none
    based on minimum individual residue distance.
    Returns (pose, d_nterm, d_middle, d_cterm)
    """
    d_nterm  = _min_dist_to_neighborhood(u, p_pos, nterm_resids)
    d_middle = _min_dist_to_neighborhood(u, p_pos, middle_resids)
    d_cterm  = _min_dist_to_neighborhood(u, p_pos, cterm_resids)

    d_min = min(d_nterm, d_middle, d_cterm)

    if d_min >= threshold:
        pose = "none"
    elif d_min == d_nterm:
        pose = "nterm"
    elif d_min == d_middle:
        pose = "middle"
    else:
        pose = "cterm"

    return pose, d_nterm, d_middle, d_cterm


def _assign_poses_both(p4_pos, p5_pos, u,
                        nterm_resids, middle_resids, cterm_resids,
                        threshold):
    """
    STEP 1: Find overall minimum of P4_nterm, P4_cterm, P5_nterm, P5_cterm.
      Assign that P first (nterm or cterm), check remaining:
        - OPPOSITE within threshold -> nterm+cterm or cterm+nterm -> DONE
        - SAME within threshold -> both_same -> Step 1b
        - NEITHER -> none -> DONE

    STEP 1b: both_same -- re-check if either P can be the opposite
        - Both qualify -> reassign the one with smaller opposite distance -> DONE
        - Only one qualifies -> reassign that one -> DONE
        - Neither -> stay both_same -> DONE

    No Step 2 -- middle/LYS90 is only tracked as extra columns, not used for assignment.
    """
    p4_d_nt = _min_dist_to_neighborhood(u, p4_pos, nterm_resids)
    p4_d_ct = _min_dist_to_neighborhood(u, p4_pos, cterm_resids)
    p5_d_nt = _min_dist_to_neighborhood(u, p5_pos, nterm_resids)
    p5_d_ct = _min_dist_to_neighborhood(u, p5_pos, cterm_resids)
    p4_d_mi = _min_dist_to_neighborhood(u, p4_pos, middle_resids)
    p5_d_mi = _min_dist_to_neighborhood(u, p5_pos, middle_resids)

    # -- Step 1 ---------------------------------------------------------------
    all_dists = {
        ("P4", "nterm"): p4_d_nt,
        ("P4", "cterm"): p4_d_ct,
        ("P5", "nterm"): p5_d_nt,
        ("P5", "cterm"): p5_d_ct,
    }
    first_atom, first_pose = min(all_dists, key=all_dists.__getitem__)
    opposite = "cterm" if first_pose == "nterm" else "nterm"
    step1b = False

    if first_atom == "P4":
        p4_pose = first_pose
        opp_dist  = p5_d_ct if opposite == "cterm" else p5_d_nt
        same_dist = p5_d_nt if first_pose == "nterm" else p5_d_ct
        if opp_dist < threshold:
            p5_pose = opposite         # DONE
        elif same_dist < threshold:
            p5_pose = first_pose
            step1b = True              # both_same -> Step 1b
        else:
            p5_pose = "none"           # DONE
    else:
        p5_pose = first_pose
        opp_dist  = p4_d_ct if opposite == "cterm" else p4_d_nt
        same_dist = p4_d_nt if first_pose == "nterm" else p4_d_ct
        if opp_dist < threshold:
            p4_pose = opposite         # DONE
        elif same_dist < threshold:
            p4_pose = first_pose
            step1b = True
        else:
            p4_pose = "none"           # DONE

    # -- Step 1b: both_same -- re-check if either can be opposite -------------
    if step1b:
        opp = opposite
        p4_opp = p4_d_ct if opp == "cterm" else p4_d_nt
        p5_opp = p5_d_ct if opp == "cterm" else p5_d_nt
        p4_can = p4_opp < threshold
        p5_can = p5_opp < threshold

        if p4_can and p5_can:
            if p4_opp <= p5_opp:
                p4_pose = opp
            else:
                p5_pose = opp
        elif p4_can:
            p4_pose = opp
        elif p5_can:
            p5_pose = opp
        # else: stay both_same

    return (p4_pose, p5_pose,
            p4_d_nt, p4_d_mi, p4_d_ct,
            p5_d_nt, p5_d_mi, p5_d_ct)



# -----------------------------------------------------------------------------
# Plotting helper
# -----------------------------------------------------------------------------

def _save_barplot(df_summary, states, title, out_path, figsize=(12, 6), fontsize=12):
    pcts = [df_summary.loc[df_summary["state"] == s, "pct"].values[0]
            if s in df_summary["state"].values else 0.0 for s in states]
    fig, ax = plt.subplots(figsize=figsize)
    x    = np.arange(len(states))
    bars = ax.bar(x, pcts,
                  color=[STATE_COLORS.get(s, "#9E9E9E") for s in states],
                  edgecolor="white", alpha=0.85)
    max_val = max(pcts) if pcts else 1
    for bar, val in zip(bars, pcts):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.01,
                    f"{val:.1f}%", va="bottom", ha="center", fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS.get(s, s) for s in states],
                       rotation=45, ha="right", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_ylabel("% of frames", fontsize=fontsize + 2)
    ax.set_title(title, fontsize=fontsize + 3)
    ax.set_ylim(top=max_val * 1.18)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_summary(df_sub):
    states_present = [s for s in STATE_LABELS.keys() if s in df_sub["state"].unique()]
    n_total = len(df_sub)
    rows = []
    for state in states_present:
        count = int((df_sub["state"] == state).sum())
        rows.append({
            "state":   state,
            "label":   STATE_LABELS[state].replace("\n", " + "),
            "count":   count,
            "pct":     round(count / n_total * 100, 3),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False), states_present, n_total


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def _run_outputs(df: pd.DataFrame, out_dir: "Path", channel_type: str, label: str = "all frames"):
    """Run all outputs (CSVs, plots, summaries) for a given dataframe of frames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = len(df)
    print(f"\n  [{label}] {n_frames} frames -> {out_dir}")
    if n_frames == 0:
        print(f"  [{label}] No frames -- skipping.")
        return

    # per-frame CSVs
    df.to_csv(out_dir / "per_frame.csv", index=False)
    for sub in df["subunit"].unique():
        df[df["subunit"] == sub].to_csv(out_dir / f"per_frame_{sub}.csv", index=False)
    print(f"\n  Per-frame CSVs saved -> {out_dir}")

    # overall summary + plot
    df_summary, states, n_total = _make_summary(df)
    df_summary.to_csv(out_dir / "summary.csv", index=False)
    _save_barplot(df_summary, states,
                  f"PIP2 Pose Assignment -- {channel_type}\n{n_total} frames (all PIPs)",
                  out_dir / "barplot_states.png")
    print(f"  Overall summary + plot saved")

    # per-subunit
    if channel_type == "G2":
        subunit_order = ["2a", "2b", "2c", "2d"]
    else:
        subunit_order = ["2a", "2b", "2c", "G1"]
    subunits_present = [s for s in subunit_order if s in df["subunit"].unique()]

    for sub in subunits_present:
        df_sub = df[df["subunit"] == sub]
        sub_dir = out_dir / f"subunit_{sub}"
        sub_dir.mkdir(exist_ok=True)
        df_sum_sub, states_sub, n_sub = _make_summary(df_sub)
        df_sum_sub.to_csv(sub_dir / "summary.csv", index=False)
        _save_barplot(df_sum_sub, states_sub,
                      f"PIP2 Pose -- {channel_type} -- Subunit {sub}\n{n_sub} frames",
                      sub_dir / "barplot_states.png")
        print(f"  Subunit {sub} saved")

    # combined per-PIP plot ordered by subunit
    pip_subunit = df.drop_duplicates("pip")[["pip","subunit"]].set_index("pip")["subunit"]
    pip_ids_sorted = sorted(
        df["pip"].unique(),
        key=lambda p: subunit_order.index(pip_subunit[p])
                      if pip_subunit[p] in subunit_order else 99
    )

    n_pips = len(pip_ids_sorted)
    fig, axes = plt.subplots(1, n_pips, figsize=(6 * n_pips, 7), sharey=False)
    if n_pips == 1: axes = [axes]

    for ax, pip_id in zip(axes, pip_ids_sorted):
        df_pip = df[df["pip"] == pip_id]
        n_pip  = len(df_pip)
        subunit = df_pip["subunit"].iloc[0]
        df_sum_pip, states_pip, _ = _make_summary(df_pip)

        pcts_pip = [df_sum_pip.loc[df_sum_pip["state"]==s,"pct"].values[0]
                    if s in df_sum_pip["state"].values else 0.0
                    for s in states_pip]
        bars = ax.bar(np.arange(len(states_pip)), pcts_pip,
                      color=[STATE_COLORS.get(s,"#9E9E9E") for s in states_pip],
                      edgecolor="white", alpha=0.85)
        max_val = max(pcts_pip) if pcts_pip else 1
        for bar, val in zip(bars, pcts_pip):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max_val*0.01,
                        f"{val:.1f}%", va="bottom", ha="center", fontsize=9)
        ax.set_xticks(np.arange(len(states_pip)))
        ax.set_xticklabels([STATE_LABELS.get(s,s) for s in states_pip],
                           rotation=45, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(f"PIP {pip_id}\nSubunit {subunit}", fontsize=12)
        ax.set_ylabel("% of frames", fontsize=11)
        ax.set_ylim(top=max_val * 1.2 if max_val > 0 else 10)

        # individual pip folder
        pip_dir = out_dir / f"PIP_{pip_id}"
        pip_dir.mkdir(exist_ok=True)
        df_sum_pip.to_csv(pip_dir / "summary.csv", index=False)
        _save_barplot(df_sum_pip, states_pip,
                      f"PIP2 Pose -- {channel_type} -- PIP {pip_id} (Subunit {subunit})\n{n_pip} frames",
                      pip_dir / "barplot_states.png")

    plt.suptitle(f"PIP2 Pose Assignment per PIP -- {channel_type}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "barplot_per_pip.png", dpi=150)
    plt.close(fig)
    print(f"  Per-PIP combined plot saved")

    # -- per P-atom marginal summary (P4 and P5 independently) ---------------
    print(f"\n  Per P-atom marginal summaries...")

    poses_order = ["nterm", "cterm", "none"]
    p_colors    = {"nterm": "#2196F3", "cterm": "#F44336", "none": "#9E9E9E"}

    marginal_rows = []
    for sub in subunits_present:
        df_sub = df[df["subunit"] == sub]
        n_sub  = len(df_sub)
        for p_atom, pose_col in [("P4", "p4_pose"), ("P5", "p5_pose")]:
            for pose in poses_order:
                count = int((df_sub[pose_col] == pose).sum())
                marginal_rows.append({
                    "subunit": sub,
                    "p_atom":  p_atom,
                    "pose":    pose,
                    "count":   count,
                    "pct":     round(count / n_sub * 100, 3),
                })

    df_marginal = pd.DataFrame(marginal_rows)
    df_marginal.to_csv(out_dir / "marginal_per_p_atom.csv", index=False)
    print(f"  Marginal CSV -> {out_dir / 'marginal_per_p_atom.csv'}")

    # combined plot: all subunits, P4 vs P5 side by side
    n_subs = len(subunits_present)
    fig, axes = plt.subplots(1, n_subs, figsize=(5 * n_subs, 5), sharey=False)
    if n_subs == 1: axes = [axes]

    for ax, sub in zip(axes, subunits_present):
        df_s  = df_marginal[df_marginal["subunit"] == sub]
        x     = np.arange(len(poses_order))
        width = 0.35

        p4_pcts = [df_s.loc[(df_s["p_atom"]=="P4") & (df_s["pose"]==p), "pct"].values[0]
                   if len(df_s.loc[(df_s["p_atom"]=="P4") & (df_s["pose"]==p)]) > 0 else 0.0
                   for p in poses_order]
        p5_pcts = [df_s.loc[(df_s["p_atom"]=="P5") & (df_s["pose"]==p), "pct"].values[0]
                   if len(df_s.loc[(df_s["p_atom"]=="P5") & (df_s["pose"]==p)]) > 0 else 0.0
                   for p in poses_order]

        bars1 = ax.bar(x - width/2, p4_pcts, width, label="P4",
                       color=[p_colors[p] for p in poses_order], alpha=0.85,
                       edgecolor="white")
        bars2 = ax.bar(x + width/2, p5_pcts, width, label="P5",
                       color=[p_colors[p] for p in poses_order], alpha=0.5,
                       edgecolor="white", hatch="//")

        max_val = max(p4_pcts + p5_pcts) if p4_pcts + p5_pcts else 1
        for bar, val in zip(bars1, p4_pcts):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max_val*0.01,
                        f"{val:.1f}%", va="bottom", ha="center", fontsize=9,
                        fontweight="bold")
        for bar, val in zip(bars2, p5_pcts):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max_val*0.01,
                        f"{val:.1f}%", va="bottom", ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(poses_order, fontsize=11)
        ax.set_ylabel("% of frames", fontsize=12)
        ax.set_title(f"Subunit {sub}", fontsize=12)
        ax.legend(fontsize=11)
        ax.set_ylim(0, min(110, max_val * 1.25) if max_val > 0 else 10)
        ax.tick_params(axis="y", labelsize=11)

    plt.suptitle(f"P4 and P5 pose distribution (independent) -- {channel_type}", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "marginal_per_p_atom_combined.png", dpi=150)
    plt.close(fig)

    # individual plot per subunit
    for sub in subunits_present:
        df_s  = df_marginal[df_marginal["subunit"] == sub]
        x     = np.arange(len(poses_order))
        width = 0.35

        p4_pcts = [df_s.loc[(df_s["p_atom"]=="P4") & (df_s["pose"]==p), "pct"].values[0]
                   if len(df_s.loc[(df_s["p_atom"]=="P4") & (df_s["pose"]==p)]) > 0 else 0.0
                   for p in poses_order]
        p5_pcts = [df_s.loc[(df_s["p_atom"]=="P5") & (df_s["pose"]==p), "pct"].values[0]
                   if len(df_s.loc[(df_s["p_atom"]=="P5") & (df_s["pose"]==p)]) > 0 else 0.0
                   for p in poses_order]

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        bars1 = ax2.bar(x - width/2, p4_pcts, width, label="P4",
                        color=[p_colors[p] for p in poses_order], alpha=0.85,
                        edgecolor="white")
        bars2 = ax2.bar(x + width/2, p5_pcts, width, label="P5",
                        color=[p_colors[p] for p in poses_order], alpha=0.5,
                        edgecolor="white", hatch="//")

        max_val = max(p4_pcts + p5_pcts) if p4_pcts + p5_pcts else 1
        for bar, val in zip(bars1, p4_pcts):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + max_val*0.01,
                         f"{val:.1f}%", va="bottom", ha="center", fontsize=11,
                         fontweight="bold")
        for bar, val in zip(bars2, p5_pcts):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + max_val*0.01,
                         f"{val:.1f}%", va="bottom", ha="center", fontsize=11)

        ax2.set_xticks(x)
        ax2.set_xticklabels(poses_order, fontsize=12)
        ax2.set_ylabel("% of frames", fontsize=13)
        ax2.set_title(f"P4 and P5 pose distribution\n{channel_type} -- Subunit {sub}",
                      fontsize=13)
        ax2.legend(fontsize=12)
        ax2.set_ylim(0, min(110, max_val * 1.25) if max_val > 0 else 10)
        ax2.tick_params(axis="y", labelsize=12)
        plt.tight_layout()
        sub_dir = out_dir / f"subunit_{sub}"
        sub_dir.mkdir(exist_ok=True)
        fig2.savefig(sub_dir / "marginal_per_p_atom.png", dpi=150)
        plt.close(fig2)

    print(f"  Marginal plots -> combined + per subunit")


    # -- LYS90 co-occurrence analysis -----------------------------------------
    print(f"\n  LYS90 co-occurrence analysis...")
    lys90_rows = []
    for sub in subunits_present:
        df_sub = df[df["subunit"] == sub]
        n_sub  = len(df_sub)
        for p_atom, pose_col, lys90_col in [
            ("P4", "p4_pose", "p4_lys90_close"),
            ("P5", "p5_pose", "p5_lys90_close"),
        ]:
            if lys90_col not in df_sub.columns:
                continue
            for pose in ["nterm", "cterm", "none", "middle"]:
                df_pose = df_sub[df_sub[pose_col] == pose]
                n_pose  = len(df_pose)
                if n_pose == 0:
                    continue
                n_lys90 = df_pose[lys90_col].sum()
                lys90_rows.append({
                    "subunit":         sub,
                    "p_atom":          p_atom,
                    "pose":            pose,
                    "n_pose_frames":   int(n_pose),
                    "pct_of_total":    round(n_pose / n_sub * 100, 2),
                    "n_lys90_close":   int(n_lys90),
                    "pct_lys90_close": round(n_lys90 / n_pose * 100, 2),
                })

    if lys90_rows:
        df_lys90 = pd.DataFrame(lys90_rows)
        df_lys90.to_csv(out_dir / "lys90_cooccurrence.csv", index=False)
        print(f"  LYS90 co-occurrence -> {out_dir / 'lys90_cooccurrence.csv'}")

        subunits_plot = [s for s in ["2a","2b","2c","G1"] if s in df_lys90["subunit"].unique()]
        for p_atom in ["P4", "P5"]:
            df_p = df_lys90[df_lys90["p_atom"] == p_atom]
            poses = ["nterm", "cterm", "none"]
            colors = {"nterm": "#2196F3", "cterm": "#F44336", "none": "#9E9E9E"}

            n_subs = len(subunits_plot)
            fig, axes = plt.subplots(1, n_subs, figsize=(5 * n_subs, 5), sharey=False)
            if n_subs == 1: axes = [axes]

            for ax, sub in zip(axes, subunits_plot):
                df_ps = df_p[df_p["subunit"] == sub]
                poses_present = [p for p in poses if p in df_ps["pose"].values]
                pcts = [df_ps.loc[df_ps["pose"]==p, "pct_lys90_close"].values[0]
                        if p in df_ps["pose"].values else 0.0
                        for p in poses_present]
                max_val = max(pcts) if pcts else 1
                bars = ax.bar(range(len(poses_present)), pcts,
                              color=[colors.get(p, "#9E9E9E") for p in poses_present],
                              edgecolor="white", alpha=0.85)
                for bar, val in zip(bars, pcts):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + max_val * 0.02,
                            f"{val:.1f}%", va="bottom", ha="center", fontsize=11)
                ax.set_xticks(range(len(poses_present)))
                ax.set_xticklabels([f"{p_atom} {p}" for p in poses_present],
                                   rotation=30, ha="right", fontsize=11)
                ax.tick_params(axis="y", labelsize=11)
                ax.set_title(f"Subunit {sub}", fontsize=12)
                ax.set_ylabel("% frames LYS90 close", fontsize=12)
                ax.set_ylim(0, min(110, max_val * 1.25) if max_val > 0 else 10)

            plt.suptitle(f"LYS90 co-occurrence with {p_atom} pose -- {channel_type}",
                         fontsize=14)
            plt.tight_layout()
            fig.savefig(out_dir / f"lys90_cooccurrence_{p_atom}.png", dpi=150)
            plt.close(fig)

            for sub in subunits_plot:
                df_ps = df_p[df_p["subunit"] == sub]
                poses_present = [p for p in poses if p in df_ps["pose"].values]
                pcts = [df_ps.loc[df_ps["pose"]==p, "pct_lys90_close"].values[0]
                        if p in df_ps["pose"].values else 0.0
                        for p in poses_present]
                max_val = max(pcts) if pcts else 1
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                bars = ax2.bar(range(len(poses_present)), pcts,
                               color=[colors.get(p, "#9E9E9E") for p in poses_present],
                               edgecolor="white", alpha=0.85)
                for bar, val in zip(bars, pcts):
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + max_val * 0.02,
                             f"{val:.1f}%", va="bottom", ha="center", fontsize=12)
                ax2.set_xticks(range(len(poses_present)))
                ax2.set_xticklabels([f"{p_atom} {p}" for p in poses_present],
                                    rotation=30, ha="right", fontsize=12)
                ax2.tick_params(axis="y", labelsize=12)
                ax2.set_ylabel("% frames LYS90 close", fontsize=13)
                ax2.set_title(f"LYS90 co-occurrence with {p_atom} -- {channel_type}\nSubunit {sub}",
                              fontsize=13)
                ax2.set_ylim(0, min(110, max_val * 1.25) if max_val > 0 else 10)
                plt.tight_layout()
                sub_dir = out_dir / f"subunit_{sub}"
                sub_dir.mkdir(exist_ok=True)
                fig2.savefig(sub_dir / f"lys90_cooccurrence_{p_atom}.png", dpi=150)
                plt.close(fig2)

            print(f"  LYS90 co-occurrence plot ({p_atom}) -> combined + per subunit")

    all_states_ordered = [
        "p4_nterm_p5_nterm", "p4_nterm_p5_cterm",  "p4_nterm_p5_none",
        "p4_cterm_p5_nterm", "p4_cterm_p5_cterm",  "p4_cterm_p5_none",
        "p4_none_p5_nterm",  "p4_none_p5_cterm",   "p4_none_p5_none",
    ]

    # -- unique residue count per state ---------------------------------------
    print(f"\n  Unique residue counts per state...")

    count_rows = []
    for sub in subunits_present:
        df_sub = df[df["subunit"] == sub]
        for state in all_states_ordered:
            df_st = df_sub[df_sub["state"] == state]
            n = len(df_st)
            if n == 0:
                continue
            for col, label in [("n_p4_close_res","P4"), ("n_p5_close_res","P5"),
                                ("n_unique_close_res","P4+P5 unique")]:
                if col not in df_st.columns:
                    continue
                vals = df_st[col]
                count_rows.append({
                    "subunit":    sub,
                    "state":      state,
                    "which":      label,
                    "n_frames":   n,
                    "mean_count": round(vals.mean(), 3),
                    "std_count":  round(vals.std(), 3),
                    "min_count":  int(vals.min()),
                    "max_count":  int(vals.max()),
                })

    df_counts = pd.DataFrame(count_rows)
    df_counts.to_csv(out_dir / "residue_counts_per_state.csv", index=False)
    print(f"  Residue counts CSV -> {out_dir / 'residue_counts_per_state.csv'}")

    # -- global unique residue count distribution -----------------------------
    print(f"\n  Global unique residue count distribution...")
    if "n_unique_close_res" in df.columns:
        global_dist_rows = []
        for sub in subunits_present:
            df_sub = df[df["subunit"] == sub]
            n_total = len(df_sub)
            for n_res, grp in df_sub.groupby("n_unique_close_res"):
                global_dist_rows.append({
                    "subunit":    sub,
                    "n_residues": int(n_res),
                    "frames":     len(grp),
                    "pct":        round(len(grp) / n_total * 100, 3),
                })

        df_global_dist = pd.DataFrame(global_dist_rows)
        df_global_dist.to_csv(out_dir / "global_count_distribution.csv", index=False)
        print(f"  Global count dist CSV -> {out_dir / 'global_count_distribution.csv'}")

        n_subs = len(subunits_present)
        fig, axes = plt.subplots(1, n_subs, figsize=(4*n_subs, 5), sharey=False)
        if n_subs == 1: axes = [axes]
        for ax, sub in zip(axes, subunits_present):
            df_s = df_global_dist[df_global_dist["subunit"]==sub].sort_values("n_residues")
            xlabels = [f"{int(r)} res" for r in df_s["n_residues"]]
            pcts    = df_s["pct"].tolist()
            max_val = max(pcts) if pcts else 1
            bars = ax.bar(range(len(xlabels)), pcts,
                          color="#5C6BC0", alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, pcts):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+max_val*0.02,
                        f"{val:.1f}%", va="bottom", ha="center", fontsize=11)
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, fontsize=11)
            ax.tick_params(axis="y", labelsize=11)
            ax.set_ylabel("% of frames", fontsize=12)
            ax.set_title(f"Subunit {sub}", fontsize=12)
            ax.set_ylim(0, min(110, max_val*1.25))
        plt.suptitle(f"Unique residues close to P4+P5 (all states)\n{channel_type}",
                     fontsize=13)
        plt.tight_layout()
        fig.savefig(out_dir / "global_count_distribution.png", dpi=150)
        plt.close(fig)
        print(f"  Global count dist plot -> {out_dir / 'global_count_distribution.png'}")

    for sub in subunits_present:
        df_s = df_counts[(df_counts["subunit"]==sub) & (df_counts["which"]=="P4+P5 unique")]
        if df_s.empty:
            continue
        states_s = df_s["state"].tolist()
        means     = df_s["mean_count"].tolist()
        stds      = df_s["std_count"].tolist()
        xlabels   = [s.replace("p4_","").replace("_p5_","+") for s in states_s]

        fig, ax = plt.subplots(figsize=(max(8, len(states_s)*1.2), 5))
        x = np.arange(len(states_s))
        bars = ax.bar(x, means, yerr=stds, color="#7E57C2", alpha=0.85,
                      edgecolor="white", capsize=4)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + max(means)*0.02,
                    f"{val:.1f}", va="bottom", ha="center", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylabel("Mean unique residues close", fontsize=13)
        ax.set_title(f"Unique residues close to P4+P5 per state\n{channel_type} -- Subunit {sub}",
                     fontsize=13)
        ax.set_ylim(0, max(means)*1.3 if means else 3)
        plt.tight_layout()
        fig.savefig(out_dir / f"residue_counts_per_state_{sub}.png", dpi=150)
        plt.close(fig)
        print(f"  Residue counts plot -> {out_dir / f'residue_counts_per_state_{sub}.png'}")

    # -- distribution of unique residue counts per state ----------------------
    print(f"\n  Unique residue count distributions per state...")

    dist_rows = []
    for sub in subunits_present:
        df_sub = df[df["subunit"] == sub]
        for state in all_states_ordered:
            df_st = df_sub[df_sub["state"] == state]
            n_total = len(df_st)
            if n_total == 0:
                continue
            if "n_unique_close_res" not in df_st.columns:
                continue
            for n_res, grp in df_st.groupby("n_unique_close_res"):
                dist_rows.append({
                    "subunit":     sub,
                    "state":       state,
                    "n_residues":  int(n_res),
                    "frames":      len(grp),
                    "pct":         round(len(grp) / n_total * 100, 3),
                })

    df_dist = pd.DataFrame(dist_rows)
    df_dist.to_csv(out_dir / "unique_residue_count_distribution.csv", index=False)
    print(f"  Distribution CSV -> {out_dir / 'unique_residue_count_distribution.csv'}")

    dist_plot_dir = out_dir / "count_distribution_per_state"
    dist_plot_dir.mkdir(exist_ok=True)

    for state in all_states_ordered:
        df_st = df_dist[df_dist["state"] == state]
        if df_st.empty:
            continue
        state_label = state.replace("p4_","").replace("_p5_","+")

        subs_in_state = [s for s in subunits_present if s in df_st["subunit"].values]
        n_subs = len(subs_in_state)
        if n_subs == 0:
            continue

        fig, axes = plt.subplots(1, n_subs, figsize=(4 * n_subs, 5), sharey=False)
        if n_subs == 1: axes = [axes]

        for ax, sub in zip(axes, subs_in_state):
            df_sub_st = df_st[df_st["subunit"] == sub].sort_values("n_residues")
            x_labels = [f"{int(r)} res" for r in df_sub_st["n_residues"]]
            pcts     = df_sub_st["pct"].tolist()
            max_val  = max(pcts) if pcts else 1

            bars = ax.bar(range(len(x_labels)), pcts,
                          color="#7E57C2", alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, pcts):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max_val * 0.02,
                        f"{val:.1f}%", va="bottom", ha="center", fontsize=11)
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, fontsize=11)
            ax.tick_params(axis="y", labelsize=11)
            ax.set_ylabel("% of frames", fontsize=12)
            ax.set_title(f"Subunit {sub}", fontsize=12)
            ax.set_ylim(0, min(110, max_val * 1.25))

        plt.suptitle(f"Unique residues close -- {state_label}\n{channel_type}", fontsize=13)
        plt.tight_layout()
        fig.savefig(dist_plot_dir / f"count_dist_{state}.png", dpi=150)
        plt.close(fig)

    print(f"  Count distribution plots -> {dist_plot_dir}")

    # -- LYS90 co-occurrence per state ----------------------------------------
    print(f"\n  LYS90 co-occurrence per state...")

    lys90_rows = []
    for sub in subunits_present:
        df_sub = df[df["subunit"] == sub]
        for state in all_states_ordered:
            df_st = df_sub[df_sub["state"] == state]
            n = len(df_st)
            if n == 0:
                continue
            n_p4 = int(df_st["p4_lys90_close"].sum()) if "p4_lys90_close" in df_st else 0
            n_p5 = int(df_st["p5_lys90_close"].sum()) if "p5_lys90_close" in df_st else 0
            lys90_rows.append({
                "subunit":          sub,
                "state":            state,
                "n_frames":         n,
                "n_p4_lys90_close": n_p4,
                "n_p5_lys90_close": n_p5,
                "pct_p4_lys90":     round(n_p4 / n * 100, 2),
                "pct_p5_lys90":     round(n_p5 / n * 100, 2),
            })

    df_lys90 = pd.DataFrame(lys90_rows)
    df_lys90.to_csv(out_dir / "lys90_per_state.csv", index=False)
    print(f"  LYS90 per-state CSV -> {out_dir / 'lys90_per_state.csv'}")

    for sub in subunits_present:
        df_s = df_lys90[df_lys90["subunit"] == sub]
        if df_s.empty:
            continue
        states_present = df_s["state"].tolist()
        pct_p4 = df_s["pct_p4_lys90"].tolist()
        pct_p5 = df_s["pct_p5_lys90"].tolist()
        x      = np.arange(len(states_present))
        width  = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(states_present) * 1.2), 6))
        bars1 = ax.bar(x - width/2, pct_p4, width, label="P4", color="#FF9800", alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width/2, pct_p5, width, label="P5", color="#2196F3", alpha=0.85, edgecolor="white")

        max_val = max(pct_p4 + pct_p5) if pct_p4 + pct_p5 else 1
        for bar, val in zip(bars1, pct_p4):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.01,
                        f"{val:.0f}%", va="bottom", ha="center", fontsize=9, color="#FF9800")
        for bar, val in zip(bars2, pct_p5):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.01,
                        f"{val:.0f}%", va="bottom", ha="center", fontsize=9, color="#2196F3")

        xlabels = [s.replace("p4_","").replace("_p5_","+") for s in states_present]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylabel("% frames LYS90 close", fontsize=13)
        ax.set_title(f"LYS90 proximity per PIP2 state\n{channel_type} -- Subunit {sub}",
                     fontsize=14)
        ax.legend(fontsize=12)
        ax.set_ylim(top=min(120, max_val * 1.2) if max_val > 0 else 10)
        plt.tight_layout()
        fig.savefig(out_dir / f"lys90_per_state_{sub}.png", dpi=150)
        plt.close(fig)
        print(f"  LYS90 per-state plot -> {out_dir / f'lys90_per_state_{sub}.png'}")


def run_pip_pose_analysis(
    u,
    mapping_json:  str,
    results_dir,
    start_frame:   int,
    end_frame:     int,
    channel_type:  str,
    pip_atoms:     List[str] = None,
    threshold:     float     = 4.0,
    end2_all:      dict      = None,   # {ion_id: [frames]} from get_end2_all_frames
    end3_all:      dict      = None,   # {ion_id: [frames]} from get_end3_all_frames
):
    if pip_atoms is None:
        pip_atoms = ["P4", "P5"]

    results_dir = Path(results_dir)
    out_dir     = results_dir / "pip_pose"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _flatten_with_prev(d):
        """Return (crossing_frames_set, prev_frames_set) from {ion_id: [frames]}."""
        crossing, prev = set(), set()
        if d:
            for frames in d.values():
                for f in frames:
                    crossing.add(int(f))
                    if int(f) - 1 >= start_frame:
                        prev.add(int(f) - 1)
        return crossing, prev

    asn_gate_frames,  asn_gate_m1_frames  = _flatten_with_prev(end2_all)
    hbc_gate_frames,  hbc_gate_m1_frames  = _flatten_with_prev(end3_all)

    with open(mapping_json) as f:
        mapping = json.load(f)

    print("\n" + "=" * 60)
    print("PIP POSE ANALYSIS  (3 neighborhoods, min-dist assignment)")
    print(f"  Channel type : {channel_type}")
    print(f"  Mapping      : {mapping_json}")
    print(f"  Frames       : {start_frame} -> {end_frame}")
    print(f"  Threshold    : {threshold} A")
    print(f"  Neighborhoods (GIRK2): Nterm(ARG60+LYS64) | Middle(LYS90) | Cterm(LYS194+LYS199+LYS200)")
    print(f"  Neighborhoods (G1):    Nterm(52+50)        | Middle(79)    | Cterm(183+188+189)")
    print(f"  Middle: tracked separately, not used for assignment")
    print("=" * 60)

    all_rows = []

    for pip_resid_str, subunit_data in mapping.items():
        pip_resid = int(pip_resid_str)
        subunit   = list(subunit_data.keys())[0]
        res_map   = subunit_data[subunit]

        print(f"\n  PIP {pip_resid} -- subunit {subunit}")

        # Get subunit-specific PDB number thresholds
        nterm_pdb, middle_pdb, cterm_pdb = _get_neighborhood_pdb_nums(subunit)

        nterm_resids  = []
        middle_resids = []
        cterm_resids  = []

        for res_name, info in res_map.items():
            pdb_num = info["pdb_number"]
            raw     = info["raw_resid"]
            if pdb_num in nterm_pdb:
                nterm_resids.append(raw)
                print(f"    Nterm:  {res_name} (raw {raw})")
            elif pdb_num in middle_pdb:
                middle_resids.append(raw)
                print(f"    Middle: {res_name} (raw {raw})")
            elif pdb_num in cterm_pdb:
                cterm_resids.append(raw)
                print(f"    Cterm:  {res_name} (raw {raw})")

        # Deduplicate in case multiple PDB nums map to same raw resid (e.g. G1 nterm 52+52)
        nterm_resids  = list(dict.fromkeys(nterm_resids))
        middle_resids = list(dict.fromkeys(middle_resids))
        cterm_resids  = list(dict.fromkeys(cterm_resids))

        if not nterm_resids or not cterm_resids:
            print(f"    WARNING: incomplete neighborhoods, skipping PIP {pip_resid}")
            continue
        if not middle_resids:
            print(f"    WARNING: no middle residue found -- middle neighborhood empty")

        p_atoms = {}
        for atom_name in pip_atoms:
            sel = u.select_atoms(f"resid {pip_resid} and name {atom_name}")
            if len(sel) == 0:
                print(f"    WARNING: {atom_name} not found in resid {pip_resid}")
                continue
            p_atoms[atom_name] = sel[0]

        if len(p_atoms) < 2:
            print(f"    WARNING: need both P4 and P5, skipping PIP {pip_resid}")
            continue

        n_frames = end_frame - start_frame + 1

        for ts in tqdm(
            u.trajectory[start_frame:end_frame + 1],
            total=n_frames,
            desc=f"  PIP{pip_resid}",
            leave=False,
        ):
            frame  = ts.frame
            p4_pos = p_atoms["P4"].position
            p5_pos = p_atoms["P5"].position

            (p4_pose, p5_pose,
             p4_d_nt, p4_d_mi, p4_d_ct,
             p5_d_nt, p5_d_mi, p5_d_ct) = _assign_poses_both(
                p4_pos, p5_pos, u,
                nterm_resids, middle_resids, cterm_resids, threshold)

            state = f"p4_{p4_pose}_p5_{p5_pose}"

            res_dists = {}
            p4_close_res = set()
            p5_close_res = set()
            for res_name, info in res_map.items():
                raw   = info["raw_resid"]
                atoms = u.select_atoms(f"resid {raw}")
                p4_d  = round(float(distance_array(p4_pos.reshape(1,3), atoms.positions)[0].min()), 3)
                p5_d  = round(float(distance_array(p5_pos.reshape(1,3), atoms.positions)[0].min()), 3)
                res_dists[f"p4_d_{res_name}"] = p4_d
                res_dists[f"p5_d_{res_name}"] = p5_d
                if p4_d < threshold:
                    p4_close_res.add(res_name)
                if p5_d < threshold:
                    p5_close_res.add(res_name)
            n_p4_close     = len(p4_close_res)
            n_p5_close     = len(p5_close_res)
            n_unique_close = len(p4_close_res | p5_close_res)

            row = {
                "frame":              frame,
                "pip":                pip_resid,
                "subunit":            subunit,
                "p4_pose":            p4_pose,
                "p5_pose":            p5_pose,
                "state":              state,
                "n_p4_close_res":     n_p4_close,
                "n_p5_close_res":     n_p5_close,
                "n_unique_close_res": n_unique_close,
                "p4_close_res":       "_".join(sorted(p4_close_res)) if p4_close_res else "none",
                "p5_close_res":       "_".join(sorted(p5_close_res)) if p5_close_res else "none",
                "p4_d_nterm_min":     round(p4_d_nt, 3),
                "p4_d_middle_min":    round(p4_d_mi, 3),
                "p4_d_cterm_min":     round(p4_d_ct, 3),
                "p5_d_nterm_min":     round(p5_d_nt, 3),
                "p5_d_middle_min":    round(p5_d_mi, 3),
                "p5_d_cterm_min":     round(p5_d_ct, 3),
                "p4_lys90_close":     p4_d_mi < threshold,
                "p5_lys90_close":     p5_d_mi < threshold,
            }
            row.update(res_dists)
            all_rows.append(row)

    if not all_rows:
        print("No data collected.")
        return

    df = pd.DataFrame(all_rows)

    _run_outputs(df, out_dir, channel_type)

    # 4 gating frame subsets: ASN gate, ASN gate -1, HBC gate, HBC gate -1
    gate_subsets = [
        (asn_gate_frames,    "asn_gate_frames",    "ASN gate (end2)"),
        (asn_gate_m1_frames, "asn_gate_m1_frames", "ASN gate -1 (pre-end2)"),
        (hbc_gate_frames,    "hbc_gate_frames",    "HBC gate (end3)"),
        (hbc_gate_m1_frames, "hbc_gate_m1_frames", "HBC gate -1 (pre-end3)"),
    ]

    for frame_set, folder_name, label in gate_subsets:
        if not frame_set:
            print(f"  No {label} frames provided -- skipping.")
            continue
        df_sub = df[df["frame"].isin(frame_set)]
        if df_sub.empty:
            print(f"  No {label} frames found in trajectory range -- skipping.")
            continue
        gate_dir = out_dir / folder_name
        gate_dir.mkdir(exist_ok=True)
        print(f"\n  Running {label} analysis ({len(df_sub)} rows)...")
        _run_outputs(df_sub, gate_dir, channel_type, label=label)

    print(f"\n✓ PIP pose analysis complete -> {out_dir}")