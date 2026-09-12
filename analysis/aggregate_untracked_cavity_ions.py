"""
analysis/pip_bound_free_hbond_analysis.py
==========================================
Splits trajectory frames into two states based on PIP2 P5 proximity to a
target residue (e.g. GLN197), then compares H-bond statistics between states.

For each PIP:
  1. Finds the subunit automatically from PIP folder name (e.g. PIP_1301_2b -> 2b)
  2. Looks for target residue column in pip_proximity_5/PIP_<X>_<sub>/P5/distances_all_pdb.csv
  3. Splits frames: dist < threshold = "bound", dist >= threshold = "free"
  4. Reads residue_neighbors/<resid>_<sub>/hbonds/ for H-bond data
  5. For each state computes H-bond statistics by bond type:
     - all H-bonds
     - backbone-backbone
     - sidechain-sidechain
     - backbone-sidechain
     - sidechain-any

Output (saved in results_dir/pip_bound_free_hbond/<resid_query>/PIP_<X>_<sub>/):
    bound/
        hbond_all.csv
        hbond_bb_bb.csv
        hbond_sc_sc.csv
        hbond_bb_sc.csv
        hbond_sc_any.csv
        barplot_all.png
        barplot_bb_bb.png
        barplot_sc_sc.png
        barplot_bb_sc.png
        barplot_sc_any.png
    free/
        (same files)
    comparison/
        hbond_comparison_all.csv       -- bound vs free side by side
        hbond_comparison_bb_bb.csv
        hbond_comparison_sc_sc.csv
        hbond_comparison_bb_sc.csv
        hbond_comparison_sc_any.csv

Called from main.py:

    from analysis.pip_bound_free_hbond_analysis import run_pip_bound_free_hbond_analysis

    run_pip_bound_free_hbond_analysis(
        results_dir      = results_dir,
        resid_query      = "197",        # PDB number without subunit
        pip_resids       = [1301, 1302, 1303, 1304],
        pip_mapping      = pip_mapping,  # {1301: "2b", ...}
        threshold        = 4.0,
        prescreen_factor = 5.0,
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BB_ATOMS = {"N", "CA", "C", "O", "H", "HA", "H1", "H2", "H3"}

BOND_TYPES = {
    "all":    "All H-bonds",
    "bb_bb":  "Backbone-Backbone",
    "sc_sc":  "Sidechain-Sidechain",
    "bb_sc":  "Backbone-Sidechain",
    "sc_any": "Sidechain Involved",
}

COLORS = {
    "all":    "steelblue",
    "bb_bb":  "steelblue",
    "sc_sc":  "darkorange",
    "bb_sc":  "mediumseagreen",
    "sc_any": "mediumpurple",
}


def _bond_type(donor_atom: str, acceptor_atom: str) -> str:
    d_bb = donor_atom in _BB_ATOMS
    a_bb = acceptor_atom in _BB_ATOMS
    if d_bb and a_bb:         return "bb_bb"
    elif d_bb and not a_bb:   return "bb_sc"
    elif not d_bb and a_bb:   return "sc_bb"
    else:                     return "sc_sc"


def _is_sc_any(donor_atom: str, acceptor_atom: str) -> bool:
    return donor_atom not in _BB_ATOMS or acceptor_atom not in _BB_ATOMS


def _make_barplot(
    partner_counts: dict,
    title:          str,
    out_path:       Path,
    color:          str,
    n_frames:       int,
):
    if not partner_counts:
        return
    partners = sorted(partner_counts.keys(), key=lambda p: -partner_counts[p])
    vals     = [partner_counts[p] for p in partners]
    fig_w    = max(8, len(partners) * 0.7)
    fig, ax  = plt.subplots(figsize=(fig_w, 5))
    x        = np.arange(len(partners))
    bars     = ax.bar(x, vals, color=color, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, vals):
        pct = round(100.0 * val / n_frames, 1) if n_frames > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + n_frames * 0.002,
                f"{val}\n({pct}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(partners, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Frames with H-bond", fontsize=13)
    ax.set_xlabel("Partner residue", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15 if vals else 1)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _compute_hbond_stats(
    frame_mask:     np.ndarray,   # boolean mask for this state
    frames:         np.ndarray,   # all frame numbers
    hb_dir:         Path,
    atom_pair_csv:  Path,
) -> dict:
    """
    For a set of frames (frame_mask), compute H-bond partner frequencies
    broken down by bond type.

    Returns dict:
        {
            "all":    {partner: n_frames},
            "bb_bb":  {partner: n_frames},
            "sc_sc":  {partner: n_frames},
            "bb_sc":  {partner: n_frames},
            "sc_any": {partner: n_frames},
        }
    """
    n_state = int(frame_mask.sum())
    if n_state == 0:
        return {bt: {} for bt in BOND_TYPES}

    state_frames = set(frames[frame_mask].tolist())

    counts = {bt: {} for bt in BOND_TYPES}

    # iterate over per-pair type CSVs
    for pair_dir in sorted(hb_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        type_csv = pair_dir / "hbond_type_per_frame.csv"
        if not type_csv.exists():
            continue

        # partner label from folder name e.g. "197_2b_LYS200_2b" -> "LYS200.2b"
        folder   = pair_dir.name
        parts    = folder.split("_")
        # folder format: <resid>_<sub>_<partner_resname><partner_num>_<partner_sub>
        # e.g. 197_2b_LYS200_2b
        # partner = everything after the second underscore, rejoined with "."
        # find split point: first two underscores are target resid + sub
        try:
            # target part is like "197_2b" (2 parts), partner is rest
            # count underscores: 197_2b_LYS200_2b -> [197, 2b, LYS200, 2b]
            # partner = parts[2] + "." + parts[3]  but resname may have numbers
            # safer: split on first 2 underscores
            target_part = "_".join(parts[:2])
            partner_raw = "_".join(parts[2:])
            # convert LYS200_2b -> LYS200.2b
            # last segment after final underscore is subunit
            partner_parts = partner_raw.rsplit("_", 1)
            if len(partner_parts) == 2:
                partner_lbl = f"{partner_parts[0]}.{partner_parts[1]}"
            else:
                partner_lbl = partner_raw
        except Exception:
            partner_lbl = folder

        df_type = pd.read_csv(type_csv)

        # filter to state frames
        df_state = df_type[df_type["frame"].isin(state_frames)]

        if df_state.empty:
            continue

        # all H-bonds
        n_all = int((df_state["bb_bb"] | df_state["sc_sc"] |
                     df_state["bb_sc"] | df_state.get("sc_bb",
                     pd.Series(np.zeros(len(df_state), dtype=int)))).sum())
        if n_all > 0:
            counts["all"][partner_lbl] = n_all

        # bb_bb
        n_bb = int(df_state["bb_bb"].sum())
        if n_bb > 0:
            counts["bb_bb"][partner_lbl] = n_bb

        # sc_sc
        n_sc = int(df_state["sc_sc"].sum())
        if n_sc > 0:
            counts["sc_sc"][partner_lbl] = n_sc

        # bb_sc
        n_bs = int(df_state["bb_sc"].sum())
        if n_bs > 0:
            counts["bb_sc"][partner_lbl] = n_bs

        # sc_any
        sc_bb_col = df_state.get("sc_bb", pd.Series(np.zeros(len(df_state), dtype=int)))
        n_sa = int(np.clip(df_state["sc_sc"].values +
                           df_state["bb_sc"].values +
                           sc_bb_col.values, 0, 1).sum())
        if n_sa > 0:
            counts["sc_any"][partner_lbl] = n_sa

    return counts


def _save_state_results(
    counts:    dict,
    state_dir: Path,
    state_lbl: str,
    n_frames:  int,
    pdb_label: str,
):
    """Save CSVs and bar plots for one state (bound or free)."""
    state_dir.mkdir(parents=True, exist_ok=True)

    for bt, bt_label in BOND_TYPES.items():
        c = counts[bt]
        # CSV
        rows = sorted(c.items(), key=lambda x: -x[1])
        df   = pd.DataFrame(rows, columns=["partner", "n_frames_hbond"])
        df["pct_frames"] = (df["n_frames_hbond"] / n_frames * 100).round(2)
        df.to_csv(state_dir / f"hbond_{bt}.csv", index=False)

        # bar plot
        title = f"{pdb_label} — {bt_label}  [{state_lbl}, n={n_frames} frames]"
        _make_barplot(c, title, state_dir / f"barplot_{bt}.png",
                      COLORS[bt], n_frames)

    print(f"    [{state_lbl}] {n_frames} frames — "
          f"partners with any H-bond: {len(counts['all'])}")


def _save_comparison(
    counts_bound: dict,
    counts_free:  dict,
    comp_dir:     Path,
    n_bound:      int,
    n_free:       int,
    pdb_label:    str,
):
    """Save side-by-side comparison CSVs for bound vs free."""
    comp_dir.mkdir(parents=True, exist_ok=True)

    for bt, bt_label in BOND_TYPES.items():
        cb = counts_bound[bt]
        cf = counts_free[bt]
        all_partners = sorted(set(list(cb.keys()) + list(cf.keys())))
        rows = []
        for p in all_partners:
            nb    = cb.get(p, 0)
            nf    = cf.get(p, 0)
            pct_b = round(100.0 * nb / n_bound, 2) if n_bound > 0 else 0
            pct_f = round(100.0 * nf / n_free,  2) if n_free  > 0 else 0
            rows.append({
                "partner":         p,
                "n_bound":         nb,
                "pct_bound":       pct_b,
                "n_free":          nf,
                "pct_free":        pct_f,
                "diff_pct":        round(pct_f - pct_b, 2),
            })
        df = (pd.DataFrame(rows)
              .sort_values("diff_pct", ascending=False)
              .reset_index(drop=True))
        df.to_csv(comp_dir / f"hbond_comparison_{bt}.csv", index=False)

    print(f"    Saved comparison CSVs -> {comp_dir.name}/")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_pip_bound_free_hbond_analysis(
    results_dir:      Path,
    resid_query:      str,          # e.g. "197"  (PDB number, no subunit)
    pip_resids:       List[int],    # e.g. [1301, 1302, 1303, 1304]
    pip_mapping:      Dict[int, str],  # e.g. {1301: "2b", ...}
    threshold:        float = 4.0,
    prescreen_factor: float = 5.0,
):
    results_dir = Path(results_dir)
    out_base    = results_dir / "pip_bound_free_hbond" / resid_query
    out_base.mkdir(parents=True, exist_ok=True)

    factor_str  = str(int(prescreen_factor)) \
                  if prescreen_factor == int(prescreen_factor) \
                  else str(prescreen_factor)
    pip_prox_base = results_dir / f"pip_proximity_{factor_str}"
    nb_base       = results_dir / "residue_neighbors"

    print("\n" + "=" * 70)
    print("PIP BOUND/FREE H-BOND ANALYSIS")
    print(f"  results_dir   : {results_dir}")
    print(f"  resid_query   : {resid_query}")
    print(f"  pip_resids    : {pip_resids}")
    print(f"  threshold     : {threshold} A")
    print(f"  pip_prox_dir  : {pip_prox_base}")
    print("=" * 70)

    for pip_resid in pip_resids:
        subunit   = pip_mapping.get(pip_resid, "")
        pip_label = f"{pip_resid}_{subunit}" if subunit else str(pip_resid)

        print(f"\n  Processing PIP {pip_label}  (subunit: {subunit})")

        # ------------------------------------------------------------------
        # 1. Find P5 distance CSV and locate target residue column
        # ------------------------------------------------------------------
        p5_csv = pip_prox_base / f"PIP_{pip_label}" / "P5" / "distances_all_pdb.csv"
        if not p5_csv.exists():
            print(f"    WARNING: {p5_csv} not found -- skipping.")
            continue

        df_p5   = pd.read_csv(p5_csv)
        frames  = df_p5["frame"].values

        # find column matching resid_query.subunit
        target_col = None
        for col in df_p5.columns:
            if col == "frame":
                continue
            # match e.g. "GLN197.2b" or "197.2b"
            if f"{resid_query}.{subunit}" in col:
                target_col = col
                break

        if target_col is None:
            print(f"    WARNING: no column matching '{resid_query}.{subunit}' "
                  f"in {p5_csv.name} -- skipping.")
            print(f"    Available columns: {[c for c in df_p5.columns if c != 'frame'][:10]}")
            continue

        print(f"    Found target column: {target_col}")

        # ------------------------------------------------------------------
        # 2. Split frames into bound / free
        # ------------------------------------------------------------------
        dist_vals  = df_p5[target_col].values
        bound_mask = dist_vals < threshold
        free_mask  = dist_vals >= threshold
        n_bound    = int(bound_mask.sum())
        n_free     = int(free_mask.sum())
        n_total    = len(frames)

        print(f"    Bound (< {threshold}A): {n_bound} frames ({100*n_bound/n_total:.1f}%)")
        print(f"    Free  (>={threshold}A): {n_free} frames ({100*n_free/n_total:.1f}%)")

        if n_bound == 0:
            print(f"    No bound frames — skipping.")
            continue
        if n_free == 0:
            print(f"    No free frames — skipping.")
            continue

        # ------------------------------------------------------------------
        # 3. Find H-bond data for this subunit
        # ------------------------------------------------------------------
        nb_folder = nb_base / f"{resid_query}_{subunit}"
        hb_dir    = nb_folder / "hbonds"

        if not hb_dir.exists():
            print(f"    WARNING: {hb_dir} not found -- skipping.")
            print(f"    Run residue_neighbors_analysis first for resid {resid_query}.{subunit}")
            continue

        atom_pair_csv = hb_dir / "hbond_atom_pair_summary.csv"
        if not atom_pair_csv.exists():
            print(f"    WARNING: hbond_atom_pair_summary.csv not found -- skipping.")
            continue

        # check type CSVs exist
        type_csvs = list(hb_dir.glob("*/hbond_type_per_frame.csv"))
        if not type_csvs:
            print(f"    WARNING: no hbond_type_per_frame.csv found in pair folders.")
            print(f"    Run residue_neighbors_analysis again to generate type CSVs.")
            continue

        print(f"    Found {len(type_csvs)} pair type CSVs")

        pdb_label = f"{resid_query}.{subunit}"

        # ------------------------------------------------------------------
        # 4. Compute H-bond stats for each state
        # ------------------------------------------------------------------
        print(f"    Computing H-bond stats for bound frames...")
        counts_bound = _compute_hbond_stats(bound_mask, frames, hb_dir, atom_pair_csv)

        print(f"    Computing H-bond stats for free frames...")
        counts_free  = _compute_hbond_stats(free_mask,  frames, hb_dir, atom_pair_csv)

        # ------------------------------------------------------------------
        # 5. Save results
        # ------------------------------------------------------------------
        pip_out = out_base / f"PIP_{pip_label}"
        _save_state_results(counts_bound, pip_out / "bound", "bound", n_bound, pdb_label)
        _save_state_results(counts_free,  pip_out / "free",  "free",  n_free,  pdb_label)
        _save_comparison(counts_bound, counts_free,
                         pip_out / "comparison", n_bound, n_free, pdb_label)

        print(f"    Results saved -> {pip_out}")

    print(f"\n✓ PIP bound/free H-bond analysis complete.")
    print(f"  Results in: {out_base}")