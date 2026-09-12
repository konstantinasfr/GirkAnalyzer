"""
analysis/pip_bound_free_hbond_analysis.py
==========================================
Splits trajectory frames into two states based on PIP2 P5 proximity to a
target residue (e.g. GLN197), then compares H-bond statistics between states.

Uses existing CSVs only — NO trajectory scanning:
    - pip_proximity_5/PIP_<X>_<sub>/P5/distances_all_pdb.csv  (PIP-residue distances)
    - residue_neighbors/<resid>_<sub>/hbonds/hbond_per_frame.csv     (per-frame partners)
    - residue_neighbors/<resid>_<sub>/hbonds/hbond_atom_pair_summary.csv (bond types)

For each partner, bond type is determined from hbond_atom_pair_summary.csv.
Per-frame partner presence is determined from hbond_per_frame.csv partners column.

Output (results_dir/pip_bound_free_hbond/<resid_query>/PIP_<X>_<sub>/):
    bound/   free/
        hbond_all.csv + barplot_all.png
        hbond_bb_bb.csv + barplot_bb_bb.png
        hbond_sc_sc.csv + barplot_sc_sc.png
        hbond_bb_sc.csv + barplot_bb_sc.png
        hbond_sc_any.csv + barplot_sc_any.png
    comparison/
        hbond_comparison_<type>.csv  -- bound vs free side by side

Called from main.py:

    from analysis.pip_bound_free_hbond_analysis import run_pip_bound_free_hbond_analysis

    run_pip_bound_free_hbond_analysis(
        results_dir      = results_dir,
        resid_query      = "197",
        pip_resids       = [1301, 1302, 1303, 1304],
        pip_mapping      = pip_mapping,
        threshold        = 4.0,
        prescreen_factor = 5.0,
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

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


def _classify_bond(donor_atom: str, acceptor_atom: str) -> str:
    d_bb = donor_atom in _BB_ATOMS
    a_bb = acceptor_atom in _BB_ATOMS
    if d_bb and a_bb:         return "bb_bb"
    elif d_bb and not a_bb:   return "bb_sc"
    elif not d_bb and a_bb:   return "sc_bb"
    else:                     return "sc_sc"


def _get_partner_bond_types(atom_pair_csv: Path, pdb_label: str) -> Dict[str, set]:
    """
    For each partner residue, determine which bond types it forms with pdb_label.
    Returns dict: {partner: set of bond types}
    e.g. {"LYS200.2b": {"bb_bb", "sc_any"}, ...}
    """
    if not atom_pair_csv.exists():
        return {}

    df = pd.read_csv(atom_pair_csv)

    # strip residue name to get bare pdb id e.g. "GLN197.2b" -> "197.2b"
    bare_label = pdb_label
    for pfx in ["GLN","LYS","ARG","VAL","ILE","LEU","PHE","TRP","SER","THR",
                "MET","CYS","ASP","GLU","ASN","HIS","HID","HIE","HIP","PRO",
                "ALA","GLY","TYR"]:
        if pdb_label.startswith(pfx):
            bare_label = pdb_label[len(pfx):]
            break

    partner_types = {}
    for _, row in df.iterrows():
        dr, da, ar, aa = str(row["donor_res"]), str(row["donor_atom"]), \
                         str(row["acceptor_res"]), str(row["acceptor_atom"])

        # identify partner (not pdb_label)
        if bare_label in dr or pdb_label in dr:
            partner = ar
        else:
            partner = dr

        bt = _classify_bond(da, aa)
        if partner not in partner_types:
            partner_types[partner] = set()
        partner_types[partner].add(bt)
        # sc_any
        if bt in ("sc_sc", "bb_sc", "sc_bb"):
            partner_types[partner].add("sc_any")

    return partner_types


def _compute_state_counts(
    state_frames:  set,
    per_frame_df:  pd.DataFrame,
    partner_types: Dict[str, set],  # kept for backward compat, not used with new format
) -> Dict[str, Dict[str, int]]:
    """
    For frames in state_frames, count how many frames each partner forms
    each bond type. Uses new-format hbond_per_frame.csv with bond-type columns.

    Returns: {bond_type: {partner: n_frames}}
    """
    counts = {bt: {} for bt in BOND_TYPES}

    # filter to state frames
    df_state = per_frame_df[per_frame_df["frame"].isin(state_frames)].copy()

    has_new_format = "bb_bb_partners" in per_frame_df.columns

    col_map = {
        "all":    "partners",
        "bb_bb":  "bb_bb_partners",
        "sc_sc":  "sc_sc_partners",
        "bb_sc":  "bb_sc_partners",
        "sc_any": "sc_any_partners",
    }

    for bt, col in col_map.items():
        # fall back to "partners" if new columns not present
        use_col = col if (has_new_format and col in df_state.columns) else "partners"
        for _, row in df_state.iterrows():
            if row["any_hbond"] == 0:
                continue
            partners_str = str(row[use_col])
            if partners_str == "nan" or not partners_str.strip():
                continue
            for p in partners_str.split("|"):
                p = p.strip()
                if p:
                    counts[bt][p] = counts[bt].get(p, 0) + 1

    return counts


def _make_barplot(partner_counts, title, out_path, color, n_frames):
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
                bar.get_height() + max(vals) * 0.02,
                f"{val}\n({pct}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(partners, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Frames with H-bond", fontsize=13)
    ax.set_xlabel("Partner residue", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(top=max(vals) * 1.2 if vals else 1)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"      Saved -> {out_path.name}")


def _save_state(counts, state_dir, state_lbl, n_frames, pdb_label):
    state_dir.mkdir(parents=True, exist_ok=True)
    for bt, bt_label in BOND_TYPES.items():
        c    = counts[bt]
        rows = sorted(c.items(), key=lambda x: -x[1])
        df   = pd.DataFrame(rows, columns=["partner", "n_frames_hbond"])
        if len(df) > 0:
            df["pct_frames"] = (df["n_frames_hbond"] / n_frames * 100).round(2)
        else:
            df["pct_frames"] = []
        df.to_csv(state_dir / f"hbond_{bt}.csv", index=False)
        if c:
            _make_barplot(c,
                          f"{pdb_label} — {bt_label}  [{state_lbl}, n={n_frames}]",
                          state_dir / f"barplot_{bt}.png",
                          COLORS[bt], n_frames)
    print(f"    [{state_lbl}] {n_frames} frames — partners: {len(counts['all'])}")


def _save_comparison(counts_bound, counts_free, comp_dir, n_bound, n_free):
    comp_dir.mkdir(parents=True, exist_ok=True)
    for bt in BOND_TYPES:
        cb  = counts_bound[bt]
        cf  = counts_free[bt]
        all_p = sorted(set(list(cb.keys()) + list(cf.keys())))
        rows  = []
        for p in all_p:
            nb    = cb.get(p, 0)
            nf    = cf.get(p, 0)
            pct_b = round(100.0 * nb / n_bound, 2) if n_bound > 0 else 0
            pct_f = round(100.0 * nf / n_free,  2) if n_free  > 0 else 0
            rows.append({"partner": p,
                         "n_bound": nb, "pct_bound": pct_b,
                         "n_free":  nf, "pct_free":  pct_f,
                         "diff_pct": round(pct_f - pct_b, 2)})
        (pd.DataFrame(rows)
           .sort_values("diff_pct", ascending=False)
           .reset_index(drop=True)
           .to_csv(comp_dir / f"hbond_comparison_{bt}.csv", index=False))
    print(f"    Saved comparison CSVs")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_pip_bound_free_hbond_analysis(
    results_dir:      Path,
    resid_query:      str,
    pip_resids:       List[int],
    pip_mapping:      Dict[int, str],
    threshold:        float = 4.0,
    prescreen_factor: float = 5.0,
):
    results_dir = Path(results_dir)
    out_base    = results_dir / "pip_bound_free_hbond" / resid_query
    out_base.mkdir(parents=True, exist_ok=True)

    factor_str    = str(int(prescreen_factor)) \
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

        # 1. Find P5 distance CSV
        p5_csv = pip_prox_base / f"PIP_{pip_label}" / "P5" / "distances_all_pdb.csv"
        if not p5_csv.exists():
            print(f"    WARNING: {p5_csv} not found -- skipping.")
            continue

        df_p5  = pd.read_csv(p5_csv)
        frames = df_p5["frame"].values

        # find target column
        target_col = None
        for col in df_p5.columns:
            if col == "frame":
                continue
            if f"{resid_query}.{subunit}" in col:
                target_col = col
                break

        if target_col is None:
            print(f"    WARNING: no column matching '{resid_query}.{subunit}' -- skipping.")
            continue

        print(f"    Found target column: {target_col}")
        pdb_label = target_col  # e.g. GLN197.2b

        # 2. Split frames
        dist_vals  = df_p5[target_col].values
        bound_mask = dist_vals < threshold
        free_mask  = ~bound_mask
        n_bound    = int(bound_mask.sum())
        n_free     = int(free_mask.sum())
        n_total    = len(frames)

        print(f"    Bound (< {threshold}A): {n_bound} frames ({100*n_bound/n_total:.1f}%)")
        print(f"    Free  (>={threshold}A): {n_free} frames ({100*n_free/n_total:.1f}%)")

        if n_bound == 0:
            print(f"    No bound frames — skipping.")
            continue

        bound_frames = set(frames[bound_mask].tolist())
        free_frames  = set(frames[free_mask].tolist())

        # 3. Find H-bond CSVs
        nb_folder     = nb_base / f"{resid_query}_{subunit}"
        hb_dir        = nb_folder / "hbonds"
        per_frame_csv = hb_dir / "hbond_per_frame.csv"
        atom_pair_csv = hb_dir / "hbond_atom_pair_summary.csv"

        if not per_frame_csv.exists():
            print(f"    WARNING: hbond_per_frame.csv not found in {hb_dir} -- skipping.")
            continue
        if not atom_pair_csv.exists():
            print(f"    WARNING: hbond_atom_pair_summary.csv not found -- skipping.")
            continue

        # 4. Load per_frame CSV
        df_pf = pd.read_csv(per_frame_csv)
        has_new = "bb_bb_partners" in df_pf.columns
        print(f"    hbond_per_frame.csv format: {'new (bond-type columns)' if has_new else 'old (partners only)'}")

        # 5. Compute counts for bound and free
        counts_bound = _compute_state_counts(bound_frames, df_pf, {})
        counts_free  = _compute_state_counts(free_frames,  df_pf, {})

        # 7. Save results
        pip_out = out_base / f"PIP_{pip_label}"
        _save_state(counts_bound, pip_out / "bound", "bound", n_bound, pdb_label)
        _save_state(counts_free,  pip_out / "free",  "free",  n_free,  pdb_label)
        _save_comparison(counts_bound, counts_free, pip_out / "comparison", n_bound, n_free)
        print(f"    Results saved -> {pip_out}")

    print(f"\n✓ PIP bound/free H-bond analysis complete.")
    print(f"  Results in: {out_base}")