"""
residue_pip_p4p5_proximity.py

For a single RUN, checks — frame by frame, over [start_frame, end_frame] —
whether each given residue is close (within a cutoff distance) to the P4
and/or P5 phosphate atoms of its paired PIP2 molecule.

Input pairs are (pip_resid, residue_resid), e.g.:
    (1301, 471), (1302, 146), (1303, 1120), (1304, 795)

For each pair, computes the minimum distance per frame between ANY atom of
the residue and the P4 atom, and separately the P5 atom, of the given PIP.
A frame counts as "close" to P4 if min_dist(residue, P4) <= cutoff, "close"
to P5 if min_dist(residue, P5) <= cutoff, and "close_either" if either holds.

Additionally, each frame is put into exactly one of 4 mutually-exclusive
categories: "P4 only", "P5 only", "Both", "None" (close to neither) — this
avoids double counting when a frame is close to both P4 and P5 at once.

RAW DISTANCE CACHING
----------------------
The per-frame P4/P5 distances computed for each (pip_resid, residue_resid)
pair are cached to:
    <results_dir>/residue_pip_p4p5_proximity/raw_distances/pip{pip}_res{res}_distances.npz
On future calls with the same [start_frame, end_frame], distances are
loaded from this cache instead of re-walking the trajectory — useful when
you just want to add more gate thresholds/filters without recomputing
everything from scratch. Pass force=True to ignore the cache and recompute.

OPTIONAL GATE-FILTERED SUBSETS
--------------------------------
You can additionally repeat the exact same analysis restricted to frames
where one or more gates (HBC, upper G-loop, etc.) are open beyond a
threshold. Pass `gate_filters` — a list of dicts, one per gate — each
containing:
    "name"       : label used for the output sub-folder, e.g. "hbc", "upper_gl"
    "csv"        : path to the gate's per-frame distance CSV
                   (columns: frame, mean, A_C, B_D)
    "thresholds" : list of thresholds to test, e.g. [5, 6, 7, 8, 9, 10, 11, 12]
    "column"     : (optional) which column to threshold on, default "mean"

For every (gate, threshold) combination, results are written to their own
sub-folder, e.g.:
    hbc_5/, hbc_6/, ..., hbc_12/
    upper_gl_5/, upper_gl_6/, ..., upper_gl_12/
using the same 5 files as the full-run output — nothing overwrites anything
else.

Output saved to <results_dir>/residue_pip_p4p5_proximity/:
    p4p5_proximity.csv       — one row per pair: frames/pct close to P4, P5, either
    category_counts.csv      — one row per pair: frame COUNT per category
    category_counts.txt      — same info, human-readable text
    barplot_counts.png       — grouped bar plot, frame counts per category per pair
    barplot_pct.png          — grouped bar plot, % of frames per category per pair
    <gate>_<threshold>/      — (only if gate_filters given) same 5 files, per gate/threshold

Columns (p4p5_proximity.csv):
    channel_type, pip_resid, residue_resid, residue_pdb,
    n_frames,
    frames_close_p4, pct_close_p4,
    frames_close_p5, pct_close_p5,
    frames_close_either, pct_close_either

Columns (category_counts.csv):
    channel_type, pip_resid, residue_resid, residue_pdb, n_frames,
    count_P4_only, count_P5_only, count_Both, count_None,
    pct_P4_only, pct_P5_only, pct_Both, pct_None

Called from main.py:

    from analysis.residue_pip_p4p5_proximity import run_p4p5_proximity

    run_p4p5_proximity(
        u             = u,
        pairs         = [(1301, 471), (1302, 146), (1303, 1120), (1304, 795)],
        results_dir   = results_dir,
        start_frame   = start_frame,
        end_frame     = end_frame,
        channel_type  = args.channel_type,
        cutoff        = 4.0,
        force         = False,   # optional, set True to ignore cache and recompute
        gate_filters  = [
            {
                "name": "hbc",
                "csv": results_dir / "hbc_distances.csv",
                "thresholds": [5, 6, 7, 8, 9, 10, 11, 12],
            },
            {
                "name": "upper_gl",
                "csv": results_dir / "upper_gl_distances.csv",
                "thresholds": [5, 6, 7, 8, 9, 10, 11, 12],
            },
        ],
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import distance_array

from analysis.converter import to_pdb

warnings.filterwarnings("ignore")

DEFAULT_CUTOFF = 4.0
CATEGORY_COLORS = {
    "P4_only": "steelblue",
    "P5_only": "darkorange",
    "Both": "mediumseagreen",
    "None": "lightgray",
}


# -----------------------------------------------------------------------------
# Core per-pair distance computation
# -----------------------------------------------------------------------------


def _get_pair_distances(u, pip_resid: int, residue_resid: int,
                         start_frame: int, end_frame: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (dist_p4, dist_p5) arrays over the full [start_frame, end_frame]
    range for one (pip_resid, residue_resid) pair, or None if the residue
    can't be found. Both distances are computed in a SINGLE pass over the
    trajectory (one seek per frame instead of two).
    """
    residue_sel = u.select_atoms(f"resid {residue_resid}")
    if len(residue_sel) == 0:
        print(f"  [WARN] residue {residue_resid} not found — skipping pair "
              f"(PIP {pip_resid}, residue {residue_resid})")
        return None

    p4_atom = u.select_atoms(f"resid {pip_resid} and name P4")
    p5_atom = u.select_atoms(f"resid {pip_resid} and name P5")

    if len(p4_atom) == 0:
        print(f"  [WARN] P4 atom not found on PIP {pip_resid}")
    if len(p5_atom) == 0:
        print(f"  [WARN] P5 atom not found on PIP {pip_resid}")

    n_frames = end_frame - start_frame + 1
    dist_p4 = np.full(n_frames, np.inf)
    dist_p5 = np.full(n_frames, np.inf)

    have_p4 = len(p4_atom) > 0
    have_p5 = len(p5_atom) > 0

    if have_p4 or have_p5:
        for i, ts in enumerate(u.trajectory[start_frame:end_frame + 1]):
            if have_p4:
                dist_p4[i] = distance_array(residue_sel.positions, p4_atom.positions, box=ts.dimensions).min()
            if have_p5:
                dist_p5[i] = distance_array(residue_sel.positions, p5_atom.positions, box=ts.dimensions).min()

    return dist_p4, dist_p5


def _compute_stats(dist_p4: np.ndarray, dist_p5: np.ndarray, cutoff: float) -> dict:
    """
    Given (already-sliced, if desired) distance arrays, computes the close
    frame counts/percentages for P4, P5, either, and the 4 mutually
    exclusive categories.
    """
    n_frames = len(dist_p4)

    close_p4 = dist_p4 <= cutoff
    close_p5 = dist_p5 <= cutoff
    close_either = close_p4 | close_p5

    frames_p4 = int(close_p4.sum())
    frames_p5 = int(close_p5.sum())
    frames_either = int(close_either.sum())

    both_mask = close_p4 & close_p5
    p4_only_mask = close_p4 & ~close_p5
    p5_only_mask = close_p5 & ~close_p4
    none_mask = ~close_p4 & ~close_p5

    count_both = int(both_mask.sum())
    count_p4_only = int(p4_only_mask.sum())
    count_p5_only = int(p5_only_mask.sum())
    count_none = int(none_mask.sum())

    return {
        "n_frames": n_frames,
        "frames_close_p4": frames_p4,
        "pct_close_p4": round(100.0 * frames_p4 / n_frames, 2) if n_frames else 0.0,
        "frames_close_p5": frames_p5,
        "pct_close_p5": round(100.0 * frames_p5 / n_frames, 2) if n_frames else 0.0,
        "frames_close_either": frames_either,
        "pct_close_either": round(100.0 * frames_either / n_frames, 2) if n_frames else 0.0,
        "count_P4_only": count_p4_only,
        "count_P5_only": count_p5_only,
        "count_Both": count_both,
        "count_None": count_none,
        "pct_P4_only": round(100.0 * count_p4_only / n_frames, 2) if n_frames else 0.0,
        "pct_P5_only": round(100.0 * count_p5_only / n_frames, 2) if n_frames else 0.0,
        "pct_Both": round(100.0 * count_both / n_frames, 2) if n_frames else 0.0,
        "pct_None": round(100.0 * count_none / n_frames, 2) if n_frames else 0.0,
    }


# -----------------------------------------------------------------------------
# Raw per-frame distance caching (avoid recomputing across reruns)
# -----------------------------------------------------------------------------

def _cache_path(cache_dir: Path, pip_resid: int, residue_resid: int) -> Path:
    return cache_dir / f"pip{pip_resid}_res{residue_resid}_distances.npz"


def _load_cached_distances(cache_path: Path, start_frame: int, end_frame: int
                            ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Loads cached (dist_p4, dist_p5) if the file exists AND was computed for
    the exact same [start_frame, end_frame] range. Otherwise returns None
    so the caller falls back to recomputing.
    """
    if not cache_path.exists():
        return None

    data = np.load(cache_path)
    if int(data["start_frame"]) != start_frame or int(data["end_frame"]) != end_frame:
        print(f"  [CACHE] {cache_path.name} exists but covers a different frame range "
              f"({int(data['start_frame'])}-{int(data['end_frame'])}) — recomputing")
        return None

    print(f"  [CACHE] Loaded distances from {cache_path}")
    return data["dist_p4"], data["dist_p5"]


def _save_cached_distances(cache_path: Path, dist_p4: np.ndarray, dist_p5: np.ndarray,
                            start_frame: int, end_frame: int):
    np.savez(cache_path, dist_p4=dist_p4, dist_p5=dist_p5,
              start_frame=start_frame, end_frame=end_frame)
    print(f"  [CACHE] Saved distances -> {cache_path}")


def _get_or_compute_pair_distances(u, pip_resid: int, residue_resid: int,
                                    start_frame: int, end_frame: int,
                                    cache_dir: Path, force: bool = False
                                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (dist_p4, dist_p5) for one pair, loading from cache if available
    (and force=False), otherwise computing from the trajectory and caching
    the result for next time.
    """
    cache_path = _cache_path(cache_dir, pip_resid, residue_resid)

    if not force:
        cached = _load_cached_distances(cache_path, start_frame, end_frame)
        if cached is not None:
            return cached

    dists = _get_pair_distances(u, pip_resid, residue_resid, start_frame, end_frame)
    if dists is None:
        return None

    dist_p4, dist_p5 = dists
    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_cached_distances(cache_path, dist_p4, dist_p5, start_frame, end_frame)
    return dist_p4, dist_p5


# -----------------------------------------------------------------------------
# Gate mask loading (works for HBC, upper G-loop, lower G-loop, ... — any
# gate-distance CSV with columns: frame, mean, A_C, B_D)
# -----------------------------------------------------------------------------

def _load_gate_mask(gate_csv, start_frame: int, end_frame: int,
                     threshold: float, column: str = "mean") -> Optional[np.ndarray]:
    """
    Loads a gate distances CSV (columns: frame, mean, A_C, B_D) and builds
    a boolean mask, aligned to [start_frame, end_frame], that is True where
    the given column exceeds threshold (gate considered "open").
    """
    gate_csv = Path(gate_csv)
    if not gate_csv.exists():
        print(f"  [WARN] gate csv not found at {gate_csv} — skipping this filter")
        return None

    gate_df = pd.read_csv(gate_csv)
    sub = gate_df[(gate_df["frame"] >= start_frame) & (gate_df["frame"] <= end_frame)].sort_values("frame")

    n_frames_expected = end_frame - start_frame + 1
    if len(sub) != n_frames_expected:
        print(f"  [WARN] gate csv frame range ({len(sub)} rows) does not match "
              f"expected n_frames ({n_frames_expected}) — skipping this filter")
        return None

    mask = (sub[column] > threshold).values
    print(f"  Gate filter ({gate_csv.name}): {mask.sum()}/{len(mask)} frames pass "
          f"({column} > {threshold})")
    return mask


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _pair_label(pip_resid: int, residue_resid: int, channel_type: str) -> str:
    pdb_label = to_pdb(residue_resid, channel_type)
    return f"PIP{pip_resid}\n{pdb_label}"


def _make_category_barplot(df: pd.DataFrame, value_cols: List[str], ylabel: str,
                            title: str, out_path: Path, as_pct: bool, channel_type: str):
    """
    Grouped bar plot: one group per (pip_resid, residue_resid) pair,
    one bar per category (P4_only, P5_only, Both, None).
    """
    labels = [_pair_label(r.pip_resid, r.residue_resid, channel_type) for r in df.itertuples()]
    n_pairs = len(df)
    n_cats = len(value_cols)

    x = np.arange(n_pairs)
    width = 0.8 / n_cats

    fig, ax = plt.subplots(figsize=(max(6, n_pairs * 1.6 + 2), 5))

    for i, col in enumerate(value_cols):
        cat_name = col.replace("pct_", "").replace("count_", "")
        color = CATEGORY_COLORS.get(cat_name, None)
        offset = (i - (n_cats - 1) / 2) * width
        bars = ax.bar(x + offset, df[col], width=width, label=cat_name,
                       color=color, edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, df[col]):
            fmt = f"{val:.1f}" if as_pct else f"{int(val)}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=n_cats)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_category_txt(df: pd.DataFrame, out_path: Path, channel_type: str):
    with open(out_path, "w") as f:
        f.write(f"P4/P5 proximity category counts — {channel_type}\n")
        f.write("=" * 60 + "\n\n")
        for r in df.itertuples():
            pdb_label = to_pdb(r.residue_resid, channel_type)
            f.write(f"PIP {r.pip_resid} <-> residue {pdb_label} (raw resid {r.residue_resid}) "
                    f"(n_frames={r.n_frames})\n")
            f.write(f"  P4 only : {r.count_P4_only:6d}  ({r.pct_P4_only:5.1f}%)\n")
            f.write(f"  P5 only : {r.count_P5_only:6d}  ({r.pct_P5_only:5.1f}%)\n")
            f.write(f"  Both    : {r.count_Both:6d}  ({r.pct_Both:5.1f}%)\n")
            f.write(f"  None    : {r.count_None:6d}  ({r.pct_None:5.1f}%)\n\n")


# -----------------------------------------------------------------------------
# Shared output writer (used for full-run and every gate/threshold combo)
# -----------------------------------------------------------------------------

def _write_all_outputs(rows: List[dict], out_dir: Path, channel_type: str) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No valid pairs — nothing written to {out_dir}.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["residue_pdb"] = df.apply(lambda r: to_pdb(r["residue_resid"], channel_type), axis=1)

    col_order = [
        "channel_type", "pip_resid", "residue_resid", "residue_pdb", "n_frames",
        "frames_close_p4", "pct_close_p4",
        "frames_close_p5", "pct_close_p5",
        "frames_close_either", "pct_close_either",
    ]
    df_main = df[col_order]
    out_path = out_dir / "p4p5_proximity.csv"
    df_main.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    cat_cols = [
        "channel_type", "pip_resid", "residue_resid", "residue_pdb", "n_frames",
        "count_P4_only", "count_P5_only", "count_Both", "count_None",
        "pct_P4_only", "pct_P5_only", "pct_Both", "pct_None",
    ]
    df_cat = df[cat_cols]
    cat_csv_path = out_dir / "category_counts.csv"
    df_cat.to_csv(cat_csv_path, index=False)
    print(f"Saved -> {cat_csv_path}")

    cat_txt_path = out_dir / "category_counts.txt"
    _write_category_txt(df_cat, cat_txt_path, channel_type)
    print(f"Saved -> {cat_txt_path}")

    counts_plot_path = out_dir / "barplot_counts.png"
    _make_category_barplot(
        df_cat,
        value_cols=["count_P4_only", "count_P5_only", "count_Both", "count_None"],
        ylabel="Frame count",
        title=f"{channel_type} — Frames close to P4 / P5 / Both / None",
        out_path=counts_plot_path,
        as_pct=False,
        channel_type=channel_type,
    )
    print(f"Saved -> {counts_plot_path}")

    pct_plot_path = out_dir / "barplot_pct.png"
    _make_category_barplot(
        df_cat,
        value_cols=["pct_P4_only", "pct_P5_only", "pct_Both", "pct_None"],
        ylabel="% of frames",
        title=f"{channel_type} — % Frames close to P4 / P5 / Both / None",
        out_path=pct_plot_path,
        as_pct=True,
        channel_type=channel_type,
    )
    print(f"Saved -> {pct_plot_path}")

    return df_main


# -----------------------------------------------------------------------------
# Public entry point (called from main.py)
# -----------------------------------------------------------------------------

def run_p4p5_proximity(
    u,
    pairs: List[Tuple[int, int]],
    results_dir,
    start_frame: int,
    end_frame: int,
    channel_type: str,
    cutoff: float = DEFAULT_CUTOFF,
    gate_filters: Optional[List[Dict[str, Any]]] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Computes P4/P5 proximity stats for a list of (pip_resid, residue_resid)
    pairs, for a single RUN (shared Universe already loaded in main.py).

    Raw per-frame distances are cached to
        <out_dir>/raw_distances/pip{pip}_res{res}_distances.npz
    On future calls with the same [start_frame, end_frame], distances are
    loaded from this cache instead of re-walking the trajectory. Pass
    force=True to ignore any cache and recompute from scratch (overwriting
    the cache file).

    gate_filters: optional list of dicts, one per gate, e.g.:
        [
            {"name": "hbc",      "csv": ".../hbc_distances.csv",      "thresholds": [5,6,7,8,9,10,11,12]},
            {"name": "upper_gl", "csv": ".../upper_gl_distances.csv", "thresholds": [5,6,7,8,9,10,11,12]},
        ]
    Each dict may also include "column" (default "mean") to threshold on
    A_C or B_D instead. For every (gate, threshold) pair, results are
    written to <out_dir>/<gate_name>_<threshold>/.
    Leave gate_filters=None to skip all gate-filtered analysis.
    """
    out_dir = Path(results_dir) / "residue_pip_p4p5_proximity"
    cache_dir = out_dir / "raw_distances"

    # Build every (gate_name, threshold) -> mask up front, reused for every residue pair
    gate_masks: Dict[Tuple[str, float], np.ndarray] = {}
    for gate in (gate_filters or []):
        name = gate["name"]
        csv_path = gate["csv"]
        thresholds = gate["thresholds"]
        column = gate.get("column", "mean")
        for threshold in thresholds:
            mask = _load_gate_mask(csv_path, start_frame, end_frame, threshold, column)
            if mask is not None:
                gate_masks[(name, threshold)] = mask

    rows_full = []
    rows_by_gate = {key: [] for key in gate_masks}

    for pip_resid, residue_resid in pairs:
        print(f"[{channel_type}] PIP {pip_resid} <-> residue {residue_resid} "
              f"(cutoff={cutoff} A)")
        dists = _get_or_compute_pair_distances(
            u, pip_resid, residue_resid, start_frame, end_frame, cache_dir, force=force,
        )
        if dists is None:
            continue
        dist_p4, dist_p5 = dists

        stats_full = _compute_stats(dist_p4, dist_p5, cutoff)
        stats_full["pip_resid"] = pip_resid
        stats_full["residue_resid"] = residue_resid
        stats_full["channel_type"] = channel_type
        rows_full.append(stats_full)

        for (gate_name, threshold), mask in gate_masks.items():
            stats_gate = _compute_stats(dist_p4[mask], dist_p5[mask], cutoff)
            stats_gate["pip_resid"] = pip_resid
            stats_gate["residue_resid"] = residue_resid
            stats_gate["channel_type"] = channel_type
            rows_by_gate[(gate_name, threshold)].append(stats_gate)

    df_full = _write_all_outputs(rows_full, out_dir, channel_type)

    for (gate_name, threshold), rows in rows_by_gate.items():
        threshold_tag = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
        gate_out_dir = out_dir / f"{gate_name}_{threshold_tag}"
        _write_all_outputs(rows, gate_out_dir, channel_type)

    return df_full