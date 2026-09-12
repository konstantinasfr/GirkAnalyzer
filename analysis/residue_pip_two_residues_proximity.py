"""
residue_pip_two_residues_proximity.py

For a single RUN, checks — frame by frame, over [start_frame, end_frame] —
whether a chosen PIP2 phosphate (P4 or P5) is close (within a cutoff
distance) to residue A (e.g. 199) and/or residue B (e.g. 64) of the matched
subunit.

Input triples are (pip_resid, residue_a_resid, residue_b_resid), e.g.:
    (1301, 471, 336), (1302, 146, 10), (1303, 1120, 985), (1304, 795, 660)

For each triple, computes the minimum distance per frame between the PIP's
chosen phosphate atom and ANY atom of residue A, and separately residue B.
A frame counts as "close" to residue A if min_dist(phosphate, res_A) <=
cutoff, "close" to residue B if min_dist(phosphate, res_B) <= cutoff.

Each frame is put into exactly one of 4 mutually-exclusive categories:
    "<label_a>_only"  — close to residue A, not residue B
    "<label_b>_only"  — close to residue B, not residue A
    "Both"            — close to both at once
    "None"            — close to neither
(default label_a="199", label_b="64" — override via res_a_label/res_b_label
if you reuse this for a different residue pair)

OUTPUT LOCATION
-----------------
Results are written under ONE general folder, with a sub-folder named
"{phosphate}_{res_b_label}_{res_a_label}" — so running this for both P4 and
P5 (or different residue pairs) never overwrites anything:
    <results_dir>/residue_pip_two_residues_proximity/P5_64_199/...
    <results_dir>/residue_pip_two_residues_proximity/P4_64_199/...

RAW DISTANCE CACHING
----------------------
Per-triple distances are cached to:
    <results_dir>/residue_pip_two_residues_proximity/<phosphate>_<res_b_label>_<res_a_label>/raw_distances/pip{pip}_resA{a}_resB{b}_distances.npz
On future calls with the same [start_frame, end_frame], distances are
loaded from cache instead of re-walking the trajectory. Pass force=True to
ignore the cache and recompute.

OPTIONAL GATE-FILTERED SUBSETS
--------------------------------
Same as residue_pip_p4p5_proximity.py — pass `gate_filters`, a list of dicts:
    {"name": "hbc", "csv": ".../hbc_distances.csv", "thresholds": [5,...,12]}
Each (gate, threshold) combination gets its own sub-folder, e.g.:
    <results_dir>/residue_pip_two_residues_proximity/P5_64_199/hbc_5/, hbc_6/, ...

Output saved to <results_dir>/residue_pip_two_residues_proximity/<phosphate>_<res_b_label>_<res_a_label>/:
    two_residue_proximity.csv   — one row per triple: frames/pct close to A, B, either
    category_counts.csv         — one row per triple: frame COUNT per category
    category_counts.txt         — same info, human-readable text
    barplot_counts.png          — grouped bar plot, frame counts per category per triple
    barplot_pct.png             — grouped bar plot, % of frames per category per triple
    <gate>_<threshold>/         — (only if gate_filters given) same 5 files, per gate/threshold

Called from main.py:

    from analysis.residue_pip_two_residues_proximity import run_phosphate_two_residue_proximity

    run_phosphate_two_residue_proximity(
        u             = u,
        triples       = [(1301, 471, 336), (1302, 146, 10), (1303, 1120, 985), (1304, 795, 660)],
        results_dir   = results_dir,
        start_frame   = start_frame,
        end_frame     = end_frame,
        channel_type  = args.channel_type,
        phosphate     = "P5",   # or "P4"
        cutoff        = 4.0,
        res_a_label   = "199",
        res_b_label   = "64",
        res_a_aa      = "LYS",
        res_b_aa      = "LYS",
        res_b_aa_overrides = {10: "ARG"},   # G1 copy of residue 64 is an Arg, not Lys
        force         = False,
        gate_filters  = [
            {"name": "hbc",      "csv": results_dir / "hbc_distances.csv",      "thresholds": [5,6,7,8,9,10,11,12]},
            {"name": "upper_gl", "csv": results_dir / "upper_gl_distances.csv", "thresholds": [5,6,7,8,9,10,11,12]},
        ],
    )

    # to also get P4, just call again with phosphate="P4" — writes to a
    # separate sub-folder, nothing gets overwritten:
    run_phosphate_two_residue_proximity(..., phosphate="P4", ...)
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


# -----------------------------------------------------------------------------
# Core per-triple distance computation
# -----------------------------------------------------------------------------

def _get_triple_distances(u, pip_resid: int, res_a_resid: int, res_b_resid: int,
                           start_frame: int, end_frame: int,
                           phosphate: str = "P5") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (dist_a, dist_b) arrays over [start_frame, end_frame]: the
    per-frame minimum distance between the PIP's phosphate atom (P4 or P5)
    and residue A, and separately residue B. Both computed in a single
    trajectory pass. Returns None if either residue can't be found.
    """
    res_a_sel = u.select_atoms(f"resid {res_a_resid}")
    res_b_sel = u.select_atoms(f"resid {res_b_resid}")

    if len(res_a_sel) == 0:
        print(f"  [WARN] residue A {res_a_resid} not found — skipping triple "
              f"(PIP {pip_resid}, {res_a_resid}, {res_b_resid})")
        return None
    if len(res_b_sel) == 0:
        print(f"  [WARN] residue B {res_b_resid} not found — skipping triple "
              f"(PIP {pip_resid}, {res_a_resid}, {res_b_resid})")
        return None

    phos_atom = u.select_atoms(f"resid {pip_resid} and name {phosphate}")
    if len(phos_atom) == 0:
        print(f"  [WARN] {phosphate} atom not found on PIP {pip_resid}")

    n_frames = end_frame - start_frame + 1
    dist_a = np.full(n_frames, np.inf)
    dist_b = np.full(n_frames, np.inf)

    have_phos = len(phos_atom) > 0

    if have_phos:
        for i, ts in enumerate(u.trajectory[start_frame:end_frame + 1]):
            dist_a[i] = distance_array(phos_atom.positions, res_a_sel.positions, box=ts.dimensions).min()
            dist_b[i] = distance_array(phos_atom.positions, res_b_sel.positions, box=ts.dimensions).min()

    return dist_a, dist_b


def _compute_stats(dist_a: np.ndarray, dist_b: np.ndarray, cutoff: float,
                    label_a: str, label_b: str) -> dict:
    """
    Given (already-sliced, if desired) distance arrays, computes close
    frame counts/percentages for A, B, either, and the 4 mutually
    exclusive categories: <label_a>_only, <label_b>_only, Both, None.
    """
    n_frames = len(dist_a)

    close_a = dist_a <= cutoff
    close_b = dist_b <= cutoff
    close_either = close_a | close_b

    frames_a = int(close_a.sum())
    frames_b = int(close_b.sum())
    frames_either = int(close_either.sum())

    both_mask = close_a & close_b
    a_only_mask = close_a & ~close_b
    b_only_mask = close_b & ~close_a
    none_mask = ~close_a & ~close_b

    count_both = int(both_mask.sum())
    count_a_only = int(a_only_mask.sum())
    count_b_only = int(b_only_mask.sum())
    count_none = int(none_mask.sum())

    a_only_col = f"{label_a}_only"
    b_only_col = f"{label_b}_only"

    return {
        "n_frames": n_frames,
        f"frames_close_{label_a}": frames_a,
        f"pct_close_{label_a}": round(100.0 * frames_a / n_frames, 2) if n_frames else 0.0,
        f"frames_close_{label_b}": frames_b,
        f"pct_close_{label_b}": round(100.0 * frames_b / n_frames, 2) if n_frames else 0.0,
        "frames_close_either": frames_either,
        "pct_close_either": round(100.0 * frames_either / n_frames, 2) if n_frames else 0.0,
        f"count_{a_only_col}": count_a_only,
        f"count_{b_only_col}": count_b_only,
        "count_Both": count_both,
        "count_None": count_none,
        f"pct_{a_only_col}": round(100.0 * count_a_only / n_frames, 2) if n_frames else 0.0,
        f"pct_{b_only_col}": round(100.0 * count_b_only / n_frames, 2) if n_frames else 0.0,
        "pct_Both": round(100.0 * count_both / n_frames, 2) if n_frames else 0.0,
        "pct_None": round(100.0 * count_none / n_frames, 2) if n_frames else 0.0,
    }


# -----------------------------------------------------------------------------
# Raw per-frame distance caching (avoid recomputing across reruns)
# -----------------------------------------------------------------------------

def _cache_path(cache_dir: Path, pip_resid: int, res_a_resid: int, res_b_resid: int) -> Path:
    return cache_dir / f"pip{pip_resid}_resA{res_a_resid}_resB{res_b_resid}_distances.npz"


def _load_cached_distances(cache_path: Path, start_frame: int, end_frame: int
                            ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not cache_path.exists():
        return None

    data = np.load(cache_path)
    if int(data["start_frame"]) != start_frame or int(data["end_frame"]) != end_frame:
        print(f"  [CACHE] {cache_path.name} exists but covers a different frame range "
              f"({int(data['start_frame'])}-{int(data['end_frame'])}) — recomputing")
        return None

    print(f"  [CACHE] Loaded distances from {cache_path}")
    return data["dist_a"], data["dist_b"]


def _save_cached_distances(cache_path: Path, dist_a: np.ndarray, dist_b: np.ndarray,
                            start_frame: int, end_frame: int):
    np.savez(cache_path, dist_a=dist_a, dist_b=dist_b,
              start_frame=start_frame, end_frame=end_frame)
    print(f"  [CACHE] Saved distances -> {cache_path}")


def _get_or_compute_triple_distances(u, pip_resid: int, res_a_resid: int, res_b_resid: int,
                                      start_frame: int, end_frame: int,
                                      cache_dir: Path, phosphate: str = "P5", force: bool = False
                                      ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    cache_path = _cache_path(cache_dir, pip_resid, res_a_resid, res_b_resid)

    if not force:
        cached = _load_cached_distances(cache_path, start_frame, end_frame)
        if cached is not None:
            return cached

    dists = _get_triple_distances(u, pip_resid, res_a_resid, res_b_resid, start_frame, end_frame, phosphate=phosphate)
    if dists is None:
        return None

    dist_a, dist_b = dists
    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_cached_distances(cache_path, dist_a, dist_b, start_frame, end_frame)
    return dist_a, dist_b


# -----------------------------------------------------------------------------
# Gate mask loading (same convention as residue_pip_p4p5_proximity.py)
# -----------------------------------------------------------------------------

def _load_gate_mask(gate_csv, start_frame: int, end_frame: int,
                     threshold: float, column: str = "mean") -> Optional[np.ndarray]:
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

def _aa_label(resid: int, channel_type: str, aa_name: str, aa_overrides: Optional[Dict[int, str]]) -> str:
    """
    Builds a label like "LYS199.2b" — amino acid 3-letter code prefixed onto
    the to_pdb() converted residue label. aa_overrides lets specific raw
    resids use a different amino acid code (e.g. the G1 copy being ARG
    instead of LYS).
    """
    pdb_label = to_pdb(resid, channel_type)
    aa = (aa_overrides or {}).get(resid, aa_name)
    return f"{aa}{pdb_label}"


def _triple_label(pip_resid: int, res_a_resid: int, res_b_resid: int, channel_type: str,
                   res_a_aa: str = "LYS", res_b_aa: str = "LYS",
                   res_b_aa_overrides: Optional[Dict[int, str]] = None) -> str:
    label_a = _aa_label(res_a_resid, channel_type, res_a_aa, None)
    label_b = _aa_label(res_b_resid, channel_type, res_b_aa, res_b_aa_overrides)
    return f"PIP{pip_resid}\n{label_a}\n{label_b}"


def _make_category_barplot(df: pd.DataFrame, value_cols: List[str], ylabel: str,
                            title: str, out_path: Path, as_pct: bool, channel_type: str,
                            category_colors: Dict[str, str],
                            res_a_aa: str = "LYS", res_b_aa: str = "LYS",
                            res_b_aa_overrides: Optional[Dict[int, str]] = None):
    labels = [_triple_label(r.pip_resid, r.residue_a_resid, r.residue_b_resid, channel_type,
                             res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides)
              for r in df.itertuples()]
    n_triples = len(df)
    n_cats = len(value_cols)

    x = np.arange(n_triples)
    width = 0.8 / n_cats

    fig, ax = plt.subplots(figsize=(max(6, n_triples * 1.6 + 2), 5))

    for i, col in enumerate(value_cols):
        cat_name = col.replace("pct_", "").replace("count_", "")
        color = category_colors.get(cat_name, None)
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


def _write_category_txt(df: pd.DataFrame, out_path: Path, channel_type: str,
                         a_only_col: str, b_only_col: str, label_a: str, label_b: str,
                         phosphate: str,
                         res_a_aa: str = "LYS", res_b_aa: str = "LYS",
                         res_b_aa_overrides: Optional[Dict[int, str]] = None):
    with open(out_path, "w") as f:
        f.write(f"{phosphate} vs {label_a}/{label_b} proximity category counts — {channel_type}\n")
        f.write("=" * 60 + "\n\n")
        for r in df.itertuples():
            aa_a = _aa_label(r.residue_a_resid, channel_type, res_a_aa, None)
            aa_b = _aa_label(r.residue_b_resid, channel_type, res_b_aa, res_b_aa_overrides)
            f.write(f"PIP {r.pip_resid} {phosphate} <-> {aa_a} / {aa_b} "
                    f"(raw resids {r.residue_a_resid}/{r.residue_b_resid}) (n_frames={r.n_frames})\n")
            f.write(f"  {label_a} only : {getattr(r, f'count_{a_only_col}'):6d}  "
                    f"({getattr(r, f'pct_{a_only_col}'):5.1f}%)\n")
            f.write(f"  {label_b} only : {getattr(r, f'count_{b_only_col}'):6d}  "
                    f"({getattr(r, f'pct_{b_only_col}'):5.1f}%)\n")
            f.write(f"  Both      : {r.count_Both:6d}  ({r.pct_Both:5.1f}%)\n")
            f.write(f"  None      : {r.count_None:6d}  ({r.pct_None:5.1f}%)\n\n")


# -----------------------------------------------------------------------------
# Shared output writer
# -----------------------------------------------------------------------------

def _write_all_outputs(rows: List[dict], out_dir: Path, channel_type: str,
                        label_a: str, label_b: str, phosphate: str,
                        res_a_aa: str = "LYS", res_b_aa: str = "LYS",
                        res_b_aa_overrides: Optional[Dict[int, str]] = None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No valid triples — nothing written to {out_dir}.")
        return pd.DataFrame()

    a_only_col = f"{label_a}_only"
    b_only_col = f"{label_b}_only"
    category_colors = {
        a_only_col: "steelblue",
        b_only_col: "darkorange",
        "Both": "mediumseagreen",
        "None": "lightgray",
    }

    df = pd.DataFrame(rows)
    df["residue_a_pdb"] = df.apply(lambda r: to_pdb(r["residue_a_resid"], channel_type), axis=1)
    df["residue_b_pdb"] = df.apply(lambda r: to_pdb(r["residue_b_resid"], channel_type), axis=1)

    col_order = [
        "channel_type", "pip_resid", "residue_a_resid", "residue_a_pdb",
        "residue_b_resid", "residue_b_pdb", "n_frames",
        f"frames_close_{label_a}", f"pct_close_{label_a}",
        f"frames_close_{label_b}", f"pct_close_{label_b}",
        "frames_close_either", "pct_close_either",
    ]
    df_main = df[col_order]
    out_path = out_dir / "two_residue_proximity.csv"
    df_main.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    cat_cols = [
        "channel_type", "pip_resid", "residue_a_resid", "residue_a_pdb",
        "residue_b_resid", "residue_b_pdb", "n_frames",
        f"count_{a_only_col}", f"count_{b_only_col}", "count_Both", "count_None",
        f"pct_{a_only_col}", f"pct_{b_only_col}", "pct_Both", "pct_None",
    ]
    df_cat = df[cat_cols]
    cat_csv_path = out_dir / "category_counts.csv"
    df_cat.to_csv(cat_csv_path, index=False)
    print(f"Saved -> {cat_csv_path}")

    cat_txt_path = out_dir / "category_counts.txt"
    _write_category_txt(df_cat, cat_txt_path, channel_type, a_only_col, b_only_col, label_a, label_b,
                         phosphate=phosphate, res_a_aa=res_a_aa, res_b_aa=res_b_aa,
                         res_b_aa_overrides=res_b_aa_overrides)
    print(f"Saved -> {cat_txt_path}")

    # build a representative amino-acid-labeled title, e.g. "LYS199 / LYS64"
    title_a = f"{res_a_aa}{label_a}"
    title_b = f"{res_b_aa}{label_b}"

    counts_plot_path = out_dir / "barplot_counts.png"
    _make_category_barplot(
        df_cat,
        value_cols=[f"count_{a_only_col}", f"count_{b_only_col}", "count_Both", "count_None"],
        ylabel="Frame count",
        title=f"{channel_type} — Frames: {phosphate} close to {title_a} / {title_b} / Both / None",
        out_path=counts_plot_path,
        as_pct=False,
        channel_type=channel_type,
        category_colors=category_colors,
        res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides,
    )
    print(f"Saved -> {counts_plot_path}")

    pct_plot_path = out_dir / "barplot_pct.png"
    _make_category_barplot(
        df_cat,
        value_cols=[f"pct_{a_only_col}", f"pct_{b_only_col}", "pct_Both", "pct_None"],
        ylabel="% of frames",
        title=f"{channel_type} — % Frames: {phosphate} close to {title_a} / {title_b} / Both / None",
        out_path=pct_plot_path,
        as_pct=True,
        channel_type=channel_type,
        category_colors=category_colors,
        res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides,
    )
    print(f"Saved -> {pct_plot_path}")

    return df_main


# -----------------------------------------------------------------------------
# Public entry point (called from main.py)
# -----------------------------------------------------------------------------

def run_phosphate_two_residue_proximity(
    u,
    triples: List[Tuple[int, int, int]],
    results_dir,
    start_frame: int,
    end_frame: int,
    channel_type: str,
    phosphate: str = "P5",
    cutoff: float = DEFAULT_CUTOFF,
    res_a_label: str = "199",
    res_b_label: str = "64",
    res_a_aa: str = "LYS",
    res_b_aa: str = "LYS",
    res_b_aa_overrides: Optional[Dict[int, str]] = None,
    gate_filters: Optional[List[Dict[str, Any]]] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Computes, for each (pip_resid, residue_a_resid, residue_b_resid) triple,
    how many frames the PIP's chosen phosphate (P4 or P5) is close to
    residue A only, residue B only, both, or neither.

    triples: list of (pip_resid, residue_a_resid, residue_b_resid), e.g.
        [(1301, 471, 336), (1302, 146, 10), (1303, 1120, 985), (1304, 795, 660)]

    phosphate: "P4" or "P5" — selects which PIP2 phosphate atom to use.
    Output is written to a sub-folder named "{phosphate}_{res_b_label}_{res_a_label}", e.g.:
        <results_dir>/residue_pip_two_residues_proximity/P5_64_199/...
        <results_dir>/residue_pip_two_residues_proximity/P4_64_199/...
    so calling this once for P5 and once for P4 never overwrites anything.

    res_a_aa / res_b_aa: amino acid 3-letter code used in plot titles/txt
    (default "LYS" for both). res_b_aa_overrides lets specific raw resids
    use a different amino acid, e.g. {10: "ARG"} for the G1 copy.

    Raw per-triple distances are cached to
        <out_dir>/raw_distances/pip{pip}_resA{a}_resB{b}_distances.npz
    Pass force=True to ignore cache and recompute.

    gate_filters: same convention as residue_pip_p4p5_proximity.py — list of
    dicts with "name", "csv", "thresholds", optional "column". Each
    (gate, threshold) combo gets its own <out_dir>/<gate_name>_<threshold>/.
    """
    out_dir = Path(results_dir) / "residue_pip_two_residues_proximity" / f"{phosphate}_{res_b_label}_{res_a_label}"
    cache_dir = out_dir / "raw_distances"

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

    for pip_resid, res_a_resid, res_b_resid in triples:
        print(f"[{channel_type}] PIP {pip_resid} {phosphate} <-> residue {res_a_resid} ({res_a_label}) / "
              f"residue {res_b_resid} ({res_b_label}) (cutoff={cutoff} A)")
        dists = _get_or_compute_triple_distances(
            u, pip_resid, res_a_resid, res_b_resid, start_frame, end_frame, cache_dir,
            phosphate=phosphate, force=force,
        )
        if dists is None:
            continue
        dist_a, dist_b = dists

        stats_full = _compute_stats(dist_a, dist_b, cutoff, res_a_label, res_b_label)
        stats_full["pip_resid"] = pip_resid
        stats_full["residue_a_resid"] = res_a_resid
        stats_full["residue_b_resid"] = res_b_resid
        stats_full["channel_type"] = channel_type
        rows_full.append(stats_full)

        for (gate_name, threshold), mask in gate_masks.items():
            stats_gate = _compute_stats(dist_a[mask], dist_b[mask], cutoff, res_a_label, res_b_label)
            stats_gate["pip_resid"] = pip_resid
            stats_gate["residue_a_resid"] = res_a_resid
            stats_gate["residue_b_resid"] = res_b_resid
            stats_gate["channel_type"] = channel_type
            rows_by_gate[(gate_name, threshold)].append(stats_gate)

    df_full = _write_all_outputs(rows_full, out_dir, channel_type, res_a_label, res_b_label, phosphate,
                                  res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides)

    for (gate_name, threshold), rows in rows_by_gate.items():
        threshold_tag = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")
        gate_out_dir = out_dir / f"{gate_name}_{threshold_tag}"
        _write_all_outputs(rows, gate_out_dir, channel_type, res_a_label, res_b_label, phosphate,
                            res_a_aa=res_a_aa, res_b_aa=res_b_aa, res_b_aa_overrides=res_b_aa_overrides)

    return df_full