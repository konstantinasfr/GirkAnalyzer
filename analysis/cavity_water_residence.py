#!/usr/bin/env python3
"""
Cavity Water Residence Analysis
=================================
Tracks every water molecule that enters the GLU/ASN cavity across the
entire trajectory and measures how long each one stays inside.

A "residence event" = one contiguous block of frames a water spends inside.
We record every such event for every water, then compute statistics.

Key design decisions
---------------------
- Waters are identified by ATOM INDEX (not resid) to avoid any duplicate-resid
  issues that can occur with AMBER WAT topology numbering.
- The cylinder geometry is recomputed every frame so it follows protein motion.
- Memory: for large systems (n_waters × n_frames > 50M), a sparse approach is
  used automatically.

What it produces
-----------------
1. cavity_water_residence.pkl   — full results dict
2. cavity_water_residence.txt   — per-water summary table
3. cavity_water_residence.png   — two-panel plot:
     left : bar = mean ± std of per-water mean residence,
            dots = each water's mean residence time
     right: histogram of all individual residence event durations

How to call from main.py
-------------------------
    from analysis.cavity_water_residence import run_cavity_water_residence

    run_cavity_water_residence(
        u=u,
        channel2=ch2,
        results_dir=results_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        water_radius=12.0,
    )
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from tqdm import tqdm


_WATER_SEL = "resname WAT SOL TIP3 and name O OW"


# ── cylinder test (vectorised) ────────────────────────────────────────────────

def _in_cylinder(water_positions, center, axis, half_length, radius):
    """
    Returns bool array (N,): True for each water O inside the cylinder.
    """
    rel        = water_positions - center           # (N, 3)
    axial      = rel @ axis                         # (N,)
    radial_vec = rel - axial[:, np.newaxis] * axis  # (N, 3)
    radial     = np.linalg.norm(radial_vec, axis=1) # (N,)
    return (np.abs(axial) <= half_length) & (radial <= radius)


# ── contiguous block finder ───────────────────────────────────────────────────

def _find_durations(bool_array):
    """
    Given a boolean array, return a list of lengths of every contiguous
    True block. E.g. [F,T,T,T,F,T,F] → [3, 1].
    Uses numpy diff for speed instead of a Python loop.
    """
    if not np.any(bool_array):
        return []

    # pad with False on both ends so diff catches edges
    padded = np.concatenate([[False], bool_array, [False]])
    diff   = np.diff(padded.astype(int))
    starts = np.where(diff ==  1)[0]   # indices where block begins
    ends   = np.where(diff == -1)[0]   # indices where block ends (exclusive)
    return (ends - starts).tolist()


# ── main ──────────────────────────────────────────────────────────────────────

def run_cavity_water_residence(
    u,
    channel2,
    results_dir,
    start_frame=0,
    end_frame=None,
    water_selection=_WATER_SEL,
    water_radius=12.0,
):
    results_dir = Path(results_dir) / "cavity_water_residence"
    results_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    n_frames = end_frame - start_frame + 1

    print("\n" + "=" * 70)
    print("CAVITY WATER RESIDENCE ANALYSIS")
    print("=" * 70)
    print(f"  Water counting radius : {water_radius:.1f} Å")
    print(f"  Frames                : {start_frame} → {end_frame} ({n_frames} total)")

    # ── select water oxygens ──────────────────────────────────────────────────
    waters = u.select_atoms(water_selection)
    if len(waters) == 0:
        print(f"  [ERROR] No waters found: '{water_selection}'")
        print(f"  Available resnames: {sorted(set(u.atoms.resnames))}")
        return None

    n_waters = len(waters)
    print(f"  Total water O atoms   : {n_waters}")

    # atom index (unique, unlike resid) used as key throughout
    atom_indices = [int(a.index) for a in waters]
    atom_resids  = [int(a.resid)  for a in waters]

    # ── build presence matrix ─────────────────────────────────────────────────
    # presence[wi, fi] = True if water wi is inside cavity at frame fi
    # Use sparse if matrix would exceed 50M booleans (~50 MB)
    use_sparse = (n_waters * n_frames) > 50_000_000

    if not use_sparse:
        presence = np.zeros((n_waters, n_frames), dtype=bool)
    else:
        print("  [INFO] Large system — using sparse storage (list of frame indices per water)")
        # sparse_presence[wi] = sorted list of frame indices (0-based) where wi is inside
        sparse_presence = [[] for _ in range(n_waters)]

    print(f"\n  Scanning {n_frames} frames ...")
    with tqdm(total=n_frames, desc="  Cavity residence",
              unit="frame", dynamic_ncols=True) as pbar:

        for fi, ts in enumerate(u.trajectory[start_frame: end_frame + 1]):
            # recompute cylinder geometry from current CA positions
            channel2.compute_geometry(gate_num=2)

            inside = _in_cylinder(
                waters.positions,
                channel2.channel_center,
                channel2.channel_axis,
                channel2.channel_length / 2.0,
                water_radius,
            )

            if not use_sparse:
                presence[:, fi] = inside
            else:
                for wi in np.where(inside)[0]:
                    sparse_presence[wi].append(fi)

            pbar.update(1)

    # ── compute residence events ──────────────────────────────────────────────
    print("\n  Computing residence events ...")

    # keyed by atom_index to avoid any resid collision
    # value: list of residence durations for that water
    residences_by_idx = {}   # {atom_index: [dur1, dur2, ...]}

    for wi in range(n_waters):
        if not use_sparse:
            bool_arr = presence[wi, :]
        else:
            # reconstruct boolean array from sparse list
            bool_arr = np.zeros(n_frames, dtype=bool)
            for fi in sparse_presence[wi]:
                bool_arr[fi] = True

        durations = _find_durations(bool_arr)

        if durations:   # only store waters that actually entered
            residences_by_idx[atom_indices[wi]] = durations

    n_visited = len(residences_by_idx)
    print(f"  Waters that visited cavity : {n_visited}")

    if n_visited == 0:
        print("  [WARN] No waters found inside cavity — check water_radius "
              "or cylinder geometry.")
        return None

    # ── statistics ────────────────────────────────────────────────────────────

    # per-water mean (mean residence duration across all visits by that water)
    per_water_mean    = {idx: float(np.mean(durs))
                         for idx, durs in residences_by_idx.items()}
    per_water_nvisits = {idx: len(durs)
                         for idx, durs in residences_by_idx.items()}
    per_water_total   = {idx: int(sum(durs))
                         for idx, durs in residences_by_idx.items()}

    # map atom_index back to resid for reporting
    idx_to_resid = dict(zip(atom_indices, atom_resids))

    # all individual residence events flattened — for histogram
    all_durations_flat = [d for durs in residences_by_idx.values() for d in durs]

    # statistics of per-water means — used for bar + dot plot
    pw_means      = np.array(list(per_water_mean.values()))
    pw_mean_mean  = float(np.mean(pw_means))
    pw_mean_std   = float(np.std(pw_means, ddof=1)) if len(pw_means) > 1 else 0.0

    # statistics of all individual events — used for histogram
    ev_mean   = float(np.mean(all_durations_flat))
    ev_std    = float(np.std(all_durations_flat, ddof=1)) if len(all_durations_flat) > 1 else 0.0
    ev_median = float(np.median(all_durations_flat))
    ev_max    = int(max(all_durations_flat))

    print(f"  Total residence events          : {len(all_durations_flat)}")
    print(f"  Per-water mean  (mean ± std)    : {pw_mean_mean:.2f} ± {pw_mean_std:.2f} frames")
    print(f"  Individual event mean ± std     : {ev_mean:.2f} ± {ev_std:.2f} frames")
    print(f"  Individual event median         : {ev_median:.2f} frames")
    print(f"  Longest single residence        : {ev_max} frames")

    # ── save pkl ──────────────────────────────────────────────────────────────
    results = {
        "water_radius"        : water_radius,
        "start_frame"         : start_frame,
        "end_frame"           : end_frame,
        "n_frames"            : n_frames,
        "n_waters_visited"    : n_visited,
        # keyed by atom index
        "residences_by_idx"   : residences_by_idx,
        "per_water_mean"      : per_water_mean,
        "per_water_nvisits"   : per_water_nvisits,
        "per_water_total"     : per_water_total,
        "idx_to_resid"        : idx_to_resid,
        # flattened events
        "all_durations_flat"  : all_durations_flat,
        # per-water mean stats (used for bar+dot plot)
        "pw_mean_mean"        : pw_mean_mean,
        "pw_mean_std"         : pw_mean_std,
        # individual event stats (used for histogram)
        "ev_mean"             : ev_mean,
        "ev_std"              : ev_std,
        "ev_median"           : ev_median,
        "ev_max"              : ev_max,
    }

    pkl_path = results_dir / "cavity_water_residence.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  ✅ Saved: {pkl_path}")

    # ── save txt ──────────────────────────────────────────────────────────────
    txt_path = results_dir / "cavity_water_residence.txt"
    with open(txt_path, "w") as f:
        f.write("CAVITY WATER RESIDENCE ANALYSIS\n")
        f.write(f"Water radius         : {water_radius:.1f} Å\n")
        f.write(f"Frames               : {start_frame} → {end_frame}\n")
        f.write(f"Waters visited       : {n_visited}\n")
        f.write(f"Total events         : {len(all_durations_flat)}\n")
        f.write(f"Per-water mean       : {pw_mean_mean:.2f} ± {pw_mean_std:.2f} frames\n")
        f.write(f"Event mean           : {ev_mean:.2f} ± {ev_std:.2f} frames\n")
        f.write(f"Event median         : {ev_median:.2f} frames\n")
        f.write(f"Longest residence    : {ev_max} frames\n")
        f.write("=" * 70 + "\n\n")
        f.write("PER WATER SUMMARY (sorted by mean residence, descending)\n")
        f.write(f"{'AtomIdx':>8}  {'ResID':>7}  {'N visits':>9}  "
                f"{'Mean dur (fr)':>14}  {'Total (fr)':>11}  Durations\n")
        f.write("-" * 80 + "\n")
        for idx in sorted(per_water_mean, key=per_water_mean.get, reverse=True):
            durs  = residences_by_idx[idx]
            resid = idx_to_resid.get(idx, "?")
            f.write(
                f"{idx:>8}  {resid:>7}  {len(durs):>9}  "
                f"{np.mean(durs):>14.2f}  {sum(durs):>11}  {durs}\n"
            )
    print(f"  ✅ Saved: {txt_path}")

    # ── plot ──────────────────────────────────────────────────────────────────
    _plot_residence(results, results_dir)

    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_residence(results, out_dir):
    """
    Left panel  : bar = mean ± std of PER-WATER means, dots = each water
    Right panel : histogram of ALL individual residence event durations
    Both stats are clearly labelled so there is no confusion between them.
    """
    pw_means     = np.array(list(results["per_water_mean"].values()))
    pw_mean_mean = results["pw_mean_mean"]
    pw_mean_std  = results["pw_mean_std"]
    n_visited    = results["n_waters_visited"]
    water_radius = results["water_radius"]
    all_durs     = results["all_durations_flat"]
    ev_mean      = results["ev_mean"]
    ev_median    = results["ev_median"]
    ev_max       = results["ev_max"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle(
        f"Water residence times in GLU/ASN cavity  (r = {water_radius:.1f} Å)\n"
        f"{n_visited} unique waters  |  "
        f"per-water mean = {pw_mean_mean:.1f} ± {pw_mean_std:.1f} frames  |  "
        f"{len(all_durs)} total events",
        fontsize=12, fontweight="bold"
    )

    # ── LEFT: bar + dots, both using per-water means ──────────────────────────
    ax = axes[0]
    ax.bar([0], [pw_mean_mean], yerr=[pw_mean_std],
           color="#2196F3", width=0.45, capsize=10,
           error_kw={"linewidth": 2.0, "ecolor": "#0D47A1"},
           edgecolor="#0D47A1", linewidth=0.8,
           zorder=2,
           label=f"Mean ± std\n({pw_mean_mean:.1f} ± {pw_mean_std:.1f} fr)")

    np.random.seed(42)
    jitter = np.random.uniform(-0.18, 0.18, size=len(pw_means))
    ax.scatter(jitter, pw_means,
               color="#E53935", s=22, alpha=0.55, zorder=5,
               label=f"Each water (n={n_visited})")

    ax.set_xticks([0])
    ax.set_xticklabels(["GLU/ASN cavity"], fontsize=10)
    ax.set_ylabel("Mean residence time per water (frames)", fontsize=11)
    ax.set_xlim(-0.5, 0.5)
    y_top = max(pw_means.max() * 1.15, pw_mean_mean + pw_mean_std * 2.5, 2)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_title("Per-water mean residence time\n"
                 "(one dot = one unique water molecule)", fontsize=10)

    # ── RIGHT: histogram of individual event durations ────────────────────────
    ax2 = axes[1]

    # cutoff: show up to mean + 6*std, cap at max
    cutoff = min(ev_max, max(int(ev_mean + 6 * results["ev_std"]) + 1, 10))

    # bins centred on integers: edges at 0.5, 1.5, 2.5, ...
    bin_edges = np.arange(0.5, cutoff + 1.5, 1.0)   # length cutoff + 1
    vals_trunc = [d for d in all_durs if d <= cutoff]
    n_truncated = len(all_durs) - len(vals_trunc)

    counts, _ = np.histogram(vals_trunc, bins=bin_edges)
    bar_x      = np.arange(1, cutoff + 1)             # length cutoff — matches counts

    ax2.bar(bar_x, counts,
            color="#2196F3", edgecolor="#0D47A1",
            linewidth=0.5, width=0.8)

    # count labels on top of bars when not too many bars
    if cutoff <= 40:
        for x, c in zip(bar_x, counts):
            if c > 0:
                ax2.text(x, c + max(counts) * 0.01, str(c),
                         ha="center", va="bottom", fontsize=7)

    ax2.axvline(ev_mean, color="#E53935", linewidth=2.0,
                linestyle="--", label=f"Mean = {ev_mean:.1f} fr")
    ax2.axvline(ev_median, color="#FF9800", linewidth=1.5,
                linestyle=":", label=f"Median = {ev_median:.1f} fr")

    ax2.set_xticks(bar_x[::max(1, len(bar_x) // 20)])  # avoid crowded x labels
    ax2.set_xlabel(
        "Residence duration (frames)"
        + (f"   [+{n_truncated} events > {cutoff} fr not shown]"
           if n_truncated > 0 else ""),
        fontsize=10
    )
    ax2.set_ylabel("Number of residence events", fontsize=11)
    ax2.set_title(
        f"Distribution of individual residence event durations\n"
        f"(n = {len(all_durs)} events from {n_visited} waters)",
        fontsize=10
    )
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out_path = out_dir / "cavity_water_residence.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")