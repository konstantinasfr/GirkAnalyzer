"""
glu_asn_geometry_analysis.py
=============================
Tracks the geometry of the GLU and ASN residue rings in the cavity over
every frame of the simulation.

Metrics computed per frame
--------------------------
GLU pair distances (Cα–Cα):
    dist_glu_pair1   distance of pair 1 (e.g. resid 99–1074)
    dist_glu_pair2   distance of pair 2 (e.g. resid 424–749)
    dist_glu_mean    mean of the two GLU pair distances  (ring diameter proxy)

ASN pair distances (Cα–Cα):
    dist_asn_pair1   distance of pair 1
    dist_asn_pair2   distance of pair 2
    dist_asn_mean    mean of the two ASN pair distances  (ring diameter proxy)

COM distances:
    dist_glu_com_to_asn_com   3D distance between GLU COM and ASN COM
    dist_glu_asn_axial        axial component (along channel axis, GLU→ASN)

Cavity size (cylinder approximation):
    cavity_radius     mean of GLU and ASN mean diameters / 2
    cavity_height     axial distance between GLU COM and ASN COM
    cavity_volume_A3  π × radius² × height  (Å³)

Outputs (saved to results_dir/glu_asn_geometry/):
    geometry_per_frame.csv      one row per frame, all metrics
    geometry_summary.json       mean ± std / min / max of all metrics
    dist_timeseries.png         time-series of all pair distances + mean
    cavity_timeseries.png       time-series of cavity radius, height, volume
    dist_boxplot.png            boxplot of all distance metrics
    cavity_boxplot.png          boxplot of cavity size metrics

Called from main.py:
    from analysis.glu_asn_geometry_analysis import run_glu_asn_geometry_analysis
    run_glu_asn_geometry_analysis(
        u                   = u,
        glu_residues        = glu_residues,
        glu_residues_pairs  = glu_residues_pairs,
        asn_residues        = asn_residues,
        asn_residues_pairs  = asn_residues_pairs,
        sf_low_res_residues = sf_low_res_residues,
        hbc_residues        = hbc_residues,
        results_dir         = results_dir,
        channel_type        = args.channel_type,
        start_frame         = start_frame,
        end_frame           = end_frame,
        frame_duration      = 0.075,
    )
"""

from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

FS_TITLE  = 20
FS_LABEL  = 18
FS_TICK   = 16
FS_LEGEND = 15


# ── geometry helpers ──────────────────────────────────────────────────────────

def _flatten(resids):
    """Flatten a list that may contain ints or (int, int) tuples."""
    flat = []
    for item in resids:
        if hasattr(item, '__iter__') and not isinstance(item, str):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _ca_pos(u, resid):
    """Cα position of a residue."""
    sel = u.select_atoms(f"resid {resid} and name CA")
    if sel.n_atoms == 0:
        sel = u.select_atoms(f"resid {resid}")   # fallback to any atom
    return sel.positions[0].copy()


def _com(u, resids):
    """COM of a group of residues (all heavy atoms).
    Accepts a flat list of resids OR a list of (resid1, resid2) pairs.
    """
    flat = _flatten(resids)
    sel = u.select_atoms(" or ".join(f"resid {r}" for r in flat))
    return sel.center_of_mass()


def _ca_dist(u, resid1, resid2):
    """Cα–Cα distance between two residues."""
    return float(np.linalg.norm(_ca_pos(u, resid1) - _ca_pos(u, resid2)))


def _make_axis(u, sf_low_res_residues, hbc_residues):
    """Unit vector from SF toward HBC (same convention as other analyses)."""
    indices = []
    for resid in sf_low_res_residues:
        sel = u.select_atoms(f"resid {resid}")
        if sel.n_atoms == 0:
            continue
        indices.append(sel[int(np.argmin(sel.positions[:, 2]))].index)
    sf_origin = u.atoms[indices].center_of_mass() if indices else \
                _com(u, sf_low_res_residues)
    hbc_end   = _com(u, hbc_residues)
    vec = hbc_end - sf_origin
    return sf_origin, vec / np.linalg.norm(vec)


# ── per-frame computation ─────────────────────────────────────────────────────

def _compute_frame(u, glu_residues, glu_pairs,
                   asn_residues, asn_pairs,
                   sf_low_res_residues, hbc_residues):
    """Compute all geometry metrics for the current trajectory frame."""

    # GLU pair distances
    d_glu = [_ca_dist(u, r1, r2) for r1, r2 in glu_pairs]
    d_glu_mean = float(np.mean(d_glu))

    # ASN pair distances
    d_asn = [_ca_dist(u, r1, r2) for r1, r2 in asn_pairs]
    d_asn_mean = float(np.mean(d_asn))

    # COMs
    glu_com = _com(u, glu_residues)
    asn_com = _com(u, asn_residues)

    # 3D GLU–ASN COM distance
    vec_glu_to_asn  = asn_com - glu_com
    dist_3d         = float(np.linalg.norm(vec_glu_to_asn))

    # axial component (GLU→ASN along channel axis)
    _, unit = _make_axis(u, sf_low_res_residues, hbc_residues)
    axial   = float(np.dot(vec_glu_to_asn, unit))

    # cavity size (cylinder approximation)
    cavity_radius = (d_glu_mean + d_asn_mean) / 4.0   # mean diameter / 2
    cavity_height = abs(axial)
    cavity_volume = np.pi * cavity_radius**2 * cavity_height

    return {
        "dist_glu_pair1":         round(d_glu[0], 3),
        "dist_glu_pair2":         round(d_glu[1], 3) if len(d_glu) > 1 else None,
        "dist_glu_mean":          round(d_glu_mean, 3),
        "dist_asn_pair1":         round(d_asn[0], 3),
        "dist_asn_pair2":         round(d_asn[1], 3) if len(d_asn) > 1 else None,
        "dist_asn_mean":          round(d_asn_mean, 3),
        "dist_glu_com_to_asn_com": round(dist_3d, 3),
        "dist_glu_asn_axial":     round(axial, 3),
        "cavity_radius":          round(cavity_radius, 3),
        "cavity_height":          round(cavity_height, 3),
        "cavity_volume_A3":       round(cavity_volume, 3),
    }


# ── CSV / JSON saving ─────────────────────────────────────────────────────────

FIELDS = [
    "frame", "t_ns",
    "dist_glu_pair1", "dist_glu_pair2", "dist_glu_mean",
    "dist_asn_pair1", "dist_asn_pair2", "dist_asn_mean",
    "dist_glu_com_to_asn_com", "dist_glu_asn_axial",
    "cavity_radius", "cavity_height", "cavity_volume_A3",
]

NUMERIC_FIELDS = FIELDS[2:]   # everything except frame, t_ns


def _save_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"    Saved: {out_path}")


def _save_summary(rows, label, out_path):
    summary = {"system": label, "n_frames": len(rows)}
    for col in NUMERIC_FIELDS:
        vals = [r[col] for r in rows if r.get(col) is not None]
        if vals:
            summary[col] = {
                "mean":   round(float(np.mean(vals)), 3),
                "std":    round(float(np.std(vals)),  3),
                "min":    round(float(np.min(vals)),  3),
                "max":    round(float(np.max(vals)),  3),
                "median": round(float(np.median(vals)), 3),
            }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved: {out_path}")


# ── plots ─────────────────────────────────────────────────────────────────────

def _time_series_plot(rows, glu_pairs, asn_pairs, label,
                      frame_duration, out_dir):
    """Two-panel time-series: (1) pair distances, (2) cavity metrics."""
    t   = np.array([r["t_ns"] for r in rows])

    # ── panel 1: distances ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    glu_colors = ["#E74C3C", "#C0392B"]
    asn_colors = ["#2980B9", "#1A5276"]

    for i, (r1, r2) in enumerate(glu_pairs):
        col = f"dist_glu_pair{i+1}"
        ax.plot(t, [r[col] for r in rows], color=glu_colors[i],
                linewidth=0.8, alpha=0.6,
                label=f"GLU pair {i+1}: resid {r1}–{r2}")
    ax.plot(t, [r["dist_glu_mean"] for r in rows],
            color="#E74C3C", linewidth=2.0, linestyle="--",
            label="GLU mean diameter")

    for i, (r1, r2) in enumerate(asn_pairs):
        col = f"dist_asn_pair{i+1}"
        ax.plot(t, [r[col] for r in rows], color=asn_colors[i],
                linewidth=0.8, alpha=0.6,
                label=f"ASN pair {i+1}: resid {r1}–{r2}")
    ax.plot(t, [r["dist_asn_mean"] for r in rows],
            color="#2980B9", linewidth=2.0, linestyle="--",
            label="ASN mean diameter")

    ax.set_ylabel("Cα–Cα distance (Å)", fontsize=FS_LABEL)
    ax.set_title(f"GLU and ASN ring diameters — {label}", fontsize=FS_TITLE, pad=12)
    ax.legend(fontsize=FS_LEGEND, ncol=2)
    ax.tick_params(labelsize=FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)

    # GLU–ASN COM distance
    ax2 = axes[1]
    ax2.plot(t, [r["dist_glu_com_to_asn_com"] for r in rows],
             color="#8E44AD", linewidth=1.0, alpha=0.7,
             label="GLU–ASN COM distance (3D)")
    ax2.plot(t, [r["dist_glu_asn_axial"] for r in rows],
             color="#D35400", linewidth=1.0, alpha=0.7,
             label="GLU–ASN axial distance")

    ax2.set_xlabel("Time (ns)", fontsize=FS_LABEL)
    ax2.set_ylabel("Distance (Å)", fontsize=FS_LABEL)
    ax2.set_title(f"GLU–ASN COM distances — {label}", fontsize=FS_TITLE, pad=12)
    ax2.legend(fontsize=FS_LEGEND)
    ax2.tick_params(labelsize=FS_TICK)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = out_dir / "dist_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")

    # ── panel: cavity metrics ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    for ax, col, ylabel, color, title_suffix in [
        (axes[0], "cavity_radius",    "Radius (Å)",       "#27AE60",
         "Cavity radius (mean ring diameter / 2)"),
        (axes[1], "cavity_height",    "Height (Å)",       "#8E44AD",
         "Cavity height (GLU–ASN axial distance)"),
        (axes[2], "cavity_volume_A3", "Volume (Å³)",      "#2980B9",
         "Cavity volume (cylinder: π·r²·h)"),
    ]:
        vals = np.array([r[col] for r in rows])
        ax.fill_between(t, vals, alpha=0.20, color=color)
        ax.plot(t, vals, color=color, linewidth=0.9)
        ax.axhline(np.mean(vals), color=color, linewidth=1.4,
                   linestyle="--", alpha=0.8,
                   label=f"mean = {np.mean(vals):.2f}")
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_title(f"{title_suffix} — {label}", fontsize=FS_TITLE, pad=10)
        ax.legend(fontsize=FS_LEGEND)
        ax.tick_params(labelsize=FS_TICK)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Time (ns)", fontsize=FS_LABEL)
    plt.tight_layout()
    out = out_dir / "cavity_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")


def _boxplot(rows, metrics, names, label, ylabel, title, out_path, color):
    """Boxplot with mean and median in x-tick labels."""
    all_vals = [[r[m] for r in rows if r.get(m) is not None] for m in metrics]

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 2.5), 7))
    ax.boxplot(
        all_vals,
        positions=range(len(names)),
        widths=0.50,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=4, alpha=0.35,
                        markerfacecolor=color, markeredgecolor="none"),
        medianprops=dict(color="black", linewidth=2.5),
        boxprops=dict(facecolor=color, alpha=0.65, linewidth=1.2),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
    )

    tick_labels = []
    for name, v in zip(names, all_vals):
        if v:
            tick_labels.append(
                f"{name}\nmean={np.mean(v):.2f}\nmedian={np.median(v):.2f}"
            )
        else:
            tick_labels.append(name)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(tick_labels, fontsize=FS_TICK - 1)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(f"{title}\n{label}  (n = {len(rows)} frames)",
                 fontsize=FS_TITLE, pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ── public entry point ────────────────────────────────────────────────────────

def run_glu_asn_geometry_analysis(
    u,
    glu_residues:        list,
    glu_residues_pairs:  list,
    asn_residues:        list,
    asn_residues_pairs:  list,
    sf_low_res_residues: list,
    hbc_residues:        list,
    results_dir,
    channel_type:   str   = "G12",
    start_frame:    int   = 0,
    end_frame:      int   = None,
    frame_duration: float = 0.075,
):
    """
    Main entry point — call from main.py.

    Parameters
    ----------
    u                   : MDAnalysis Universe
    glu_residues        : list of 4 GLU resids
    glu_residues_pairs  : list of (resid1, resid2) tuples for diagonal pairs
    asn_residues        : list of 4 ASN resids
    asn_residues_pairs  : list of (resid1, resid2) tuples for diagonal pairs
    sf_low_res_residues : SF bottom residues (axis origin)
    hbc_residues        : HBC residues (axis end)
    results_dir         : Path to RUN-level results directory
    channel_type        : label for plots
    start_frame         : first frame (default 0)
    end_frame           : last frame inclusive (default last)
    frame_duration      : ns per frame (default 0.075)
    """
    results_dir = Path(results_dir)
    out_dir     = results_dir / "glu_asn_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = channel_type

    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    # skip if already computed
    out_csv = out_dir / "geometry_per_frame.csv"
    if out_csv.exists():
        print(f"\n  glu_asn_geometry: already exists — skipping. "
              f"Delete {out_csv} to recompute.")
        return

    print(f"\n  GLU/ASN geometry analysis ({label})")
    print(f"  Frames         : {start_frame} – {end_frame}")
    print(f"  GLU residues   : {_flatten(glu_residues)}")
    print(f"  GLU pairs      : {glu_residues_pairs}")
    print(f"  ASN residues   : {_flatten(asn_residues)}")
    print(f"  ASN pairs      : {asn_residues_pairs}")
    print(f"  Output dir     : {out_dir}")

    rows = []
    for ts in tqdm(u.trajectory[start_frame:end_frame + 1],
                   total=end_frame - start_frame + 1,
                   desc="  computing geometry"):
        frame   = ts.frame
        metrics = _compute_frame(u, glu_residues, glu_residues_pairs,
                                 asn_residues, asn_residues_pairs,
                                 sf_low_res_residues, hbc_residues)
        row = {"frame": frame,
               "t_ns":  round(frame * frame_duration, 4)}
        row.update(metrics)
        rows.append(row)

    print(f"  Computed {len(rows)} frames")

    # ── save files ────────────────────────────────────────────────────────
    _save_csv(rows, out_csv)
    _save_summary(rows, label, out_dir / "geometry_summary.json")

    # ── time-series plots ─────────────────────────────────────────────────
    _time_series_plot(rows, glu_residues_pairs, asn_residues_pairs,
                      label, frame_duration, out_dir)

    # ── distance boxplots ─────────────────────────────────────────────────
    dist_metrics = [
        "dist_glu_pair1", "dist_glu_pair2", "dist_glu_mean",
        "dist_asn_pair1", "dist_asn_pair2", "dist_asn_mean",
        "dist_glu_com_to_asn_com", "dist_glu_asn_axial",
    ]
    dist_names = [
        f"GLU\npair1\n{glu_residues_pairs[0][0]}–{glu_residues_pairs[0][1]}",
        f"GLU\npair2\n{glu_residues_pairs[1][0]}–{glu_residues_pairs[1][1]}",
        "GLU\nmean\ndiameter",
        f"ASN\npair1\n{asn_residues_pairs[0][0]}–{asn_residues_pairs[0][1]}",
        f"ASN\npair2\n{asn_residues_pairs[1][0]}–{asn_residues_pairs[1][1]}",
        "ASN\nmean\ndiameter",
        "GLU–ASN\nCOM\n(3D)",
        "GLU–ASN\naxial",
    ]
    _boxplot(rows, dist_metrics, dist_names, label,
             "Distance (Å)",
             "GLU and ASN residue pair distances over full trajectory",
             out_dir / "dist_boxplot.png", color="#E74C3C")

    # ── cavity size boxplots ──────────────────────────────────────────────
    cav_metrics = ["cavity_radius", "cavity_height", "cavity_volume_A3"]
    cav_names   = ["Cavity\nradius (Å)", "Cavity\nheight (Å)", "Cavity\nvolume (Å³)"]
    _boxplot(rows, cav_metrics, cav_names, label,
             "Value",
             "GLU/ASN cavity size over full trajectory\n"
             "(cylinder approximation: radius = mean ring diameter / 2)",
             out_dir / "cavity_boxplot.png", color="#27AE60")

    # ── terminal summary ──────────────────────────────────────────────────
    print(f"\n  SUMMARY ({len(rows)} frames):")
    for col in dist_metrics + cav_metrics:
        vals = [r[col] for r in rows if r.get(col) is not None]
        if vals:
            print(f"    {col:<32}  {np.mean(vals):7.2f} ± {np.std(vals):.2f}  "
                  f"[{np.min(vals):.2f} – {np.max(vals):.2f}]")

    print(f"\n  Done. All outputs in: {out_dir}")