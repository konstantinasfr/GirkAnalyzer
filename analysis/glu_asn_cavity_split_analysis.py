"""
cavity_split_analysis.py

Splits the GLU/ASN cavity into an UPPER and a LOWER half and counts K+ ions
in each, per frame.

Geometry (recomputed every frame, so it follows the protein):
    z_glu  = z of the COM of the four GLU ring residues
    z_asn  = z of the COM of the four ASN/ASP ring residues
    z_mid  = (z_glu + z_asn) / 2          <- the cut

    UPPER half = z between z_mid and z_glu
    LOWER half = z between z_asn and z_mid

Two radial definitions are counted independently:
    CAVITY   — cylinder of radius `cavity_radius` (default 12 A) about the
               channel axis (the x,y of the GLU/ASN COM midpoint)
    CYLINDER — narrow cylinder of radius `cylinder_radius` (default 3 A),
               same axis, same z limits — i.e. the on-axis core only

Outputs (saved to results_dir/cavity_split/):
    per_frame_counts.csv      — one row per frame, both radii, upper/lower/total
    end2_counts.csv           — one row per permeation event, counted at its
                                end_2 frame, WITH and WITHOUT the permeating ion
    end2_summary.csv          — means/medians of the end_2 columns
    summary.txt               — geometry + headline numbers
    plots/
        per_frame_counts.png      — occupancy vs frame (cavity and cylinder)
        end2_distributions.png    — histograms of the end_2 counts

Call from main.py:
    from analysis.cavity_split_analysis import run_cavity_split_analysis

    run_cavity_split_analysis(
        u                 = u,
        results_dir       = results_dir,
        channel_type      = args.channel_type,
        permeation_events = analyzer.permeation_events,
        frame_duration    = 0.075,
        force_rerun       = False,
    )

If `permeation_events` is omitted the module reads permeation_table.json from
results_dir (or its parents), so it also works as a post-hoc standalone run.
"""

import csv
import json
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:                                    # tqdm optional
    def tqdm(x, **kwargs):
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Residue definitions per channel type (simulation resids, 4 subunits each)
# ─────────────────────────────────────────────────────────────────────────────

RING_RESIDUES = {
    "G12":     {"glu": [99, 424, 748, 1073], "asn": [131, 456, 780, 1105]},
    "G12_ML":  {"glu": [99, 424, 749, 1074], "asn": [131, 456, 781, 1106]},
    "G12_GAT": {"glu": [98, 422, 747, 1073], "asn": [130, 454, 779, 1105]},
    "G2":      {"glu": [98, 426, 754, 1082], "asn": [130, 458, 786, 1114]},
}

ALIASES = {"ML": "G12_ML", "GAT": "G12_GAT", "G12_WT": "G12", "G2_WT": "G2"}


def resolve_rings(channel_type, glu=None, asn=None):
    """Returns (glu_resids, asn_resids), with explicit overrides winning."""
    if glu is not None and asn is not None:
        return list(glu), list(asn)
    ct = ALIASES.get(str(channel_type).upper().strip(),
                     str(channel_type).upper().strip())
    if ct not in RING_RESIDUES:
        raise ValueError(
            f"Unknown channel_type '{channel_type}'. Known: "
            f"{', '.join(sorted(RING_RESIDUES))}. Pass glu=/asn= explicitly."
        )
    return list(RING_RESIDUES[ct]["glu"]), list(RING_RESIDUES[ct]["asn"])


# ─────────────────────────────────────────────────────────────────────────────
# Pure geometry — no MDAnalysis, so it is unit-testable on plain arrays
# ─────────────────────────────────────────────────────────────────────────────

def count_split(ion_pos, axis_xy, z_glu, z_asn, radius, exclude_index=None):
    """
    Counts ions inside a cylinder of `radius` about `axis_xy`, spanning the z
    range between the GLU and ASN rings, split at the midpoint z.

    ion_pos       (N, 3) array of ion coordinates
    axis_xy       (2,)   x,y of the channel axis
    exclude_index index into ion_pos to leave out (the permeating ion), or None

    Returns dict(total, upper, lower).
    """
    ion_pos = np.asarray(ion_pos, dtype=float)
    if ion_pos.size == 0:
        return {"total": 0, "upper": 0, "lower": 0}

    keep = np.ones(len(ion_pos), dtype=bool)
    if exclude_index is not None and 0 <= exclude_index < len(ion_pos):
        keep[exclude_index] = False
    pos = ion_pos[keep]
    if len(pos) == 0:
        return {"total": 0, "upper": 0, "lower": 0}

    z_lo, z_hi = (z_asn, z_glu) if z_asn <= z_glu else (z_glu, z_asn)
    z_mid = 0.5 * (z_glu + z_asn)

    r = np.hypot(pos[:, 0] - axis_xy[0], pos[:, 1] - axis_xy[1])
    z = pos[:, 2]

    in_cyl = (r <= radius) & (z >= z_lo) & (z <= z_hi)
    upper = int(np.count_nonzero(in_cyl & (z >= z_mid)))
    lower = int(np.count_nonzero(in_cyl & (z < z_mid)))
    return {"total": upper + lower, "upper": upper, "lower": lower}


def classify_ion(pos, axis_xy, z_glu, z_asn, radius):
    """Where a single ion sits: 'upper', 'lower' or 'outside' that cylinder."""
    z_lo, z_hi = (z_asn, z_glu) if z_asn <= z_glu else (z_glu, z_asn)
    z_mid = 0.5 * (z_glu + z_asn)
    r = float(np.hypot(pos[0] - axis_xy[0], pos[1] - axis_xy[1]))
    if r > radius or pos[2] < z_lo or pos[2] > z_hi:
        return "outside"
    return "upper" if pos[2] >= z_mid else "lower"


# ─────────────────────────────────────────────────────────────────────────────
# Permeation events / end_2 frames
# ─────────────────────────────────────────────────────────────────────────────

def _load_permeation_events(results_dir):
    """Finds permeation_table.json in results_dir or its parents."""
    results_dir = Path(results_dir)
    for d in [results_dir, results_dir.parent, results_dir.parent.parent]:
        f = d / "permeation_table.json"
        if f.is_file():
            with open(f) as fh:
                return json.load(fh)
    return []


def _get_end2(event):
    """end_2 under any of the key spellings used across the pipeline."""
    for key in ("end_2", "end2", "R2_end", "r2_end"):
        if key in event and event[key] is not None:
            try:
                return int(event[key])
            except (TypeError, ValueError):
                continue
    return None


def _get_ion_id(event):
    for key in ("ion_id", "ion", "id"):
        if key in event and event[key] is not None:
            try:
                return int(event[key])
            except (TypeError, ValueError):
                return event[key]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

MEAN_KW   = dict(color='red',   linestyle='--',  linewidth=1.3)
MEDIAN_KW = dict(color='green', linestyle='-.',  linewidth=1.3)
C_UPPER, C_LOWER, C_TOTAL = '#C44E52', '#4C72B0', '#2E4057'


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, plots_dir, name):
    out = Path(plots_dir) / name
    fig.savefig(out, dpi=150, bbox_inches='tight')
    _mpl().close(fig)
    print(f"  Plot saved: {name}")


def _rolling(y, w):
    if w <= 1 or len(y) < w:
        return None
    k = np.ones(w) / w
    return np.convolve(y, k, mode='same')


# ---- 1. time series: upper and lower in their own panels -------------------

def _plot_timeseries(rows, plots_dir, tag, label, radius, frame_duration):
    plt = _mpl()
    t = np.array([r["time_ns"] for r in rows], dtype=float)
    xlabel = 'Time (ns)' if frame_duration else 'Frame'
    w = max(1, len(t) // 100)

    series = [("upper", f'{label} UPPER half', C_UPPER),
              ("lower", f'{label} LOWER half', C_LOWER),
              ("total", f'{label} TOTAL', C_TOTAL)]

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.6), sharex=True)
    for ax, (key, title, color) in zip(axes, series):
        y = np.array([r[f"{tag}_{key}"] for r in rows], dtype=float)
        # raw values as faint dots — a line through 1000s of integer counts
        # renders as a solid block and hides the trend
        step_pts = max(1, len(y) // 4000)
        ax.scatter(t[::step_pts], y[::step_pts], s=2, color=color, alpha=0.12,
                   linewidths=0, zorder=1)
        roll = _rolling(y, w)
        if roll is not None:
            ax.plot(t, roll, color=color, lw=2.0, zorder=3,
                    label=f'rolling mean ({w} frames)')
        ax.axhline(y.mean(), label=f'mean = {y.mean():.2f}', **MEAN_KW)
        ax.axhline(np.median(y), label=f'median = {np.median(y):.1f}', **MEDIAN_KW)
        ax.set_ylabel('Ions')
        ax.set_title(f'{title}  (r <= {radius} A)', fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0, frameon=False)
    axes[-1].set_xlabel(xlabel, fontsize=12)
    fig.tight_layout()
    _save(fig, plots_dir, f'timeseries_{tag}.png')


# ---- 2. boxplot, one dot per frame -----------------------------------------

def _plot_frame_boxplot(rows, plots_dir, cavity_radius, cylinder_radius,
                        max_dots=3000, seed=0):
    plt = _mpl()
    cols = [("cav_upper", f'Cavity upper\n(r<={cavity_radius})',   C_UPPER),
            ("cav_lower", f'Cavity lower\n(r<={cavity_radius})',   C_LOWER),
            ("cav_total", f'Cavity total\n(r<={cavity_radius})',   C_TOTAL),
            ("cyl_upper", f'Cylinder upper\n(r<={cylinder_radius})', '#8172B2'),
            ("cyl_lower", f'Cylinder lower\n(r<={cylinder_radius})', '#55A868'),
            ("cyl_total", f'Cylinder total\n(r<={cylinder_radius})', '#937860')]

    data = [np.array([r[c] for r in rows], dtype=float) for c, _, _ in cols]
    rng = np.random.default_rng(seed)

    # counts are integers, so dots are jittered in x AND y to show density
    n = len(rows)
    idx = (rng.choice(n, size=max_dots, replace=False)
           if n > max_dots else np.arange(n))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
               medianprops=dict(color='green', linewidth=1.8),
               boxprops=dict(facecolor='#EAF2FB', edgecolor='#2E4057'),
               whiskerprops=dict(color='#2E4057'),
               capprops=dict(color='#2E4057'))

    for i, (d, (_, _, color)) in enumerate(zip(data, cols), start=1):
        xj = i + rng.uniform(-0.17, 0.17, len(idx))
        yj = d[idx] + rng.uniform(-0.17, 0.17, len(idx))
        ax.scatter(xj, yj, s=4, color=color, alpha=0.25, linewidths=0, zorder=2)
        ax.scatter([i], [d.mean()], marker='D', s=42, color='red', zorder=4)

    ax.plot([], [], 'rD', label='mean')
    ax.plot([], [], color='green', lw=1.8, label='median')
    ax.scatter([], [], s=12, color='grey', alpha=0.5,
               label=f'one dot = one frame (n={len(idx)}'
                     + (f' of {n} sampled)' if n > max_dots else ')'))

    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels([lab for _, lab, _ in cols], fontsize=8)
    ax.set_ylabel('Ions in region', fontsize=12)
    ax.set_title('Per-frame occupancy — each dot is one frame (jittered)',
                 fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False)
    fig.tight_layout()
    _save(fig, plots_dir, 'boxplot_per_frame.png')


# ---- 3. how much time at each occupancy ------------------------------------

def _plot_occupancy_histograms(rows, plots_dir, cavity_radius, cylinder_radius):
    plt = _mpl()
    n = len(rows)
    panels = [("cav", f'Cavity (r <= {cavity_radius} A)'),
              ("cyl", f'Cylinder (r <= {cylinder_radius} A)')]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (tag, title) in zip(axes, panels):
        up  = np.array([r[f"{tag}_upper"] for r in rows], dtype=int)
        low = np.array([r[f"{tag}_lower"] for r in rows], dtype=int)
        kmax = int(max(up.max(), low.max()))
        ks = np.arange(0, kmax + 1)
        f_up  = [np.count_nonzero(up == k) / n * 100 for k in ks]
        f_low = [np.count_nonzero(low == k) / n * 100 for k in ks]
        w = 0.4
        ax.bar(ks - w / 2, f_up,  width=w, color=C_UPPER, edgecolor='white',
               label=f'upper (mean {up.mean():.2f})')
        ax.bar(ks + w / 2, f_low, width=w, color=C_LOWER, edgecolor='white',
               label=f'lower (mean {low.mean():.2f})')
        for k, a, b in zip(ks, f_up, f_low):
            if a > 1:
                ax.text(k - w / 2, a + 0.5, f'{a:.0f}', ha='center', fontsize=7)
            if b > 1:
                ax.text(k + w / 2, b + 0.5, f'{b:.0f}', ha='center', fontsize=7)
        ax.set_xticks(ks)
        ax.set_xlabel('Ions in half', fontsize=11)
        ax.set_ylabel('% of frames', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, plots_dir, 'occupancy_histograms.png')


# ---- 4. end_2 against a LOCAL baseline -------------------------------------

def add_local_baselines(rows, end2_rows, window_frames=200):
    """
    For each end_2 event, the mean occupancy over frames within
    +/- window_frames of it, excluding the event frame itself.

    This is the comparison that matters: occupancy often drifts over a
    trajectory, so measuring an event against the whole-run mean mixes the
    drift into the effect. Adds *_local_base keys to each end2 row in place.
    """
    if not end2_rows:
        return
    frames = np.array([r["frame"] for r in rows])
    cols = ["cav_upper", "cav_lower", "cav_total",
            "cyl_upper", "cyl_lower", "cyl_total"]
    data = {c: np.array([r[c] for r in rows], dtype=float) for c in cols}

    for e in end2_rows:
        f = e["end2_frame"]
        sel = (frames >= f - window_frames) & (frames <= f + window_frames) & (frames != f)
        if not np.any(sel):
            sel = frames != f
        for c in cols:
            e[f"{c}_local_base"] = round(float(data[c][sel].mean()), 4)
        e["local_base_n_frames"] = int(np.count_nonzero(sel))


def _paired_stats(end2_rows, col):
    """(mean difference, p-value) for end_2 minus its local baseline."""
    e = np.array([r[f"{col}_without"] for r in end2_rows], dtype=float)
    b = np.array([r[f"{col}_local_base"] for r in end2_rows], dtype=float)
    d = e - b
    p = float("nan")
    try:
        from scipy import stats
        if np.any(d != 0):
            p = float(stats.wilcoxon(d).pvalue)
    except Exception:
        pass
    return float(d.mean()), p, e, b


def _plot_local_baseline(end2_rows, plots_dir, window_frames):
    if not end2_rows or "cav_upper_local_base" not in end2_rows[0]:
        return
    plt = _mpl()
    panels = [("cav_upper", "Cavity UPPER", C_UPPER),
              ("cav_lower", "Cavity LOWER", C_LOWER),
              ("cyl_upper", "Cylinder UPPER", '#8172B2'),
              ("cyl_lower", "Cylinder LOWER", '#55A868')]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.4), sharey=False)
    rng = np.random.default_rng(0)
    for ax, (col, title, color) in zip(axes, panels):
        mean_d, p, e, b = _paired_stats(end2_rows, col)
        x = rng.uniform(-0.06, 0.06, len(e))
        for xi, bi, ei in zip(x, b, e):
            ax.plot([1 + xi, 2 + xi], [bi, ei], color='grey', lw=0.6, alpha=0.5,
                    zorder=1)
        ax.scatter(1 + x, b, s=26, color='lightgrey', edgecolor='#2E4057',
                   linewidth=0.5, zorder=3, label='local baseline')
        ax.scatter(2 + x, e, s=26, color=color, edgecolor='white',
                   linewidth=0.5, zorder=3, label='at end_2')
        ax.plot([1, 2], [b.mean(), e.mean()], color='red', ls='--', lw=1.8,
                marker='D', ms=7, zorder=4, label='mean')
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'local\nbaseline\n(+/-{window_frames} fr)', 'at end_2\n(no perm. ion)'],
                           fontsize=8)
        ax.set_xlim(0.6, 2.4)
        ptxt = 'p = n/a' if np.isnan(p) else f'p = {p:.3f}'
        ax.set_title(f'{title}\nDelta = {mean_d:+.3f} ions, {ptxt}', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Ions', fontsize=11)
    axes[0].legend(fontsize=7, frameon=False, loc='best')
    fig.suptitle(f'Each event against its own local baseline '
                 f'(n={len(end2_rows)} events, paired Wilcoxon)', fontsize=12)
    fig.tight_layout()
    _save(fig, plots_dir, 'end2_vs_local_baseline.png')


# ---- 5. where the ions actually sit ----------------------------------------

def _plot_density(z_hist, rz_hist, z_edges, r_edges, plots_dir,
                  n_frames, cavity_radius, cylinder_radius):
    plt = _mpl()
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # axial profile (ions per frame per bin), 0 = ASN ring, 1 = GLU ring
    ax1.plot(z_centres, z_hist / n_frames, color=C_TOTAL, lw=2)
    ax1.fill_between(z_centres, z_hist / n_frames, color=C_TOTAL, alpha=0.25)
    ax1.axvline(0.5, color='black', ls='--', lw=1.2, label='mid-plane (the cut)')
    ax1.axvline(0.0, color=C_LOWER, ls=':', lw=1.5, label='ASN/ASP ring')
    ax1.axvline(1.0, color=C_UPPER, ls=':', lw=1.5, label='GLU ring')
    ax1.set_xlabel('Normalised axial position  (0 = ASN, 1 = GLU)', fontsize=11)
    ax1.set_ylabel('Ions per frame per bin', fontsize=11)
    ax1.set_title(f'Axial ion density, r <= {cavity_radius} A', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, frameon=False)

    # r-z map
    im = ax2.imshow((rz_hist / n_frames).T, origin='lower', aspect='auto',
                    cmap='magma',
                    extent=[z_edges[0], z_edges[-1], r_edges[0], r_edges[-1]])
    ax2.axvline(0.5, color='white', ls='--', lw=1.2)
    ax2.axhline(cylinder_radius, color='cyan', ls='-', lw=1.4,
                label=f'narrow cylinder r = {cylinder_radius} A')
    ax2.set_xlabel('Normalised axial position  (0 = ASN, 1 = GLU)', fontsize=11)
    ax2.set_ylabel('Radial distance from axis (A)', fontsize=11)
    ax2.set_title('Ion density map', fontsize=12)
    ax2.legend(fontsize=8, loc='upper right', frameon=True)
    fig.colorbar(im, ax=ax2, label='Ions per frame per bin')

    fig.tight_layout()
    _save(fig, plots_dir, 'ion_density.png')


# ---- 6. convergence --------------------------------------------------------

def _plot_running_mean(rows, plots_dir, frame_duration):
    plt = _mpl()
    t = np.array([r["time_ns"] for r in rows], dtype=float)
    xlabel = 'Time (ns)' if frame_duration else 'Frame'
    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, key, color, style in [("cav", "upper", C_UPPER, '-'),
                                   ("cav", "lower", C_LOWER, '-'),
                                   ("cyl", "upper", '#8172B2', '--'),
                                   ("cyl", "lower", '#55A868', '--')]:
        y = np.array([r[f"{tag}_{key}"] for r in rows], dtype=float)
        run = np.cumsum(y) / np.arange(1, len(y) + 1)
        ax.plot(t, run, color=color, ls=style, lw=1.8,
                label=f'{tag} {key} -> {run[-1]:.2f}')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Cumulative mean ions', fontsize=12)
    ax.set_title('Running mean occupancy (flat = converged sampling)', fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False)
    fig.tight_layout()
    _save(fig, plots_dir, 'running_mean.png')


# ---- 7. end_2 vs the all-frame baseline ------------------------------------

def _plot_end2_vs_baseline(rows, end2_rows, plots_dir):
    if not end2_rows:
        return
    plt = _mpl()
    panels = [("cav_upper", "Cavity UPPER", C_UPPER),
              ("cav_lower", "Cavity LOWER", C_LOWER),
              ("cyl_upper", "Cylinder UPPER", '#8172B2'),
              ("cyl_lower", "Cylinder LOWER", '#55A868')]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, (col, title, color) in zip(axes, panels):
        base = np.array([r[col] for r in rows], dtype=float)
        e_wo = np.array([r[f"{col}_without"] for r in end2_rows], dtype=float)
        kmax = int(max(base.max(), e_wo.max() if len(e_wo) else 0))
        ks = np.arange(0, kmax + 1)
        fb = [np.count_nonzero(base == k) / len(base) * 100 for k in ks]
        fe = [np.count_nonzero(e_wo == k) / len(e_wo) * 100 for k in ks]
        w = 0.4
        ax.bar(ks - w / 2, fb, width=w, color='lightgrey', edgecolor='white',
               label=f'all frames ({base.mean():.2f})')
        ax.bar(ks + w / 2, fe, width=w, color=color, edgecolor='white',
               label=f'at end_2, no perm. ion ({e_wo.mean():.2f})')
        ax.set_xticks(ks)
        ax.set_xlabel('Ions', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel('% of frames / events', fontsize=11)
    fig.suptitle('Occupancy at permeation (end_2) vs the whole trajectory',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, plots_dir, 'end2_vs_baseline.png')


# ---- 8. end_2 histograms (with vs without the permeating ion) --------------

def _plot_end2(end2_rows, plots_dir):
    if not end2_rows:
        return
    plt = _mpl()
    panels = [
        ('cav_upper_with',    'Cavity upper (with perm. ion)',   C_UPPER),
        ('cav_lower_with',    'Cavity lower (with perm. ion)',   C_LOWER),
        ('cav_upper_without', 'Cavity upper (without)',          C_UPPER),
        ('cav_lower_without', 'Cavity lower (without)',          C_LOWER),
        ('cyl_upper_with',    'Cylinder upper (with perm. ion)', '#8172B2'),
        ('cyl_lower_with',    'Cylinder lower (with perm. ion)', '#55A868'),
        ('cyl_upper_without', 'Cylinder upper (without)',        '#8172B2'),
        ('cyl_lower_without', 'Cylinder lower (without)',        '#55A868'),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for ax, (col, title, color) in zip(axes.ravel(), panels):
        vals = np.array([r[col] for r in end2_rows], dtype=float)
        bins = np.arange(-0.5, max(vals.max(), 1) + 1.5, 1)
        ax.hist(vals, bins=bins, color=color, edgecolor='white')
        ax.axvline(vals.mean(), **MEAN_KW)
        ax.axvline(np.median(vals), **MEDIAN_KW)
        ax.set_title(f'{title}\nmean={vals.mean():.2f} med={np.median(vals):.1f}',
                     fontsize=9)
        ax.set_xlabel('Ions')
        ax.grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylabel('Events')
    axes[1, 0].set_ylabel('Events')
    fig.tight_layout()
    _save(fig, plots_dir, 'end2_distributions.png')


# ---- 9. end_2 boxplot, one dot per event -----------------------------------

def _plot_end2_boxplot(end2_rows, plots_dir, seed=0):
    if not end2_rows:
        return
    plt = _mpl()
    cols = [('cav_upper_with', 'Cav upper\nwith', C_UPPER),
            ('cav_upper_without', 'Cav upper\nwithout', C_UPPER),
            ('cav_lower_with', 'Cav lower\nwith', C_LOWER),
            ('cav_lower_without', 'Cav lower\nwithout', C_LOWER),
            ('cyl_upper_with', 'Cyl upper\nwith', '#8172B2'),
            ('cyl_upper_without', 'Cyl upper\nwithout', '#8172B2'),
            ('cyl_lower_with', 'Cyl lower\nwith', '#55A868'),
            ('cyl_lower_without', 'Cyl lower\nwithout', '#55A868')]
    data = [np.array([r[c] for r in end2_rows], dtype=float) for c, _, _ in cols]
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
               medianprops=dict(color='green', linewidth=1.8),
               boxprops=dict(facecolor='#EAF2FB', edgecolor='#2E4057'),
               whiskerprops=dict(color='#2E4057'),
               capprops=dict(color='#2E4057'))
    for i, (d, (_, _, color)) in enumerate(zip(data, cols), start=1):
        xj = i + rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(xj, d, s=26, color=color, edgecolor='white', linewidth=0.5,
                   alpha=0.9, zorder=3)
        ax.scatter([i], [d.mean()], marker='D', s=40, color='red', zorder=4)
    ax.plot([], [], 'rD', label='mean')
    ax.plot([], [], color='green', lw=1.8, label='median')
    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels([lab for _, lab, _ in cols], fontsize=8)
    ax.set_ylabel('Ions', fontsize=12)
    ax.set_title(f'Occupancy at end_2 — each dot is one permeation event '
                 f'(n={len(end2_rows)})', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False)
    fig.tight_layout()
    _save(fig, plots_dir, 'end2_boxplot.png')


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_cavity_split_analysis(u,
                              results_dir,
                              channel_type,
                              permeation_events=None,
                              glu_resids=None,
                              asn_resids=None,
                              cavity_radius=12.0,
                              cylinder_radius=3.0,
                              ion_selection="name K+ or resname K+ or name K",
                              frame_duration=0.075,
                              local_window=200,
                              start_frame=0,
                              stop_frame=None,
                              step=1,
                              force_rerun=False):
    """
    u                 MDAnalysis Universe (topology + trajectory already loaded)
    results_dir       per-run results folder; output goes to <results_dir>/cavity_split/
    channel_type      G12 / G12_ML / G12_GAT / G2 (or pass glu_resids/asn_resids)
    permeation_events list of event dicts with ion_id and end_2; if None, read
                      from permeation_table.json
    cavity_radius     radius of the wide cavity cylinder, A (default 12)
    cylinder_radius   radius of the narrow on-axis cylinder, A (default 3)
    frame_duration    ns per frame, for the time column (set None to skip)
    local_window      half-width in frames of the local baseline window used to
                      compare each end_2 event against its own surroundings
    """
    results_dir = Path(results_dir)
    out_dir = results_dir / "cavity_split"
    plots_dir = out_dir / "plots"
    per_frame_csv = out_dir / "per_frame_counts.csv"
    end2_csv = out_dir / "end2_counts.csv"
    summary_csv = out_dir / "end2_summary.csv"
    summary_txt = out_dir / "summary.txt"

    if per_frame_csv.exists() and not force_rerun:
        print(f"  {per_frame_csv} already exists — skipping (force_rerun=False).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # left over from an earlier version of this module
    stale = plots_dir / "per_frame_counts.png"
    if stale.exists():
        stale.unlink()

    glu_resids, asn_resids = resolve_rings(channel_type, glu_resids, asn_resids)

    glu_sel = "resid " + " ".join(str(r) for r in glu_resids)
    asn_sel = "resid " + " ".join(str(r) for r in asn_resids)
    glu_ag = u.select_atoms(glu_sel)
    asn_ag = u.select_atoms(asn_sel)
    ions = u.select_atoms(ion_selection)

    for ag, label, sel in [(glu_ag, "GLU ring", glu_sel),
                           (asn_ag, "ASN/ASP ring", asn_sel),
                           (ions, "K+ ions", ion_selection)]:
        if len(ag) == 0:
            raise RuntimeError(f"{label} selection matched 0 atoms: '{sel}'")

    # ion_id -> row index in the ions AtomGroup.
    # ion_id is whatever the main pipeline uses; try the plausible meanings in
    # order rather than mixing them into one dict (their number ranges overlap,
    # so a single dict would silently mis-resolve).
    by_index  = {int(a.index): i for i, a in enumerate(ions)}       # 0-based
    by_index1 = {int(a.index) + 1: i for i, a in enumerate(ions)}   # 1-based / AMBER
    by_id     = {int(a.id): i for i, a in enumerate(ions)}
    try:
        by_resid = {int(a.resid): i for i, a in enumerate(ions)}
    except AttributeError:
        by_resid = {}

    def resolve_ion_index(ion_id):
        """Row index of this ion in `ions`, and which key matched."""
        if ion_id is None:
            return None, "none"
        try:
            k = int(ion_id)
        except (TypeError, ValueError):
            return None, "unparseable"
        for table, name in ((by_index, "atom_index"), (by_id, "atom_id"),
                            (by_index1, "atom_index+1"), (by_resid, "resid")):
            if k in table:
                return table[k], name
        return None, "no_match"

    if permeation_events is None:
        permeation_events = _load_permeation_events(results_dir)

    # end_2 frame -> list of (event_index, ion_id)
    end2_map = {}
    skipped_events = 0
    for ei, ev in enumerate(permeation_events or []):
        f2 = _get_end2(ev)
        if f2 is None:
            skipped_events += 1
            continue
        end2_map.setdefault(f2, []).append((ei, _get_ion_id(ev)))

    print("\n" + "=" * 80)
    print(f"CAVITY SPLIT ANALYSIS — {channel_type}")
    print("=" * 80)
    print(f"  GLU ring resids:     {glu_resids}")
    print(f"  ASN/ASP ring resids: {asn_resids}")
    print(f"  Cavity cylinder radius:  {cavity_radius} A")
    print(f"  Narrow cylinder radius:  {cylinder_radius} A")
    print(f"  K+ ions in system:   {len(ions)}")
    print(f"  Permeation events with an end_2 frame: "
          f"{sum(len(v) for v in end2_map.values())}"
          + (f" ({skipped_events} without end_2, skipped)" if skipped_events else ""))

    rows = []
    end2_rows = []

    # Accumulators for the ion-density map: normalised axial position
    # (0 = ASN ring, 1 = GLU ring) vs radial distance from the axis.
    Z_BINS, R_BINS = 60, 40
    z_edges = np.linspace(0.0, 1.0, Z_BINS + 1)
    r_edges = np.linspace(0.0, cavity_radius, R_BINS + 1)
    z_hist = np.zeros(Z_BINS)
    rz_hist = np.zeros((Z_BINS, R_BINS))

    traj = u.trajectory[start_frame:stop_frame:step]
    for ts in tqdm(traj, total=len(traj), desc="  frames", unit="frame"):
        frame = int(ts.frame)

        glu_com = glu_ag.center_of_mass()
        asn_com = asn_ag.center_of_mass()
        z_glu, z_asn = float(glu_com[2]), float(asn_com[2])
        z_mid = 0.5 * (z_glu + z_asn)
        axis_xy = (0.5 * (glu_com[0] + asn_com[0]),
                   0.5 * (glu_com[1] + asn_com[1]))

        ion_pos = ions.positions.copy()

        cav = count_split(ion_pos, axis_xy, z_glu, z_asn, cavity_radius)
        cyl = count_split(ion_pos, axis_xy, z_glu, z_asn, cylinder_radius)

        # density accumulation (ions inside the wide cylinder only)
        z_lo, z_hi = (z_asn, z_glu) if z_asn <= z_glu else (z_glu, z_asn)
        span = (z_hi - z_lo) or 1.0
        r_all = np.hypot(ion_pos[:, 0] - axis_xy[0], ion_pos[:, 1] - axis_xy[1])
        zn_all = (ion_pos[:, 2] - z_lo) / span
        sel = (zn_all >= 0.0) & (zn_all <= 1.0) & (r_all <= cavity_radius)
        if np.any(sel):
            h2, _, _ = np.histogram2d(zn_all[sel], r_all[sel],
                                      bins=[z_edges, r_edges])
            rz_hist += h2
            z_hist += h2.sum(axis=1)

        rows.append({
            "frame":       frame,
            "time_ns":     round(frame * frame_duration, 4) if frame_duration else frame,
            "z_glu":       round(z_glu, 3),
            "z_asn":       round(z_asn, 3),
            "z_mid":       round(z_mid, 3),
            "cav_total":   cav["total"],
            "cav_upper":   cav["upper"],
            "cav_lower":   cav["lower"],
            "cyl_total":   cyl["total"],
            "cyl_upper":   cyl["upper"],
            "cyl_lower":   cyl["lower"],
        })

        # ---- end_2 frames: with and without the permeating ion ----
        for event_index, ion_id in end2_map.get(frame, []):
            idx, match_kind = resolve_ion_index(ion_id)
            cav_wo = count_split(ion_pos, axis_xy, z_glu, z_asn,
                                 cavity_radius, exclude_index=idx)
            cyl_wo = count_split(ion_pos, axis_xy, z_glu, z_asn,
                                 cylinder_radius, exclude_index=idx)
            if idx is not None:
                p = ion_pos[idx]
                perm_z = round(float(p[2]), 3)
                perm_r = round(float(np.hypot(p[0] - axis_xy[0],
                                              p[1] - axis_xy[1])), 3)
                perm_half_cav = classify_ion(p, axis_xy, z_glu, z_asn, cavity_radius)
                perm_half_cyl = classify_ion(p, axis_xy, z_glu, z_asn, cylinder_radius)
            else:
                perm_z = perm_r = None
                perm_half_cav = perm_half_cyl = "unknown"

            end2_rows.append({
                "event_index":        event_index,
                "ion_id":             ion_id,
                "end2_frame":         frame,
                "time_ns":            round(frame * frame_duration, 4) if frame_duration else frame,
                "z_glu":              round(z_glu, 3),
                "z_asn":              round(z_asn, 3),
                "z_mid":              round(z_mid, 3),
                "perm_ion_matched":   match_kind,
                "perm_ion_z":         perm_z,
                "perm_ion_r":         perm_r,
                "perm_ion_half_cav":  perm_half_cav,
                "perm_ion_half_cyl":  perm_half_cyl,
                "cav_total_with":     cav["total"],
                "cav_upper_with":     cav["upper"],
                "cav_lower_with":     cav["lower"],
                "cav_total_without":  cav_wo["total"],
                "cav_upper_without":  cav_wo["upper"],
                "cav_lower_without":  cav_wo["lower"],
                "cyl_total_with":     cyl["total"],
                "cyl_upper_with":     cyl["upper"],
                "cyl_lower_with":     cyl["lower"],
                "cyl_total_without":  cyl_wo["total"],
                "cyl_upper_without":  cyl_wo["upper"],
                "cyl_lower_without":  cyl_wo["lower"],
            })

    if not rows:
        print("  No frames processed — nothing written.")
        return

    # ── CSVs ──
    with open(per_frame_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Wrote {per_frame_csv} ({len(rows)} frames)")

    # Local baselines: each event compared against its own neighbourhood, not
    # the whole-run mean, so a drifting occupancy cannot masquerade as a signal.
    add_local_baselines(rows, end2_rows, window_frames=local_window)

    # ── why "with" and "without" may be identical ──
    if end2_rows:
        n = len(end2_rows)
        n_unmatched = sum(1 for r in end2_rows
                          if r["perm_ion_matched"] in ("no_match", "none", "unparseable"))
        n_outside = sum(1 for r in end2_rows if r["perm_ion_half_cav"] == "outside")
        n_differ = sum(1 for r in end2_rows
                       if r["cav_total_with"] != r["cav_total_without"])
        print("\n  PERMEATING-ION EXCLUSION CHECK")
        print(f"    events: {n}   with != without: {n_differ}")
        print(f"    ion_id not matched to any K+ atom: {n_unmatched}")
        print(f"    permeating ion outside the cavity at its end_2 frame: {n_outside}")
        if n_unmatched:
            kinds = sorted({r["perm_ion_matched"] for r in end2_rows})
            print(f"    -> ion_id could not be resolved (match kinds seen: {kinds}). "
                  f"Check what ion_id means in permeation_table.json; the 'without' "
                  f"columns are NOT excluding anything for those events.")
        elif n_differ == 0:
            print("    -> ion_id resolved fine, so the permeating ion simply is not "
                  "inside the GLU/ASN cylinder at end_2 (expected: end_2 is the frame "
                  "it leaves the cavity). 'with' == 'without' is the correct answer.")

        with open(end2_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(end2_rows[0].keys()))
            w.writeheader()
            w.writerows(end2_rows)
        print(f"  Wrote {end2_csv} ({len(end2_rows)} events)")

        print(f"\n  end_2 vs LOCAL baseline (+/-{local_window} frames, paired)")
        for c in ("cav_upper", "cav_lower", "cyl_upper", "cyl_lower"):
            mean_d, p, e, b = _paired_stats(end2_rows, c)
            ptxt = "n/a" if np.isnan(p) else f"{p:.3f}"
            print(f"    {c:<10} baseline {b.mean():.3f} -> end_2 {e.mean():.3f}  "
                  f"delta {mean_d:+.3f}  wilcoxon p={ptxt}")

        count_cols = [c for c in end2_rows[0]
                      if c.startswith(("cav_", "cyl_"))]
        with open(summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["column", "mean", "median", "std", "min", "max", "n"])
            for c in count_cols:
                v = np.array([r[c] for r in end2_rows], dtype=float)
                w.writerow([c, round(v.mean(), 4), round(float(np.median(v)), 4),
                            round(v.std(ddof=0), 4), int(v.min()), int(v.max()),
                            len(v)])
        print(f"  Wrote {summary_csv}")
    else:
        print("  No end_2 frames fell inside the analysed frame range — "
              "end2_counts.csv not written.")

    # ── summary.txt ──
    cav_tot = np.array([r["cav_total"] for r in rows], dtype=float)
    cav_up  = np.array([r["cav_upper"] for r in rows], dtype=float)
    cav_low = np.array([r["cav_lower"] for r in rows], dtype=float)
    cyl_tot = np.array([r["cyl_total"] for r in rows], dtype=float)
    cyl_up  = np.array([r["cyl_upper"] for r in rows], dtype=float)
    cyl_low = np.array([r["cyl_lower"] for r in rows], dtype=float)
    z_sep = np.array([abs(r["z_glu"] - r["z_asn"]) for r in rows], dtype=float)

    with open(summary_txt, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"CAVITY SPLIT ANALYSIS — {channel_type}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"GLU ring resids:        {glu_resids}\n")
        f.write(f"ASN/ASP ring resids:    {asn_resids}\n")
        f.write(f"Cavity cylinder radius: {cavity_radius} A\n")
        f.write(f"Narrow cylinder radius: {cylinder_radius} A\n")
        f.write(f"Frames analysed:        {len(rows)} "
                f"({rows[0]['frame']} to {rows[-1]['frame']}, step {step})\n")
        f.write(f"K+ ions in system:      {len(ions)}\n\n")
        f.write(f"GLU-ASN z separation:   mean {z_sep.mean():.3f} A "
                f"(min {z_sep.min():.3f}, max {z_sep.max():.3f})\n\n")
        f.write("PER-FRAME OCCUPANCY (mean +/- sd):\n")
        f.write(f"  Cavity  r<={cavity_radius} A   total {cav_tot.mean():.3f} +/- {cav_tot.std():.3f}, "
                f"upper {cav_up.mean():.3f} +/- {cav_up.std():.3f}, "
                f"lower {cav_low.mean():.3f} +/- {cav_low.std():.3f}\n")
        f.write(f"  Cylinder r<={cylinder_radius} A  total {cyl_tot.mean():.3f} +/- {cyl_tot.std():.3f}, "
                f"upper {cyl_up.mean():.3f} +/- {cyl_up.std():.3f}, "
                f"lower {cyl_low.mean():.3f} +/- {cyl_low.std():.3f}\n")
        if end2_rows:
            f.write(f"\nAT end_2 FRAMES ({len(end2_rows)} events), mean:\n")
            for c in [c for c in end2_rows[0] if c.startswith(("cav_", "cyl_"))]:
                v = np.array([r[c] for r in end2_rows], dtype=float)
                f.write(f"  {c:<22} {v.mean():.3f}\n")
            f.write("\n'with' includes the permeating ion; 'without' excludes it.\n")
            f.write(f"\nEND_2 vs LOCAL BASELINE (+/-{local_window} frames, "
                    f"paired Wilcoxon):\n")
            for c in ("cav_upper", "cav_lower", "cyl_upper", "cyl_lower"):
                mean_d, p, e, b = _paired_stats(end2_rows, c)
                ptxt = "n/a" if np.isnan(p) else f"{p:.3f}"
                f.write(f"  {c:<10} baseline {b.mean():.3f} -> end_2 {e.mean():.3f}"
                        f"   delta {mean_d:+.3f}   p = {ptxt}\n")
            f.write("The local baseline excludes the event frame itself. Compare "
                    "against this, not the\nwhole-run mean, when occupancy drifts "
                    "over the trajectory (see running_mean.png).\n")
    print(f"  Wrote {summary_txt}")

    # ── plots ──
    _plot_timeseries(rows, plots_dir, "cav", "Cavity", cavity_radius, frame_duration)
    _plot_timeseries(rows, plots_dir, "cyl", "Cylinder", cylinder_radius, frame_duration)
    _plot_frame_boxplot(rows, plots_dir, cavity_radius, cylinder_radius)
    _plot_occupancy_histograms(rows, plots_dir, cavity_radius, cylinder_radius)
    _plot_local_baseline(end2_rows, plots_dir, local_window)
    _plot_density(z_hist, rz_hist, z_edges, r_edges, plots_dir, len(rows),
                  cavity_radius, cylinder_radius)
    _plot_running_mean(rows, plots_dir, frame_duration)
    _plot_end2(end2_rows, plots_dir)
    _plot_end2_boxplot(end2_rows, plots_dir)
    _plot_end2_vs_baseline(rows, end2_rows, plots_dir)

    print("\n  PER-FRAME MEANS")
    print(f"    cavity   (r<={cavity_radius} A): total {cav_tot.mean():.2f}, "
          f"upper {cav_up.mean():.2f}, lower {cav_low.mean():.2f}")
    print(f"    cylinder (r<={cylinder_radius} A): total {cyl_tot.mean():.2f}, "
          f"upper {cyl_up.mean():.2f}, lower {cyl_low.mean():.2f}")
    if end2_rows:
        def m(c):
            return np.mean([r[c] for r in end2_rows])
        print(f"  AT end_2 ({len(end2_rows)} events)")
        print(f"    cavity   with:    upper {m('cav_upper_with'):.2f}, lower {m('cav_lower_with'):.2f}")
        print(f"    cavity   without: upper {m('cav_upper_without'):.2f}, lower {m('cav_lower_without'):.2f}")
        print(f"    cylinder with:    upper {m('cyl_upper_with'):.2f}, lower {m('cyl_lower_with'):.2f}")
        print(f"    cylinder without: upper {m('cyl_upper_without'):.2f}, lower {m('cyl_lower_without'):.2f}")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI (optional — normal use is the call from main.py)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Split the GLU/ASN cavity in half and count ions per frame")
    ap.add_argument("-prmtop", required=True)
    ap.add_argument("-traj", required=True)
    ap.add_argument("-results_dir", required=True,
                    help="Run results folder (holds permeation_table.json)")
    ap.add_argument("-channel", default="G12")
    ap.add_argument("-cavity_radius", type=float, default=12.0)
    ap.add_argument("-cylinder_radius", type=float, default=3.0)
    ap.add_argument("-glu", nargs="+", type=int, default=None)
    ap.add_argument("-asn", nargs="+", type=int, default=None)
    ap.add_argument("-frame_duration", type=float, default=0.075)
    ap.add_argument("-local_window", type=int, default=200,
                    help="Half-width in frames of the end_2 local baseline window")
    ap.add_argument("-start", type=int, default=0)
    ap.add_argument("-stop", type=int, default=None)
    ap.add_argument("-step", type=int, default=1)
    ap.add_argument("-force_rerun", action="store_true")
    args = ap.parse_args()

    try:
        import MDAnalysis as mda
    except ImportError:
        raise SystemExit("ERROR: MDAnalysis required (conda activate Amber)")

    u = mda.Universe(args.prmtop, args.traj)
    run_cavity_split_analysis(
        u=u,
        results_dir=args.results_dir,
        channel_type=args.channel,
        glu_resids=args.glu,
        asn_resids=args.asn,
        cavity_radius=args.cavity_radius,
        cylinder_radius=args.cylinder_radius,
        frame_duration=args.frame_duration,
        local_window=args.local_window,
        start_frame=args.start,
        stop_frame=args.stop,
        step=args.step,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()


'''
python3 cavity_split_analysis.py -prmtop com_4fs.prmtop -traj protein.nc \
    -results_dir /media/ziyue/.../girk_analyser_results/G12/RUN3 -channel G12
'''