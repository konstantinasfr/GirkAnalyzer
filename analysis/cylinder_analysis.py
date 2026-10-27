#!/usr/bin/env python3
"""
Cylinder Analysis — where K+ sits along each subunit's GLU -> ASN/ASP line
==========================================================================
The MD counterpart of esp_cylinders.py / esp_maps.py. Every frame, each
subunit gets a cylinder whose axis runs from its GLU CA to its ASN (or ASP)
CA, built exactly like the QM ESP cylinders:

    axis    GLU CA -> ASN CA, recomputed every frame (follows the protein)
    radius  `radius` A perpendicular to that axis          (default 3.0)
    length  CA-CA distance + `extend` A past each CA        (default 3.0)

Every K+ inside a cylinder is recorded with its axial position t
(0 = GLU CA, L = ASN CA, same x-axis as the ESP axial profile), its
fractional position s = t / L, and its radial distance r from the axis.

Making the histogram statistically meaningful
---------------------------------------------
- Density, not counts: every histogram is ions per frame per A
  (counts / (n_frames x bin width)). Its area is the mean number of K+ in
  the cylinder, and runs of different length are directly comparable.
- Frames are NOT independent (an ion sitting for 200 frames gives 200
  correlated counts), so no per-frame p-values are computed. Within one
  run the uncertainty is a block SEM: the run is cut into `n_blocks`
  contiguous blocks, the density is computed per block, SEM over blocks.
  Across runs (aggregate_cylinder_analysis.py) each run is one independent
  sample.
- Visits: one unbroken stretch of frames an ion spends inside a cylinder.
  Each visit contributes ONE mean position, a far less correlated sample
  than frames. A visit still running at the last frame is censored = 1.

Which end is GLU: decided by membership of glu_residues / asn_residues;
if a pair is in neither, the CA with the higher z is taken as GLU.

What it produces   <results_dir>/cylinder_analysis/
---------------------------------------------------
1. cylinder_ion_hits.csv              every (frame, subunit, ion) inside: t, s, r, L
2. cylinder_per_frame.csv             per frame: n ions and CA-CA length per subunit
3. cylinder_visits.csv                every visit: start, end, n frames, mean t / s
4. cylinder_density.csv               axial density + block SEM per subunit
5. cylinder_summary.txt               occupancy, positions, visits per subunit
6. cylinder_analysis.pkl              all of the above, for the aggregator
7. cylinder_density_panels.png        one panel per subunit, block SEM band
8. cylinder_density_overlay.png       all subunits on the ESP-profile axis
9. cylinder_density_fractional.png    same, along s = t / L
10. cylinder_radial.png               r distribution per subunit (radius check)
11. cylinder_counts_timeseries.png    K+ in each cylinder vs time

How to call it from main.py
----------------------------
    from analysis.cylinder_analysis import run_cylinder_analysis

    run_cylinder_analysis(
        u=u,
        subunit_groups=subunit_groups,
        glu_residues=glu_residues,
        asn_residues=asn_residues,
        results_dir=results_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_duration=frame_duration,
        channel_type=channel_type,
    )
"""

import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


_ION_SEL = "resname K+ K"

# same colours as the esp_maps.py axial profile (tab10 0, 1, 2, 4)
SUBUNIT_COLORS = [plt.cm.tab10(x) for x in np.linspace(0, 0.4, 4)]

TITLE_SIZE  = 16
LABEL_SIZE  = 14
TICK_SIZE   = 12
LEGEND_SIZE = 11

AXIAL_XLABEL = "position along Glu-CA $\\rightarrow$ Asn-CA axis (Å)"


# =============================================================================
# Geometry
# =============================================================================

def order_pair(u, pair, glu_residues, asn_residues):
    """Returns (glu_resid, asn_resid) for one subunit_groups entry."""
    a, b = pair
    glu, asn = set(glu_residues or []), set(asn_residues or [])
    if a in glu and b in asn:
        return a, b
    if b in glu and a in asn:
        return b, a
    za = u.select_atoms(f"resid {a} and name CA")[0].position[2]
    zb = u.select_atoms(f"resid {b} and name CA")[0].position[2]
    print(f"  [NOTE] pair {pair} not split by glu/asn lists, using CA z "
          f"(higher z = GLU)")
    return (a, b) if za >= zb else (b, a)


def axial_radial(points, p_start, p_end):
    """t along p_start -> p_end, r perpendicular to it, and the CA-CA length."""
    vec = p_end - p_start
    length = float(np.linalg.norm(vec))
    u_ax = vec / length
    rel = points - p_start
    t = rel @ u_ax
    r = np.linalg.norm(rel - np.outer(t, u_ax), axis=1)
    return t, r, length


# =============================================================================
# Visits
# =============================================================================

class VisitTracker:
    """Unbroken runs of frames each ion spends inside one cylinder."""

    def __init__(self, n_ions):
        self.entry = np.full(n_ions, -1, dtype=int)
        self.n = np.zeros(n_ions, dtype=int)
        self.t_sum = np.zeros(n_ions)
        self.s_sum = np.zeros(n_ions)
        self.visits = []

    def update(self, inside, t, s, frame, ion_resids):
        left = (self.entry >= 0) & ~inside
        for i in np.nonzero(left)[0]:
            self._record(i, frame - 1, ion_resids, censored=0)
        new = inside & (self.entry < 0)
        self.entry[new] = frame
        self.n[inside] += 1
        self.t_sum[inside] += t[inside]
        self.s_sum[inside] += s[inside]

    def finish(self, last_frame, ion_resids):
        for i in np.nonzero(self.entry >= 0)[0]:
            self._record(i, last_frame, ion_resids, censored=1)

    def _record(self, i, end, ion_resids, censored):
        n = int(self.n[i])
        self.visits.append(dict(ion_resid=int(ion_resids[i]),
                                start=int(self.entry[i]), end=int(end), n_frames=n,
                                mean_t=self.t_sum[i] / n, mean_s=self.s_sum[i] / n,
                                censored=censored))
        self.entry[i] = -1
        self.n[i] = 0
        self.t_sum[i] = 0.0
        self.s_sum[i] = 0.0


# =============================================================================
# Densities
# =============================================================================

def density(values, n_frames, edges):
    """Ions per frame per unit of x."""
    counts, _ = np.histogram(values, bins=edges)
    return counts / (n_frames * np.diff(edges))


def block_density(values, frame_pos, n_frames, edges, n_blocks):
    """
    Density of the whole run plus its block SEM.
    frame_pos : 0-based frame position (within the analysed window) of each value
    """
    dens = density(values, n_frames, edges)
    n_blocks = max(1, min(n_blocks, n_frames))
    bounds = np.linspace(0, n_frames, n_blocks + 1).astype(int)
    blocks = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = (frame_pos >= lo) & (frame_pos < hi)
        blocks.append(density(values[sel], hi - lo, edges))
    blocks = np.array(blocks)
    sem = (blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)) if n_blocks > 1 \
        else np.zeros_like(dens)
    return dens, sem, blocks


def axial_edges(extend, max_length, bin_width):
    hi = np.ceil((max_length + extend) / bin_width) * bin_width
    return np.arange(-extend, hi + bin_width / 2, bin_width)


def fractional_edges(extend, min_length, bin_width=0.05):
    pad = np.ceil(extend / min_length / bin_width) * bin_width
    return np.arange(-pad, 1 + pad + bin_width / 2, bin_width)


def describe(values):
    if len(values) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, q1=np.nan, q3=np.nan)
    values = np.asarray(values, dtype=float)
    return dict(n=int(values.size), mean=float(values.mean()),
                median=float(np.median(values)),
                std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
                q1=float(np.percentile(values, 25)), q3=float(np.percentile(values, 75)))


# =============================================================================
# Main entry point
# =============================================================================

def run_cylinder_analysis(
    u,
    subunit_groups,
    glu_residues,
    asn_residues,
    results_dir,
    start_frame=0,
    end_frame=None,
    frame_duration=0.075,
    channel_type="",
    radius=3.0,
    extend=3.0,
    bin_width=0.5,
    n_blocks=5,
    ion_selection=_ION_SEL,
):
    """
    Records every K+ inside each subunit's GLU-CA -> ASN-CA cylinder, then
    saves the per-subunit axial histograms.

    Parameters
    ----------
    u              : MDAnalysis Universe
    subunit_groups : {name: [res1, res2]} GLU and ASN/ASP of each subunit, any order
    glu_residues   : GLU ring resids (decides which end of each pair is GLU)
    asn_residues   : ASN/ASP ring resids
    results_dir    : Path
    start_frame    : int
    end_frame      : int or None (last frame)
    frame_duration : float  ns per frame (0.075 G12, 0.4 G2)
    channel_type   : str, used in titles
    radius         : float  cylinder radius, A
    extend         : float  extension past each CA along the axis, A
    bin_width      : float  axial histogram bin width, A
    n_blocks       : int    contiguous blocks for the within-run SEM
    ion_selection  : str    MDAnalysis selection for K+
    """
    out_dir = Path(results_dir) / "cylinder_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(u.trajectory) - 1
    n_frames = end_frame - start_frame + 1

    print("\n" + "=" * 70)
    print("CYLINDER ANALYSIS (K+ along GLU-CA -> ASN-CA)")
    print("=" * 70)

    ions = u.select_atoms(ion_selection)
    if len(ions) == 0:
        print(f"  [ERROR] No ions found with: '{ion_selection}'")
        return None
    ion_resids = ions.resids.copy()

    names = list(subunit_groups)
    pairs = {n: order_pair(u, subunit_groups[n], glu_residues, asn_residues) for n in names}
    ca_idx = []
    for n in names:
        for resid in pairs[n]:
            ca_idx.append(u.select_atoms(f"resid {resid} and name CA")[0].index)
    cas = u.atoms[ca_idx]

    print(f"  K+ ions in system : {len(ions)}")
    print(f"  Cylinder          : r = {radius} Å, extend = {extend} Å past each CA")
    print(f"  Frames            : {start_frame} → {end_frame}  ({n_frames} total)")
    for n in names:
        print(f"  {n:<4} GLU {pairs[n][0]:>5}  →  ASN/ASP {pairs[n][1]:>5}")

    # ── per-frame loop ─────────────────────────────────────────────────────────
    hits = {n: dict(frame=[], ion_resid=[], t=[], s=[], r=[], L=[]) for n in names}
    n_in = {n: np.zeros(n_frames, dtype=int) for n in names}
    lengths = {n: np.zeros(n_frames) for n in names}
    trackers = {n: VisitTracker(len(ions)) for n in names}
    n_multi = 0   # ion-frames counted in more than one cylinder

    for fi, ts in enumerate(tqdm(u.trajectory[start_frame: end_frame + 1],
                                 total=n_frames, desc="  Cylinders",
                                 unit="frame", dynamic_ncols=True)):
        pos = ions.positions
        ca_pos = cas.positions
        in_any = np.zeros(len(ions), dtype=int)
        for k, n in enumerate(names):
            t, r, L = axial_radial(pos, ca_pos[2 * k], ca_pos[2 * k + 1])
            s = t / L
            inside = (r <= radius) & (t >= -extend) & (t <= L + extend)
            in_any += inside
            lengths[n][fi] = L
            n_in[n][fi] = int(inside.sum())
            trackers[n].update(inside, t, s, ts.frame, ion_resids)
            idx = np.nonzero(inside)[0]
            if idx.size:
                h = hits[n]
                h["frame"].extend([ts.frame] * idx.size)
                h["ion_resid"].extend(ion_resids[idx].tolist())
                h["t"].extend(t[idx].tolist())
                h["s"].extend(s[idx].tolist())
                h["r"].extend(r[idx].tolist())
                h["L"].extend([L] * idx.size)
        n_multi += int(np.sum(in_any > 1))

    for n in names:
        trackers[n].finish(end_frame, ion_resids)
    hits = {n: {k: np.asarray(v) for k, v in h.items()} for n, h in hits.items()}
    visits = {n: trackers[n].visits for n in names}

    # ── densities ──────────────────────────────────────────────────────────────
    max_L = max(lengths[n].max() for n in names)
    min_L = min(lengths[n].min() for n in names)
    t_edges = axial_edges(extend, max_L, bin_width)
    s_edges = fractional_edges(extend, min_L)
    dens = {}
    for n in names:
        fpos = hits[n]["frame"] - start_frame
        d, sem, _ = block_density(hits[n]["t"], fpos, n_frames, t_edges, n_blocks)
        ds, sems, _ = block_density(hits[n]["s"], fpos, n_frames, s_edges, n_blocks)
        dens[n] = dict(t=d, t_sem=sem, s=ds, s_sem=sems)

    # ── summary ────────────────────────────────────────────────────────────────
    stats = {}
    for n in names:
        complete = [v for v in visits[n] if not v["censored"]]
        peak = dens[n]["t"]
        stats[n] = dict(
            glu=pairs[n][0], asn=pairs[n][1],
            mean_L=float(lengths[n].mean()), std_L=float(lengths[n].std()),
            occupancy=float(np.mean(n_in[n] > 0)),
            mean_ions=float(n_in[n].mean()),
            t=describe(hits[n]["t"]), s=describe(hits[n]["s"]), r=describe(hits[n]["r"]),
            peak_t=float(0.5 * (t_edges[:-1] + t_edges[1:])[np.argmax(peak)])
            if peak.sum() > 0 else np.nan,
            n_visits=len(visits[n]), n_censored=len(visits[n]) - len(complete),
            visit_ns=describe([v["n_frames"] * frame_duration for v in complete]),
            visit_t=describe([v["mean_t"] for v in visits[n]]),
        )

    title = f"{channel_type}  |  " if channel_type else ""
    lines = [
        "CYLINDER ANALYSIS — K+ along GLU-CA -> ASN-CA",
        f"Frames      : {start_frame} → {end_frame}  ({n_frames} frames, "
        f"{(n_frames - 1) * frame_duration:.2f} ns, {frame_duration} ns/frame)",
        f"Cylinder    : r = {radius} Å, extend = {extend} Å, bin = {bin_width} Å",
        f"Block SEM   : {n_blocks} contiguous blocks",
        f"Ion-frames in >1 cylinder : {n_multi}",
        "",
        "Density = K+ per frame per Å. t = 0 at GLU CA, t = L at ASN CA.",
        "",
        f"{'subunit':<8}{'GLU':>6}{'ASN':>6}{'L (Å)':>12}{'occup.':>8}{'<n>':>7}"
        f"{'hits':>8}{'median t':>10}{'IQR t':>16}{'peak t':>8}{'median r':>10}"
        f"{'visits':>8}{'med. visit ns':>15}",
        "-" * 122,
    ]
    for n in names:
        st = stats[n]
        lines.append(
            f"{n:<8}{st['glu']:>6}{st['asn']:>6}"
            f"{st['mean_L']:>7.2f}±{st['std_L']:<4.2f}"
            f"{st['occupancy']:>8.3f}{st['mean_ions']:>7.3f}{st['t']['n']:>8}"
            f"{st['t']['median']:>10.2f}"
            f"{'[' + format(st['t']['q1'], '.2f') + ', ' + format(st['t']['q3'], '.2f') + ']':>16}"
            f"{st['peak_t']:>8.2f}{st['r']['median']:>10.2f}"
            f"{st['n_visits']:>8}{st['visit_ns']['median']:>15.3f}")
    lines += ["", "occup. = fraction of frames with >= 1 K+ in the cylinder",
              "<n>    = mean K+ in the cylinder per frame (area under the density)"]
    summary = "\n".join(lines)
    print("\n" + summary)
    (out_dir / "cylinder_summary.txt").write_text(summary + "\n")

    # ── CSVs ───────────────────────────────────────────────────────────────────
    with open(out_dir / "cylinder_ion_hits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ns", "subunit", "ion_resid", "t_A", "s", "r_A", "L_A"])
        for n in names:
            h = hits[n]
            for i in range(len(h["t"])):
                w.writerow([int(h["frame"][i]), round(h["frame"][i] * frame_duration, 4), n,
                            int(h["ion_resid"][i]), round(h["t"][i], 4), round(h["s"][i], 5),
                            round(h["r"][i], 4), round(h["L"][i], 4)])

    frames = np.arange(start_frame, end_frame + 1)
    with open(out_dir / "cylinder_per_frame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ns"] + [f"n_{n}" for n in names] + [f"L_{n}" for n in names])
        for fi, fr in enumerate(frames):
            w.writerow([int(fr), round(fr * frame_duration, 4)]
                       + [int(n_in[n][fi]) for n in names]
                       + [round(lengths[n][fi], 4) for n in names])

    with open(out_dir / "cylinder_visits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subunit", "ion_resid", "start", "end", "n_frames", "duration_ns",
                    "mean_t_A", "mean_s", "censored"])
        for n in names:
            for v in visits[n]:
                w.writerow([n, v["ion_resid"], v["start"], v["end"], v["n_frames"],
                            round(v["n_frames"] * frame_duration, 4),
                            round(v["mean_t"], 4), round(v["mean_s"], 5), v["censored"]])

    centres = 0.5 * (t_edges[:-1] + t_edges[1:])
    with open(out_dir / "cylinder_density.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_centre_A"] + [c for n in names for c in (f"density_{n}", f"sem_{n}")])
        for b, c in enumerate(centres):
            w.writerow([round(c, 4)] + [round(x, 6) for n in names
                                        for x in (dens[n]["t"][b], dens[n]["t_sem"][b])])

    result = dict(
        channel_type=channel_type, names=names, pairs=pairs,
        start_frame=start_frame, end_frame=end_frame, n_frames=n_frames,
        frame_duration=frame_duration, radius=radius, extend=extend,
        bin_width=bin_width, n_blocks=n_blocks,
        hits=hits, n_in=n_in, lengths=lengths, visits=visits, stats=stats,
        t_edges=t_edges, s_edges=s_edges, density=dens, n_multi=n_multi,
    )
    with open(out_dir / "cylinder_analysis.pkl", "wb") as f:
        pickle.dump(result, f)

    # ── plots ──────────────────────────────────────────────────────────────────
    mean_L = {n: stats[n]["mean_L"] for n in names}
    labels = {n: f"{n}  (<n> = {stats[n]['mean_ions']:.2f})" for n in names}
    plot_density_panels(t_edges, {n: dens[n]["t"] for n in names},
                        {n: dens[n]["t_sem"] for n in names}, mean_L,
                        out_dir / "cylinder_density_panels.png",
                        f"{title}K+ density per subunit (r = {radius} Å), "
                        f"band = SEM over {n_blocks} blocks",
                        panel_titles={n: f"{n}   GLU {pairs[n][0]} → ASN {pairs[n][1]}   "
                                         f"occupancy {stats[n]['occupancy']:.2f}"
                                      for n in names})
    plot_density_overlay(t_edges, {n: dens[n]["t"] for n in names},
                         {n: dens[n]["t_sem"] for n in names}, mean_L, labels,
                         out_dir / "cylinder_density_overlay.png",
                         f"{title}K+ axial density (r = {radius} Å cylinder)")
    plot_density_overlay(s_edges, {n: dens[n]["s"] for n in names},
                         {n: dens[n]["s_sem"] for n in names}, {n: 1.0 for n in names},
                         labels, out_dir / "cylinder_density_fractional.png",
                         f"{title}K+ density along s = t / L",
                         xlabel="fractional position s  (0 = Glu CA, 1 = Asn CA)",
                         ylabel="K+ per frame per unit s")
    _plot_radial(hits, names, n_frames, radius, out_dir / "cylinder_radial.png", title)
    _plot_counts(frames * frame_duration, n_in, names, out_dir / "cylinder_counts_timeseries.png",
                 title)

    print(f"\n  Saved to {out_dir}")
    return result


# =============================================================================
# Plots (also used by aggregate_cylinder_analysis.py)
# =============================================================================

def _ca_markers(ax, mean_L, names, colors):
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    for n, col in zip(names, colors):
        ax.axvline(mean_L[n], color=col, linestyle=":", linewidth=1.2, alpha=0.8)


def _ca_labels(ax, mean_L):
    yhi = ax.get_ylim()[1]
    ax.text(0, yhi, " Glu CA", va="top", fontsize=TICK_SIZE)
    ax.text(np.mean(list(mean_L.values())), yhi, " Asn CA", va="top", fontsize=TICK_SIZE)


def plot_density_panels(edges, dens, band, mean_L, out_path, suptitle,
                        panel_titles=None, per_run=None, band_label="SEM",
                        xlabel=AXIAL_XLABEL, ylabel="K+ per frame per Å"):
    """
    One panel per subunit, shared x and y.
    dens / band : {name: array}  band is drawn as dens ± band
    per_run     : optional {name: 2D array (runs x bins)} drawn as thin lines
    """
    names = list(dens)
    centres = 0.5 * (edges[:-1] + edges[1:])
    colors = SUBUNIT_COLORS[:len(names)] if len(names) <= 4 \
        else [plt.cm.tab10(i % 10) for i in range(len(names))]
    ncol = 2
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 4.6 * nrow), squeeze=False,
                             sharex=True, sharey=True, layout="constrained")
    for ax, n, col in zip(axes.ravel(), names, colors):
        if per_run is not None:
            for row in per_run[n]:
                ax.step(centres, row, where="mid", color="grey", alpha=0.35, linewidth=0.8)
        ax.fill_between(centres, np.clip(dens[n] - band[n], 0, None), dens[n] + band[n],
                        step="mid", color=col, alpha=0.25, linewidth=0, label=band_label)
        ax.step(centres, dens[n], where="mid", color=col, linewidth=2)
        _ca_markers(ax, mean_L, [n], [col])
        ax.set_title(panel_titles[n] if panel_titles else n, fontsize=LABEL_SIZE)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=TICK_SIZE)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=TITLE_SIZE)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def plot_density_overlay(edges, dens, band, mean_L, labels, out_path, title,
                         xlabel=AXIAL_XLABEL, ylabel="K+ per frame per Å"):
    """All subunits on one axis, styled like the esp_maps.py axial profile."""
    names = list(dens)
    centres = 0.5 * (edges[:-1] + edges[1:])
    colors = SUBUNIT_COLORS[:len(names)] if len(names) <= 4 \
        else [plt.cm.tab10(i % 10) for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(11, 6.0))
    for n, col in zip(names, colors):
        ax.plot(centres, dens[n], color=col, linewidth=2, label=labels[n])
        ax.fill_between(centres, np.clip(dens[n] - band[n], 0, None), dens[n] + band[n],
                        color=col, alpha=0.18, linewidth=0)
    _ca_markers(ax, mean_L, names, colors)
    ax.set_ylim(bottom=0)
    _ca_labels(ax, mean_L)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def _plot_radial(hits, names, n_frames, radius, out_path, title):
    """
    K+ per frame per Å^2 of cross-section vs r: each annulus is divided by
    its area, so a flat line means ions fill the cylinder evenly and a
    curve still high at r = radius means ions sit just outside it too.
    """
    edges = np.linspace(0, radius, 13)
    centres = 0.5 * (edges[:-1] + edges[1:])
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n, col in zip(names, SUBUNIT_COLORS if len(names) <= 4
                      else [plt.cm.tab10(i % 10) for i in range(len(names))]):
        counts, _ = np.histogram(hits[n]["r"], bins=edges)
        ax.plot(centres, counts / (n_frames * area), "o-", color=col, linewidth=2, label=n)
    ax.set_xlabel("radial distance from the Glu-CA → Asn-CA axis (Å)", fontsize=LABEL_SIZE)
    ax.set_ylabel("K+ per frame per Å² (any t)", fontsize=LABEL_SIZE)
    ax.set_title(f"{title}radial K+ distribution inside the cylinders", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def _plot_counts(time_ns, n_in, names, out_path, title):
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.8 * len(names) + 1),
                             sharex=True, squeeze=False)
    for ax, n, col in zip(axes[:, 0], names, SUBUNIT_COLORS if len(names) <= 4
                          else [plt.cm.tab10(i % 10) for i in range(len(names))]):
        ax.step(time_ns, n_in[n], where="post", color=col, linewidth=0.8)
        ax.set_ylabel(n, fontsize=LABEL_SIZE)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time (ns)", fontsize=LABEL_SIZE)
    fig.suptitle(f"{title}K+ inside each cylinder", fontsize=TITLE_SIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")
