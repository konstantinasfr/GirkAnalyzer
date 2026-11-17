#!/usr/bin/env python3
"""
esp_maps.py
===========
Spatial views of the QM electrostatic potential from a Jaguar .resp grid.
Companion to esp_cylinders.py (which it imports helpers from, so keep the two
files in the same directory).

Where esp_cylinders.py collapses each cylinder to a single number, this script
keeps the geometry and shows you the potential as a field.

FIGURES PRODUCED
  1. <label>_axial_profile.png
        ESP vs position along each subunit's Glu-CA -> Asn-CA axis.
        Median line + interquartile band, one curve per subunit.
        Answers: WHERE between the two rings does the difference sit?

  2. <label>_xy_map.png
        Top-down view of the inter-ring slab. Grid points projected onto xy
        (relative to the pore axis), binned, coloured by mean ESP, with the
        Glu and Asn CA positions marked.
        Answers: is the negative potential four-fold symmetric, or is one
        subunit deeper?

  3. <label>_rz_maps.png
        Per-subunit heatmap of ESP in (axial position, radial distance) space.
        Answers: what is the shape of each well?

  4. <label>_radial_profile.png
        ESP vs distance from the pore axis, within the inter-ring slab.
        Answers: is the potential deepest on-axis or off-axis?

  5. <label>_ecdf.png
        Empirical CDF per subunit. A whole-distribution version of the boxplot
        -- if one curve is shifted bodily left, that is a real shift and not a
        tail artefact.

USAGE (same arguments as esp_cylinders.py):

    python3 esp_maps.py --resp frame5606.resp \
        --struct com_4fs.prmtop --coords window_01.rst \
        --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
        --glu 99 424 748 1073 --asn 131 456 780 1105 \
        --radius 3.0 --extend 3.0 --label G12_frame5606

  or, if you already know the cluster atom indices:

    python3 esp_maps.py --resp frame5606.resp \
        --pairs 7:30 50:73 95:118 140:163 --names G1 A B C

UNITS: kcal/mol per +1e, Angstrom. Negative = stabilising for a cation.

CAVEAT: the Jaguar grid is built on vdW shells, so coverage is uneven in
space. Bins with few points are masked out (--min-bin) rather than plotted as
noise. Always read the companion count panels before interpreting a colour.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from esp_cylinders import (read_resp, load_reference_ca, map_resid_to_cluster,
                           parse_group_members, classify_groups, cylinder_mask,
                           cyl_color, cyl_order)

CMAP = "RdBu_r"

# ──────────────────────────────────────────────────────────────────────
#  Figure styling
# ──────────────────────────────────────────────────────────────────────

_FONT_SCALE = 1.0


def FS(size):
    """Scale a font size by the current --fontscale."""
    return size * _FONT_SCALE


def set_font_scale(scale):
    """
    Scales every text element in the figures. Base sizes are tuned for a
    full-width figure; raise this when the figures will be shrunk into a
    manuscript column or a slide, so the labels stay readable.
    """
    global _FONT_SCALE
    _FONT_SCALE = float(scale)
    import matplotlib
    matplotlib.rcParams.update({
        "font.size":         FS(10),
        "axes.titlesize":    FS(12),
        "axes.labelsize":    FS(11),
        "xtick.labelsize":   FS(10),
        "ytick.labelsize":   FS(10),
        "legend.fontsize":   FS(10),
        "figure.titlesize":  FS(13),
        "axes.linewidth":    1.0 + 0.3 * (_FONT_SCALE - 1),
        "xtick.major.width": 1.0 + 0.3 * (_FONT_SCALE - 1),
        "ytick.major.width": 1.0 + 0.3 * (_FONT_SCALE - 1),
        "savefig.bbox":      "tight",
    })
      # negative -> blue, less negative -> red


# ──────────────────────────────────────────────────────────────────────
#  Binned helpers
# ──────────────────────────────────────────────────────────────────────


def binned_stat_1d(x, v, edges, min_count=5):
    """Median, Q1, Q3 and count of v in bins of x. NaN where count < min_count."""
    idx = np.digitize(x, edges) - 1
    n = len(edges) - 1
    med = np.full(n, np.nan)
    q1 = np.full(n, np.nan)
    q3 = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=int)
    for b in range(n):
        sel = v[idx == b]
        cnt[b] = sel.size
        if sel.size >= min_count:
            med[b] = np.median(sel)
            q1[b], q3[b] = np.percentile(sel, [25, 75])
    return med, q1, q3, cnt


def binned_mean_2d(x, y, v, xedges, yedges, min_count=3):
    """Mean of v per 2D bin, masked where count < min_count."""
    cnt, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    tot, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=v)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = tot / cnt
    return np.ma.masked_where(cnt < min_count, mean), cnt


def symmetric_norm(arrays, lo_pct=1, hi_pct=99):
    """Shared colour limits across panels, robust to outliers."""
    parts = []
    for a in arrays:
        parts.append(a.compressed() if np.ma.isMaskedArray(a) else np.asarray(a).ravel())
    pooled = np.concatenate(parts) if parts else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return Normalize(-1, 1)
    return Normalize(np.percentile(pooled, lo_pct), np.percentile(pooled, hi_pct))


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resp", required=True)
    ap.add_argument("--groups", nargs="+", default=None,
                    help="NAME:RES,RES per subunit, e.g. A:780,748 G1:99,131")
    ap.add_argument("--struct", default=None, help="prmtop or PDB (needed with --groups)")
    ap.add_argument("--coords", default=None, help="rst/ncrst/nc frame for --struct")
    ap.add_argument("--glu", nargs="+", type=int, default=None,
                    help="Upper-ring (Glu) residue ids. Optional: if omitted, each "
                         "--groups pair is ordered by CA z-coordinate.")
    ap.add_argument("--asn", nargs="+", type=int, default=None,
                    help="Lower-ring (Asn/Asp) residue ids. Optional, see --glu.")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Alternative to --groups: GLU_CA:ASN_CA cluster atom indices")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--match-tol", type=float, default=0.6)
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--extend", type=float, default=3.0)
    ap.add_argument("--label", default="esp_maps",
                    help="Prefix for all output file NAMES")
    ap.add_argument("--title", default=None,
                    help="Short title printed on the figures, e.g. 'GIRK1/2, frame 5606' "
                         "(default: --label)")
    ap.add_argument("--fontscale", type=float, default=1.5,
                    help="Multiplier for every font size in the figures "
                         "(default 1.5). Raise it if you shrink the figures into "
                         "a manuscript column or a slide.")
    ap.add_argument("--dpi", type=int, default=200, help="Figure resolution (default 200)")
    ap.add_argument("--ylim", nargs=2, type=float, default=[-280.0, -60.0], metavar=("YMIN", "YMAX"),
                    help="Fix the y-axis of the axial profile, e.g. --ylim -280 -60, "
                         "so profiles from different frames/constructs can be compared "
                         "directly. Default -280 -60, the same as the ensemble profiles, so every "
                         "frame and construct is drawn on the same scale.")
    ap.add_argument("--xlim", nargs=2, type=float, default=[-3.0, 14.0], metavar=("XMIN", "XMAX"),
                    help="Fix the x-axis of the axial profile (default -3 14, the same for "
                         "every frame and construct)")
    ap.add_argument("--radial-ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
                    help="Fix the y-axis of the radial profile (top panel). Default: automatic.")
    ap.add_argument("--outdir", "--folder", "-o", dest="outdir", default=None,
                    help="Folder to save everything into. Created if it does not "
                         "exist. Defaults to a folder named after --label. "
                         "Pass '.' to write into the current directory instead.")
    ap.add_argument("--axial-bin", type=float, default=0.5, help="Axial bin width, A")
    ap.add_argument("--xy-bin", type=float, default=0.75, help="xy map bin width, A")
    ap.add_argument("--rz-bin", type=float, default=0.5, help="r-z map bin width, A")
    ap.add_argument("--radial-bin", type=float, default=0.5, help="Radial bin width, A")
    ap.add_argument("--xy-halfwidth", type=float, default=16.0,
                    help="Half-width of the top-down xy map, A")
    ap.add_argument("--slab-pad", type=float, default=0.0,
                    help="Extra A above/below the two CA planes for the slab views")
    ap.add_argument("--min-bin", type=int, default=3, help="Min points for a 2D bin to be drawn")
    ap.add_argument("--min-count", type=int, default=5, help="Min points for a 1D profile bin")
    args = ap.parse_args()
    set_font_scale(args.fontscale)

    outdir = Path(args.outdir) if args.outdir is not None else Path(args.label)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {outdir.resolve()}")

    atoms, grid_xyz, esp = read_resp(Path(args.resp))
    print(f"Read {Path(args.resp).name}: {len(atoms)} atoms, {len(grid_xyz)} grid points")
    print(f"  ESP range {esp.min():.1f} to {esp.max():.1f} kcal/mol per +1e")

    # ---- resolve subunit pairs ----
    if args.groups:
        if not args.struct:
            raise SystemExit("ERROR: --groups needs --struct")
        raw_groups = parse_group_members(args.groups)
        resids = sorted({r for _, members in raw_groups for r in members})
        ref_ca = load_reference_ca(args.struct, args.coords, resids)
        mapping = map_resid_to_cluster(ref_ca, atoms, tol=args.match_tol)
        groups = classify_groups(raw_groups, args.glu, args.asn, mapping, atoms)
        pairs = [(mapping[g][0], mapping[a][0]) for _, g, a in groups]
        names = [n for n, _, _ in groups]
        print("\nResidue -> cluster atom:")
        for (name, g, a), (gi, ai) in zip(groups, pairs):
            print(f"  {name:<4} Glu {g} -> atom {gi} ({mapping[g][1]:.3f} A)   "
                  f"Asn {a} -> atom {ai} ({mapping[a][1]:.3f} A)")
    elif args.pairs:
        pairs = [tuple(int(x) for x in s.split(":")) for s in args.pairs]
        names = args.names or [f"cyl{i+1}" for i in range(len(pairs))]
    else:
        raise SystemExit("ERROR: pass --groups (with --struct) or --pairs")
    if len(names) != len(pairs):
        raise SystemExit("ERROR: --names length does not match the number of subunits")

    glu_pts = np.array([atoms[a - 1] for a, _ in pairs])
    asn_pts = np.array([atoms[b - 1] for _, b in pairs])

    # pore axis: xy centroid of all CA atoms used
    axis_xy = np.vstack([glu_pts, asn_pts])[:, :2].mean(axis=0)
    z_glu = glu_pts[:, 2].mean()
    z_asn = asn_pts[:, 2].mean()
    print(f"\nPore axis (xy centroid of CAs): {axis_xy[0]:.2f}, {axis_xy[1]:.2f}")
    print(f"Glu CA plane z = {z_glu:.2f} A   |   Asn CA plane z = {z_asn:.2f} A")

    title = args.title or args.label
    colours = [cyl_color(n, i) for i, n in enumerate(names)]

    # ---- per-cylinder point sets ----
    cyl = {}
    for (ia, ib), name in zip(pairs, names):
        p0, p1 = atoms[ia - 1], atoms[ib - 1]
        mask, _, length = cylinder_mask(grid_xyz, p0, p1, args.radius, args.extend, "ca-ca")
        rel = grid_xyz[mask] - p0
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        t = rel @ u
        r = np.linalg.norm(rel - np.outer(t, u), axis=1)
        cyl[name] = dict(v=esp[mask], t=t, r=r, length=float(np.linalg.norm(p1 - p0)),
                         xyz=grid_xyz[mask])
        print(f"  {name:<4} n={mask.sum():>6}  CA-CA sep {cyl[name]['length']:.2f} A")

    # ══════════════════════════════════════════════════════════════
    # 1. Axial profile
    # ══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10 * (0.9 + 0.1 * _FONT_SCALE), 6.0))
    tmin = -args.extend
    tmax = max(c["length"] for c in cyl.values()) + args.extend
    edges = np.arange(tmin, tmax + args.axial_bin, args.axial_bin)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for i, name in enumerate(cyl_order(list(cyl))):
        c, col = cyl[name], cyl_color(name, i)
        med, q1, q3, cnt = binned_stat_1d(c["t"], c["v"], edges, args.min_count)
        ax.plot(centres, med, color=col, linewidth=2, label=name)
        ax.fill_between(centres, q1, q3, color=col, alpha=0.18, linewidth=0)
        ax.axvline(c["length"], color=col, linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    if args.ylim is not None:
        ax.set_ylim(min(args.ylim), max(args.ylim))
        for name, c in cyl.items():
            med, _, _, _ = binned_stat_1d(c["t"], c["v"], edges, args.min_count)
            q = med[np.isfinite(med)]
            if q.size and (q.min() < min(args.ylim) or q.max() > max(args.ylim)):
                print(f"  WARNING: {name} median runs outside --ylim "
                      f"({q.min():.1f} to {q.max():.1f}); it will be clipped")
    if args.xlim is not None:
        ax.set_xlim(min(args.xlim), max(args.xlim))
    ylo, yhi = ax.get_ylim()
    # both CA labels at the top, so the Asn label is never hidden by the legend
    ax.text(0, yhi, " Glu CA", va="top", fontsize=FS(9))
    ax.text(np.mean([c["length"] for c in cyl.values()]), yhi, " Asn/Asp CA",
            va="top", fontsize=FS(9))
    ax.set_xlabel("position along Glu-CA $\\rightarrow$ Asn/Asp-CA axis (Å)", fontsize=FS(12))
    ax.set_ylabel("ESP (kcal/mol per +1e)", fontsize=FS(12))
    ax.set_title(f"{title}\naxial QM ESP profile, median and IQR", fontsize=FS(12))
    ax.legend(fontsize=FS(9))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / f"{args.label}_axial_profile.png"
    fig.savefig(p, dpi=args.dpi); plt.close(fig); print(f"\nWrote {p}")

    # ══════════════════════════════════════════════════════════════
    # 2. Top-down xy map of the inter-ring slab
    # ══════════════════════════════════════════════════════════════
    zlo = min(z_glu, z_asn) - args.slab_pad
    zhi = max(z_glu, z_asn) + args.slab_pad
    slab = (grid_xyz[:, 2] >= zlo) & (grid_xyz[:, 2] <= zhi)
    sx = grid_xyz[slab, 0] - axis_xy[0]
    sy = grid_xyz[slab, 1] - axis_xy[1]
    sv = esp[slab]
    print(f"Inter-ring slab {zlo:.1f} < z < {zhi:.1f} A: {slab.sum()} grid points")

    H = args.xy_halfwidth
    xe = np.arange(-H, H + args.xy_bin, args.xy_bin)
    mean_xy, cnt_xy = binned_mean_2d(sx, sy, sv, xe, xe, args.min_bin)
    norm = symmetric_norm([mean_xy])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
    im = axes[0].pcolormesh(xe, xe, mean_xy.T, cmap=CMAP, norm=norm, shading="flat")
    axes[0].set_title("mean ESP in the inter-ring slab", fontsize=FS(12))
    fig.colorbar(im, ax=axes[0], label="ESP (kcal/mol per +1e)", fraction=0.046, pad=0.04)
    im2 = axes[1].pcolormesh(xe, xe, np.ma.masked_where(cnt_xy < 1, cnt_xy).T,
                             cmap="viridis", shading="flat")
    axes[1].set_title("points per bin (sampling check)", fontsize=FS(12))
    fig.colorbar(im2, ax=axes[1], label="points", fraction=0.046, pad=0.04)

    for axp in axes:
        for g, a, name, col in zip(glu_pts, asn_pts, names, colours):
            axp.plot(g[0] - axis_xy[0], g[1] - axis_xy[1], "o", color="black",
                     markersize=7 * (0.6 + 0.4 * _FONT_SCALE), markerfacecolor="white", markeredgewidth=1.6)
            axp.plot(a[0] - axis_xy[0], a[1] - axis_xy[1], "s", color="black",
                     markersize=7 * (0.6 + 0.4 * _FONT_SCALE), markerfacecolor="yellow", markeredgewidth=1.4)
            axp.annotate(name, (g[0] - axis_xy[0], g[1] - axis_xy[1]),
                         textcoords="offset points", xytext=(8, 6),
                         fontsize=FS(11), fontweight="bold")
        axp.plot(0, 0, "+", color="black", markersize=12 * (0.6 + 0.4 * _FONT_SCALE), markeredgewidth=2)
        axp.set_xlabel("x relative to pore axis (Å)", fontsize=FS(11))
        axp.set_ylabel("y relative to pore axis (Å)", fontsize=FS(11))
        axp.set_aspect("equal")
    fig.suptitle(f"{title}  |  top-down view, {zlo:.1f} < z < {zhi:.1f} Å   "
                 f"(○ Glu CA,  □ Asn CA,  + pore axis)", fontsize=FS(12))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = outdir / f"{args.label}_xy_map.png"
    fig.savefig(p, dpi=args.dpi); plt.close(fig); print(f"Wrote {p}")

    # ══════════════════════════════════════════════════════════════
    # 3. r-z heatmaps per subunit
    # ══════════════════════════════════════════════════════════════
    te = np.arange(-args.extend, tmax + args.rz_bin, args.rz_bin)
    re_ = np.arange(0, args.radius + args.rz_bin, args.rz_bin)
    grids = {}
    for name, c in cyl.items():
        grids[name], _ = binned_mean_2d(c["t"], c["r"], c["v"], te, re_, args.min_bin)
    norm = symmetric_norm(list(grids.values()))

    ncol = 2
    nrow = int(np.ceil(len(cyl) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 4.2 * nrow),
                             squeeze=False, layout="constrained")
    for ax, (name, g) in zip(axes.ravel(), grids.items()):
        im = ax.pcolormesh(te, re_, g.T, cmap=CMAP, norm=norm, shading="flat")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.axvline(cyl[name]["length"], color="black", linestyle=":", linewidth=1.2)
        ax.set_title(f"{name}   (Glu CA at 0, Asn CA at {cyl[name]['length']:.1f} Å)",
                     fontsize=FS(11))
        ax.set_xlabel("axial position (Å)", fontsize=FS(10))
        ax.set_ylabel("radial dist. from axis (Å)", fontsize=FS(10))
    for ax in axes.ravel()[len(cyl):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="mean ESP (kcal/mol per +1e)",
                 fraction=0.025, pad=0.02)
    fig.suptitle(f"{title}  |  ESP inside each cylinder", fontsize=FS(13))
    p = outdir / f"{args.label}_rz_maps.png"
    fig.savefig(p, dpi=args.dpi); plt.close(fig); print(f"Wrote {p}")

    # ══════════════════════════════════════════════════════════════
    # 4. Radial profile from the pore axis (inter-ring slab)
    # ══════════════════════════════════════════════════════════════
    rad = np.sqrt(sx ** 2 + sy ** 2)
    redges = np.arange(0, min(H, rad.max()) + args.radial_bin, args.radial_bin)
    rcent = 0.5 * (redges[:-1] + redges[1:])
    med, q1, q3, cnt = binned_stat_1d(rad, sv, redges, args.min_count)

    fig, (ax, axc) = plt.subplots(2, 1, figsize=(10 * (0.9 + 0.1 * _FONT_SCALE), 7.5),
                                  height_ratios=[3, 1], sharex=True)
    ax.plot(rcent, med, color="#1f4e79", linewidth=2.2, label="median ESP")
    ax.fill_between(rcent, q1, q3, color="#1f4e79", alpha=0.2, linewidth=0, label="IQR")
    if np.any(np.isfinite(med)):
        rmin = rcent[np.nanargmin(med)]
        ax.axvline(rmin, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"deepest median at r = {rmin:.2f} Å")
    if args.radial_ylim is not None:
        ax.set_ylim(min(args.radial_ylim), max(args.radial_ylim))
    ax.set_ylabel("ESP (kcal/mol per +1e)", fontsize=FS(12))
    ax.set_title(f"{title}  |  radial ESP profile, {zlo:.1f} < z < {zhi:.1f} Å",
                 fontsize=FS(12))
    ax.legend(fontsize=FS(9)); ax.grid(alpha=0.3)
    axc.bar(rcent, cnt, width=args.radial_bin * 0.9, color="grey", alpha=0.7)
    axc.set_xlabel("distance from pore axis (Å)", fontsize=FS(12))
    axc.set_ylabel("points", fontsize=FS(10))
    axc.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / f"{args.label}_radial_profile.png"
    fig.savefig(p, dpi=args.dpi); plt.close(fig); print(f"Wrote {p}")

    # ══════════════════════════════════════════════════════════════
    # 5. ECDF overlay
    # ══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(9 * (0.9 + 0.1 * _FONT_SCALE), 6.0))
    for i, name in enumerate(cyl_order(list(cyl))):
        c, col = cyl[name], cyl_color(name, i)
        v = np.sort(c["v"])
        ax.plot(v, np.arange(1, v.size + 1) / v.size, color=col, linewidth=2,
                label=f"{name}   median {np.median(v):.1f}")
    ax.set_xlabel("ESP (kcal/mol per +1e)", fontsize=FS(12))
    ax.set_ylabel("cumulative fraction of grid points", fontsize=FS(12))
    ax.set_title(f"{title}  |  ESP distribution per subunit (ECDF)", fontsize=FS(12))
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.legend(fontsize=FS(10)); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / f"{args.label}_ecdf.png"
    fig.savefig(p, dpi=args.dpi); plt.close(fig); print(f"Wrote {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()

'''
python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_spe_B3LYP-D3_LACVPs/jag_frame5606_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
    --ylim -280 -60 \
    --label G12_frame5606 --folder ./results/G12/RUN3/frame5606


python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_E141Q_spe_B3LYP-D3_LACVPs/jag_frame5606_E141Q_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
    --ylim -280 -60 \
    --label G12_frame5606_E141Q --folder ./results/G12/RUN3/frame5606_E141Q

python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_E141Q/RUN5/frame6220.prj/jobs/jag_frame6220_spe_B3LYP-D3_LACVPs/jag_frame6220_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_E141Q/RUN5/pdbs/frame6220_K.pdb \
    --groups A:422,454 B:98,130 C:747,779 G1:1073,1105 \
    --ylim -280 -60 \
    --label G12_E141Q_frame6220 --folder ./results/G12_E141Q/RUN5/frame6220
'''

'''
python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_spe_B3LYP-D3_LACVPs/jag_frame5606_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
    --ylim -280 -60 \
    --label G12_frame5606 --folder ./results/G12/RUN3/frame5606

python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK2_WT/RUN14/frame814_3ions.prj/jobs/jag_frame814_spe_B3LYP-D3_LACVPs/jag_frame814_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK2_WT/RUN14/pdbs/frame814_K.pdb \
    --groups A:98,130 B:754,786 C:426,458 D:1082,1114 \
    --ylim -280 -60 \
    --label G2_frame814 --folder ./results/G2/RUN14/frame814

python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_E141Q_spe_B3LYP-D3_LACVPs/jag_frame5606_E141Q_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
    --ylim -280 -60 \
    --label G12_frame5606_E141Q --folder ./results/G12/RUN3/frame5606_E141Q

python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/frame5606.prj/jobs/jag_frame5606_mutated_spe_B3LYP-D3_LACVPs/jag_frame5606_mutated_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN3/pdbs/frame5606_K.pdb \
    --groups A:780,748 B:456,424 C:1073,1105 G1:99,131 \
    --ylim -280 -60 \
    --label G12_frame5606_D173N --folder ./results/G12/RUN3/frame5606_D173N

python3 esp_maps.py --resp /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_E141Q/RUN5/frame6220.prj/jobs/jag_frame6220_spe_B3LYP-D3_LACVPs/jag_frame6220_spe_B3LYP-D3_LACVPs.resp \
--struct /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_E141Q/RUN5/pdbs/frame6220_K.pdb \
    --groups A:422,454 B:98,130 C:747,779 G1:1073,1105 \
    --ylim -280 -60 \
    --label G12_E141Q_frame6220 --folder ./results/G12_E141Q/RUN5/frame6220
'''