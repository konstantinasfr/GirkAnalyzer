"""
cavity_esp_classical.py
=======================
Classical electrostatic potential (ESP) in the Glu -> Asn/Asp cylinders, over the whole trajectory.

The same quantity as the QM ESP of esp_cylinders.py (Jaguar), computed with fixed force-field charges
(prmtop) so that it can be done on many frames. Validated against the QM on frames 5606 (G12) and 814 (G2)
with esp_classical_check.py: at the Jaguar grid points r = 0.99, G12 - G2 33.6 vs 32.2 (QM).

ATOMS (the QM cluster): the 4 upper-ring residues (Glu / Gln) and the 4 lower-ring residues (Asn / Asp),
complete, plus caps like the ACE / NME of the QM cluster: from the residue before each of them C, O, CA,
HA; from the residue after N, H, CA, HA. Each cap is shifted to net charge 0 (the QM caps are neutral), so
the cluster carries -5 (G12), -4 (G2), the charges of the ring residues only. No ions, no water.

CYLINDERS (as esp_cylinders.py, --axis ca-ca): one per subunit, axis from the Glu CA to the Asn/Asp CA of
the same subunit (the lower-ring residue next in sequence), radius 3 A, extended 3 A beyond each CA. They
are rebuilt in every frame from that frame's CA atoms, so they move with the protein.

POINTS, two grids per cylinder:
  regular mesh (columns mean, median, sd, n_points): even sampling, the fair grid between frames and systems;
  Jaguar-like (columns jag_mean, jag_median, jag_sd, jag_n; ONLY for the frames of interest, NaN elsewhere):
  spheres around every cluster atom at the radii of
  the Jaguar ESP grid (JAG_RADII), points nearer than an element's minimum distance (JAG_RMIN) removed, as
  measured from the frame-5606 .resp; comparable to the QM numbers of esp_cylinders.py.
Regular mesh: Jaguar points exist only for the QM frames, so every frame gets a regular mesh inside each
cylinder: every `step` A along the axis, rings at 0, step, 2 step ... radius, points on each ring every ~step
A around it. Points closer than `exclude` A (1.4) to a cluster atom are dropped (the potential of a point
charge diverges there); the number dropped is recorded.

PER FRAME AND CYLINDER: mean, median, SD (over the cylinder's points), Q1, Q3, the number of points, the
CA-CA length, the contribution of each of the 8 ring residues and of the caps to the mean (they add up to
it), and how far the upper-ring side chain reaches toward the pore axis: distances from the axis of its CA,
CD and carboxylate oxygens (r_o_upper), and the largest r_o of the four in that frame (max_r_o_upper); the
pore axis goes through the centroid of the 8 CAs, along Glu ring -> lower ring. "All Glu pointing in" is
decided when pooling (aggregate_cavity_esp.py). Axial profile: the mean in 0.5 A slices along the cylinder.

Frames: every `stride` frames (default 1 = every frame) plus any `extra_frames` (frames of interest, e.g. the
QM frames), from frame 0; the equilibration cut is applied when pooling (aggregate_cavity_esp.py).

OUTPUT  <results_dir>/cavity_esp/esp_frames_run<run>.csv     one row per frame and cylinder
        <results_dir>/cavity_esp/esp_profile_run<run>.csv.gz  frame, cylinder, axial bin, mean
        <results_dir>/cavity_esp/esp_atoms_run<run>.txt       the atoms used and their charges
        <results_dir>/cavity_esp/esp_run<run>.png             over time, distribution, axial profile
        <results_dir>/cavity_esp/esp_axial_run<run>.npz / .png axial profile like esp_maps.py: median and IQR of all
                                                               points in each 0.5 A slice (2D histograms summed
                                                               over blocks of 10 frames)
        <results_dir>/cavity_esp/esp_hist_run<run>.npz / .png  histogram of the ESP over the cylinder points
                                                               (5 kcal/mol bins), per frame; plot pooled over
                                                               frames, like esp_cylinders.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

COULOMB = 332.0637
# Jaguar-like ESP grid, measured from the .resp of the G12 frame-5606 single point (jaguar_grid_info.py):
# spheres around every atom at these radii (A, the same for every element), points closer to any atom than
# its element's minimum distance removed. Reproduces the Jaguar-point cylinder means within ~2 kcal/mol.
JAG_RADII = (1.6184, 1.7185, 1.8282, 1.9493, 2.0841, 2.2355, 2.4078, 2.6069, 2.8419, 3.1270, 3.4878, 3.9758, 4.7242)
JAG_NPTS = (302, 302, 302, 146, 146, 146, 146, 146, 110, 110, 110, 110, 110)
JAG_RMIN = {"H": 1.60, "O": 1.71, "N": 1.85, "C": 1.94}


def _fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


HIST_EDGES = np.arange(-500.0, 0.0 + 1e-9, 5.0)     # 5 kcal/mol bins, as in esp_cylinders.py
AX_EDGES = np.arange(-3.0, 20.0 + 1e-9, 0.5)        # 0.5 A slices along the cylinder (distance from the Glu CA)
BLOCK = 10                                           # axial 2D histograms are summed over blocks of 10 frames


_JAG_DIRS = np.concatenate([r * _fib(n) for r, n in zip(JAG_RADII, JAG_NPTS)])



CAP_BEFORE = ("C", "O", "CA", "HA", "HA2", "HA3")
CAP_AFTER = ("N", "H", "CA", "HA", "HA2", "HA3")


def pair_rings(glu_residues, asn_residues):
    """[(upper resid, lower resid)]: each upper-ring residue with the lower-ring residue next in sequence."""
    lower = sorted(int(r) for r in asn_residues)
    out = []
    for g in glu_residues:
        after = [r for r in lower if r > int(g)]
        if not after:
            raise ValueError(f"no lower-ring residue after {g}")
        out.append((int(g), min(after)))
    if len({b for _, b in out}) != len(out):
        raise ValueError(f"lower-ring residues paired twice: {out}")
    return out


def cluster_atoms(universe, pairs):
    """(AtomGroup, charges with neutral caps, group label per atom: ring resid or 'caps')."""
    ring = [r for p in pairs for r in p]
    idx, grp = [], []
    for r in ring:
        ag = universe.select_atoms(f"resid {r}")
        if not len(ag):
            raise ValueError(f"resid {r} not found")
        idx += list(ag.indices)
        grp += [str(r)] * len(ag)
    caps = []
    for r in ring:
        for nb, names in ((r - 1, CAP_BEFORE), (r + 1, CAP_AFTER)):
            if nb in ring:
                continue
            ag = universe.select_atoms(f"resid {nb} and name {' '.join(names)}")
            if len(ag):
                caps.append(ag)
                idx += list(ag.indices)
                grp += ["caps"] * len(ag)
    atoms = universe.atoms[idx]
    q = atoms.charges.astype(float).copy()
    n_ring = sum(len(universe.select_atoms(f"resid {r}")) for r in ring)
    pos = n_ring
    for ag in caps:                                 # each cap -> net 0, as the neutral ACE / NME caps
        sl = slice(pos, pos + len(ag))
        q[sl] -= q[sl].mean()
        pos += len(ag)
    return atoms, q, np.array(grp)


def mesh_local(length, radius=3.0, extend=3.0, step=0.5):
    """Points (t along the axis from the Glu CA, r, angle) of the regular cylinder mesh."""
    ts = np.arange(-extend, length + extend + 1e-9, step)
    ring = [(0.0, 0.0)]
    for r in np.arange(step, radius + 1e-9, step):
        n = max(6, int(round(2 * np.pi * r / step)))
        ring += [(r, a) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    ring = np.array(ring)
    t = np.repeat(ts, len(ring))
    rr = np.tile(ring[:, 0], len(ts))
    aa = np.tile(ring[:, 1], len(ts))
    return t, rr, aa


def mesh_xyz(p0, p1, t, r, a):
    u = p1 - p0
    u = u / np.linalg.norm(u)
    tmp = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return p0 + np.outer(t, u) + np.outer(r * np.cos(a), e1) + np.outer(r * np.sin(a), e2)


def potential_by_group(points, xyz, q, groups, names):
    """(total V per point, {group: V per point}); V = 332.06 sum q/r."""
    d = _dist(points, xyz)
    w = COULOMB * q[None, :] / np.maximum(d, 1e-6)
    return w.sum(axis=1), {g: w[:, groups == g].sum(axis=1) for g in names}, d.min(axis=1)


def inward(universe, pairs):
    """{upper resid: (r_CA, r_CD, r_O)}: distances (A) from the pore axis of the CA, the CD and the mean of
    the side-chain terminal O / N (Glu OE1 OE2, Gln OE1 NE2). Small r_O = the carboxylate faces the lumen."""
    cas = np.array([universe.select_atoms(f"resid {r} and name CA").positions[0] for p in pairs for r in p])
    up = cas[0::2].mean(axis=0)
    lo = cas[1::2].mean(axis=0)
    c = cas.mean(axis=0)
    ax = (lo - up) / np.linalg.norm(lo - up)
    rad = lambda x: np.linalg.norm((x - c) - np.dot(x - c, ax) * ax)
    out = {}
    for g, _ in pairs:
        ca = universe.select_atoms(f"resid {g} and name CA")
        cd = universe.select_atoms(f"resid {g} and name CD")
        tip = universe.select_atoms(f"resid {g} and name OE1 OE2 NE2")
        out[g] = (rad(ca.positions[0]),
                  rad(cd.positions[0]) if len(cd) else np.nan,
                  rad(tip.positions.mean(axis=0)) if len(tip) else np.nan)
    return out


def _dist(a, b):
    """Pairwise distances (fast C routine of MDAnalysis when available)."""
    try:
        from MDAnalysis.lib.distances import distance_array
        return distance_array(np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64))
    except ImportError:
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def cyl_mask(points, p0, p1, radius, extend):
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    rel = points - p0
    t = rel @ u
    rad = np.linalg.norm(rel - np.outer(t, u), axis=1)
    return (rad <= radius) & (t >= -extend) & (t <= np.linalg.norm(p1 - p0) + extend)


def frame_esp(universe, pairs, atoms, q, groups, labels, radius=3.0, extend=3.0, step=0.5, exclude=1.4,
              bin_w=0.5, jaguar=True):
    """(rows, profile rows) for the current frame."""
    xyz = atoms.positions.astype(float)
    rmin = np.array([JAG_RMIN.get(n[0].upper(), 1.94) for n in atoms.names])

    names = [str(r) for p in pairs for r in p] + ["caps"]
    inw = inward(universe, pairs)
    max_r_o = float(np.nanmax([v[2] for v in inw.values()]))
    rows, prof = [], []
    for g, l in pairs:
        p0 = universe.select_atoms(f"resid {g} and name CA").positions[0].astype(float)
        p1 = universe.select_atoms(f"resid {l} and name CA").positions[0].astype(float)
        length = float(np.linalg.norm(p1 - p0))
        t, r, a = mesh_local(length, radius, extend, step)
        pts = mesh_xyz(p0, p1, t, r, a)
        V, parts, dmin = potential_by_group(pts, xyz, q, groups, names)
        keep = dmin >= exclude
        v = V[keep]
        row = dict(cylinder=labels[(g, l)], upper=g, lower=l, length_A=length, n_points=int(keep.sum()),
                   n_excluded=int((~keep).sum()), mean=v.mean(), median=np.median(v), sd=v.std(ddof=1),
                   q1=np.percentile(v, 25), q3=np.percentile(v, 75), r_ca_upper=inw[g][0], r_cd_upper=inw[g][1],
                   r_o_upper=inw[g][2], max_r_o_upper=max_r_o)
        for nmx in names:
            key = "caps" if nmx == "caps" else labels.get(int(nmx), nmx)
            row[f"contrib_{key}"] = parts[nmx][keep].mean()
        # the Jaguar-like grid in the same cylinder (only when asked: the frames of interest)
        # only atoms whose spheres can reach the cylinder carry candidate points
        if jaguar:
            near = cyl_mask(xyz, p0, p1, radius + JAG_RADII[-1], extend + JAG_RADII[-1])
            cand = (xyz[near][:, None, :] + _JAG_DIRS[None, :, :]).reshape(-1, 3)
            jp = cand[cyl_mask(cand, p0, p1, radius, extend)]
        else:
            jp = np.empty((0, 3))
        if len(jp):
            dj = _dist(jp, xyz)
            jp, dj = jp[(dj >= rmin[None, :] - 1e-6).all(axis=1)], dj[(dj >= rmin[None, :] - 1e-6).all(axis=1)]
        if len(jp):
            vj = COULOMB * (q[None, :] / np.maximum(dj, 1e-6)).sum(axis=1)
            row.update(jag_n=len(vj), jag_mean=vj.mean(), jag_median=np.median(vj), jag_sd=vj.std(ddof=1))
        else:
            row.update(jag_n=0, jag_mean=np.nan, jag_median=np.nan, jag_sd=np.nan)
        row["_hist"] = np.histogram(np.clip(v, HIST_EDGES[0], HIST_EDGES[-1] - 1e-6), HIST_EDGES)[0].astype(np.uint16)
        row["_h2"] = np.histogram2d(np.clip(t[keep], AX_EDGES[0], AX_EDGES[-1] - 1e-6),
                                    np.clip(v, HIST_EDGES[0], HIST_EDGES[-1] - 1e-6), [AX_EDGES, HIST_EDGES])[0]
        rows.append(row)
        b = np.floor((t[keep] + extend) / bin_w).astype(int)
        for k in np.unique(b):
            prof.append((labels[(g, l)], round(-extend + (k + 0.5) * bin_w, 3), float(v[b == k].mean())))
    return rows, prof


def run_cavity_esp(universe, glu_residues, asn_residues, results_dir, run, channel_type="G12",
                   frame_duration=0.075, stride=None, extra_frames=(), start_frame=0, end_frame=None,
                   force_rerun=False):
    """One pass over the trajectory -> <results_dir>/cavity_esp/ (cached)."""
    from tqdm import tqdm
    from analysis.converter import convert_to_pdb_numbering
    from analysis.closest_residue_analysis import residue_names_from_topology
    from analysis.site_chain import site_label
    od = Path(results_dir) / "cavity_esp"
    od.mkdir(parents=True, exist_ok=True)
    f_rows, f_prof = od / f"esp_frames_run{run}.csv", od / f"esp_profile_run{run}.csv.gz"
    if f_rows.exists() and f_prof.exists() and not force_rerun:
        print(f"cavity_esp: using {f_rows}")
        return
    pairs = pair_rings(glu_residues, asn_residues)
    resn = residue_names_from_topology(universe, [r for p in pairs for r in p])
    lab1 = {r: site_label(convert_to_pdb_numbering(r, channel_type), resn.get(r, "")) for p in pairs for r in p}
    labels = dict(lab1)
    labels.update({(g, l): f"{lab1[g]}/{lab1[l]}" for g, l in pairs})
    atoms, q, groups = cluster_atoms(universe, pairs)
    with open(od / f"esp_atoms_run{run}.txt", "w") as fh:
        fh.write(f"cluster: {len(atoms)} atoms, total charge {q.sum():+.4f} (ring residues "
                 f"{q[groups != 'caps'].sum():+.4f}, caps {q[groups == 'caps'].sum():+.4f})\n")
        fh.write("cylinders (upper/lower): " + ", ".join(labels[p] for p in pairs) + "\n\n")
        for a_, qq, g in zip(atoms, q, groups):
            fh.write(f"{a_.resid:>6} {a_.resname:<5} {a_.name:<5} {qq:+.4f}  {g}\n")
    n = len(universe.trajectory)
    stride = stride or 1                      # default: every frame
    end = n - 1 if end_frame is None else min(end_frame, n - 1)
    frames = sorted(set(range(start_frame, end + 1, stride)) | {int(f) for f in extra_frames if 0 <= int(f) < n})
    extra = {int(f) for f in extra_frames}
    f_hist = od / f"esp_hist_run{run}.npz"
    f_ax = od / f"esp_axial_run{run}.npz"
    rows, prof, hists = [], [], []
    ax_blocks = {}                                   # (block, cylinder) -> summed 2D histogram
    for f in tqdm(frames, desc=f"cavity_esp RUN{run}", unit="frame"):
        universe.trajectory[f]
        r_, p_ = frame_esp(universe, pairs, atoms, q, groups, labels, jaguar=f in extra)
        for r in r_:
            r.update(run=run, frame=f, time_ns=f * frame_duration, frame_of_interest=f in extra)
            hists.append(r.pop("_hist"))
            h2 = r.pop("_h2")
            k = (f // BLOCK, r["cylinder"])
            ax_blocks[k] = ax_blocks.get(k, 0) + h2
        rows += r_
        prof += [(run, f, c, t, v) for c, t, v in p_]
    cols = ["run", "frame", "time_ns", "frame_of_interest", "cylinder", "upper", "lower", "length_A",
            "n_points", "n_excluded", "mean", "median", "sd", "q1", "q3", "r_ca_upper", "r_cd_upper", "r_o_upper",
            "max_r_o_upper", "jag_n", "jag_mean", "jag_median", "jag_sd"]
    df = pd.DataFrame(rows)
    df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(f_rows, index=False, float_format="%.3f")
    pd.DataFrame(prof, columns=["run", "frame", "cylinder", "t_A", "mean"]).to_csv(f_prof, index=False,
                                                                                    float_format="%.3f")
    np.savez_compressed(f_hist, edges=HIST_EDGES, frame=df["frame"].to_numpy(), cylinder=df["cylinder"].to_numpy().astype(str),
                        counts=np.array(hists))
    print(f"cavity_esp: wrote {f_rows} ({len(frames)} frames, cluster charge {q.sum():+.2f})")
    plot_hist(df, np.array(hists), od / f"esp_hist_run{run}.png", f"{channel_type} RUN{run}")
    cyl = list(dict.fromkeys(df["cylinder"]))
    blocks = sorted({b for b, _ in ax_blocks})
    A = np.zeros((len(blocks), len(cyl), len(AX_EDGES) - 1, len(HIST_EDGES) - 1), np.uint16)   # <= 10 frames x ~2,500 points per bin
    for (b, c), h in ax_blocks.items():
        A[blocks.index(b), cyl.index(c)] = h
    lengths = df.groupby("cylinder")["length_A"].mean()
    np.savez_compressed(f_ax, ax_edges=AX_EDGES, esp_edges=HIST_EDGES, block=np.array(blocks), block_size=BLOCK,
                        cylinder=np.array(cyl), counts=A, length=np.array([lengths[c] for c in cyl]))
    plot_axial({c: A[:, i].sum(axis=0) for i, c in enumerate(cyl)}, {c: lengths[c] for c in cyl},
               od / f"esp_axial_run{run}.png", f"{channel_type} RUN{run}\naxial ESP profile, median and IQR, all frames")
    plot_run(df, pd.DataFrame(prof, columns=["run", "frame", "cylinder", "t_A", "mean"]), od / f"esp_run{run}.png",
             f"{channel_type} RUN{run}")


def hist_stats(counts, edges):
    """Mean and median of the values behind a histogram (bin centres; the median interpolated in its bin)."""
    c = counts.astype(float)
    if c.sum() == 0:
        return np.nan, np.nan
    mid = 0.5 * (edges[:-1] + edges[1:])
    cum = np.cumsum(c) / c.sum()
    k = int(np.searchsorted(cum, 0.5))
    lo = cum[k - 1] if k else 0.0
    med = edges[k] + (0.5 - lo) / max(cum[k] - lo, 1e-12) * (edges[k + 1] - edges[k])
    return float((mid * c).sum() / c.sum()), float(med)


def big_fonts():
    """Large text in every plot (the figures get shrunk into panels)."""
    import matplotlib
    matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": "large", "axes.labelsize": "large",
                                "xtick.labelsize": "medium", "ytick.labelsize": "medium",
                                "legend.fontsize": "medium", "figure.titlesize": "x-large"})


def _auto_xlim(cyl_counts, edges, tail=0.0005, step=25):
    """Range that keeps all but `tail` of the points on each side, over all panels, rounded to `step`."""
    tot = sum(np.asarray(c, float) for c in cyl_counts.values())
    if tot.sum() == 0:
        return (-300, -50)
    cum = np.cumsum(tot) / tot.sum()
    lo = edges[int(np.searchsorted(cum, tail))]
    hi = edges[min(int(np.searchsorted(cum, 1 - tail)) + 1, len(edges) - 1)]
    return (np.floor(lo / step) * step, np.ceil(hi / step) * step)


def plot_hist_panels(cyl_counts, edges, path, title, overlay=None, xlim=None):
    """2 x 2 panels like esp_cylinders.py: % of points per 5 kcal/mol bin, mean and median lines.
    cyl_counts: {cylinder: counts}; overlay: {cylinder: counts} drawn as a black step line (e.g. a QM frame)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    big_fonts()
    names = cyl_order(list(cyl_counts))
    if xlim is None:
        xlim = _auto_xlim(cyl_counts, edges)
    fig, axes = plt.subplots(int(np.ceil(len(names) / 2)), 2, figsize=(11, 3.9 * int(np.ceil(len(names) / 2))), squeeze=False)
    for ax, c in zip(axes.ravel(), names):
        cnt = cyl_counts[c].astype(float)
        frac = 100 * cnt / max(cnt.sum(), 1)
        ax.bar(edges[:-1], frac, width=np.diff(edges), align="edge", color="#3b6ea5", alpha=0.85, edgecolor="white", lw=0.3)
        m, md = hist_stats(cnt, edges)
        ax.axvline(m, color="crimson", ls="--", lw=1.5, label=f"mean {m:.1f}")
        ax.axvline(md, color="darkorange", ls="-.", lw=1.5, label=f"median {md:.1f}")
        if overlay and c in overlay:
            o = overlay[c].astype(float)
            ax.step(edges[:-1], 100 * o / max(o.sum(), 1), where="post", color="#1B2631", lw=1.2, label="frame of interest")
        ax.set_title(c)
        ax.set_xlim(*xlim)
        ax.set_xlabel("ESP (kcal/mol per +1e)")
        ax.set_ylabel("% of points")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hist(df, hists, path, title):
    cyl = list(dict.fromkeys(df["cylinder"]))
    allc = {c: hists[(df["cylinder"] == c).to_numpy()].sum(axis=0) for c in cyl}
    plot_hist_panels(allc, HIST_EDGES, path, f"{title}\nESP over the cylinder points, all frames")


def hist_quantiles(counts, edges, ps=(0.25, 0.5, 0.75)):
    """Quantiles of the values behind a histogram (linear inside the bin); NaN if empty."""
    c = counts.astype(float)
    tot = c.sum()
    if tot == 0:
        return [np.nan] * len(ps)
    cum = np.cumsum(c) / tot
    out = []
    for p in ps:
        k = int(np.searchsorted(cum, p))
        lo = cum[k - 1] if k else 0.0
        out.append(edges[k] + (p - lo) / max(cum[k] - lo, 1e-12) * (edges[k + 1] - edges[k]))
    return out


SUBUNIT_COLORS = {"a": "#3b6ea5", "b": "#D98A1C", "c": "#2E8B57", "d": "#8E6BB0"}
GIRK1_COLOR = "#C0392B"


def cyl_color(name, i=0):
    """Same colour for the same cylinder in every plot: the GIRK1 cylinder (".G1") red, E152 cylinders by subunit letter."""
    head = name.split("/")[0]
    if head.endswith(".G1"):
        return GIRK1_COLOR
    return SUBUNIT_COLORS.get(head[-1:], ["#555555", "#999999"][i % 2])


def cyl_order(names):
    """GIRK1 cylinder first, then the E152 cylinders by subunit letter."""
    return sorted(names, key=lambda n: (not n.split("/")[0].endswith(".G1"), n.split("/")[0][-1:]))


def plot_axial(h2_by_cyl, length_by_cyl, path, title, ylim=(-280, -60), min_points=50):
    """esp_maps.py-style axial profile: median and IQR of all points in each 0.5 A slice, one line per cylinder,
    dashed line at the Glu CA (0), dotted lines at each cylinder's mean Asn/Asp CA position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    big_fonts()
    mid = 0.5 * (AX_EDGES[:-1] + AX_EDGES[1:])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, c in enumerate(cyl_order(h2_by_cyl)):
        h, col = h2_by_cyl[c], cyl_color(c, i)
        q = np.array([hist_quantiles(h[i], HIST_EDGES) if h[i].sum() >= min_points else [np.nan] * 3
                      for i in range(len(mid))])
        L = length_by_cyl[c]
        keep = mid <= L + 3.0 + 1e-9
        ax.plot(mid[keep], q[keep, 1], color=col, lw=2, label=c)
        ax.fill_between(mid[keep], q[keep, 0], q[keep, 2], color=col, alpha=0.18, lw=0)
        ax.axvline(L, color=col, ls=":", lw=1, alpha=0.7)
    ax.axvline(0, color="black", ls="--", lw=1.2)
    ax.set_ylim(*ylim)
    ax.text(0, ylim[1], " Glu CA", va="top", fontsize="medium")
    ax.text(np.mean(list(length_by_cyl.values())), ylim[1], " Asn/Asp CA", va="top", fontsize="medium")
    ax.set_xlabel("position along Glu-CA → Asn/Asp-CA axis (Å)")
    ax.set_ylabel("ESP (kcal/mol per +1e)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_run(df, prof, path, title):
    """One run: the cylinder ESP over time, its distribution, and the axial profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    big_fonts()
    cols = ["#C0392B", "#D98A1C", "#13896B", "#3B7DD8"]
    cyl = list(dict.fromkeys(df["cylinder"]))
    four = df.groupby("frame")[["time_ns", "mean"]].mean()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    win = max(1, len(four) // 100)
    for c, col in zip(cyl, cols):
        g = df[df["cylinder"] == c].sort_values("frame")
        ax[0].plot(g["time_ns"], g["mean"], color=col, lw=0.4, alpha=0.35)
        ax[0].plot(g["time_ns"], g["mean"].rolling(win, center=True, min_periods=1).mean(), color=col, lw=1.8, label=c)
        ax[1].hist(g["mean"], bins=60, histtype="step", lw=1.8, color=col, label=f"{c}: {g['mean'].mean():.1f} ± {g['mean'].std():.1f}")
        p = prof[prof["cylinder"] == c].groupby("t_A")["mean"].agg(["mean", "std"])
        ax[2].plot(p.index, p["mean"], color=col, lw=2, label=c)
        ax[2].fill_between(p.index, p["mean"] - p["std"], p["mean"] + p["std"], color=col, alpha=0.15, lw=0)
    ax[0].plot(four["time_ns"], four["mean"].rolling(win, center=True, min_periods=1).mean(), color="#1B2631", lw=2,
               label=f"four-subunit mean ({four['mean'].mean():.1f})")
    foi = df[df["frame_of_interest"]]
    for f_, g in foi.groupby("frame"):
        ax[0].axvline(g["time_ns"].iloc[0], color="#1B2631", ls=":", lw=1)
        ax[0].text(g["time_ns"].iloc[0], ax[0].get_ylim()[1], f" frame {f_}", fontsize="medium", va="top")
    ax[0].set_xlabel("time (ns)"); ax[0].set_ylabel("cylinder ESP (kcal/mol per +1e)")
    ax[0].set_title("over time\n(thin = every frame, thick = running mean)")
    ax[1].set_xlabel("cylinder ESP (kcal/mol per +1e)"); ax[1].set_ylabel("frames"); ax[1].set_title("distribution over frames")
    ax[2].axvline(0, color="#1B2631", ls="--", lw=1)
    ax[2].set_xlabel("distance from the Glu CA along the cylinder (Å)"); ax[2].set_ylabel("ESP (kcal/mol per +1e)")
    ax[2].set_title("axial profile\n(mean ± SD over frames)")
    for a_ in ax:
        a_.legend(frameon=False)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.suptitle(f"{title}\nclassical cavity ESP, all frames from 0 (equilibration cut only when pooling)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
