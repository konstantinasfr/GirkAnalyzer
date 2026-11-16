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

Frames: every `stride` frames (default: ~0.75 ns) plus any `extra_frames` (frames of interest, e.g. the
QM frames), from frame 0; the equilibration cut is applied when pooling (aggregate_cavity_esp.py).

OUTPUT  <results_dir>/cavity_esp/esp_frames_run<run>.csv     one row per frame and cylinder
        <results_dir>/cavity_esp/esp_profile_run<run>.csv.gz  frame, cylinder, axial bin, mean
        <results_dir>/cavity_esp/esp_atoms_run<run>.txt       the atoms used and their charges
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
    stride = stride or max(1, int(round(0.75 / frame_duration)))
    end = n - 1 if end_frame is None else min(end_frame, n - 1)
    frames = sorted(set(range(start_frame, end + 1, stride)) | {int(f) for f in extra_frames if 0 <= int(f) < n})
    extra = {int(f) for f in extra_frames}
    rows, prof = [], []
    for f in tqdm(frames, desc=f"cavity_esp RUN{run}", unit="frame"):
        universe.trajectory[f]
        r_, p_ = frame_esp(universe, pairs, atoms, q, groups, labels, jaguar=f in extra)
        for r in r_:
            r.update(run=run, frame=f, time_ns=f * frame_duration, frame_of_interest=f in extra)
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
    print(f"cavity_esp: wrote {f_rows} ({len(frames)} frames, cluster charge {q.sum():+.2f})")
