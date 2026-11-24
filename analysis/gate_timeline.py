"""
analysis/gate_timeline.py
=========================
Tests of the "bend and latch" idea for the HBC gate, called from
analysis/bd_jump_residues.py. Output: <minjump folder>/timeline/

    bend   the lower part of TM2 kinks at a glycine hinge and carries the gate
           Phe outward,
    latch  the Phe ring reaches the pocket of the neighbouring subunit and
           T-stacks on its F83 (GIRK1: F72), which holds it there and keeps
           the gate wide.

Per subunit and per frame (timeline_all_frames.csv):
    kink_G189   angle (deg) between the TM2 helix axis above and below the
                lower glycine (pos -3: G189 in GIRK2, G178 in GIRK1); axis =
                first principal axis of 8 CA on each side; 0 = straight
    kink_G180   the same at the upper glycine (pos -12: G180 / GIRK1 G169)
                NOTE: this broad 8+8 measure reads ~18 deg for an ideal straight
                helix and underestimates kinks (20 -> 10 deg): use it only to
                compare states, not as an absolute angle
    bend_G189   local bending angle at the lower glycine, standard HELANAL
                definition: local helix axes of the CA windows h-3..h and
                h..h+3 (bisector method), angle between them; 0 for a straight
                ideal helix, exact for a kink at h
    bend_G180   the same at the upper glycine
    bend_ref    the same at A185 (pos -7, no glycine): noise floor of the local bend
    NOTE: in these runs the TM2 alpha-helix ends at ~195 (local bend 60-100 deg
    at 195-198 in every subunit), 3 residues after the gate Phe; the measures
    below stay inside the helix (up to 194) except "post"
    seg_G189    segment-axis bend at G189: angle between the axes of the
                helical segments 182-188 and 190-194 (7 / 5 CA, the glycine
                left out), each axis fitted through the HELANAL axis points
                (Kink Finder / ProKink style): exact on ideal helices and much
                less noisy than the local bend. _rad = signed part of the bend
                away from (+) / towards (-) the pore axis, _tan = part around
                the pore
    seg_G180    the same at G180 (173-179 vs 181-187)
    bprof<p>    local bend (HELANAL) at every TM2 position p = -16..+6 from the
                gate Phe (GIRK2 176..198): the bend profile along TM2
    tilt_up / tilt_end
                tilt of the segments 182-188 / 189-194 (end of the helix,
                with the gate) relative to the
                pore axis (axis through the HELANAL axis points, pointing down);
                _rad = signed lean away from (+) / towards (-) the pore axis
                going down, _tan = lean around the pore
    post        direction of the chain after the helix end, CA 195 -> CA 199,
                relative to the pore axis (same signs); not a helix
    helix_G189  fraction of the alpha-helix H-bonds O(i)...N(i+4) (N-O <= 3.5 A)
                present for i = 182..190, i.e. up to N(194) (covers every residue the G189
                measures use); low = the helix is broken and the angles mean
                little.  helix_G180: i = 173..187
    r_gate      distance of the gate Phe CA from the pore axis (A)
    chi1        chi1 of the gate Phe (deg, 0-360)
    d_pocket    ring centroid distance gate Phe - pocket Phe (pos -109: F83 /
                GIRK1 F72 of the neighbouring subunit closest on average)
    latch       1 if that pair is pi-stacked (criteria of phe_interactions.py)
    d_pocket186 / latch186   the same with the second docking site of that
                pocket, F186 (pos -6; GIRK1 F175) of the same neighbouring subunit
    in_pocket   1 if pi-stacked with F83 OR F186 of the neighbouring subunit
    d_gate      ring distance to the nearest other gate Phe
    gate_stack  1 if pi-stacked with another gate Phe

1. Timeline around each jump: event<k>_frame<F>_timeline.png (the signals
   from jump - W to jump + W, W = 500 frames, light = raw, dark = running mean
   of 25 frames) and event<k>_frame<F>_order.csv: for every signal x subunit
   the mean before / after, the change, z = change / noise, and the frame at
   which the smoothed trace crosses halfway between the two levels (t_half;
   lag = t_half - jump frame). Sorted by lag: what changes first? Resolution
   ~ +-10 frames because of the smoothing.
2. Correlation over the whole run: correlation.csv / correlation.png. Per
   subunit Spearman correlations kink_G189 vs d_pocket, kink_G189 vs r_gate,
   d_pocket vs the gate's own diagonal, and the diagonal / kink with and
   without the latch.
3. Narrowing should reverse it: latch_events.csv, one row per jump x
   subunit with latch / gate-stack %, kink, chi1, r_gate before and after.
   mover = subunit whose gate CA moves most radially; fits_bend_and_latch =
   widening with the latch gained and the kink increased, or narrowing with
   the latch lost and the kink decreased.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.phe_interactions import PI_PARALLEL, PI_TILTED, PI_T

SEG = 8              # CA per helix segment for the kink angle
SMOOTH = 25          # frames, running mean
SUB_COLOURS = {"A": "#4c72b0", "B": "#dd8452", "C": "#55a868", "D": "#c44e52"}


def _axis(X):
    """X (nf, n, 3) -> unit axis (nf, 3) pointing from the first to the last point."""
    Xc = X - X.mean(1, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    a = vt[:, 0]
    sgn = np.sign(np.einsum("ij,ij->i", a, X[:, -1] - X[:, 0]))
    return a * sgn[:, None]


def local_axes(X):
    """Local helix axes (HELANAL; Sugeta & Miyazawa 1967, Bansal et al. 2000):
    X (nf, n, 3) consecutive CA -> (nf, n-3, 3) unit axes of the windows
    i..i+3, from the cross product of the bisectors at i+1 and i+2, pointing
    from residue i to i+3."""
    A = X[:, :-3] + X[:, 2:-1] - 2 * X[:, 1:-2]
    B = X[:, 1:-2] + X[:, 3:] - 2 * X[:, 2:-1]
    ax = np.cross(A, B)
    ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
    s = np.sign(np.einsum("fij,fij->fi", ax, X[:, 3:] - X[:, :-3]))
    return ax * s[..., None]


def local_bend(ca, rows7):
    """HELANAL local bending angle at the middle residue h of rows7 (CA rows of
    h-3 .. h+3): angle between the local axes of windows h-3..h and h..h+3.
    0 for an ideal straight helix, exactly theta for a helix kinked by theta at h."""
    a = local_axes(ca[:, rows7])
    return np.degrees(np.arccos(np.clip(np.einsum("fi,fi->f", a[:, 0], a[:, 3]), -1, 1)))


def axis_points(X):
    """HELANAL local helix-axis points: X (nf, n, 3) consecutive CA -> points
    (nf, 2*(n-3), 3) on the helix axis (two per window i..i+3, at residues i+1
    and i+2: CA moved along its bisector by the local radius). For an ideal
    helix they lie exactly on the axis."""
    A = X[:, :-3] + X[:, 2:-1] - 2 * X[:, 1:-2]
    B = X[:, 1:-2] + X[:, 3:] - 2 * X[:, 2:-1]
    nA = np.linalg.norm(A, axis=-1, keepdims=True)
    nB = np.linalg.norm(B, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("fij,fij->fi", A, B)[..., None] / (nA * nB)
        r = np.sqrt(nA * nB) / (2 * (1 - cos))
        P1 = X[:, 1:-2] + r * A / nA
        P2 = X[:, 2:-1] + r * B / nB
    return np.concatenate([P1, P2], axis=1)


def segment_axis(X):
    """axis of a helical segment (nf, n, 3), n >= 4: straight line through its
    HELANAL axis points, pointing from the first to the last residue."""
    P = axis_points(X)
    Pc = P - np.nanmean(P, 1, keepdims=True)
    Pc = np.nan_to_num(Pc)
    _, _, vt = np.linalg.svd(Pc, full_matrices=False)
    a = vt[:, 0]
    s = np.sign(np.einsum("ij,ij->i", a, X[:, -1] - X[:, 0]))
    return a * s[:, None]


def segment_bend(ca, rows_up, rows_down, hinge_xyz=None, axis=None):
    """Bend between two helical segments (Kink Finder / ProKink style): angle
    between their axes, each fitted through HELANAL axis points (exact on ideal
    helices, ~2 turns per side so much less noisy than the local bend). With
    hinge_xyz (nf, 3) and the pore axis (point, unit direction), also the signed
    parts of the bend: radial (+ = the lower segment bends away from the pore
    axis, - = towards it) and tangential (around the pore, right-handed about
    the pore axis pointing to the filter).
    Returns (total, radial, tangential) in degrees."""
    u = segment_axis(ca[:, rows_up])
    d = segment_axis(ca[:, rows_down])
    tot = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u, d), -1, 1)))
    if hinge_xyz is None or axis is None:
        return tot, None, None
    c0, a0 = axis
    # outward direction: from the pore axis to the helix axis of the upper segment
    # (the CA of the glycine sits 2.3 A off the helix axis, which would mix the parts)
    v = np.nanmean(axis_points(ca[:, rows_up]), 1) - c0
    rad = v - np.outer(v @ a0, a0)                          # outward, perpendicular to the pore
    rad = rad - np.einsum("ij,ij->i", rad, u)[:, None] * u  # perpendicular to the upper axis
    rad /= np.linalg.norm(rad, axis=1, keepdims=True)
    tan = np.cross(a0[None], rad)                           # around the pore, right-handed
    tan -= np.einsum("ij,ij->i", tan, u)[:, None] * u
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    du = np.einsum("ij,ij->i", d, u)
    r_ = np.degrees(np.arctan2(np.einsum("ij,ij->i", d, rad), du))
    t_ = np.degrees(np.arctan2(np.einsum("ij,ij->i", d, tan), du))
    return tot, r_, t_


def segment_tilt(ca, rows, axis):
    """Tilt of a helical segment relative to the pore axis. The segment axis
    (through its HELANAL axis points, pointing N -> C, i.e. down towards the
    cytoplasm) is compared with the pore axis pointing the same way (chosen
    from the average over all frames, so tilts are 0-90 deg). Returns (total,
    radial, tangential) in degrees: radial + = the segment leans away from the
    pore axis going down (its lower end further out), - = towards it;
    tangential = lean around the pore (right-handed about the axis pointing to
    the filter). Needs >= 4 consecutive CA."""
    c0, a0 = axis
    u = segment_axis(ca[:, rows])
    # reference = the pore-axis direction along which the segment runs (N -> C), chosen once
    # from the average so it does not depend on which end of the filter the pore axis points to
    down = a0 if np.nanmean(u @ a0) > 0 else -a0
    v = np.nanmean(axis_points(ca[:, rows]), 1) - c0
    rad = v - np.outer(v @ a0, a0)
    rad /= np.linalg.norm(rad, axis=1, keepdims=True)
    tan = np.cross(a0[None], rad)
    ud = u @ down
    tot = np.degrees(np.arccos(np.clip(ud, -1, 1)))
    r_ = np.degrees(np.arctan2(np.einsum("ij,ij->i", u, rad), ud))
    t_ = np.degrees(np.arctan2(np.einsum("ij,ij->i", u, tan), ud))
    return tot, r_, t_


PROFILE = range(-16, 7)            # local bend at every TM2 position 176..198 (GIRK2 numbering)
TILT_SEGMENTS = (("up", -10, -4), ("end", -3, 2))   # 182-188, 189-194 (helix end with the gate)
POST = (3, 7)                                        # CA 195 -> 199: chain after the helix end


def vector_tilt(p_from, p_to, axis):
    """direction p_from -> p_to (nf, 3) relative to the pore axis, same sign
    conventions as segment_tilt: (total, radial: + = away from the pore going
    down, tangential) in degrees."""
    c0, a0 = axis
    u = p_to - p_from
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    down = a0 if np.nanmean(u @ a0) > 0 else -a0
    v = 0.5 * (p_from + p_to) - c0
    rad = v - np.outer(v @ a0, a0)
    rad /= np.linalg.norm(rad, axis=1, keepdims=True)
    tan = np.cross(a0[None], rad)
    ud = u @ down
    return (np.degrees(np.arccos(np.clip(ud, -1, 1))),
            np.degrees(np.arctan2(np.einsum("ij,ij->i", u, rad), ud)),
            np.degrees(np.arctan2(np.einsum("ij,ij->i", u, tan), ud)))


class HelixTracker:
    """Per frame, for every gate subunit, which backbone alpha-helix H-bonds
    O(i) ... N(i+4) (N-O <= 3.5 A) are present for i at positions lo..hi from
    the gate Phe (resid offsets in the same chain). update(i) once per frame."""

    def __init__(self, residues, gate_resids, nf, lo=-21, hi=3, cut=3.5):
        self.cut, self.lo, self.hi = cut, lo, hi
        self.pairs = []                      # per gate: (O atoms, N atoms, positions)
        for gr in gate_resids:
            O, N, pos = [], [], []
            for p in range(lo, hi + 1):
                o = residues.atoms.select_atoms(f"resid {gr + p} and name O")
                n = residues.atoms.select_atoms(f"resid {gr + p + 4} and name N")
                if len(o) == 1 and len(n) == 1:
                    O.append(o[0].index)
                    N.append(n[0].index)
                    pos.append(p)
            self.pairs.append((np.array(O, int), np.array(N, int), np.array(pos, int)))
        self.u = residues.universe
        self.hb = [np.zeros((nf, len(pz)), bool) for _, _, pz in self.pairs]

    def update(self, i):
        X = self.u.atoms.positions
        for k, (O, N, _) in enumerate(self.pairs):
            if len(O):
                self.hb[k][i] = np.linalg.norm(X[O] - X[N], axis=1) <= self.cut

    def fraction(self, k, lo, hi):
        """per-frame fraction of the H-bonds O(i)...N(i+4) present, lo <= i <= hi."""
        O, N, pos = self.pairs[k]
        m = (pos >= lo) & (pos <= hi)
        if not m.any():
            return None
        return self.hb[k][:, m].mean(1)


def kink_angle(ca, rows_above, rows_below):
    a, b = _axis(ca[:, rows_above]), _axis(ca[:, rows_below])
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1, 1)))


def _pi_ok(d, ang):
    return (((d <= PI_PARALLEL) & (ang < 30)) | ((d <= PI_TILTED) & (ang >= 30) & (ang <= 60))
            | ((d <= PI_T) & (ang > 60)))


def gate_signals(traj, info, axis, pocket_pos=-109):
    """Per-frame signals of every gate (see module doc). Returns (df, gates)
    where gates = list of dicts (subunit, gate pdb, info row, pocket name)."""
    ca = traj["ca"]
    nf = ca.shape[0]
    rows = {(c, int(p)): i for i, (c, p) in enumerate(zip(info.chain, info.pos)) if np.isfinite(p)}
    rings = traj.get("rings")
    c0, a0 = axis
    out = {}
    gates = []
    gate_rows = list(traj.get("gate_idx") or [])
    for gi in gate_rows:
        g = info.iloc[gi]
        s = str(g.subunit)
        ent = dict(subunit=s, gate=g.pdb, row=gi, pocket="")
        for name, hinge in (("kink_G189", -3), ("kink_G180", -12)):
            above = [rows.get((g.chain, hinge - k)) for k in range(SEG, 0, -1)]
            below = [rows.get((g.chain, hinge + k)) for k in range(1, SEG + 1)]
            if None in above or None in below:
                out[f"{name}_{s}"] = np.full(nf, np.nan)
            else:
                out[f"{name}_{s}"] = kink_angle(ca, above, below)
        for name, hinge in (("bend_G189", -3), ("bend_G180", -12), ("bend_ref", -7)):
            r7 = [rows.get((g.chain, hinge + k)) for k in range(-3, 4)]
            out[f"{name}_{s}"] = (np.full(nf, np.nan) if None in r7 else local_bend(ca, r7))
        # segment-axis bends (7 CA per side) with signed radial / tangential parts
        # (the TM2 helix ends at ~195, right after the gate: the lower side of G189 stops at 194)
        for name, up, down, hinge in (("seg_G189", (-10, -4), (-2, 2), -3),
                                      ("seg_G180", (-19, -13), (-11, -5), -12)):
            ru = [rows.get((g.chain, p)) for p in range(up[0], up[1] + 1)]
            rd = [rows.get((g.chain, p)) for p in range(down[0], down[1] + 1)]
            rh = rows.get((g.chain, hinge))
            if None in ru or None in rd or rh is None:
                for suf in ("", "_rad", "_tan"):
                    out[f"{name}{suf}_{s}"] = np.full(nf, np.nan)
                continue
            tot, rad, tan = segment_bend(ca, ru, rd, ca[:, rh], axis)
            out[f"{name}_{s}"], out[f"{name}_rad_{s}"], out[f"{name}_tan_{s}"] = tot, rad, tan
        # bend profile along TM2 and tilt of three segments relative to the pore axis
        for p_ in PROFILE:
            r7 = [rows.get((g.chain, p_ + k)) for k in range(-3, 4)]
            out[f"bprof{p_:+d}_{s}"] = (np.full(nf, np.nan) if None in r7 else local_bend(ca, r7))
        for name, lo, hi in TILT_SEGMENTS:
            rr = [rows.get((g.chain, p_)) for p_ in range(lo, hi + 1)]
            if None in rr:
                for suf in ("", "_rad", "_tan"):
                    out[f"tilt_{name}{suf}_{s}"] = np.full(nf, np.nan)
                continue
            tot, rad, tan = segment_tilt(ca, rr, axis)
            out[f"tilt_{name}_{s}"], out[f"tilt_{name}_rad_{s}"], out[f"tilt_{name}_tan_{s}"] = tot, rad, tan
        # direction of the chain after the helix end (CA 195 -> CA 199), not a helix
        r0, r1 = rows.get((g.chain, POST[0])), rows.get((g.chain, POST[1]))
        if r0 is None or r1 is None:
            for suf in ("", "_rad", "_tan"):
                out[f"post{suf}_{s}"] = np.full(nf, np.nan)
        else:
            tot, rad, tan = vector_tilt(ca[:, r0], ca[:, r1], axis)
            out[f"post_{s}"], out[f"post_rad_{s}"], out[f"post_tan_{s}"] = tot, rad, tan
        ht = traj.get("helix")
        if ht is not None:
            k_ = gate_rows.index(gi)
            for name, lo, hi in (("helix_G189", -10, -2), ("helix_G180", -19, -5)):
                fr_ = ht.fraction(k_, lo, hi)
                if fr_ is not None:
                    out[f"{name}_{s}"] = fr_
        v = ca[:, gi] - c0
        out[f"r_gate_{s}"] = np.linalg.norm(v - np.outer(v @ a0, a0), axis=1)
        gates.append(ent)
    if rings is not None and getattr(rings, "ready", False):
        A = rings.arom
        gate_rows_rings = [ri for _, ri, _ in rings.gates]
        for g_, ri in enumerate(gate_rows_rings):
            ent = next((e for e in gates if e["row"] == ri), None)
            if ent is None:
                continue
            s, gch = ent["subunit"], info.chain.iloc[ri]
            out[f"chi1_{s}"] = np.mod(rings.chi1[:, g_], 360)
            part = info.iloc[A["res"]] if len(A["res"]) else info.iloc[[]]
            pk = [j for j in range(len(A["res"]))
                  if part.pos.iloc[j] == pocket_pos and part.chain.iloc[j] != gch]
            if pk:
                j = min(pk, key=lambda j_: np.nanmedian(A["d"][:, g_, j_]))
                ent["pocket"] = part.pdb.iloc[j]
                d, ang = A["d"][:, g_, j], A["ang"][:, g_, j]
                out[f"d_pocket_{s}"] = d
                ok83 = _pi_ok(d, ang)
                out[f"latch_{s}"] = ok83.astype(float)
                # second docking site of the same pocket: F186 (pos -6) of the same subunit
                pch = part.chain.iloc[j]
                p186 = [k for k in range(len(A["res"]))
                        if part.pos.iloc[k] == pocket_pos + 103 and part.chain.iloc[k] == pch]
                if p186:
                    k = p186[0]
                    ent["pocket186"] = part.pdb.iloc[k]
                    d2, ang2 = A["d"][:, g_, k], A["ang"][:, g_, k]
                    ok186 = _pi_ok(d2, ang2)
                    out[f"d_pocket186_{s}"] = d2
                    out[f"latch186_{s}"] = ok186.astype(float)
                    out[f"in_pocket_{s}"] = (ok83 | ok186).astype(float)
            og = [j for j in range(len(A["res"])) if A["res"][j] in gate_rows_rings
                  and A["res"][j] != ri]
            if og:
                d = A["d"][:, g_, og]
                ok = _pi_ok(d, A["ang"][:, g_, og])
                out[f"d_gate_{s}"] = np.nanmin(d, 1)
                out[f"gate_stack_{s}"] = ok.any(1).astype(float)
    df = pd.DataFrame(out)
    df.insert(0, "frame", traj["frames"])
    return df, gates


SIGNALS = [("bend_G189", "local bend at G189, HELANAL (deg)"),
           ("seg_G189_rad", "segment bend G189, outward part (deg)"),
           ("tilt_up_rad", "upper TM2 182-188, lean out (deg)"),
           ("bend_G180", "local bend at G180, HELANAL (deg)"),
           ("r_gate", "gate Phe CA from pore axis (Å)"), ("chi1", "gate Phe χ1 (deg)"),
           ("d_pocket", "ring distance to pocket F83 (Å)"),
           ("d_gate", "ring distance to nearest gate Phe (Å)")]


def _smooth(x):
    return pd.Series(x).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()


def order_table(sig, gates, j, b, a, frames, bd, lo, hi):
    """Per signal x subunit: before / after means, z, and the halfway crossing."""
    rows = []
    series = [("B_D distance", "-", bd)] + [
        (lab, e["subunit"], sig[f"{key}_{e['subunit']}"].to_numpy())
        for key, lab in SIGNALS + [("latch", "latch (pi-stack with pocket F83)"),
                                   ("gate_stack", "pi-stack with a gate Phe")]
        for e in gates if f"{key}_{e['subunit']}" in sig]
    for lab, s, x in series:
        x = np.asarray(x, float)
        if np.all(np.isnan(x[b])) or np.all(np.isnan(x[a])):
            continue
        mb, ma = np.nanmean(x[b]), np.nanmean(x[a])
        sd = np.sqrt((np.nanvar(x[b]) + np.nanvar(x[a])) / 2)
        z = abs(ma - mb) / sd if sd > 0 else np.inf
        sm = _smooth(x[lo:hi])
        mid = (mb + ma) / 2
        sgn = np.sign(sm - mid)
        cross = np.flatnonzero(np.diff(sgn) != 0) + lo
        t = cross[np.argmin(np.abs(cross - j))] if len(cross) else None
        gate = next((e["gate"] for e in gates if e["subunit"] == s), "")
        rows.append(dict(signal=lab, subunit=s, gate=gate, before=mb, after=ma, change=ma - mb,
                         z=z, changes="yes" if z >= 1 else "no",
                         t_half=int(frames[t]) if t is not None else np.nan,
                         lag=int(frames[t] - frames[j]) if t is not None else np.nan))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_k"] = (df.changes != "yes").astype(int)
    return df.sort_values(["_k", "lag"]).drop(columns="_k").reset_index(drop=True)


def plot_timeline(sig, gates, bd, ac, frames, j, lo, hi, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k, _ in SIGNALS if any(f"{k}_{e['subunit']}" in sig for e in gates)]
    fig, axes = plt.subplots(len(keys) + 1, 1, figsize=(14, 2.6 * (len(keys) + 1)), sharex=True)
    f = frames[lo:hi]
    ax = axes[0]
    for x, lab, c in ((bd, "B_D", "black"), (ac, "A_C", "0.55")):
        ax.plot(f, x[lo:hi], color=c, alpha=0.25, lw=0.6)
        ax.plot(f, _smooth(x[lo:hi]), color=c, lw=1.8, label=lab)
    ax.set_ylabel("HBC diagonal\nCα-Cα (Å)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    for ax, key in zip(axes[1:], keys):
        lab = dict(SIGNALS)[key]
        for e in gates:
            col = f"{key}_{e['subunit']}"
            if col not in sig:
                continue
            x = sig[col].to_numpy()[lo:hi]
            c = SUB_COLOURS.get(e["subunit"], "0.3")
            ax.plot(f, x, color=c, alpha=0.2, lw=0.5)
            name = e["gate"] + (f" → {e['pocket']}" if key == "d_pocket" and e["pocket"] else "")
            ax.plot(f, _smooth(x), color=c, lw=1.8, label=name)
        if key in ("d_pocket", "d_gate"):
            ax.axhline(PI_T, color="0.4", ls=":", lw=1)
        ax.set_ylabel(lab.replace(" (", "\n("), fontsize=12)
        ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=4)
    for ax in axes:
        ax.axvline(frames[j], color="purple", ls="--", lw=1.2)
        ax.tick_params(labelsize=11)
        ax.grid(color="0.92", lw=0.6)
    axes[-1].set_xlabel("frame", fontsize=12)
    fig.suptitle(title + f"\nlight = per frame, dark = running mean of {SMOOTH} frames; "
                 f"purple = jump; dotted = π-stack limit {PI_T:g} Å", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def correlation(sig, gates, bd, ac, diag_of):
    rows = []
    for e in gates:
        s = e["subunit"]
        diag = bd if diag_of.get(s) == "B_D" else ac
        r = dict(subunit=s, gate=e["gate"], pocket=e["pocket"], diagonal=diag_of.get(s, ""))
        k, dp, rg = (sig.get(f"kink_G189_{s}"), sig.get(f"d_pocket_{s}"), sig.get(f"r_gate_{s}"))
        lt = sig.get(f"latch_{s}")
        D = pd.Series(diag)
        if k is not None and dp is not None:
            r["spearman_kinkG189_vs_dpocket"] = k.corr(dp, method="spearman")
        bnd = sig.get(f"bend_G189_{s}")
        if bnd is not None and dp is not None:
            r["spearman_bendG189_vs_dpocket"] = bnd.corr(dp, method="spearman")
        if k is not None and rg is not None:
            r["spearman_kinkG189_vs_rgate"] = k.corr(rg, method="spearman")
        if dp is not None:
            r["spearman_dpocket_vs_diagonal"] = dp.corr(D, method="spearman")
        if lt is not None:
            on = lt.to_numpy() > 0.5
            r["latched_pct"] = 100 * on.mean()
            r["diagonal_latched"] = D[on].mean() if on.any() else np.nan
            r["diagonal_unlatched"] = D[~on].mean() if (~on).any() else np.nan
            if k is not None:
                r["kinkG189_latched"] = k[on].mean() if on.any() else np.nan
                r["kinkG189_unlatched"] = k[~on].mean() if (~on).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def plot_correlation(sig, gates, bd, ac, diag_of, corr, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(gates)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 9), squeeze=False)
    for i, e in enumerate(gates):
        s = e["subunit"]
        k, dp, lt = sig.get(f"kink_G189_{s}"), sig.get(f"d_pocket_{s}"), sig.get(f"latch_{s}")
        ax = axes[0, i]
        if k is not None and dp is not None and k.notna().any() and dp.notna().any():
            ax.hexbin(k, dp, gridsize=40, mincnt=1, cmap="viridis", bins="log")
            rr = corr.loc[corr.subunit == s, "spearman_kinkG189_vs_dpocket"]
            ax.set_title(f"{e['gate']} → {e['pocket']}\nSpearman r = {rr.iloc[0]:.2f}", fontsize=12)
            ax.axhline(PI_T, color="w", ls=":", lw=1)
        ax.set_xlabel("kink at G189 (deg)", fontsize=11)
        ax.set_ylabel("ring distance to pocket F83 (Å)", fontsize=11)
        ax = axes[1, i]
        diag = bd if diag_of.get(s) == "B_D" else ac
        if lt is not None:
            on = lt.to_numpy() > 0.5
            bins = np.linspace(np.nanmin(diag), np.nanmax(diag), 40)
            ax.hist(diag[~on], bins, color="#9aa5b1", alpha=0.8, density=True,
                    label=f"not latched ({100 * (~on).mean():.0f} %)")
            ax.hist(diag[on], bins, color="#2a9d8f", alpha=0.7, density=True,
                    label=f"latched ({100 * on.mean():.0f} %)")
            ax.legend(fontsize=10, frameon=False)
        ax.set_xlabel(f"{diag_of.get(s, '')} Cα-Cα (Å)", fontsize=11)
        ax.set_ylabel("density", fontsize=11)
    fig.suptitle(title + "\ntop: kink at the lower glycine vs distance of the gate ring to the "
                 "pocket F83 (all frames); bottom: the gate's own diagonal with / without the "
                 "latch (π-stack with F83)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def latch_table(sig, gates, jumps, wins, bd):
    rows = []
    for k, (j, b, a) in wins.items():
        jr = jumps.iloc[k]
        sub = []
        for e in gates:
            s = e["subunit"]
            r = dict(event=k + 1, jump_frame=int(jr.frame), direction=jr.direction,
                     bd_step=float(jr.step), subunit=s, gate=e["gate"], pocket=e["pocket"])
            for key in ("latch", "gate_stack"):
                c = f"{key}_{s}"
                if c in sig:
                    x = sig[c].to_numpy()
                    r[f"{key}_before_pct"] = 100 * np.nanmean(x[b])
                    r[f"{key}_after_pct"] = 100 * np.nanmean(x[a])
            for key in ("kink_G189", "kink_G180", "bend_G189", "bend_G180", "seg_G189",
                        "seg_G189_rad", "tilt_up_rad", "tilt_end_rad", "post_rad", "seg_G180",
                        "helix_G189", "chi1", "r_gate",
                        "d_pocket", "d_gate"):
                c = f"{key}_{s}"
                if c in sig:
                    x = sig[c].to_numpy()
                    r[f"{key}_before"] = np.nanmean(x[b])
                    r[f"{key}_after"] = np.nanmean(x[a])
            sub.append(r)
        if not sub:
            continue
        df = pd.DataFrame(sub)
        if "r_gate_before" in df:
            mv = (df.r_gate_after - df.r_gate_before).abs().idxmax()
            df["mover"] = False
            df.loc[mv, "mover"] = True
        dl = df.get("latch_after_pct", np.nan) - df.get("latch_before_pct", np.nan)
        dk = df.get("kink_G189_after", np.nan) - df.get("kink_G189_before", np.nan)
        wid = df.direction == "widening"
        df["latch_change_pct"] = dl
        df["kink_G189_change"] = dk
        db = df.get("bend_G189_after", np.nan) - df.get("bend_G189_before", np.nan)
        df["bend_G189_change"] = db
        df["fits_bend_and_latch_local"] = np.where((wid & (dl > 0) & (db > 0)) | (~wid & (dl < 0) & (db < 0)),
                                                   "yes", "no")
        df["fits_bend_and_latch"] = np.where((wid & (dl > 0) & (dk > 0)) | (~wid & (dl < 0) & (dk < 0)),
                                             "yes", "no")
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_gate_timeline(traj, info, axis, bd, ac, bounds, jumps, out, hbc_diagonal_pairs, cfg,
                      run_name=""):
    from analysis.bd_jump_residues import event_windows
    tdir = Path(out) / "timeline"
    tdir.mkdir(exist_ok=True)
    frames = traj["frames"]
    sig, gates = gate_signals(traj, info, axis, cfg.get("pocket_pos", -109))
    if not gates:
        return
    allsig = sig.copy()
    allsig.insert(1, "B_D", bd)
    allsig.insert(2, "A_C", ac)
    allsig.to_csv(tdir / "timeline_all_frames.csv", index=False, float_format="%.3f")

    (a1, a2), (b1, b2) = hbc_diagonal_pairs
    diag_of = {}
    for e in gates:
        r = int(info.resid.iloc[e["row"]])
        diag_of[e["subunit"]] = "B_D" if r in (b1, b2) else "A_C"

    # 2. correlation over the whole run
    corr = correlation(sig, gates, bd, ac, diag_of)
    corr.to_csv(tdir / "correlation.csv", index=False, float_format="%.3f")
    plot_correlation(sig, gates, bd, ac, diag_of, corr, tdir / "correlation.png",
                     f"{run_name}  bend and latch over all {len(frames)} frames")
    print(f"\n  [timeline] whole run, per gate (latch = π-stack with the pocket F83):")
    print(corr.round(2).to_string(index=False))

    # 1. timeline around each jump
    W, buf = cfg.get("timeline_window", 500), cfg["buffer"]
    wins = {}
    for k, jr in jumps.reset_index(drop=True).iterrows():
        j = bounds[k + 1]
        b, a = event_windows(bounds, k + 1, "window", dict(buffer=buf, window=W))
        if len(b) < 10 or len(a) < 10:
            continue
        wins[k] = (j, b, a)
        lo, hi = max(0, j - W), min(len(frames), j + W)
        tag = f"event{k + 1}_frame{int(jr.frame)}"
        title = (f"{run_name}  jump at {int(jr.frame)} ({jr.direction}, {jr.step:+.2f} Å)  "
                 f"timeline ±{W} frames")
        plot_timeline(sig, gates, bd, ac, frames, j, lo, hi, tdir / f"{tag}_timeline.png", title)
        od = order_table(sig, gates, j, b, a, frames, bd, lo, hi)
        od.insert(0, "event", k + 1)
        od.insert(1, "jump_frame", int(jr.frame))
        od.insert(2, "direction", jr.direction)
        od.to_csv(tdir / f"{tag}_order.csv", index=False, float_format="%.3f")
        ch = od[od.changes == "yes"]
        print(f"\n  [timeline] jump at {int(jr.frame)} ({jr.direction}): what changes, "
              f"in order (lag = frames from the jump, halfway point):")
        print(ch[["signal", "gate", "before", "after", "z", "lag"]].round(2).to_string(index=False))

    # 3. does narrowing reverse it?
    lt = latch_table(sig, gates, jumps.reset_index(drop=True), wins, bd)
    if not lt.empty:
        lt.to_csv(tdir / "latch_events.csv", index=False, float_format="%.3f")
        mv = lt[lt.get("mover", False) == True]  # noqa: E712
        show = [c for c in ("event", "jump_frame", "direction", "gate", "latch_before_pct",
                            "latch_after_pct", "gate_stack_before_pct", "gate_stack_after_pct",
                            "kink_G189_before", "kink_G189_after", "fits_bend_and_latch")
                if c in mv]
        print("\n  [timeline] gate that moves most at every jump (does narrowing reverse it?):")
        print(mv[show].round(1).to_string(index=False))
    print(f"  -> {tdir}")
