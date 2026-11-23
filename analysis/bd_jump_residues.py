"""
analysis/bd_jump_residues.py
============================
Which residues move when the GIRK1 (B_D) HBC diagonal suddenly widens or
narrows? One run, called from main.py:

    python3 main.py --channel_type G12 --run_number 4 --bd_jump_only
    python3 main.py --channel_type G12 --run_number 4 --bd_jump_only --bd_min_jump 0.8

Results:  <results_dir>/bd_jump_residues/minjump_<X>/   (X = --bd_min_jump, e.g. 1.0)


IDEA
----
1. Find the jumps. The Calpha-Calpha distance of the B_D diagonal (the two
   HBC Phe of hbc_diagonal_pairs[1]; GIRK1 is in B_D by the repo convention)
   is split into flat stretches: the trace is cut where the average level
   before and after differs most, then each part again, until no cut would
   separate two levels that differ by >= min_jump (default 1.0 A) or a part
   would be shorter than min_stretch (150 frames). Neighbouring stretches
   whose levels differ by < min_jump are merged. Every boundary between two
   stretches is a jump (widening or narrowing). The jumps do not depend on
   the gate being open or closed.
2. "Before" and "after" frames of every jump, two ways (both are run):
     window   the `window` frames (default 300) just before / just after the
              jump, never crossing the previous / next jump
     segment  the whole flat stretch before / after the jump (optionally at
              most `segment_max` frames, the ones closest to the jump)
   `buffer` frames (25) on each side of the jump belong to neither.
3. Hold the channel still: every frame is superimposed (least squares, as
   for an RMSD) on frame 0 using the Calpha of the selectivity filter and
   the pore helix of all four subunits (per subunit: lower1 - pore_helix ..
   upper1 of the main.py config; pore_helix = 12 residues). The whole frame
   is moved with that fit, so a residue that moves did move relative to the
   filter, not because the channel drifted or turned. The fit RMSD of every
   frame is saved: the filter must stay rigid for this to make sense.
4. Every residue, every subunit: average Calpha position over the "before"
   frames and over the "after" frames.
       disp          distance between the two average positions (A)
       d_radial      change of the distance from the pore axis
                     (+ = moved OUT, away from the axis; what widens a gate)
       d_axial       change along the pore axis (+ = up, towards the filter)
       d_around      sideways, around the axis (signed)
       sc_disp       the same displacement for the side-chain centre
                     (heavy atoms beyond CA; NaN for Gly). sc_disp >> disp =
                     the side chain turned; sc_disp ~ disp = whole residue moved
       noise         how much the average position moves WITHOUT a jump: the
                     "before" frames split in two halves, and the "after"
                     frames too; mean of the two half-to-half displacements
       ratio         disp / noise. Large = a real move at the jump
       rmsf_before / rmsf_after   positional fluctuation inside each set
   Pore axis: through the centres of the lower1 and upper1 Calpha rings of
   the filter (both lie on the axis by symmetry), pointing up.

OUTPUT, bd_jump_residues/minjump_<X>/
    bd_trace.png             B_D and A_C Calpha-Calpha vs frame, flat
                             stretches and jumps marked
    jumps.csv                every jump: frame, direction, levels, size,
                             size / noise, the A_C change at the same time
    alignment_rmsd.csv       filter + pore-helix fit RMSD per frame
    window<W>/ and segment_<all|maxN>/   one folder per way of choosing frames:
        event<k>_frame<F>_residues.csv   one row per residue (all the above,
                                         plus subunit, GIRK1 flag, and `pos` =
                                         resid - HBC Phe resid of the same
                                         subunit, so equivalent positions of
                                         different subunits line up; HBC = 0)
        event<k>_frame<F>.png            displacement along the sequence, one
                                         row per subunit (whole protein)
        event<k>_frame<F>_TM.png         the same, zoomed on the transmembrane
                                         part (pos -50 .. +15) with its own
                                         y-scale; the gate (pos 0) is outlined
                                         and labelled in every subunit
        event<k>_frame<F>_gate.csv       gate table: the four HBC Phe and the 3
                                         residues on each side (pos -3 .. +3)
        event<k>_frame<F>_ranked_raw.csv every residue, sorted by Calpha
                                         displacement, biggest first, with the
                                         topology resid (raw numbers)
        event<k>_frame<F>_ranked_pdb.csv the same list with PDB numbering, e.g.
                                         E141.G1 / F192.2b (analysis/converter.py,
                                         to_pdb, for this channel_type)
                                         Both lists have `real_move` = yes when the
                                         displacement is >= 3 x its own noise; floppy
                                         loops and chain ends can have a large
                                         displacement that is still only noise
        event<k>_frame<F>_disp.pdb       frame-0 structure with the Calpha
                                         displacement in the B-factor column
                                         (colour by B-factor in PyMOL / VMD)
        all_events_residues.csv          all events, long format
        all_events_gate.csv              the gate tables of all events
        consistent_residues.csv          per residue over all events of the
                                         run: mean disp, mean d_radial with the
                                         sign of narrowing events flipped (so
                                         + = moves out when B_D WIDENS), how
                                         many events it moved in (ratio >= 3)
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIRNAME = "bd_jump_residues"
DEFAULTS = dict(min_jump=1.0, min_stretch=150, buffer=25, window=300, segment_max=0,
                min_frames=100, pore_helix=12, ratio_moved=3.0,
                tm_range=(-50, 15), gate_flank=3)
BACKBONE = ("N", "CA", "C", "O", "OXT")
GATE_SHOW = ["event", "direction", "resid", "resname", "pdb", "subunit", "girk1", "pos", "disp",
             "d_radial", "d_axial", "d_around", "sc_disp", "noise", "ratio", "r_before"]


# --------------------------------------------------------------------------
# entry point (main.py)
# --------------------------------------------------------------------------
def run_bd_jump_residues(u, hbc_diagonal_pairs, sf_lower, sf_upper, results_dir,
                         subunit_groups=None, asn_residues=None, modes=("window", "segment"),
                         channel_type=None, **settings):
    cfg = {**DEFAULTS, **settings}
    out = Path(results_dir) / OUT_DIRNAME / f"minjump_{cfg['min_jump']:.1f}"
    out.mkdir(parents=True, exist_ok=True)
    run_name = Path(results_dir).name
    print(f"\n{'=' * 60}\nB_D JUMP RESIDUE ANALYSIS -> {out}\n{'=' * 60}")

    traj = collect_aligned_positions(u, sf_lower, sf_upper, cfg["pore_helix"])
    info = residue_table(u, traj["residues"], hbc_diagonal_pairs, subunit_groups, asn_residues,
                         channel_type)
    pd.DataFrame(dict(frame=traj["frames"], fit_rmsd=traj["fit_rmsd"])).to_csv(
        out / "alignment_rmsd.csv", index=False, float_format="%.3f")
    print(f"  filter + pore-helix fit RMSD: median {np.median(traj['fit_rmsd']):.2f} A, "
          f"max {traj['fit_rmsd'].max():.2f} A")

    ca = traj["ca"]
    idx = {r: i for i, r in enumerate(info.resid)}
    (a1, a2), (b1, b2) = hbc_diagonal_pairs
    bd = np.linalg.norm(ca[:, idx[b1]] - ca[:, idx[b2]], axis=1)
    ac = np.linalg.norm(ca[:, idx[a1]] - ca[:, idx[a2]], axis=1)

    bounds = flat_stretches(bd, cfg["min_jump"], cfg["min_stretch"])
    jumps = jump_table(bd, ac, bounds, traj["frames"], cfg["buffer"])
    jumps.to_csv(out / "jumps.csv", index=False)
    plot_trace(traj["frames"], bd, ac, bounds, out / "bd_trace.png",
               f"B_D jumps -- {run_name}  (min jump {cfg['min_jump']:.1f} A)")
    print(f"  {len(jumps)} jump(s):")
    if len(jumps):
        print(jumps.to_string(index=False))
    else:
        print("  no jump of B_D Calpha-Calpha >= min_jump: nothing to compare")
        return

    axis = pore_axis(ca, info, sf_lower, sf_upper)
    for mode in modes:
        name = (f"window{cfg['window']}" if mode == "window"
                else f"segment_{'all' if not cfg['segment_max'] else 'max%d' % cfg['segment_max']}")
        mdir = out / name
        mdir.mkdir(exist_ok=True)
        rows, gates = [], []
        for k, j in jumps.iterrows():
            b, a = event_windows(bounds, k + 1, mode, cfg)
            if len(b) < cfg["min_frames"] or len(a) < cfg["min_frames"]:
                print(f"  [{name}] jump at frame {j.frame}: before/after shorter than "
                      f"{cfg['min_frames']} frames -> skipped")
                continue
            res = compare_windows(traj, info, axis, b, a)
            res.insert(0, "event", k + 1)
            res.insert(1, "jump_frame", int(j.frame))
            res.insert(2, "direction", j.direction)
            res.insert(3, "before_frames", f"{traj['frames'][b[0]]}-{traj['frames'][b[-1]]}")
            res.insert(4, "after_frames", f"{traj['frames'][a[0]]}-{traj['frames'][a[-1]]}")
            tag = f"event{k + 1}_frame{int(j.frame)}"
            res.to_csv(mdir / f"{tag}_residues.csv", index=False, float_format="%.3f")
            title = (f"{run_name}  jump at {int(j.frame)} ({j.direction}, {j.step:+.2f} A)  "
                     f"[{name}]  before {res.before_frames.iloc[0]}, "
                     f"after {res.after_frames.iloc[0]}")
            plot_event(res, mdir / f"{tag}.png", title)
            plot_event(res, mdir / f"{tag}_TM.png", title + "  — transmembrane part",
                       pos_range=cfg["tm_range"])
            gate = gate_table(res, cfg["gate_flank"])
            gate.to_csv(mdir / f"{tag}_gate.csv", index=False, float_format="%.3f")
            raw, pdb = ranked_lists(res, cfg["ratio_moved"])
            raw.to_csv(mdir / f"{tag}_ranked_raw.csv", index=False, float_format="%.3f")
            pdb.to_csv(mdir / f"{tag}_ranked_pdb.csv", index=False, float_format="%.3f")
            real = pdb[pdb.real_move == "yes"]
            print(f"\n  [{name}] jump at {int(j.frame)}: {len(real)} residues moved >= "
                  f"{cfg['ratio_moved']:g} x noise; the 20 biggest:")
            print(real.head(20)[["rank", "residue", "subunit", "disp", "d_radial",
                                 "sc_disp", "ratio"]].round(2).to_string(index=False))
            write_disp_pdb(u, traj, info, res, mdir / f"{tag}_disp.pdb")
            rows.append(res)
            gates.append(gate)
        if not rows:
            continue
        allev = pd.concat(rows, ignore_index=True)
        allev.to_csv(mdir / "all_events_residues.csv", index=False, float_format="%.3f")
        allgate = pd.concat(gates, ignore_index=True)
        allgate.to_csv(mdir / "all_events_gate.csv", index=False, float_format="%.3f")
        print(f"\n  [{name}] the four HBC gate Phe at every jump:")
        print(allgate[allgate.pos == 0][GATE_SHOW].round(2).to_string(index=False))
        cons = consistent_residues(allev, cfg["ratio_moved"])
        cons.to_csv(mdir / "consistent_residues.csv", index=False, float_format="%.3f")
        print(f"\n  [{name}] residues that move most (ratio >= {cfg['ratio_moved']:g} in most events):")
        print(cons.head(15).to_string(index=False))
        print(f"  -> {mdir}")


# --------------------------------------------------------------------------
# trajectory pass: align on filter + pore helix, keep CA and side-chain centres
# --------------------------------------------------------------------------
def kabsch(P, Q):
    """Rotation R (3x3) minimising |(P - Pc) R - (Q - Qc)|; P, Q (n, 3)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    return V @ D @ Wt


def alignment_selection(u, sf_lower, sf_upper, pore_helix):
    parts = [f"(resid {lo - pore_helix}:{hi} and name CA)" for lo, hi in zip(sf_lower, sf_upper)]
    sel = u.select_atoms(" or ".join(parts))
    if len(sel) < 12:
        raise ValueError(f"alignment selection has only {len(sel)} CA atoms")
    return sel


def collect_aligned_positions(u, sf_lower, sf_upper, pore_helix=12):
    """One pass over the trajectory. Returns frames, fit_rmsd, residues
    (MDAnalysis ResidueGroup of the protein), ca (nf, nres, 3) and sc
    (nf, nres, 3; NaN where a residue has no side chain), all in the frame-0
    coordinate system of the alignment selection."""
    prot = u.select_atoms("protein")
    residues = prot.residues
    ca_ag = residues.atoms.select_atoms("name CA")
    if len(ca_ag) != len(residues):
        keep = {a.resindex for a in ca_ag}
        residues = residues[[r.resindex in keep for r in residues]]
        ca_ag = residues.atoms.select_atoms("name CA")
    resmap = {r.resindex: i for i, r in enumerate(residues)}
    sc_ag = residues.atoms.select_atoms(
        "not name " + " ".join(BACKBONE) + " and not name H*")
    sc_res = np.array([resmap[a.resindex] for a in sc_ag], int)
    nres = len(residues)
    sc_count = np.bincount(sc_res, minlength=nres).astype(float)

    align = alignment_selection(u, sf_lower, sf_upper, pore_helix)
    nf = len(u.trajectory)
    frames = np.empty(nf, int)
    rmsd = np.empty(nf)
    ca = np.empty((nf, nres, 3), np.float32)
    sc = np.full((nf, nres, 3), np.nan, np.float32)
    ref = None
    for i, ts in enumerate(u.trajectory):
        frames[i] = ts.frame
        P = align.positions.astype(float)
        if ref is None:
            ref = P.copy()
            ref_c = ref.mean(0)
        R = kabsch(P, ref)
        Pc = P.mean(0)
        move = lambda X: (X - Pc) @ R + ref_c
        rmsd[i] = np.sqrt(np.mean(np.sum((move(P) - ref) ** 2, axis=1)))
        ca[i] = move(ca_ag.positions.astype(float))
        if len(sc_ag):
            X = move(sc_ag.positions.astype(float))
            with np.errstate(invalid="ignore", divide="ignore"):
                sc[i] = np.stack([np.bincount(sc_res, X[:, d], nres) for d in range(3)], 1) \
                        / sc_count[:, None]
    sc[:, sc_count == 0] = np.nan
    return dict(frames=frames, fit_rmsd=rmsd, residues=residues, ca=ca, sc=sc)


ONE_LETTER = dict(ALA="A", ARG="R", ASN="N", ASP="D", ASH="D", CYS="C", CYX="C", CYM="C",
                  GLN="Q", GLU="E", GLH="E", GLY="G", HIS="H", HID="H", HIE="H", HIP="H",
                  HSD="H", HSE="H", HSP="H", ILE="I", LEU="L", LYS="K", LYN="K", MET="M",
                  PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")


def pdb_label(resid, resname, channel_type):
    """'E141.G1'-style label from analysis/converter.py; the plain resid if the
    channel type is not known to the converter."""
    try:
        from analysis.converter import to_pdb
        num = to_pdb(int(resid), channel_type) if channel_type else str(resid)
    except Exception:
        num = str(resid)
    return f"{ONE_LETTER.get(resname, 'X')}{num}"


def residue_table(u, residues, hbc_diagonal_pairs, subunit_groups=None, asn_residues=None,
                  channel_type=None):
    """resid, resname, PDB label, subunit letter, GIRK1 flag, pos (resid - HBC
    Phe resid of the same subunit) for every residue."""
    try:
        chain = np.array([r.atoms[0].fragindex for r in residues])
    except Exception:
        segs = np.array([r.segment.segid for r in residues])
        chain = np.unique(segs, return_inverse=True)[1]
    resid = np.array([r.resid for r in residues])
    by_resid = dict(zip(resid, chain))
    chain_to_sub = {}
    for letter, rs in (subunit_groups or {}).items():
        for r in rs:
            if r in by_resid:
                chain_to_sub[by_resid[r]] = letter
    girk1 = set()
    names = {r.resid: r.resname for r in residues}
    for r in asn_residues or []:
        if names.get(r) in ("ASP", "ASH") and r in by_resid:
            girk1.add(by_resid[r])
    hbc_of_chain = {by_resid[r]: r for pair in hbc_diagonal_pairs for r in pair if r in by_resid}
    pos = np.array([rid - hbc_of_chain[c] if c in hbc_of_chain else np.nan
                    for rid, c in zip(resid, chain)])
    sub = [chain_to_sub.get(c, f"chain{c}") for c in chain]
    rn = [r.resname for r in residues]
    return pd.DataFrame(dict(resid=resid, resname=rn,
                             pdb=[pdb_label(i, n, channel_type) for i, n in zip(resid, rn)],
                             chain=chain, subunit=sub, girk1=[c in girk1 for c in chain], pos=pos))


# --------------------------------------------------------------------------
# jumps
# --------------------------------------------------------------------------
def flat_stretches(x, min_jump=1.0, min_stretch=150):
    """Boundaries [0, b1, ..., len(x)] of flat stretches (binary segmentation
    on the mean, then merging of neighbours closer than min_jump)."""
    x = np.asarray(x, float)

    def best_split(lo, hi):
        c = np.cumsum(np.r_[0.0, x[lo:hi]])
        n, tot = hi - lo, c[-1]
        k = np.arange(min_stretch, n - min_stretch + 1)
        if not len(k):
            return None
        score = k * (n - k) / n * (c[k] / k - (tot - c[k]) / (n - k)) ** 2
        return lo + int(k[np.argmax(score)])

    cuts, stack = [], [(0, len(x))]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2 * min_stretch:
            continue
        k = best_split(lo, hi)
        if k is None or abs(x[lo:k].mean() - x[k:hi].mean()) < min_jump:
            continue
        cuts.append(k)
        stack += [(lo, k), (k, hi)]
    b = [0] + sorted(cuts) + [len(x)]
    merged = True
    while merged:
        merged = False
        steps = [abs(x[b[i - 1]:b[i]].mean() - x[b[i]:b[i + 1]].mean())
                 for i in range(1, len(b) - 1)]
        if steps and min(steps) < min_jump:
            del b[1 + int(np.argmin(steps))]
            merged = True
    return b


def jump_table(bd, ac, bounds, frames, buffer):
    rows = []
    for i in range(1, len(bounds) - 1):
        lo, j, hi = bounds[i - 1], bounds[i], bounds[i + 1]
        b, a = bd[lo:max(lo + 1, j - buffer)], bd[min(hi - 1, j + buffer):hi]
        step = a.mean() - b.mean()
        rows.append(dict(event=i, frame=int(frames[j]),
                         direction="widening" if step > 0 else "narrowing",
                         bd_before=round(b.mean(), 3), bd_after=round(a.mean(), 3),
                         step=round(step, 3),
                         step_over_noise=round(abs(step) / np.sqrt((b.var() + a.var()) / 2), 2),
                         ac_step=round(ac[min(hi - 1, j + buffer):hi].mean()
                                       - ac[lo:max(lo + 1, j - buffer)].mean(), 3),
                         stretch_before=f"{frames[lo]}-{frames[j - 1]}",
                         stretch_after=f"{frames[j]}-{frames[hi - 1]}"))
    return pd.DataFrame(rows)


def event_windows(bounds, i, mode, cfg):
    """Row indices of the before / after frames of the jump at bounds[i]."""
    lo, j, hi = bounds[i - 1], bounds[i], bounds[i + 1]
    buf = cfg["buffer"]
    b_lo, b_hi = (lo + buf if i > 1 else lo), j - buf     # never into the previous jump
    a_lo, a_hi = j + buf, (hi - buf if i < len(bounds) - 2 else hi)
    if mode == "window":
        b_lo, a_hi = max(b_lo, b_hi - cfg["window"]), min(a_hi, a_lo + cfg["window"])
    elif cfg.get("segment_max"):
        b_lo, a_hi = max(b_lo, b_hi - cfg["segment_max"]), min(a_hi, a_lo + cfg["segment_max"])
    return np.arange(b_lo, max(b_lo, b_hi)), np.arange(a_lo, max(a_lo, a_hi))


# --------------------------------------------------------------------------
# residue comparison
# --------------------------------------------------------------------------
def pore_axis(ca, info, sf_lower, sf_upper):
    """Point and unit direction (pointing up, to the filter) of the pore axis,
    from the mean positions of the lower1 / upper1 filter CA rings."""
    idx = {r: i for i, r in enumerate(info.resid)}
    mean = ca.mean(axis=0)
    low = mean[[idx[r] for r in sf_lower]].mean(0)
    up = mean[[idx[r] for r in sf_upper]].mean(0)
    d = up - low
    return low, d / np.linalg.norm(d)


def _radial(p, c, a):
    v = p - c
    perp = v - np.outer(v @ a, a)
    return perp, np.linalg.norm(perp, axis=1)


def compare_windows(traj, info, axis, b, a):
    c, ax = axis
    ca, sc = traj["ca"], traj["sc"]
    pb, pa = ca[b].mean(0).astype(float), ca[a].mean(0).astype(float)
    d = pa - pb
    perp_b, rb = _radial(pb, c, ax)
    _, ra = _radial(pa, c, ax)
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = perp_b / rb[:, None]
    that = np.cross(ax, rhat)

    def halves(rows):
        h = len(rows) // 2
        return np.linalg.norm(ca[rows[:h]].mean(0) - ca[rows[h:]].mean(0), axis=1)

    noise = (halves(b) + halves(a)) / 2
    with np.errstate(invalid="ignore"):
        scb, sca = np.nanmean(sc[b], 0), np.nanmean(sc[a], 0)
    disp = np.linalg.norm(d, axis=1)
    rmsf = lambda rows, p: np.sqrt(np.mean(np.sum((ca[rows] - p) ** 2, axis=2), axis=0))
    res = info[["resid", "resname", "pdb", "subunit", "girk1", "pos"]].copy()
    res["disp"] = disp
    res["d_radial"] = ra - rb
    res["d_axial"] = d @ ax
    res["d_around"] = np.sum(d * that, axis=1)
    res["sc_disp"] = np.linalg.norm(sca - scb, axis=1)
    res["noise"] = noise
    with np.errstate(invalid="ignore", divide="ignore"):
        res["ratio"] = disp / noise
    res["rmsf_before"] = rmsf(b, pb)
    res["rmsf_after"] = rmsf(a, pa)
    res["r_before"] = rb
    return res


def gate_table(res, flank=3):
    """The HBC Phe (pos 0) of every subunit and `flank` residues on each side."""
    g = res[res.pos.abs() <= flank]
    cols = ["event", "jump_frame", "direction", "before_frames", "after_frames"] + GATE_SHOW[2:] \
        + ["rmsf_before", "rmsf_after"]
    return g.sort_values(["subunit", "pos"])[cols].reset_index(drop=True)


def ranked_lists(res, ratio_moved=3.0):
    """Every residue sorted by Calpha displacement, biggest first. Returns the
    list with raw topology resids and the same list with PDB labels."""
    r = res.sort_values("disp", ascending=False).reset_index(drop=True)
    r.insert(0, "rank", np.arange(1, len(r) + 1))
    r["real_move"] = np.where(r.ratio >= ratio_moved, "yes", "no")
    common = ["subunit", "girk1", "pos", "disp", "d_radial", "d_axial", "d_around", "sc_disp",
              "noise", "ratio", "real_move", "rmsf_before", "rmsf_after"]
    raw = r[["rank", "resid", "resname"] + common]
    pdb = r[["rank", "pdb"] + common].rename(columns={"pdb": "residue"})
    return raw, pdb


def consistent_residues(allev, ratio_moved=3.0):
    df = allev.copy()
    sign = np.where(df.direction == "widening", 1.0, -1.0)
    df["d_radial_w"] = df.d_radial * sign       # + = moves out when B_D widens
    df["d_axial_w"] = df.d_axial * sign
    df["moved"] = df.ratio >= ratio_moved
    g = df.groupby(["resid", "resname", "pdb", "subunit", "girk1", "pos"], dropna=False)
    out = g.agg(n_events=("event", "nunique"), n_moved=("moved", "sum"),
                mean_disp=("disp", "mean"), mean_ratio=("ratio", "mean"),
                mean_d_radial_widening=("d_radial_w", "mean"),
                mean_d_axial_widening=("d_axial_w", "mean"),
                mean_sc_disp=("sc_disp", "mean")).reset_index()
    out["frac_moved"] = out.n_moved / out.n_events
    return out.sort_values(["frac_moved", "mean_disp"], ascending=False)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def write_disp_pdb(u, traj, info, res, path):
    """Frame-0 structure (already the alignment reference) with the CA
    displacement of each residue as B-factor."""
    try:
        u.trajectory[0]
        prot = traj["residues"].atoms
        if not hasattr(u.atoms, "tempfactors"):
            u.add_TopologyAttr("tempfactors")
        disp = dict(zip(res.resid, res.disp))
        prot.tempfactors = [disp.get(a.resid, 0.0) for a in prot]
        prot.write(str(path))
    except Exception as exc:   # PDB is a convenience; never fail the analysis on it
        print(f"  could not write {path.name}: {exc}")


def plot_trace(frames, bd, ac, bounds, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sm = lambda v: pd.Series(v).rolling(50, center=True, min_periods=1).median()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(frames, bd, color="#ff7f0e", lw=0.3, alpha=0.35)
    ax.plot(frames, sm(bd), color="#ff7f0e", lw=1.4, label="B_D Cα–Cα (GIRK1 diagonal)")
    ax.plot(frames, ac, color="#1f77b4", lw=0.3, alpha=0.35)
    ax.plot(frames, sm(ac), color="#1f77b4", lw=1.4, label="A_C Cα–Cα")
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        ax.hlines(bd[lo:hi].mean(), frames[lo], frames[hi - 1], color="black", lw=2,
                  label="flat stretches (B_D)" if i == 0 else None)
    for i in range(1, len(bounds) - 1):
        ax.axvline(frames[bounds[i]], color="purple", ls="--", lw=1.2,
                   label="jump" if i == 1 else None)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Cα–Cα (Å)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_event(res, out_png, title, pos_range=None):
    """Displacement along the sequence, one row per subunit. With pos_range
    (lo, hi): only that part, its own y-scale, the gate (pos 0) outlined and
    labelled, every residue with ratio >= 3 labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = sorted(res.subunit.unique())
    zoom = pos_range is not None
    fig, axes = plt.subplots(len(subs), 1, figsize=(13, 2.3 * len(subs) + 0.6),
                             sharex=True, sharey=zoom)
    axes = np.atleast_1d(axes)
    if zoom:
        lo, hi = pos_range
        part = res[(res.pos >= lo) & (res.pos <= hi)]
        ymax = 1.25 * max(part.disp.max(), part.sc_disp.max(), part.d_radial.abs().max(), 0.5)
        ymin = min(part.d_radial.min() * 1.25, -0.3)
    for ax, s in zip(axes, subs):
        r = res[res.subunit == s].sort_values("pos")
        if zoom:
            r = r[(r.pos >= lo) & (r.pos <= hi)]
        x = r.pos.to_numpy()
        ax.bar(x, r.d_radial, width=1.0, color=np.where(r.d_radial > 0, "#2a9d8f", "#e76f51"),
               alpha=0.55, label="outward (+) / inward (−)")
        ax.plot(x, r.disp, color="black", lw=0.9, label="Cα displacement")
        ax.plot(x, 3 * r.noise, color="0.55", lw=0.8, ls="--", label="3 × noise")
        ax.scatter(x, r.sc_disp, s=4, color="#6a4c93", label="side-chain displacement", zorder=3)
        ax.axvline(0, color="purple", lw=0.8)
        ax.axhline(0, color="black", lw=0.5)
        lab = s + ("  (GIRK1)" if r.girk1.any() else "")
        ax.set_ylabel(f"{lab}\nÅ", fontsize=9)
        top = r.loc[r.ratio >= 3].nlargest(12 if zoom else 5, "disp")
        for _, t in top.iterrows():
            ax.annotate(t.pdb if "pdb" in t else f"{t.resname}{t.resid}", (t.pos, t.disp), fontsize=6,
                        xytext=(0, 3), textcoords="offset points", ha="center")
        if zoom:
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(lo - 0.5, hi + 0.5)
            g = r[r.pos == 0]
            for _, t in g.iterrows():
                ax.bar([0], [t.d_radial], width=1.0, fill=False, edgecolor="purple", lw=1.5)
                ax.text(0.995, 0.95, f"gate {t.pdb} (resid {t.resid}): {t.disp:.2f} Å, "
                            f"out {t.d_radial:+.2f} Å, ratio {t.ratio:.1f}",
                            transform=ax.transAxes, fontsize=7.5, color="purple",
                            ha="right", va="top",
                            bbox=dict(facecolor="white", edgecolor="purple", lw=0.6, alpha=0.9))
    axes[0].legend(fontsize=7, ncol=4, loc="upper left")
    axes[0].set_title(title, fontsize=9)
    axes[-1].set_xlabel("position relative to the HBC Phe of the same subunit (HBC = 0)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
