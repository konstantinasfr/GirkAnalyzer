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
     window   the `window` frames (default 300 and 500, one folder each) just before / just after the
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
        event<k>_frame<F>_top10_pdb.png  one panel per subunit: the 10 residues
        event<k>_frame<F>_top10_raw.png  with the biggest Calpha displacement
        event<k>_frame<F>_top10_TM_*.png (the _TM versions rank only the
                                         transmembrane part, pos -50 .. +15)
                                         (bar; green = out, red = in; faded =
                                         under 3 x noise), side-chain displacement
                                         (dot) and 3 x noise (grey tick); labels in
                                         PDB numbering or raw resid. The first and
                                         last `top_exclude_ends` (15) residues of
                                         each subunit are left out (floppy ends)
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
        event<k>_frame<F>_neighbours.csv residues near each of the four HBC gate
                                         Phe IN 3D (not by sequence): every residue
                                         with a heavy atom within `neighbour_cutoff`
                                         (8 A) of the gate Phe in the before or the
                                         after frames, from any subunit. Per
                                         neighbour: distance to the gate Phe before
                                         and after (median of the per-frame minimum
                                         heavy-atom distance), the change, % of
                                         frames in contact (< 4.5 A) before / after,
                                         its own displacement at the jump and
                                         real_move. Sorted per gate by how much the
                                         distance changes (biggest first)
        contacts/event<k>_frame<F>_contacts_change_<pdb|raw>.png
        contacts/event<k>_frame<F>_contacts_present_<pdb|raw>.png
                                         2 x 2 grid, one panel per gate Phe. x axis =
                                         residue POSITIONS, the same in every panel
                                         (union over the four gates): GIRK2 numbering
                                         for _pdb (GIRK1 shifted +11, A174.G1 = 185),
                                         position from the HBC Phe for _raw. One bar
                                         per neighbouring residue, coloured by its
                                         subunit: height = % frames in contact (< 4.5
                                         A) AFTER the jump, black line = BEFORE, number
                                         under it = after - before (percentage points)
                                         Selected per gate: within 8 A, >= 5 % contact
                                         before or after, not same subunit within +-2
                                         ('present') / +-1 ('change'); both plots also
                                         >= 10 points change. 'present' = which pocket
                                         each gate Phe sits in
        contacts/event<k>_frame<F>_contacts_<change|present>.csv
                                         the numbers of that plot: one row per gate
                                         Phe x neighbour shown, with position_girk2,
                                         position_from_gate, neighbour (PDB) and resid
                                         (raw), contact_before_pct, contact_after_pct,
                                         change_pct (after - before, percentage
                                         points), distances before / after and the
                                         neighbour's own displacement
        contacts/all_events_contacts_<change|present>.csv   all jumps together
        interactions/event<k>_frame<F>_interactions.csv
                                         chemistry of each gate Phe ring (see
                                         analysis/phe_interactions.py): per gate x
                                         partner x type (pi-pi with geometry
                                         parallel / tilted / T-shaped, S...pi with
                                         Cys/Met sulfur, S-H...pi, CH-pi, cation-pi)
                                         % frames before / after, change, median
                                         distance and angle before / after
        interactions/event<k>_frame<F>_interactions.png
                                         2 x 2 grid, bars before / after of the
                                         interactions that change by >= 10 points
        interactions/all_events_interactions.csv   all jumps together
        all_events_residues.csv          all events, long format
    timeline/                            "bend and latch" tests (analysis/gate_timeline.py):
        timeline_all_frames.csv          per frame: B_D, A_C and per gate the kink at
                                         G189 / G180, r_gate, chi1, ring distance to the
                                         pocket F83, latch (pi-stack), nearest gate ring
        event<k>_frame<F>_timeline.png   those signals +- 500 frames around the jump
        event<k>_frame<F>_order.csv      before / after / z and the halfway frame of
                                         every signal: what changes first?
        correlation.csv / .png           whole run: kink vs pocket distance, diagonal
                                         with / without the latch
        latch_events.csv                 every jump x gate: latch, gate-gate stack, kink
                                         before / after; fits_bend_and_latch
        all_events_neighbours.csv        the neighbour lists of all events
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

from analysis.phe_interactions import plot_interactions

OUT_DIRNAME = "bd_jump_residues"
DEFAULTS = dict(min_jump=1.0, min_stretch=150, buffer=25, window=(300, 500), segment_max=0,
                min_frames=100, pore_helix=12, ratio_moved=3.0,
                tm_range=(-50, 15), gate_flank=3, neighbour_cutoff=8.0, contact_cutoff=4.5,
                contact_sets=(("heavy", 4.5), ("all", 3.5)),
                top_n=10, top_exclude_ends=15, contact_top_n=0, contact_min_pct=5.0,
                contact_min_change=0.0, timeline_window=500, pocket_pos=-109)
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

    gate_resids = [r for pair in hbc_diagonal_pairs for r in pair]
    contact_sets = [(str(a), float(c)) for a, c in cfg["contact_sets"]]
    traj = collect_aligned_positions(u, sf_lower, sf_upper, cfg["pore_helix"], gate_resids,
                                     atom_sets=sorted({a for a, _ in contact_sets}))
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
    from analysis.gate_timeline import run_gate_timeline
    run_gate_timeline(traj, info, axis, bd, ac, bounds, jumps, out, hbc_diagonal_pairs, cfg,
                      run_name)
    windows = cfg["window"] if isinstance(cfg["window"], (list, tuple)) else [cfg["window"]]
    runs = [("window", w) for w in windows if "window" in modes] \
        + ([("segment", None)] if "segment" in modes else [])
    for mode, w in runs:
        mcfg = dict(cfg, window=w) if w else cfg
        name = (f"window{w}" if mode == "window"
                else f"segment_{'all' if not cfg['segment_max'] else 'max%d' % cfg['segment_max']}")
        mdir = out / name
        mdir.mkdir(exist_ok=True)
        sub_of = {cs: mdir / f"contacts_{cs[0]}_{cs[1]:g}" for cs in contact_sets}
        for sd in sub_of.values():
            sd.mkdir(exist_ok=True)
        idir = mdir / "interactions"
        idir.mkdir(exist_ok=True)
        rows, gates, inters = [], [], []
        neighbours = {cs: [] for cs in contact_sets}
        contact_tabs = {cs: {"change": [], "present": []} for cs in contact_sets}
        for k, j in jumps.iterrows():
            b, a = event_windows(bounds, k + 1, mode, mcfg)
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
            for lab in ("pdb", "raw"):
                plot_top_residues(res, mdir / f"{tag}_top{cfg['top_n']}_{lab}.png", title,
                                  label=lab, n=cfg["top_n"],
                                  exclude_ends=cfg["top_exclude_ends"])
                plot_top_residues(res, mdir / f"{tag}_top{cfg['top_n']}_TM_{lab}.png",
                                  title + "  — transmembrane part", label=lab, n=cfg["top_n"],
                                  exclude_ends=0, pos_range=cfg["tm_range"])
            gate = gate_table(res, cfg["gate_flank"])
            gate.to_csv(mdir / f"{tag}_gate.csv", index=False, float_format="%.3f")
            for cs in contact_sets:
                atoms, ccut = cs
                sd = sub_of[cs]
                nb_cs = neighbour_table(traj, info, res, b, a, cfg, atoms, ccut)
                nb_cs.to_csv(sd / f"{tag}_neighbours.csv", index=False, float_format="%.3f")
                neighbours[cs].append(nb_cs)
                ctitle = f"{title}\ncontact = {atoms} atoms closer than {ccut:g} Å"
                for kind in ("change", "present"):
                    sel = select_contacts(nb_cs, kind, cfg["contact_top_n"], cfg["contact_min_pct"],
                                          cfg["contact_min_change"])
                    tab = contacts_table(fill_union(nb_cs, sel, kind))
                    tab.to_csv(sd / f"{tag}_contacts_{kind}.csv", index=False,
                               float_format="%.3f")
                    contact_tabs[cs][kind].append(tab)
                    for lab in ("pdb", "raw"):
                        plot_contacts(nb_cs, sd / f"{tag}_contacts_{kind}_{lab}.png", ctitle,
                                      kind=kind, label=lab, n=cfg["contact_top_n"],
                                      min_pct=cfg["contact_min_pct"], cutoff=ccut,
                                      min_change=cfg["contact_min_change"])
                if cs == contact_sets[0]:
                    nb = nb_cs
            it = traj["rings"].summary(b, a, info) if traj.get("rings") is not None else pd.DataFrame()
            if not it.empty:
                for c_, v_ in (("event", k + 1), ("jump_frame", int(j.frame)),
                               ("direction", j.direction))[::-1]:
                    it.insert(0, c_, v_)
                it.to_csv(idir / f"{tag}_interactions.csv", index=False, float_format="%.3f")
                inters.append(it)
                plot_interactions(it, idir / f"{tag}_interactions.png", title,
                                  cfg["contact_min_pct"], cfg["contact_min_change"])
            mover = gate.loc[gate.pos == 0].sort_values("disp", ascending=False).iloc[0]
            near = nb[nb.gate == mover.pdb]
            print(f"\n  [{name}] jump at {int(j.frame)}: 3D neighbours of the gate Phe that moved "
                  f"most, {mover.pdb} ({mover.disp:.2f} A), biggest distance change first:")
            print(near.head(15)[["neighbour", "subunit", "dist_before", "dist_after", "d_dist",
                                 "contact_before_pct", "contact_after_pct", "disp",
                                 "real_move"]].round(2).to_string(index=False))
            if not it.empty:
                im = it[(it.gate == mover.pdb) & (it[["pct_before", "pct_after"]].max(axis=1) >= 5)]
                print(f"\n  [{name}] chemical interactions of the ring of {mover.pdb}:")
                print(im[["partner", "interaction", "pct_before", "pct_after", "change_pct",
                          "dist_before", "dist_after", "geometry_before", "geometry_after"]]
                      .round(1).to_string(index=False))
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
        if inters:
            pd.concat(inters, ignore_index=True).to_csv(
                idir / "all_events_interactions.csv", index=False, float_format="%.3f")
        allgate = pd.concat(gates, ignore_index=True)
        allgate.to_csv(mdir / "all_events_gate.csv", index=False, float_format="%.3f")
        for cs in contact_sets:
            if neighbours[cs]:
                pd.concat(neighbours[cs], ignore_index=True).to_csv(
                    sub_of[cs] / "all_events_neighbours.csv", index=False, float_format="%.3f")
            for kind, tabs in contact_tabs[cs].items():
                if tabs:
                    pd.concat(tabs, ignore_index=True).to_csv(
                        sub_of[cs] / f"all_events_contacts_{kind}.csv", index=False,
                        float_format="%.3f")
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


def collect_aligned_positions(u, sf_lower, sf_upper, pore_helix=12, gate_resids=(),
                              atom_sets=("heavy",)):
    """One pass over the trajectory. Returns frames, fit_rmsd, residues
    (MDAnalysis ResidueGroup of the protein), ca (nf, nres, 3) and sc
    (nf, nres, 3; NaN where a residue has no side chain), all in the frame-0
    coordinate system of the alignment selection. With gate_resids also
    gate_dist {atom_set: (nf, n_gates, nres)}: per frame the minimum distance
    from each gate residue to every residue (NaN for the gate itself), using
    heavy atoms only ('heavy') or all atoms including hydrogens ('all')."""
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

    # atoms of every residue, grouped by residue, for the gate distances
    gsets = {}
    for aset in atom_sets:
        sel = "not name H*" if aset == "heavy" else "all"
        grp = residues.atoms.select_atoms(sel)
        g_res = np.array([resmap[a.resindex] for a in grp], int)
        order = np.argsort(g_res, kind="stable")
        grp, g_res = grp[order], g_res[order]
        starts = np.r_[0, np.flatnonzero(np.diff(g_res)) + 1]
        gates_ag = []
        for gr in gate_resids:
            ag = residues.atoms.select_atoms(f"resid {gr} and ({sel})")
            if len(ag):
                gates_ag.append(ag)
        gsets[aset] = dict(atoms=grp, starts=starts, present=g_res[starts], gates=gates_ag)
    gate_idx = []
    for gr in gate_resids:
        ag = residues.atoms.select_atoms(f"resid {gr}")
        if len(ag):
            gate_idx.append(resmap[ag[0].resindex])
    gate_dist = ({aset: np.full((nf, len(gate_idx), nres), np.nan, np.float32) for aset in gsets}
                 if gate_idx else None)
    if gate_idx:
        from MDAnalysis.lib.distances import distance_array
    from analysis.phe_interactions import RingTracker
    rings = RingTracker(residues, resmap, gate_resids, nf)
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
        if gate_idx:
            for aset, gs in gsets.items():
                H = gs["atoms"].positions
                for k, ag in enumerate(gs["gates"]):
                    dmin = distance_array(ag.positions, H).min(axis=0)
                    gate_dist[aset][i, k, gs["present"]] = np.minimum.reduceat(dmin, gs["starts"])
                    gate_dist[aset][i, k, gate_idx[k]] = np.nan
        rings.update(i)
        if len(sc_ag):
            X = move(sc_ag.positions.astype(float))
            with np.errstate(invalid="ignore", divide="ignore"):
                sc[i] = np.stack([np.bincount(sc_res, X[:, d], nres) for d in range(3)], 1) \
                        / sc_count[:, None]
    sc[:, sc_count == 0] = np.nan
    return dict(frames=frames, fit_rmsd=rmsd, residues=residues, ca=ca, sc=sc,
                gate_dist=gate_dist, gate_idx=gate_idx, rings=rings)


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
def flat_stretches(x, min_jump=1.0, min_stretch=150, exhaustive=False):
    """Boundaries [0, b1, ..., len(x)] of flat stretches (binary segmentation
    on the mean, then merging of neighbours closer than min_jump).
    exhaustive=False: a part is not split further when its best cut separates
    levels closer than min_jump (fast, but misses level changes hidden inside a
    part whose two halves happen to have similar means, e.g. high - low - high).
    exhaustive=True: every part is split down to min_stretch and the merging
    alone decides which cuts survive."""
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
        if k is None or (not exhaustive and abs(x[lo:k].mean() - x[k:hi].mean()) < min_jump):
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


def neighbour_table(traj, info, res, b, a, cfg, atoms="heavy", contact_cutoff=None):
    """3D neighbours of each gate Phe for one jump (rows b = before, a = after),
    distances from the `atoms` set ('heavy' or 'all'), contact = < contact_cutoff."""
    gd = traj.get("gate_dist")
    if gd is None:
        return pd.DataFrame()
    gd = gd[atoms] if isinstance(gd, dict) else gd
    cut = cfg["neighbour_cutoff"]
    ccut = cfg["contact_cutoff"] if contact_cutoff is None else contact_cutoff
    rows = []
    for k, gi in enumerate(traj["gate_idx"]):
        db, da = gd[b, k, :], gd[a, k, :]
        with np.errstate(invalid="ignore"):
            med_b, med_a = np.nanmedian(db, 0), np.nanmedian(da, 0)
            cb, ca_ = 100 * np.mean(db < ccut, 0), 100 * np.mean(da < ccut, 0)
        near = np.flatnonzero((med_b <= cut) | (med_a <= cut))
        g = info.iloc[gi]
        for j in near:
            n, r = info.iloc[j], res.iloc[j]
            same = n.chain == g.chain
            rows.append(dict(gate=g.pdb, gate_resid=int(g.resid), gate_subunit=g.subunit,
                             neighbour=n.pdb, resid=int(n.resid), resname=n.resname,
                             subunit=n.subunit, girk1=bool(n.girk1), pos=n.pos,
                             same_subunit=bool(same),
                             seq_distance=int(abs(n.resid - g.resid)) if same else np.nan,
                             dist_before=med_b[j], dist_after=med_a[j],
                             d_dist=med_a[j] - med_b[j],
                             contact_before_pct=cb[j], contact_after_pct=ca_[j],
                             disp=r.disp, d_radial=r.d_radial, sc_disp=r.sc_disp,
                             noise=r.noise, ratio=r.ratio,
                             real_move="yes" if r.ratio >= cfg["ratio_moved"] else "no"))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["abs_d_dist"] = df.d_dist.abs()
    df = df.sort_values(["gate_subunit", "abs_d_dist"], ascending=[True, False])
    head = res.iloc[0]
    df.insert(0, "event", head.event)
    df.insert(1, "jump_frame", head.jump_frame)
    df.insert(2, "direction", head.direction)
    return df.drop(columns="abs_d_dist").reset_index(drop=True)


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


HBC_GIRK2 = 192        # GIRK2 numbering of the HBC Phe: position label = 192 + pos


def select_contacts(nb, kind="present", n=0, min_pct=5.0, min_change=0.0):
    """The neighbour rows shown in a contact plot (see plot_contacts), with the
    position columns added: pos (from the HBC Phe of the neighbour's own
    subunit), position_girk2 (192 + pos), change_pct (after - before)."""
    if nb is None or nb.empty:
        return pd.DataFrame()
    d = nb[np.isfinite(nb.pos)].copy()
    d["chg"] = d.contact_after_pct - d.contact_before_pct
    skip = 1 if kind == "change" else 2
    helix = d.same_subunit.astype(bool) & (d.seq_distance <= skip)
    d = d[~helix & ((d.contact_before_pct >= min_pct) | (d.contact_after_pct >= min_pct))]
    if min_change:
        d = d[d.chg.abs() >= min_change]      # optional: only residues whose contact changes
    if kind == "change":
        d["key"] = d.chg.abs()
    else:
        d["key"] = d[["contact_before_pct", "contact_after_pct"]].max(axis=1)
    if n:
        d = d.sort_values("key", ascending=False).groupby("gate").head(n)
    d["pos"] = d.pos.astype(int)
    d["passes_filter"] = True
    return d


def fill_union(nb, d, kind="present"):
    """Same residues in every panel: for every position in the union over the
    gates, a gate that has no selected residue there gets the neighbour at that
    position with the most contact (closest if tied), even below the filter
    (passes_filter = False). A gate with no neighbour within 8 A there stays empty."""
    if d.empty:
        return d
    skip = 1 if kind == "change" else 2
    a = nb[np.isfinite(nb.pos)].copy()
    a = a[~(a.same_subunit.astype(bool) & (a.seq_distance <= skip))]
    a["chg"] = a.contact_after_pct - a.contact_before_pct
    a["pos"] = a.pos.astype(int)
    a["key"] = a[["contact_before_pct", "contact_after_pct"]].max(axis=1)
    a["dmin"] = a[["dist_before", "dist_after"]].min(axis=1)
    have = set(zip(d.gate, d.pos))
    add = []
    for (gate, p), grp in a[a.pos.isin(set(d.pos))].groupby(["gate", "pos"]):
        if (gate, p) not in have:
            add.append(grp.sort_values(["key", "dmin"], ascending=[False, True]).iloc[[0]])
    if not add:
        return d
    f = pd.concat(add).drop(columns="dmin")
    f["passes_filter"] = False
    return pd.concat([d, f], ignore_index=True)


def res_label(resname, pos, pdb_name, subunit, two_lines=False):
    """Residue name for the plots: GIRK2 residues as F83.2c (GIRK2 number +
    subunit); GIRK1 residues with their own number and the GIRK2 position in
    brackets, F72.G1 (=F83). pos = position from the HBC Phe."""
    pdb_name = str(pdb_name)
    sub = pdb_name.split(".")[-1] if "." in pdb_name else str(subunit)
    g2 = f"{ONE_LETTER.get(resname, 'X')}{HBC_GIRK2 + int(pos)}"
    if sub.upper().startswith("G") and "." in pdb_name:          # GIRK1
        if pdb_name.split(".")[0] == g2:
            return pdb_name
        return f"{pdb_name}{chr(10) if two_lines else ' '}(={g2})"
    return f"{g2}.{sub}"


def contacts_table(d):
    """CSV version of a contact plot: one row per gate Phe x neighbour shown."""
    if d.empty:
        return d
    out = pd.DataFrame(dict(
        event=d.event, jump_frame=d.jump_frame, direction=d.direction,
        gate=d.gate, gate_resid=d.gate_resid, gate_subunit=d.gate_subunit,
        label=[res_label(rn, ps, nb_, su) for rn, ps, nb_, su in
               zip(d.resname, d.pos, d.neighbour, d.subunit)],
        name_girk2=[f"{ONE_LETTER.get(rn, 'X')}{HBC_GIRK2 + int(ps)}."
                    f"{str(nb_).split('.')[-1] if '.' in str(nb_) else su}"
                    for rn, ps, nb_, su in zip(d.resname, d.pos, d.neighbour, d.subunit)],
        position_girk2=HBC_GIRK2 + d.pos, position_from_gate=d.pos,
        neighbour=d.neighbour, resid=d.resid, resname=d.resname, subunit=d.subunit,
        same_subunit=d.same_subunit,
        contact_before_pct=d.contact_before_pct, contact_after_pct=d.contact_after_pct,
        change_pct=d.chg,
        dist_before=d.dist_before, dist_after=d.dist_after, d_dist=d.d_dist,
        neighbour_disp=d.disp, neighbour_ratio=d.ratio, neighbour_real_move=d.real_move,
        passes_filter=d.passes_filter))
    return out.sort_values(["gate_subunit", "position_girk2", "subunit"]).reset_index(drop=True)


def plot_contacts(nb, out_png, title, kind="change", label="pdb", n=0, min_pct=5.0,
                  cutoff=4.5, min_change=0.0):
    """2 x 2 grid, one panel per gate Phe. x axis = residue POSITIONS, the same
    in every panel: pos = resid - HBC Phe resid of the residue's own subunit,
    labelled in GIRK2 numbering (192 + pos, so GIRK1 A174 = 185) for label
    'pdb', or as pos from the gate Phe for label 'raw'. The columns are the
    union over the four gates of the positions of the selected neighbours.
    Each bar pair = one neighbouring residue at that position: % frames in
    contact (< cutoff) before (grey) / after (green gained, red lost); under it
    its subunit (and raw resid for 'raw') and the change in percentage points.
    If two subunits touch the same gate at the same position, both are drawn.
    Selected per gate: neighbours within 8 A with >= min_pct contact before or
    after, not in the same subunit within +-1 ('change') / +-2 ('present');
    with min_change > 0 also a change of >= min_change points (default 0 = all);
    n > 0 keeps only the top n."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = fill_union(nb, select_contacts(nb, kind, n, min_pct, min_change), kind)
    if d.empty:
        return
    skip = 1 if kind == "change" else 2
    positions = sorted(d.pos.unique())
    col = {p: i for i, p in enumerate(positions)}
    if label == "pdb":
        xl = []
        for p in positions:
            letters = "/".join(sorted(set(ONE_LETTER.get(r, "X") for r in d.loc[d.pos == p, "resname"])))
            xl.append(f"{letters}{HBC_GIRK2 + p}")
    else:
        xl = [f"{p:+d}" for p in positions]
    gates = nb.drop_duplicates("gate").sort_values("gate_subunit")

    def sub_of(r):
        return str(r.neighbour).split(".")[-1] if "." in str(r.neighbour) else str(r.subunit)
    d["sub"] = [sub_of(r) for _, r in d.iterrows()]
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    subs = sorted(d["sub"].unique(), key=lambda x: (x.startswith("G"), x))
    scol = {s_: palette[i % len(palette)] for i, s_ in enumerate(subs)}

    # every residue gets one slot of width 1 (2 bars); a position is as wide as
    # the most residues any gate has there, so there is no empty padding
    nk = d.groupby(["gate", "pos"]).size().groupby("pos").max()
    x0, x = {}, 0.0
    for p in positions:
        x0[p] = x
        x += nk[p]
    xmax = x
    fs = 16
    fig, axes = plt.subplots(2, 2, figsize=(max(16, 1.25 * xmax * 2 + 3), 13), sharey=True)
    for ax, (_, g) in zip(axes.flat, gates.iterrows()):
        s = d[d.gate == g.gate].sort_values(["pos", "sub"])
        for p, grp in s.groupby("pos"):
            off = (nk[p] - len(grp)) / 2                  # centre in the position's column
            for j, (_, r) in enumerate(grp.iterrows()):
                xc = x0[p] + off + j + 0.5
                bw = 0.4
                ax.bar(xc - bw / 2, r.contact_before_pct, bw, color="#9aa5b1")
                ax.bar(xc + bw / 2, r.contact_after_pct, bw,
                       color="#2a9d8f" if r.chg >= 0 else "#e76f51")
                name = (res_label(r.resname, r.pos, r.neighbour, r.subunit, two_lines=True)
                        if label == "pdb" else f"{r.resname}{int(r.resid)}")
                nl = name.count("\n") + 1
                ax.annotate(name, (xc, 0), xytext=(0, -4), textcoords="offset points",
                            ha="center", va="top", fontsize=fs, annotation_clip=False,
                            linespacing=1.05, color="0.1" if r.passes_filter else "0.5",
                            style="normal" if r.passes_filter else "italic")
                c = round(float(r.chg)) + 0          # no "-0 %"
                ax.annotate(f"{c:+d} %", (xc, 0), xytext=(0, -8 - nl * fs * 1.2),
                            textcoords="offset points", ha="center", va="top", fontsize=fs,
                            annotation_clip=False, fontweight="bold",
                            color="#1f7a6f" if c > 0 else "#c0503a" if c < 0 else "0.45")
        ax.set_xticks([])
        gname = g.gate if label == "pdb" else f"resid {int(g.gate_resid)}"
        ax.set_title(f"gate {gname}  ({g.gate_subunit})", fontsize=fs + 3, fontweight="bold")
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_ylim(0, 105)
        ax.set_xlim(0, xmax)
        for p in positions[1:]:
            ax.axvline(x0[p], color="0.9", lw=0.8, zorder=0)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"% frames in contact (< {cutoff:g} Å)", fontsize=fs)
    fig.legend(handles=[Patch(color="#9aa5b1", label="before the jump"),
                        Patch(color="#2a9d8f", label="after: contact gained"),
                        Patch(color="#e76f51", label="after: contact lost")],
               loc="lower center", ncol=3, frameon=False, fontsize=fs)
    rule = f"≥ {min_pct:g} % contact before or after"
    if min_change:
        rule += f" and a change of ≥ {min_change:g} points"
    if n:
        rule += f", top {n} per gate"
    rule += (", not same subunit ±{skip}\ngrey italic name = shown only so every panel has the "
             "same positions (below the filter for this gate)").replace("{skip}", str(skip))
    xinfo = ("position in GIRK2 numbering (GIRK1 shifted +11: A174.G1 = 185)" if label == "pdb"
             else "position from the HBC Phe of the residue's own subunit (gate = 0)")
    naming = ("residue names: GIRK2 number + subunit; GIRK1 residues with their own number and "
              "the GIRK2 position in brackets, e.g. F72.G1 (=F83)" if label == "pdb" else "raw resid")
    fig.suptitle(f"{title}\nfor every neighbouring residue: % frames in contact before (grey) and after "
                 f"the jump; under it the residue and the change (after - before)"
                 f"\n{naming}\nneighbours within 8 Å, {rule}", fontsize=fs - 2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.88), h_pad=7)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_top_residues(res, out_png, title, label="pdb", n=10, exclude_ends=15, pos_range=None):
    """One horizontal-bar panel per subunit with its n residues of biggest Calpha
    displacement (chain ends left out). label = 'pdb' or 'raw'. With pos_range
    (lo, hi) only residues at those positions from the HBC Phe are ranked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    subs = sorted(res.subunit.unique())
    fig, axes = plt.subplots(1, len(subs), figsize=(4.2 * len(subs), 4.6), sharex=True)
    axes = np.atleast_1d(axes)
    xmax = 0.0
    for ax, s in zip(axes, subs):
        r = res[res.subunit == s].sort_values("resid")
        if exclude_ends and len(r) > 2 * exclude_ends + n:
            r = r.iloc[exclude_ends:len(r) - exclude_ends]
        if pos_range is not None:
            r = r[(r.pos >= pos_range[0]) & (r.pos <= pos_range[1])]
        top = r.nlargest(n, "disp").iloc[::-1]          # biggest at the top
        y = np.arange(len(top))
        real = top.ratio >= 3
        colors = np.where(top.d_radial > 0, "#2a9d8f", "#e76f51")
        ax.barh(y, top.disp, color=colors, alpha=0.35, edgecolor="none")
        ax.barh(y[real.values], top.disp[real.values], color=colors[real.values], edgecolor="black",
                lw=0.6)
        ax.scatter(top.sc_disp, y, color="#6a4c93", s=14, zorder=3)
        ax.scatter(3 * top.noise, y, marker="|", color="0.4", s=120, zorder=3)
        labels = (top.pdb if label == "pdb" else top.resname + top.resid.astype(str)).tolist()
        labels = [f"{l}  ({int(p):+d})" if np.isfinite(p) else l for l, p in zip(labels, top.pos)]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        name = s + ("  (GIRK1)" if r.girk1.any() else "")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("displacement (Å)")
        xmax = max(xmax, top.disp.max(), np.nanmax(top.sc_disp.values), 3 * top.noise.max())
    for ax in axes:
        ax.set_xlim(0, xmax * 1.08)
    handles = [Patch(color="#2a9d8f", label="Cα moved out"), Patch(color="#e76f51", label="Cα moved in"),
               Patch(facecolor="white", edgecolor="black", label="solid + outline = real move (≥ 3× noise)"),
               Line2D([], [], marker="o", ls="", color="#6a4c93", label="side-chain displacement"),
               Line2D([], [], marker="|", ls="", color="0.4", markersize=10, label="3 × noise")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    numbering = "PDB numbering" if label == "pdb" else "raw resid"
    fig.suptitle(f"{title}\ntop {n} residues by Cα displacement per subunit ({numbering}; "
                 f"(+n) = position from the HBC Phe; chain ends left out)", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(out_png, dpi=180)
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
