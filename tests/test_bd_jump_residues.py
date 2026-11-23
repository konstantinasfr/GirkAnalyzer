"""Synthetic check for analysis/bd_jump_residues.py.
Four chains of 40 residues; the whole channel is randomly rotated and shifted
in every frame (alignment must undo it). From frame 450 on, residues 25-39 of
chain B move 1.5 A outward and residues 28-39 of chain D 1.0 A outward, so the
B_D diagonal widens by ~2.5 A. Run from the repo root:
    python3 tests/test_bd_jump_residues.py"""
import sys, tempfile; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from analysis.bd_jump_residues import (run_bd_jump_residues, flat_stretches, kabsch, OUT_DIRNAME,
                                      pdb_label)

rng = np.random.default_rng(1)

# PDB numbering through analysis/converter.py (G12 WT: resid 99 = GIRK1 E141, 464 = F192 of 2b)
assert pdb_label(99, "GLU", "G12") == "E141.G1" and pdb_label(464, "PHE", "G12") == "F192.2b"
assert pdb_label(1073, "GLU", "G12_E141Q") == "E141.G1"
assert pdb_label(5, "ALA", None) == "A5"

# kabsch recovers a known rotation
P = rng.normal(size=(20, 3)); t = 0.7
Rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
assert np.allclose((P - P.mean(0)) @ kabsch(P, P @ Rz.T + 3) , (P @ Rz.T) - (P @ Rz.T).mean(0), atol=1e-8)

# flat_stretches: two clean levels
x = np.r_[np.full(400, 10.0), np.full(400, 12.0)] + rng.normal(0, 0.3, 800)
b = flat_stretches(x, 1.0, 150)
assert len(b) == 3 and abs(b[1] - 400) <= 5, b
assert flat_stretches(np.full(800, 10.0) + rng.normal(0, 0.3, 800), 1.0, 150) == [0, 800]

NCH, NRES, NF, JUMP = 4, 40, 900, 450
names = ["N", "CA", "C", "O", "CB"]
nat = NCH * NRES * len(names)
u = mda.Universe.empty(nat, n_residues=NCH * NRES, atom_resindex=np.repeat(np.arange(NCH * NRES), len(names)),
                       trajectory=False)
u.add_TopologyAttr("names", names * NCH * NRES)
u.add_TopologyAttr("resids", np.arange(1, NCH * NRES + 1))
rn = ["ALA"] * (NCH * NRES)
for k in range(NCH):
    rn[k * NRES + 30] = "PHE"                       # HBC Phe = residue index 30 -> resid k*40+31
u.add_TopologyAttr("resnames", rn)
u.add_TopologyAttr("bonds", [(i, i + 1) for i in range(nat - 1)
                             if (i + 1) % (NRES * len(names)) != 0])   # one fragment per chain

def base_coords(moved):
    X = np.zeros((nat, 3))
    for k in range(NCH):
        th = np.pi / 2 * k
        rdir = np.array([np.cos(th), np.sin(th), 0.0])
        tdir = np.array([-np.sin(th), np.cos(th), 0.0])
        for i in range(NRES):
            if i < 18:     # pore helix (0-11) + filter (12-17): near the top
                r, z = 4 + 0.25 * i, 30 - 0.8 * i
            else:          # TM2 going down
                r, z = 7 + 0.05 * (i - 18), 14 - 1.5 * (i - 18)
            if moved:
                if k == 1 and i >= 25: r += 1.5
                if k == 3 and i >= 28: r += 1.0
            ca = r * rdir + np.array([0, 0, z])
            base = (k * NRES + i) * len(names)
            for a, off in enumerate([(0, 0, 0.8), (0, 0, 0), (0.6, 0, -0.5), (1.2, 0, -0.5), (0, 0, 0)]):
                X[base + a] = ca + off[0] * tdir + np.array([0, 0, off[1] + off[2]])
            X[base + 4] = ca + 1.2 * rdir             # CB points outward
    return X

X0, X1 = base_coords(False), base_coords(True)
traj = np.empty((NF, nat, 3), np.float32)
for f in range(NF):
    X = (X1 if f >= JUMP else X0) + rng.normal(0, 0.25, (nat, 3))
    q = rng.normal(size=4); q /= np.linalg.norm(q)             # random rotation
    w, x_, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x_ * y - z * w), 2 * (x_ * z + y * w)],
                  [2 * (x_ * y + z * w), 1 - 2 * (x_ * x_ + z * z), 2 * (y * z - x_ * w)],
                  [2 * (x_ * z - y * w), 2 * (y * z + x_ * w), 1 - 2 * (x_ * x_ + y * y)]])
    traj[f] = X @ R.T + rng.normal(0, 20, 3)
u.load_new(traj, format=MemoryReader)

sf_lower = [k * NRES + 13 for k in range(NCH)]
sf_upper = [k * NRES + 18 for k in range(NCH)]
pairs = [(31, 111), (71, 151)]                       # A_C, B_D
groups = {"A": [31], "B": [71], "C": [111], "D": [151]}

with tempfile.TemporaryDirectory() as tmp:
    run_bd_jump_residues(u, pairs, sf_lower, sf_upper, tmp, subunit_groups=groups)
    out = Path(tmp) / OUT_DIRNAME / "minjump_1.0"
    j = pd.read_csv(out / "jumps.csv")
    assert len(j) == 1 and abs(j.frame[0] - JUMP) <= 5 and j.direction[0] == "widening", j
    assert abs(j.step[0] - 2.5) < 0.2 and abs(j.ac_step[0]) < 0.2, j
    rm = pd.read_csv(out / "alignment_rmsd.csv")
    # 0.25 A noise per coordinate in the frame and in the reference -> RMSD ~ 0.25*sqrt(3)*sqrt(2) = 0.61
    assert rm.fit_rmsd.median() < 0.75, rm.fit_rmsd.median()          # rotation undone (else ~20 A)
    for mode in ("window300", "window500", "segment_all"):
        d = out / mode
        r = pd.read_csv(next(d.glob("event1_frame*_residues.csv")))
        assert (d / "consistent_residues.csv").exists() and list(d.glob("*_disp.pdb"))
        B = r[(r.subunit == "B")].set_index("pos")
        D = r[(r.subunit == "D")].set_index("pos")
        A = r[(r.subunit == "A")].set_index("pos")
        # positions relative to the HBC Phe (index 30): chain B moved from index 25 = pos -5
        assert np.allclose(B.loc[-5:9, "d_radial"], 1.5, atol=0.15), B.loc[-5:9, "d_radial"]
        assert np.allclose(D.loc[-2:9, "d_radial"], 1.0, atol=0.15)
        assert (B.loc[-5:9, "ratio"] > 3).all()
        assert (A.disp < 3 * A.noise + 0.05).all()                    # nothing moved in A
        assert np.all(np.abs(B.loc[-5:9, "d_axial"]) < 0.15)          # outward, not up/down
        assert r.loc[r.resname == "PHE", "pos"].eq(0).all()
        assert list(d.glob("event1_frame*_TM.png"))
        assert all(list(d.glob(f"event1_frame*_top10_{k}.png")) for k in ("pdb", "raw", "TM_pdb", "TM_raw"))
        g = pd.read_csv(next(d.glob("event1_frame*_gate.csv")))
        assert sorted(g.pos.unique()) == list(range(-3, 4)) and len(g) == 4 * 7
        gB = g[(g.subunit == "B") & (g.pos == 0)].iloc[0]
        assert abs(gB.d_radial - 1.5) < 0.15 and gB.resid == 71
        assert (d / "all_events_gate.csv").exists()
        raw = pd.read_csv(next(d.glob("event1_frame*_ranked_raw.csv")))
        pdb = pd.read_csv(next(d.glob("event1_frame*_ranked_pdb.csv")))
        assert raw.disp.is_monotonic_decreasing and len(raw) == NCH * NRES
        assert set(raw.head(27).subunit) <= {"B", "D"}          # the 27 moved residues come first
        assert (raw.resid == raw.resid).all() and "residue" in pdb and len(pdb) == len(raw)
        assert (raw.head(15).real_move == "yes").all()
        nb = pd.read_csv(next(d.glob("event1_frame*_neighbours.csv")))
        nbB = nb[nb.gate_resid == 71]
        assert 71 not in set(nbB.resid) and {70, 72} <= set(nbB.resid)
        assert (nbB.dist_before.le(8) | nbB.dist_after.le(8)).all()
        # brute force: median over the "before" frames of the min heavy-atom distance F71 - A74
        row = nbB[nbB.resid == 74].iloc[0]
        lo, hi = map(int, str(row_frames := r.before_frames.iloc[0]).split("-"))
        g71 = u.select_atoms("resid 71"); a74 = u.select_atoms("resid 74")
        dmins = []
        for ts in u.trajectory[lo:hi + 1]:
            dmins.append(np.linalg.norm(g71.positions[:, None] - a74.positions[None], axis=2).min())
        assert abs(np.median(dmins) - row.dist_before) < 1e-3, (np.median(dmins), row.dist_before)
        assert (d / "all_events_neighbours.csv").exists()
        for kind in ("change", "present"):
            for lab in ("pdb", "raw"):
                assert list((d / "contacts").glob(f"event1_frame*_contacts_{kind}_{lab}.png")), (kind, lab)
            ct = pd.read_csv(next((d / "contacts").glob(f"event1_frame*_contacts_{kind}.csv")))
            assert np.allclose(ct.change_pct, ct.contact_after_pct - ct.contact_before_pct, atol=2e-3)
            assert (ct.position_girk2 == 192 + ct.position_from_gate).all()
            assert (d / "contacts" / f"all_events_contacts_{kind}.csv").exists()
    print("test_bd_jump_residues: OK")
