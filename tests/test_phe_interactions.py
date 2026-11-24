"""Hand-built geometry check for analysis/phe_interactions.py. A gate Phe ring
lies flat at the origin (normal = z). Over 100 frames: PHE 2 is far away in
frames 0-49 and stacked parallel 3.8 A above in 50-99; PHE 3 is T-shaped (ring
plane xz) 5 A along -x all the time; CYS 4 has its SG 4 A below the ring
only after frame 50, with its HG pointing at the ring; ALA 5 has CB over
the ring face (CH-pi); LYS 6 NZ 5.8 A away in the ring plane (cation-pi).
    python3 tests/test_phe_interactions.py"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from analysis.phe_interactions import RingTracker, plot_interactions

ring = ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]
hexa = np.array([[1.4 * np.cos(np.pi / 3 * k), 1.4 * np.sin(np.pi / 3 * k), 0] for k in range(6)])
res = [("PHE", ring), ("PHE", ring), ("PHE", ring), ("CYS", ["SG", "HG"]), ("ALA", ["CB"]),
       ("LYS", ["NZ"]), ("HIS", ["CG", "ND1", "CE1", "NE2", "CD2"])]
names = [n for _, ns in res for n in ns]
rix = np.repeat(np.arange(len(res)), [len(ns) for _, ns in res])
u = mda.Universe.empty(len(names), n_residues=len(res), atom_resindex=rix, trajectory=False)
u.add_TopologyAttr("names", names)
u.add_TopologyAttr("resnames", [r for r, _ in res])
u.add_TopologyAttr("resids", np.arange(1, len(res) + 1))

penta = np.array([[1.2 * np.cos(2 * np.pi / 5 * k), 1.2 * np.sin(2 * np.pi / 5 * k), 0]
                  for k in range(5)])
NF = 100
X = np.zeros((NF, len(names), 3), np.float32)
for f in range(NF):
    after = f >= 50
    X[f] = np.vstack([
        hexa,                                                          # gate at origin
        hexa + ([0, 0, 3.8] if after else [9, 0, 0]),                  # stacked after 50
        hexa[:, [0, 2, 1]] + [-5.0, 0, 0],        # ring in xz plane: T-shaped
        np.array([[0, 0, -4.0], [0, 0, -2.7]]) + ([0, 0, 0] if after else [0, -9, 0]),
        [[1.0, 0.5, 3.6 if after else -3.6]],                          # CB over the ring face
        [[0, 5.8, 0]],                                                 # NZ in the ring plane
        penta + [0, 0, -10.0]])                                        # HIS (5-ring), tracked, no contact
u.load_new(X, format=MemoryReader)

residues = u.residues
resmap = {r.resindex: i for i, r in enumerate(residues)}
tr = RingTracker(residues, resmap, [1], NF)
for i, ts in enumerate(u.trajectory):
    tr.update(i)
info = pd.DataFrame(dict(resid=[1, 2, 3, 4, 5, 6, 7], resname=[r for r, _ in res],
                         pdb=["F192.2a", "F186.2b", "F83.2b", "C190.2b", "A185.2b", "K194.2b",
                              "H200.2b"],
                         subunit=["A", "B", "B", "B", "B", "B", "B"],
                         pos=[0, -6, -109, -2, -7, 2, 8], chain=[0, 1, 1, 1, 1, 1, 1]))
s = tr.summary(np.arange(0, 50), np.arange(50, 100), info).set_index(["partner", "interaction"])
print(s[["pct_before", "pct_after", "dist_before", "dist_after", "geometry_before",
         "geometry_after"]].round(2))
r = s.loc[("F186.2b", "pi-pi")]
assert r.pct_before == 0 and r.pct_after == 100 and abs(r.dist_after - 3.8) < 1e-3
assert r.geometry_after.startswith("parallel")
r = s.loc[("F83.2b", "pi-pi")]
assert r.pct_before == 100 == r.pct_after and r.geometry_before.startswith("T-shaped")
assert abs(r.angle_before - 90) < 1e-3
r = s.loc[("C190.2b", "S...pi")]
assert r.pct_before == 0 and r.pct_after == 100 and abs(r.dist_after - 4.0) < 1e-3
r = s.loc[("C190.2b", "S-H...pi")]
assert r.pct_after == 100 and abs(r.dist_after - 2.7) < 1e-3 and abs(r.angle_after - 180) < 1e-3
r = s.loc[("A185.2b", "CH-pi")]
assert r.pct_before == 100 == r.pct_after
r = s.loc[("K194.2b", "cation-pi")]
assert r.pct_before == 100 == r.pct_after and abs(r.dist_before - 5.8) < 1e-3
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    plot_interactions(s.reset_index(), Path(tmp) / "x.png", "test")
    assert (Path(tmp) / "x.png").exists()
print("test_phe_interactions: OK")

from analysis.bd_jump_residues import res_label
assert res_label("PHE", -109, "F72.G1", "D") == "F72.G1 (=F83)"
assert res_label("PHE", -109, "F72.G1", "D", two_lines=True) == "F72.G1\n(=F83)"
assert res_label("PHE", -109, "F83.2c", "C") == "F83.2c"
print("res_label: OK")

# every panel gets the same columns: a second gate without the C190 S...pi still gets a column
t2 = s.reset_index().copy()
t2b = t2.copy(); t2b["gate"] = "F192.2c"; t2b["gate_subunit"] = "C"
t2b = t2b[t2b.interaction != "S...pi"]
with tempfile.TemporaryDirectory() as tmp:
    plot_interactions(pd.concat([t2, t2b]), Path(tmp) / "y.png", "test 2 gates")
    assert (Path(tmp) / "y.png").exists()
print("union plot: OK")
