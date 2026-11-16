"""Ground truth for analysis/cavity_esp_classical.py on a synthetic 4-subunit cavity: cluster charge,
neutral caps, mesh geometry, the cylinder mean against a brute-force Coulomb sum, and the per-residue
contributions adding up to it. Run from the repo root: python3 tests/test_cavity_esp.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from analysis.cavity_esp_classical import cluster_atoms, pair_rings, frame_esp, mesh_local, mesh_xyz, COULOMB
ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(("OK      " if cond else "FAIL    ") + name)

def build(glu, low, lowtype, nframes=30, seed=0):
    rng = np.random.default_rng(seed)
    names, resn, rid, rix, q, xyz = [], [], [], [], [], []
    def add_res(r, rn, atoms):
        k = len(resn); resn.append(rn); rid.append(r)
        for nm, c, p in atoms:
            names.append(nm); rix.append(k); q.append(c); xyz.append(p)
    def gly(r, centre):
        ch = rng.normal(0, 0.3, 7); ch -= ch.mean()
        add_res(r, "GLY", [(n, c, centre + rng.normal(0, 1.0, 3)) for n, c in zip(["N","H","CA","HA2","HA3","C","O"], ch)])
    for k, (g, l) in enumerate(zip(glu, low)):
        ang = k * np.pi / 2
        u = np.array([np.cos(ang), np.sin(ang), 0])
        up, lo = u * 9 + [0, 0, 0], u * 8 + [0, 0, -10]
        gly(g - 1, up + [0, 0, 2]); 
        add_res(g, "GLU", [("N", -0.4, up + [0,0,1]), ("H", 0.3, up + [0,0,1.5]), ("CA", 0.0, up), ("HA", 0.1, up + [0.5,0,0]),
                           ("CB", 0.0, up - u * 1.2), ("CG", 0.0, up - u * 2.0), ("CD", 0.7, up - u * 3.0),
                           ("OE1", -0.85, up - u * 3.6 + [0, 0.8, 0]), ("OE2", -0.85, up - u * 3.6 - [0, 0.8, 0]),
                           ("C", 0.5, up + [0,0,-1]), ("O", -0.5, up + [0,0.6,-1.6])])
        gly(g + 1, up + [0, 0, -2.5])
        gly(l - 1, lo + [0, 0, 2])
        if lowtype[k] == "ASP":
            sc = [("CB", 0.0, lo - u*1.2), ("CG", 0.6, lo - u*2.2), ("OD1", -0.8, lo - u*2.8 + [0,.8,0]), ("OD2", -0.8, lo - u*2.8 - [0,.8,0])]
        else:
            sc = [("CB", 0.0, lo - u*1.2), ("CG", 0.6, lo - u*2.2), ("OD1", -0.6, lo - u*2.8 + [0,.8,0]), ("ND2", 0.0, lo - u*2.8 - [0,.8,0])]
        add_res(l, lowtype[k], [("N", -0.4, lo + [0,0,1]), ("H", 0.3, lo + [0,0,1.5]), ("CA", 0.0, lo), ("HA", 0.1, lo + [.5,0,0])] + sc +
                [("C", 0.5, lo + [0,0,-1]), ("O", -0.5 - (0 if lowtype[k]=="ASP" else 0), lo + [0,.6,-1.6])])
        gly(l + 1, lo + [0, 0, -2.5])
    # fix exact residue totals: Glu -1, Asp -1, Asn 0
    q = np.array(q, float); rix = np.array(rix)
    for k, rn in enumerate(resn):
        sel = rix == k; target = {"GLU": -1, "ASP": -1, "ASN": 0, "GLY": 0}[rn]
        q[sel] += (target - q[sel].sum()) / sel.sum()
    u = mda.Universe.empty(len(names), n_residues=len(resn), atom_resindex=rix, trajectory=False)
    for at, v in (("names", names), ("resnames", resn), ("resids", rid), ("charges", q)): u.add_TopologyAttr(at, v)
    X = np.array(xyz, np.float32)
    traj = np.stack([X + rng.normal(0, 0.15, X.shape).astype(np.float32) for _ in range(nframes)])
    u.load_new(traj, format=MemoryReader)
    return u


glu, low = [99, 424, 748, 1073], [131, 456, 780, 1105]
u = build(glu, low, ["ASP", "ASN", "ASN", "ASN"], nframes=5)
pairs = pair_rings(glu, low)
chk("pairs by sequence", pairs == [(99, 131), (424, 456), (748, 780), (1073, 1105)])
atoms, q, groups = cluster_atoms(u, pairs)
chk(f"cluster charge -5 ({q.sum():.4f})", abs(q.sum() + 5) < 1e-6)
chk("caps neutral", abs(q[groups == "caps"].sum()) < 1e-6)
u.trajectory[3]
labels = {**{x: str(x) for p in pairs for x in p}, **{p: f"{p[0]}/{p[1]}" for p in pairs}}
rows, prof = frame_esp(u, pairs, atoms, q, groups, labels)
g, l = pairs[0]
p0 = u.select_atoms(f"resid {g} and name CA").positions[0].astype(float)
p1 = u.select_atoms(f"resid {l} and name CA").positions[0].astype(float)
t, r, a = mesh_local(np.linalg.norm(p1 - p0)); pts = mesh_xyz(p0, p1, t, r, a)
d = np.linalg.norm(pts[:, None] - atoms.positions[None], axis=2)
keep = d.min(1) >= 1.4
brute = (COULOMB * (q / np.maximum(d, 1e-6)).sum(1))[keep].mean()
chk(f"cylinder mean = brute force ({rows[0]['mean']:.3f} vs {brute:.3f})", abs(rows[0]["mean"] - brute) < 1e-6)
chk("contributions add up to the mean", abs(sum(v for k, v in rows[0].items() if k.startswith("contrib_")) - rows[0]["mean"]) < 1e-6)
uax = (p1 - p0) / np.linalg.norm(p1 - p0); rel = pts - p0; tt = rel @ uax
chk("mesh inside radius 3 A", np.linalg.norm(rel - np.outer(tt, uax), axis=1).max() <= 3 + 1e-9)
chk("mesh spans -3 .. L+3 A", tt.min() >= -3 - 1e-9 and tt.max() <= np.linalg.norm(p1 - p0) + 3 + 1e-9)
chk("carboxylate distance from the axis recorded", rows[0]["r_o_upper"] < rows[0]["r_ca_upper"] and rows[0]["max_r_o_upper"] >= rows[0]["r_o_upper"])
from analysis.cavity_esp_classical import _JAG_DIRS, JAG_RMIN
X = atoms.positions.astype(float)
jp = (X[:, None] + _JAG_DIRS[None]).reshape(-1, 3)
rel = jp - p0; tt = rel @ uax; rad = np.linalg.norm(rel - np.outer(tt, uax), axis=1)
jp = jp[(rad <= 3) & (tt >= -3) & (tt <= np.linalg.norm(p1 - p0) + 3)]
rmin = np.array([JAG_RMIN.get(n[0], 1.94) for n in atoms.names])
dj = np.linalg.norm(jp[:, None] - X[None], axis=2); okp = (dj >= rmin[None] - 1e-6).all(1)
jm = (COULOMB * (q / dj[okp]).sum(1)).mean()
chk(f"Jaguar-like grid mean = independent rebuild ({rows[0]['jag_mean']:.3f} vs {jm:.3f}, n {rows[0]['jag_n']} vs {okp.sum()})",
    abs(rows[0]["jag_mean"] - jm) < 1e-6 and rows[0]["jag_n"] == okp.sum())
print("\nALL OK" if ok else "\nSOME FAILED"); sys.exit(not ok)
