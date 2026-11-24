"""Synthetic 4-chain channel shared by the tests: 40 residues per chain, HBC Phe
at residue index 30, rigidly rotated + shifted every frame. From frame JUMP
on, chain B residues 25-39 move 1.5 A outward and chain D 28-39 1.0 A, so the
B_D diagonal widens by ~2.5 A and A_C stays."""
import numpy as np, MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader


def build(NF=900, JUMP=450, seed=1):
    rng = np.random.default_rng(seed)
    NCH, NRES = 4, 40
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

    return u, sf_lower, sf_upper, pairs, groups
