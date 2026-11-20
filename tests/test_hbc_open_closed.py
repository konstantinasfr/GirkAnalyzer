"""Synthetic check for analysis/hbc_open_closed.py (step 1): closed (3 A) / open (5 A) / closed trace with
noise, a short spike and a short dip. Run from the repo root: python3 tests/test_hbc_open_closed.py"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from analysis.hbc_open_closed import find_states, parse_manual

rng = np.random.default_rng(0)
x = np.full(3000, 3.0); x[1000:2000] = 5.0
x[500:540] = 5.0        # 40-frame spike while closed -> must be absorbed
x[1400:1440] = 3.0      # 40-frame dip while open     -> must be absorbed
x += rng.normal(0, 0.4, x.size)
df = pd.DataFrame(dict(frame=np.arange(x.size), mean=x, A_C=np.full(x.size, 7.0), B_D=x))

per, segs = find_states(df, "B_D", buffer=25)
assert list(segs.state) == ["closed", "open", "closed"], segs
o = segs[segs.state == "open"].iloc[0]
assert abs(o.start_frame - 1025) <= 10 and abs(o.end_frame - 1974) <= 10, o
assert (per.state == "transition").sum() == 100
assert per.loc[per.frame.between(1100, 1900), "state"].eq("open").all()

lab = parse_manual("open:10-19;closed:0-4,30-39", 50)
assert (lab[10:20] == "open").all() and (lab[30:40] == "closed").all() and lab[5] == "excluded"
try: parse_manual("open:0-10;closed:5-20", 50); raise SystemExit("overlap not caught")
except ValueError: pass
print("test_hbc_open_closed: OK")

# ---- step 2: synthetic 4-Phe gate; the Phe 12 (diagonal B_D) chi1 flips g+ -> t when open ----
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.lib.distances import calc_dihedrals
from analysis.hbc_open_closed import dihedral, compute_gate_geometry, summarize_by_state, rotamer_table

P = rng.normal(size=(4, 500, 3))
assert np.allclose(dihedral(*P), np.degrees(calc_dihedrals(*P)), atol=1e-3)   # same as MDAnalysis

def rot(v, axis, deg):                      # Rodrigues rotation of v about unit axis
    a = np.radians(deg); axis = axis / np.linalg.norm(axis)
    return v * np.cos(a) + np.cross(axis, v) * np.sin(a) + axis * np.dot(axis, v) * (1 - np.cos(a))

def phe(ca, chi1):
    """N, CA, CB, CG, CD1, CD2, CE1, CE2, CZ with the given chi1 (N-CA-CB-CG)."""
    inward = -ca / np.linalg.norm(ca); z = np.array([0, 0, 1.])
    N, CB = ca + 1.5 * z, ca + 1.5 * inward
    g0 = CB + 1.5 * rot(-inward, z, 70)                 # some starting CG
    d0 = dihedral(N[None], ca[None], CB[None], g0[None])[0]
    CG = CB + rot(g0 - CB, CB - ca, chi1 - d0)          # rotate about CA->CB to the target chi1
    e = (CG - CB) / 1.5
    ring = [CG + 1.4 * e + 0.7 * s for s in (z, -z, np.cross(z, e), -np.cross(z, e))]
    return [N, ca, CB, CG, ring[0], ring[1], ring[2], ring[3], CG + 2.8 * e]

names = ["N", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
nf = 300
u = mda.Universe.empty(36, n_residues=4, atom_resindex=np.repeat(np.arange(4), 9), trajectory=False)
u.add_TopologyAttr("resids", [11, 12, 13, 14]); u.add_TopologyAttr("resnames", ["PHE"] * 4)
u.add_TopologyAttr("names", names * 4)
is_open = np.zeros(nf, bool); is_open[100:200] = True
cas = [np.array(v, float) for v in ([5, 0, 0], [0, 5, 0], [-5, 0, 0], [0, -5, 0])]
X = np.array([[a for k, ca in enumerate(cas)
               for a in phe(ca, 180.0 if (k == 1 and o) else 60.0)] for o in is_open], np.float32)
u.load_new(X, format=MemoryReader)

pairs = [(11, 13), (12, 14)]
labels = {11: "A:F11", 13: "C:F13", 12: "B:F12", 14: "D:F14"}
geo = compute_gate_geometry(u, pairs, labels)
assert np.allclose(geo["ca_ca A_C"], 10) and np.allclose(geo["ca_ca B_D"], 10)
c = geo["chi1 B:F12"].to_numpy()
assert np.allclose(np.abs(c[is_open]), 180, atol=1e-2) and np.allclose(c[~is_open], 60, atol=1e-2)
assert np.allclose(geo["chi1 A:F11"], 60, atol=1e-2)
geo["state"] = np.where(is_open, "open", "closed")
r = rotamer_table(geo).set_index(["residue", "state"])
assert r.loc[("B:F12", "open"), "pct_t"] == 100 and r.loc[("B:F12", "closed"), "pct_g+"] == 100
s = summarize_by_state(geo).set_index(["measure", "state"])
assert abs(abs(s.loc[("chi1 B:F12", "open-closed"), "median"]) - 120) < 1e-2
assert abs(s.loc[("chi1 A:F11", "open-closed"), "median"]) < 1e-2
assert abs(s.loc[("ca_ca B_D", "open-closed"), "median"]) < 1e-6
print("test_hbc_open_closed step 2: OK")
