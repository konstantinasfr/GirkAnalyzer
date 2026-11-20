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
A_C = np.full(x.size, 7.0)
df = pd.DataFrame(dict(frame=np.arange(x.size), mean=(A_C + x) / 2, A_C=A_C, B_D=x))

def check(per, segs):
    assert list(segs.state) == ["closed", "open", "closed"], segs
    o = segs[segs.state == "open"].iloc[0]
    assert abs(o.start_frame - 1025) <= 10 and abs(o.end_frame - 1974) <= 10, o
    assert (per.state == "transition").sum() == 100
    assert per.loc[per.frame.between(1100, 1900), "state"].eq("open").all()

# default: mean of the diagonals (5 A closed, 6 A open) with 5.69 A
per, segs = find_states(df)
check(per, segs)
assert (per.distance == df["mean"]).all()
assert abs(segs.median_B_D.iloc[0] - 3) < 0.1 and abs(segs.median_A_C.iloc[1] - 7) < 1e-9
# the GIRK1 diagonal alone with 4.5 / 4.0 A gives the same windows
check(*find_states(df, "B_D", open_above=4.5, close_below=4.0))

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

# ---- step 2b: backbone vs side chain on known motions ----
# open frames: Phe 12 (B) only reorients its side chain (Calpha fixed);
#              Phe 14 (D) moves 1 A straight out as a rigid whole (backbone)
from analysis.hbc_open_closed import (compute_decomposition, decomposition_by_state,
                                      decomposition_by_switch, correlation_table, _units,
                                      segment_table)
X3 = X.copy()
X3[is_open, 27:36, 1] -= 1.0            # residue index 3 (resid 14) sits at y = -5: push to -6
# resid 12 (atoms 9-17, CA = 10): swing its side chain (CB..CZ) 50 deg around the
# z axis through its own CA -> pure reorientation, CA stays where it is
R = np.array([[np.cos(np.radians(50)), -np.sin(np.radians(50)), 0],
              [np.sin(np.radians(50)), np.cos(np.radians(50)), 0], [0, 0, 1]])
ca12 = X3[:, 10:11, :]
X3[is_open, 11:18] = ((X3[is_open, 11:18] - ca12[is_open]) @ R.T + ca12[is_open])
u.load_new(X3, format=MemoryReader)
geo3, pos3 = compute_gate_geometry(u, pairs, labels, keep_positions=True)
# the 4 test Phe are the whole "protein", so their centroid moves with D; real
# channels have ~1000 CA. Use the true pore centre (origin) as the reference:
pos3["_ref_center"] = np.zeros((nf, 3))
dec = compute_decomposition(pos3, pairs, labels)
dec.insert(0, "frame", np.arange(nf)); dec.insert(1, "state", np.where(is_open, "open", "closed"))
units = _units(pairs, labels)
bs = decomposition_by_state(dec, units).set_index("unit")
# D: rigid -> all backbone, no side-chain part
assert abs(bs.loc["D:F14", "d_backbone"] - 1.0) < 0.02 and abs(bs.loc["D:F14", "d_sidechain"]) < 0.02
assert abs(bs.loc["D:F14", "pct_backbone"] - 100) < 2
# B: chi1 flip -> no backbone part, all side chain
assert abs(bs.loc["B:F12", "d_backbone"]) < 0.02 and abs(bs.loc["B:F12", "d_sidechain"]) > 0.1
assert abs(bs.loc["B:F12", "pct_sidechain"] - 100) < 2
# A, C untouched
assert abs(bs.loc["A:F11", "d_total"]) < 0.02 and abs(bs.loc["C:F13", "d_total"]) < 0.02
# parts add up exactly
assert np.allclose(bs.d_backbone + bs.d_sidechain, bs.d_total, atol=2e-3)
# the axis-based radial distance equals the in-plane distance (pore axis = z here)
assert np.allclose(dec["ca_r A:F11"], 5) and np.allclose(dec.loc[is_open, "ca_r D:F14"], 6)
segs3 = segment_table(np.arange(nf), np.zeros(nf), dec.state.to_numpy())
sw = decomposition_by_switch(dec, segs3, units, smooth=1).set_index(["switch", "unit"])
assert list(sw.index.get_level_values(0).unique()) == ["closed->open", "open->closed"]
assert abs(sw.loc[("closed->open", "D:F14"), "d_backbone"] - 1) < 0.02
assert sw.loc[("closed->open", "D:F14"), "t_half_backbone"] == 100
assert abs(sw.loc[("open->closed", "D:F14"), "d_backbone"] + 1) < 0.02
assert sw.loc[("open->closed", "D:F14"), "t_half_backbone"] == 200
assert sw.loc[("open->closed", "B:F12"), "t_half_sidechain"] == 200
hbc = pd.DataFrame(dict(frame=np.arange(nf), A_C=dec["radial_gap A_C"], B_D=dec["radial_gap B_D"]))
c = correlation_table(dec, hbc).set_index("diagonal")
assert abs(c.loc["B_D", "R2_both"] - 1) < 1e-6
print("test_hbc_open_closed step 2b: OK")
