"""Synthetic check for analysis/gate_states.py: B_D widens at frame 450, A_C
does not change -> B_D has 2 states, A_C 1 (the whole run), joint 2.
    python3 tests/test_gate_states.py"""
import sys, json, tempfile; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from synthetic_channel import build
from analysis.gate_states import run_gate_states, states_of
from analysis.gate_timeline import local_bend, kink_angle

# bending measures on ideal helices (CA: radius 2.3 A, 1.5 A rise, 100 deg per residue)
tt = np.radians(100) * np.arange(25)
H = np.c_[2.3 * np.cos(tt), 2.3 * np.sin(tt), 1.5 * np.arange(25)]
h = 12
assert local_bend(H[None], list(range(h - 3, h + 4)))[0] < 1e-6          # straight -> 0
for th in (10, 30, 45):
    r = np.radians(th)
    R = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    X = H.copy(); X[h + 1:] = (X[h + 1:] - X[h]) @ R.T + X[h]
    assert abs(local_bend(X[None], list(range(h - 3, h + 4)))[0] - th) < 0.5, th
# segment-axis bend (7 CA per side through HELANAL axis points): 0 for a straight helix,
# exact for a kink, and its outward / sideways parts exact for a pure outward / sideways tilt
from analysis.gate_timeline import segment_bend, HelixTracker
tt30 = np.radians(100) * np.arange(30)
H30 = np.c_[2.3 * np.cos(tt30), 2.3 * np.sin(tt30), 1.5 * np.arange(30)]
hh = 15
up, dn = list(range(hh - 7, hh)), list(range(hh + 1, hh + 8))
pore = (np.array([-10.0, 0, 0]), np.array([0, 0, 1.0]))      # pore axis 10 A away along -x
assert segment_bend(H30[None], up, dn)[0][0] < 1e-6
for kvec, th, exp_rad in (((0, 1, 0), 25, 25), ((0, 1, 0), -15, -15), ((1, 0, 0), 20, 0)):
    k = np.array(kvec, float); r = np.radians(th)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(r) * K + (1 - np.cos(r)) * K @ K
    X = H30.copy(); X[hh + 1:] = (X[hh + 1:] - X[hh]) @ R.T + X[hh]
    tot, rad, tan = segment_bend(X[None], up, dn, X[None, hh], pore)
    assert abs(tot[0] - abs(th)) < 1e-6 and abs(rad[0] - exp_rad) < 1e-6, (th, tot, rad, tan)
    assert abs(abs(tan[0]) - (abs(th) if exp_rad == 0 else 0)) < 1e-6
# segment tilt relative to the pore axis: vertical helix 0, leaning by theta -> exactly theta,
# sign + when the lower end is further from the pore axis
from analysis.gate_timeline import segment_tilt
Hd = np.c_[2.3 * np.cos(tt30), 2.3 * np.sin(tt30), -1.5 * np.arange(30)]   # N -> C going down
assert np.allclose(segment_tilt(Hd[None], list(range(5, 12)), pore)[0], 0, atol=1e-6)
for th in (10, -25):
    r = np.radians(th)
    R = np.array([[np.cos(r), 0, -np.sin(r)], [0, 1, 0], [np.sin(r), 0, np.cos(r)]])  # lower end -> +x (away)
    tot, rad, tan = segment_tilt((Hd @ R.T)[None], list(range(5, 12)), pore)
    assert abs(tot[0] - abs(th)) < 1e-6 and abs(rad[0] - th) < 1e-6 and abs(tan[0]) < 1e-6, (th, tot, rad)
# direction of the chain after the helix end (CA -> CA vector) relative to the pore axis
from analysis.gate_timeline import vector_tilt
for th in (0, 20, -30):
    rr_ = np.radians(th)
    tot, rad, tan = vector_tilt(np.zeros((1, 3)), np.array([[6 * np.sin(rr_), 0, -6 * np.cos(rr_)]]), pore)
    assert abs(tot[0] - abs(th)) < 1e-6 and abs(rad[0] - th) < 1e-6 and abs(tan[0]) < 1e-6
# bend profile: the local bend peaks within +-1 residue of a kink
r = np.radians(20)
R = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
X = H30.copy(); X[hh + 1:] = (X[hh + 1:] - X[hh]) @ R.T + X[hh]
prof = [local_bend(X[None], list(range(k - 3, k + 4)))[0] for k in range(hh - 5, hh + 6)]
assert abs(int(np.argmax(prof)) - 5) <= 1 and prof[0] < 1e-6 and prof[-1] < 1e-6, prof
# the broad 8+8 measure has an offset of ~18 deg for a straight ideal helix (documented)
k0 = kink_angle(H[None], list(range(h - 8, h)), list(range(h + 1, h + 9)))[0]
assert 15 < k0 < 21, k0

b, st, near = states_of(np.r_[np.full(300, 10.0), np.full(300, 12.0)], 1.0, 150, 25)
assert b == [0, 300, 600] and st[299] == 1 and st[300] == 2 and near[280:320].all() and not near[:270].any()
# high - low - high: the first cut alone is < 1 A, the exhaustive search still finds both jumps
rng0 = np.random.default_rng(3)
y = np.r_[np.full(2000, 16.0), np.full(2800, 14.0), np.full(2000, 16.0)] + rng0.normal(0, .5, 6800)
b, st, near = states_of(y, 1.0, 150, 25)
assert len(b) == 4 and abs(b[1] - 2000) < 30 and abs(b[2] - 4800) < 30, b
b, st, near = states_of(15 + 0.8 * np.sin(np.arange(6800) / 900) + rng0.normal(0, .5, 6800), 1.0, 150, 25)
assert b == [0, 6800], b                                                  # slow drift: one state
b, st, near = states_of(np.full(600, 10.0) + np.random.default_rng(0).normal(0, .3, 600), 1.0, 150, 25)
assert b == [0, 600] and (st == 1).all() and not near.any()

# HelixTracker: O(i)...N(i+4) present in half of the frames
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
nres = 12
uu = mda.Universe.empty(2 * nres, n_residues=nres, atom_resindex=np.repeat(np.arange(nres), 2),
                        trajectory=False)
uu.add_TopologyAttr("names", ["N", "O"] * nres)
uu.add_TopologyAttr("resids", np.arange(1, nres + 1))
uu.add_TopologyAttr("resnames", ["ALA"] * nres)
XX = np.zeros((10, 2 * nres, 3), np.float32)
for f in range(10):
    for i in range(nres):
        XX[f, 2 * i] = [0, 0, 10.0 * i]                                   # N of residue i
        XX[f, 2 * i + 1] = [0, 0, 10.0 * (i + 4) + (2.9 if f < 5 else 6.0)]  # O(i) near N(i+4)?
uu.load_new(XX, format=MemoryReader)
ht = HelixTracker(uu.residues, [9], 10, lo=-8, hi=-1)
for i, ts in enumerate(uu.trajectory):
    ht.update(i)
fr_h = ht.fraction(0, -8, -1)
assert np.allclose(fr_h[:5], 1) and np.allclose(fr_h[5:], 0), fr_h

u, sf_lower, sf_upper, pairs, groups = build()
with tempfile.TemporaryDirectory() as tmp:
    out = run_gate_states(u, pairs, sf_lower, sf_upper, tmp, subunit_groups=groups)
    s = pd.read_csv(out / "states.csv")
    assert (s.diagonal == "A_C").sum() == 1 and (s.diagonal == "B_D").sum() == 2
    assert (s.diagonal == "joint").sum() == 2
    bd = s[s.diagonal == "B_D"].set_index("state")
    assert abs(bd.loc[2, "B_D"] - bd.loc[1, "B_D"] - 2.5) < 0.2
    assert abs(bd.loc[1, "last_frame"] - 449) <= 5
    assert bd.loc[1, "n_used"] < bd.loc[1, "n_frames"]                 # frames near the jump left out
    ac = s[s.diagonal == "A_C"].iloc[0]
    assert ac.first_frame == 0 and ac.last_frame == 899
    assert bd.loc[2, "r_gate_F71"] - bd.loc[1, "r_gate_F71"] > 1.3
    assert bd.loc[2, "kink_G189_F71"] > bd.loc[1, "kink_G189_F71"] + 2
    fr = pd.read_csv(out / "frames.csv.gz")
    assert len(fr) == 900 and set(fr.state_BD) == {1, 2} and set(fr.state_AC) == {1}
    assert {"bend_G189_A", "bend_G180_D", "seg_G189_B", "seg_G189_rad_B", "post_rad_C", "post_A",
            "bend_ref_A", "helix_G189_B", "helix_G180_A"} <= set(fr.columns)
    assert fr.helix_G189_B.between(0, 1).all()
    assert {"bprof-3_B", "bprof+6_A", "bprof-16_D", "tilt_up_rad_B", "tilt_end_B", "tilt_end_tan_C"} <= set(fr.columns)
    bp = pd.read_csv(out / "bend_profile.csv")
    assert set(bp.position_girk2) == set(range(176, 199)) and "tilt_end_rad_F71" in s and "post_rad_F71" in s
    assert all((out / f"states_{k}_{d}.png").exists() for k in ("tilt", "bend_profile")
               for d in ("A_C", "B_D", "joint"))
    assert "post_sd_F71" in s and "seg_G189_rad_F71" in s and "ctl_G189_F71" not in s
    assert "kink_G189_sd_F71" in s and "bend_G180_sd_F151" in s
    assert all((out / f"states_bending_{d}.png").exists() for d in ("A_C", "B_D", "joint"))
    assert (out / "states_trace.png").exists() and json.load(open(out / "run_info.json"))["n_frames"] == 900
print("test_gate_states: OK")
