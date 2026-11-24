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
# the broad 8+8 measure has an offset of ~18 deg for a straight ideal helix (documented)
k0 = kink_angle(H[None], list(range(h - 8, h)), list(range(h + 1, h + 9)))[0]
assert 15 < k0 < 21, k0

b, st, near = states_of(np.r_[np.full(300, 10.0), np.full(300, 12.0)], 1.0, 150, 25)
assert b == [0, 300, 600] and st[299] == 1 and st[300] == 2 and near[280:320].all() and not near[:270].any()
b, st, near = states_of(np.full(600, 10.0) + np.random.default_rng(0).normal(0, .3, 600), 1.0, 150, 25)
assert b == [0, 600] and (st == 1).all() and not near.any()

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
    assert {"bend_G189_A", "bend_G180_D"} <= set(fr.columns)
    assert "kink_G189_sd_F71" in s and "bend_G180_sd_F151" in s
    assert all((out / f"states_bending_{d}.png").exists() for d in ("A_C", "B_D", "joint"))
    assert (out / "states_trace.png").exists() and json.load(open(out / "run_info.json"))["n_frames"] == 900
print("test_gate_states: OK")
