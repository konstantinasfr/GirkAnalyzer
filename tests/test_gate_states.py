"""Synthetic check for analysis/gate_states.py: B_D widens at frame 450, A_C
does not change -> B_D has 2 states, A_C 1 (the whole run), joint 2.
    python3 tests/test_gate_states.py"""
import sys, json, tempfile; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from synthetic_channel import build
from analysis.gate_states import run_gate_states, states_of

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
    assert (out / "states_trace.png").exists() and json.load(open(out / "run_info.json"))["n_frames"] == 900
print("test_gate_states: OK")
