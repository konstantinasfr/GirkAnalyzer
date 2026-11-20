"""Synthetic check for hbc_state_windows: closed (3 A) / open (5 A) / closed trace with
noise, a short spike and a short dip. Run from the repo root: python3 tests/test_hbc_state_windows.py"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from hbc_state_windows import find_states, parse_manual

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
print("test_hbc_state_windows: OK")
