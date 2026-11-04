"""visit_hbond.load_bonds on the real hbond_per_frame.csv format (hbond_present, bonds_detail):
backbone-only bonds must not count, carboxylate bonds must. Run: python3 tests/test_hbond_reader.py"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from analysis.visit_hbond import load_bonds

bb = {"donor_res": "GLU141.G1", "donor_atom": "N", "hydrogen": "H", "acceptor_res": "PHE137.G1",
      "acceptor_atom": "O", "dist_HA_A": 2.1, "angle_DHA_deg": 160.0}
sc = {"donor_res": "SER148.2a", "donor_atom": "OG", "hydrogen": "HG", "acceptor_res": "GLU152.2a",
      "acceptor_atom": "OE2", "dist_HA_A": 1.8, "angle_DHA_deg": 170.0}
rows = [(0, []), (1, [bb]), (2, [sc]), (3, [bb, sc])]
with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "RUN1" / "hbond_analysis" / "P"
    f.mkdir(parents=True)
    pd.DataFrame([dict(frame=fr, n_hbonds=len(b), hbond_present=int(bool(b)), bonds_detail=str(b) if b else "")
                  for fr, b in rows]).to_csv(f / "hbond_per_frame.csv", index=False)
    carb = load_bonds(d, {"E": "P"})[(1, "E")]
    anyb = load_bonds(d, {"E": "P"}, carboxylate_only=False)[(1, "E")]
ok = carb == {0: 0, 1: 0, 2: 1, 3: 1} and anyb == {0: 0, 1: 1, 2: 1, 3: 1}
print(("OK      " if ok else "FAIL    ") + f"carboxylate only {carb}, any bond {anyb}")
sys.exit(not ok)
