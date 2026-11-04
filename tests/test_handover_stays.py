"""Ground-truth tests of handover_table_stays (ion tracking with the stay rule), on hand-made frames.
Run from the repo root: python3 tests/test_handover_stays.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import analysis.site_chain as sc
ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(("OK      " if cond else "FAIL    ") + name)

# B. synthetic frames: helper
def frames(occ_by_frame, f_a, f_b, cav_ids, sf_ids=lambda f: ""):
    rows = []
    for f in range(f_a, f_b + 1):
        o = occ_by_frame(f)
        rows.append(dict(run=1, event=0, frame=f, contact="", occupied=";".join(sorted(x.split(":")[0] for x in o.split(";") if x)),
                         occupants=o, cavity_ions=";".join(f"{i}:0.0" for i in cav_ids(f)), sf_ions=sf_ids(f),
                         n_above=0, n_below=0, n_sf=0, **{f"n_{k}_c{j}": 0 for j in (1, 2, 3) for k in ("above", "below")}))
    return pd.DataFrame(rows)
EV = pd.DataFrame([dict(run=1, event=0, ion_id=500, end_1=100, end_2=200, starts_at_frame0=False)])
ST = pd.DataFrame([dict(run=1, event=0, residue_id=1, residue_pdb="152.2a", resname="GLU", ring="GLU", start_frame=130, end_frame=160)])
def run(occ, cav=lambda f: [900, 901], sf=lambda f: ""):
    fr = frames(occ, 100, 200, cav, sf)
    v, e = sc.build_visits(fr, ST, EV, window=4, occ_frames=10)
    return sc.handover_table(e, fr, v, 5, 0.075).iloc[0], sc.handover_table_stays(e, fr, v, 5, 0.075, hold=10).iloc[0]
# B1 flicker: resident at E152.2a all the time except one frame (131) at N184.2c -> old: pushed + went to N184.2c; new: still there
old, new = run(lambda f: "E152.2a:900" if f != 131 else "N184.2c:900")
print("   flicker  old:", old.leave_lag_ns, old.resident_went, old.resident_went_to, "| new:", new.leave_lag_ns, new.resident_went)
chk("flicker: stay rule says 'still there'", new.resident_went == "still there" and new.leave_lag_ns != new.leave_lag_ns)
# B2 passer-by: 901 at E152.2a only 2 frames of the 29-frame trip (110-111), nobody else -> old: resident; new: none
old, new = run(lambda f: "E152.2a:901" if f in (110, 111) else "")
chk("passer-by (2 of 29 frames): no resident with the stay rule (old rule calls it a resident)", new.resident != new.resident and old.resident == 901)
# B3 real push: 900 at E152.2a 101-131, then N184.2c 132-200 -> lag (132-130)*0.075 = +0.15, went N184.2c
old, new = run(lambda f: "E152.2a:900" if f <= 131 else "N184.2c:900")
chk("real push: lag +0.15 ns, went to N184.2c", abs(new.leave_lag_ns - 0.15) < 1e-9 and new.resident_went_to == "N184.2c" and new.resident_went == "lower ring")
# B4 leaves, brushes D173 for 3 frames, then stays at E141 -> destination E141 (old: D173)
def occ4(f):
    if f <= 125: return "E152.2a:900"
    if f <= 128: return "D173.G1:900"
    return "E141.G1:900"
old, new = run(occ4)
print("   brush    old:", old.resident_went_to, "| new:", new.resident_went_to, new.leave_lag_ns)
chk("brushes D173 3 frames then stays at E141: destination E141 (lag -0.3 ns)", new.resident_went_to == "E141.G1" and abs(new.leave_lag_ns + 0.3) < 1e-9)
# B5 leaves the cavity for good at 150 -> out of the cavity
old, new = run(lambda f: "E152.2a:900" if f < 120 else "", cav=lambda f: [900] if f < 150 else [])
chk("free 120-149 then outside: leaves at 120 (lag -0.75 ns), goes out of the cavity", abs(new.leave_lag_ns + 0.75) < 1e-9 and new.resident_went == "out of the cavity")
# B6 goes back into the SF for good at 125
old, new = run(lambda f: "E152.2a:900" if f < 125 else "", cav=lambda f: [900] if f < 125 else [], sf=lambda f: "900" if f >= 125 else "")
chk("into the SF at 125: goes to SF", new.resident_went == "SF")
# B7 two candidates: 900 sits 101-115 (15 frames) then leaves; 901 sits 116-131 (16) -> resident 901 (seen last)
old, new = run(lambda f: "E152.2a:900" if f <= 115 else ("E152.2a:901" if f <= 131 else ""))
chk("two candidates: the one seen last (901)", new.resident == 901)
print("\nALL OK" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
