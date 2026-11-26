"""Check for aggregate_hbond_by_cavity_ions.py --groups: runs split by any run->group table
(here ML297 ligand categories E / D) get their own pooled tables with the right %.
    python3 tests/test_hbond_groups.py"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp) / "G12_ML"
    spec = {"RUN1": ("E", 0.8), "RUN2": ("E", 0.8), "RUN3": ("D", 0.2), "RUN5": ("D", 0.2)}
    for r, (_, f) in spec.items():
        d = base / r / "hbond_by_cavity_ions" / "SER148_2a_GLU152_2a"; d.mkdir(parents=True)
        n = 500; on = (np.arange(n) < f * n).astype(int)
        pd.DataFrame(dict(frame=np.arange(n), n_k_cavity=3, any=on, sc_any=on, sc_sc=on)).to_csv(d / "per_frame.csv", index=False)
    lig = base / "ligand_inter_class"; lig.mkdir()
    pd.DataFrame(dict(run=list(spec), category=[v[0] for v in spec.values()])).to_csv(lig / "ligand_categories.csv", index=False)
    r = subprocess.run([sys.executable, str(ROOT / "aggregate_hbond_by_cavity_ions.py"), str(base),
                        "--groups", str(lig / "ligand_categories.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    t = base / "hbond_by_cavity_ions_summary" / "tables" / "by_ligand"
    for grp, want in (("E", 80.0), ("D", 20.0)):
        tab = pd.read_csv(t / grp / "table_sc_sc.csv")
        assert abs(float(tab["all_pct"].iloc[0]) - want) < 0.01, (grp, tab)
        assert abs(float(tab["3_pct"].iloc[0]) - want) < 0.01, (grp, tab)
    allt = pd.read_csv(base / "hbond_by_cavity_ions_summary" / "tables" / "table_sc_sc.csv")
    assert abs(float(allt["all_pct"].iloc[0]) - 50.0) < 0.01
    assert (base / "hbond_by_cavity_ions_summary" / "SER148_2a_GLU152_2a" / "by_ligand" / "E" / "by_ion_count_pooled.csv").exists()
print("test_hbond_groups: OK")
