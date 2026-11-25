"""Check for aggregate_gate_interactions.py on synthetic runs: pooled % = all frames
with the interaction / all frames, a column missing in a run counts 0, and the
HIGH / TYPICAL / LOW split follows the permeation categories.
    python3 tests/test_aggregate_gate_interactions.py"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
META = [dict(column="F192.2a|F83.2b|pi-pi", gate="F192.2a", gate_subunit="A", partner="F83.2b", resid=10, resname="PHE", subunit="B", pos=-109, interaction="pi-pi"),
        dict(column="F181.G1|F72.G1|CH-pi", gate="F181.G1", gate_subunit="D", partner="F72.G1", resid=20, resname="PHE", subunit="D", pos=-109, interaction="CH-pi")]
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp) / "G12"
    # run: (frames, fraction with pi-pi, fraction with CH-pi or None = column absent), category
    spec = {"RUN1": (100, 0.5, 0.2, "HIGH"), "RUN2": (300, 0.1, None, "LOW"), "RUN3": (200, 0.25, 0.5, "TYPICAL"), "RUN4": (100, 1.0, 0.0, "HIGH")}
    for r, (n, a, b, _) in spec.items():
        d = base / r / "gate_states"; d.mkdir(parents=True)
        fr = pd.DataFrame(dict(frame=np.arange(n), A_C=10.0, B_D=10.0))
        fr[META[0]["column"]] = (np.arange(n) < a * n).astype(int)
        m = [META[0]]
        if b is not None:
            fr[META[1]["column"]] = (np.arange(n) < b * n).astype(int); m.append(META[1])
        fr.to_csv(d / "frames.csv.gz", index=False); pd.DataFrame(m).to_csv(d / "interaction_columns.csv", index=False)
    (base / "permeation_summary").mkdir()
    pd.DataFrame(dict(Run=list(spec), Category=[v[3] for v in spec.values()])).to_csv(base / "permeation_summary/run_categories_full_permeations_total.csv", index=False)
    subprocess.run([sys.executable, str(ROOT / "aggregate_gate_interactions.py"), str(base), "--min_pct", "1"], check=True, capture_output=True)
    out = base / "gate_interactions_pooled"
    A = pd.read_csv(out / "interactions_ALL.csv").set_index("interaction")
    assert abs(A.loc["pi-pi", "pct"] - 100 * (50 + 30 + 50 + 100) / 700) < 2e-3, A
    assert abs(A.loc["CH-pi", "pct"] - 100 * (20 + 0 + 100 + 0) / 700) < 2e-3, A      # RUN2 has no column: 0
    H = pd.read_csv(out / "interactions_HIGH.csv").set_index("interaction")
    assert abs(H.loc["pi-pi", "pct"] - 75.0) < 2e-3 and H.loc["pi-pi", "n_runs"] == 2
    L = pd.read_csv(out / "interactions_LOW.csv").set_index("interaction")
    assert abs(L.loc["pi-pi", "pct"] - 10.0) < 2e-3 and L.loc["CH-pi", "pct"] == 0
    assert abs(H.loc["pi-pi", "run_sd"] - np.std([50, 100], ddof=1)) < 2e-3
    for f in ("interactions_ALL.png", "interactions_HIGH.png", "interactions_by_category.png", "interactions_by_category.csv", "runs.csv"):
        assert (out / f).exists(), f
    import shutil; shutil.copy(out / "interactions_by_category.png", "/tmp/claude-0/-home-user/a251b40a-f58e-5536-8526-74788a10acce/scratchpad/test_gate_by_cat.png")
print("test_aggregate_gate_interactions: OK")
