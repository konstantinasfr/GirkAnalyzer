"""Check for analysis/gate_states_pool.py: the numpy Gaussian mixture finds
known blobs, and the pooling of two synthetic runs (B_D widens at frame 450)
gives two clusters with the right distances and occupancy.
    python3 tests/test_gate_states_pool.py"""
import sys, tempfile; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from analysis.gate_states_pool import gmm_fit, gmm_predict, bic, choose_k, run_gate_states_pool

rng = np.random.default_rng(0)
X = np.vstack([rng.normal([14, 16], 0.4, (3000, 2)), rng.normal([15, 19], 0.4, (2000, 2)),
               rng.normal([16.5, 20.5], 0.4, (1000, 2))])
bics = [bic(gmm_fit(X, k), len(X)) for k in range(1, 7)]
assert choose_k(bics) == 3, bics
g = gmm_fit(X, 3)
assert np.allclose(np.sort(g["mu"][:, 1]), [16, 19, 20.5], atol=0.1), g["mu"]
lab = gmm_predict(X, g)
assert np.mean(lab[:3000] == np.bincount(lab[:3000]).argmax()) > 0.98 and np.bincount(lab[:3000]).argmax() != np.bincount(lab[3000:5000]).argmax()

from synthetic_channel import build
from analysis.gate_states import run_gate_states
with tempfile.TemporaryDirectory() as tmp:
    for i, seed in enumerate((1, 2)):
        u, a, b, p, gr = build(seed=seed)
        run_gate_states(u, p, a, b, Path(tmp) / "G12" / f"RUN{i + 1}", subunit_groups=gr,
                        channel_type="G12")
    out = run_gate_states_pool(tmp)
    cl = pd.read_csv(out / "clusters.csv")
    assert len(cl) == 2, cl
    assert abs(cl.B_D[0] - 15.2) < 0.2 and abs(cl.B_D[1] - 17.7) < 0.2, cl
    assert abs(cl.pct_frames[0] - 50) < 2
    occ = pd.read_csv(out / "occupancy.csv")
    assert len(occ) == 2 and np.allclose(occ.cluster_1, 50, atol=2)
    assert (cl.r_gate_B[1] - cl.r_gate_B[0]) > 1.3
    fc = pd.read_csv(out / "frame_clusters.csv.gz")
    assert len(fc) == 1800 and set(fc.cluster) == {1, 2}
    assert all((out / f).exists() for f in ("map.png", "occupancy.png", "cluster_gates.png", "bic.csv"))
print("test_gate_states_pool: OK")
