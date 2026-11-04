"""Ground truth for visit_water.collect_water: water counts vs brute-force distances on a random
synthetic system. Run from the repo root: python3 tests/test_visit_water.py"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from analysis.markov_check import G12_CONFIG
from analysis.visit_water import collect_water
rng = np.random.default_rng(3)
glu, asn = G12_CONFIG["glu_residues"][:2], G12_CONFIG["asn_residues"][:1]
names, rn, rid, rix = [], [], [], []
res = glu + asn
for i, r in enumerate(res):
    for nm in (["CD","OE1","OE2"] if r in glu else ["CG","OD1","ND2"]):
        names.append(nm); rix.append(i)
    rn.append("GLU" if r in glu else "ASN"); rid.append(r)
nw = 300
for k in range(nw):
    for nm in ("O","H1","H2"): names.append(nm); rix.append(len(res)+k)
    rn.append("WAT"); rid.append(2000+k)
for k in range(2):
    names.append("K"); rix.append(len(res)+nw+k); rn.append("K+"); rid.append(1400+k)
u = mda.Universe.empty(len(names), n_residues=len(rid), atom_resindex=rix, trajectory=False)
for a,v in (("names",names),("resnames",rn),("resids",rid)): u.add_TopologyAttr(a,v)
nf = 12
X = rng.uniform(40, 60, (nf, len(names), 3)).astype(np.float32)
u.load_new(X, format=MemoryReader, dimensions=[200,200,200,90,90,90])
events = [dict(ion_id=1400, end_1=2, end_2=7), dict(ion_id=1401, end_1=5, end_2=10)]
ions, sites = collect_water(u, events, asn, glu, 3, "G12")
wO = [i for i,n in enumerate(names) if n=="O"]
bad = 0
for r in ions.itertuples():
    ion = names.index("K") + (0 if events[r.event]["ion_id"]==1400 else 1)
    d = np.linalg.norm(X[r.frame, wO] - X[r.frame, ion], axis=1)
    bad += int((d <= 3.4).sum() != r.n_water)
labs = [c for c in sites.columns if c not in ("run","frame")]
for r in sites.to_dict("records"):
    for j, res_ in enumerate(res):
        idx = [i for i,(n,x) in enumerate(zip(names,rix)) if x==j and n in ("OE1","OE2","OD1","OD2")]
        d = np.linalg.norm(X[r["frame"], wO][:,None] - X[r["frame"], idx][None], axis=2)
        bad += int((d <= 3.5).any(1).sum() != r[labs[j]])
print(("OK      " if bad == 0 else "FAIL    ") + f"water counts ({len(ions)} ion rows, {len(sites)} frames): {bad} mismatches")
sys.exit(bad != 0)
