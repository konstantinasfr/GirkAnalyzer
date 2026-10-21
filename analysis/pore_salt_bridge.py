#!/usr/bin/env python3
"""
pore_salt_bridge.py
===================
Pore-loop salt bridge behind the selectivity filter: the conserved ARG two
residues after the GYG motif and the GLU behind the filter, per subunit.

    Kir3.1 (G1):  R149 <-> E139      GYG = G145-Y146-G147
    Kir3.2 (G2):  R160 <-> E150      GYG = G156-Y157-G158
(to_pdb numbering of this system - see SALT_BRIDGE_PAIRS. Kir3.2 here is
2 above the R158/E148, GYG 154-156 numbering: G12 RUN3 shows
TETTI-GYG-YRVIT with the GYG at 156-158 through to_pdb.)

NUMBERING
---------
No topology resid is hardcoded. Each sequence number is turned into a
topology resid with converter.from_pdb(), the reverse of converter.to_pdb(),
built by inverting to_pdb() itself so the two can never disagree.
Only "G12" and "G2" are supported (the layouts to_pdb() knows for them):
    G2   4 x 328 residues, offset 54, chains 2a 2c 2b 2d (resid order)
    G12  chain 1 (resid 1-326) G1, offset 42; then 2b, 2a, 2c, offset 54

SUBUNIT TYPE
------------
From the to_pdb() chain label: G1 -> Kir3.1, 2x -> Kir3.2. Chains are the
protein residues grouped by that label. The filter motif (T[IV]GYG) and the
residue 3 after the last GYG glycine are printed per chain for eyeballing; they do not stop
the run.

DIAGNOSTICS (printed, and saved to diagnostics.txt; any failure -> stop)
-----------------------------------------------------------------------
    1. every chain: label, segid, n residues, first/last resid
    2. subunit type + filter-motif context
    3. every selected residue: resname, topology resid, sequence number, chain;
       ARG/GLU resname asserted
    4. +/-7 sequence context, selected residue in [brackets]
    5. atom names of each selected residue; NE NH1 NH2 / OE1 OE2 asserted
    6. protonation: GLH/ASH-style resname, an HE1/HE2 hydrogen on the GLU, or
       a residue charge that is not -1 (GLU) / +1 (ARG) stops the run
    7. first-frame minimum distance per pair

MEASUREMENT (minimum image, box of each frame)
----------------------------------------------
    min_dist      min over {NE,NH1,NH2} x {OE1,OE2}
    state_<c>     at cutoff c (3.5 and 4.0 A):
                    bidentate   two contacts <= c using two different ARG N
                                AND two different GLU O
                    monodentate min_dist <= c but not bidentate
                    broken      min_dist > c
    water_bridge  only when min_dist > 3.5: one water O within 3.5 A of an
                  ARG N and within 3.5 A of a GLU O. n_bridge_waters and
                  their resids are kept.

OUTPUTS   <results_dir>/pore_salt_bridge/
-----------------------------------------
    diagnostics.txt                    everything printed before the scan
    salt_bridge_frames.csv             one row per frame per subunit
    salt_bridge_summary.csv            occupancy per subunit at 3.5 and 4.0 A
    salt_bridge_timeseries.png         one panel per subunit, cutoffs drawn

USAGE - from main.py
--------------------
    from analysis.pore_salt_bridge import run_pore_salt_bridge

    run_pore_salt_bridge(
        universe       = u,
        results_dir    = results_dir,
        channel_type   = channel_type,
        frame_duration = frame_duration,
        start_frame    = start_frame,
        end_frame      = end_frame,
        max_frames     = None,     # e.g. 50 for a quick check
        diagnostics_only = False,  # True: print diagnostics and return
        force_rerun    = False,
    )
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from analysis.converter import to_pdb, from_pdb
except ImportError:
    try:
        from converter import to_pdb, from_pdb
    except ImportError:
        sys.exit("ERROR: could not import to_pdb/from_pdb from analysis/converter.py.")

try:
    from tqdm import tqdm
except ImportError:                                    # tqdm optional
    def tqdm(x, **kwargs):
        return x

MODULE_DIR = "pore_salt_bridge"
SUPPORTED_CHANNELS = ("G12", "G2")

# Sequence numbering (human full length), per subunit type
SALT_BRIDGE_PAIRS = {
    "Kir3.1": {"arg": 149, "glu": 139, "gyg_first_g": 145},
    "Kir3.2": {"arg": 160, "glu": 150, "gyg_first_g": 156},   # = R158/E148 in GYG-154 numbering
}
# residue 3 after the last GYG glycine (GYG-Y-R-[x]) -> subunit type
# Kir3.1 ...GYGYR[Y]IT...   Kir3.2 ...GYGYR[V]IT...
GYG3_TYPE = {"Y": "Kir3.1", "V": "Kir3.2"}
GYG3_OFFSET = 5                                        # from the first GYG glycine
FILTER_MOTIF = re.compile(r"T[IV]GYG")

ARG_N = ("NE", "NH1", "NH2")
GLU_O = ("OE1", "OE2")
CUTOFFS = (3.5, 4.0)
WATER_CUTOFF = 3.5
WATER_SELECTION = "resname WAT HOH SOL TIP3 and name O OW OH2"
CONTEXT = 7

ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ARN": "R", "ASN": "N", "ASP": "D", "ASH": "D",
    "CYS": "C", "CYX": "C", "CYM": "C", "GLN": "Q", "GLU": "E", "GLH": "E",
    "GLY": "G", "HIS": "H", "HIE": "H", "HID": "H", "HIP": "H", "HSD": "H",
    "HSE": "H", "HSP": "H", "ILE": "I", "LEU": "L", "LYS": "K", "LYN": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


class SaltBridgeCheckError(RuntimeError):
    """A diagnostic check failed - nothing is computed."""


class _Log:
    """print() that also keeps the lines for diagnostics.txt."""
    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        print(msg)
        self.lines.append(str(msg))


def _fail(log, msg):
    log(f"\n!!! STOP: {msg}")
    raise SaltBridgeCheckError(msg)


def _split_label(label):
    num, chain = str(label).split(".", 1)
    return int(num), chain


def _subunit_type_from_label(chain):
    return "Kir3.1" if chain == "G1" else "Kir3.2" if chain.startswith("2") else None


# ---------------------------------------------------------------------------
# chain detection
# ---------------------------------------------------------------------------

def find_chains(u, channel_type, log):
    """Protein residues grouped by their to_pdb() chain label."""
    protein = u.select_atoms("protein")
    if len(protein) == 0:
        _fail(log, "no atoms match 'protein' in this topology.")
    groups = {}
    for res in protein.residues:
        label = to_pdb(int(res.resid), channel_type)
        if "." not in label:
            continue                                   # outside every chain range
        groups.setdefault(label.split(".", 1)[1], []).append(res)

    log("1. CHAINS (protein residues grouped by to_pdb chain label)")
    log(f"   {'chain':>5} {'type':>7} {'segid':>7} {'n_res':>5} {'first':>6} {'last':>6}"
        f"  {'to_pdb(first)':>14} {'to_pdb(last)':>13}")
    chains = []
    for label, residues in sorted(groups.items(), key=lambda kv: kv[1][0].resid):
        res = u.residues[[r.ix for r in residues]]
        labels = [to_pdb(int(r), channel_type) for r in res.resids]
        chain = {"label": label, "type": _subunit_type_from_label(label), "residues": res,
                 "segid": ",".join(sorted(set(res.segids))),
                 "seq": "".join(ONE_LETTER.get(rn, "X") for rn in res.resnames),
                 "seqnums": [_split_label(lab)[0] for lab in labels]}
        log(f"   {label:>5} {chain['type']:>7} {chain['segid']:>7} {len(res):>5} "
            f"{res.resids[0]:>6} {res.resids[-1]:>6}  {labels[0]:>14} {labels[-1]:>13}")
        chains.append(chain)
    if len(chains) != 4:
        _fail(log, f"expected 4 chains from to_pdb, found {len(chains)}: {[c['label'] for c in chains]}")
    return chains


def show_filter_motif(chain, log):
    """Sequence evidence, printed only - the subunit type comes from to_pdb."""
    seq = chain["seq"]
    hits = [m.start() + 2 for m in FILTER_MOTIF.finditer(seq)]
    if not hits:
        log(f"   chain {chain['label']:>3} ({chain['type']}):  no {FILTER_MOTIF.pattern} motif found")
        return
    for g in hits:
        res = chain["residues"]
        gyg3 = seq[g + GYG3_OFFSET] if g + GYG3_OFFSET < len(seq) else "-"
        log(f"   chain {chain['label']:>3} ({chain['type']}):  "
            f"{seq[g-5:g]}-{seq[g:g+3]}-{seq[g+3:g+8]}   "
            f"GYG at resid {res.resids[g]}-{res.resids[g+2]} = seq "
            f"{chain['seqnums'][g]}-{chain['seqnums'][g+2]};  last G+3 = {gyg3} "
            f"(sequence suggests {GYG3_TYPE.get(gyg3, '??')})")


# ---------------------------------------------------------------------------
# residue checks
# ---------------------------------------------------------------------------

def _context(chain, idx):
    seq = chain["seq"]
    lo, hi = max(0, idx - CONTEXT), min(len(seq), idx + CONTEXT + 1)
    s = seq[lo:idx] + "[" + seq[idx] + "]" + seq[idx + 1:hi]
    return f"seq {chain['seqnums'][lo]}-{chain['seqnums'][hi-1]}:  {s}"


def check_residue(u, chain, seqnum, role, channel_type, log):
    """role 'ARG' or 'GLU'. Returns the residue."""
    resid = from_pdb(seqnum, chain["label"], channel_type)
    if resid is None:
        _fail(log, f"from_pdb({seqnum}, {chain['label']}, {channel_type}) found no resid.")
    back = to_pdb(resid, channel_type)
    if back != f"{seqnum}.{chain['label']}":
        _fail(log, f"round trip failed: from_pdb -> {resid} -> to_pdb -> {back}")
    resids = list(chain["residues"].resids)
    if resid not in resids:
        _fail(log, f"resid {resid} ({seqnum}.{chain['label']}) is not in chain {chain['label']}.")
    idx = resids.index(resid)
    res = chain["residues"][idx]
    atoms = res.atoms
    names = list(atoms.names)

    log(f"   {role}: resname {res.resname:<4} resid(topology) {resid:<5} seq {seqnum:<4} "
        f"chain {chain['label']}  (segid {res.segid})")
    log(f"        context  {_context(chain, idx)}")
    log(f"        atoms    {' '.join(names)}")
    try:
        charge = float(atoms.charges.sum())
        log(f"        residue charge (topology) {charge:+.3f}")
    except Exception:
        charge = None
        log("        residue charge: not available in topology")

    if role == "ARG":
        if res.resname != "ARG":
            _fail(log, f"{seqnum}.{chain['label']} (resid {resid}) is {res.resname}, not ARG.")
        need = ARG_N
        if charge is not None and abs(charge - 1.0) > 0.05:
            _fail(log, f"ARG {seqnum}.{chain['label']} charge {charge:+.3f}, expected +1 "
                       f"(neutral ARG would kill the interaction).")
    else:
        if res.resname in ("GLH", "GLUP", "GLUH"):
            _fail(log, f"GLU {seqnum}.{chain['label']} is parameterized as {res.resname} "
                       f"(protonated) - the salt bridge cannot form as a charge pair.")
        if res.resname != "GLU":
            _fail(log, f"{seqnum}.{chain['label']} (resid {resid}) is {res.resname}, not GLU.")
        h_on_carboxyl = [n for n in names if n in ("HE1", "HE2")]
        if h_on_carboxyl:
            _fail(log, f"GLU {seqnum}.{chain['label']} carries {h_on_carboxyl} - protonated "
                       f"carboxylate under a GLU resname.")
        if charge is not None and abs(charge + 1.0) > 0.05:
            _fail(log, f"GLU {seqnum}.{chain['label']} charge {charge:+.3f}, expected -1.")
        need = GLU_O
    missing = [n for n in need if n not in names]
    if missing:
        _fail(log, f"{role} {seqnum}.{chain['label']} is missing atoms {missing}.")
    log(f"        required atoms {' '.join(need)}: present")
    return res


# ---------------------------------------------------------------------------
# per-frame measurement
# ---------------------------------------------------------------------------

def _bidentate(d, cutoff):
    """d: 3x2 N-O distance matrix. Two contacts, distinct N and distinct O."""
    c = d <= cutoff
    return any(c[i, 0] and c[j, 1] for i in range(3) for j in range(3) if i != j)


def classify(d, cutoff):
    if d.min() > cutoff:
        return "broken"
    return "bidentate" if _bidentate(d, cutoff) else "monodentate"


def measure_frame(pairs, waters, box):
    from MDAnalysis.lib.distances import distance_array, capped_distance
    rows = []
    for p in pairs:
        n_pos, o_pos = p["n_atoms"].positions, p["o_atoms"].positions
        d = distance_array(n_pos, o_pos, box=box)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        row = {"subunit": p["name"], "min_dist": float(d[i, j]),
               "closest_N": ARG_N[i], "closest_O": GLU_O[j]}
        for nm_i, nm in enumerate(ARG_N):
            for go_i, go in enumerate(GLU_O):
                row[f"d_{nm}_{go}"] = float(d[nm_i, go_i])
        for c in CUTOFFS:
            row[f"state_{c}"] = classify(d, c)
        row["water_bridge"] = False
        row["n_bridge_waters"] = 0
        row["bridge_water_resids"] = ""
        if waters is None:
            row["water_bridge"] = np.nan
            row["n_bridge_waters"] = np.nan
        elif row["min_dist"] > WATER_CUTOFF:
            near_n = capped_distance(n_pos, waters.positions, WATER_CUTOFF,
                                     box=box, return_distances=False)
            cand = np.unique(near_n[:, 1]) if len(near_n) else np.array([], int)
            if len(cand):
                near_o = capped_distance(o_pos, waters.positions[cand], WATER_CUTOFF,
                                         box=box, return_distances=False)
                if len(near_o):
                    bridged = waters[cand[np.unique(near_o[:, 1])]]
                    row["water_bridge"] = True
                    row["n_bridge_waters"] = len(bridged)
                    row["bridge_water_resids"] = ";".join(str(r) for r in bridged.resids)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def summarize(df, log):
    tables = []
    for c in CUTOFFS:
        g = df.groupby("subunit", sort=False)
        t = pd.DataFrame({
            "cutoff_A": c,
            "n_frames": g.size(),
            "occupancy": g["min_dist"].apply(lambda s: float((s <= c).mean())),
            "bidentate": g[f"state_{c}"].apply(lambda s: float((s == "bidentate").mean())),
            "monodentate": g[f"state_{c}"].apply(lambda s: float((s == "monodentate").mean())),
            "broken": g[f"state_{c}"].apply(lambda s: float((s == "broken").mean())),
            "mean_min_dist": g["min_dist"].mean(),
            "median_min_dist": g["min_dist"].median(),
        })
        if c == WATER_CUTOFF:
            t["water_bridged_of_broken"] = g.apply(
                lambda x: float(x.loc[x["min_dist"] > c, "water_bridge"].astype(float).mean())
                if (x["min_dist"] > c).any() else np.nan)
        t = t.reset_index()
        tables.append(t)
        log(f"\nOCCUPANCY SUMMARY  (cutoff {c} A, fractions of frames, per subunit)")
        log(t.drop(columns="cutoff_A").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return pd.concat(tables, ignore_index=True)


def plot_timeseries(df, pairs, out_path, title):
    fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 2.2 * len(pairs)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, pairs):
        s = df[df["subunit"] == p["name"]]
        ax.plot(s["time_ns"], s["min_dist"], lw=0.6, color="#2a6f97")
        wb = s[s["water_bridge"] == True]                  # noqa: E712 (NaN-safe)
        if len(wb):
            ax.scatter(wb["time_ns"], wb["min_dist"], s=8, color="#d1495b", zorder=3,
                       label="water-bridged")
            ax.legend(loc="best", frameon=False, fontsize=8)
        ax.axhline(3.5, color="0.3", lw=1, ls="--")
        ax.axhline(4.0, color="0.6", lw=1, ls=":")
        ax.text(1.005, 3.5, "3.5", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        ax.text(1.005, 4.0, "4.0", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        occ = (s["min_dist"] <= 3.5).mean()
        ax.set_title(f"{p['name']}   occupancy(3.5 A) = {occ:.2f}", fontsize=9, loc="left")
        ax.set_ylabel("min N-O (A)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("time (ns)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def setup_pairs(u, channel_type, log):
    """All diagnostics 1-6. Returns the list of pairs, or raises SaltBridgeCheckError."""
    chains = find_chains(u, channel_type, log)
    log("\n2. SUBUNIT TYPE (from to_pdb label) and filter motif per chain")
    for ch in chains:
        show_filter_motif(ch, log)
    counts = {t: sum(c["type"] == t for c in chains) for t in SALT_BRIDGE_PAIRS}
    log(f"   stoichiometry: {counts}")

    log("\n3-6. SELECTED RESIDUES (resname, numbering, context, atoms, protonation)")
    pairs = []
    for ch in chains:
        nums = SALT_BRIDGE_PAIRS[ch["type"]]
        log(f"\n   --- {ch['label']} ({ch['type']}): R{nums['arg']} <-> E{nums['glu']}")
        arg = check_residue(u, ch, nums["arg"], "ARG", channel_type, log)
        glu = check_residue(u, ch, nums["glu"], "GLU", channel_type, log)
        n_atoms = sum((arg.atoms.select_atoms(f"name {n}") for n in ARG_N[1:]),
                      arg.atoms.select_atoms(f"name {ARG_N[0]}"))
        o_atoms = glu.atoms.select_atoms(f"name {GLU_O[0]}") + glu.atoms.select_atoms(f"name {GLU_O[1]}")
        if len(n_atoms) != 3 or len(o_atoms) != 2:
            _fail(log, f"{ch['label']}: expected 3 ARG N and 2 GLU O, got {len(n_atoms)} / {len(o_atoms)}.")
        pairs.append({"name": f"R{nums['arg']}-E{nums['glu']}.{ch['label']}",
                      "chain": ch["label"], "type": ch["type"],
                      "arg_resid": int(arg.resid), "glu_resid": int(glu.resid),
                      "n_atoms": n_atoms, "o_atoms": o_atoms})
    return pairs


def run_pore_salt_bridge(universe, results_dir, channel_type, frame_duration,
                         start_frame=0, end_frame=None, step=1, max_frames=None,
                         diagnostics_only=False, force_rerun=False):
    u = universe
    ct = channel_type.upper().strip()
    out_dir = Path(results_dir) / MODULE_DIR
    out_csv = out_dir / "salt_bridge_frames.csv"
    print("\n" + "=" * 80)
    print("PORE-LOOP SALT BRIDGE (ARG GYG+4 <-> GLU behind the filter)")
    print("=" * 80)
    if ct not in SUPPORTED_CHANNELS:
        print(f"  {channel_type}: not supported (only {SUPPORTED_CHANNELS}). Skipped.")
        return None
    if out_csv.exists() and not force_rerun and not diagnostics_only:
        print(f"  {out_csv} exists. Nothing was recomputed. Pass force_rerun=True to overwrite.")
        return pd.read_csv(out_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = _Log()
    log(f"channel_type {ct}   topology {u.filename}   trajectory {u.trajectory.filename}")
    log(f"n_frames {len(u.trajectory)}   frame_duration {frame_duration} ns\n")
    try:
        pairs = setup_pairs(u, ct, log)
    finally:
        (out_dir / "diagnostics.txt").write_text("\n".join(log.lines) + "\n")

    waters = u.select_atoms(WATER_SELECTION)
    if len(waters) == 0:
        log(f"\n   WARNING: no waters match '{WATER_SELECTION}' - water bridges NOT computed (NaN).")
        waters = None
    else:
        log(f"\n   waters for bridging: {len(waters)} O atoms")

    if end_frame is None:
        end_frame = len(u.trajectory) - 1
    frames = list(range(start_frame, end_frame + 1, step))
    if max_frames is not None:
        frames = frames[:int(max_frames)]

    log(f"\n7. FIRST-FRAME DISTANCES (frame {frames[0]})")
    ts = u.trajectory[frames[0]]
    for r in measure_frame(pairs, waters, ts.dimensions):
        log(f"   {r['subunit']:<18} min {r['min_dist']:.2f} A  ({r['closest_N']}-{r['closest_O']})"
            f"  state@3.5 {r['state_3.5']:<11} state@4.0 {r['state_4.0']:<11}"
            f"  water_bridge {r['water_bridge']}")
    (out_dir / "diagnostics.txt").write_text("\n".join(log.lines) + "\n")
    log(f"\n   diagnostics saved to {out_dir / 'diagnostics.txt'}")
    if diagnostics_only:
        log("   diagnostics_only=True - no frames scanned.")
        return None

    rows = []
    for f in tqdm(frames, desc="salt bridge"):
        ts = u.trajectory[f]
        for r in measure_frame(pairs, waters, ts.dimensions):
            r["frame"] = f
            r["time_ns"] = f * frame_duration
            rows.append(r)
    df = pd.DataFrame(rows)
    meta = {p["name"]: p for p in pairs}
    df["chain"] = df["subunit"].map(lambda n: meta[n]["chain"])
    df["subunit_type"] = df["subunit"].map(lambda n: meta[n]["type"])
    df["arg_resid"] = df["subunit"].map(lambda n: meta[n]["arg_resid"])
    df["glu_resid"] = df["subunit"].map(lambda n: meta[n]["glu_resid"])
    front = ["frame", "time_ns", "subunit", "chain", "subunit_type", "arg_resid", "glu_resid",
             "min_dist", "closest_N", "closest_O"] + [f"state_{c}" for c in CUTOFFS] + \
            ["water_bridge", "n_bridge_waters", "bridge_water_resids"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(out_csv, index=False)

    summary = summarize(df, print)
    summary.to_csv(out_dir / "salt_bridge_summary.csv", index=False)
    plot_timeseries(df, pairs, out_dir / "salt_bridge_timeseries.png",
                    f"{ct} pore-loop salt bridge - min N(ARG)-O(GLU)")
    print(f"\n  saved: {out_csv}\n         {out_dir / 'salt_bridge_summary.csv'}"
          f"\n         {out_dir / 'salt_bridge_timeseries.png'}")
    return df


if __name__ == "__main__":
    import argparse
    import MDAnalysis as mda
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--top", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--channel_type", required=True, choices=SUPPORTED_CHANNELS)
    ap.add_argument("--frame_duration", type=float, required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--diagnostics_only", action="store_true")
    a = ap.parse_args()
    try:
        run_pore_salt_bridge(mda.Universe(a.top, a.traj), a.results_dir, a.channel_type,
                             a.frame_duration, max_frames=a.max_frames,
                             diagnostics_only=a.diagnostics_only, force_rerun=True)
    except SaltBridgeCheckError:
        sys.exit(1)
