"""
analysis/hbond_by_cavity_ions.py
================================
H-bond of a residue pair split by how many K+ are in the Glu-Asn cavity, frame by frame.

For EVERY frame of the run:
    1. count the K+ in the cavity: inside main.py's channel 2 cylinder and not in the SF
       (channel 1) -- the same cavity definition as stay_chain.py / site_chain.py
    2. find the H-bonds between the two residues (same criteria as hbond_pair_analysis.py:
       H...A <= 3.5 A, D-H...A >= 120 deg)
Then per bin of K+ in the cavity (0, 1, 2, 3, 4, 5+): number of frames and the % of them with
the H-bond, for three definitions of "bonded":
    any     any H-bond between the two residues (includes backbone-backbone, e.g. the helix
            C=O(i)...H-N(i+4) between a residue and the one 4 after it)
    sc_any  at least one H-bond that uses a side-chain atom
    sc_sc   at least one side chain - side chain H-bond (e.g. Ser OG-H ... OE Glu)

No ion tracking and no events: "of all frames with n K+ in the cavity, X% have the bond".

Output  results_dir/hbond_by_cavity_ions/<RESA><pdb>_<RESB><pdb>/
    per_frame.csv        frame, n_k_cavity, n_hbonds, any, sc_any, sc_sc, bonds_detail
    by_ion_count.csv     bin, n_frames, pct_of_frames, pct_any, pct_sc_any, pct_sc_sc
    by_ion_count.png     % of frames with the side chain - side chain H-bond per bin
Pool the runs with aggregate_hbond_by_cavity_ions.py.

Called from main.py with --hbond_cavity_pairs, e.g.
    python3 main.py --channel_type G12 --run_number 1 --hbond_cavity_pairs "144-148,..."
    python3 main.py --channel_type G12 --run_number 1 --hbond_cavity_pairs glu
("glu" = every Glu of glu_residues with the residue 4 before it: GIRK2 S148-E152, GIRK1 F137-E141)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.hbond_pair_analysis import _find_hbonds, BACKBONE_ATOMS
from analysis.stay_chain import PoreFrames

MAX_K = 5            # last bin is "5+"
BOND_TYPES = ["any", "sc_any", "sc_sc"]


def bin_label(n, max_k=MAX_K):
    return f"{max_k}+" if n >= max_k else str(int(n))


def bin_order(max_k=MAX_K):
    return [str(i) for i in range(max_k)] + [f"{max_k}+"]


def _flags(bonds):
    """(any, sc_any, sc_sc) for the H-bonds of one frame."""
    if not bonds:
        return 0, 0, 0
    sc_any = sc_sc = 0
    for b in bonds:
        d_sc = b["donor_atom"] not in BACKBONE_ATOMS
        a_sc = b["acceptor_atom"] not in BACKBONE_ATOMS
        sc_any |= int(d_sc or a_sc)
        sc_sc |= int(d_sc and a_sc)
    return 1, sc_any, sc_sc


def summarize(df, max_k=MAX_K):
    """by_ion_count table from a per_frame table (also used by the aggregator)."""
    df = df.assign(bin=df["n_k_cavity"].map(lambda n: bin_label(n, max_k)))
    rows = []
    for b in bin_order(max_k):
        g = df[df["bin"] == b]
        row = {"bin": b, "n_frames": len(g),
               "pct_of_frames": 100.0 * len(g) / len(df) if len(df) else np.nan}
        for bt in BOND_TYPES:
            row[f"pct_{bt}"] = 100.0 * g[bt].mean() if len(g) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bins(tab, title, out_path, bond="sc_sc"):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tab))
    vals = tab[f"pct_{bond}"].fillna(0).values
    ax.bar(x, vals, color="#2f5d9e", width=0.6)
    for xi, v, n in zip(x, vals, tab["n_frames"]):
        ax.text(xi, v + 2, f"{v:.1f}%\nn={n}" if n else "no\nframes", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(tab["bin"])
    ax.set_xlabel("K+ in the Glu-Asn cavity")
    ax.set_ylabel({"sc_sc": "% frames with side chain-side chain H-bond",
                   "sc_any": "% frames with a side-chain H-bond",
                   "any": "% frames with any H-bond"}[bond])
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title(title, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_hbond_by_cavity_ions(u, results_dir, channel_type, residue_pairs, channel1, channel2,
                             start_frame=0, end_frame=None, hbond_dist=3.5, hbond_angle=120.0,
                             max_k=MAX_K):
    from analysis.converter import to_pdb

    results_dir = Path(results_dir)
    end_frame = len(u.trajectory) - 1 if end_frame is None else end_frame
    pore = PoreFrames(u, channel1, channel2)

    pairs = []
    for a, b in residue_pairs:
        ra, rb = u.select_atoms(f"resid {a}"), u.select_atoms(f"resid {b}")
        if not len(ra) or not len(rb):
            print(f"  [hbond_by_cavity_ions] resid {a} or {b} not found -- skipped")
            continue
        la = f"{ra.residues[0].resname}{to_pdb(a, channel_type)}"
        lb = f"{rb.residues[0].resname}{to_pdb(b, channel_type)}"
        pairs.append((ra, la, rb, lb))
        print(f"  H-bond vs cavity K+: {la} - {lb}  (resid {a} - {b})")
    if not pairs:
        return

    rows = {i: [] for i in range(len(pairs))}
    k_counts = []
    for ts in tqdm(u.trajectory[start_frame:end_frame + 1], desc="H-bond vs cavity K+"):
        f = ts.frame
        _, in_cav, _ = pore.get(f)
        pore._cache.pop(f, None)               # one pass, nothing to reuse
        n_k = int(in_cav.sum())
        k_counts.append(n_k)
        for i, (ra, la, rb, lb) in enumerate(pairs):
            bonds = _find_hbonds(ra, la, rb, lb, hbond_dist, hbond_angle)
            any_, sc_any, sc_sc = _flags(bonds)
            rows[i].append({"frame": f, "n_k_cavity": n_k, "n_hbonds": len(bonds),
                            "any": any_, "sc_any": sc_any, "sc_sc": sc_sc,
                            "bonds_detail": str(bonds) if bonds else ""})

    base = results_dir / "hbond_by_cavity_ions"
    for i, (ra, la, rb, lb) in enumerate(pairs):
        od = base / f"{la}_{lb}".replace(".", "_")
        od.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows[i])
        df.to_csv(od / "per_frame.csv", index=False)
        tab = summarize(df, max_k)
        tab.to_csv(od / "by_ion_count.csv", index=False, float_format="%.2f")
        plot_bins(tab, f"{la} - {lb}   ({results_dir.name}, {len(df)} frames)", od / "by_ion_count.png")
        print(f"\n  {la} - {lb}  ->  {od}")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
