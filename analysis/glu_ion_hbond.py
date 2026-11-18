"""
analysis/glu_ion_hbond.py
=========================
Per frame, for each cavity glutamate: which side chain - side chain H-bonds it has and how many
K+ are next to it.

For EVERY frame of the run and EVERY residue of glu_residues (GIRK2 Glu152, GIRK1 Glu141; the
same position in a mutant, e.g. Gln141 in E141Q):
    partners   H-bond yes/no with the residues at -4, -46, +24, +32 in the same chain
               (GIRK2: Ser148, Trp106, Gln176, Asn184; GIRK1: Phe137, Trp95, Gln165, Asp173)
               side chain - side chain only (sc_sc), as hbond_by_cavity_ions.py
    n_partners how many of them are bonded
    n_k_any    K+ within 3.5 A of ANY atom of the Glu (the site-chain / visit definition)
    n_k_carbox K+ within 3.5 A of its side-chain oxygens (OE1/OE2; OD1/OD2 for Asp)
    n_k_cavity K+ in the Glu-Asn cavity (channel 2, not the SF), as hbond_by_cavity_ions.py

H-bonds are NOT recomputed when RUN*/hbond_by_cavity_ions/<PARTNER>_<GLU>/per_frame.csv exists
(main.py --hbond_cavity_pairs); only missing pairs or frames are computed here (same criteria:
H...A <= 3.5 A, D-H...A >= 120 deg).

Output  results_dir/glu_ion_hbond/<GLU><pdb>/   (e.g. GLU152_2a)
    per_frame.csv   frame, n_k_cavity, n_k_any, n_k_carbox, <PARTNER> 0/1 ..., n_partners
    summary.csv     per partner: frames with the bond formed / broken and the % of them with
                    a K+ at the Glu (any atom and carboxylate), mean K+ near the Glu
Pool the runs with aggregate_glu_ion_hbond.py.

main.py:  python3 main.py --channel_type G12 --run_number 1 --glu_ion_hbond
"""

from pathlib import Path

import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array
from tqdm import tqdm

from analysis.hbond_pair_analysis import _find_hbonds
from analysis.hbond_by_cavity_ions import _flags
from analysis.stay_chain import PoreFrames

PARTNER_OFFSETS = (-4, -46, 24, 32)       # GIRK2: S148 W106 Q176 N184 from E152; GIRK1: F137 W95 Q165 D173 from E141
CUTOFF = 3.5
CARBOX_O = {"OE1", "OE2", "OD1", "OD2"}


def _label(u, resid, channel_type):
    from analysis.converter import to_pdb
    return f"{u.select_atoms(f'resid {resid}').residues[0].resname}{to_pdb(resid, channel_type)}"


def _load_existing(results_dir, folder):
    f = Path(results_dir) / "hbond_by_cavity_ions" / folder / "per_frame.csv"
    if not f.is_file():
        return None
    t = pd.read_csv(f, usecols=["frame", "sc_sc"])
    return dict(zip(t["frame"].astype(int), t["sc_sc"].astype(int)))


def run_glu_ion_hbond(u, results_dir, channel_type, glu_residues, channel1, channel2,
                      start_frame=0, end_frame=None, cutoff=CUTOFF,
                      hbond_dist=3.5, hbond_angle=120.0):
    results_dir = Path(results_dir)
    end_frame = len(u.trajectory) - 1 if end_frame is None else end_frame
    pore = PoreFrames(u, channel1, channel2)
    ions = pore.ions

    sites = []
    for g in glu_residues:
        g = int(g)
        glu = u.select_atoms(f"resid {g}")
        if not len(glu):
            print(f"  [glu_ion_hbond] resid {g} not found -- skipped")
            continue
        gl = _label(u, g, channel_type)
        carbox = glu.select_atoms("name " + " ".join(sorted(CARBOX_O)))
        partners = []
        for off in PARTNER_OFFSETS:
            p = g + off
            pa = u.select_atoms(f"resid {p}")
            if not len(pa):
                print(f"  [glu_ion_hbond] {gl}: partner resid {p} not found -- skipped")
                continue
            pl = _label(u, p, channel_type)
            folder = f"{pl}_{gl}".replace(".", "_")
            cache = _load_existing(results_dir, folder)
            partners.append(dict(resid=p, atoms=pa, label=pl, col=pl.split(".")[0], cache=cache or {}))
        src = ", ".join(f"{d['label']} ({'existing' if d['cache'] else 'computed here'})" for d in partners)
        print(f"  {gl} (resid {g}): carboxylate O = {list(carbox.names)}; partners: {src}")
        sites.append(dict(resid=g, atoms=glu, carbox=carbox, label=gl, partners=partners, rows=[]))
    if not sites:
        return

    n_computed = 0
    for ts in tqdm(u.trajectory[start_frame:end_frame + 1], desc="Glu ions + H-bonds"):
        f = ts.frame
        _, in_cav, _ = pore.get(f)
        pore._cache.pop(f, None)
        n_cav = int(in_cav.sum())
        kpos = ions.positions
        for s in sites:
            d_any = distance_array(s["atoms"].positions, kpos, box=ts.dimensions).min(axis=0)
            n_any = int((d_any <= cutoff).sum())
            if len(s["carbox"]):
                d_cx = distance_array(s["carbox"].positions, kpos, box=ts.dimensions).min(axis=0)
                n_cx = int((d_cx <= cutoff).sum())
            else:
                n_cx = 0
            row = {"frame": f, "n_k_cavity": n_cav, "n_k_any": n_any, "n_k_carbox": n_cx}
            n_p = 0
            for p in s["partners"]:
                v = p["cache"].get(f)
                if v is None:
                    bonds = _find_hbonds(p["atoms"], p["label"], s["atoms"], s["label"], hbond_dist, hbond_angle)
                    v = _flags(bonds)[2]
                    n_computed += 1
                row[p["col"]] = int(v)
                n_p += int(v)
            row["n_partners"] = n_p
            s["rows"].append(row)

    base = results_dir / "glu_ion_hbond"
    for s in sites:
        od = base / s["label"].replace(".", "_")
        od.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(s["rows"])
        df.to_csv(od / "per_frame.csv", index=False)
        summ = summarize_partners(df, [p["col"] for p in s["partners"]])
        summ.to_csv(od / "summary.csv", index=False, float_format="%.2f")
        print(f"\n  {s['label']} -> {od}")
        print(summ.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if n_computed:
        print(f"\n  ({n_computed} frame-pair H-bond checks computed here; the rest read from hbond_by_cavity_ions/)")


def summarize_partners(df, partner_cols):
    """Per partner and bond state (formed / broken): frames, % with a K+ at the Glu (any atom,
    carboxylate) and mean K+ near the Glu. Plus one row per number of bonded partners."""
    rows = []
    for c in partner_cols:
        for state, g in (("formed", df[df[c] == 1]), ("broken", df[df[c] == 0])):
            rows.append(_row(f"{c}", state, g))
    for n in sorted(df["n_partners"].unique()):
        rows.append(_row("n_partners", str(int(n)), df[df["n_partners"] == n]))
    return pd.DataFrame(rows)


def _row(what, state, g):
    return {"partner": what, "state": state, "n_frames": len(g),
            "pct_k_any": 100.0 * (g["n_k_any"] > 0).mean() if len(g) else np.nan,
            "pct_k_carbox": 100.0 * (g["n_k_carbox"] > 0).mean() if len(g) else np.nan,
            "mean_k_any": g["n_k_any"].mean() if len(g) else np.nan,
            "mean_k_carbox": g["n_k_carbox"].mean() if len(g) else np.nan}
