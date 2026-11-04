"""
visit_water.py
==============
Water at the site-chain stops. Two numbers per frame of every permeation (end_1 .. end_2):
  - the waters in the first hydration shell of the permeating K+ (water O within `shell` A, 3.4 A as
    in ion_hydration_regions.py);
  - the waters around each ring residue: water O within `site_cutoff` A (3.5) of its side-chain
    oxygens (Glu OE1/OE2, Asp OD1/OD2, Asn OD1, Gln OE1; all of the residue's heavy atoms if none).
Collected once per run from the trajectory (run_visit_water, called from main.py after the permeation
events exist), then joined to the pooled site-chain visits by aggregate_visit_water.py.

Questions it answers (aggregate step, analyze()):
  - how many waters the K+ keeps while it stops at each residue, compared to while it moves;
  - how wet each residue is, overall and with its H-bond to the residue above formed / broken
    (optional --pairs, as in aggregate_visit_hbond.py);
  - at visit 1, do new ions go first to a residue more when it is drier or wetter (its waters during
    the new ion's trip, below / above that residue's median), and do they leave the cavity more;
  - ion tracking (stay rule): is the K+ sitting there pushed off or already gone, by the residue's
    water at the arrival (below / above median).

OUTPUTS (per run)  <results_dir>/visit_water/water_ions_run<run>.csv  (run, event, frame, n_water)
                   <results_dir>/visit_water/water_sites_run<run>.csv (run, frame, one column per site)
        (pooled)   <site_chain residue-model folder>/water_visits/
                   stop_hydration.csv, site_water.csv, first_stop_by_water.csv, tracking_by_water.csv,
                   water_visits.txt, stop_hydration.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WATER_SEL = "resname WAT SOL TIP3 HOH and name O OW"
ION_SEL = "resname K+ K"
SIDECHAIN_O = "name OE1 OE2 OD1 OD2"
PUSH_NS = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Per run: collect from the trajectory
# ─────────────────────────────────────────────────────────────────────────────

def site_atoms(universe, resids, channel_type):
    """{site label (as the site chain, e.g. E152.2a): AtomGroup of its side-chain oxygens}."""
    from analysis.converter import convert_to_pdb_numbering
    from analysis.closest_residue_analysis import residue_names_from_topology
    from analysis.site_chain import site_label
    names = residue_names_from_topology(universe, list(resids))
    out = {}
    for r in resids:
        ag = universe.select_atoms(f"resid {int(r)} and ({SIDECHAIN_O})")
        if not len(ag):
            ag = universe.select_atoms(f"resid {int(r)} and not name H*")
        out[site_label(convert_to_pdb_numbering(r, channel_type), names.get(int(r), ""))] = ag
    return out


def collect_water(universe, permeation_events, asn_residues, glu_residues, run, channel_type,
                  shell=3.4, site_cutoff=3.5):
    """(ions, sites) DataFrames for the frames end_1 .. end_2 of every permeation event."""
    from MDAnalysis.lib.distances import capped_distance
    from tqdm import tqdm
    water = universe.select_atoms(WATER_SEL)
    ions = universe.select_atoms(ION_SEL)
    col = {int(r): i for i, r in enumerate(ions.resids)}
    sites = site_atoms(universe, list(glu_residues) + list(asn_residues), channel_type)
    evs = []
    for k, ev in enumerate(permeation_events):
        e1 = int(ev["end_1"] if "end_1" in ev else ev["R1_end"])
        e2 = int(ev["end_2"] if "end_2" in ev else ev["R2_end"])
        if e2 >= e1:
            evs.append((k, int(ev["ion_id"]), e1, e2))
    by_frame = {}
    for k, ion, e1, e2 in evs:
        for f in range(e1, e2 + 1):
            by_frame.setdefault(f, []).append((k, ion))
    ion_rows, site_rows = [], []
    box = universe.dimensions
    for f in tqdm(sorted(by_frame), desc=f"visit_water RUN{run}", unit="frame"):
        universe.trajectory[f]
        box = universe.dimensions
        wpos = water.positions
        for k, ion in by_frame[f]:
            p = ions.positions[col[ion]][None, :]
            pairs = capped_distance(p, wpos, max_cutoff=shell, box=box, return_distances=False)
            ion_rows.append((run, k, f, len(pairs)))
        row = {"run": run, "frame": f}
        for lab, ag in sites.items():
            pairs = capped_distance(ag.positions, wpos, max_cutoff=site_cutoff, box=box, return_distances=False)
            row[lab] = len(set(pairs[:, 1])) if len(pairs) else 0
        site_rows.append(row)
    return (pd.DataFrame(ion_rows, columns=["run", "event", "frame", "n_water"]), pd.DataFrame(site_rows))


def run_visit_water(universe, permeation_events, asn_residues, glu_residues, results_dir, run,
                    channel_type="G12", shell=3.4, site_cutoff=3.5, force_rerun=False):
    """Writes <results_dir>/visit_water/water_ions_run<run>.csv and water_sites_run<run>.csv (cached)."""
    od = Path(results_dir) / "visit_water"
    od.mkdir(parents=True, exist_ok=True)
    fi, fs = od / f"water_ions_run{run}.csv", od / f"water_sites_run{run}.csv"
    if fi.exists() and fs.exists() and not force_rerun:
        print(f"visit_water: using {fi}")
        return
    ions, sites = collect_water(universe, permeation_events, asn_residues, glu_residues, run, channel_type,
                                shell, site_cutoff)
    ions.to_csv(fi, index=False)
    sites.to_csv(fs, index=False)
    print(f"visit_water: wrote {fi} and {fs}")


# ─────────────────────────────────────────────────────────────────────────────
# Pooled: join to the site-chain visits
# ─────────────────────────────────────────────────────────────────────────────

def load_water(base):
    ions, sites = [], []
    for run_dir in Path(base).glob("RUN*"):
        d = run_dir / "visit_water"
        ions += [pd.read_csv(f) for f in d.glob("water_ions_run*.csv")]
        sites += [pd.read_csv(f) for f in d.glob("water_sites_run*.csv")]
    if not ions or not sites:
        return None, None
    return pd.concat(ions, ignore_index=True), pd.concat(sites, ignore_index=True)


def _mse(x):
    x = pd.Series(x).dropna()
    return (x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan, len(x))


def stop_hydration(visits, events, wi):
    """Per residue: waters around the K+ while it stops there (mean over the stop's frames, then over
    stops), all visits and visit 1; and around the K+ while it moves (frames not in a stop)."""
    w = {k: g.set_index("frame")["n_water"] for k, g in wi.groupby(["run", "event"])}
    per_stop = []
    in_stop = {}
    for v in visits.to_dict("records"):
        s_ = w.get((v["run"], v["event"]))
        if s_ is None:
            continue
        fr = s_.loc[(s_.index >= v["start"]) & (s_.index <= v["end"])]
        if len(fr):
            per_stop.append(dict(site=v["site"], visit=v["visit"], n_water=fr.mean()))
        in_stop.setdefault((v["run"], v["event"]), []).append((v["start"], v["end"]))
    moving = []
    for e in events.to_dict("records"):
        s_ = w.get((e["run"], e["event"]))
        if s_ is None:
            continue
        mask = (s_.index > e["end_1"]) & (s_.index <= e["end_2"])
        for a, b in in_stop.get((e["run"], e["event"]), []):
            mask &= ~((s_.index >= a) & (s_.index <= b))
        if mask.any():
            moving.append(s_[mask].mean())
    ps = pd.DataFrame(per_stop)
    rows = []
    for site, g in ps.groupby("site"):
        m, se, n = _mse(g["n_water"])
        m1, se1, n1 = _mse(g[g["visit"] == 0]["n_water"])
        rows.append(dict(site=site, stops=n, water_mean=m, water_sem=se, visit1_stops=n1, visit1_water_mean=m1,
                         visit1_water_sem=se1))
    m, se, n = _mse(moving)
    rows.append(dict(site="moving (no stop)", stops=n, water_mean=m, water_sem=se))
    return pd.DataFrame(rows)


def _trip_mean(ws, run, site, a, b, decision):
    """Mean waters around `site` over frames a..b of the run (the frame `decision` when a > b)."""
    s_ = ws.get(run)
    if s_ is None or site not in s_:
        return np.nan
    col = s_[site]
    fr = col.loc[(col.index >= a) & (col.index <= b)] if b >= a else col.loc[col.index == decision]
    return fr.mean() if len(fr) else np.nan


def analyze(base, site_chain_dir, pairs=None):
    from analysis.visit_hbond import load_bonds, bond_state
    sc = Path(site_chain_dir)
    events = pd.read_csv(sc / "events.csv", keep_default_na=False)
    visits = pd.read_csv(sc / "visits.csv")
    ho_f = sc / "handover_stays" / "handover_visits.csv"
    ho = pd.read_csv(ho_f) if ho_f.exists() else pd.DataFrame()
    wi, wsites = load_water(base)
    od = sc / "water_visits"
    od.mkdir(exist_ok=True)
    lines = [f"WATER AT THE STOPS  ({base})"]
    if wi is None:
        lines.append("no visit_water CSVs in RUN*/visit_water/: run main.py with run_visit_water first")
        (od / "water_visits.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    sites = [c for c in wsites.columns if c not in ("run", "frame")]
    ws = {r: g.drop_duplicates("frame").set_index("frame") for r, g in wsites.groupby("run")}
    lines.append(f"runs with water data: {sorted(ws)}")

    # 1. hydration of the stopping K+
    sh = stop_hydration(visits, events, wi)
    sh.to_csv(od / "stop_hydration.csv", index=False, float_format="%.3f")
    lines.append("\n1. WATERS AROUND THE PERMEATING K+ while it stops at each residue (mean over its stops), vs moving")
    for r in sh.to_dict("records"):
        lines.append(f"  {r['site']:<17} {r['water_mean']:5.2f} +- {r['water_sem']:.2f}  (n={int(r['stops'])})")

    # 2. how wet each residue is (all transit frames of the runs), and by bond state
    bonds = load_bonds(base, pairs) if pairs else {}
    rows = []
    for site in sites:
        vals = pd.concat([g[site] for g in ws.values() if site in g])
        row = dict(site=site, water_mean=vals.mean(), water_median=vals.median())
        if pairs and site in pairs:
            f_, b_ = [], []
            for run, g in ws.items():
                m = bonds.get((run, site))
                if m is None:
                    continue
                st = g.index.map(lambda f: m.get(f, -1))
                f_.append(g[site][st == 1])
                b_.append(g[site][st == 0])
            if f_:
                row["water_bond_formed"] = pd.concat(f_).mean()
                row["water_bond_broken"] = pd.concat(b_).mean()
        rows.append(row)
    swt = pd.DataFrame(rows)
    swt.to_csv(od / "site_water.csv", index=False, float_format="%.3f")
    lines.append("\n2. WATERS AROUND EACH RESIDUE'S SIDE-CHAIN OXYGENS (all frames of the transits)"
                 + ("; H-bond to the residue above formed / broken" if pairs else ""))
    for r in swt.to_dict("records"):
        extra = (f"   bond formed {r['water_bond_formed']:.2f} / broken {r['water_bond_broken']:.2f}"
                 if r.get("water_bond_formed") == r.get("water_bond_formed") and "water_bond_formed" in r else "")
        lines.append(f"  {r['site']:<9} mean {r['water_mean']:5.2f}  median {r['water_median']:4.1f}{extra}")

    # 3. visit 1: does a drier / wetter residue get more new ions?
    first = visits[visits["visit"] == 0].set_index(["run", "event"])
    fs = []
    for ev in events.to_dict("records"):
        key = (ev["run"], ev["event"])
        e1 = int(ev["end_1"])
        f_start = int(first.loc[key, "start"]) if key in first.index else int(ev["end_2"]) + 1
        target = first.loc[key, "site"] if key in first.index else "Exit"
        for site in sites:
            w = _trip_mean(ws, ev["run"], site, e1 + 1, f_start - 1, e1)
            if w == w:
                fs.append(dict(site=site, water=w, to_it=target == site, out=target == "Exit",
                               bond=bond_state(bonds, ev["run"], site, e1 + 1, f_start - 1, e1) if pairs else ""))
    fs = pd.DataFrame(fs)
    # split at the median of the trip waters themselves (balanced halves); dry = at or below it
    fs["wet"] = fs["water"] > fs.groupby("site")["water"].transform("median")
    rows = []
    for site, g0 in fs.groupby("site", sort=False):
        row = dict(site=site, median_water=g0["water"].median())
        for lab, g in (("dry", g0[~g0["wet"]]), ("wet", g0[g0["wet"]])):
            row[f"n_{lab}"], row[f"to_it_{lab}"], row[f"out_{lab}"] = len(g), 100 * g["to_it"].mean(), 100 * g["out"].mean()
        if pairs:
            for st in ("formed", "broken"):
                g = g0[g0["bond"] == st]
                row[f"water_trip_bond_{st}"] = g["water"].mean() if len(g) else np.nan
        rows.append(row)
    fst = pd.DataFrame(rows)
    fst.to_csv(od / "first_stop_by_water.csv", index=False, float_format="%.2f")
    lines.append("\n3. VISIT 1: residue drier (<= median of its waters over the new ions' trips) / wetter: % going to it first, % out")
    for r in fst.to_dict("records"):
        lines.append(f"  {r['site']:<9} (median {r['median_water']:.1f})  dry n={r['n_dry']:>3}: to it {r['to_it_dry']:5.1f}% "
                     f"out {r['out_dry']:5.1f}%  |  wet n={r['n_wet']:>3}: to it {r['to_it_wet']:5.1f}% out {r['out_wet']:5.1f}%")

    # 4. ion tracking (stay rule) by the residue's water at the arrival
    if len(ho):
        m = ho[ho["resident"].notna()].merge(visits[["run", "event", "visit", "start"]].rename(columns={"start": "arrive"}),
                                              on=["run", "event", "visit"])
        m["water"] = [_trip_mean(ws, r, s, int(a), int(a), int(a)) for r, s, a in zip(m["run"], m["site"], m["arrive"])]
        m = m[m["water"].notna()]
        rows = []
        for scope, t in (("all visits", m), ("visit 1", m[m["visit"] == 0])):
            for site, g0 in t.groupby("site", sort=False):
                row = dict(site=site, scope=scope)
                cut = g0["water"].median()
                row["median_water"] = cut
                for lab, g in (("dry", g0[g0["water"] <= cut]), ("wet", g0[g0["water"] > cut])):
                    lag = g["leave_lag_ns"]
                    row[f"n_{lab}"] = len(g)
                    row[f"pushed_{lab}"] = 100 * (lag.abs() <= PUSH_NS).mean() if len(g) else np.nan
                    row[f"left_before_{lab}"] = 100 * (lag < -PUSH_NS).mean() if len(g) else np.nan
                rows.append(row)
        trt = pd.DataFrame(rows)
        trt.to_csv(od / "tracking_by_water.csv", index=False, float_format="%.2f")
        lines.append("\n4. ION TRACKING (stay rule): the K+ sitting there, by the residue's water at the arrival frame "
                     "(dry <= median of those / wet); the rest left > 1 ns after the arrival or stayed")
        for r in trt.to_dict("records"):
            lines.append(f"  {r['scope']:<10} {r['site']:<9} (median {r['median_water']:.1f}) dry n={r['n_dry']:>3}: pushed {r['pushed_dry']:5.1f}% "
                         f"left before {r['left_before_dry']:5.1f}%  |  wet n={r['n_wet']:>3}: pushed {r['pushed_wet']:5.1f}% "
                         f"left before {r['left_before_wet']:5.1f}%")

    s_ = sh[sh["site"] != "moving (no stop)"]
    mv = sh[sh["site"] == "moving (no stop)"]
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(s_) + 3), 4.8))
    x = np.arange(len(s_))
    ax.bar(x, s_["water_mean"], yerr=s_["water_sem"], color="#3B7DD8", capsize=3)
    if len(mv):
        ax.axhline(float(mv["water_mean"].iloc[0]), color="#6B6B6B", ls="--", label="while moving")
        ax.legend(frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels(s_["site"], rotation=30, ha="right")
    ax.set_ylabel("waters around the K+ (first shell)")
    ax.set_title("Hydration of the permeating K+ while it stops at each residue")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(od / "stop_hydration.png", dpi=180)
    plt.close(fig)
    (od / "water_visits.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"visit_water: results in {od}")
