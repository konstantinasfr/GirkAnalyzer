"""
visit_hbond.py
==============
Does the H-bond of a glutamate to the residue above it decide how the glutamate behaves at
visit 1? Joins the per-frame H-bond state of residue pairs (analysis/hbond_pair_analysis.py,
hbond_per_frame.csv of every run) to the site-chain visits (analysis/site_chain.py, pooled by
aggregate_site_chain.py), and splits every result by the bond being formed or broken.

Idea: in GIRK2 the serine above each E152 H-bonds the carboxylate; in GIRK1 a phenylalanine sits
above E141 (no H-bond). If the serine bond makes E152 a storage / refill seat, an E152 whose bond
is broken should behave more like E141.

BOND STATE of a glutamate for a new ion:
  visit 1 (where it goes first)   formed when the bond is there in >= half of the frames of the
                                  new ion's trip from the SF (end_1+1 .. first stop - 1; the frame
                                  end_1 when the trip is empty)
  ion tracking (the K+ sitting there when a new ion lands, handover_stays/)
                                  the same over the trip to that visit
BOND = an H-bond that uses the glutamate's carboxylate oxygens (OE1/OE2), read from bonds_detail;
backbone-only bonds (the helix C=O(i)...H-N(i+4) to the residue 4 before) are not counted.
Frames without H-bond data (runs missing, frames outside the H-bond analysis range) are left out
(state unknown).

OUTPUTS  <site_chain residue-model folder>/hbond_visits/
    bond_occupancy.csv          per glutamate: % of the new ions whose trip sees the bond formed,
                                all and per other-K+ bin
    first_stop_by_bond.csv      per glutamate and K+ bin: ions with the bond formed / broken, % of
                                them going to the glutamate first, % leaving the cavity without a
                                stop; also split by the glutamate holding another K+ (occupied)
    tracking_by_bond.csv        per glutamate, visit 1 and all visits: residents with the bond formed /
                                broken, % pushed off at the arrival (|lag| <= 1 ns), % already left
                                (> 1 ns before), % still there
    hbond_visits.txt, first_stop_by_bond.png, tracking_by_bond.png

USAGE  python3 aggregate_visit_hbond.py <system folder with RUN*/> --site_chain <residue-model folder>
           --pairs pairs.csv
       pairs.csv: columns site,hbond_dir  (site = the glutamate label of the site chain, e.g. E152.2a;
       hbond_dir = the folder of that pair under RUN*/hbond_analysis/, e.g. SER148_2a_GLU152_2a)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUSH_NS = 1.0


CARBOXYLATE_O = {"OE1", "OE2", "OD1", "OD2"}
_ATOMS = re.compile(r"'donor_atom':\s*'([^']*)'.*?'acceptor_atom':\s*'([^']*)'", re.S)


def carboxylate_bond(detail):
    """True when at least one H-bond in the frame's bonds_detail uses a carboxylate oxygen
    (OE1/OE2/OD1/OD2: the Glu side chain; Ser and Phe have none). Backbone-only bonds, like the
    helix C=O(i)...H-N(i+4) between residue i and the Glu at i+4, do not count."""
    if not isinstance(detail, str) or not detail:
        return False
    return any(d in CARBOXYLATE_O or a in CARBOXYLATE_O for d, a in _ATOMS.findall(detail))


_PAIR_DIR = re.compile(r"^[A-Z]+\d+_[A-Za-z0-9]+_([A-Z]+)(\d+)_([A-Za-z0-9]+)$")


def auto_pairs(base):
    """{site: hbond_dir} from the folder names under RUN*/hbond_analysis/, e.g.
    SER148_2a_GLU152_2a -> E152.2a (the site is the second residue, the glutamate)."""
    from analysis.site_chain import site_label
    out = {}
    for d in sorted(Path(base).glob("RUN*/hbond_analysis/*")):
        m = _PAIR_DIR.match(d.name)
        if d.is_dir() and m:
            out.setdefault(site_label(f"{m.group(2)}.{m.group(3)}", m.group(1)), d.name)
    return out


def read_pairs(base, arg):
    """--pairs: 'auto' (from the folder names) or a CSV with columns site,hbond_dir."""
    if arg is None:
        return None
    pairs = auto_pairs(base) if str(arg).lower() == "auto" else dict(pd.read_csv(arg)[["site", "hbond_dir"]].values)
    print("H-bond pairs (site <- hbond_analysis folder):")
    for k, v in pairs.items():
        print(f"  {k:<9} <- {v}")
    if not pairs:
        print("  none found")
    return pairs


def load_bonds(base, pairs, carboxylate_only=True):
    """{(run, site): {frame: 0/1}} from RUN<run>/hbond_analysis/<hbond_dir>/hbond_per_frame.csv.
    carboxylate_only (default): a frame counts as bonded only if a bond uses the Glu carboxylate
    oxygens (from bonds_detail); otherwise any H-bond between the two residues counts."""
    out = {}
    for run_dir in Path(base).glob("RUN*"):
        try:
            run = int(run_dir.name[3:])
        except ValueError:
            continue
        for site, d in pairs.items():
            f = run_dir / "hbond_analysis" / d / "hbond_per_frame.csv"
            if f.exists():
                t = pd.read_csv(f, keep_default_na=False)
                if carboxylate_only and "bonds_detail" in t:
                    on = t["bonds_detail"].map(carboxylate_bond).astype(int)
                else:
                    col = "hbond_present" if "hbond_present" in t else "hbond"
                    on = (pd.to_numeric(t[col], errors="coerce").fillna(0) > 0).astype(int)
                out[(run, site)] = dict(zip(t["frame"].astype(int), on))
    return out


def bond_state(bonds, run, site, a, b, decision):
    """'formed' if the bond is there in >= half of the frames a..b with data (the frame `decision`
    when a > b), 'broken' otherwise, '' when there is no data."""
    m = bonds.get((run, site))
    if m is None:
        return ""
    fr = range(a, b + 1) if b >= a else [decision]
    v = [m[f] for f in fr if f in m]
    if not v:
        return ""
    return "formed" if 2 * sum(v) >= len(v) else "broken"


def _occ_set(text):
    return set(t for t in str(text).split(";") if t and t != "nan")


def bin_label(n, top):
    return f"{top}+" if n >= top else str(int(n))


def first_stop_rows(events, visits, bonds, sites, max_k=5):
    """One row per (ion, glutamate): K+ bin, bond state during the trip from the SF, occupied, target."""
    first = visits[visits["visit"] == 0].set_index(["run", "event"])
    rows = []
    for ev in events.to_dict("records"):
        key = (ev["run"], ev["event"])
        end_1, end_2 = int(ev["end_1"]), int(ev["end_2"])
        f_start = int(first.loc[key, "start"]) if key in first.index else end_2 + 1
        target = first.loc[key, "site"] if key in first.index else "Exit"
        occ = _occ_set(ev.get("occupied_sf", ""))
        for site in sites:
            st = bond_state(bonds, ev["run"], site, end_1 + 1, f_start - 1, end_1)
            if st:
                rows.append(dict(run=ev["run"], event=ev["event"], start=bin_label(ev["sf_count"], max_k), site=site,
                                 bond=st, occupied=site in occ, to_it=target == site, out=target == "Exit"))
    return pd.DataFrame(rows)


def first_stop_table(fs):
    rows = []
    for site, g0 in fs.groupby("site", sort=False):
        for b in ["all"] + sorted(g0["start"].unique()):
            g1 = g0 if b == "all" else g0[g0["start"] == b]
            for occ_lab, g2 in (("all", g1), ("free", g1[~g1["occupied"]]), ("occupied", g1[g1["occupied"]])):
                row = dict(site=site, start=b, occupied=occ_lab)
                for st in ("formed", "broken"):
                    g = g2[g2["bond"] == st]
                    row[f"n_{st}"] = len(g)
                    row[f"to_it_{st}"] = 100 * g["to_it"].mean() if len(g) else np.nan
                    row[f"out_{st}"] = 100 * g["out"].mean() if len(g) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def tracking_rows(events, visits, ho, bonds, sites):
    """One row per visit to a glutamate with a resident (handover_stays): bond state over the trip there."""
    if not len(ho):
        return pd.DataFrame()
    end1 = dict(zip(zip(events["run"], events["event"]), events["end_1"]))
    v = visits.sort_values(["run", "event", "visit"])
    prev_end = v.groupby(["run", "event"])["end"].shift(1)
    v = v.assign(trip_from=[(int(p) + 1) if p == p else int(end1[(r, e)]) + 1
                            for p, r, e in zip(prev_end, v["run"], v["event"])])
    m = ho[ho["site"].isin(sites) & ho["resident"].notna()].merge(
        v[["run", "event", "visit", "start", "trip_from"]].rename(columns={"start": "arrive"}), on=["run", "event", "visit"])
    m["bond"] = [bond_state(bonds, r, s, int(a), int(b) - 1, int(a))
                 for r, s, a, b in zip(m["run"], m["site"], m["trip_from"], m["arrive"])]
    return m[m["bond"] != ""]


def tracking_table(tr):
    rows = []
    for scope, t in (("all visits", tr), ("visit 1", tr[tr["visit"] == 0])):
        for site, g0 in t.groupby("site", sort=False):
            row = dict(site=site, scope=scope)
            for st in ("formed", "broken"):
                g = g0[g0["bond"] == st]
                lag = g["leave_lag_ns"]
                row[f"n_{st}"] = len(g)
                row[f"pushed_{st}"] = 100 * (lag.abs() <= PUSH_NS).mean() if len(g) else np.nan
                row[f"left_before_{st}"] = 100 * (lag < -PUSH_NS).mean() if len(g) else np.nan
                row[f"still_there_{st}"] = 100 * lag.isna().mean() if len(g) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _pair_bars(ax, sites, a, b, ylab, la="bond formed", lb="bond broken"):
    x = np.arange(len(sites))
    for dx, vals, lab, alpha in ((-0.2, a, la, 1.0), (0.2, b, lb, 0.45)):
        ax.bar(x + dx, vals, 0.38, color="#D98A1C", alpha=alpha, label=lab, edgecolor="white")
        for xi, v in zip(x + dx, vals):
            if v == v:
                ax.text(xi, v + 0.8, f"{v:.0f}%", ha="center", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, fontsize=12)
    ax.set_ylabel(ylab)
    ax.legend(frameon=False)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def analyze(base, site_chain_dir, pairs, max_k=5):
    sc = Path(site_chain_dir)
    events = pd.read_csv(sc / "events.csv", keep_default_na=False)
    if "occupied_sf" not in events:
        events["occupied_sf"] = ""
    visits = pd.read_csv(sc / "visits.csv")
    ho_f = sc / "handover_stays" / "handover_visits.csv"
    ho = pd.read_csv(ho_f) if ho_f.exists() else pd.DataFrame()
    sites = list(pairs)
    bonds = load_bonds(base, pairs)
    od = sc / "hbond_visits"
    od.mkdir(exist_ok=True)
    lines = [f"H-BOND STATE AND VISIT 1  ({base})", "pairs: " + ", ".join(f"{s} <- {d}" for s, d in pairs.items()),
             f"runs with H-bond data: {sorted({r for r, _ in bonds})}"]
    fs = first_stop_rows(events, visits, bonds, sites, max_k)
    if not len(fs):
        lines.append("no H-bond data matching the site-chain runs: nothing to analyse")
        (od / "hbond_visits.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    occ = fs.groupby(["site", "start"])["bond"].apply(lambda s: 100 * (s == "formed").mean()).unstack()
    occ["all"] = fs.groupby("site")["bond"].apply(lambda s: 100 * (s == "formed").mean())
    occ.to_csv(od / "bond_occupancy.csv", float_format="%.2f")
    tab = first_stop_table(fs)
    tab.to_csv(od / "first_stop_by_bond.csv", index=False, float_format="%.2f")
    tr = tracking_rows(events, visits, ho, bonds, sites)
    trt = tracking_table(tr) if len(tr) else pd.DataFrame()
    if len(trt):
        trt.to_csv(od / "tracking_by_bond.csv", index=False, float_format="%.2f")
    lines.append("\nBOND FORMED during the new ion's trip from the SF (% of the new ions), all and by other K+")
    for site, r in occ.iterrows():
        lines.append(f"  {site:<9} all {r['all']:5.1f}%   " + "  ".join(f"{c}: {r[c]:5.1f}%" for c in occ.columns
                                                                       if c != "all" and r[c] == r[c]))
    lines.append("\nVISIT 1: % of the new ions going to the glutamate first / leaving without a stop, bond formed vs broken")
    for r in tab[tab["start"] == "all"].to_dict("records"):
        lines.append(f"  {r['site']:<9} {r['occupied']:<8} formed n={r['n_formed']:>3}: to it {r['to_it_formed']:5.1f}% out "
                     f"{r['out_formed']:5.1f}%  |  broken n={r['n_broken']:>3}: to it {r['to_it_broken']:5.1f}% out "
                     f"{r['out_broken']:5.1f}%")
    if len(trt):
        lines.append("\nION TRACKING (stay rule): the K+ sitting there when the new ion lands, bond formed vs broken")
        for r in trt.to_dict("records"):
            lines.append(f"  {r['scope']:<10} {r['site']:<9} formed n={r['n_formed']:>3}: pushed {r['pushed_formed']:5.1f}% "
                         f"left before {r['left_before_formed']:5.1f}% still {r['still_there_formed']:5.1f}%  |  broken "
                         f"n={r['n_broken']:>3}: pushed {r['pushed_broken']:5.1f}% left before {r['left_before_broken']:5.1f}% "
                         f"still {r['still_there_broken']:5.1f}%")
    a = tab[(tab["start"] == "all") & (tab["occupied"] == "all")].set_index("site").reindex(sites)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    _pair_bars(axes[0], sites, a["to_it_formed"], a["to_it_broken"], "% of new ions going there first")
    _pair_bars(axes[1], sites, a["out_formed"], a["out_broken"], "% of new ions leaving without a stop")
    fig.suptitle("Visit 1 by the glutamate's H-bond to the residue above (state during the trip from the SF)")
    fig.tight_layout()
    fig.savefig(od / "first_stop_by_bond.png", dpi=180)
    plt.close(fig)
    if len(trt):
        t = trt[trt["scope"] == "all visits"].set_index("site").reindex(sites)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        _pair_bars(axes[0], sites, t["pushed_formed"], t["pushed_broken"], "% pushed off as the new ion lands")
        _pair_bars(axes[1], sites, t["left_before_formed"], t["left_before_broken"], "% already left (> 1 ns before)")
        fig.suptitle("Ion tracking (stay rule, all visits) by the glutamate's H-bond state")
        fig.tight_layout()
        fig.savefig(od / "tracking_by_bond.png", dpi=180)
        plt.close(fig)
    (od / "hbond_visits.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"visit_hbond: results in {od}")
