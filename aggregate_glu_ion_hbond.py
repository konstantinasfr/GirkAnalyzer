#!/usr/bin/env python3
"""
aggregate_glu_ion_hbond.py
==========================
Pools RUN*/glu_ion_hbond/<GLU>/per_frame.csv (analysis/glu_ion_hbond.py) over the runs of one
system. For each glutamate:
    by partner        frames with the side chain-side chain H-bond formed / broken, and the % of
                      them with a K+ at the Glu (within 3.5 A of any atom; of the carboxylate O)
    by n partners     the same for 0, 1, 2, ... bonded partners
two ways: pooled (all frames of all runs) and per run (mean +- SD over the runs with at least
--min_frames frames in that group). Also HIGH / TYPICAL / LOW permeation runs when
<system>/permeation_summary/run_categories_full_permeations_total.csv exists.

USAGE
    python3 aggregate_glu_ion_hbond.py /path/girk_analyser_results/G12

OUTPUT  <system>/glu_ion_hbond_summary/[HIGH|TYPICAL|LOW/]
    <GLU>_by_partner.csv, <GLU>_by_n_partners.csv     pooled + per-run mean/SD, frames, runs
    table_partner_<any|carbox>.csv/.png               every Glu: % frames with K+ at the Glu,
                                                      bond formed vs broken, per partner
    table_npartners_<any|carbox>.csv/.png             every Glu: % with K+ by number of partners
    fig_occupancy / fig_partners / fig_effect _<any|carbox>.png   slide figures (per-run mean +- SD;
                                                      see analysis/glu_ion_plots.py), fig_values_*.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_hbond_by_cavity_ions import _table_png, natural_key, CATEGORY_FILE, AA1
from analysis.glu_ion_plots import make_figures

NONPARTNER = {"frame", "n_k_cavity", "n_k_any", "n_k_carbox", "n_partners"}
KDEF = {"any": ("n_k_any", "a K+ ≤3.5 Å from any Glu atom"),
        "carbox": ("n_k_carbox", "a K+ ≤3.5 Å from the carboxylate O")}


def load(base):
    """{glu: {run: DataFrame}}"""
    out = {}
    for rd in sorted(base.glob("RUN*"), key=lambda d: natural_key(d.name)):
        for f in sorted((rd / "glu_ion_hbond").glob("*/per_frame.csv")):
            out.setdefault(f.parent.name, {})[rd.name] = pd.read_csv(f)
    return out


def _short(res):
    """SER148 -> S148, GLU152_2a -> E152.2a"""
    head, _, chain = res.partition("_")
    s = AA1.get(head[:3], head[:3]) + head[3:]
    return f"{s}.{chain}" if chain else s


def _glu_order(g):
    sub = g.rsplit("_", 1)[-1]
    return (0 if sub.upper().startswith("G") else 1, sub)


def groups_of(df):
    """[(group label, state, mask)] for every partner (formed / broken) and n_partners value."""
    cols = [c for c in df.columns if c not in NONPARTNER]
    out = []
    for c in cols:
        out.append((c, "formed", df[c] == 1))
        out.append((c, "broken", df[c] == 0))
    for n in sorted(df["n_partners"].unique()):
        out.append(("n_partners", str(int(n)), df["n_partners"] == n))
    return out


def stats(runs, min_frames):
    allf = pd.concat(runs.values(), ignore_index=True)
    rows = []
    for name, state, _ in groups_of(allf):
        row = {"partner": name, "state": state}
        g = allf[(allf[name] == 1) if state == "formed" else (allf[name] == 0)] if name != "n_partners" \
            else allf[allf["n_partners"] == int(state)]
        row["n_frames"] = len(g)
        row["n_runs"] = 0
        for k, (col, _) in KDEF.items():
            row[f"pooled_pct_{k}"] = 100.0 * (g[col] > 0).mean() if len(g) else np.nan
            row[f"pooled_mean_n_{k}"] = g[col].mean() if len(g) else np.nan
            per = []
            for df in runs.values():
                gg = df[(df[name] == 1) if state == "formed" else (df[name] == 0)] if name != "n_partners" \
                    else df[df["n_partners"] == int(state)]
                if len(gg) >= min_frames:
                    per.append(100.0 * (gg[col] > 0).mean())
            row[f"run_mean_pct_{k}"] = np.mean(per) if per else np.nan
            row[f"run_sd_pct_{k}"] = np.std(per) if per else np.nan
            row["n_runs"] = len(per)
        rows.append(row)
    return pd.DataFrame(rows)


def tables(per_glu, out_dir, title):
    glus = sorted(per_glu, key=_glu_order)
    for k, (_, desc) in KDEF.items():
        # partner table: rows = Glu-partner, columns = formed / broken
        rows = []
        for g in glus:
            s = per_glu[g]
            for p in [x for x in s["partner"].unique() if x != "n_partners"]:
                r = {"site": f"{_short(p)}–{_short(g)}", "pair": f"{p}_{g}"}
                for state in ("formed", "broken"):
                    q = s[(s["partner"] == p) & (s["state"] == state)].iloc[0]
                    r[f"{state}_pct"] = q[f"pooled_pct_{k}"]
                    r[f"{state}_n_frames"] = int(q["n_frames"])
                rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(out_dir / f"table_partner_{k}.csv", index=False, float_format="%.2f")
        _table_png(t, ["formed", "broken"], f"{title}: % of frames with {desc}\nside chain–side chain H-bond formed vs broken",
                   out_dir / f"table_partner_{k}.png", corner="H-bond →", first_w=1.6)
        # n_partners table: rows = Glu, columns = 0, 1, 2, ...
        ns = sorted({int(x) for g in glus for x in per_glu[g].query("partner == 'n_partners'")["state"]})
        cols = [str(n) for n in ns]
        rows = []
        for g in glus:
            s = per_glu[g].query("partner == 'n_partners'").set_index("state")
            r = {"site": _short(g), "pair": g}
            for c in cols:
                r[f"{c}_pct"] = s.loc[c, f"pooled_pct_{k}"] if c in s.index else np.nan
                r[f"{c}_n_frames"] = int(s.loc[c, "n_frames"]) if c in s.index else 0
            rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(out_dir / f"table_npartners_{k}.csv", index=False, float_format="%.2f")
        _table_png(t, cols, f"{title}: % of frames with {desc}\nby number of side-chain H-bond partners",
                   out_dir / f"table_npartners_{k}.png", corner="partners →", first_w=1.2)


def run_group(data, out_dir, title, min_frames):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_glu = {}
    for glu, runs in data.items():
        s = stats(runs, min_frames)
        per_glu[glu] = s
        s[s["partner"] != "n_partners"].to_csv(out_dir / f"{glu}_by_partner.csv", index=False, float_format="%.2f")
        s[s["partner"] == "n_partners"].to_csv(out_dir / f"{glu}_by_n_partners.csv", index=False, float_format="%.2f")
        show = s[["partner", "state", "n_frames", "n_runs", "pooled_pct_any", "pooled_pct_carbox",
                  "run_mean_pct_carbox", "run_sd_pct_carbox"]]
        print(f"\n{title} — {glu}  ({len(runs)} runs)\n" + show.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    tables(per_glu, out_dir, title)
    make_figures(data, out_dir, title, min_frames)
    print(f"  -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system_dir", help="folder with RUN*/ (e.g. .../girk_analyser_results/G12)")
    ap.add_argument("--min_frames", type=int, default=50,
                    help="a run counts in a group's mean +- SD only with at least this many frames there (default 50)")
    ap.add_argument("--no_categories", action="store_true", help="skip the HIGH/TYPICAL/LOW split")
    args = ap.parse_args()

    base = Path(args.system_dir)
    data = load(base)
    if not data:
        sys.exit(f"no RUN*/glu_ion_hbond/<GLU>/per_frame.csv under {base} -- run main.py --glu_ion_hbond first")
    for glu, runs in data.items():
        print(f"{glu}: {len(runs)} runs ({', '.join(runs)})")
    out = base / "glu_ion_hbond_summary"
    run_group(data, out, base.name, args.min_frames)

    cf = base / CATEGORY_FILE
    if not args.no_categories and cf.is_file():
        c = pd.read_csv(cf)
        cats = {str(r).strip(): str(k).strip().upper() for r, k in zip(c["Run"], c["Category"]) if pd.notna(k)}
        for cat in ("HIGH", "TYPICAL", "LOW"):
            sub = {g: {r: d for r, d in runs.items() if cats.get(r) == cat} for g, runs in data.items()}
            sub = {g: r for g, r in sub.items() if r}
            if sub:
                run_group(sub, out / cat, f"{base.name} [{cat}]", args.min_frames)
    elif not args.no_categories:
        print(f"(no {CATEGORY_FILE}: no HIGH/TYPICAL/LOW split)")


if __name__ == "__main__":
    main()
