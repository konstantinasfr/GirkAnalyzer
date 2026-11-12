#!/usr/bin/env python3
"""
aggregate_hbond_by_cavity_ions.py
=================================
Pools RUN*/hbond_by_cavity_ions/<pair>/per_frame.csv (analysis/hbond_by_cavity_ions.py) over the
runs of one system: per number of K+ in the Glu-Asn cavity (0..4, 5+), the % of frames with the
H-bond, two ways --
    pooled       all frames of all runs together (frames with the bond / frames in the bin)
    per run      the % in each run, then mean +- SD over the runs that have at least
                 --min_frames frames in that bin (a run with 3 frames in a bin says little)
Also split into HIGH / TYPICAL / LOW permeation runs when
<system>/permeation_summary/run_categories_full_permeations_total.csv exists.

USAGE
    python3 aggregate_hbond_by_cavity_ions.py /path/girk_analyser_results/G12            # every pair
    python3 aggregate_hbond_by_cavity_ions.py /path/girk_analyser_results/G12 --pair SER148_2a_GLU152_2a

OUTPUT  <system>/hbond_by_cavity_ions_summary/<pair>/[HIGH|TYPICAL|LOW/]
    by_ion_count_pooled.csv     bin, n_frames, n_runs, pct_any, pct_sc_any, pct_sc_sc
    by_ion_count_per_run.csv    one row per run and bin
    by_ion_count_mean_sd.csv    bin, n_runs_used, mean/sd of pct_any, pct_sc_any, pct_sc_sc
    by_ion_count_<type>.png     pooled bars (n frames on top) with the per-run mean +- SD
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.hbond_by_cavity_ions import BOND_TYPES, MAX_K, bin_label, bin_order, summarize

CATEGORY_FILE = "permeation_summary/run_categories_full_permeations_total.csv"
YLAB = {"any": "% frames with any H-bond",
        "sc_any": "% frames with a side-chain H-bond",
        "sc_sc": "% frames with side chain-side chain H-bond"}


def natural_key(name):
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def load_runs(base, pair):
    runs = {}
    for rd in sorted(base.glob("RUN*"), key=lambda d: natural_key(d.name)):
        f = rd / "hbond_by_cavity_ions" / pair / "per_frame.csv"
        if f.is_file():
            runs[rd.name] = pd.read_csv(f, usecols=["frame", "n_k_cavity"] + BOND_TYPES)
    return runs


def aggregate(runs, out_dir, title, min_frames, max_k=MAX_K):
    out_dir.mkdir(parents=True, exist_ok=True)
    order = bin_order(max_k)

    # pooled: all frames together
    allf = pd.concat(runs.values(), ignore_index=True)
    pooled = summarize(allf, max_k).drop(columns="pct_of_frames")
    pooled.insert(2, "n_runs", [sum((df["n_k_cavity"].map(lambda n: bin_label(n, max_k)) == b).any()
                                    for df in runs.values()) for b in order])
    pooled.to_csv(out_dir / "by_ion_count_pooled.csv", index=False, float_format="%.2f")

    # per run, then mean +- SD over runs with enough frames in the bin
    per_run = pd.concat([summarize(df, max_k).assign(run=r) for r, df in runs.items()], ignore_index=True)
    per_run = per_run[["run"] + [c for c in per_run.columns if c != "run"]]
    per_run.to_csv(out_dir / "by_ion_count_per_run.csv", index=False, float_format="%.2f")
    rows = []
    for b in order:
        g = per_run[(per_run["bin"] == b) & (per_run["n_frames"] >= min_frames)]
        row = {"bin": b, "n_runs_used": len(g)}
        for bt in BOND_TYPES:
            row[f"mean_pct_{bt}"] = g[f"pct_{bt}"].mean() if len(g) else np.nan
            row[f"sd_pct_{bt}"] = g[f"pct_{bt}"].std(ddof=0) if len(g) else np.nan
        rows.append(row)
    msd = pd.DataFrame(rows)
    msd.to_csv(out_dir / "by_ion_count_mean_sd.csv", index=False, float_format="%.2f")

    for bt in BOND_TYPES:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        x = np.arange(len(order))
        v = pooled[f"pct_{bt}"].fillna(0).values
        ax.bar(x, v, width=0.6, color="#2f5d9e", label="pooled (all frames)")
        m, s = msd[f"mean_pct_{bt}"].values, msd[f"sd_pct_{bt}"].values
        ok = ~np.isnan(m)
        ax.errorbar(x[ok], m[ok], yerr=s[ok], fmt="o", color="#c8651b", ms=5, capsize=4,
                    label=f"per-run mean ± SD (runs with ≥{min_frames} frames)")
        for xi, vi, n, nr in zip(x, v, pooled["n_frames"], msd["n_runs_used"]):
            ax.text(xi, max(vi, 0) + 3, f"{vi:.1f}%\n{n} fr, {nr} runs" if n else "no frames",
                    ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_xlabel("K+ in the Glu-Asn cavity")
        ax.set_ylabel(YLAB[bt])
        ax.set_ylim(0, 130)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / f"by_ion_count_{bt}.png", dpi=150)
        plt.close(fig)

    show = pooled[["bin", "n_frames", "n_runs", "pct_sc_sc", "pct_sc_any", "pct_any"]].copy()
    show["per_run_sc_sc"] = [f"{a:.1f} ± {b:.1f} ({n})" if n else "-"
                             for a, b, n in zip(msd["mean_pct_sc_sc"], msd["sd_pct_sc_sc"], msd["n_runs_used"])]
    print(f"\n{title}\n" + show.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"  -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system_dir", help="folder with RUN*/ (e.g. .../girk_analyser_results/G12)")
    ap.add_argument("--pair", default=None, help="pair folder name, e.g. SER148_2a_GLU152_2a (default: all)")
    ap.add_argument("--min_frames", type=int, default=50,
                    help="a run counts in a bin's mean +- SD only with at least this many frames there (default 50)")
    ap.add_argument("--no_categories", action="store_true", help="skip the HIGH/TYPICAL/LOW split")
    args = ap.parse_args()

    base = Path(args.system_dir)
    pairs = [args.pair] if args.pair else sorted({p.name for p in base.glob("RUN*/hbond_by_cavity_ions/*")
                                                   if (p / "per_frame.csv").is_file()})
    if not pairs:
        sys.exit(f"no RUN*/hbond_by_cavity_ions/<pair>/per_frame.csv under {base} -- run main.py "
                 f"--hbond_cavity_pairs first")

    cats = {}
    cf = base / CATEGORY_FILE
    if not args.no_categories and cf.is_file():
        c = pd.read_csv(cf)
        cats = {str(r).strip(): str(k).strip().upper() for r, k in zip(c["Run"], c["Category"]) if pd.notna(k)}
    elif not args.no_categories:
        print(f"(no {CATEGORY_FILE}: no HIGH/TYPICAL/LOW split)")

    for pair in pairs:
        runs = load_runs(base, pair)
        if not runs:
            print(f"{pair}: no runs")
            continue
        out = base / "hbond_by_cavity_ions_summary" / pair
        print(f"\n=== {pair}: {len(runs)} runs ({', '.join(runs)}) ===")
        aggregate(runs, out, f"{pair}  ({base.name}, {len(runs)} runs)", args.min_frames)
        for cat in ("HIGH", "TYPICAL", "LOW"):
            sub = {r: d for r, d in runs.items() if cats.get(r) == cat}
            if sub:
                aggregate(sub, out / cat, f"{pair} [{cat}]  ({base.name}, {len(sub)} runs)", args.min_frames)


if __name__ == "__main__":
    main()
