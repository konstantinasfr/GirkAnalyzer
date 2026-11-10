#!/usr/bin/env python3
"""
compare_hbc_girk1_diagonal.py
=============================
Is the HBC diagonal that contains the GIRK1 subunit narrower in a mutant than
in WT?

The channel is 1 GIRK1 : 3 GIRK2. In hbc_distances.csv the two diagonals are
called A_C and B_D, but which chains those are depends on the
hbc_diagonal_pairs of each system in main.py. This script renames them by
subunit before comparing:

    GIRK1_diag   the diagonal that contains the GIRK1 HBC Phe (GIRK1 vs GIRK2)
    GIRK2_diag   the other one (GIRK2 vs GIRK2)

GIRK1 is the subunit with Asp at the cavity position (GIRK1 D173, GIRK2 N184),
i.e. the Asp in asn_residues; its HBC Phe is that resid + 8.
Default mapping:
    G12 WT, runs <  21   GIRK1 = Asp131 / Phe139, B_D = (139, 464)
                         -> GIRK1_diag = B_D   (default --wt_girk1_col)
    G12_E141Q            GIRK1 = Gln1073 / Phe1113, A_C = (138, 1113)
                         -> GIRK1_diag = A_C   (default --mut_girk1_col)
    G12 WT, runs >= 21   not checked -- find the Asp before using them.
For any other system, check which asn_residues entry is ASP and pass
--wt_girk1_col / --mut_girk1_col accordingly.

PER RUN (one row per run, both systems)
    median / mean distance of each diagonal
    frac_open_<diag>     fraction of frames with that diagonal > threshold
    frac_both_open       both diagonals > threshold
    median_asym          median(GIRK1_diag - GIRK2_diag); negative = the
                         GIRK1 diagonal is the narrower one
    n_permeations        from <system>/permeation_summary/*.csv (same reader
                         as gate_openness.py)

HOW TO READ IT
    GIRK1_diag lower in the mutant than in WT, GIRK2_diag similar
        -> the mutation narrows the GIRK1 side of the gate
    GIRK1_diag equally low in WT
        -> the asymmetry is a property of the 1:3 heteromer, not the mutation

USAGE
    python3 compare_hbc_girk1_diagonal.py \\
        --wt_dir  /media/ziyue/.../girk_analyser_results/G12 \\
        --mut_dir /media/ziyue/.../girk_analyser_results/G12_E141Q \\
        --out_dir /media/ziyue/.../girk_analyser_results/hbc_girk1_comparison

    # only some WT runs, a different threshold, a different mutant label
    python3 compare_hbc_girk1_diagonal.py --wt_dir ... --mut_dir ... \\
        --wt_runs 1-20 --threshold 5.69 --mut_label E141Q

OUTPUTS   <out_dir>/
    hbc_girk1_per_run.csv       one row per run
    hbc_girk1_summary.csv       per system: mean, sd, median over runs, plus a
                                Mann-Whitney test WT vs mutant per metric
    hbc_girk1_open_fraction.png box plot, one dot per run
    hbc_girk1_distance_hist.png pooled distance distributions per diagonal
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gate_openness import (DEFAULT_PERM_COL, expand_runs, find_run_dirs,
                           load_gate, load_permeations, natural_key)

DIAGS = ["GIRK1_diag", "GIRK2_diag"]
METRICS = ["median_GIRK1_diag", "median_GIRK2_diag",
           "frac_open_GIRK1_diag", "frac_open_GIRK2_diag",
           "frac_both_open", "median_asym", "n_permeations"]
COLORS = {"WT": "#4C72B0", "MUT": "#DD8452"}


def other(col):
    return "B_D" if col == "A_C" else "A_C"


def analyse_system(base, runs, girk1_col, threshold, perm_col, label):
    base = Path(base)
    print(f"\n[{label}] {base}   GIRK1 diagonal = {girk1_col}")
    perms, chosen = load_permeations(base, perm_col)
    rows, pooled = [], {d: [] for d in DIAGS}
    for run, gate_dir in sorted(find_run_dirs(base, runs),
                                key=lambda x: natural_key(x[0])):
        d = load_gate(gate_dir / "hbc_distances.csv")
        if d is None:
            print(f"  [warn] {run}: empty hbc_distances.csv — skipped")
            continue
        g1, g2 = d[girk1_col], d[other(girk1_col)]
        pooled["GIRK1_diag"].append(g1)
        pooled["GIRK2_diag"].append(g2)
        o1, o2 = g1 > threshold, g2 > threshold
        rec = perms.get(run, {})
        rows.append(dict(
            system=label, run=run, n_frames=int(g1.size),
            girk1_column=girk1_col, threshold=threshold,
            median_GIRK1_diag=round(float(np.median(g1)), 3),
            median_GIRK2_diag=round(float(np.median(g2)), 3),
            mean_GIRK1_diag=round(float(g1.mean()), 3),
            mean_GIRK2_diag=round(float(g2.mean()), 3),
            frac_open_GIRK1_diag=round(float(o1.mean()), 4),
            frac_open_GIRK2_diag=round(float(o2.mean()), 4),
            frac_both_open=round(float((o1 & o2).mean()), 4),
            median_asym=round(float(np.median(g1 - g2)), 3),
            corr_diagonals=round(float(np.corrcoef(g1, g2)[0, 1]), 3),
            n_permeations=rec.get(chosen, "") if chosen else "",
        ))
        r = rows[-1]
        print(f"  {run:>6}  GIRK1 med {r['median_GIRK1_diag']:5.2f} "
              f"open {r['frac_open_GIRK1_diag']:5.1%} | GIRK2 med "
              f"{r['median_GIRK2_diag']:5.2f} open "
              f"{r['frac_open_GIRK2_diag']:5.1%} | both "
              f"{r['frac_both_open']:5.1%} | perm {r['n_permeations']}")
    pooled = {k: np.concatenate(v) if v else np.array([])
              for k, v in pooled.items()}
    return rows, pooled


def values(rows, metric):
    out = []
    for r in rows:
        v = r.get(metric, "")
        if str(v).strip() != "":
            out.append(float(v))
    return np.array(out)


def summarise(wt_rows, mut_rows, wt_label, mut_label):
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        mannwhitneyu = None
    out = []
    for m in METRICS:
        a, b = values(wt_rows, m), values(mut_rows, m)
        row = dict(metric=m)
        for lab, v in ((wt_label, a), (mut_label, b)):
            row[f"{lab}_n"] = v.size
            row[f"{lab}_mean"] = round(float(v.mean()), 4) if v.size else ""
            row[f"{lab}_sd"] = round(float(v.std(ddof=1)), 4) if v.size > 1 else ""
            row[f"{lab}_median"] = round(float(np.median(v)), 4) if v.size else ""
        p = ""
        if mannwhitneyu is not None and a.size >= 2 and b.size >= 2:
            try:
                p = round(float(mannwhitneyu(a, b, alternative="two-sided").pvalue), 4)
            except ValueError:
                p = ""
        row["mannwhitney_p"] = p
        out.append(row)
    return out


def write_csv(rows, path):
    if not rows:
        print(f"  [warn] nothing to write to {path.name}")
        return
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def plot_open_fraction(wt_rows, mut_rows, wt_label, mut_label, threshold, path):
    panels = [("frac_open_GIRK1_diag", "GIRK1–GIRK2 diagonal"),
              ("frac_open_GIRK2_diag", "GIRK2–GIRK2 diagonal"),
              ("frac_both_open", "Both diagonals")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (m, title) in zip(axes, panels):
        data = [values(wt_rows, m) * 100, values(mut_rows, m) * 100]
        ax.boxplot(data, widths=0.5, showfliers=False,
                   medianprops=dict(color="black"))
        for i, (v, key) in enumerate(zip(data, ("WT", "MUT")), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, s=28,
                       color=COLORS[key], edgecolors="black", linewidths=0.5,
                       zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"{wt_label}\n(n={data[0].size})",
                            f"{mut_label}\n(n={data[1].size})"])
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"% of frames > {threshold} Å")
    axes[0].set_ylim(0, 100)
    fig.suptitle("HBC openness per run, diagonals matched by subunit",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_hist(wt_pool, mut_pool, wt_label, mut_label, threshold, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.linspace(1.5, 12, 64)
    for ax, d, title in zip(axes, DIAGS,
                            ("GIRK1–GIRK2 diagonal", "GIRK2–GIRK2 diagonal")):
        for pool, lab, key in ((wt_pool, wt_label, "WT"),
                               (mut_pool, mut_label, "MUT")):
            if pool[d].size:
                ax.hist(pool[d], bins=bins, density=True, histtype="step",
                        linewidth=2, color=COLORS[key], label=lab)
        ax.axvline(threshold, color="grey", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Min. atom–atom distance (Å)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Density (all runs pooled)")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wt_dir", required=True, help="e.g. .../girk_analyser_results/G12")
    ap.add_argument("--mut_dir", required=True, help="e.g. .../girk_analyser_results/G12_E141Q")
    ap.add_argument("--out_dir", default=None,
                    help="default: <parent of mut_dir>/hbc_girk1_comparison")
    ap.add_argument("--wt_runs", default=None, help='e.g. "1-20" (default: all RUN*)')
    ap.add_argument("--mut_runs", default=None, help='e.g. "1-6" (default: all RUN*)')
    ap.add_argument("--wt_girk1_col", default="B_D", choices=["A_C", "B_D"])
    ap.add_argument("--mut_girk1_col", default="A_C", choices=["A_C", "B_D"])
    ap.add_argument("--threshold", type=float, default=5.6)
    ap.add_argument("--perm_col", default=DEFAULT_PERM_COL)
    ap.add_argument("--wt_label", default="WT")
    ap.add_argument("--mut_label", default="E141Q")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(args.mut_dir).resolve().parent / "hbc_girk1_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    wt_rows, wt_pool = analyse_system(
        args.wt_dir, expand_runs(args.wt_runs) if args.wt_runs else None,
        args.wt_girk1_col, args.threshold, args.perm_col, args.wt_label)
    mut_rows, mut_pool = analyse_system(
        args.mut_dir, expand_runs(args.mut_runs) if args.mut_runs else None,
        args.mut_girk1_col, args.threshold, args.perm_col, args.mut_label)

    print(f"\nWriting to {out_dir}")
    write_csv(wt_rows + mut_rows, out_dir / "hbc_girk1_per_run.csv")
    summary = summarise(wt_rows, mut_rows, args.wt_label, args.mut_label)
    write_csv(summary, out_dir / "hbc_girk1_summary.csv")
    plot_open_fraction(wt_rows, mut_rows, args.wt_label, args.mut_label,
                       args.threshold, out_dir / "hbc_girk1_open_fraction.png")
    plot_hist(wt_pool, mut_pool, args.wt_label, args.mut_label,
              args.threshold, out_dir / "hbc_girk1_distance_hist.png")

    print("\nSummary (mean over runs; Mann-Whitney p, WT vs mutant):")
    for r in summary:
        print(f"  {r['metric']:<22} {args.wt_label}={r[f'{args.wt_label}_mean']!s:<8} "
              f"{args.mut_label}={r[f'{args.mut_label}_mean']!s:<8} p={r['mannwhitney_p']}")


if __name__ == "__main__":
    main()
