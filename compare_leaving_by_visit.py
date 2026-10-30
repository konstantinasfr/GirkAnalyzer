"""
compare_leaving_by_visit.py
===========================
Compares systems on when their ions leave the GLU/ASN cavity: per visit k
(1 = left without a stop, 2 = after 1 stop, ..., 10+), the % of the permeating
ions that leave at visit k (top) and their mean time from leaving the SF to
leaving the cavity +- standard error (bottom), one bar colour per system.

Input: the events.csv of each system's aggregated site chain, e.g.
    <base>/aggregated_site_chain/t3.5_d10/residue_a5b3/events.csv   (G2: t3.5_d2)

USAGE
-----
    python3 compare_leaving_by_visit.py G12=/path/G12/.../events.csv G2=/path/G2/.../events.csv \\
        [--out leaving_by_visit_compare.png] [--max_visit 10]
Writes the figure and a CSV with the same numbers next to it.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.site_chain import LEAVE_VISITS, _ns_label, leaving_by_visit_table

COLORS = ["#7f7f7f", "#4f81bd", "#c0504d", "#9bbb59", "#8064a2", "#f79646"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("systems", nargs="+", help="LABEL=path/to/events.csv")
    p.add_argument("--out", default="leaving_by_visit_compare.png")
    p.add_argument("--max_visit", type=int, default=LEAVE_VISITS)
    args = p.parse_args(argv)

    tabs = {}
    for item in args.systems:
        label, path = item.split("=", 1)
        ev = pd.read_csv(path)
        tabs[label] = leaving_by_visit_table(ev.assign(transit_frames=ev["transit_ns"]), 1.0, args.max_visit)
        tabs[label].insert(0, "system", label)
        tabs[label].insert(1, "n_permeating", len(ev))

    labels = list(tabs)
    x = np.arange(args.max_visit)
    w = 0.8 / len(labels)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    for i, lab in enumerate(labels):
        t = tabs[lab]
        off = x - 0.4 + w * (i + 0.5)
        c = COLORS[i % len(COLORS)]
        ax1.bar(off, t["pct"], w, color=c, label=f"{lab} ({t['n_permeating'].iloc[0]} permeating ions)")
        for xi, (pc, n) in zip(off, zip(t["pct"], t["n_ions"])):
            if n:
                ax1.text(xi, pc + 0.6, f"{pc:.0f}%", ha="center", va="bottom", fontsize=15, fontweight="bold")
        ax2.bar(off, t["mean_ns"], w, yerr=t["sem_ns"], capsize=4, color=c, error_kw=dict(lw=1.5))
        for xi, (m, e) in zip(off, zip(t["mean_ns"], t["sem_ns"].fillna(0))):
            if np.isfinite(m):
                ax2.text(xi, m + e + 3, _ns_label(m).replace(" ns", ""), ha="center", va="bottom", fontsize=12,
                         rotation=90)
    ax1.set_ylabel("% of permeating ions leaving", fontsize=19)
    ax1.set_ylim(0, max(t["pct"].max() for t in tabs.values()) * 1.25)
    ax1.legend(fontsize=16)
    ax1.set_title("ions leaving the cavity at each visit (visit 1 = no stop)", fontsize=19)
    ax2.set_ylabel("time SF to exit (ns)\nmean ± SE", fontsize=19)
    ax2.set_ylim(0, max(np.nanmax((t["mean_ns"] + t["sem_ns"].fillna(0)).to_numpy()) for t in tabs.values()) * 1.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"visit {v}" for v in tabs[labels[0]]["visit"]], rotation=30, ha="right", fontsize=17)
    for ax in (ax1, ax2):
        ax.tick_params(axis="y", labelsize=17)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    pd.concat(tabs.values()).to_csv(out.with_suffix(".csv"), index=False, float_format="%.3f")
    print(f"compare_leaving_by_visit: {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
