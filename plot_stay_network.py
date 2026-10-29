"""
plot_stay_network.py
====================
Network figure of the stay-to-stay chain (analysis/stay_chain.py) of one
system, from the pooled tables of aggregate_stay_chain.py.

Panel A  the cavity residues as nodes, laid out like the channel: GLU ring on
         top, lower ring (ASN/ASP) below, one column per subunit, Entry (the
         ion has left the SF) above and out below. Visits are merged (repeats
         on one residue joined, as in the <level>_merged outputs).
           node size   number of visits; label: residue, visits, median visit time
           arrow       P(next site | current site), all K+ counts together;
                       width ~ probability; arrows below --min_prob are left out,
                       labels below --label_prob
Panel B  P(out next) against the K+ above the ion when it leaves, for one
         site (default: the site most visits leave to 'out' from), real
         (+/- 2 SD, bootstrap over events) and the time-shifted control.

USAGE
-----
    python3 plot_stay_network.py /path/G12/aggregated_stay_chain/t3.5_d10 \\
        [--site 173.G1] [--min_prob 0.12] [--label_prob 0.15] [--title "GIRK1/2 WT"] [--out network.png]
Output (default): <folder>/stay_network.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from analysis.stay_chain import (ENTRY, OUT, CONTROL_SHIFTS, merge_repeat_stays, event_paths,
                                 p_out_by_above, bootstrap_p_out)

ONE_LETTER = {"GLU": "E", "GLH": "E", "GLN": "Q", "ASP": "D", "ASH": "D", "ASN": "N"}
COLORS = {"GLU": "#e8766f", "GLN": "#f2a35e", "ASN": "#5aa5d6", "ASP": "#5cbf6a"}
GREY = "#9a9a9a"
GLU_Y, LOW_Y = 2.6, 0.6
RAD = 0.15


def site_tables(stays, events, max_bin, dt):
    """Merged visits -> (paths, per-site info, site->site counts)."""
    merged = merge_repeat_stays(stays)
    per_event = merged.groupby(["run", "event"]).size()
    events = events.copy()
    events["n_stays"] = [int(per_event.get((r, e), 0)) for r, e in zip(events["run"], events["event"])]
    paths = event_paths(merged, events, "residue", max_bin)

    info = {}
    for site, g in merged.groupby("residue_pdb"):
        frames = (g["end_frame"] - g["start_frame"] + 1).to_numpy()
        info[site] = dict(resname=g["resname"].iloc[0], ring=g["ring"].iloc[0], visits=len(g),
                          median_ns=float(np.median(frames)) * dt)
    counts = {}
    for p in paths:
        for _, fw, _, _, tw, _, _ in p["steps"]:
            counts.setdefault(fw, {}).setdefault(tw, 0)
            counts[fw][tw] += 1
    return merged, events, paths, info, counts


def layout(info):
    """Positions: one column per subunit (suffix after '.'), GLU ring above the lower ring."""
    def suffix(site):
        return site.split(".")[-1]
    subunits = sorted({suffix(s) for s in info}, key=lambda x: (0 if x.upper().startswith("G") else 1, x))
    col = {u: i for i, u in enumerate(subunits)}
    n = max(len(subunits), 1)
    pos = {}
    for site, d in info.items():
        x = col[suffix(site)] - (n - 1) / 2
        pos[site] = (x * 2.0, GLU_Y if d["ring"] == "GLU" else LOW_Y)
    pos[ENTRY] = (0.0, GLU_Y + 1.7)
    pos[OUT] = (0.0, LOW_Y - 1.7)
    return pos


def label(site, info):
    if site in (ENTRY, OUT):
        return site
    return f"{ONE_LETTER.get(info[site]['resname'], '')}{site}"


def draw_network(ax, info, counts, pos, n_events, min_prob, label_prob):
    sizes = {s: 1700 + 2800 * np.sqrt(d["visits"] / max(v["visits"] for v in info.values()))
             for s, d in info.items()}
    sizes[ENTRY] = sizes[OUT] = 2600
    radius = {s: np.sqrt(a) / 2 for s, a in sizes.items()}            # points

    # arrows first, under the nodes
    for src, dests in counts.items():
        total = sum(dests.values())
        for dst, c in sorted(dests.items(), key=lambda kv: kv[1]):
            p = c / total
            if p < min_prob or dst == src or src not in pos or dst not in pos:
                continue
            color = COLORS.get(info.get(src, {}).get("resname"), GREY)
            straight = (src == ENTRY and dst == OUT)
            arrow = FancyArrowPatch(pos[src], pos[dst], arrowstyle="-|>", mutation_scale=12 + 16 * p,
                                    lw=0.8 + 12 * p, color=color, alpha=0.35 + 0.55 * min(p / 0.4, 1),
                                    connectionstyle="arc3,rad=0.0" if straight else f"arc3,rad={RAD}",
                                    shrinkA=radius[src], shrinkB=radius[dst], zorder=1)
            ax.add_patch(arrow)
            if p >= label_prob:
                (x1, y1), (x2, y2) = pos[src], pos[dst]
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                if not straight:                                     # middle of the arc3 curve
                    dx, dy = x2 - x1, y2 - y1
                    mx, my = mx + 0.5 * RAD * dy, my - 0.5 * RAD * dx
                ax.text(mx, my, f"{100 * p:.0f}%", fontsize=8, ha="center", va="center", zorder=4,
                        color=color, fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    for site, (x, y) in pos.items():
        color = COLORS.get(info.get(site, {}).get("resname"), GREY)
        ax.scatter([x], [y], s=sizes[site], color=color, edgecolor="black", linewidth=1.2, zorder=3)
        if site == ENTRY:
            text = f"Entry\n(left the SF)\n{n_events} events"
        elif site == OUT:
            exits = sum(d.get(OUT, 0) for d in counts.values())
            text = f"out\n{exits} exits"
        else:
            d = info[site]
            p_out = counts.get(site, {}).get(OUT, 0) / max(sum(counts.get(site, {}).values()), 1)
            text = (f"{label(site, info)}\n{d['visits']} visits\n{d['median_ns']:.1f} ns"
                    f"\nout {100 * p_out:.0f}%")
        ax.text(x, y, text, ha="center", va="center", fontsize=8, fontweight="bold", zorder=5)

    ax.text(-0.02, GLU_Y, "GLU ring", transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=10, fontweight="bold")
    ax.text(-0.02, LOW_Y, "lower ring", transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=10, fontweight="bold")
    xs = [x for x, _ in pos.values()]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(LOW_Y - 2.4, GLU_Y + 2.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"A  Jumps between cavity sites (arrow width = P(next | site), shown if >= {100 * min_prob:.0f}%)\n"
                 "node size = visits; in the node: visits, median visit time, P(out next)", fontsize=10, loc="left")


def draw_gate(ax, paths, ctrl_paths, site, label_site, max_bin, n_boot, rng):
    bins = [str(k) for k in range(max_bin)] + [f"{max_bin}+"]
    real = p_out_by_above(paths, site, bins)
    sd = bootstrap_p_out(paths, site, bins, n_boot, rng)
    ctrl = p_out_by_above(ctrl_paths, site, bins)
    x = np.arange(len(bins))
    n = np.array([k for _, k in real])
    ok = n >= 5
    p = np.array([v for v, _ in real], dtype=float)
    pc = np.array([v for v, _ in ctrl], dtype=float)
    lo = np.minimum(200 * sd, 100 * p)
    hi = np.minimum(200 * sd, 100 - 100 * p)
    ax.errorbar(x[ok], 100 * p[ok], yerr=[lo[ok], hi[ok]], marker="o", ms=7, lw=2, capsize=5,
                color="#2f7d3a", label="real (+/- 2 SD)")
    ax.plot(x[ok], 100 * pc[ok], marker="x", ls="--", lw=1.5, color="0.4",
            label="control (other K+ from another time)")
    for xi, pi, ni in zip(x[ok], p[ok], n[ok]):
        ax.text(xi + 0.1, 100 * pi + 2, f"n={ni}", fontsize=8, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel("K+ above the ion when it leaves")
    ax.set_ylabel("% going out next")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"B  Exit from {label_site}", fontsize=10, loc="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="aggregated_stay_chain/<settings> (holds stays_all.csv, events_all.csv)")
    ap.add_argument("--site", help="site of panel B (default: the one most visits leave to 'out' from)")
    ap.add_argument("--min_prob", type=float, default=0.12, help="smallest jump probability drawn")
    ap.add_argument("--label_prob", type=float, default=0.15, help="smallest jump probability labelled")
    ap.add_argument("--max_bin", type=int, default=4)
    ap.add_argument("--frame_duration", type=float, default=0.075)
    ap.add_argument("--title")
    ap.add_argument("--out")
    args = ap.parse_args()

    folder = Path(args.folder)
    stays, events = pd.read_csv(folder / "stays_all.csv"), pd.read_csv(folder / "events_all.csv")
    merged, events, paths, info, counts = site_tables(stays, events, args.max_bin, args.frame_duration)
    ctrl_paths = []
    for j in range(1, len(CONTROL_SHIFTS) + 1):
        if f"n_above_c{j}" in merged:
            ctrl_paths += event_paths(merged, events, "residue", args.max_bin, above_col=f"n_above_c{j}")
    site = args.site or max(info, key=lambda s: counts.get(s, {}).get(OUT, 0))

    fig = plt.figure(figsize=(17, 8.5))
    ax_net = fig.add_axes([0.06, 0.02, 0.62, 0.86])
    ax_gate = fig.add_axes([0.75, 0.16, 0.23, 0.6])
    pos = layout(info)
    draw_network(ax_net, info, counts, pos, len(events), args.min_prob, args.label_prob)
    draw_gate(ax_gate, paths, ctrl_paths, site, label(site, info), args.max_bin, 200,
              np.random.default_rng(0))
    n_runs = events["run"].nunique()
    fig.suptitle(f"{args.title or folder.parent.parent.name}: K+ permeation through the cavity as jumps "
                 f"between binding sites ({n_runs} runs, {len(events)} permeations, {len(merged)} visits)",
                 fontsize=12, fontweight="bold")
    out = Path(args.out) if args.out else folder / "stay_network.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
