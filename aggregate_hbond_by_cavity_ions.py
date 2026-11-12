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
and <system>/hbond_by_cavity_ions_summary/tables/[HIGH|TYPICAL|LOW/]table_<type>.csv/.png:
    every pair in one table, rows = pairs, columns = K+ in the cavity and all, each cell =
    % of frames with the bond (all runs pooled) and the number of frames
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


AA1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "HIE": "H", "HID": "H", "HIP": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M",
       "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
TYPE_TITLE = {"sc_sc": "side chain–side chain H-bond", "sc_any": "H-bond with a side chain",
              "any": "any H-bond (incl. backbone)"}


def pair_label(pair):
    """SER148_2a_GLU152_2a -> 'S148–E152.2a';  PHE137_G1_GLU141_G1 -> 'F137–E141.G1'."""
    import re
    m = re.match(r"([A-Z]{3})(\d+)_([^_]+)_([A-Z]{3})(\d+)_([^_]+)$", pair)
    if not m:
        return pair
    ra, na, ca, rb, nb, cb = m.groups()
    a, b = AA1.get(ra, ra), AA1.get(rb, rb)
    return f"{a}{na}–{b}{nb}.{cb}" if ca == cb else f"{a}{na}.{ca}–{b}{nb}.{cb}"


def _row_order(pair):
    sub = pair.rsplit("_", 1)[-1]
    return (0 if sub.upper().startswith("G") else 1, sub, pair)


def combined_tables(frames_by_pair, out_dir, title, max_k=MAX_K):
    """One table per bond type: rows = pairs, columns = K+ in the cavity (0 .. 5+) and all;
    each cell = % of frames with the bond and the number of frames. CSV + PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    order = bin_order(max_k)
    pairs = sorted(frames_by_pair, key=_row_order)
    tabs = {p: summarize(frames_by_pair[p], max_k).set_index("bin") for p in pairs}
    # leave out K+ bins without frames in any pair
    cols = [b for b in order if any(tabs[p].loc[b, "n_frames"] > 0 for p in pairs)] + ["all"]
    for bt in BOND_TYPES:
        rows = []
        for p in pairs:
            row = {"site": pair_label(p), "pair": p}
            df = frames_by_pair[p]
            for b in cols:
                if b == "all":
                    n, v = len(df), 100.0 * df[bt].mean() if len(df) else np.nan
                else:
                    n, v = int(tabs[p].loc[b, "n_frames"]), tabs[p].loc[b, f"pct_{bt}"]
                row[f"{b}_pct"] = v
                row[f"{b}_n_frames"] = n
            rows.append(row)
        tab = pd.DataFrame(rows)
        tab.to_csv(out_dir / f"table_{bt}.csv", index=False, float_format="%.2f")
        _table_png(tab, cols, f"{title}\n{TYPE_TITLE[bt]}: % of frames, by K+ in the Glu–Asn cavity",
                   out_dir / f"table_{bt}.png")
    print(f"  combined tables -> {out_dir}")


def _table_png(tab, cols, title, out_path):
    from matplotlib.colors import to_rgb
    navy, light, accent = "#1f3b5c", "#ffffff", np.array(to_rgb("#e8a35c"))
    nr, nc = len(tab) + 1, len(cols) + 1
    fig_w, fig_h = 1.6 + 1.25 * len(cols), 0.9 + 0.62 * nr
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.axis("off")
    first_w = 1.0

    def cell(x, y, w, text, fc, color="#1b2233", weight="normal", size=11):
        ax.add_patch(plt.Rectangle((x, y), w, 1, facecolor=fc, edgecolor="#d9d6cf", lw=0.8))
        ax.text(x + w / 2, y + 0.5, text, ha="center", va="center", color=color, fontsize=size,
                fontweight=weight)

    # the frame count goes under the K+ number in the header when it is the same for every pair
    # (same runs); otherwise it stays in each cell
    same_n = all(tab[f"{b}_n_frames"].nunique() == 1 for b in cols)
    cell(0, 0, first_w, "K+ in cavity →" + ("\n(frames)" if same_n else ""), navy, "#ffffff", "bold", 10)
    for j, b in enumerate(cols):
        ax.add_patch(plt.Rectangle((first_w + j, 0), 1, 1, facecolor=navy, edgecolor="#d9d6cf", lw=0.8))
        ax.text(first_w + j + 0.5, 0.36 if same_n else 0.5, b, ha="center", va="center",
                color="#ffffff", fontsize=13, fontweight="bold")
        if same_n:
            ax.text(first_w + j + 0.5, 0.74, f"{int(tab[f'{b}_n_frames'].iloc[0]):,} fr",
                    ha="center", va="center", color="#c9d2e3", fontsize=9.5)
    for i, (_, r) in enumerate(tab.iterrows(), start=1):
        cell(0, i, first_w, r["site"], light, "#c8651b", "bold", 11)
        for j, b in enumerate(cols):
            v, n = r[f"{b}_pct"], int(r[f"{b}_n_frames"])
            if n == 0 or v != v:
                cell(first_w + j, i, 1, "–", light, "#9a968c")
                continue
            t = min(max(v, 0) / 80.0, 1.0)                     # 80 % and above = full shade
            fc = tuple(1 - t * (1 - accent))
            txt = f"{v:.1f}%" if same_n else f"{v:.1f}%\n{n:,} fr"
            cell(first_w + j, i, 1, txt, fc, "#1b2233", "bold" if b == "all" else "normal", 12)
    ax.set_title(title, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


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

    per_group = {"ALL": {}, "HIGH": {}, "TYPICAL": {}, "LOW": {}}
    for pair in pairs:
        runs = load_runs(base, pair)
        if not runs:
            print(f"{pair}: no runs")
            continue
        per_group["ALL"][pair] = pd.concat(runs.values(), ignore_index=True)
        for cat in ("HIGH", "TYPICAL", "LOW"):
            sub = [d for r, d in runs.items() if cats.get(r) == cat]
            if sub:
                per_group[cat][pair] = pd.concat(sub, ignore_index=True)
        out = base / "hbond_by_cavity_ions_summary" / pair
        print(f"\n=== {pair}: {len(runs)} runs ({', '.join(runs)}) ===")
        aggregate(runs, out, f"{pair}  ({base.name}, {len(runs)} runs)", args.min_frames)
        for cat in ("HIGH", "TYPICAL", "LOW"):
            sub = {r: d for r, d in runs.items() if cats.get(r) == cat}
            if sub:
                aggregate(sub, out / cat, f"{pair} [{cat}]  ({base.name}, {len(sub)} runs)", args.min_frames)

    # one table per bond type with every pair (all runs pooled), also per permeation group
    tdir = base / "hbond_by_cavity_ions_summary" / "tables"
    for grp, fb in per_group.items():
        if fb:
            combined_tables(fb, tdir if grp == "ALL" else tdir / grp,
                            f"{base.name}" + ("" if grp == "ALL" else f" [{grp}]"))


if __name__ == "__main__":
    main()
