#!/usr/bin/env python3
"""
compare_hbc_diagonals_multi.py
==============================
HBC gate diagonals of several systems in one figure, e.g. G12 WT, G2 and a
G12 mutant together (compare_hbc_girk1_diagonal.py does two systems only).

Each system is given as  LABEL=PATH[:COLUMN]  with PATH the folder that holds
RUN*/hbc_distances.csv. COLUMN says which column of hbc_distances.csv is the
diagonal that contains the GIRK1 subunit:

    B_D    (default) GIRK1 diagonal = B_D, GIRK2 diagonal = A_C
           (G12 WT, D173N, E141Q: see main.py hbc_diagonal_pairs and
           main.py --subunit_check_only)
    A_C    the other way round (old, unswapped CSVs)
    pool   homomer without GIRK1 (G2): both diagonals are GIRK2-GIRK2, so the
           two columns are pooled and the same G2 curve is drawn in both panels
           as the reference

PANELS
    GIRK1-GIRK2 diagonal  |  GIRK2-GIRK2 diagonal   (G2: both diagonals pooled)
    pooled frame density per system (area = 1); dotted line = open threshold.
    Saved twice: plain, and with a 95% band from resampling whole runs (--boot 0: plain only).

OUTPUTS   <out_dir>/
    hbc_diagonals_hist.png / .pdf       the figure, plain lines (as compare_hbc_girk1_diagonal.py)
    hbc_diagonals_hist_band.png / .pdf  the same with the 95% band from resampling whole runs
    hbc_diagonals_per_run.csv       per system, run and diagonal: frames, median,
                                    mean, fraction of frames above the threshold
    hbc_diagonals_summary.csv       per system and diagonal: runs, frames, pooled
                                    median / mean / % open, and the per-run mean +- SD
                                    of the median and of the % open

USAGE
    R=/media/ziyue/.../girk_analyser_results
    python3 compare_hbc_diagonals_multi.py \\
        --system "G12 WT=$R/G12" --system "G2=$R/G2:pool" --system "E141Q=$R/G12_E141Q_4fs"
    # -> $R/hbc_diagonals_G12-WT_G2_E141Q/
    # options: --runs "E141Q=1-6" (only some runs of one system), --threshold 5.69,
    #          --boot 500, --out_dir DIR
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gate_openness import expand_runs, find_run_dirs, load_gate, natural_key

# categorical slots in fixed order (validated palette); line style as secondary encoding
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
STYLES = ["-", "-", "--", "--", "-.", "-."]
PANELS = [("GIRK1", "GIRK1–GIRK2 diagonal"), ("GIRK2", "GIRK2–GIRK2 diagonal")]


def parse_system(spec):
    if "=" not in spec:
        raise SystemExit(f"--system {spec!r}: expected LABEL=PATH[:COLUMN]")
    label, rest = spec.split("=", 1)
    path, col = rest, "B_D"
    head, sep, tail = rest.rpartition(":")
    if sep and tail in ("A_C", "B_D", "pool"):
        path, col = head, tail
    return label.strip(), Path(path), col


def load_system(label, base, col, runs):
    """{"GIRK1": [per-run arrays], "GIRK2": [...]}, and per-run rows."""
    print(f"\n[{label}] {base}   GIRK1 diagonal column = {col}")
    per = {"GIRK1": [], "GIRK2": []}
    names = []
    for run, gate_dir in sorted(find_run_dirs(base, runs), key=lambda x: natural_key(x[0])):
        d = load_gate(gate_dir / "hbc_distances.csv")
        if d is None:
            print(f"  [warn] {run}: empty hbc_distances.csv - skipped")
            continue
        if col == "pool":
            both = np.concatenate([d["A_C"], d["B_D"]])
            per["GIRK1"].append(both)
            per["GIRK2"].append(both)
        else:
            other = "A_C" if col == "B_D" else "B_D"
            per["GIRK1"].append(d[col])
            per["GIRK2"].append(d[other])
        names.append(run)
        print(f"  {run:>6}  frames {d[col if col != 'pool' else 'A_C'].size}")
    if not names:
        raise SystemExit(f"[{label}] no usable runs in {base}")
    return per, names


def boot_band(runs_list, bins, n_boot, rng):
    """2.5 / 97.5 percentiles of the pooled density when whole runs are resampled."""
    hists = np.array([np.histogram(r, bins=bins)[0] for r in runs_list], dtype=float)
    widths = np.diff(bins)
    out = np.empty((n_boot, len(widths)))
    for b in range(n_boot):
        pick = rng.integers(0, len(hists), len(hists))
        h = hists[pick].sum(axis=0)
        out[b] = h / (h.sum() * widths) if h.sum() else 0
    return np.percentile(out, [2.5, 97.5], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, help='LABEL=PATH[:B_D|A_C|pool], repeat per system')
    ap.add_argument("--runs", action="append", default=[], help='LABEL=SPEC, e.g. "E141Q=1-6" (default: all RUN*)')
    ap.add_argument("--threshold", type=float, default=5.69,
                    help="open threshold in A (default 5.69, Li et al. 2019 J Biol Chem)")
    ap.add_argument("--boot", type=int, default=500, help="run resamples for the 95%% band (0 = no band)")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    systems = [parse_system(s) for s in args.system]
    run_spec = dict(r.split("=", 1) for r in args.runs)
    data = []
    for label, base, col in systems:
        runs = expand_runs(run_spec[label]) if label in run_spec else None
        per, names = load_system(label, base, col, runs)
        data.append((label, col, per, names))

    out_dir = Path(args.out_dir) if args.out_dir else \
        systems[0][1].resolve().parent / ("hbc_diagonals_" + "_".join(l.replace(" ", "-") for l, _, _ in systems))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- tables ----------
    rows, summ = [], []
    for label, col, per, names in data:
        for diag, _ in PANELS:
            for run, v in zip(names, per[diag]):
                rows.append(dict(system=label, run=run, diagonal=diag, column=col, n_frames=v.size,
                                 median=round(float(np.median(v)), 3), mean=round(float(v.mean()), 3),
                                 frac_open=round(float((v > args.threshold).mean()), 4)))
            pool = np.concatenate(per[diag])
            meds = np.array([np.median(v) for v in per[diag]])
            opens = np.array([(v > args.threshold).mean() for v in per[diag]])
            summ.append(dict(system=label, diagonal=diag, column=col, n_runs=len(per[diag]), n_frames=pool.size,
                             pooled_median=round(float(np.median(pool)), 3), pooled_mean=round(float(pool.mean()), 3),
                             pooled_pct_open=round(100 * float((pool > args.threshold).mean()), 2),
                             run_median_mean=round(float(meds.mean()), 3),
                             run_median_sd=round(float(meds.std(ddof=1)) if meds.size > 1 else 0.0, 3),
                             run_pct_open_mean=round(100 * float(opens.mean()), 2),
                             run_pct_open_sd=round(100 * float(opens.std(ddof=1)) if opens.size > 1 else 0.0, 2)))
    for name, recs in (("hbc_diagonals_per_run.csv", rows), ("hbc_diagonals_summary.csv", summ)):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
        print(f"  Saved: {out_dir / name}")
    print("\nsystem / diagonal: pooled median (A), % open, runs")
    for s in summ:
        print(f"  {s['system']:>10} {s['diagonal']}: median {s['pooled_median']:5.2f}  open {s['pooled_pct_open']:5.1f}%"
              f"  (per run {s['run_pct_open_mean']:.1f} +- {s['run_pct_open_sd']:.1f}%, n={s['n_runs']})")

    # ---------- figures: plain (as compare_hbc_girk1_diagonal.py) and with the run band ----------
    allv = np.concatenate([np.concatenate(per[d]) for _, _, per, _ in data for d, _ in PANELS])
    lo, hi = min(1.5, float(allv.min())), max(12.0, float(allv.max()))
    bins = np.arange(np.floor(lo * 4) / 4, np.ceil(hi * 4) / 4 + 0.25 + 1e-9, 0.25)
    centers = 0.5 * (bins[1:] + bins[:-1])
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})
    for band in ([False, True] if args.boot else [False]):
        rng = np.random.default_rng(0)
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
        for ax, (diag, title) in zip(axes, PANELS):
            for k, (label, col, per, names) in enumerate(data):
                pool = np.concatenate(per[diag])
                dens, _ = np.histogram(pool, bins=bins, density=True)
                c, ls = COLORS[k % len(COLORS)], STYLES[k % len(STYLES)]
                lab = f"{label} (both diagonals, {len(names)} runs)" if col == "pool" else f"{label} ({len(names)} runs)"
                if band and len(per[diag]) > 1:
                    lo_b, hi_b = boot_band(per[diag], bins, args.boot, rng)
                    ax.fill_between(centers, lo_b, hi_b, color=c, alpha=0.18, linewidth=0, step="mid")
                ax.step(centers, dens, where="mid", color=c, ls=ls, lw=1.8, label=lab)
            ax.axvline(args.threshold, color="#6B6B6B", ls=":", lw=1)
            ax.text(args.threshold, 1.0, f" open > {args.threshold:g} \u00c5", transform=ax.get_xaxis_transform(),
                    fontsize=8, color="#6B6B6B", va="top")
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_xlabel("min. atom\u2013atom distance across the HBC (\u00c5)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.yaxis.grid(True, color="#E3E3E3", lw=0.5); ax.set_axisbelow(True)
        axes[0].set_ylabel("density (all frames of all runs)")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        if band:
            fig.text(0.99, 0.01, "shaded: 95% band from resampling whole runs", ha="right", fontsize=7.5, color="#6B6B6B")
        fig.tight_layout()
        stem = "hbc_diagonals_hist_band" if band else "hbc_diagonals_hist"
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
