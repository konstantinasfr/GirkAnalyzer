#!/usr/bin/env python3
"""
aggregate_ion_state_sequence.py
===============================
Pools `ion_state_sequence` output across runs (and optionally channels).

Every number is reported two ways:
  per_run   the value is computed in each run first, then averaged over runs
            (mean +/- SEM), so every run weighs the same
  pooled    every permeation event counted once (busy runs weigh more)

TABLES   (one row per channel x split x group x metric, etc.)
------
  pooled_events.csv          every event of every run, all channels
  splits.csv                 transit_ns, n_bounces, ever_upper (fraction),
                             longest_visit_ns, split by
                               entry GLU load      <=1 / >=2
                               entry free cavity   0 / 1 / >=2
                             plus 'all'
  last_before_out.csv        last_state_before_out and last_residue_before_out
                             percentages
  exit_vs_background.csv     distribution of exit_n_glu_occupied and
                             exit_n_free_cavity compared with the same
                             quantities at random frames of the same runs
                             (ion_state_background.csv). Rows with value
                             'mean' give the per-run mean at exit, at random
                             frames, and a paired Wilcoxon p over runs (if
                             scipy is installed).
                             Exits leave out the permeating ion; to match,
                             the background leaves out the lowest-z cavity
                             K+ of each frame (background_*). The same
                             counts with every K+ are reported alongside
                             (background_all_*) for reference.

PLOTS
-----
  exit_vs_background_<CH>.png     exit vs random-frame distributions
  transit_by_glu_load.png         per-run mean transit, entry GLU load <=1 / >=2,
                                  one group of bars per channel
  transit_by_free_cavity.png      the same, by entry free cavity ions

OUTPUT FOLDER
-------------
  one channel     <channel>/ion_state_sequence_across_runs/
  several         --out, or <parent of the first channel>/ion_state_sequence_comparison/

USAGE
-----
    python3 aggregate_ion_state_sequence.py .../girk_analyser_results/G12
    python3 aggregate_ion_state_sequence.py .../G12 .../G12_D173N .../G12_E141Q \\
        .../G12_GAT .../G12_ML .../G2 --out .../ion_state_sequence_comparison
    python3 aggregate_ion_state_sequence.py .../G12 --runs 1-20
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SCRIPT_VERSION = "2026-10-21b  (matched background + all-ion reference)"

MODULE_DIR = "ion_state_sequence"
OUT_SUBDIR = "ion_state_sequence_across_runs"
EVENTS_NAME = "ion_state_sequence.csv"
BACKGROUND_NAME = "ion_state_background.csv"

METRICS = ["transit_ns", "n_bounces", "ever_upper", "longest_visit_ns"]
SPLITS = {
    "all": [("all", lambda r: True)],
    "entry_glu_load": [("<=1", lambda r: r["entry_n_glu_occupied"] <= 1),
                       (">=2", lambda r: r["entry_n_glu_occupied"] >= 2)],
    "entry_free_cavity": [("0", lambda r: r["entry_n_free_cavity"] == 0),
                          ("1", lambda r: r["entry_n_free_cavity"] == 1),
                          (">=2", lambda r: r["entry_n_free_cavity"] >= 2)],
}
LOAD_QUANTITIES = {"n_glu_occupied": [0, 1, 2, 3, 4],
                   "n_free_cavity": [0, 1, 2, 3]}        # last value = ">= that"

CHANNEL_COLORS = ["#5AA9C9", "#E8746B", "#6FBF73", "#E8A85B", "#9B7BC8", "#8E9AAF"]
FIGSIZE = (8, 6.6)


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def expand_runs(spec):
    names = []
    for token in str(spec).replace(",", " ").split():
        t = token.strip()
        if not t:
            continue
        body = t[3:] if t.upper().startswith("RUN") else t
        if "-" in body and all(p.isdigit() for p in body.split("-", 1)):
            a, b = body.split("-", 1)
            names.extend(f"RUN{i}" for i in range(int(a), int(b) + 1))
        else:
            names.append(t if t.upper().startswith("RUN") else f"RUN{t}")
    return names


def natural(run):
    digits = "".join(c for c in run if c.isdigit())
    return (int(digits) if digits else 0, run)


def find_files(base, name, runs=None):
    base = Path(base)
    if not base.is_dir():
        sys.exit(f"ERROR: not a directory: {base}")
    names = runs if runs else sorted((d.name for d in base.glob("RUN*") if d.is_dir()), key=natural)
    return [(run, base / run / MODULE_DIR / name) for run in names
            if (base / run / MODULE_DIR / name).is_file()]


INT_COLS = ("ion_id", "end_1", "end_2", "n_visits", "n_bounces",
            "entry_n_glu_occupied", "entry_n_asn_occupied", "entry_n_free_cavity", "entry_n_cavity_total",
            "exit_n_glu_occupied", "exit_n_asn_occupied", "exit_n_free_cavity", "exit_n_cavity_total",
            "frame", "n_glu_occupied", "n_asn_occupied", "n_free_cavity", "n_cavity_total",
            "n_glu_occupied_all", "n_asn_occupied_all", "n_free_cavity_all", "n_cavity_total_all",
            "excluded_ion_id")
FLOAT_COLS = ("transit_ns", "entry_n_glu_occupied_mean", "exit_n_glu_occupied_mean",
              "time_in_U_ns", "time_in_L_ns", "time_in_F_ns", "longest_visit_ns")


def _typed(row):
    for k in INT_COLS:
        if k in row:
            row[k] = int(row[k]) if str(row[k]).strip() != "" else None
    for k in FLOAT_COLS:
        if k in row:
            row[k] = float(row[k]) if str(row[k]).strip() != "" else None
    if "ever_upper" in row:
        row["ever_upper"] = str(row["ever_upper"]).strip().lower() == "true"
    return row


def load(base, name, runs, channel):
    rows = []
    for run, path in find_files(base, name, runs):
        with open(path, newline="") as f:
            got = [_typed(r) for r in csv.DictReader(f)]
        for r in got:
            r["run_label"] = run
            r["channel_type"] = channel
        rows.extend(got)
        if got:
            print(f"    {run:<8} {len(got):>6} rows   {name}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def _mean_sem(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None
    return m, s


def _r(v, n=4):
    return "" if v is None else round(v, n)


def split_table(events, channel):
    out = []
    usable = [r for r in events if r["entry_n_glu_occupied"] is not None and r["entry_n_free_cavity"] is not None]
    for split, groups in SPLITS.items():
        for group, test in groups:
            sel = [r for r in usable if test(r)]
            by_run = {}
            for r in sel:
                by_run.setdefault(r["run_label"], []).append(r)
            for metric in METRICS:
                vals = [float(r[metric]) for r in sel if r[metric] is not None]
                pr = [float(np.mean([float(r[metric]) for r in rs if r[metric] is not None]))
                      for rs in by_run.values() if any(r[metric] is not None for r in rs)]
                m, s = _mean_sem(pr)
                out.append({"channel_type": channel, "split": split, "group": group, "metric": metric,
                            "n_runs": len(pr), "n_events": len(vals),
                            "per_run_mean": _r(m), "per_run_sem": _r(s),
                            "pooled_mean": _r(float(np.mean(vals)) if vals else None),
                            "pooled_median": _r(float(np.median(vals)) if vals else None)})
    return out


def last_table(events, channel):
    out = []
    for key in ("last_state_before_out", "last_residue_before_out"):
        labels = sorted({r[key] or "(none)" for r in events})
        by_run = {}
        for r in events:
            by_run.setdefault(r["run_label"], []).append(r[key] or "(none)")
        for label in labels:
            pr = [100.0 * v.count(label) / len(v) for v in by_run.values() if v]
            m, s = _mean_sem(pr)
            n = sum(1 for r in events if (r[key] or "(none)") == label)
            out.append({"channel_type": channel, "what": key, "label": label,
                        "n_runs": len(pr), "per_run_mean_pct": _r(m, 2), "per_run_sem_pct": _r(s, 2),
                        "pooled_count": n, "pooled_pct": _r(100.0 * n / len(events) if events else None, 2)})
    return out


def _bin(v, values):
    return min(v, values[-1])


def _dist(vals, values):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {k: None for k in values}
    return {k: 100.0 * sum(1 for v in vals if _bin(v, values) == k) / len(vals) for k in values}


def exit_vs_background(events, background, channel):
    """
    Exit distribution vs the matched background (lowest-z cavity ion left
    out, columns <q>) and the all-ion background (columns <q>_all).
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    out = []
    runs = sorted({r["run_label"] for r in events}, key=natural)

    def per_run(rows, key, values, run_key="run_label"):
        return {run: _dist([r.get(key) for r in rows if r[run_key] == run], values) for run in runs}

    def paired(key, bg_key):
        pairs = []
        for run in runs:
            e = [r[key] for r in events if r["run_label"] == run and r[key] is not None]
            b = [x.get(bg_key) for x in background if x["run_label"] == run and x.get(bg_key) is not None]
            if e and b:
                pairs.append((float(np.mean(e)), float(np.mean(b))))
        p = ""
        if wilcoxon and len(pairs) >= 5 and any(a != b for a, b in pairs):
            p = round(float(wilcoxon([a for a, _ in pairs], [b for _, b in pairs]).pvalue), 4)
        return pairs, p

    for q, values in LOAD_QUANTITIES.items():
        ex_key = f"exit_{q}"
        ex_pool = _dist([r[ex_key] for r in events], values)
        ex_run = per_run(events, ex_key, values)
        bg = {tag: (_dist([b.get(key) for b in background], values), per_run(background, key, values))
              for tag, key in (("background", q), ("background_all", f"{q}_all"))}
        for k in values:
            em, es = _mean_sem([d[k] for d in ex_run.values() if d[k] is not None])
            row = {"channel_type": channel, "quantity": q,
                   "value": f">={k}" if k == values[-1] else str(k),
                   "exit_pooled_pct": _r(ex_pool[k], 2),
                   "exit_per_run_mean_pct": _r(em, 2), "exit_per_run_sem_pct": _r(es, 2)}
            for tag, (pool, by_run) in bg.items():
                bm, bs = _mean_sem([d[k] for d in by_run.values() if d[k] is not None])
                row.update({f"{tag}_pooled_pct": _r(pool[k], 2),
                            f"{tag}_per_run_mean_pct": _r(bm, 2), f"{tag}_per_run_sem_pct": _r(bs, 2)})
            row.update({"n_runs": len(runs), "wilcoxon_p_runs": "", "wilcoxon_p_runs_all": ""})
            out.append(row)
        # per-run means at exit vs at random frames (matched and all-ion)
        pairs, p = paired(ex_key, q)
        pairs_all, p_all = paired(ex_key, f"{q}_all")
        em, es = _mean_sem([a for a, _ in pairs])
        bm, bs = _mean_sem([b for _, b in pairs])
        am, as_ = _mean_sem([b for _, b in pairs_all])
        out.append({"channel_type": channel, "quantity": q, "value": "mean",
                    "exit_pooled_pct": "", "exit_per_run_mean_pct": _r(em, 3), "exit_per_run_sem_pct": _r(es, 3),
                    "background_pooled_pct": "", "background_per_run_mean_pct": _r(bm, 3),
                    "background_per_run_sem_pct": _r(bs, 3),
                    "background_all_pooled_pct": "", "background_all_per_run_mean_pct": _r(am, 3),
                    "background_all_per_run_sem_pct": _r(as_, 3),
                    "n_runs": len(pairs), "wilcoxon_p_runs": p, "wilcoxon_p_runs_all": p_all})
    return out


def write_csv(rows, path):
    if not rows:
        print(f"  [warn] no rows for {path.name}")
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_exit_vs_background(rows, channel, out_path):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    for ax, q in zip(axes, LOAD_QUANTITIES):
        sel = [r for r in rows if r["quantity"] == q and r["value"] != "mean"]
        x = np.arange(len(sel))
        for off, key, color, label in ((-0.27, "exit_per_run_mean_pct", "#C62828", "at cavity exit"),
                                       (0.0, "background_per_run_mean_pct", "#90A4AE",
                                        "random frames\n(lowest-z cavity ion left out)"),
                                       (0.27, "background_all_per_run_mean_pct", "white",
                                        "random frames, all ions\n(reference)")):
            vals = [r.get(key) if r.get(key) not in ("", None) else 0 for r in sel]
            errs = [r.get(key.replace("mean", "sem")) or 0 for r in sel]
            ax.bar(x + off, vals, width=0.27, yerr=errs, capsize=2, color=color, edgecolor="black",
                   linewidth=0.8, label=label)
        mean = next((r for r in rows if r["quantity"] == q and r["value"] == "mean"), None)
        p = (f"\nmean exit {mean['exit_per_run_mean_pct']} vs random {mean['background_per_run_mean_pct']}"
             + (f"\np = {mean['wilcoxon_p_runs']} (paired, runs)" if mean['wilcoxon_p_runs'] != '' else "")) if mean else ""
        ax.set_title(q.replace("n_", "").replace("_", " ") + p, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([r["value"] for r in sel])
        ax.set_xlabel("number of K⁺ / residues", fontsize=11)
        ax.set_ylabel("% of frames (per-run mean ± SEM)", fontsize=11)
        _style(ax)
    fig.suptitle(f"{channel}: cavity load at exit vs random frames", fontsize=14, fontweight="bold")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_channel_split(per_channel_events, split, metric, ylabel, title, out_path):
    groups = [g for g, _ in SPLITS[split]]
    tests = dict(SPLITS[split])
    channels = list(per_channel_events)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    width = 0.8 / max(len(channels), 1)
    rng = np.random.default_rng(0)
    for ci, ch in enumerate(channels):
        events = [r for r in per_channel_events[ch]
                  if r["entry_n_glu_occupied"] is not None and r["entry_n_free_cavity"] is not None]
        color = CHANNEL_COLORS[ci % len(CHANNEL_COLORS)]
        for gi, g in enumerate(groups):
            by_run = {}
            for r in events:
                if tests[g](r) and r[metric] is not None:
                    by_run.setdefault(r["run_label"], []).append(float(r[metric]))
            pr = [float(np.mean(v)) for v in by_run.values()]
            m, s = _mean_sem(pr)
            x = gi - 0.4 + width * (ci + 0.5)
            if m is not None:
                ax.bar(x, m, width=width * 0.9, yerr=s or 0, capsize=3, color=color,
                       edgecolor="black", linewidth=0.8)
                ax.scatter(x + rng.uniform(-width * 0.25, width * 0.25, len(pr)), pr, s=14,
                           color="black", alpha=0.6, zorder=3, linewidths=0)
                ax.text(x, 0, f"{len(pr)}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_xlabel(split.replace("_", " "), fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    _style(ax)
    handles = [Patch(facecolor=CHANNEL_COLORS[i % len(CHANNEL_COLORS)], edgecolor="black", label=ch)
               for i, ch in enumerate(channels)]
    handles.append(Line2D([], [], marker="o", color="black", linestyle="", alpha=0.6, label="run"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Aggregate ion_state_sequence over runs and channels")
    ap.add_argument("base_dirs", nargs="+", help="Channel folder(s) holding RUN*/")
    ap.add_argument("-runs", "--runs", default="", help="e.g. 1-20 (applied to every channel)")
    ap.add_argument("-out", "--out", default=None)
    args = ap.parse_args()

    bases = [Path(b) for b in args.base_dirs]
    runs = expand_runs(args.runs) if args.runs else None
    if args.out:
        out_dir = Path(args.out)
    elif len(bases) == 1:
        out_dir = bases[0] / OUT_SUBDIR
    else:
        out_dir = bases[0].parent / "ion_state_sequence_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"aggregate_ion_state_sequence.py  version {SCRIPT_VERSION}")
    all_events, splits, lasts, exits, per_channel = [], [], [], [], {}
    for base in bases:
        ch = base.name
        print(f"\n{ch}:")
        events = load(base, EVENTS_NAME, runs, ch)
        background = load(base, BACKGROUND_NAME, runs, ch)
        if not events:
            print(f"  no {EVENTS_NAME} found under {base}/RUN*/{MODULE_DIR}/")
            continue
        per_channel[ch] = events
        all_events += events
        splits += split_table(events, ch)
        lasts += last_table(events, ch)
        ev_bg = exit_vs_background(events, background, ch)
        exits += ev_bg
        plot_exit_vs_background(ev_bg, ch, out_dir / f"exit_vs_background_{ch}.png")

    if not all_events:
        sys.exit("No ion_state_sequence results found.")
    print()
    write_csv(all_events, out_dir / "pooled_events.csv")
    write_csv(splits, out_dir / "splits.csv")
    write_csv(lasts, out_dir / "last_before_out.csv")
    write_csv(exits, out_dir / "exit_vs_background.csv")
    plot_channel_split(per_channel, "entry_glu_load", "transit_ns", "Transit time (ns), per-run mean ± SEM",
                       "Transit time by GLU load at SF exit", out_dir / "transit_by_glu_load.png")
    plot_channel_split(per_channel, "entry_free_cavity", "transit_ns", "Transit time (ns), per-run mean ± SEM",
                       "Transit time by free cavity ions at SF exit", out_dir / "transit_by_free_cavity.png")
    print(f"\n✓ Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
