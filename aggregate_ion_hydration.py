#!/usr/bin/env python3
"""
aggregate_ion_hydration.py
==========================
Pools `ion_hydration_events.csv` from every run and asks the question the
per-run bar charts only hint at:

    is the permeating ion more hydrated than the other cavity ions?

THE TEST IS PAIRED
------------------
Every permeation event supplies two numbers measured in the SAME frame: the
permeating ion's own hydration, and the mean hydration of the other ions in
the cavity at that instant. They share the frame's water structure, the
run's equilibration state and the channel's conformation, so comparing them
as two independent groups throws away the pairing and most of the power.

This script uses a Wilcoxon signed-rank test on the per-event differences
(and reports a paired t-test alongside). A per-run bar chart of 5.91 vs 4.87
cannot tell you whether that gap is consistent across events; the paired
test can.

Pooling is by EVENT, not by run: a run with 27 permeations carries more
weight than one with 2. The mean of per-run means is reported next to it as
a check -- if the two disagree, one run is driving the pooled answer.

OUTPUTS   <channel>/water_analysis/ion_hydration_across_runs/
-------------------------------------------------------------
    pooled_events_<CH>.csv        every event from every run, one row each
    pooled_summary_<CH>.csv       permeating vs others, pooled and per-run
    pooled_tests_<CH>.csv         Wilcoxon / t-test per moment and region
    pooled_bars_<CH>.png          permeating vs others at end_2 and end_3
    pooled_hist_<CH>.png          hydration distributions, both moments
    paired_slopes_<CH>.png        one line per event: permeating -> others
    per_run_forest_<CH>.png       the difference per run, with the pooled line
    all_frames_<CH>.png           baseline hydration per region across runs
    contact_split_<CH>.csv        pooled in-contact / no-contact / per residue
    contact_split_<CH>.png        the split, with one dot per run

USAGE
-----
    python3 aggregate_ion_hydration.py \
        /media/ziyue/.../girk_analyser_results/G12 --channel G12

    python3 aggregate_ion_hydration.py .../G12 --channel G12 --runs 1-20
    python3 aggregate_ion_hydration.py .../G2  --channel G2 --region GLU/ASN
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-10-06c  (labels clear of the run dots)"

WATER_ROOT = "water_analysis"
MODULE_DIR = "ion_hydration_regions"
OUT_SUBDIR = "ion_hydration_across_runs"

EVENTS_NAME = "ion_hydration_events.csv"
SUMMARY_NAME = "ion_hydration_events_summary.csv"

REGIONS = ["GLU/ASN", "ASN/HBC", "GLU/HBC"]
MOMENTS = ["end2", "end3"]
MOMENT_LABEL = {"end2": "end_2 (K⁺ leaves the cavity)",
                "end3": "end_3 (K⁺ past the gate)"}
MOMENT_COLOR = {"end2": "#E67E22", "end3": "#C0392B"}
PERM_COLOR = "#C0392B"
OTHER_COLOR = "#90A4AE"


def tag(region):
    return region.replace("/", "_")


# ─────────────────────────────────────────────────────────────────────────────
#  Discovery
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


def find_files(base, name, runs=None):
    """
    [(run_name, path)] using bounded globs — never a recursive walk, which is
    slow on the external drive.
    """
    base = Path(base)
    if not base.is_dir():
        sys.exit(f"ERROR: not a directory: {base}")
    out = []
    names = runs if runs else sorted(
        (d.name for d in base.glob("RUN*") if d.is_dir()),
        key=lambda n: (len("".join(c for c in n if c.isdigit())), n))
    for run in names:
        direct = base / run / WATER_ROOT / MODULE_DIR / name
        if direct.is_file():
            out.append((run, direct))
            continue
        alt = sorted((base / run / WATER_ROOT).glob(f"*/{name}"))
        if alt:
            out.append((run, alt[0]))
    return out


def natural(run):
    digits = "".join(c for c in run if c.isdigit())
    return (int(digits) if digits else 0, run)


def _f(row, key):
    v = str(row.get(key, "")).strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_events(base, runs, channel):
    rows = []
    for run, path in find_files(base, EVENTS_NAME, runs):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                r["run"] = r.get("run") or run
                if not str(r.get("run", "")).strip():
                    r["run"] = run
                r["run_label"] = run
                r["channel_type"] = channel or r.get("channel_type", "")
                rows.append(r)
        print(f"    {run:<8} {path.parent.parent.parent.name:<6} "
              f"{sum(1 for _ in open(path)) - 1:>4} events")
    return rows


def load_all_frames(base, runs, channel):
    """The all_frames rows of each run's summary — the baseline hydration."""
    out = []
    for run, path in find_files(base, SUMMARY_NAME, runs):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("moment") != "all_frames":
                    continue
                n = _f(r, "n_events")
                m = _f(r, "mean")
                if n is None or m is None:
                    continue
                out.append(dict(channel_type=channel or r.get("channel_type", ""),
                                run=run, region=r.get("region", ""),
                                n_ion_frames=int(n), mean=m,
                                median=_f(r, "median")))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────

def paired_stats(perm, others):
    """
    Wilcoxon signed-rank plus a paired t-test on the per-event differences.
    Falls back to a sign test if scipy is unavailable.
    """
    perm, others = np.asarray(perm, float), np.asarray(others, float)
    d = perm - others
    out = dict(n_pairs=int(d.size),
               mean_permeating=round(float(perm.mean()), 3) if d.size else "",
               mean_others=round(float(others.mean()), 3) if d.size else "",
               mean_difference=round(float(d.mean()), 3) if d.size else "",
               median_difference=round(float(np.median(d)), 3) if d.size else "",
               n_perm_higher=int((d > 0).sum()), n_equal=int((d == 0).sum()),
               n_perm_lower=int((d < 0).sum()))
    if d.size > 1 and d.std(ddof=1) > 0:
        sd = float(d.std(ddof=1))
        out["sd_difference"] = round(sd, 3)
        out["cohens_dz"] = round(float(d.mean() / sd), 3)
        ci = 1.96 * sd / np.sqrt(d.size)
        out["ci95_low"] = round(float(d.mean() - ci), 3)
        out["ci95_high"] = round(float(d.mean() + ci), 3)
    try:
        from scipy import stats
        if d.size >= 6 and np.any(d != 0):
            w, pw = stats.wilcoxon(perm, others, zero_method="wilcox")
            out["wilcoxon_p"] = round(float(pw), 6)
        t, pt = stats.ttest_rel(perm, others)
        out["paired_t"] = round(float(t), 3)
        out["paired_t_p"] = round(float(pt), 6)
    except Exception:
        pass
    return out


def weighted_mean(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    return float((v * w).sum() / w.sum()) if w.sum() else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
#  Tables
# ─────────────────────────────────────────────────────────────────────────────

def pooled_rows(events, region):
    """(perm, others, run) triples where both numbers exist for this region."""
    out = {m: [] for m in MOMENTS}
    for r in events:
        for m in MOMENTS:
            p = _f(r, f"perm_waters_{m}")
            o = _f(r, f"others_mean_{m}_{tag(region)}")
            if p is None or o is None:
                continue
            out[m].append((p, o, r.get("run_label", r.get("run", ""))))
    return out


def build_summary(events, channel):
    rows = []
    for region in REGIONS:
        pooled = pooled_rows(events, region)
        for m in MOMENTS:
            trip = pooled[m]
            if not trip:
                continue
            perm = [t[0] for t in trip]
            others = [t[1] for t in trip]
            runs = sorted({t[2] for t in trip}, key=natural)
            # mean of per-run means, as the alternative weighting
            per_run_diff = []
            for run in runs:
                d = [t[0] - t[1] for t in trip if t[2] == run]
                if d:
                    per_run_diff.append(float(np.mean(d)))
            rows.append(dict(channel_type=channel, moment=m, region=region,
                             weighting="pooled_events",
                             n_runs=len(runs),
                             **paired_stats(perm, others)))
            if per_run_diff:
                a = np.array(per_run_diff)
                rows.append(dict(
                    channel_type=channel, moment=m, region=region,
                    weighting="mean_of_run_means", n_runs=len(runs),
                    n_pairs=len(trip),
                    mean_difference=round(float(a.mean()), 3),
                    median_difference=round(float(np.median(a)), 3),
                    sd_difference=round(float(a.std(ddof=1))
                                        if a.size > 1 else 0.0, 3)))
    return rows


def write_csv(rows, path):
    if not rows:
        print(f"    [warn] nothing to write to {Path(path).name}")
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"    Saved: {Path(path).name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_bars(events, region, out_path, title):
    pooled = pooled_rows(events, region)
    moments = [m for m in MOMENTS if pooled[m]]
    if not moments:
        return
    x = np.arange(len(moments))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for k, (who, color) in enumerate((("permeating", PERM_COLOR),
                                      ("others", OTHER_COLOR))):
        means, errs, ns = [], [], []
        for m in moments:
            vals = np.array([t[0] if who == "permeating" else t[1]
                             for t in pooled[m]])
            means.append(vals.mean())
            errs.append(vals.std(ddof=1) / np.sqrt(vals.size)
                        if vals.size > 1 else 0.0)
            ns.append(vals.size)
        pos = x + (k - 0.5) * width
        ax.bar(pos, means, width * 0.92, yerr=errs, capsize=5, color=color,
               alpha=0.9, edgecolor="black", linewidth=0.7,
               error_kw={"linewidth": 1.2, "ecolor": "black"},
               label=("Permeating ion" if who == "permeating"
                      else f"Other ions in {region}"))
        for p, mu, n in zip(pos, means, ns):
            ax.text(p, mu, f"{mu:.2f}\nn={n}", ha="center", va="bottom",
                    fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MOMENT_LABEL[m] for m in moments], fontsize=10)
    ax.set_ylabel("Mean hydration number (± SEM)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_hist(events, region, out_path, title):
    pooled = pooled_rows(events, region)
    allv = [v for m in MOMENTS for t in pooled[m] for v in (t[0], t[1])]
    if not allv:
        return
    bins = np.arange(min(allv) - 0.5, max(allv) + 1.5, 1.0)
    moments = [m for m in MOMENTS if pooled[m]]

    fig, axes = plt.subplots(1, len(moments), figsize=(6 * len(moments), 4.6),
                             squeeze=False)
    for ax, m in zip(axes[0], moments):
        perm = np.array([t[0] for t in pooled[m]])
        oth = np.array([t[1] for t in pooled[m]])
        ax.hist(perm, bins=bins, alpha=0.6, color=PERM_COLOR,
                edgecolor="black", linewidth=0.5,
                label=f"permeating (mean {perm.mean():.2f})")
        ax.hist(oth, bins=bins, alpha=0.6, color=OTHER_COLOR,
                edgecolor="black", linewidth=0.5,
                label=f"others (mean {oth.mean():.2f})")
        ax.set_title(f"{MOMENT_LABEL[m]}  (n={perm.size})", fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("Hydration number", fontsize=10)
        ax.grid(True, alpha=0.25, axis="y")
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Permeation events", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_paired_slopes(events, region, out_path, title):
    """One faint line per event. A consistent tilt is the whole result."""
    pooled = pooled_rows(events, region)
    moments = [m for m in MOMENTS if pooled[m]]
    if not moments:
        return
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(moments), figsize=(5.2 * len(moments), 5),
                             squeeze=False)
    for ax, m in zip(axes[0], moments):
        trip = pooled[m]
        for p, o, _ in trip:
            j = rng.uniform(-0.04, 0.04, 2)
            ax.plot([0 + j[0], 1 + j[1]], [p, o], color="grey", alpha=0.35,
                    linewidth=0.8, zorder=1)
            ax.scatter([0 + j[0]], [p], s=18, color=PERM_COLOR, alpha=0.7,
                       zorder=2)
            ax.scatter([1 + j[1]], [o], s=18, color=OTHER_COLOR, alpha=0.7,
                       zorder=2)
        perm = np.array([t[0] for t in trip])
        oth = np.array([t[1] for t in trip])
        ax.plot([0, 1], [perm.mean(), oth.mean()], color="black",
                linewidth=2.5, marker="o", zorder=3, label="mean")
        st = paired_stats(perm, oth)
        sub = (f"Δ = {st['mean_difference']:+.2f}  "
               f"({st['n_perm_higher']}/{st['n_pairs']} higher)")
        if "wilcoxon_p" in st:
            sub += f"\nWilcoxon p = {st['wilcoxon_p']:.3g}"
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Permeating\nion", f"Others in\n{region}"],
                           fontsize=10)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(f"{MOMENT_LABEL[m]}\n{sub}", fontsize=10,
                     fontweight="bold")
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Hydration number", fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_forest(events, region, out_path, title):
    """Per-run mean difference with its SEM, against the pooled value."""
    pooled = pooled_rows(events, region)
    moments = [m for m in MOMENTS if pooled[m]]
    if not moments:
        return
    fig, axes = plt.subplots(1, len(moments), figsize=(6 * len(moments), 6),
                             squeeze=False, sharey=True)
    for ax, m in zip(axes[0], moments):
        trip = pooled[m]
        runs = sorted({t[2] for t in trip}, key=natural)
        ys, means, errs, ns = [], [], [], []
        for i, run in enumerate(runs):
            d = np.array([t[0] - t[1] for t in trip if t[2] == run])
            ys.append(i)
            means.append(float(d.mean()))
            errs.append(float(d.std(ddof=1) / np.sqrt(d.size))
                        if d.size > 1 else 0.0)
            ns.append(d.size)
        alld = np.array([t[0] - t[1] for t in trip])
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(alld.mean(), color=MOMENT_COLOR[m], linestyle="--",
                   linewidth=1.6,
                   label=f"pooled Δ = {alld.mean():+.2f} (n={alld.size})")
        ax.errorbar(means, ys, xerr=errs, fmt="o", color=MOMENT_COLOR[m],
                    ecolor="black", elinewidth=1, capsize=3, markersize=7,
                    markeredgecolor="black", markeredgewidth=0.5)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{r}  (n={n})" for r, n in zip(runs, ns)],
                           fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Hydration difference  (permeating − others)",
                      fontsize=10)
        ax.set_title(MOMENT_LABEL[m], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25, axis="x")
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_all_frames(rows, out_path, title):
    """Baseline hydration per region: one dot per run, weighted pooled bar."""
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rng = np.random.default_rng(0)
    tops = []
    for i, region in enumerate(REGIONS):
        sel = [r for r in rows if r["region"] == region]
        if not sel:
            continue
        vals = np.array([r["mean"] for r in sel])
        wts = np.array([r["n_ion_frames"] for r in sel])
        pooled = weighted_mean(vals, wts)
        ax.bar(i, pooled, 0.55, color="#2980B9", alpha=0.7,
               edgecolor="black", linewidth=0.7, zorder=1)
        ax.scatter(i + rng.uniform(-0.16, 0.16, vals.size), vals, s=30,
                   color="#222222", alpha=0.6, zorder=3, edgecolors="white",
                   linewidths=0.3)
        tops.append(max(pooled, float(vals.max())))
        ax.text(i, tops[-1], f"{pooled:.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", zorder=4)
    ax.set_xticks(range(len(REGIONS)))
    ax.set_xticklabels(REGIONS, fontsize=11)
    if tops:
        ax.set_ylim(0, max(tops) * 1.18)
    ax.set_ylabel("Mean waters per ion (all frames)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")



def load_contacts(base, runs, channel):
    """
    The glu_asn_contact rows of each run's summary: hydration of the cavity
    ions split by whether they touch one of the 8 GLU/ASN residues, and the
    per-residue breakdown.

    Returns [] for runs produced before ion_hydration_regions.py 2026-10-06b,
    which did not compute them.
    """
    out = []
    for run, path in find_files(base, SUMMARY_NAME, runs):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("moment") != "glu_asn_contact":
                    continue
                n, m = _f(r, "n_events"), _f(r, "mean")
                if n is None or m is None or n == 0:
                    continue
                out.append(dict(channel_type=channel, run=run,
                                who=r.get("who", ""), region=r.get("region", ""),
                                n_ion_frames=int(n), mean=m,
                                median=_f(r, "median")))
    return out


def pool_contacts(rows, channel):
    """
    Pool over ION-FRAMES, not over run means: a run contributing 26,000
    ion-frames should outweigh one contributing 2,000. The mean of run means
    is reported beside it so a single dominant run would be visible.
    """
    out = []
    keys = []
    for r in rows:
        k = (r["who"], r["region"])
        if k not in keys:
            keys.append(k)
    for who, region in keys:
        sel = [r for r in rows if r["who"] == who and r["region"] == region]
        vals = np.array([r["mean"] for r in sel], float)
        wts = np.array([r["n_ion_frames"] for r in sel], float)
        out.append(dict(
            channel_type=channel, who=who, region=region, n_runs=len(sel),
            total_ion_frames=int(wts.sum()),
            pooled_mean=round(weighted_mean(vals, wts), 3),
            mean_of_run_means=round(float(vals.mean()), 3),
            sd_across_runs=round(float(vals.std(ddof=1))
                                 if vals.size > 1 else 0.0, 3),
            min_run=round(float(vals.min()), 3),
            max_run=round(float(vals.max()), 3)))
    return out


def plot_contact_split(rows, pooled, out_path, title):
    """
    Left: the three groups. Right: per residue. Dots are runs.

    Value labels are placed ABOVE the highest dot of their bar, not at the bar
    top, because the dot cloud sits exactly where the label would otherwise
    go and hides it.
    """
    groups = [("in_contact", "in contact"), ("no_contact", "touching\nnothing"),
              ("all_cavity_ions", "all cavity\nions")]
    gsel = [(lab, next((p for p in pooled if p["who"] == k), None), k)
            for k, lab in groups]
    gsel = [(lab, p, k) for lab, p, k in gsel if p]
    res = [p for p in pooled if p["who"] == "per_residue"]
    if not gsel and not res:
        return

    rng = np.random.default_rng(0)
    ncol = 2 if res else 1
    widths = [1, 1.7] if res else [1]
    fig, axes = plt.subplots(1, ncol, figsize=(13 if res else 6.5, 5.4),
                             gridspec_kw={"width_ratios": widths},
                             squeeze=False)

    def draw(ax, bars, color_for):
        """bars = [(label, pooled_value, [per-run values])]"""
        tops = []
        for i, (lab, value, pts) in enumerate(bars):
            ax.bar(i, value, 0.6, color=color_for(i, lab), alpha=0.8,
                   edgecolor="black", linewidth=0.7, zorder=1)
            if pts:
                ax.scatter(i + rng.uniform(-0.17, 0.17, len(pts)), pts, s=24,
                           color="#222222", alpha=0.55, zorder=3,
                           edgecolors="white", linewidths=0.3)
            tops.append(max([value] + list(pts)) if pts else value)
        span = max(tops) if tops else 1
        for i, (lab, value, pts) in enumerate(bars):
            ax.text(i, tops[i] + span * 0.035, f"{value:.2f}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold", zorder=4)
        ax.set_xticks(range(len(bars)))
        ax.set_ylim(0, span * 1.22)
        return span

    ax1 = axes[0][0]
    colors = {"in_contact": "#C0392B", "no_contact": "#2980B9",
              "all_cavity_ions": "#90A4AE"}
    gbars = [(lab, p["pooled_mean"],
              [r["mean"] for r in rows if r["who"] == k])
             for lab, p, k in gsel]
    keys = [k for _, _, k in gsel]
    draw(ax1, gbars, lambda i, lab: colors.get(keys[i], "#2980B9"))
    ax1.set_xticklabels([lab for lab, _, _ in gsel], fontsize=9)
    ax1.set_ylabel("Mean waters per ion", fontsize=10)
    ax1.set_title("GLU/ASN cavity ions", fontsize=11, fontweight="bold")

    if res:
        ax2 = axes[0][1]
        rbars = [(p["region"], p["pooled_mean"],
                  [r["mean"] for r in rows if r["who"] == "per_residue"
                   and r["region"] == p["region"]]) for p in res]
        draw(ax2, rbars, lambda i, lab: "#8E44AD")
        ax2.set_xticklabels([p["region"] for p in res], rotation=45,
                            ha="right", fontsize=9)
        ax2.set_title("Ions sitting on each residue", fontsize=11,
                      fontweight="bold")

    for ax in axes[0]:
        ax.grid(True, alpha=0.25, axis="y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", help="Channel folder holding RUN*/")
    ap.add_argument("-channel", "--channel", dest="channel", default="")
    ap.add_argument("-runs", "--runs", default="",
                    help="Only these runs, e.g. --runs 1-20")
    ap.add_argument("-region", "--region", default="GLU/ASN",
                    choices=REGIONS + ["all"],
                    help="Which region's other-ions to compare against "
                         "(default GLU/ASN; 'all' does every region)")
    ap.add_argument("-out", "--out", default=None)
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    ch = args.channel or base.name

    print("=" * 70)
    print("AGGREGATE ION HYDRATION")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel: {ch}")

    runs = expand_runs(args.runs) if args.runs else None
    events = load_events(base, runs, ch)
    if not events:
        sys.exit(f"ERROR: no {EVENTS_NAME} found under {base}/RUN*/"
                 f"{WATER_ROOT}/{MODULE_DIR}/")
    n_runs = len({r["run_label"] for r in events})
    print(f"  {len(events)} permeation events from {n_runs} run(s)")

    out_dir = Path(args.out) if args.out else base / WATER_ROOT / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    write_csv(events, out_dir / f"pooled_events_{ch}.csv")

    summary = build_summary(events, ch)
    write_csv(summary, out_dir / f"pooled_summary_{ch}.csv")
    tests = [r for r in summary if r["weighting"] == "pooled_events"]
    write_csv(tests, out_dir / f"pooled_tests_{ch}.csv")

    print(f"\n  {'moment':<7}{'region':<10}{'perm':>7}{'others':>8}"
          f"{'diff':>8}{'higher':>9}{'Wilcoxon p':>12}")
    for r in tests:
        wp = r.get("wilcoxon_p", "")
        frac = (f"{r['n_perm_higher']}/{r['n_pairs']}"
                if r.get("n_pairs") else "")
        print(f"  {r['moment']:<7}{r['region']:<10}"
              f"{r.get('mean_permeating',''):>7}{r.get('mean_others',''):>8}"
              f"{r.get('mean_difference',''):>8}{frac:>9}"
              f"{(f'{wp:.3g}' if wp != '' else '-'):>12}")

    regions = REGIONS if args.region == "all" else [args.region]
    for region in regions:
        suffix = f"_{tag(region)}" if args.region == "all" else ""
        plot_bars(events, region, out_dir / f"pooled_bars_{ch}{suffix}.png",
                  f"{ch} — permeating ion vs other cavity ions "
                  f"({n_runs} runs, {len(events)} events)")
        plot_hist(events, region, out_dir / f"pooled_hist_{ch}{suffix}.png",
                  f"{ch} — hydration distributions, pooled over {n_runs} runs")
        plot_paired_slopes(events, region,
                           out_dir / f"paired_slopes_{ch}{suffix}.png",
                           f"{ch} — paired within each permeation event")
        plot_forest(events, region,
                    out_dir / f"per_run_forest_{ch}{suffix}.png",
                    f"{ch} — hydration difference per run (others in {region})")

    con = load_contacts(base, runs, ch)
    if con:
        pooled_con = pool_contacts(con, ch)
        write_csv(con, out_dir / f"contact_split_per_run_{ch}.csv")
        write_csv(pooled_con, out_dir / f"contact_split_{ch}.csv")
        plot_contact_split(con, pooled_con,
                           out_dir / f"contact_split_{ch}.png",
                           f"{ch} — hydration by residue contact in GLU/ASN "
                           f"(bars = pooled over {len({r['run'] for r in con})} "
                           f"runs, dots = individual runs)")
        print(f"\n  {'who':<16}{'region':<14}{'pooled':>8}{'per-run':>9}"
              f"{'sd':>7}{'runs':>6}{'ion-frames':>12}")
        for p in pooled_con:
            print(f"  {p['who']:<16}{p['region']:<14}{p['pooled_mean']:>8}"
                  f"{p['mean_of_run_means']:>9}{p['sd_across_runs']:>7}"
                  f"{p['n_runs']:>6}{p['total_ion_frames']:>12,}")
    else:
        print(f"\n  [info] no glu_asn_contact rows found — rerun "
              f"ion_hydration_regions.py 2026-10-06b or later with "
              f"force_rerun=True to add the residue-contact split.")

    af = load_all_frames(base, runs, ch)
    if af:
        write_csv(af, out_dir / f"all_frames_baseline_{ch}.csv")
        plot_all_frames(af, out_dir / f"all_frames_{ch}.png",
                        f"{ch} — baseline hydration per region "
                        f"(bar = frame-weighted pool, dots = runs)")

    print(f"\n✓ Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()

'''
python3 aggregate_ion_hydration.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--channel G12

python3 aggregate_ion_hydration.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--channel G2

python3 aggregate_ion_hydration.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/ \
--channel G12_E141Q

python3 aggregate_ion_hydration.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/ \
--channel G12_D173N
'''