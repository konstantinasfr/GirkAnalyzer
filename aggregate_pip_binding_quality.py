#!/usr/bin/env python3
"""
aggregate_pip_binding_quality.py
================================
Cross-run summary of the pip_binding_quality pipeline, at FOUR levels:

  PER-RUN     one row per run: was binding good in RUN1, RUN2, ...
  PER-SITE    2a / 2b / 2c / G1 pooled across runs: is one interface weaker?
  OVERALL     one verdict per channel type
  PER-RESIDUE which contacts reproduce across every run, and which don't

Several channel folders can be given at once. Each gets its own outputs, and
with two or more the channels are compared directly:

    python3 aggregate_pip_binding_quality.py .../G12 .../G2 -stats

INPUT LAYOUT
------------
    <base_dir>/                   e.g. .../girk_analyser_results/G12
        RUN1/pip_binding_quality/pip_timeseries.csv     (required)
        RUN1/pip_binding_quality/pip_occupancy.csv      (optional)
        RUN2/...

POOLING RULES (deliberate — don't change casually)
--------------------------------------------------
Frame-level quantities (salt bridges, contacts, drift, frac_bound) are pooled
over FRAMES, so a 6800-frame run counts more than a 2000-frame one.

UNBINDING EVENTS are counted WITHIN each run and summed, never on a
concatenated timeseries: joining the last frame of one run to the first frame
of the next invents transitions that never happened. Rates are total events
over total simulated time.

Per-run spread is reported as the SD across runs of each run's own mean,
next to the pooled value. Pooled value = the estimate; SD = whether the runs
agree. A strong pooled number with a large SD means one run is carrying it.

frac_bound saturates at 1.0 in well-behaved systems and then ranks nothing.
-strict_sb raises the bar (default 3 salt bridges) for a discriminating
version; both are reported.

TESTING (with -stats)
---------------------
Channel comparisons are run-clustered Mann-Whitney over per-run means, n =
number of runs. The trajectory is the unit of replication here — frames within
a run are not independent, so a frame-level test would be meaningless, and
unlike the per-ion pipeline there is no ion-level alternative worth printing.
Per-site tests carry Benjamini-Hochberg FDR across the sites.

With 3-5 runs per channel a Mann-Whitney cannot reach p < 0.05 no matter how
large the difference (minimum attainable p at n=3 vs 3 is 0.1). Treat the
effect size as the result and the p-value as a formality at these n.

OUTPUTS (in <base_dir>/pip_binding_quality_aggregate/)
------------------------------------------------------
    per_run_summary.csv          one row per run, with a verdict label
    per_site_per_run.csv         the raw run x site grid everything builds on
    per_site_summary.csv         sites pooled across runs
    overall_summary.csv          one row for the channel
    per_residue_occupancy.csv    pooled occupancy + reproducibility flag
    per_run_binding.png          salt bridges and drift, per run per site
    per_site_binding.png         per site, each dot one run
    per_residue_reproducibility.png

  cross-channel (written to the FIRST channel's folder, or to -out)
    channel_comparison.csv
    channel_binding_box.png
    channel_tests.csv            with -stats

USAGE
-----
    python3 aggregate_pip_binding_quality.py .../G12
    python3 aggregate_pip_binding_quality.py .../G12 .../G2 -stats
    python3 aggregate_pip_binding_quality.py .../G12 -runs 1 2 3
    python3 aggregate_pip_binding_quality.py .../G12 -strict_sb 4 -fontscale 1.2
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-10-14a  (per-run + per-site + per-residue)"

RUN_PATTERN = re.compile(r"^RUN(\d+)$", re.IGNORECASE)

CHANNEL_COLOR = {"G12": "#2980B9", "G2": "#E67E22",
                 "G12_ML": "#27AE60", "G12_GAT": "#8E44AD",
                 "G12_D173N": "#C0392B", "G12_E141Q": "#16A085"}

SITE_COLOR = {"2a": "#2980B9", "2b": "#E67E22",
              "2c": "#27AE60", "2d": "#8E44AD", "G1": "#C0392B"}

# ns per frame, by channel type; override with -frame_duration
FRAME_DURATION = {"G12": 0.075, "G12_ML": 0.075, "G12_GAT": 0.075,
                  "G12_D173N": 0.075, "G12_E141Q": 0.075, "G2": 0.4}

# ─────────────────────────────────────────────────────────────────────────────
#  FONT SIZES — all multiplied by -fontscale at run time.
# ─────────────────────────────────────────────────────────────────────────────
FS_TITLE   = 23
FS_AXLABEL = 21
FS_TICK    = 19
FS_VALUE   = 16
FS_LEGEND  = 18

DPI = 200
BAR_EDGE_LW = 1.0

FIG_PER_RUN    = (16.0, 9.0)
FIG_PER_SITE   = (10.0, 7.0)
FIG_RESIDUE    = (13.0, 9.0)
FIG_CHANNEL    = (9.5, 7.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _finish(ax, fs):
    ax.tick_params(axis="both", labelsize=FS_TICK * fs)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)


def save_df(df, path):
    if df is None or df.empty:
        print(f"  [warn] nothing to write to {Path(path).name}")
        return
    df.to_csv(path, index=False)
    print(f"  Saved: {Path(path).name}  ({len(df)} rows)")


def mannwhitney(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or b.size < 3:
        return float("nan"), float("nan"), f"too few runs (n={a.size}, {b.size})"
    try:
        from scipy import stats
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p), ""
    except ImportError:
        return float("nan"), float("nan"), "scipy not installed"
    except Exception as e:
        return float("nan"), float("nan"), str(e)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    if not ok.any():
        return q
    sub = p[ok]
    m = sub.size
    order = np.argsort(sub)
    adj = sub[order] * m / np.arange(1, m + 1)
    adj = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0, 1)
    out = np.empty(m)
    out[order] = adj
    q[ok] = out
    return q


def stars(q):
    if q != q:
        return ""
    for thr, s in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if q < thr:
            return s
    return "ns"


def _fmt_p(v):
    if v != v:
        return ""
    return f"{v:.2e}" if v < 1e-4 else round(float(v), 6)


def _sd(values):
    v = np.asarray(values, float)
    return round(float(v.std(ddof=1)), 3) if v.size > 1 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Bound-state statistics (inlined so this script stands alone)
# ─────────────────────────────────────────────────────────────────────────────

def debounce(b, min_len):
    """Drop runs shorter than min_len so single-frame flickers aren't events."""
    b = np.asarray(b, dtype=bool).copy()
    if min_len <= 1 or b.size == 0:
        return b
    i = 0
    while i < b.size:
        j = i
        while j < b.size and b[j] == b[i]:
            j += 1
        if (j - i) < min_len and i > 0:
            b[i:j] = b[i - 1]
        i = j
    return b


def bound_stats(bound, min_event):
    """-> (frac_bound, longest bound run, number of bound->unbound events)"""
    b = debounce(bound, min_event)
    if b.size == 0:
        return 0.0, 0, 0
    frac = float(b.mean())
    longest, run = 0, 0
    for v in b:
        run = run + 1 if v else 0
        longest = max(longest, run)
    n_ev = int(np.sum((~b[1:]) & b[:-1])) if b.size > 1 else 0
    return frac, longest, n_ev


def verdict(frac_bound, mean_sb, events_per_us, drift_max):
    """Heuristic label. These thresholds are conventions, not measurements."""
    if frac_bound >= 0.95 and mean_sb >= 2.0 and events_per_us < 1.0 and drift_max < 5.0:
        return "good"
    if frac_bound >= 0.80 and mean_sb >= 1.0 and drift_max < 8.0:
        return "marginal"
    return "poor"


# ─────────────────────────────────────────────────────────────────────────────
#  Discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_runs(base_dir, runs=None):
    found = []
    for sub in base_dir.iterdir():
        if not sub.is_dir():
            continue
        m = RUN_PATTERN.match(sub.name)
        if not m:
            continue
        n = int(m.group(1))
        if runs is not None and n not in set(runs):
            continue
        d = sub / "pip_binding_quality"
        ts, occ = d / "pip_timeseries.csv", d / "pip_occupancy.csv"
        if not ts.is_file():
            print(f"    [missing] {sub.name}: no pip_timeseries.csv")
            continue
        found.append(dict(run=sub.name, index=n, dir=d, ts=ts,
                          occ=occ if occ.is_file() else None))
    found.sort(key=lambda d: d["index"])
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  One channel
# ─────────────────────────────────────────────────────────────────────────────

def analyse_channel(base_dir, channel, out_dir, args, fs):
    """Returns (df_run, df_sites, overall_row, df_site_run) or None."""
    runs = find_runs(base_dir, args.runs)
    if not runs:
        print(f"  [warn] no usable RUN*/pip_binding_quality/ in {base_dir}")
        return None

    fd = args.frame_duration or FRAME_DURATION.get(channel)
    if fd is None:
        print(f"  [warn] no frame_duration known for '{channel}' — event rates "
              f"will be per 1000 frames instead of per us. Pass -frame_duration.")
        fd = 1.0 / 1000.0
    print(f"  {len(runs)} runs: {', '.join(r['run'] for r in runs)}   "
          f"(frame_duration {fd} ns)")

    site_rows, occ_frames, ts_frames = [], [], []
    for r in runs:
        ts = pd.read_csv(r["ts"])
        missing = {"pip", "subunit", "n_salt_bridges", "frame"} - set(ts.columns)
        if missing:
            print(f"    [skip] {r['run']}: pip_timeseries.csv missing {missing}")
            continue
        ts["run"] = r["index"]
        ts_frames.append(ts)

        if r["occ"]:
            occ = pd.read_csv(r["occ"])
            occ["run"] = r["index"]
            occ["n_frames"] = ts.frame.nunique()
            occ_frames.append(occ)

        for (pip, subunit), g in ts.groupby(["pip", "subunit"]):
            g = g.sort_values("frame")
            frac, longest, n_ev = bound_stats(
                g.n_salt_bridges.values >= args.bound_sb, args.min_event)
            frac_s, _, n_ev_s = bound_stats(
                g.n_salt_bridges.values >= args.strict_sb, args.min_event)
            site_rows.append(dict(
                run=r["index"], pip=pip, subunit=subunit, n_frames=len(g),
                time_ns=len(g) * fd,
                frac_bound=frac, frac_bound_strict=frac_s,
                longest_bound_frames=longest,
                n_unbinding_events=n_ev, n_unbinding_events_strict=n_ev_s,
                mean_salt_bridges=float(g.n_salt_bridges.mean()),
                mean_contact_residues=float(g.n_contact_residues.mean())
                    if "n_contact_residues" in g else np.nan,
                drift_mean=float(g.headgroup_drift.mean())
                    if "headgroup_drift" in g else np.nan,
                drift_max=float(g.headgroup_drift.max())
                    if "headgroup_drift" in g else np.nan,
                drift_final=float(g.headgroup_drift.values[-1])
                    if "headgroup_drift" in g else np.nan))

    if not site_rows:
        print("  [warn] no usable timeseries")
        return None

    df_site_run = pd.DataFrame(site_rows)
    df_ts = pd.concat(ts_frames, ignore_index=True)
    has_drift = df_site_run.drift_max.notna().any()
    save_df(df_site_run, out_dir / "per_site_per_run.csv")

    # ── PER-RUN ──────────────────────────────────────────────────────────
    run_rows = []
    for run, g in df_site_run.groupby("run"):
        ts_r = df_ts[df_ts.run == run]
        time_us = g.time_ns.sum() / 1000.0
        events = int(g.n_unbinding_events.sum())
        rate = events / time_us if time_us else np.nan
        frac = float((ts_r.n_salt_bridges >= args.bound_sb).mean())
        mean_sb = float(ts_r.n_salt_bridges.mean())
        dmax = float(g.drift_max.max()) if has_drift else np.nan
        worst = g.loc[g.mean_salt_bridges.idxmin()]
        run_rows.append(dict(
            run=f"RUN{run}", channel_type=channel, n_sites=len(g),
            n_frames=int(g.n_frames.max()), time_ns=round(g.time_ns.max(), 1),
            frac_bound=round(frac, 4),
            frac_bound_strict=round(float((ts_r.n_salt_bridges >= args.strict_sb).mean()), 4),
            mean_salt_bridges=round(mean_sb, 3),
            sd_across_sites=_sd(g.mean_salt_bridges),
            mean_contact_residues=round(float(ts_r.n_contact_residues.mean()), 3)
                if "n_contact_residues" in ts_r else np.nan,
            unbinding_events=events,
            events_per_us=round(rate, 3) if rate == rate else np.nan,
            drift_mean=round(float(g.drift_mean.mean()), 3) if has_drift else np.nan,
            drift_max=round(dmax, 3) if has_drift else np.nan,
            weakest_site=f"{worst.subunit} ({worst.mean_salt_bridges:.2f} sb)",
            verdict=verdict(frac, mean_sb, rate if rate == rate else 0,
                            dmax if dmax == dmax else 0)))
    df_run = pd.DataFrame(run_rows)
    save_df(df_run, out_dir / "per_run_summary.csv")

    # ── PER-SITE ─────────────────────────────────────────────────────────
    site_sum = []
    for subunit, g in df_site_run.groupby("subunit"):
        ts_s = df_ts[df_ts.subunit == subunit]
        time_us = g.time_ns.sum() / 1000.0
        events = int(g.n_unbinding_events.sum())
        site_sum.append(dict(
            subunit=subunit, channel_type=channel, n_runs=g.run.nunique(),
            n_frames_total=int(g.n_frames.sum()),
            time_ns_total=round(g.time_ns.sum(), 1),
            frac_bound=round(float((ts_s.n_salt_bridges >= args.bound_sb).mean()), 4),
            frac_bound_strict=round(float((ts_s.n_salt_bridges >= args.strict_sb).mean()), 4),
            mean_salt_bridges=round(float(ts_s.n_salt_bridges.mean()), 3),
            sd_across_runs=_sd(g.mean_salt_bridges),
            weakest_run=f"RUN{int(g.loc[g.mean_salt_bridges.idxmin(), 'run'])}",
            mean_contact_residues=round(float(ts_s.n_contact_residues.mean()), 3)
                if "n_contact_residues" in ts_s else np.nan,
            unbinding_events=events,
            events_per_us=round(events / time_us, 3) if time_us else np.nan,
            drift_mean=round(float(g.drift_mean.mean()), 3) if has_drift else np.nan,
            drift_max=round(float(g.drift_max.max()), 3) if has_drift else np.nan))
    df_sites = pd.DataFrame(site_sum).sort_values("mean_salt_bridges",
                                                  ascending=False)
    save_df(df_sites, out_dir / "per_site_summary.csv")

    # ── OVERALL ──────────────────────────────────────────────────────────
    time_us_all = df_site_run.time_ns.sum() / 1000.0
    events_all = int(df_site_run.n_unbinding_events.sum())
    frac_all = float((df_ts.n_salt_bridges >= args.bound_sb).mean())
    mean_sb_all = float(df_ts.n_salt_bridges.mean())
    dmax_all = float(df_site_run.drift_max.max()) if has_drift else np.nan
    rate_all = events_all / time_us_all if time_us_all else np.nan
    overall = dict(
        channel_type=channel, n_runs=int(df_site_run.run.nunique()),
        n_sites=int(df_site_run.groupby("run").size().max()),
        total_time_ns=round(df_site_run.time_ns.sum(), 1),
        total_site_time_us=round(time_us_all, 3),
        frac_bound=round(frac_all, 4),
        frac_bound_strict=round(float((df_ts.n_salt_bridges >= args.strict_sb).mean()), 4),
        mean_salt_bridges=round(mean_sb_all, 3),
        sd_across_runs=_sd(df_run.mean_salt_bridges),
        mean_contact_residues=round(float(df_ts.n_contact_residues.mean()), 3)
            if "n_contact_residues" in df_ts else np.nan,
        unbinding_events=events_all,
        events_per_us=round(rate_all, 3) if rate_all == rate_all else np.nan,
        drift_mean=round(float(df_site_run.drift_mean.mean()), 3) if has_drift else np.nan,
        drift_max=round(dmax_all, 3) if has_drift else np.nan,
        n_runs_not_good=int((df_run.verdict != "good").sum()),
        verdict=verdict(frac_all, mean_sb_all,
                        rate_all if rate_all == rate_all else 0,
                        dmax_all if dmax_all == dmax_all else 0))
    save_df(pd.DataFrame([overall]), out_dir / "overall_summary.csv")

    # ── PER-RESIDUE ──────────────────────────────────────────────────────
    df_occ_agg = None
    if occ_frames:
        df_occ = pd.concat(occ_frames, ignore_index=True)
        n_runs_total = df_site_run.run.nunique()
        rows = []
        for (subunit, residue), g in df_occ.groupby(["subunit", "residue"]):
            w = g.n_frames.values
            rows.append(dict(
                channel_type=channel, subunit=subunit, residue=residue,
                n_runs_present=int(g.run.nunique()),
                occupancy_pooled=round(float(np.average(g.contact_occupancy, weights=w)), 4),
                occupancy_sd=_sd(g.contact_occupancy),
                occupancy_min=round(float(g.contact_occupancy.min()), 4),
                occupancy_max=round(float(g.contact_occupancy.max()), 4),
                saltbridge_pooled=round(float(np.average(g.saltbridge_occupancy, weights=w)), 4)
                    if "saltbridge_occupancy" in g else np.nan))
        df_occ_agg = pd.DataFrame(rows)
        df_occ_agg["reproducible"] = ((df_occ_agg.n_runs_present == n_runs_total)
                                      & (df_occ_agg.occupancy_min > args.repro_min))
        df_occ_agg = df_occ_agg.sort_values(["subunit", "occupancy_pooled"],
                                            ascending=[True, False])
        save_df(df_occ_agg, out_dir / "per_residue_occupancy.csv")

    # ── report ───────────────────────────────────────────────────────────
    with pd.option_context("display.width", 240, "display.max_columns", 60):
        print(f"\n  --- per run ---\n{df_run.to_string(index=False)}")
        print(f"\n  --- per site, pooled across runs ---\n"
              f"{df_sites.to_string(index=False)}")
        print(f"\n  --- {channel} overall ---")
        for k, v in overall.items():
            print(f"    {k:<24} {v}")
        if df_occ_agg is not None:
            flaky = df_occ_agg[~df_occ_agg.reproducible
                               & (df_occ_agg.occupancy_max > args.repro_min)]
            if len(flaky):
                print(f"\n  Contacts in some runs but not others "
                      f"(max > {args.repro_min}, not reproducible):")
                print(flaky[["subunit", "residue", "n_runs_present",
                             "occupancy_min", "occupancy_max"]].to_string(index=False))

    plot_per_run(df_site_run, df_run, out_dir / "per_run_binding.png",
                 channel, has_drift, fs)
    plot_per_site(df_site_run, df_sites, out_dir / "per_site_binding.png",
                  channel, fs)
    if df_occ_agg is not None:
        plot_per_residue(df_occ_agg, out_dir / "per_residue_reproducibility.png",
                         channel, args.repro_min, fs)

    return df_run, df_sites, overall, df_site_run


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_run(df_site_run, df_run, out_path, channel, has_drift, fs):
    subs = sorted(df_site_run.subunit.unique())
    runs = sorted(df_site_run.run.unique())
    width = 0.8 / max(len(subs), 1)
    nrow = 2 if has_drift else 1
    fig, axes = plt.subplots(nrow, 1, figsize=(FIG_PER_RUN[0] * fs,
                                               FIG_PER_RUN[1] * fs * nrow / 2),
                             sharex=True)
    axes = np.atleast_1d(axes)

    for i, s in enumerate(subs):
        g = df_site_run[df_site_run.subunit == s].set_index("run")
        axes[0].bar(np.arange(len(runs)) + i * width,
                    [g.mean_salt_bridges.get(r, np.nan) for r in runs],
                    width * 0.92, color=SITE_COLOR.get(s, "#777777"),
                    alpha=0.85, edgecolor="black", linewidth=BAR_EDGE_LW,
                    label=s)
    axes[0].axhline(df_run.mean_salt_bridges.mean(), color="red", ls="--",
                    linewidth=1.8 * fs, label="Mean (all runs)")
    axes[0].set_ylabel("Mean salt bridges", fontsize=FS_AXLABEL * fs)
    axes[0].set_title(f"{channel} — PIP binding per run and site\n"
                      f"({len(runs)} runs, {len(subs)} sites)",
                      fontsize=FS_TITLE * fs, fontweight="bold")
    axes[0].legend(fontsize=FS_LEGEND * fs, loc="upper left",
                   bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    _finish(axes[0], fs)

    if has_drift:
        for i, s in enumerate(subs):
            g = df_site_run[df_site_run.subunit == s].set_index("run")
            axes[1].bar(np.arange(len(runs)) + i * width,
                        [g.drift_max.get(r, np.nan) for r in runs],
                        width * 0.92, color=SITE_COLOR.get(s, "#777777"),
                        alpha=0.85, edgecolor="black", linewidth=BAR_EDGE_LW,
                        label=s)
        axes[1].axhline(df_site_run.drift_max.mean(), color="red", ls="--",
                        linewidth=1.8 * fs, label="Mean (all runs)")
        axes[1].set_ylabel("Max headgroup drift (Å)", fontsize=FS_AXLABEL * fs)
        axes[1].legend(fontsize=FS_LEGEND * fs, loc="upper left",
                       bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
        _finish(axes[1], fs)

    axes[-1].set_xticks(np.arange(len(runs)) + 0.4 - width / 2)
    axes[-1].set_xticklabels([f"RUN{r}" for r in runs], rotation=45,
                             ha="right", fontsize=FS_TICK * fs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {Path(out_path).name}")


def plot_per_site(df_site_run, df_sites, out_path, channel, fs):
    """One box per site over the RUNS, each dot a run."""
    subs = list(df_sites.subunit)
    data = [df_site_run[df_site_run.subunit == s].mean_salt_bridges.values
            for s in subs]
    n_runs = max(len(d) for d in data) if data else 0
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(max(FIG_PER_SITE[0], 2.6 * len(subs)) * fs,
                                    FIG_PER_SITE[1] * fs))
    pos = np.arange(1, len(subs) + 1)
    if n_runs >= 3:
        bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="green", linewidth=2.2 * fs),
                        showmeans=True, meanline=True,
                        meanprops=dict(color="red", linestyle="--",
                                       linewidth=1.8 * fs),
                        whiskerprops=dict(linewidth=1.3 * fs),
                        capprops=dict(linewidth=1.3 * fs))
        for box, s in zip(bp["boxes"], subs):
            box.set_facecolor(SITE_COLOR.get(s, "#777777"))
            box.set_alpha(0.6)
            box.set_edgecolor("black")
            box.set_linewidth(BAR_EDGE_LW)
        ax.plot([], [], color="green", linewidth=2.2 * fs, label="Median")
        ax.plot([], [], color="red", linestyle="--", linewidth=1.8 * fs,
                label="Mean")
    else:
        ax.bar(pos, [np.mean(d) for d in data], width=0.55,
               color=[SITE_COLOR.get(s, "#777777") for s in subs], alpha=0.7,
               edgecolor="black", linewidth=BAR_EDGE_LW)
        print(f"  [note] {n_runs} runs — per-site drawn as bars, not boxes")

    for p, vals in zip(pos, data):
        v = np.asarray(vals, float)
        ax.scatter(p + rng.uniform(-0.16, 0.16, v.size), v, s=46 * fs,
                   color="black", alpha=0.45, edgecolors="none", zorder=4)
    ax.scatter([], [], s=46 * fs, color="black", alpha=0.45, label="One run")

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{s}\n({len(d)} runs)" for s, d in zip(subs, data)],
                       fontsize=FS_TICK * fs)
    ax.set_ylabel("Mean salt bridges", fontsize=FS_AXLABEL * fs)
    ax.set_title(f"{channel} — PIP binding per site\n(each dot is one run)",
                 fontsize=FS_TITLE * fs, fontweight="bold")
    ax.legend(fontsize=FS_LEGEND * fs, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    _finish(ax, fs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {Path(out_path).name}")


def plot_per_residue(df_occ, out_path, channel, repro_min, fs):
    top = df_occ[df_occ.occupancy_max > 0.3]
    if top.empty:
        print("  [skip] per-residue plot — no residue above 0.3 occupancy")
        return
    fig, ax = plt.subplots(figsize=(FIG_RESIDUE[0] * fs,
                                    max(FIG_RESIDUE[1], 0.30 * len(top)) * fs))
    y = np.arange(len(top))
    ax.barh(y, top.occupancy_pooled,
            xerr=top.occupancy_sd.fillna(0).values, capsize=3,
            color=["#27AE60" if r else "#C0392B" for r in top.reproducible],
            alpha=0.85, edgecolor="black", linewidth=BAR_EDGE_LW)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}  {r}" for s, r in zip(top.subunit, top.residue)],
                       fontsize=FS_TICK * fs * 0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Contact occupancy (pooled over runs)",
                  fontsize=FS_AXLABEL * fs)
    ax.set_xlim(0, 1)
    ax.plot([], [], color="#27AE60", linewidth=10 * fs,
            label=f"In every run, min > {repro_min}")
    ax.plot([], [], color="#C0392B", linewidth=10 * fs, label="Not reproducible")
    ax.set_title(f"{channel} — per-residue contacts\n"
                 f"(error bars = SD across runs)",
                 fontsize=FS_TITLE * fs, fontweight="bold")
    ax.legend(fontsize=FS_LEGEND * fs, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.tick_params(axis="x", labelsize=FS_TICK * fs)
    ax.grid(True, alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {Path(out_path).name}")


def plot_channel_comparison(per_channel, out_path, fs):
    """One box per channel over its RUNS, each dot a run."""
    channels = list(per_channel)
    data = [per_channel[c]["df_run"].mean_salt_bridges.values for c in channels]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(FIG_CHANNEL[0], 2.8 * len(channels)) * fs,
                                    FIG_CHANNEL[1] * fs))
    pos = np.arange(1, len(channels) + 1)
    if min(len(d) for d in data) >= 3:
        bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="green", linewidth=2.2 * fs),
                        showmeans=True, meanline=True,
                        meanprops=dict(color="red", linestyle="--",
                                       linewidth=1.8 * fs))
        for box, c in zip(bp["boxes"], channels):
            box.set_facecolor(CHANNEL_COLOR.get(c, "#777777"))
            box.set_alpha(0.6)
            box.set_edgecolor("black")
            box.set_linewidth(BAR_EDGE_LW)
        for w in bp["whiskers"] + bp["caps"]:
            w.set_linewidth(1.3 * fs)
        ax.plot([], [], color="green", linewidth=2.2 * fs, label="Median")
        ax.plot([], [], color="red", linestyle="--", linewidth=1.8 * fs,
                label="Mean")
    else:
        ax.bar(pos, [np.mean(d) for d in data], width=0.55,
               color=[CHANNEL_COLOR.get(c, "#777777") for c in channels],
               alpha=0.7, edgecolor="black", linewidth=BAR_EDGE_LW)
    for p, vals in zip(pos, data):
        v = np.asarray(vals, float)
        ax.scatter(p + rng.uniform(-0.16, 0.16, v.size), v, s=46 * fs,
                   color="black", alpha=0.45, edgecolors="none", zorder=4)
    ax.scatter([], [], s=46 * fs, color="black", alpha=0.45, label="One run")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{c}\n({len(d)} runs)" for c, d in zip(channels, data)],
                       fontsize=FS_TICK * fs)
    ax.set_ylabel("Mean salt bridges per run", fontsize=FS_AXLABEL * fs)
    ax.set_title("PIP binding by channel\n(each dot is one run)",
                 fontsize=FS_TITLE * fs, fontweight="bold")
    ax.legend(fontsize=FS_LEGEND * fs, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    _finish(ax, fs)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-channel tests
# ─────────────────────────────────────────────────────────────────────────────

def channel_tests(per_channel, out_dir):
    channels = list(per_channel)
    rows = []
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            a, b = channels[i], channels[j]
            ra = per_channel[a]["df_run"]
            rb = per_channel[b]["df_run"]

            for quantity, col in [("mean salt bridges", "mean_salt_bridges"),
                                  ("frac_bound (strict)", "frac_bound_strict"),
                                  ("max drift", "drift_max")]:
                va = ra[col].dropna().values
                vb = rb[col].dropna().values
                if va.size == 0 or vb.size == 0:
                    continue
                u, p, note = mannwhitney(va, vb)
                rows.append(dict(
                    quantity=quantity, level="run-clustered", group_a=a, group_b=b,
                    n_a=va.size, n_b=vb.size,
                    median_a=round(float(np.median(va)), 3),
                    median_b=round(float(np.median(vb)), 3),
                    statistic=round(u, 3) if u == u else "",
                    p_value=_fmt_p(p), q_value="", stars="", note=note))

            # per-site, FDR-corrected across sites
            sa, sb = per_channel[a]["df_site_run"], per_channel[b]["df_site_run"]
            shared = sorted(set(sa.subunit) & set(sb.subunit))
            per_site = []
            for s in shared:
                va = sa[sa.subunit == s].mean_salt_bridges.values
                vb = sb[sb.subunit == s].mean_salt_bridges.values
                u, p, note = mannwhitney(va, vb)
                per_site.append(dict(
                    quantity=f"salt bridges {s}", level="run-clustered",
                    group_a=a, group_b=b, n_a=va.size, n_b=vb.size,
                    median_a=round(float(np.median(va)), 3) if va.size else "",
                    median_b=round(float(np.median(vb)), 3) if vb.size else "",
                    statistic=round(u, 3) if u == u else "",
                    p_value=_fmt_p(p), q_value="", stars="", note=note))
            qs = bh_fdr([float(r["p_value"]) if r["p_value"] != "" else np.nan
                         for r in per_site])
            for r, q in zip(per_site, qs):
                r["q_value"] = _fmt_p(q)
                r["stars"] = stars(q)
            rows.extend(per_site)

    if not rows:
        return
    df = pd.DataFrame(rows)
    save_df(df, out_dir / "channel_tests.csv")
    print("\n  === channel comparison (run-clustered, n = runs) ===")
    for r in rows:
        if r["level"] == "run-clustered" and not r["quantity"].startswith("salt bridges "):
            print(f"    {r['quantity']:<22} {r['group_a']} (med {r['median_a']}, "
                  f"n={r['n_a']}) vs {r['group_b']} (med {r['median_b']}, "
                  f"n={r['n_b']}): p={r['p_value']}"
                  + (f"  [{r['note']}]" if r["note"] else ""))
    site_rows = [r for r in rows if r["quantity"].startswith("salt bridges ")]
    if site_rows:
        print("\n    per-site salt bridges, BH-FDR across sites:")
        for r in site_rows:
            print(f"      {r['quantity'][13:]:<6} {r['median_a']:>7} vs "
                  f"{r['median_b']:>7}  p={r['p_value']} q={r['q_value']} "
                  f"{r['stars']}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", nargs="+",
                    help="Channel folder(s) holding RUN1, RUN2, ... "
                         "Pass G12 and G2 together to compare them.")
    ap.add_argument("-runs", nargs="+", type=int, default=None,
                    help="Only these run numbers (default: all found)")
    ap.add_argument("-bound_sb", type=int, default=1,
                    help="Salt bridges needed to call a site bound (default 1)")
    ap.add_argument("-strict_sb", type=int, default=3,
                    help="Stricter bar, reported alongside (default 3). Use this "
                         "when frac_bound saturates at 1.0 and ranks nothing.")
    ap.add_argument("-min_event", type=int, default=5,
                    help="Minimum run length in frames for a state change to "
                         "count as real (default 5)")
    ap.add_argument("-frame_duration", type=float, default=None,
                    help="ns per frame, overriding the built-in per-channel "
                         "table (G12 0.075, G2 0.4)")
    ap.add_argument("-repro_min", type=float, default=0.5,
                    help="A contact counts as reproducible if it is present in "
                         "every run with occupancy above this (default 0.5)")
    ap.add_argument("-channel", default=None,
                    help="Label for plot titles when ONE folder is given "
                         "(default: the folder name)")
    ap.add_argument("-out", default=None,
                    help="Write everything to this one directory instead of "
                         "each channel's own pip_binding_quality_aggregate/")
    ap.add_argument("-fontscale", "--fontscale", type=float, default=1.0,
                    help="Multiply every font size (and figure size) by this")
    ap.add_argument("-stats", action="store_true",
                    help="Mann-Whitney between channels, run-clustered")
    args = ap.parse_args()

    fs = args.fontscale
    dirs = [Path(d) for d in args.base_dir]
    for d in dirs:
        if not d.is_dir():
            sys.exit(f"ERROR: not a directory: {d}")

    print("=" * 70)
    print("AGGREGATE PIP BINDING QUALITY")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  bound_sb={args.bound_sb}  strict_sb={args.strict_sb}  "
          f"min_event={args.min_event}  repro_min={args.repro_min}")

    forced_out = Path(args.out).resolve() if args.out else None
    if forced_out:
        forced_out.mkdir(parents=True, exist_ok=True)

    per_channel, first_out = {}, None
    overall_rows = []

    for d in dirs:
        channel = args.channel if (args.channel and len(dirs) == 1) else d.name
        out_dir = forced_out or (d / "pip_binding_quality_aggregate")
        out_dir.mkdir(parents=True, exist_ok=True)
        if first_out is None:
            first_out = out_dir

        print(f"\n{'=' * 70}\n  {channel}   ({d})\n{'=' * 70}")
        res = analyse_channel(d, channel, out_dir, args, fs)
        if res is None:
            continue
        df_run, df_sites, overall, df_site_run = res
        per_channel[channel] = dict(df_run=df_run, df_sites=df_sites,
                                    df_site_run=df_site_run)
        overall_rows.append(overall)

    if len(per_channel) >= 2:
        out_dir = forced_out or first_out
        print(f"\n{'=' * 70}\n  CHANNEL COMPARISON\n{'=' * 70}")
        df_cmp = pd.DataFrame(overall_rows)
        save_df(df_cmp, out_dir / "channel_comparison.csv")
        with pd.option_context("display.width", 240, "display.max_columns", 60):
            print(df_cmp[["channel_type", "n_runs", "frac_bound",
                          "frac_bound_strict", "mean_salt_bridges",
                          "sd_across_runs", "events_per_us", "drift_max",
                          "verdict"]].to_string(index=False))
        plot_channel_comparison(per_channel, out_dir / "channel_binding_box.png", fs)
        if args.stats:
            channel_tests(per_channel, out_dir)
    elif args.stats:
        print("\n  [note] only one channel loaded — pass both channel folders "
              "in the same call for the comparison.")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()


'''
python3 aggregate_pip_binding_quality.py \
  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12

python3 aggregate_pip_binding_quality.py \
  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2

python3 aggregate_pip_binding_quality.py \
  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N

python3 aggregate_pip_binding_quality.py \
  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q

# both channels in one call, with run-clustered tests
python3 aggregate_pip_binding_quality.py \
  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
  /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
  -stats

# only some runs, stricter bound criterion
python3 aggregate_pip_binding_quality.py <base_dir> -runs 1 2 3 -strict_sb 4
'''