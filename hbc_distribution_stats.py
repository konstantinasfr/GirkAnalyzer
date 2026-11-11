"""
hbc_distribution_stats.py
=========================
Distribution-level statistics and plots for compare_hbc_girk1_diagonal.py.

THE INDEPENDENT UNIT IS THE RUN, NOT THE FRAME
    Consecutive frames of one run are strongly correlated, so treating the
    6800 frames of a run as 6800 independent samples gives absurdly small
    p-values. Every test and confidence interval here resamples or shuffles
    WHOLE RUNS. The pooled curve of a system is the histogram of all its frames
    (each run weighs as much as it has frames), exactly as in the density plot.

PER DIAGONAL, WT vs mutant
    peak (mode)    position of the maximum of the pooled curve, smoothed with a
                   0.15 A Gaussian so single-bin noise does not set it
    median         median of the pooled curve
    diff           mutant - WT
    95% CI         bootstrap: resample runs with replacement inside each system
                   (2000 times), recompute the pooled curves, take the 2.5 and
                   97.5 percentiles of the difference
    permutation p  are the two pooled curves from the same population? The run
                   labels (WT / mutant) are shuffled 10000 times, keeping the
                   number of runs per system, and the p-value is the fraction of
                   shuffles giving a difference at least as large as the real one
                     - |median difference|        a shift
                     - Kolmogorov-Smirnov D       largest gap between the two
                                                  cumulative curves (any shape
                                                  difference), 0..1
                     - Wasserstein distance (A)   how far, in A, one curve must
                                                  be moved to become the other
    Mann-Whitney   on the per-run medians (one number per run)
    Cliff's delta  effect size of the per-run medians, -1..1;
                   |d| < 0.15 negligible, < 0.33 small, < 0.47 medium, else large

The smallest possible permutation p is 1/(n_perm+1) ~ 1e-4; with few runs the
number of distinct label shuffles is limited (e.g. 20 WT + 3 mutant runs give
only 1771), so p cannot go below ~1/1771 there however different the curves are.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINE_BIN = 0.02          # A, for the statistics
PLOT_BIN = 1.0 / 6       # A, for the figures (as before)
MODE_SMOOTH = 0.15       # A, Gaussian sigma used to locate the peak
N_BOOT = 2000
N_PERM = 10000
TITLES = {"GIRK1_diag": "GIRK1–GIRK2 diagonal", "GIRK2_diag": "GIRK2–GIRK2 diagonal"}


# ── binned per-run curves ──────────────────────────────────────────────────

def _edges(per_run_all, width):
    """Bins covering all data (nothing cut at either end)."""
    vals = [a for runs in per_run_all for a in runs if a.size]
    lo = min([1.5] + [float(np.min(a)) for a in vals])
    hi = max([12.0] + [float(np.max(a)) for a in vals])
    lo = 0.0 if width == FINE_BIN else np.floor(lo / width) * width
    hi = np.ceil((hi + 1e-9) / width) * width + width
    return np.arange(lo, hi + width / 2, width)


def _counts(runs, edges):
    """(n_runs, n_bins) frame counts per run."""
    return np.array([np.histogram(a, bins=edges)[0] for a in runs], dtype=float)


def _cdf(c):
    s = c.sum()
    return np.cumsum(c) / s if s else np.zeros_like(c)


def _median(c, edges):
    cdf = _cdf(c)
    i = int(np.searchsorted(cdf, 0.5))
    lo_cdf = cdf[i - 1] if i else 0.0
    frac = (0.5 - lo_cdf) / max(cdf[i] - lo_cdf, 1e-12)
    return float(edges[i] + frac * (edges[i + 1] - edges[i]))


_KERNEL = None


def _mode(c, edges):
    global _KERNEL
    if _KERNEL is None:
        x = np.arange(-4 * MODE_SMOOTH, 4 * MODE_SMOOTH + FINE_BIN / 2, FINE_BIN)
        _KERNEL = np.exp(-0.5 * (x / MODE_SMOOTH) ** 2)
        _KERNEL /= _KERNEL.sum()
    sm = np.convolve(c, _KERNEL, mode="same")
    i = int(np.argmax(sm))
    return float(0.5 * (edges[i] + edges[i + 1]))


def _ks(c1, c2):
    return float(np.max(np.abs(_cdf(c1) - _cdf(c2))))


def _w1(c1, c2):
    return float(np.sum(np.abs(_cdf(c1) - _cdf(c2))) * FINE_BIN)


def _cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not a.size or not b.size:
        return float("nan")
    gt = (b[:, None] > a[None, :]).sum()
    lt = (b[:, None] < a[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))     # > 0: mutant larger


def _magnitude(d):
    d = abs(d)
    return ("negligible" if d < 0.15 else "small" if d < 0.33
            else "medium" if d < 0.47 else "large")


# ── statistics ─────────────────────────────────────────────────────────────

def distribution_tests(wt_runs, mut_runs, wt_label, mut_label, seed=0):
    """
    wt_runs / mut_runs: {diag: [array per run]}.
    Returns a list of rows (one per diagonal and statistic) for a CSV.
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        mannwhitneyu = None
    rng = np.random.default_rng(seed)
    rows = []
    for diag in TITLES:
        A, B = wt_runs[diag], mut_runs[diag]
        if len(A) < 2 or len(B) < 2:
            continue
        edges = _edges([A, B], FINE_BIN)
        CA, CB = _counts(A, edges), _counts(B, edges)
        ca, cb = CA.sum(0), CB.sum(0)
        nA, nB = len(A), len(B)

        obs = dict(median=_median(cb, edges) - _median(ca, edges),
                   peak=_mode(cb, edges) - _mode(ca, edges),
                   ks=_ks(ca, cb), w1=_w1(ca, cb))

        # bootstrap over runs -> CI of the differences
        boot = {"median": [], "peak": []}
        for _ in range(N_BOOT):
            a = CA[rng.integers(0, nA, nA)].sum(0)
            b = CB[rng.integers(0, nB, nB)].sum(0)
            boot["median"].append(_median(b, edges) - _median(a, edges))
            boot["peak"].append(_mode(b, edges) - _mode(a, edges))
        ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in boot.items()}

        # permutation of run labels -> "same population?"
        allC = np.vstack([CA, CB])
        hits = {"median": 0, "ks": 0, "w1": 0}
        for _ in range(N_PERM):
            idx = rng.permutation(nA + nB)
            a, b = allC[idx[:nA]].sum(0), allC[idx[nA:]].sum(0)
            hits["median"] += abs(_median(b, edges) - _median(a, edges)) >= abs(obs["median"]) - 1e-12
            hits["ks"] += _ks(a, b) >= obs["ks"] - 1e-12
            hits["w1"] += _w1(a, b) >= obs["w1"] - 1e-12
        p = {k: (v + 1) / (N_PERM + 1) for k, v in hits.items()}

        run_med_a = [float(np.median(x)) for x in A]
        run_med_b = [float(np.median(x)) for x in B]
        mw = (float(mannwhitneyu(run_med_a, run_med_b, alternative="two-sided").pvalue)
              if mannwhitneyu else float("nan"))
        cd = _cliffs_delta(run_med_a, run_med_b)

        base = dict(diagonal=diag, n_runs_wt=nA, n_runs_mut=nB,
                    frames_wt=int(ca.sum()), frames_mut=int(cb.sum()))
        rows += [
            dict(base, statistic="peak (mode) of pooled curve, A",
                 wt=round(_mode(ca, edges), 2), mut=round(_mode(cb, edges), 2),
                 diff_mut_minus_wt=round(obs["peak"], 2),
                 ci95_low=round(ci["peak"][0], 2), ci95_high=round(ci["peak"][1], 2),
                 p_value="", test="bootstrap over runs"),
            dict(base, statistic="median of pooled curve, A",
                 wt=round(_median(ca, edges), 2), mut=round(_median(cb, edges), 2),
                 diff_mut_minus_wt=round(obs["median"], 2),
                 ci95_low=round(ci["median"][0], 2), ci95_high=round(ci["median"][1], 2),
                 p_value=float(f"{p['median']:.3g}"), test="permutation of run labels"),
            dict(base, statistic="Kolmogorov-Smirnov D (0-1)", wt="", mut="",
                 diff_mut_minus_wt=round(obs["ks"], 3), ci95_low="", ci95_high="",
                 p_value=float(f"{p['ks']:.3g}"), test="permutation of run labels"),
            dict(base, statistic="Wasserstein distance, A", wt="", mut="",
                 diff_mut_minus_wt=round(obs["w1"], 3), ci95_low="", ci95_high="",
                 p_value=float(f"{p['w1']:.3g}"), test="permutation of run labels"),
            dict(base, statistic="per-run medians, A (median over runs)",
                 wt=round(float(np.median(run_med_a)), 2), mut=round(float(np.median(run_med_b)), 2),
                 diff_mut_minus_wt=round(float(np.median(run_med_b) - np.median(run_med_a)), 2),
                 ci95_low="", ci95_high="", p_value=float(f"{mw:.3g}"),
                 test=f"Mann-Whitney on run medians; Cliff's delta {cd:+.2f} ({_magnitude(cd)})"),
        ]
    for r in rows:
        r["wt_label"], r["mut_label"] = wt_label, mut_label
    return rows


# ── figures ────────────────────────────────────────────────────────────────

def _plot_edges(wt_runs, mut_runs):
    return _edges([wt_runs[d] for d in TITLES] + [mut_runs[d] for d in TITLES], PLOT_BIN)


def plot_counts(wt_runs, mut_runs, wt_label, mut_label, colors, threshold, path, source=""):
    """Unscaled: raw frame counts (top) and frames per run (bottom)."""
    edges = _plot_edges(wt_runs, mut_runs)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True)
    for j, diag in enumerate(TITLES):
        for runs, lab, key in ((wt_runs, wt_label, "WT"), (mut_runs, mut_label, "MUT")):
            if not runs[diag]:
                continue
            c = _counts(runs[diag], edges).sum(0)
            n = len(runs[diag])
            for i, (y, tag) in enumerate(((c, ""), (c / n, ""))):
                axes[i, j].stairs(y, edges, linewidth=2, color=colors[key],
                                  label=f"{lab} ({n} runs)")
        for i in range(2):
            axes[i, j].axvline(threshold, color="grey", linestyle="--", linewidth=1)
            axes[i, j].spines[["top", "right"]].set_visible(False)
        axes[0, j].set_title(TITLES[diag])
        axes[1, j].set_xlabel("Min. atom–atom distance (Å)")
    axes[0, 0].set_ylabel("Frames (all runs, unscaled)")
    axes[1, 0].set_ylabel("Frames per run (counts / n runs)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(("Unscaled histograms. Top: raw counts — a system with more runs has a "
                  "bigger curve.\nBottom: divided by the number of runs, so the two "
                  "systems are comparable.") + (f"\n{source}" if source else ""), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_bootstrap_bands(wt_runs, mut_runs, wt_label, mut_label, colors, threshold,
                         path, source="", seed=0):
    """Density curves with a 95% band from resampling whole runs."""
    rng = np.random.default_rng(seed)
    edges = _plot_edges(wt_runs, mut_runs)
    width = np.diff(edges)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, diag in zip(axes, TITLES):
        for runs, lab, key in ((wt_runs, wt_label, "WT"), (mut_runs, mut_label, "MUT")):
            if not runs[diag]:
                continue
            C = _counts(runs[diag], edges)
            n = len(C)
            dens = lambda c: c / (c.sum() * width)
            boot = np.array([dens(C[rng.integers(0, n, n)].sum(0)) for _ in range(N_BOOT)])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            ax.stairs(hi, edges, baseline=lo, fill=True, alpha=0.25, color=colors[key])
            ax.stairs(dens(C.sum(0)), edges, linewidth=2, color=colors[key],
                      label=f"{lab} ({n} runs)")
        ax.axvline(threshold, color="grey", linestyle="--", linewidth=1)
        ax.set_title(TITLES[diag])
        ax.set_xlabel("Min. atom–atom distance (Å)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Density (all runs pooled)")
    axes[0].legend(frameon=False)
    fig.suptitle("Shaded: 95% band from resampling whole runs (bootstrap). Where the bands "
                 "do not overlap, the difference does not depend on which runs you have."
                 + (f"\n{source}" if source else ""), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_run(wt_runs, mut_runs, wt_label, mut_label, colors, threshold, path, source=""):
    """One thin density curve per run, pooled curve on top."""
    edges = _plot_edges(wt_runs, mut_runs)
    width = np.diff(edges)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, diag in zip(axes, TITLES):
        for runs, lab, key in ((wt_runs, wt_label, "WT"), (mut_runs, mut_label, "MUT")):
            if not runs[diag]:
                continue
            C = _counts(runs[diag], edges)
            for c in C:
                if c.sum():
                    ax.stairs(c / (c.sum() * width), edges, linewidth=0.7,
                              color=colors[key], alpha=0.35)
            tot = C.sum(0)
            ax.stairs(tot / (tot.sum() * width), edges, linewidth=2.4, color=colors[key],
                      label=f"{lab}: {len(C)} runs (thin) + pooled (thick)")
        ax.axvline(threshold, color="grey", linestyle="--", linewidth=1)
        ax.set_title(TITLES[diag])
        ax.set_xlabel("Min. atom–atom distance (Å)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=8)
    if source:
        fig.suptitle(source, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
