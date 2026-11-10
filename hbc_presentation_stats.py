"""
hbc_presentation_stats.py
=========================
Mann-Whitney U and Kolmogorov-Smirnov tests, tables and figures for
compare_hbc_girk1_diagonal.py.

ONE VALUE PER RUN
    Both tests compare the per-run values (e.g. the median HBC distance of each
    run: 20 numbers for WT, N for the mutant). Frames within a run are strongly
    correlated, so testing all frames as independent samples would give
    meaningless p-values; the run is the independent sample.

    Mann-Whitney U   are the values generally larger in one group? (a shift)
    Kolmogorov-      are the two sets of values from the same distribution?
    Smirnov          D = largest gap between the two cumulative curves (0-1),
                     sensitive to shifts and to differences in spread/shape
    Neither assumes a normal distribution. For small groups the exact p-values
    are used.

THE TABLE also describes the pooled histograms (all frames of all runs) with
numbers: median, mean, 25-75 % range (IQR) and peak position. These describe
the curves; the p-values come from the per-run tests.

OUTPUTS (written by compare_hbc_girk1_diagonal.py)
    stats_table.csv / stats_table.png    one row per diagonal and measure
    stats_per_run.png                    per-run medians and means, both tests
    stats_ks_cdf.png                     the cumulative curves the KS test
                                         compares, with D marked
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu

DIAG_TITLES = {"GIRK1_diag": "GIRK1–GIRK2 diagonal", "GIRK2_diag": "GIRK2–GIRK2 diagonal"}
MEASURES = {"median": "median distance per run", "mean": "mean distance per run"}


def _vals(rows, key):
    return np.array([float(r[key]) for r in rows if str(r.get(key, "")).strip() != ""])


def _sem(v):
    return float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")


def _peak(x, width=0.05, sigma=0.15):
    """Position of the maximum of the histogram, lightly smoothed."""
    if not x.size:
        return float("nan")
    edges = np.arange(np.floor(x.min()), np.ceil(x.max()) + width, width)
    c = np.histogram(x, bins=edges)[0].astype(float)
    k = np.arange(-4 * sigma, 4 * sigma + width / 2, width)
    k = np.exp(-0.5 * (k / sigma) ** 2)
    c = np.convolve(c, k / k.sum(), mode="same")
    i = int(np.argmax(c))
    return float(0.5 * (edges[i] + edges[i + 1]))


def _fmt_p(p):
    return "" if p != p else (f"{p:.2g}" if p < 0.001 else f"{p:.3f}")


def _stars(p):
    return "ns" if p != p or p >= 0.05 else "*" if p >= 0.01 else "**" if p >= 0.001 else "***"


def run_tests(wt_rows, mut_rows, wt_pool, mut_pool, wt_label, mut_label):
    out = []
    for diag in DIAG_TITLES:
        for meas in MEASURES:
            key = f"{meas}_{diag}"
            a, b = _vals(wt_rows, key), _vals(mut_rows, key)
            if a.size < 2 or b.size < 2:
                continue
            mw = mannwhitneyu(a, b, alternative="two-sided")
            ks = ks_2samp(a, b, method="exact" if a.size * b.size < 10000 else "auto")
            pa, pb = wt_pool[diag], mut_pool[diag]
            q = lambda x: np.percentile(x, [25, 75]) if x.size else (np.nan, np.nan)
            qa, qb = q(pa), q(pb)
            out.append({
                "diagonal": DIAG_TITLES[diag], "measure": MEASURES[meas],
                "n_runs_wt": a.size, "n_runs_mut": b.size,
                f"{wt_label}_mean_sem": f"{a.mean():.2f} ± {_sem(a):.2f}",
                f"{mut_label}_mean_sem": f"{b.mean():.2f} ± {_sem(b):.2f}",
                f"{wt_label}_median_of_runs": round(float(np.median(a)), 2),
                f"{mut_label}_median_of_runs": round(float(np.median(b)), 2),
                "diff_median_mut_minus_wt": round(float(np.median(b) - np.median(a)), 2),
                "mannwhitney_U": round(float(mw.statistic), 1),
                "mannwhitney_p": float(f"{mw.pvalue:.3g}"),
                "ks_D": round(float(ks.statistic), 3),
                "ks_p": float(f"{ks.pvalue:.3g}"),
                # descriptors of the pooled histograms (all frames)
                f"{wt_label}_hist_median": round(float(np.median(pa)), 2) if pa.size else "",
                f"{mut_label}_hist_median": round(float(np.median(pb)), 2) if pb.size else "",
                f"{wt_label}_hist_mean": round(float(pa.mean()), 2) if pa.size else "",
                f"{mut_label}_hist_mean": round(float(pb.mean()), 2) if pb.size else "",
                f"{wt_label}_hist_IQR": f"{qa[0]:.2f}–{qa[1]:.2f}",
                f"{mut_label}_hist_IQR": f"{qb[0]:.2f}–{qb[1]:.2f}",
                f"{wt_label}_hist_peak": round(_peak(pa), 2),
                f"{mut_label}_hist_peak": round(_peak(pb), 2),
            })
    return out


def plot_table(stats, wt_label, mut_label, path, source=""):
    """The main numbers as a figure, for slides."""
    cols = ["Diagonal", "Per-run\nvalue",
            f"{wt_label}\nmean ± SEM", f"{mut_label}\nmean ± SEM",
            "Mann–Whitney\np", "KS\nD", "KS\np",
            f"Histogram peak\n{wt_label} / {mut_label}",
            f"Histogram IQR\n{wt_label} / {mut_label}"]
    cells = []
    for r in stats:
        cells.append([
            r["diagonal"].replace(" diagonal", ""), r["measure"].split()[0],
            r[f"{wt_label}_mean_sem"] + " Å", r[f"{mut_label}_mean_sem"] + " Å",
            f"{_fmt_p(r['mannwhitney_p'])} {_stars(r['mannwhitney_p'])}",
            f"{r['ks_D']:.2f}",
            f"{_fmt_p(r['ks_p'])} {_stars(r['ks_p'])}",
            f"{r[f'{wt_label}_hist_peak']:.1f} / {r[f'{mut_label}_hist_peak']:.1f} Å",
            f"{r[f'{wt_label}_hist_IQR']} / {r[f'{mut_label}_hist_IQR']}",
        ])
    fig, ax = plt.subplots(figsize=(15, 0.5 * len(cells) + 1.2))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center",
                 colWidths=[0.10, 0.07, 0.11, 0.11, 0.10, 0.06, 0.09, 0.13, 0.19])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.9)
    for (i, j), c in t.get_celld().items():
        if i == 0:
            c.set_facecolor("#E8E8E8")
            c.set_text_props(fontweight="bold")
        elif j in (4, 6) and "ns" not in c.get_text().get_text():
            c.set_text_props(fontweight="bold")
    n = stats[0] if stats else {}
    ax.set_title(f"{wt_label} (n = {n.get('n_runs_wt', '?')} runs) vs {mut_label} "
                 f"(n = {n.get('n_runs_mut', '?')} runs): tests on one value per run"
                 + (f"\n{source}" if source else ""), fontsize=10, pad=8)
    fig.text(0.5, 0.02, "* p < 0.05   ** p < 0.01   *** p < 0.001   ns not significant.   "
             "Histogram peak and IQR (25–75 %) describe the pooled curves (all frames).",
             ha="center", fontsize=8)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_per_run(wt_rows, mut_rows, stats, wt_label, mut_label, colors, path, source=""):
    """Per-run medians and means, one dot per run, with both p-values."""
    lookup = {(r["diagonal"], r["measure"]): r for r in stats}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    panels = [(d, m) for d in DIAG_TITLES for m in MEASURES]
    for ax, (diag, meas) in zip(axes, panels):
        key = f"{meas}_{diag}"
        data = [_vals(wt_rows, key), _vals(mut_rows, key)]
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops=dict(color="black"))
        for i, (v, c) in enumerate(zip(data, (colors["WT"], colors["MUT"])), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, s=30, color=c,
                       edgecolors="black", linewidths=0.5, zorder=3)
        r = lookup.get((DIAG_TITLES[diag], MEASURES[meas]))
        txt = (f"MW p = {_fmt_p(r['mannwhitney_p'])} {_stars(r['mannwhitney_p'])}\n"
               f"KS D = {r['ks_D']:.2f}, p = {_fmt_p(r['ks_p'])} {_stars(r['ks_p'])}") if r else ""
        ax.set_title(f"{DIAG_TITLES[diag]}\n{MEASURES[meas]}\n{txt}", fontsize=9)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"{wt_label}\n(n={data[0].size})", f"{mut_label}\n(n={data[1].size})"])
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("HBC min. atom–atom distance (Å)")
    if source:
        fig.suptitle(source, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_ks_cdf(wt_rows, mut_rows, stats, wt_label, mut_label, colors, path, source=""):
    """Cumulative curves of the per-run medians (what the KS test compares), D marked."""
    lookup = {(r["diagonal"], r["measure"]): r for r in stats}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, diag in zip(axes, DIAG_TITLES):
        key = f"median_{diag}"
        a, b = np.sort(_vals(wt_rows, key)), np.sort(_vals(mut_rows, key))
        if not a.size or not b.size:
            continue
        for v, lab, c in ((a, wt_label, colors["WT"]), (b, mut_label, colors["MUT"])):
            ax.step(np.r_[v[0] - 0.5, v, v[-1] + 0.5], np.r_[0, np.arange(1, v.size + 1) / v.size, 1],
                    where="post", linewidth=2, color=c, label=f"{lab} ({v.size} runs)")
        grid = np.sort(np.r_[a, b])
        Fa = np.searchsorted(a, grid, side="right") / a.size
        Fb = np.searchsorted(b, grid, side="right") / b.size
        i = int(np.argmax(np.abs(Fa - Fb)))
        ax.annotate("", xy=(grid[i], Fa[i]), xytext=(grid[i], Fb[i]),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.4))
        r = lookup.get((DIAG_TITLES[diag], MEASURES["median"]))
        ax.text(grid[i] + 0.1, (Fa[i] + Fb[i]) / 2,
                f"D = {r['ks_D']:.2f}\np = {_fmt_p(r['ks_p'])}" if r else "", fontsize=9, va="center")
        ax.set_title(DIAG_TITLES[diag])
        ax.set_xlabel("Median HBC distance of a run (Å)")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction of runs with median ≤ x")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Kolmogorov–Smirnov test: D is the largest vertical gap between the curves"
                 + (f"\n{source}" if source else ""), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
