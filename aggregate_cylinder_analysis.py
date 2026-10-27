#!/usr/bin/env python3
"""
Aggregate Cylinder Analysis
===========================
Loads cylinder_analysis.pkl from every RUN* folder of one channel and builds
the across-run K+ histograms along each subunit's GLU-CA -> ASN-CA axis.

Input per run
-------------
    <base_dir>/RUN<n>/cylinder_analysis/cylinder_analysis.pkl
(written by analysis/cylinder_analysis.py)

How runs are combined
---------------------
- Every run is re-binned on ONE common set of bin edges and turned into a
  density (K+ per frame per Å), so runs of different length weigh the same.
- Each run is one independent sample. The curve is the mean over runs, the
  band its 95 % bootstrap CI (runs resampled with replacement), and every
  run is drawn as a thin grey line so a single run dominating is visible.
- Subunit comparison uses per-run numbers only (never pooled frames, which
  are autocorrelated): a Friedman test across subunits, runs as blocks,
  then pairwise Wilcoxon signed-rank tests, Holm-corrected. With fewer
  than ~6 runs these tests have little power; read the per-run table.

What it produces   <output_dir>  (default <base_dir>/cylinder_analysis_across_runs)
-------------------------------------------------------------------------------------
1. cylinder_per_run_<CH>.csv           one row per run and subunit
2. cylinder_density_<CH>.csv           mean, SEM, 95 % CI per bin and subunit
3. cylinder_visits_<CH>.csv            every visit of every run
4. cylinder_across_runs_<CH>.txt       across-run summary + subunit tests
5. cylinder_across_runs_<CH>.pkl       all of the above
6. cylinder_density_panels_<CH>.png    per subunit: runs (grey) + mean ± CI
7. cylinder_density_overlay_<CH>.png   all subunits, ESP-profile axis
8. cylinder_density_fractional_<CH>.png  same along s = t / L
9. cylinder_metrics_<CH>.png           occupancy / <n> / median t per run

How to run
----------
    python aggregate_cylinder_analysis.py \\
        --base_dir /path/to/girk_analyser_results/G12 --channel_type G12

    # only some runs, and label subunit D as G1 (as in the ESP plots)
    python aggregate_cylinder_analysis.py --base_dir .../G12 --channel_type G12 \\
        --runs 1-5,9 --rename D:G1
"""

import argparse
import csv
import pickle
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from analysis.cylinder_analysis import (
    SUBUNIT_COLORS, LABEL_SIZE, TICK_SIZE, TITLE_SIZE,
    axial_edges, fractional_edges, density, describe,
    plot_density_panels, plot_density_overlay,
)


PKL_REL = Path("cylinder_analysis") / "cylinder_analysis.pkl"
METRICS = ("occupancy", "mean_ions", "median_t", "peak_t")


def _run_number(path):
    m = re.search(r"RUN(\d+)", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def _parse_runs(spec):
    """'1-5,9' or '3 5 14' -> {1,2,3,4,5,9}"""
    out = set()
    for tok in re.split(r"[,\s]+", spec.strip()):
        tok = re.sub(r"(?i)^run", "", tok)
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return out


def _parse_rename(items):
    """['D:G1', 'A:2a'] -> {'D': 'G1', 'A': '2a'}"""
    out = {}
    for it in items or []:
        old, new = it.split(":", 1)
        out[old] = new
    return out


def _renamed(d, rename):
    """Applies --rename to every subunit-keyed dict of one run."""
    if not rename:
        return d
    d = dict(d)
    d["names"] = [rename.get(n, n) for n in d["names"]]
    for key in ("pairs", "hits", "n_in", "lengths", "visits", "stats", "density"):
        d[key] = {rename.get(n, n): v for n, v in d[key].items()}
    return d


def load_runs(base_dir, runs=None, rename=None):
    run_dirs = sorted((p for p in Path(base_dir).glob("RUN*") if p.is_dir()),
                      key=_run_number)
    if runs:
        run_dirs = [p for p in run_dirs if _run_number(p) in runs]

    loaded = {}
    for rd in run_dirs:
        pkl = rd / PKL_REL
        if not pkl.exists():
            print(f"  [SKIP] {rd.name}: {PKL_REL} not found")
            continue
        with open(pkl, "rb") as f:
            loaded[rd.name] = _renamed(pickle.load(f), rename)
        print(f"  [OK]   {rd.name}")
    return loaded


def bootstrap_ci(mat, n_boot=2000, seed=0):
    """95 % CI of the across-run mean, resampling runs (rows) with replacement."""
    if mat.shape[0] < 2:
        return mat.mean(axis=0), mat.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, mat.shape[0], size=(n_boot, mat.shape[0]))
    boots = mat[idx].mean(axis=1)
    return np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as given."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def subunit_tests(per_run, names, metric):
    """
    Friedman across subunits + pairwise Wilcoxon (Holm), runs as blocks.
    Runs with a NaN for any subunit (e.g. no K+ ever entered) are dropped.
    """
    mat = np.array([[per_run[r][n][metric] for n in names] for r in per_run], dtype=float)
    mat = mat[~np.isnan(mat).any(axis=1)]
    lines = [f"  {metric}: {mat.shape[0]} runs with data for every subunit"]
    if mat.shape[0] < 3 or len(names) < 2:
        lines.append("    too few runs for a test")
        return lines
    if np.allclose(mat, mat[:, :1]):
        lines.append("    identical for every subunit in every run, no test")
        return lines
    if len(names) >= 3:
        try:
            stat, p = sps.friedmanchisquare(*mat.T)
            lines.append(f"    Friedman chi2 = {stat:.3f}, p = {p:.4g}")
        except ValueError as e:
            lines.append(f"    Friedman not computed ({e})")
    pairs = list(combinations(range(len(names)), 2))
    raw = []
    for i, j in pairs:
        diff = mat[:, i] - mat[:, j]
        if np.allclose(diff, 0):
            raw.append(1.0)
            continue
        raw.append(sps.wilcoxon(mat[:, i], mat[:, j]).pvalue)
    for (i, j), p, pa in zip(pairs, raw, holm(raw)):
        md = np.median(mat[:, i] - mat[:, j])
        lines.append(f"    {names[i]:>4} vs {names[j]:<4} median diff {md:+8.3f}   "
                     f"Wilcoxon p = {p:.4g}   Holm p = {pa:.4g}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True,
                    help="folder holding RUN1, RUN2, ... of one channel")
    ap.add_argument("--channel_type", default="", help="used in file names and titles")
    ap.add_argument("--runs", default=None, help="e.g. '1-5,9'; default: every RUN*")
    ap.add_argument("--output_dir", default=None,
                    help="default <base_dir>/cylinder_analysis_across_runs")
    ap.add_argument("--rename", nargs="+", default=None,
                    help="relabel subunits, e.g. D:G1")
    ap.add_argument("--bin_width", type=float, default=None,
                    help="axial bin width, Å (default: the runs' own)")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    runs = _parse_runs(args.runs) if args.runs else None
    data = load_runs(args.base_dir, runs, _parse_rename(args.rename))
    if not data:
        raise SystemExit("No runs loaded.")

    out_dir = Path(args.output_dir) if args.output_dir \
        else Path(args.base_dir) / "cylinder_analysis_across_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = args.channel_type or "channel"
    title = f"{args.channel_type}  |  " if args.channel_type else ""

    first = next(iter(data.values()))
    names = first["names"]
    for r, d in data.items():
        if d["names"] != names:
            raise SystemExit(f"{r}: subunits {d['names']} differ from {names}")
        for key in ("radius", "extend"):
            if d[key] != first[key]:
                print(f"  [WARN] {r}: {key} = {d[key]}, first run has {first[key]}")
    radius, extend = first["radius"], first["extend"]
    bin_width = args.bin_width or first["bin_width"]

    # ── common bins ────────────────────────────────────────────────────────────
    max_L = max(d["lengths"][n].max() for d in data.values() for n in names)
    min_L = min(d["lengths"][n].min() for d in data.values() for n in names)
    t_edges = axial_edges(extend, max_L, bin_width)
    s_edges = fractional_edges(extend, min_L)
    t_centres = 0.5 * (t_edges[:-1] + t_edges[1:])

    t_mat = {n: np.array([density(d["hits"][n]["t"], d["n_frames"], t_edges)
                          for d in data.values()]) for n in names}
    s_mat = {n: np.array([density(d["hits"][n]["s"], d["n_frames"], s_edges)
                          for d in data.values()]) for n in names}

    def summarise(mat):
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0]) if mat.shape[0] > 1 \
            else np.zeros_like(mean)
        lo, hi = bootstrap_ci(mat, args.n_boot)
        return dict(mean=mean, sem=sem, ci_lo=lo, ci_hi=hi)

    t_sum = {n: summarise(t_mat[n]) for n in names}
    s_sum = {n: summarise(s_mat[n]) for n in names}

    # ── per-run metrics ────────────────────────────────────────────────────────
    per_run = {}
    for r, d in data.items():
        per_run[r] = {}
        for k, n in enumerate(names):
            st = d["stats"][n]
            dens = t_mat[n][list(data).index(r)]
            per_run[r][n] = dict(
                n_frames=d["n_frames"], mean_L=st["mean_L"],
                occupancy=st["occupancy"], mean_ions=st["mean_ions"],
                median_t=st["t"]["median"], mean_t=st["t"]["mean"],
                peak_t=float(t_centres[np.argmax(dens)]) if dens.sum() > 0 else np.nan,
                median_r=st["r"]["median"], n_visits=st["n_visits"],
                median_visit_ns=st["visit_ns"]["median"],
            )

    with open(out_dir / f"cylinder_per_run_{ch}.csv", "w", newline="") as f:
        cols = ["n_frames", "mean_L", "occupancy", "mean_ions", "median_t", "mean_t",
                "peak_t", "median_r", "n_visits", "median_visit_ns"]
        w = csv.writer(f)
        w.writerow(["run", "subunit"] + cols)
        for r in per_run:
            for n in names:
                w.writerow([r, n] + [per_run[r][n][c] for c in cols])

    with open(out_dir / f"cylinder_density_{ch}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_centre_A"] + [f"{c}_{n}" for n in names
                                     for c in ("mean", "sem", "ci_lo", "ci_hi")])
        for b, c in enumerate(t_centres):
            w.writerow([round(c, 4)] + [round(t_sum[n][k][b], 6) for n in names
                                        for k in ("mean", "sem", "ci_lo", "ci_hi")])

    with open(out_dir / f"cylinder_visits_{ch}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "subunit", "ion_resid", "start", "end", "n_frames",
                    "duration_ns", "mean_t_A", "mean_s", "censored"])
        for r, d in data.items():
            for n in names:
                for v in d["visits"][n]:
                    w.writerow([r, n, v["ion_resid"], v["start"], v["end"], v["n_frames"],
                                round(v["n_frames"] * d["frame_duration"], 4),
                                round(v["mean_t"], 4), round(v["mean_s"], 5), v["censored"]])

    # ── text summary ───────────────────────────────────────────────────────────
    lines = [
        f"CYLINDER ANALYSIS ACROSS RUNS — {args.channel_type}",
        f"Runs        : {len(data)}  ({', '.join(data)})",
        f"Cylinder    : r = {radius} Å, extend = {extend} Å, bin = {bin_width} Å",
        "Per-run values, mean ± SEM over runs (each run = one sample)",
        "",
        f"{'subunit':<8}" + "".join(f"{m:>22}" for m in METRICS),
        "-" * (8 + 22 * len(METRICS)),
    ]
    for n in names:
        row = f"{n:<8}"
        for m in METRICS:
            v = np.array([per_run[r][n][m] for r in per_run], dtype=float)
            v = v[~np.isnan(v)]
            sem = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
            row += f"{(f'{v.mean():.3f} ± {sem:.3f}' if v.size else 'n/a'):>22}"
        lines.append(row)
    lines += ["", "occupancy = fraction of frames with >= 1 K+ in the cylinder",
              "mean_ions = mean K+ in the cylinder per frame",
              "median_t / peak_t = Å from the GLU CA", "",
              "SUBUNIT COMPARISON (per-run values, runs as blocks)"]
    for m in ("mean_ions", "median_t"):
        lines += subunit_tests(per_run, names, m)
    vis = {n: describe([v["mean_t"] for d in data.values() for v in d["visits"][n]])
           for n in names}
    lines += ["", "VISITS (one mean position per visit, all runs pooled)",
              f"{'subunit':<8}{'visits':>8}{'median t':>10}{'IQR t':>18}"]
    for n in names:
        q = vis[n]
        lines.append(f"{n:<8}{q['n']:>8}{q['median']:>10.2f}"
                     f"{'[' + format(q['q1'], '.2f') + ', ' + format(q['q3'], '.2f') + ']':>18}")
    summary = "\n".join(lines)
    print("\n" + summary)
    (out_dir / f"cylinder_across_runs_{ch}.txt").write_text(summary + "\n")

    with open(out_dir / f"cylinder_across_runs_{ch}.pkl", "wb") as f:
        pickle.dump(dict(names=names, runs=list(data), radius=radius, extend=extend,
                         t_edges=t_edges, s_edges=s_edges, t_per_run=t_mat,
                         s_per_run=s_mat, t_summary=t_sum, s_summary=s_sum,
                         per_run=per_run), f)

    # ── plots ──────────────────────────────────────────────────────────────────
    mean_L = {n: float(np.mean([per_run[r][n]["mean_L"] for r in per_run])) for n in names}
    labels = {n: f"{n}  (<n> = {t_sum[n]['mean'].sum() * bin_width:.2f})" for n in names}

    def ci_band(summ):
        # plotting helpers draw mean ± band; use the wider side of the CI
        return {n: np.maximum(summ[n]["mean"] - summ[n]["ci_lo"],
                              summ[n]["ci_hi"] - summ[n]["mean"]) for n in names}

    plot_density_panels(t_edges, {n: t_sum[n]["mean"] for n in names}, ci_band(t_sum),
                        mean_L, out_dir / f"cylinder_density_panels_{ch}.png",
                        f"{title}K+ density, {len(data)} runs (grey), mean ± 95 % CI "
                        f"(r = {radius} Å)",
                        per_run=t_mat, band_label="95 % CI")
    plot_density_overlay(t_edges, {n: t_sum[n]["mean"] for n in names}, ci_band(t_sum),
                         mean_L, labels, out_dir / f"cylinder_density_overlay_{ch}.png",
                         f"{title}K+ axial density, mean of {len(data)} runs ± 95 % CI")
    plot_density_overlay(s_edges, {n: s_sum[n]["mean"] for n in names}, ci_band(s_sum),
                         {n: 1.0 for n in names}, labels,
                         out_dir / f"cylinder_density_fractional_{ch}.png",
                         f"{title}K+ density along s = t / L, {len(data)} runs",
                         xlabel="fractional position s  (0 = Glu CA, 1 = Asn CA)",
                         ylabel="K+ per frame per unit s")
    _plot_metrics(per_run, names, out_dir / f"cylinder_metrics_{ch}.png", title)
    print(f"\n  Saved to {out_dir}")


def _plot_metrics(per_run, names, out_path, title):
    """Bar = mean over runs, dots = individual runs."""
    metrics = (("occupancy", "occupancy"), ("mean_ions", "<n> K+ per frame"),
               ("median_t", "median t (Å from Glu CA)"))
    colors = SUBUNIT_COLORS[:len(names)] if len(names) <= 4 \
        else [plt.cm.tab10(i % 10) for i in range(len(names))]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 5))
    rng = np.random.default_rng(1)
    x = np.arange(len(names))
    for ax, (m, lab) in zip(axes, metrics):
        for i, (n, col) in enumerate(zip(names, colors)):
            v = np.array([per_run[r][n][m] for r in per_run], dtype=float)
            v = v[~np.isnan(v)]
            if v.size:
                ax.bar(i, v.mean(), color=col, alpha=0.5, width=0.6)
                ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, color=col,
                           edgecolor="black", s=30, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=TICK_SIZE)
        ax.set_ylabel(lab, fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{title}per-run cylinder metrics (bar = mean, dots = runs)",
                 fontsize=TITLE_SIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
