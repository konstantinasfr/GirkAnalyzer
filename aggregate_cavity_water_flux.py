#!/usr/bin/env python3
"""
Aggregate Cavity Water Flux & Residence
=======================================
Loads cavity_water_flux.pkl from every RUN* folder of one channel and
combines the water crossings and the water residence times.

Input per run
-------------
    <base_dir>/RUN<n>/water_analysis/cavity_water_flux/cavity_water_flux.pkl
(written by analysis/cavity_water_flux.py)

How runs are combined
---------------------
- Fluxes: each run's crossings are turned into a rate (per ns) first, so
  runs of different length weigh the same. Mean ± std is over runs.
- Residence: every complete visit from every run is pooled for the overall
  histogram / boxplot / mean / median. The per-run mean and median are also
  reported, and the per-run boxplot shows whether one run dominates the pool.
  Censored visits (still inside at a run's last frame) are left out, as in
  the per-run analysis.

Do not mix channels with different frame spacing (G12 0.075 ns vs G2
0.4 ns) in one call: the script warns if the runs disagree.

What it produces   <output_dir>  (default <base_dir>/water_analysis/cavity_water_flux_across_runs)
-------------------------------------------------------------------------------------------------
1. cavity_water_flux_per_run_<CH>.csv     one row per run: crossings, rates, residence stats
2. cavity_water_flux_across_runs_<CH>.txt across-run flux mean ± std, pooled residence
3. cavity_water_flux_across_runs_<CH>.pkl all of the above
4. flux_rates_<CH>.png                    forward / back / net per ns, bar = mean, dots = runs
5. flux_net_per_run_<CH>.png              net flux per run at each boundary
6. residence_hist_<CH>.png                pooled histogram per region
7. residence_box_<CH>.png                 pooled boxplot per region
8. residence_box_per_run_<CH>.png         one panel per region, one box per run

How to run
----------
    python aggregate_cavity_water_flux.py \\
        --base_dir /path/to/girk_analyser_results/G12 --channel_type G12

    # only some runs
    python aggregate_cavity_water_flux.py --base_dir .../G12 --channel_type G12 \\
        --runs 1-5,9
"""

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.cavity_water_flux import (
    BOUNDARIES, CROSSINGS, RES_REGIONS, RES_COLORS, describe,
    _plot_residence_hist, _plot_residence_box,
)


PKL_REL = Path("water_analysis") / "cavity_water_flux" / "cavity_water_flux.pkl"


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


def load_runs(base_dir, runs=None):
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
            loaded[rd.name] = pickle.load(f)
        print(f"  [OK]   {rd.name}")
    return loaded


def run_row(name, d):
    fd = d["frame_duration"]
    sim_ns = (len(d["frame_indices"]) - 1) * fd
    row = {"run": name, "sim_ns": round(sim_ns, 3), "frame_duration": fd,
           "n_permeations": d["n_permeation_events"]}
    for k in CROSSINGS:
        row[k] = d["totals"][k]
        row[f"{k}_per_ns"] = d["totals"][k] / sim_ns if sim_ns > 0 else np.nan
    for b, (f, k) in BOUNDARIES.items():
        row[f"net_{_slug(b)}_per_ns"] = row[f"{f}_per_ns"] - row[f"{k}_per_ns"]
    for r in RES_REGIONS:
        st = d["residence_stats"].get(r)
        if st is None:  # run analysed before this region existed
            continue
        row[f"res_{_slug(r)}_n"] = st["n"]
        row[f"res_{_slug(r)}_censored"] = st["n_censored"]
        row[f"res_{_slug(r)}_mean_ns"] = st["mean"]
        row[f"res_{_slug(r)}_median_ns"] = st["median"]
    return row


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def complete_times(d, region):
    v = d["residence_visits"].get(region)
    if v is None:
        return np.zeros(0)
    return v["n_frames"][~v["censored"]] * d["frame_duration"]


def run_aggregate_flux(base_dir, output_dir=None, channel_type="G12", runs=None):
    base_dir = Path(base_dir)
    out_dir = (Path(output_dir) if output_dir
               else base_dir / "water_analysis" / "cavity_water_flux_across_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = channel_type

    print("\n" + "=" * 70)
    print(f"AGGREGATE CAVITY WATER FLUX & RESIDENCE — {ch}")
    print("=" * 70)

    data = load_runs(base_dir, runs)
    if not data:
        print("  [ERROR] no runs loaded")
        return None

    fds = sorted({d["frame_duration"] for d in data.values()})
    if len(fds) > 1:
        print(f"  [WARN] runs have different frame spacing {fds} ns — "
              f"residence times and rates are not comparable across them")
    fd = fds[0]

    rows = [run_row(n, d) for n, d in data.items()]
    names = [r["run"] for r in rows]

    # ── per-run csv ───────────────────────────────────────────────────────────
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    csv_path = out_dir / f"cavity_water_flux_per_run_{ch}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ── across-run flux ───────────────────────────────────────────────────────
    flux = {}
    for k in [f"{c}_per_ns" for c in CROSSINGS] + \
             [f"net_{_slug(b)}_per_ns" for b in BOUNDARIES]:
        vals = np.array([r[k] for r in rows], dtype=float)
        flux[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                   "median": float(np.nanmedian(vals)), "values": vals}

    # ── pooled residence ──────────────────────────────────────────────────────
    pooled = {r: np.concatenate([complete_times(d, r) for d in data.values()])
              for r in RES_REGIONS}
    pooled_stats = {r: describe(t) for r, t in pooled.items()}
    for r in RES_REGIONS:
        pooled_stats[r]["n_censored"] = int(sum(
            d["residence_visits"][r]["censored"].sum()
            for d in data.values() if r in d["residence_visits"]))
        run_means = np.array([describe(complete_times(d, r))["mean"] for d in data.values()])
        run_medians = np.array([describe(complete_times(d, r))["median"] for d in data.values()])
        pooled_stats[r]["mean_of_run_means"] = float(np.nanmean(run_means))
        pooled_stats[r]["std_of_run_means"] = float(np.nanstd(run_means))
        pooled_stats[r]["mean_of_run_medians"] = float(np.nanmean(run_medians))

    # ── summary txt ───────────────────────────────────────────────────────────
    lines = [
        f"CAVITY WATER FLUX & RESIDENCE ACROSS RUNS — {ch}",
        f"Runs ({len(rows)}): {', '.join(names)}",
        f"Frame spacing: {fds} ns",
        f"Total simulated: {sum(r['sim_ns'] for r in rows):.1f} ns, "
        f"K+ permeations: {sum(r['n_permeations'] for r in rows)}",
        "",
        "FLUX (per ns, statistics over runs)",
        "=" * 60,
        f"{'':<22} {'mean':>9} {'std':>9} {'median':>9}",
        "-" * 60,
    ]
    for k, st in flux.items():
        lines.append(f"{k:<22} {st['mean']:>9.3f} {st['std']:>9.3f} {st['median']:>9.3f}")
    lines += [
        "",
        "RESIDENCE (ns, all complete visits of all runs pooled)",
        "=" * 96,
        f"{'region':<9} {'n':>8} {'cens.':>7} {'mean':>8} {'median':>8} {'std':>8} "
        f"{'q1':>7} {'q3':>7} {'max':>8} {'run-mean ± std':>17}",
        "-" * 96,
    ]
    for r, st in pooled_stats.items():
        lines.append(
            f"{r:<9} {st['n']:>8} {st['n_censored']:>7} {st['mean']:>8.3f} "
            f"{st['median']:>8.3f} {st['std']:>8.3f} {st['q1']:>7.3f} {st['q3']:>7.3f} "
            f"{st['max']:>8.3f} {st['mean_of_run_means']:>9.3f} ± {st['std_of_run_means']:<6.3f}")
    print("\n".join("  " + s for s in lines))
    (out_dir / f"cavity_water_flux_across_runs_{ch}.txt").write_text("\n".join(lines) + "\n")

    result = {"channel_type": ch, "runs": names, "per_run": rows, "flux": flux,
              "residence_pooled": pooled, "residence_stats": pooled_stats,
              "frame_durations": fds}
    with open(out_dir / f"cavity_water_flux_across_runs_{ch}.pkl", "wb") as f:
        pickle.dump(result, f)

    # ── plots ─────────────────────────────────────────────────────────────────
    _plot_rates_across(rows, out_dir / f"flux_rates_{ch}.png",
                       f"{ch}: water crossings per boundary ({len(rows)} runs)")
    _plot_net_per_run(rows, out_dir / f"flux_net_per_run_{ch}.png",
                      f"{ch}: net water flux per run (downward +)")
    _plot_residence_hist(pooled, pooled_stats, fd, out_dir / f"residence_hist_{ch}.png",
                         title=f"{ch}: water residence, all runs pooled")
    _plot_residence_box(pooled, pooled_stats, out_dir / f"residence_box_{ch}.png",
                        title=f"{ch}: water residence, all runs pooled (◇ = mean)")
    _plot_box_per_run(data, out_dir / f"residence_box_per_run_{ch}.png",
                      f"{ch}: water residence per run (◇ = mean)")

    print(f"  ✅ All outputs in: {out_dir}")
    return result


# ── plotting helpers ───────────────────────────────────────────────────────────

def _plot_rates_across(rows, out_path, title):
    names = list(BOUNDARIES)
    x = np.arange(len(names))
    wdt = 0.27
    series = (
        ("forward (down)", "#1E88E5", [f"{f}_per_ns" for f, _ in BOUNDARIES.values()]),
        ("backward (up)", "#90CAF9", [f"{k}_per_ns" for _, k in BOUNDARIES.values()]),
        ("net", "#212121", [f"net_{_slug(b)}_per_ns" for b in BOUNDARIES]),
    )
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (label, color, keys), off in zip(series, (-wdt, 0, wdt)):
        vals = np.array([[r[k] for r in rows] for k in keys], dtype=float)
        mean, std = np.nanmean(vals, axis=1), np.nanstd(vals, axis=1)
        bars = ax.bar(x + off, mean, wdt, yerr=std, capsize=3, color=color,
                      alpha=0.8, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        for i in range(len(keys)):
            jit = rng.uniform(-wdt / 4, wdt / 4, vals.shape[1])
            ax.scatter(x[i] + off + jit, vals[i], s=12, color="white",
                       edgecolor="black", linewidth=0.6, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, names, fontsize=11)
    ax.set_ylabel("waters per ns", fontsize=11)
    ax.set_title(f"{title}\nbar = mean ± std over runs, dots = runs",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_net_per_run(rows, out_path, title):
    x = np.arange(len(rows))
    wdt = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(rows) + 3), 5))
    for (b, color), off in zip(zip(BOUNDARIES, ("#1E88E5", "#FB8C00", "#43A047")),
                               (-wdt, 0, wdt)):
        ax.bar(x + off, [r[f"net_{_slug(b)}_per_ns"] for r in rows], wdt,
               color=color, label=b)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{r['run']}\n({r['n_permeations']} K⁺)" for r in rows],
                  fontsize=9)
    ax.set_ylabel("net waters per ns", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


def _plot_box_per_run(data, out_path, title):
    regions = list(RES_REGIONS)
    names = list(data)
    fig, axes = plt.subplots(len(regions), 1, sharex=True,
                             figsize=(max(8, 0.6 * len(names) + 3), 3.2 * len(regions)))
    for ax, r in zip(np.atleast_1d(axes), regions):
        times = [complete_times(d, r) for d in data.values()]
        keep = [i for i, t in enumerate(times) if t.size]
        if keep:
            bp = ax.boxplot([times[i] for i in keep], positions=[i + 1 for i in keep],
                            showmeans=True, patch_artist=True,
                            flierprops={"markersize": 2, "alpha": 0.3},
                            meanprops={"marker": "D", "markerfacecolor": "white",
                                       "markeredgecolor": "black"})
            for patch in bp["boxes"]:
                patch.set_facecolor(RES_COLORS[r])
                patch.set_alpha(0.7)
            ax.set_yscale("log")
        ax.set_ylabel(f"{r}\nresidence (ns)", fontsize=10)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    axes[-1].set_xticks(range(1, len(names) + 1), names, fontsize=9, rotation=45)
    axes[0].set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_dir", required=True,
                        help="channel folder holding RUN* subfolders")
    parser.add_argument("--output_dir", default=None,
                        help="default: <base_dir>/water_analysis/cavity_water_flux_across_runs")
    parser.add_argument("--channel_type", default="G12")
    parser.add_argument("--runs", default=None,
                        help="subset of runs, e.g. '1-5,9' (default: all RUN* folders)")
    args = parser.parse_args()

    run_aggregate_flux(args.base_dir, args.output_dir, args.channel_type,
                       _parse_runs(args.runs) if args.runs else None)


'''
python3 aggregate_cavity_water_flux.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 --channel_type G12
python3 aggregate_cavity_water_flux.py --base_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 --channel_type G2
'''
