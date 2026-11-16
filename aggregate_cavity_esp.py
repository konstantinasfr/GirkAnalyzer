#!/usr/bin/env python3
"""
aggregate_cavity_esp.py
=======================
Pools the classical cylinder ESP (analysis/cavity_esp_classical.py, RUN*/cavity_esp/) over the runs of one
or more systems, after the equilibration cut, and compares the systems.

USAGE
  python3 aggregate_cavity_esp.py \\
      --system G12=/path/girk_analyser_results/G12 \\
      --system G2=/path/girk_analyser_results/G2 \\
      --system D173N=/path/girk_analyser_results/G12_D173N,/path/girk_analyser_results/G12_D173N_4fs \\
      --system E141Q=/path/girk_analyser_results/G12_E141Q,/path/girk_analyser_results/G12_E141Q_4fs \\
      [--out cavity_esp_summary] [--no_eq] [--boot 5000]
The first --system is the reference for the differences. The equilibration start of each run is read from
<system>/aggregated_site_chain/<settings>/equilibration.csv (used_start_frame) when it exists; frames of
interest (e.g. the QM frames) are always reported, whether before or after the cut.

OUTPUTS (in --out)
  frames_of_interest.csv   the exact values at the frames of interest (each cylinder + four-subunit mean)
  per_run.csv              per run and cylinder: frames, mean, median, SD over frames (all frames and
                           frames with all four upper-ring side chains pointing in)
  summary.csv              per system and cylinder: runs, mean of the run means, median, SD between runs,
                           SEM, bootstrap 95% CI over runs; pooled over frames: mean, median, SD; the mean
                           spatial SD inside the cylinder; the same for inward frames
  differences.csv          four-subunit mean of each system minus the reference: estimate, bootstrap 95% CI
                           over runs (all frames and inward frames)
  contributions.csv        per system and cylinder: the mean contribution of each ring residue and the caps
  profiles.csv             axial profile (distance from the Glu CA): mean over runs, SD between runs
  cavity_esp.txt, per_run.png, profiles.png, frames_hist.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOUR = "four-subunit mean"
COLORS = ["#1F4E79", "#D9784F", "#13896B", "#C0392B", "#6B6B6B", "#3B7DD8"]


def load_one(base, use_eq=True):
    base = Path(base)
    fr = [pd.read_csv(f) for f in sorted(base.glob("RUN*/cavity_esp/esp_frames_run*.csv"))]
    pr = [pd.read_csv(f) for f in sorted(base.glob("RUN*/cavity_esp/esp_profile_run*.csv.gz"))]
    if not fr:
        raise SystemExit(f"no RUN*/cavity_esp/esp_frames_run*.csv under {base}")
    frames = pd.concat(fr, ignore_index=True)
    prof = pd.concat(pr, ignore_index=True) if pr else pd.DataFrame()
    eq = {}
    if use_eq:
        for f in sorted(base.glob("aggregated_site_chain/*/equilibration.csv")):
            t = pd.read_csv(f)
            eq = dict(zip(t["run"].astype(int), t["used_start_frame"].fillna(0).astype(int)))
            print(f"  {base.name}: equilibration starts from {f}")
            break
        if not eq:
            print(f"  {base.name}: no equilibration.csv found, all frames used")
    frames["after_eq"] = frames["frame"] >= frames["run"].map(eq).fillna(0)
    if len(prof):
        prof = prof[prof["frame"] >= prof["run"].map(eq).fillna(0)]
    return frames, prof


def load_system(paths, use_eq=True):
    """One or more folders (comma-separated) pooled as one system; with several folders the runs are
    renamed <folder>:<run> so that RUN1 of one folder is not mixed with RUN1 of another."""
    paths = [p for p in paths.split(",") if p]
    F, P = [], []
    for p in paths:
        fr, pr = load_one(p, use_eq)
        if len(paths) > 1:
            tag = Path(p).name
            fr["run"] = tag + ":" + fr["run"].astype(str)
            if len(pr):
                pr["run"] = tag + ":" + pr["run"].astype(str)
        F.append(fr)
        if len(pr):
            P.append(pr)
    return pd.concat(F, ignore_index=True), (pd.concat(P, ignore_index=True) if P else pd.DataFrame()), None


def add_four(frames):
    """Append, per frame, a 'four-subunit mean' row (mean of the four cylinder means)."""
    keys = ["run", "frame", "time_ns", "frame_of_interest", "all_inward", "after_eq"]
    g = frames.groupby(keys, as_index=False).agg(mean=("mean", "mean"), median=("median", "mean"),
                                                 sd=("sd", "mean"), n_points=("n_points", "sum"))
    g["cylinder"] = FOUR
    return pd.concat([frames, g], ignore_index=True)


def boot_ci(values, n=5000, seed=1):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n, len(v))).mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_diff(a, b, n=5000, seed=1):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(axis=1) - rng.choice(b, (n, len(b))).mean(axis=1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, help="LABEL=PATH[,PATH2...] (folders holding RUN*/; several folders = one system, runs kept apart)")
    ap.add_argument("--out", default="cavity_esp_summary")
    ap.add_argument("--no_eq", action="store_true", help="use all frames (no equilibration cut)")
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)

    systems, allf, allp = [], [], []
    for s in a.system:
        lab, path = s.split("=", 1)
        fr, pr, eq = load_system(path, not a.no_eq)
        fr = add_four(fr)
        fr.insert(0, "system", lab)
        allf.append(fr)
        if len(pr):
            pr.insert(0, "system", lab)
            allp.append(pr)
        systems.append(lab)
    F = pd.concat(allf, ignore_index=True)
    P = pd.concat(allp, ignore_index=True) if allp else pd.DataFrame()
    lines = ["CLASSICAL CAVITY ESP (kcal/mol per +1e), force-field charges of the QM cluster atoms, "
             "regular mesh in the Glu -> Asn/Asp cylinders"]

    # 1. frames of interest
    foi = F[F["frame_of_interest"]].copy()
    foi[["system", "run", "frame", "cylinder", "mean", "median", "sd", "n_points", "all_inward", "after_eq"]].to_csv(
        od / "frames_of_interest.csv", index=False, float_format="%.2f")
    if len(foi):
        lines.append("\nFRAMES OF INTEREST (exact values on the mesh)")
        for (sy, run, fr_), g in foi.groupby(["system", "run", "frame"], sort=False):
            lines.append(f"  {sy} RUN{run} frame {fr_}  (all Glu pointing in: {bool(g['all_inward'].iloc[0])})")
            for r in g.to_dict("records"):
                lines.append(f"    {r['cylinder']:<20} mean {r['mean']:8.1f}  median {r['median']:8.1f}  "
                             f"SD in cylinder {r['sd']:6.1f}")

    # 2. per run
    use = F[F["after_eq"]]
    rows = []
    for (sy, run, cyl), g in use.groupby(["system", "run", "cylinder"], sort=False):
        gi = g[g["all_inward"]]
        rows.append(dict(system=sy, run=run, cylinder=cyl, n_frames=len(g), mean=g["mean"].mean(),
                         median=g["mean"].median(), sd_frames=g["mean"].std(ddof=1), spatial_sd=g["sd"].mean(),
                         n_frames_inward=len(gi), mean_inward=gi["mean"].mean() if len(gi) else np.nan,
                         frac_inward=len(gi) / len(g)))
    R = pd.DataFrame(rows)
    R.to_csv(od / "per_run.csv", index=False, float_format="%.3f")

    # 3. summary per system and cylinder
    rows = []
    for (sy, cyl), g in R.groupby(["system", "cylinder"], sort=False):
        fr_ = use[(use["system"] == sy) & (use["cylinder"] == cyl)]
        fi = fr_[fr_["all_inward"]]
        lo, hi = boot_ci(g["mean"], a.boot)
        gi = g["mean_inward"].dropna()
        loi, hii = boot_ci(gi, a.boot)
        rows.append(dict(system=sy, cylinder=cyl, n_runs=len(g), n_frames=len(fr_),
                         mean_of_runs=g["mean"].mean(), median_of_runs=g["mean"].median(),
                         sd_between_runs=g["mean"].std(ddof=1), sem=g["mean"].std(ddof=1) / np.sqrt(len(g)),
                         ci95_lo=lo, ci95_hi=hi,
                         pooled_mean=fr_["mean"].mean(), pooled_median=fr_["mean"].median(),
                         pooled_sd_frames=fr_["mean"].std(ddof=1), mean_spatial_sd=fr_["sd"].mean(),
                         frac_frames_inward=len(fi) / len(fr_) if len(fr_) else np.nan,
                         inward_mean_of_runs=gi.mean() if len(gi) else np.nan, inward_ci95_lo=loi,
                         inward_ci95_hi=hii, inward_pooled_median=fi["mean"].median() if len(fi) else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(od / "summary.csv", index=False, float_format="%.3f")
    lines.append("\nPER SYSTEM (after equilibration; runs are the unit)")
    for r in S.to_dict("records"):
        lines.append(f"  {r['system']:<7} {r['cylinder']:<20} runs {r['n_runs']:>2} frames {r['n_frames']:>6}  "
                     f"mean {r['mean_of_runs']:7.1f} [{r['ci95_lo']:.1f}, {r['ci95_hi']:.1f}]  median {r['median_of_runs']:7.1f}  "
                     f"SD runs {r['sd_between_runs']:5.1f}  SD frames {r['pooled_sd_frames']:5.1f}  "
                     f"inward frames {100 * r['frac_frames_inward']:4.0f}% -> {r['inward_mean_of_runs']:7.1f}")

    # 4. differences vs the reference (four-subunit mean)
    ref = systems[0]
    rows = []
    for sy in systems[1:]:
        for col, tag in (("mean", "all frames"), ("mean_inward", "inward frames")):
            a_ = R[(R["system"] == ref) & (R["cylinder"] == FOUR)][col].dropna()
            b_ = R[(R["system"] == sy) & (R["cylinder"] == FOUR)][col].dropna()
            lo, hi = boot_diff(b_, a_, a.boot)
            rows.append(dict(system=sy, reference=ref, frames=tag, difference=b_.mean() - a_.mean(),
                             ci95_lo=lo, ci95_hi=hi, n_runs=len(b_), n_runs_ref=len(a_)))
    D = pd.DataFrame(rows)
    D.to_csv(od / "differences.csv", index=False, float_format="%.3f")
    if len(D):
        lines.append(f"\nFOUR-SUBUNIT MEAN MINUS {ref} (bootstrap 95% CI over runs; positive = less negative than {ref})")
        for r in D.to_dict("records"):
            lines.append(f"  {r['system']:<7} {r['frames']:<14} {r['difference']:+7.1f} [{r['ci95_lo']:+.1f}, {r['ci95_hi']:+.1f}]")

    # 5. contributions
    cc = [c for c in use.columns if c.startswith("contrib_")]
    C = (use[use["cylinder"] != FOUR].groupby(["system", "run", "cylinder"])[cc].mean()
         .groupby(["system", "cylinder"]).mean().reset_index())
    C.to_csv(od / "contributions.csv", index=False, float_format="%.2f")
    lines.append("\nWHERE EACH CYLINDER'S POTENTIAL COMES FROM (mean contribution, kcal/mol per +1e)")
    for r in C.to_dict("records"):
        parts = sorted(((k[8:], v) for k, v in r.items() if k.startswith("contrib_") and v == v), key=lambda x: x[1])
        lines.append(f"  {r['system']:<7} {r['cylinder']:<20} " + ", ".join(f"{k} {v:.1f}" for k, v in parts[:5]))

    # 6. profiles
    if len(P):
        pr = (P.groupby(["system", "run", "cylinder", "t_A"])["mean"].mean().reset_index()
              .groupby(["system", "cylinder", "t_A"])["mean"].agg(["mean", "std", "count"]).reset_index())
        pr.to_csv(od / "profiles.csv", index=False, float_format="%.3f")
        cyls = {sy: list(dict.fromkeys(pr[pr["system"] == sy]["cylinder"])) for sy in systems}
        fig, axes = plt.subplots(1, len(systems), figsize=(5.2 * len(systems), 4.4), squeeze=False, sharey=True)
        for ax, sy in zip(axes[0], systems):
            for c, col in zip(cyls[sy], COLORS):
                g = pr[(pr["system"] == sy) & (pr["cylinder"] == c)]
                ax.plot(g["t_A"], g["mean"], color=col, lw=2, label=c)
                ax.fill_between(g["t_A"], g["mean"] - g["std"], g["mean"] + g["std"], color=col, alpha=0.15, lw=0)
            ax.axvline(0, color="#1B2631", ls="--", lw=1)
            ax.set_title(sy)
            ax.set_xlabel("distance from the Glu CA along the cylinder (Å)")
            ax.legend(frameon=False, fontsize=8)
        axes[0][0].set_ylabel("ESP (kcal/mol per +1e)")
        fig.tight_layout()
        fig.savefig(od / "profiles.png", dpi=180)
        plt.close(fig)

    # 7. per-run four-subunit means, and the frame distribution
    fig, ax = plt.subplots(figsize=(1.6 * len(systems) + 2.5, 4.6))
    for i, sy in enumerate(systems):
        g = R[(R["system"] == sy) & (R["cylinder"] == FOUR)]
        x = i + np.random.default_rng(i).uniform(-0.12, 0.12, len(g))
        ax.scatter(x, g["mean"], s=28, color=COLORS[i % len(COLORS)], alpha=0.8, edgecolors="white", lw=0.8, zorder=3)
        s = S[(S["system"] == sy) & (S["cylinder"] == FOUR)].iloc[0]
        ax.errorbar(i + 0.3, s["mean_of_runs"], yerr=[[s["mean_of_runs"] - s["ci95_lo"]], [s["ci95_hi"] - s["mean_of_runs"]]],
                    fmt="o", color="#1B2631", capsize=4, zorder=4)
    for _, r in foi[foi["cylinder"] == FOUR].iterrows():
        i = systems.index(r["system"])
        ax.scatter(i - 0.3, r["mean"], marker="*", s=120, color="#F4D03F", edgecolors="#1B2631", zorder=5)
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems)
    ax.set_ylabel("four-subunit mean ESP (kcal/mol per +1e)")
    ax.set_title("each dot = one run; black = mean ± 95% CI; star = frame of interest", fontsize=10)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(od / "per_run.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, sy in enumerate(systems):
        v = use[(use["system"] == sy) & (use["cylinder"] == FOUR)]["mean"]
        ax.hist(v, bins=50, histtype="step", lw=2, color=COLORS[i % len(COLORS)], density=True, label=f"{sy} (n={len(v)})")
    ax.set_xlabel("four-subunit mean ESP per frame (kcal/mol per +1e)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(od / "frames_hist.png", dpi=180)
    plt.close(fig)

    (od / "cavity_esp.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\ncavity_esp: results in {od}")


if __name__ == "__main__":
    main()
