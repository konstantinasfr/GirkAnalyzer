"""
combine_pip_headgroup_distances.py
====================================
Combines pip_headgroup_distances results across multiple RUN folders.
Reads per_frame.csv from each RUN/subunit, concatenates all frames,
makes combined histograms, boxplots, violin plots and summary statistics.

Usage
-----
    python combine_pip_headgroup_distances.py \
        --runs_dir     /path/to/G12 \
        --channel_type G12 \
        --output_dir   /path/to/output  (default: runs_dir/combined_pip_headgroup_distances)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

P_ATOMS   = ["P1", "P4", "P5"]
CLUSTERS  = ["Nterm", "Cterm", "LYS90", "LYS194"]
DIST_COLS = [f"{p}_{c}" for p in P_ATOMS for c in CLUSTERS]

CLUSTER_LABELS = {
    "Nterm":  "N-term\n(ARG60+LYS64)",
    "Cterm":  "C-term\n(LYS199+LYS200)",
    "LYS90":  "LYS90",
    "LYS194": "LYS194",
}
P_COLORS    = {"P1": "#4CAF50", "P4": "#FF9800", "P5": "#2196F3"}
CLUS_COLORS = {"Nterm": "#1565C0", "Cterm": "#B71C1C", "LYS90": "#7B1FA2", "LYS194": "#E91E63"}


def _subunit_order(channel_type: str) -> list[str]:
    if channel_type == "G2":
        return ["2a", "2b", "2c", "2d"]
    return ["2a", "2b", "2c", "G1"]


def _sorted_run_dirs(runs_dir: Path, exclude_runs: list[str] | None = None) -> list:
    exclude = {r.upper() for r in (exclude_runs or [])}
    return sorted(
        [d for d in runs_dir.iterdir()
         if d.is_dir() and d.name.upper().startswith("RUN") and d.name[3:].isdigit()
         and d.name.upper() not in exclude],
        key=lambda d: int(d.name[3:]),
    )


def _load_all_frames(runs_dir: Path, subunit: str,
                     exclude_runs: list[str] | None = None,
                     subset: str | None = None) -> pd.DataFrame:
    run_dirs  = _sorted_run_dirs(runs_dir, exclude_runs=exclude_runs)
    base_path = ("pip_headgroup_distances" + (f"/{subset}" if subset else "")
                 + f"/subunit_{subunit}")
    dfs = []
    for run_dir in run_dirs:
        csv = run_dir / base_path / "per_frame.csv"
        if not csv.exists():
            print(f"    MISSING: {csv}")
            continue
        df = pd.read_csv(csv)
        df["run"] = run_dir.name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    expected = len(dfs) * dfs[0]["frame"].nunique() if dfs else 0
    print(f"    Subunit {subunit}: {len(dfs)} runs, {len(combined)} total frames "
          f"(expected ~{expected})")
    return combined


def _make_summary(df: pd.DataFrame, n_runs: int) -> pd.DataFrame:
    rows = []
    for col in DIST_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        p_atom, cluster = col.split("_", 1)
        rows.append({
            "distance": col,
            "p_atom":   p_atom,
            "cluster":  cluster,
            "n_frames": len(vals),
            "n_runs":   n_runs,
            "mean":     round(float(vals.mean()), 3),
            "median":   round(float(np.median(vals)), 3),
            "std":      round(float(vals.std()), 3),
            "min":      round(float(vals.min()), 3),
            "max":      round(float(vals.max()), 3),
            "q25":      round(float(np.percentile(vals, 25)), 3),
            "q75":      round(float(np.percentile(vals, 75)), 3),
        })
    return pd.DataFrame(rows)


def _histograms(df: pd.DataFrame, out_dir: Path, channel_type: str,
                subunit: str, n_runs: int):
    from scipy.stats import gaussian_kde
    from sklearn.mixture import GaussianMixture

    hist_dir     = out_dir / "histograms"
    gaussian_dir = out_dir / "gaussians"
    hist_dir.mkdir(exist_ok=True)
    gaussian_dir.mkdir(exist_ok=True)

    gauss_colors     = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]
    BIC_PENALTY_FACTOR = 1.5
    MIN_WEIGHT         = 0.12
    MIN_SEP            = 0.8

    col_data = {}

    for col in DIST_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue
        p_atom, cluster = col.split("_", 1)
        color = P_COLORS.get(p_atom, "#9E9E9E")
        short_title = f"{p_atom} -> {CLUSTER_LABELS.get(cluster, cluster)}"
        full_title  = f"{short_title}\n{channel_type} | Subunit {subunit} [{n_runs} runs]"

        x_range  = np.linspace(vals.min(), vals.max(), 500)
        kde      = gaussian_kde(vals, bw_method=0.18)
        kde_vals = kde(x_range)

        gmm_data = None
        try:
            vals_2d = vals.reshape(-1, 1)
            best_gmm, best_bic = None, np.inf
            for nc in range(1, 5):
                g = GaussianMixture(n_components=nc, random_state=42,
                                    max_iter=1000, n_init=15, reg_covar=1e-4)
                g.fit(vals_2d)
                bic = g.bic(vals_2d) * BIC_PENALTY_FACTOR
                if bic < best_bic:
                    best_bic, best_gmm = bic, g

            order   = np.argsort(best_gmm.means_.flatten())
            means   = best_gmm.means_.flatten()[order]
            stds    = np.sqrt(best_gmm.covariances_.flatten())[order]
            weights = best_gmm.weights_[order]

            keep    = weights >= MIN_WEIGHT
            means, stds, weights = means[keep], stds[keep], weights[keep]
            weights /= weights.sum()
            n_comp  = len(means)

            merged_means, merged_stds, merged_weights = [], [], []
            used = [False] * n_comp
            for k in range(n_comp):
                if used[k]: continue
                gw = weights[k]; gmu = means[k]*weights[k]
                gvar = (stds[k]**2 + means[k]**2)*weights[k]
                used[k] = True
                for j in range(k+1, n_comp):
                    if not used[j] and abs(means[k]-means[j]) < MIN_SEP:
                        gw += weights[j]; gmu += means[j]*weights[j]
                        gvar += (stds[j]**2 + means[j]**2)*weights[j]
                        used[j] = True
                mu_m = gmu/gw
                merged_means.append(mu_m)
                merged_stds.append(np.sqrt(max(gvar/gw - mu_m**2, 1e-6)))
                merged_weights.append(gw)

            merged_means   = np.array(merged_means)
            merged_stds    = np.array(merged_stds)
            merged_weights = np.array(merged_weights)
            merged_weights /= merged_weights.sum()

            components_scaled = []
            kde_max = kde_vals.max()
            for k in range(len(merged_means)):
                c = (merged_weights[k] / (merged_stds[k]*np.sqrt(2*np.pi)) *
                     np.exp(-0.5*((x_range-merged_means[k])/merged_stds[k])**2))
                peak_idx = np.argmax(c)
                scale_k  = min(kde_vals[peak_idx], kde_max) / c[peak_idx] if c[peak_idx] > 0 else 1.0
                components_scaled.append((c * scale_k, merged_means[k],
                                          merged_stds[k], merged_weights[k]))
            gmm_data = components_scaled
        except Exception as e:
            print(f"    GMM fit failed for {col}: {e}")

        col_data[col] = dict(vals=vals, x_range=x_range, kde_vals=kde_vals,
                             color=color, full_title=full_title,
                             short_title=short_title, gmm=gmm_data)

        # individual histogram
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.fill_between(x_range, kde_vals, color=color, alpha=0.7)
        ax.plot(x_range, kde_vals, color=color, linewidth=1.5)
        ax.axvline(np.mean(vals), color="black", linestyle="--",
                   linewidth=1.2, label=f"mean={np.mean(vals):.2f}A")
        ax.axvline(np.median(vals), color="grey", linestyle=":",
                   linewidth=1.2, label=f"median={np.median(vals):.2f}A")
        ax.set_xlabel("Distance (A)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(full_title, fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        fig.savefig(hist_dir / f"hist_{col}.png", dpi=150)
        plt.close(fig)

        # individual gaussian
        if gmm_data is not None:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.fill_between(x_range, kde_vals, color=color, alpha=0.3, label="KDE")
            ax.plot(x_range, kde_vals, color=color, linewidth=1.5)
            for k, (comp, mu, sigma, w) in enumerate(gmm_data):
                ax.plot(x_range, comp,
                        color=gauss_colors[k % len(gauss_colors)],
                        linewidth=2, linestyle="--", alpha=0.9,
                        label=f"G{k+1}: mu={mu:.2f}A, s={sigma:.2f}A, w={w:.2f}")
            ax.set_xlabel("Distance (A)", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(f"GMM -- {full_title}", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            fig.savefig(gaussian_dir / f"gauss_{col}.png", dpi=150)
            plt.close(fig)

    # 3x3 grids
    fig_g, axes_g = plt.subplots(3, 4, figsize=(16, 9))
    fig_g.suptitle(f"KDE histograms -- {channel_type} | Subunit {subunit} [{n_runs} runs]",
                   fontsize=15)
    fig_k, axes_k = plt.subplots(3, 4, figsize=(16, 9))
    fig_k.suptitle(f"GMM fits -- {channel_type} | Subunit {subunit} [{n_runs} runs]",
                   fontsize=15)

    for i, p_atom in enumerate(P_ATOMS):
        for j, cluster in enumerate(CLUSTERS):
            col = f"{p_atom}_{cluster}"
            ax_g = axes_g[i][j]
            ax_k = axes_k[i][j]
            if col not in col_data:
                ax_g.set_visible(False); ax_k.set_visible(False)
                continue
            d = col_data[col]
            color, x_range, kde_vals = d["color"], d["x_range"], d["kde_vals"]
            vals = d["vals"]
            stitle = d["short_title"]

            ax_g.fill_between(x_range, kde_vals, color=color, alpha=0.7)
            ax_g.plot(x_range, kde_vals, color=color, linewidth=1.2)
            ax_g.axvline(np.mean(vals), color="black", linestyle="--", linewidth=1)
            ax_g.axvline(np.median(vals), color="grey", linestyle=":", linewidth=1)
            ax_g.set_title(stitle, fontsize=11)
            ax_g.set_xlabel("Distance (A)", fontsize=10)
            ax_g.set_ylabel("Density", fontsize=10)
            ax_g.tick_params(labelsize=9)

            ax_k.fill_between(x_range, kde_vals, color=color, alpha=0.3)
            ax_k.plot(x_range, kde_vals, color=color, linewidth=1.2)
            if d["gmm"] is not None:
                for k, (comp, mu, sigma, w) in enumerate(d["gmm"]):
                    ax_k.plot(x_range, comp,
                              color=gauss_colors[k % len(gauss_colors)],
                              linewidth=1.5, linestyle="--", alpha=0.9,
                              label=f"G{k+1}: {mu:.1f}A, w={w:.2f}")
                ax_k.legend(fontsize=8, loc="upper right")
            ax_k.set_title(stitle, fontsize=11)
            ax_k.set_xlabel("Distance (A)", fontsize=10)
            ax_k.set_ylabel("Density", fontsize=10)
            ax_k.tick_params(labelsize=9)

    fig_g.tight_layout()
    fig_g.savefig(hist_dir / "hist_all_3x3.png", dpi=150)
    plt.close(fig_g)

    fig_k.tight_layout()
    fig_k.savefig(gaussian_dir / "gauss_all_3x3.png", dpi=150)
    plt.close(fig_k)

    # paired overlay plots
    overlay_dir = out_dir / "gaussians" / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    clus_colors = {
        "Nterm":  "#1565C0",
        "Cterm":  "#B71C1C",
        "LYS90":  "#7B1FA2",
        "LYS194": "#E91E63",
    }

    pairs = [
        ("Nterm",  "Cterm",  "Nterm vs Cterm"),
        ("Nterm",  "LYS90",  "Nterm vs LYS90"),
        ("Cterm",  "LYS194", "Cterm vs LYS194"),
    ]

    for p_atom in P_ATOMS:
        for clus_a, clus_b, pair_label in pairs:
            col_a = f"{p_atom}_{clus_a}"
            col_b = f"{p_atom}_{clus_b}"
            if col_a not in col_data or col_b not in col_data:
                continue

            da = col_data[col_a]
            db = col_data[col_b]

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.fill_between(da["x_range"], da["kde_vals"],
                            color=clus_colors[clus_a], alpha=0.15)
            ax.fill_between(db["x_range"], db["kde_vals"],
                            color=clus_colors[clus_b], alpha=0.15)
            ax.plot(da["x_range"], da["kde_vals"],
                    color=clus_colors[clus_a], linewidth=1.5,
                    label=f"{clus_a} KDE")
            ax.plot(db["x_range"], db["kde_vals"],
                    color=clus_colors[clus_b], linewidth=1.5,
                    label=f"{clus_b} KDE")

            if da["gmm"]:
                for k, (comp, mu, sigma, w) in enumerate(da["gmm"]):
                    ax.plot(da["x_range"], comp,
                            color=clus_colors[clus_a],
                            linewidth=2, linestyle="--", alpha=0.9,
                            label=f"{clus_a} G{k+1}: {mu:.1f}A w={w:.2f}")
            if db["gmm"]:
                for k, (comp, mu, sigma, w) in enumerate(db["gmm"]):
                    ax.plot(db["x_range"], comp,
                            color=clus_colors[clus_b],
                            linewidth=2, linestyle=":", alpha=0.9,
                            label=f"{clus_b} G{k+1}: {mu:.1f}A w={w:.2f}")

            ax.set_xlabel("Distance (A)", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(f"{p_atom} | {pair_label} | {channel_type} | Subunit {subunit} [{n_runs} runs]",
                         fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.tick_params(labelsize=10)
            plt.tight_layout()
            fname = f"overlay_{p_atom}_{clus_a}_vs_{clus_b}.png"
            fig.savefig(overlay_dir / fname, dpi=150)
            plt.close(fig)

    print(f"  Overlay plots -> {overlay_dir}")


def _residue_gaussians(runs_dir: Path, out_dir: Path, channel_type: str,
                        subunit: str, n_runs: int,
                        exclude_runs: list | None = None):
    from scipy.stats import gaussian_kde
    from sklearn.mixture import GaussianMixture

    run_dirs = _sorted_run_dirs(runs_dir, exclude_runs=exclude_runs)
    dfs = []
    for run_dir in run_dirs:
        csv = run_dir / "pip_pose" / f"per_frame_{subunit}.csv"
        if not csv.exists():
            continue
        dfs.append(pd.read_csv(csv))
    if not dfs:
        print(f"    No pip_pose per_frame found for subunit {subunit}")
        return
    df = pd.concat(dfs, ignore_index=True)

    resid_dir = out_dir / "gaussians" / "per_residue"
    resid_dir.mkdir(exist_ok=True)

    residue_colors = {
        "ARG60":  "#1565C0",
        "LYS64":  "#00ACC1",
        "LYS199": "#E53935",
        "LYS200": "#FB8C00",
    }

    BIC_PENALTY_FACTOR = 1.5
    MIN_WEIGHT         = 0.12
    MIN_SEP            = 0.8
    gauss_colors       = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]

    gmm_store = {}

    for p_atom in P_ATOMS:
        for res_name, color in residue_colors.items():
            col = None
            for candidate in [
                f"p{p_atom[1]}_d_{res_name}.{subunit}",
                f"p{p_atom[1]}_d_{res_name}",
                f"{p_atom.lower()}_d_{res_name}.{subunit}",
                f"{p_atom.lower()}_d_{res_name}",
            ]:
                if candidate in df.columns:
                    col = candidate
                    break
            if col is None:
                continue

            vals = df[col].dropna().values
            if len(vals) == 0:
                continue

            x_range  = np.linspace(vals.min(), vals.max(), 500)
            kde      = gaussian_kde(vals, bw_method=0.18)
            kde_vals = kde(x_range)

            gmm_data = None
            try:
                vals_2d = vals.reshape(-1, 1)
                best_gmm, best_bic = None, np.inf
                for nc in range(1, 5):
                    g = GaussianMixture(n_components=nc, random_state=42,
                                        max_iter=1000, n_init=15, reg_covar=1e-4)
                    g.fit(vals_2d)
                    bic = g.bic(vals_2d) * BIC_PENALTY_FACTOR
                    if bic < best_bic:
                        best_bic, best_gmm = bic, g

                order   = np.argsort(best_gmm.means_.flatten())
                means   = best_gmm.means_.flatten()[order]
                stds    = np.sqrt(best_gmm.covariances_.flatten())[order]
                weights = best_gmm.weights_[order]
                keep    = weights >= MIN_WEIGHT
                means, stds, weights = means[keep], stds[keep], weights[keep]
                weights /= weights.sum()
                n_comp  = len(means)

                merged_means, merged_stds, merged_weights = [], [], []
                used = [False] * n_comp
                for k in range(n_comp):
                    if used[k]: continue
                    gw = weights[k]; gmu = means[k]*weights[k]
                    gvar = (stds[k]**2 + means[k]**2)*weights[k]
                    used[k] = True
                    for j in range(k+1, n_comp):
                        if not used[j] and abs(means[k]-means[j]) < MIN_SEP:
                            gw += weights[j]; gmu += means[j]*weights[j]
                            gvar += (stds[j]**2 + means[j]**2)*weights[j]
                            used[j] = True
                    mu_m = gmu/gw
                    merged_means.append(mu_m)
                    merged_stds.append(np.sqrt(max(gvar/gw - mu_m**2, 1e-6)))
                    merged_weights.append(gw)

                merged_means   = np.array(merged_means)
                merged_stds    = np.array(merged_stds)
                merged_weights = np.array(merged_weights)
                merged_weights /= merged_weights.sum()

                components_scaled = []
                kde_max = kde_vals.max()
                for k in range(len(merged_means)):
                    c = (merged_weights[k] / (merged_stds[k]*np.sqrt(2*np.pi)) *
                         np.exp(-0.5*((x_range-merged_means[k])/merged_stds[k])**2))
                    peak_idx = np.argmax(c)
                    scale_k  = min(kde_vals[peak_idx], kde_max) / c[peak_idx] if c[peak_idx] > 0 else 1.0
                    components_scaled.append((c * scale_k, merged_means[k],
                                              merged_stds[k], merged_weights[k]))
                gmm_data = components_scaled
            except Exception as e:
                print(f"    GMM failed for {p_atom} {res_name}: {e}")

            gmm_store[(p_atom, res_name)] = dict(
                x_range=x_range, kde_vals=kde_vals, gmm=gmm_data, color=color)

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.fill_between(x_range, kde_vals, color=color, alpha=0.3, label="KDE")
            ax.plot(x_range, kde_vals, color=color, linewidth=1.5)
            if gmm_data:
                for k, (comp, mu, sigma, w) in enumerate(gmm_data):
                    ax.plot(x_range, comp,
                            color=gauss_colors[k % len(gauss_colors)],
                            linewidth=2, linestyle="--", alpha=0.9,
                            label=f"G{k+1}: {mu:.1f}A w={w:.2f}")
            ax.set_xlabel("Distance (A)", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(f"{p_atom} -> {res_name} | {channel_type} | Subunit {subunit}",
                         fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            fig.savefig(resid_dir / f"gauss_{p_atom}_{res_name}.png", dpi=150)
            plt.close(fig)

    overlay_pairs = [
        ("ARG60", "LYS64",  "Nterm: ARG60 vs LYS64"),
        ("LYS199","LYS200", "Cterm: LYS199 vs LYS200"),
    ]
    for p_atom in P_ATOMS:
        for res_a, res_b, pair_label in overlay_pairs:
            da = gmm_store.get((p_atom, res_a))
            db = gmm_store.get((p_atom, res_b))
            if not da or not db:
                continue

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.fill_between(da["x_range"], da["kde_vals"], color=da["color"], alpha=0.15)
            ax.fill_between(db["x_range"], db["kde_vals"], color=db["color"], alpha=0.15)
            ax.plot(da["x_range"], da["kde_vals"], color=da["color"], linewidth=1.5, label=f"{res_a} KDE")
            ax.plot(db["x_range"], db["kde_vals"], color=db["color"], linewidth=1.5, label=f"{res_b} KDE")
            if da["gmm"]:
                for k, (comp, mu, sigma, w) in enumerate(da["gmm"]):
                    ax.plot(da["x_range"], comp, color=da["color"], linewidth=2,
                            linestyle="--", alpha=0.9, label=f"{res_a} G{k+1}: {mu:.1f}A w={w:.2f}")
            if db["gmm"]:
                for k, (comp, mu, sigma, w) in enumerate(db["gmm"]):
                    ax.plot(db["x_range"], comp, color=db["color"], linewidth=2,
                            linestyle=":", alpha=0.9, label=f"{res_b} G{k+1}: {mu:.1f}A w={w:.2f}")

            ax.set_xlabel("Distance (A)", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(f"{p_atom} | {pair_label} | {channel_type} | Subunit {subunit}",
                         fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.tick_params(labelsize=10)
            plt.tight_layout()
            fig.savefig(resid_dir / f"overlay_{p_atom}_{res_a}_vs_{res_b}.png", dpi=150)
            plt.close(fig)

    print(f"  Per-residue gaussians -> {resid_dir}")


def _boxplot(df: pd.DataFrame, out_path: Path, channel_type: str,
             subunit: str, n_frames: int, n_runs: int):
    fig, axes = plt.subplots(3, 4, figsize=(11, 6), sharey=False)
    fig.suptitle(f"PIP2 headgroup distances — {channel_type} | Subunit {subunit}\n"
                 f"[{n_runs} runs, {n_frames} frames]", fontsize=14)

    for i, p_atom in enumerate(P_ATOMS):
        for j, cluster in enumerate(CLUSTERS):
            col = f"{p_atom}_{cluster}"
            ax  = axes[i][j]
            if col not in df.columns:
                ax.set_visible(False)
                continue
            vals  = df[col].dropna().values
            color = P_COLORS.get(p_atom, "#9E9E9E")

            bp = ax.boxplot([vals], patch_artist=True,
                            medianprops=dict(color="black", linewidth=2),
                            flierprops=dict(marker=".", markersize=1, alpha=0.1))
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)

            mean   = np.mean(vals)
            median = np.median(vals)
            ax.scatter(1, mean, marker="D", color="white", s=40,
                       zorder=5, edgecolors="black", linewidths=1)
            ax.set_title(f"{p_atom} → {cluster}", fontsize=11)
            ax.set_ylabel("Å", fontsize=10)
            ax.text(1.3, mean,   f"μ={mean:.1f}",   va="center", fontsize=9)
            ax.text(1.3, median, f"m={median:.1f}", va="center", fontsize=9, color="grey")
            ax.tick_params(axis="both", labelsize=9)
            ax.set_xticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _violinplot(df: pd.DataFrame, out_path: Path, channel_type: str,
                subunit: str, n_frames: int, n_runs: int):
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=False)
    fig.suptitle(f"PIP2 headgroup distances — {channel_type} | Subunit {subunit}\n"
                 f"[{n_runs} runs, {n_frames} frames]", fontsize=14)

    for j, cluster in enumerate(CLUSTERS):
        ax = axes[j]
        vals_list, labels, colors = [], [], []
        for p_atom in P_ATOMS:
            col = f"{p_atom}_{cluster}"
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue
            vals_list.append(vals)
            labels.append(p_atom)
            colors.append(P_COLORS.get(p_atom, "#9E9E9E"))

        if not vals_list:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(vals_list, positions=range(1, len(vals_list)+1),
                              showmeans=True, showmedians=True)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("white")

        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel("Distance (Å)", fontsize=13)
        ax.set_title(f"→ {CLUSTER_LABELS.get(cluster, cluster)}", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


GATE_SUBSETS = [
    ("asn_gate_frames",    "ASN gate (end2)"),
    ("asn_gate_m1_frames", "ASN gate -1 (pre-end2)"),
    ("hbc_gate_frames",    "HBC gate (end3)"),
    ("hbc_gate_m1_frames", "HBC gate -1 (pre-end3)"),
]


def _run_all_for_subset(runs_dir: Path, output_dir: Path, channel_type: str,
                         n_runs: int, exclude_runs: list | None = None,
                         subset: str | None = None, label: str = "all frames"):
    print(f"\n  === {label} ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    for sub in _subunit_order(channel_type):
        print(f"\n  Subunit {sub}...")
        df = _load_all_frames(runs_dir, sub,
                              exclude_runs=exclude_runs, subset=subset)
        if df.empty:
            continue

        sub_dir = output_dir / f"subunit_{sub}"
        sub_dir.mkdir(exist_ok=True)
        n = len(df)

        df.to_csv(sub_dir / "per_frame_all_runs.csv", index=False)
        summary = _make_summary(df, n_runs)
        summary.to_csv(sub_dir / "summary.csv", index=False)
        print(summary[["distance","mean","median","std"]].to_string(index=False))

        _histograms(df, sub_dir, channel_type, sub, n_runs)
        _boxplot(df, sub_dir / "boxplot.png", channel_type, sub, n, n_runs)
        _violinplot(df, sub_dir / "violin.png", channel_type, sub, n, n_runs)
        _residue_gaussians(runs_dir, sub_dir, channel_type, sub, n_runs,
                           exclude_runs=exclude_runs)
        print(f"  -> saved to {sub_dir}")


def run_combine_pip_headgroup_distances(
    runs_dir:     str,
    channel_type: str,
    output_dir:   Optional[str] = None,
    exclude_runs: list[str] | None = None,
):
    runs_dir = Path(runs_dir)
    if output_dir is None:
        output_dir = runs_dir / "combined_pip_headgroup_distances"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _sorted_run_dirs(runs_dir, exclude_runs=exclude_runs)
    n_runs   = len(run_dirs)

    print("\n" + "=" * 60)
    print("COMBINE PIP HEADGROUP DISTANCES")
    print(f"  Runs dir     : {runs_dir}")
    print(f"  Channel type : {channel_type}")
    print(f"  Subunit order: {_subunit_order(channel_type)}")
    print(f"  Runs found   : {n_runs}")
    print(f"  Output dir   : {output_dir}")
    if exclude_runs:
        print(f"  Excluding    : {exclude_runs}")
    print("=" * 60)

    # all frames
    _run_all_for_subset(runs_dir, output_dir, channel_type, n_runs,
                        exclude_runs=exclude_runs, subset=None, label="all frames")

    # gate subsets
    subunit_order = _subunit_order(channel_type)
    for folder_name, human_label in GATE_SUBSETS:
        has_subset = any(
            (rd / "pip_headgroup_distances" / folder_name /
             f"subunit_{subunit_order[0]}" / "per_frame.csv").exists()
            for rd in run_dirs
        )
        if has_subset:
            _run_all_for_subset(runs_dir, output_dir / folder_name,
                                channel_type, n_runs,
                                exclude_runs=exclude_runs,
                                subset=folder_name, label=human_label)
        else:
            print(f"\n  [{human_label}] not found in any RUN -- skipping.")

    print(f"\n✓ Combined headgroup distances -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir",      required=True)
    parser.add_argument("--channel_type",  required=True)
    parser.add_argument("--output_dir",    default=None)
    parser.add_argument("--exclude_runs",  nargs="+", default=None,
                        help="Run folders to exclude, e.g. --exclude_runs RUN4 RUN7")
    args = parser.parse_args()

    run_combine_pip_headgroup_distances(
        runs_dir     = args.runs_dir,
        channel_type = args.channel_type,
        output_dir   = args.output_dir,
        exclude_runs = args.exclude_runs if args.exclude_runs else None,
    )

# python3 combine_pip_headgroup_distances.py --runs_dir .../G12     --channel_type G12
# python3 combine_pip_headgroup_distances.py --runs_dir .../G12_ML  --channel_type G12_ML --exclude_runs RUN4
# python3 combine_pip_headgroup_distances.py --runs_dir .../G12_GAT --channel_type G12_GAT
# python3 combine_pip_headgroup_distances.py --runs_dir .../G2      --channel_type G2


"""
python3 combine_pip_headgroup_distances.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12 \
    --channel_type G12

python3 combine_pip_headgroup_distances.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_ML \
    --channel_type G12_ML \
    --exclude_runs RUN4

python3 combine_pip_headgroup_distances.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_GAT \
    --channel_type G12_GAT

python3 combine_pip_headgroup_distances.py \
    --runs_dir /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2 \
    --channel_type G2
"""