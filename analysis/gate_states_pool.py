"""
analysis/gate_states_pool.py
============================
Pools the per-run results of --gate_states_only over all runs and clusters
the frames by their two HBC diagonals:

    python3 main.py --gate_states_pool
    python3 main.py --gate_states_pool /path/to/girk_analyser_results --gs_pool_k 4

Reads every <root>/**/gate_states/frames.csv.gz (+ run_info.json,
interaction_columns.csv). Results: <root>/gate_states_pooled/

IDEA
----
Every frame of every run is a point (A_C, B_D) (CA-CA of the gate Phe of the
two diagonals). The points are clustered with a Gaussian mixture (full
covariance, EM, k-means++ start, several restarts). The number of clusters is
--gs_pool_k, or chosen automatically: the smallest k that gives >= 90 % of the
improvement in BIC between k = 1 and k = 8 (frames of MD are correlated, so the
plain BIC minimum would always ask for more clusters). Clusters are numbered
by their mean A_C + B_D, so cluster 1 is the narrowest gate.

For each cluster: how many frames, which runs / channel types visit it, the
mean CA-CA distances, and per gate subunit (A, B, C, D) the % frames in the
pocket (pi-stack with F83 or F186 of the neighbour), latched on F83, on
F186, pi-stacked with another gate Phe, the local TM2 bend (HELANAL) at G189
and G180, chi1 and the distance of the gate CA from the pore axis, and the %
frames of every gate-ring interaction. Interactions are pooled by gate
subunit, partner subunit, partner position from its HBC Phe and kind (e.g.
B:C-109:pi-pi = gate B with F83 of subunit C, pi-pi), so they match between
runs and channel types.

OUTPUT
------
runs.csv                     the runs found (run, channel type, frames, folder)
bic.csv                      BIC for k = 1..8 and the k used
frame_clusters.csv.gz        run, channel_type, frame, A_C, B_D, cluster
clusters.csv                 one row per cluster: frames, %, A_C / B_D mean +- sd,
                             per subunit in_pocket / latch / latch186 / gate_stack
                             %, bend_G189 / bend_G180 (mean, sd), r_gate, chi1
occupancy.csv                % of the frames of every run (and channel type) in
                             each cluster
cluster_interactions.csv     % frames per cluster x pooled interaction
map.png                      A_C vs B_D of all frames coloured by cluster (all
                             runs, and one panel per channel type)
occupancy.png                per run: fraction of frames in each cluster
cluster_gates.png            per subunit and cluster: in pocket, gate-gate stack,
                             local bend G189 / G180, distance from the pore axis
cluster_interactions.png     2 x 2 (one panel per gate subunit), bars per cluster
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIRNAME = "gate_states_pooled"
KMAX = 8
CL_COLOURS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860",
              "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]
GATE_KEYS = (("in_pocket", 100, "in pocket (F83 or F186) %"), ("latch", 100, "on F83 %"),
             ("latch186", 100, "on F186 %"), ("gate_stack", 100, "π-stack with a gate Phe %"),
             ("bend_G189", 1, "local bend G189 (deg)"), ("bend_G180", 1, "local bend G180 (deg)"),
             ("r_gate", 1, "gate CA from pore axis (Å)"), ("chi1", 1, "χ1 (deg)"))
ONE = dict(PHE="F", TYR="Y", TRP="W", HIS="H", HSD="H", HSE="H", HIE="H", HID="H", CYS="C",
           MET="M", ALA="A", VAL="V", LEU="L", ILE="I", THR="T", LYS="K", ARG="R")


# --------------------------------------------------------------------------
# Gaussian mixture (numpy)
# --------------------------------------------------------------------------
def _kpp(X, k, rng):
    c = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min(((X[:, None] - np.array(c)[None]) ** 2).sum(-1), 1)
        c.append(X[rng.choice(len(X), p=d / d.sum())])
    return np.array(c)


def _logpdf(X, mu, cov):
    d = X.shape[1]
    out = np.empty((len(X), len(mu)))
    for j in range(len(mu)):
        L = np.linalg.cholesky(cov[j])
        z = np.linalg.solve(L, (X - mu[j]).T)
        out[:, j] = -0.5 * (z ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)
    return out


def gmm_fit(X, k, n_init=4, iters=300, tol=1e-6, seed=0):
    """Full-covariance Gaussian mixture by EM. Returns dict(w, mu, cov, ll)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    reg = 1e-6 * np.eye(d)
    best = None
    for _ in range(n_init):
        mu = _kpp(X, k, rng)
        cov = np.array([np.cov(X.T) + reg] * k)
        w = np.full(k, 1 / k)
        prev = -np.inf
        for _ in range(iters):
            lp = _logpdf(X, mu, cov) + np.log(w)
            m = lp.max(1, keepdims=True)
            ll_i = m[:, 0] + np.log(np.exp(lp - m).sum(1))
            r = np.exp(lp - ll_i[:, None])
            ll = ll_i.sum()
            nk = r.sum(0) + 1e-12
            w = nk / n
            mu = (r.T @ X) / nk[:, None]
            for j in range(k):
                Xc = X - mu[j]
                cov[j] = (r[:, j, None] * Xc).T @ Xc / nk[j] + reg
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        if best is None or ll > best["ll"]:
            best = dict(w=w.copy(), mu=mu.copy(), cov=cov.copy(), ll=ll)
    return best


def gmm_predict(X, g):
    return np.argmax(_logpdf(X, g["mu"], g["cov"]) + np.log(g["w"]), 1)


def bic(g, n, d=2):
    k = len(g["w"])
    p = k * (d + d * (d + 1) / 2) + k - 1
    return -2 * g["ll"] + p * np.log(n)


def choose_k(bics, frac=0.9):
    b = np.asarray(bics)
    drop = b[0] - b.min()
    if drop <= 0:
        return 1
    for k, v in enumerate(b, start=1):
        if b[0] - v >= frac * drop:
            return k
    return int(np.argmin(b)) + 1


# --------------------------------------------------------------------------
def find_runs(root):
    runs = []
    for f in sorted(Path(root).glob("**/gate_states/frames.csv.gz")):
        d = f.parent
        info = json.load(open(d / "run_info.json")) if (d / "run_info.json").exists() else {}
        runs.append(dict(run=info.get("run", d.parent.name),
                         channel_type=info.get("channel_type", d.parent.parent.name),
                         folder=str(d), info=info))
    return runs


def load_run(r):
    d = Path(r["folder"])
    fr = pd.read_csv(d / "frames.csv.gz")
    try:
        meta = pd.read_csv(d / "interaction_columns.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        meta = pd.DataFrame()
    ren, names = {}, {}
    for _, m in meta.iterrows():
        if m.column not in fr:
            continue
        key = f"{m.gate_subunit}:{m.subunit}{int(m.pos):+d}:{m.interaction}"
        ren[m.column] = "int|" + key
        names[key] = (m.resname, int(m.pos), m.subunit, m.gate_subunit, m.interaction)
    fr = fr.rename(columns=ren)
    fr.insert(0, "run", r["run"])
    fr.insert(1, "channel_type", r["channel_type"])
    return fr, names


def run_gate_states_pool(root, k=None, out=None, max_fit=40000, seed=0):
    root = Path(root)
    out = Path(out) if out else root / OUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\nGATE STATES POOLED -> {out}\n{'=' * 60}")
    runs = find_runs(root)
    if not runs:
        print(f"  no */gate_states/frames.csv.gz under {root}: run --gate_states_only first")
        return None
    frames, names = [], {}
    for r in runs:
        fr, nm = load_run(r)
        frames.append(fr)
        names.update(nm)
        r["n_frames"] = len(fr)
    pd.DataFrame([{k_: v for k_, v in r.items() if k_ != "info"} for r in runs]).to_csv(
        out / "runs.csv", index=False)
    print(f"  {len(runs)} run(s): " + ", ".join(f"{r['channel_type']}/{r['run']}" for r in runs))
    df = pd.concat(frames, ignore_index=True)
    icols = [c for c in df.columns if c.startswith("int|")]
    df[icols] = df[icols].fillna(0)            # absent in a run = present in < 1 % there

    # ---- clustering
    X = df[["A_C", "B_D"]].to_numpy(float)
    ok = np.isfinite(X).all(1)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    fit_idx = idx if len(idx) <= max_fit else np.sort(rng.choice(idx, max_fit, replace=False))
    Xf = X[fit_idx]
    fits, bics = [], []
    for kk in range(1, KMAX + 1):
        g = gmm_fit(Xf, kk, seed=seed)
        fits.append(g)
        bics.append(bic(g, len(Xf)))
    kk = int(k) if k else choose_k(bics)
    pd.DataFrame(dict(k=range(1, KMAX + 1), bic=bics, used=[i + 1 == kk for i in range(KMAX)])).to_csv(
        out / "bic.csv", index=False)
    g = fits[kk - 1]
    order = np.argsort(g["mu"].sum(1))                         # narrowest first
    relabel = {int(o): i + 1 for i, o in enumerate(order)}
    lab = np.zeros(len(df), int)
    lab[ok] = [relabel[int(v)] for v in gmm_predict(X[ok], g)]
    df["cluster"] = lab
    print(f"  clusters: k = {kk}" + ("" if k else " (automatic, see bic.csv)"))
    df[["run", "channel_type", "frame", "A_C", "B_D", "cluster"]].to_csv(
        out / "frame_clusters.csv.gz", index=False, float_format="%.3f")

    # ---- per cluster
    subs = sorted({c.rsplit("_", 1)[1] for c in df.columns
                   if c.startswith(("in_pocket_", "bend_G189_", "r_gate_")) and len(c.rsplit("_", 1)[1]) == 1})
    rows = []
    for c in range(1, kk + 1):
        d = df[df.cluster == c]
        row = dict(cluster=c, n_frames=len(d), pct_frames=100 * len(d) / ok.sum(),
                   n_runs=d.run.nunique(), A_C=d.A_C.mean(), A_C_sd=d.A_C.std(),
                   B_D=d.B_D.mean(), B_D_sd=d.B_D.std())
        for s in subs:
            for key, f, _ in GATE_KEYS:
                col = f"{key}_{s}"
                if col in d:
                    row[f"{key}{'_pct' if f == 100 else ''}_{s}"] = f * d[col].mean()
                    if key.startswith("bend"):
                        row[f"{key}_sd_{s}"] = d[col].std()
        rows.append(row)
    cl = pd.DataFrame(rows)
    cl.to_csv(out / "clusters.csv", index=False, float_format="%.3f")
    occ = (pd.crosstab([df.channel_type, df.run], df.cluster, normalize="index") * 100).reset_index()
    occ.columns = [str(c) if not isinstance(c, int) else f"cluster_{c}" for c in occ.columns]
    occ.to_csv(out / "occupancy.csv", index=False, float_format="%.2f")
    ci = []
    for col in icols:
        key = col[4:]
        rn, pos, psub, gsub, kind = names.get(key, ("", 0, "", key.split(":")[0], key.rsplit(":", 1)[1]))
        for c in range(1, kk + 1):
            ci.append(dict(cluster=c, gate_subunit=gsub, partner_subunit=psub, pos=pos, resname=rn,
                           interaction=kind, key=key,
                           label=f"{ONE.get(rn, 'X')}{192 + pos}.{psub}",
                           pct=100 * df.loc[df.cluster == c, col].mean()))
    ci = pd.DataFrame(ci)
    ci.to_csv(out / "cluster_interactions.csv", index=False, float_format="%.3f")

    # ---- plots
    gate_names = {}
    for r in runs:
        for gg in r["info"].get("gates", []):
            gate_names.setdefault(gg["subunit"], set()).add(gg["gate"])
    gl = {s: f"{s} ({'/'.join(sorted(v))})" for s, v in gate_names.items()}
    plot_map(df, cl, g, relabel, out / "map.png")
    plot_occupancy(occ, kk, out / "occupancy.png")
    plot_cluster_gates(cl, subs, gl, out / "cluster_gates.png")
    plot_cluster_interactions(ci, cl, gl, out / "cluster_interactions.png")
    print(cl[["cluster", "n_frames", "pct_frames", "n_runs", "A_C", "B_D"] +
             [c for c in cl.columns if c.startswith("in_pocket_pct_")]].round(1).to_string(index=False))
    print(occ.round(1).to_string(index=False))
    print(f"  -> {out}")
    return out


# --------------------------------------------------------------------------
def _legend_cl(cl):
    from matplotlib.patches import Patch
    return [Patch(color=CL_COLOURS[(int(r.cluster) - 1) % len(CL_COLOURS)],
                  label=f"cluster {int(r.cluster)}: A_C {r.A_C:.1f} / B_D {r.B_D:.1f} Å, "
                        f"{r.pct_frames:.0f} % of frames, {int(r.n_runs)} run(s)")
            for _, r in cl.iterrows()]


def plot_map(df, cl, g, relabel, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    types = sorted(df.channel_type.unique())
    n = 1 + (len(types) if len(types) > 1 else 0)
    fig, axes = plt.subplots(1, n, figsize=(max(12, 6.5 * n), 7 + 0.35 * len(cl)), squeeze=False,
                             sharex=True, sharey=True)
    sets = [("all runs", df)] + ([(t, df[df.channel_type == t]) for t in types] if n > 1 else [])
    for ax, (name, d) in zip(axes[0], sets):
        sub = d.sample(min(len(d), 60000), random_state=0) if len(d) > 60000 else d
        cols = [CL_COLOURS[(c - 1) % len(CL_COLOURS)] if c > 0 else "0.7" for c in sub.cluster]
        ax.scatter(sub.A_C, sub.B_D, s=2, c=cols, alpha=0.3, rasterized=True)
        for j, mu in enumerate(g["mu"]):
            cnum = relabel[j]
            vals, vecs = np.linalg.eigh(g["cov"][j])
            t = np.linspace(0, 2 * np.pi, 100)
            e = (vecs @ (2 * np.sqrt(vals)[:, None] * np.vstack([np.cos(t), np.sin(t)]))).T + mu
            ax.plot(e[:, 0], e[:, 1], color="black", lw=1)
            ax.text(mu[0], mu[1], str(cnum), ha="center", va="center", fontsize=14, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
        ax.set_title(f"{name} ({len(d)} frames)", fontsize=14)
        ax.set_xlabel("A_C Cα-Cα (Å)", fontsize=13)
        ax.grid(color="0.92", lw=0.6)
    axes[0, 0].set_ylabel("B_D Cα-Cα (Å)", fontsize=13)
    fig.legend(handles=_legend_cl(cl), loc="lower center", ncol=1, frameon=False, fontsize=11)
    fig.suptitle("HBC gate states of all frames\n(Gaussian mixture on A_C, B_D; ellipses = 2 sd)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0.03 + 0.035 * len(cl), 1, 0.92))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_occupancy(occ, k, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{t}\n{r}" for t, r in zip(occ.channel_type, occ.run)]
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(labels) + 3), 6))
    bottom = np.zeros(len(occ))
    for c in range(1, k + 1):
        v = occ.get(f"cluster_{c}", pd.Series(np.zeros(len(occ)))).to_numpy()
        ax.bar(range(len(occ)), v, bottom=bottom, color=CL_COLOURS[(c - 1) % len(CL_COLOURS)],
               label=f"cluster {c}")
        bottom += v
    ax.set_xticks(range(len(occ)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% of the run's frames", fontsize=13)
    ax.set_ylim(0, 100)
    ax.legend(ncol=k, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_cluster_gates(cl, subs, gl, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("in_pocket", "_pct", "in pocket (%)"), ("gate_stack", "_pct", "π-stack with\na gate Phe (%)"),
            ("bend_G189", "", "local bend\nG189 (deg)"), ("bend_G180", "", "local bend\nG180 (deg)"),
            ("r_gate", "", "gate CA from\npore axis (Å)")]
    keys = [k_ for k_ in keys if any(f"{k_[0]}{k_[1]}_{s}" in cl for s in subs)]
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), len(subs), figsize=(4.2 * len(subs), 2.9 * len(keys)),
                             squeeze=False, sharey="row")
    for i, (key, suf, lab) in enumerate(keys):
        for j, s in enumerate(subs):
            ax = axes[i, j]
            col = f"{key}{suf}_{s}"
            if col not in cl:
                ax.axis("off")
                continue
            sd = cl.get(f"{key}_sd_{s}")
            ax.bar(cl.cluster, cl[col], color=[CL_COLOURS[(c - 1) % len(CL_COLOURS)] for c in cl.cluster],
                   yerr=sd if sd is not None else None, capsize=3)
            ax.set_xticks(cl.cluster)
            ax.grid(axis="y", color="0.92", lw=0.6)
            if i == 0:
                ax.set_title(gl.get(s, s), fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(lab, fontsize=12)
            if i == len(keys) - 1:
                ax.set_xlabel("cluster", fontsize=12)
    fig.legend(handles=_legend_cl(cl), loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("per gate subunit and cluster (error bars = sd over frames)", fontsize=14)
    fig.tight_layout(rect=(0, 0.04 + 0.025 * ((len(cl) + 1) // 2), 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_cluster_interactions(ci, cl, gl, out_png, min_pct=5.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.phe_interactions import LABEL
    if ci.empty:
        return
    best = ci.groupby("key").pct.max()
    ci = ci[ci.key.isin(best[best >= min_pct].index)]
    if ci.empty:
        return
    gsubs = sorted(ci.gate_subunit.unique())
    k = int(cl.cluster.max())
    nmax = max(ci[ci.gate_subunit == s].key.nunique() for s in gsubs)
    fig, axes = plt.subplots(2, 2, figsize=(max(16, 0.45 * k * nmax * 2 + 3), 12), sharey=True)
    bw = 0.8 / k
    for ax, s in zip(axes.flat, gsubs):
        d = ci[ci.gate_subunit == s].sort_values(["pos", "partner_subunit", "interaction"])
        keys = list(dict.fromkeys(d.key))
        for x, key in enumerate(keys):
            dd = d[d.key == key].set_index("cluster")
            for c in range(1, k + 1):
                ax.bar(x + 0.1 + (c - 0.5) * bw, dd.pct.get(c, 0), bw * 0.92,
                       color=CL_COLOURS[(c - 1) % len(CL_COLOURS)])
            r = dd.iloc[0]
            ax.annotate(f"{r.label}\n{LABEL.get(r.interaction, r.interaction)}", (x + 0.5, 0),
                        xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                        fontsize=12, annotation_clip=False)
        ax.set_xlim(0, max(len(keys), 1))
        ax.set_xticks([])
        ax.set_ylim(0, 105)
        ax.set_title(f"gate {gl.get(s, s)}", fontsize=15, fontweight="bold")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.tick_params(axis="y", labelsize=12)
    for ax in axes.flat[len(gsubs):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("% frames of the cluster", fontsize=13)
    fig.legend(handles=_legend_cl(cl), loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle("gate Phe ring interactions per cluster, pooled over runs (partner = residue at "
                 "the GIRK2 position, subunit letter; ≥ 5 % in some cluster)", fontsize=14)
    fig.tight_layout(rect=(0, 0.06 + 0.025 * ((len(cl) + 1) // 2), 1, 0.94), h_pad=6)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
