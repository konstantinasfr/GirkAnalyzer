"""
analysis/glu_ion_plots.py
=========================
Slide-ready figures for aggregate_glu_ion_hbond.py. Every value is a per-run mean +- SD (runs
with at least min_frames frames in the group), so error bars show run-to-run spread.

    fig_occupancy_<k>.png   % of frames with a K+ at each Glu (GIRK1 vs GIRK2)
    fig_partners_<k>.png    one panel per Glu: % with K+ when each partner's H-bond is formed vs
                            broken; partners in the same order for every Glu (GIRK1 equivalents
                            matched by position: F137~S148, Q165~Q176, W95~W106, D173~N184)
    fig_effect_<k>.png      formed - broken (percentage points, per run, then mean +- SD) per
                            partner, all Glus in one plot: < 0 = the bond keeps K+ away
    fig_values_<k>.csv      the numbers drawn
k = carbox (K+ <= 3.5 A from the carboxylate O) or any (from any Glu atom).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#1b2233", "#5f6675", "#e6e4de"
FORMED, BROKEN = "#2f5d9e", "#c3cad6"
GIRK1_C = "#c8651b"
GIRK2_C = ["#1f3b5c", "#2f5d9e", "#6f93c7", "#a9c0e0"]     # 2a 2b 2c 2d: same hue, stepped lightness
NONPARTNER = {"frame", "n_k_cavity", "n_k_any", "n_k_carbox", "n_partners"}
# display order of the partner positions (per_frame.csv lists them as offsets -4, -46, +24, +32)
POS_ORDER = [0, 2, 1, 3]                                    # S148/F137, Q176/Q165, W106/W95, N184/D173
POS_NAME = ["Ser148 / Phe137", "Gln176 / Gln165", "Trp106 / Trp95", "Asn184 / Asp173"]
KCOL = {"carbox": "n_k_carbox", "any": "n_k_any"}
KDESC = {"carbox": "K+ ≤ 3.5 Å from the carboxylate O", "any": "K+ ≤ 3.5 Å from any Glu atom"}

AA1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "HIE": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
       "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def _style():
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 13,
                         "axes.edgecolor": "#b9b4a6", "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 14})


def _one(res):
    return AA1.get(res[:3], res[:3]) + res[3:]


def glu_title(g):
    """GLU152_2a -> 'E152 · GIRK2 2a';  GLU141_G1 -> 'E141 · GIRK1'"""
    head, _, chain = g.partition("_")
    return f"{_one(head)} · GIRK1" if chain.upper().startswith("G") else f"{_one(head)} · GIRK2 {chain}"


def is_girk1(g):
    return g.rsplit("_", 1)[-1].upper().startswith("G")


def partners_of(df):
    cols = [c for c in df.columns if c not in NONPARTNER]
    return [cols[i] if i < len(cols) else None for i in POS_ORDER]


def _ms(vals):
    v = [x for x in vals if x == x]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


def run_pct(runs, col, kcol, state, min_frames):
    out = []
    for df in runs.values():
        g = df if col is None else df[df[col] == (1 if state == "formed" else 0)]
        if len(g) >= min_frames:
            out.append(100.0 * (g[kcol] > 0).mean())
    return out


def run_delta(runs, col, kcol, min_frames):
    out = []
    for df in runs.values():
        f, b = df[df[col] == 1], df[df[col] == 0]
        if len(f) >= min_frames and len(b) >= min_frames:
            out.append(100.0 * (f[kcol] > 0).mean() - 100.0 * (b[kcol] > 0).mean())
    return out


def make_figures(data, out_dir, title, min_frames=50):
    """data: {glu: {run: per_frame DataFrame}}"""
    _style()
    glus = sorted(data, key=lambda g: (0 if is_girk1(g) else 1, g))
    for k, kcol in KCOL.items():
        rows = []
        # ---- values ----
        occ = {}
        for g in glus:
            m, s, n = _ms(run_pct(data[g], None, kcol, None, 1))
            occ[g] = (m, s, n)
            rows.append(dict(figure="occupancy", glu=g, partner="", state="all", mean=m, sd=s, n_runs=n))
        part = {}
        for g in glus:
            any_df = next(iter(data[g].values()))
            cols = partners_of(any_df)
            part[g] = []
            for i, c in enumerate(cols):
                if c is None:
                    part[g].append(None)
                    continue
                fm = _ms(run_pct(data[g], c, kcol, "formed", min_frames))
                br = _ms(run_pct(data[g], c, kcol, "broken", min_frames))
                de = _ms(run_delta(data[g], c, kcol, min_frames))
                nf = int(sum((df[c] == 1).sum() for df in data[g].values()))
                part[g].append(dict(col=c, formed=fm, broken=br, delta=de, n_formed_frames=nf))
                for st, v in (("formed", fm), ("broken", br), ("formed-broken", de)):
                    rows.append(dict(figure="partners", glu=g, partner=c, state=st,
                                     mean=v[0], sd=v[1], n_runs=v[2]))
        pd.DataFrame(rows).to_csv(out_dir / f"fig_values_{k}.csv", index=False, float_format="%.2f")

        _fig_occupancy(glus, occ, f"{title}: how often each Glu holds a K+",
                       f"% of frames with {KDESC[k]} · mean ± SD over runs", out_dir / f"fig_occupancy_{k}.png")
        _fig_partners(glus, part, f"{title}: K+ at the Glu when each H-bond is formed or broken",
                      f"% of frames with {KDESC[k]} · side chain–side chain H-bonds · mean ± SD over runs",
                      out_dir / f"fig_partners_{k}.png")
        _fig_effect(glus, part, f"{title}: which H-bonds keep K+ away from the Glu",
                    f"formed − broken, percentage points of frames with {KDESC[k]} · per run, mean ± SD",
                    out_dir / f"fig_effect_{k}.png")
    for k in KCOL:
        table_matrix(data, out_dir, title, k)
    print(f"  slide figures -> {out_dir}/fig_*.png, table_glu_partner_*.png")


def _finish(fig, title, sub, path, top_in=1.25):
    """Title and subtitle in a band of top_in inches above the plots (no overlap)."""
    H = fig.get_figheight()
    fig.subplots_adjust(top=1 - top_in / H)
    fig.text(0.01, 1 - 0.22 / H, title, ha="left", va="top", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.62 / H, sub, ha="left", va="top", fontsize=12, color=MUTED)
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _clean(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    ax.tick_params(length=0)


def _fig_occupancy(glus, occ, title, sub, path):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(glus))
    j = 0
    cols = []
    for g in glus:
        if is_girk1(g):
            cols.append(GIRK1_C)
        else:
            cols.append(GIRK2_C[min(j, 3)])
            j += 1
    m = np.array([occ[g][0] for g in glus])
    s = np.array([occ[g][1] for g in glus])
    ax.bar(x, m, width=0.62, color=cols, zorder=2)
    ax.errorbar(x, m, yerr=s, fmt="none", ecolor=INK, elinewidth=1.4, capsize=5, zorder=3)
    for xi, mi, si, g in zip(x, m, s, glus):
        if mi == mi:
            ax.text(xi, mi + si + 3, f"{mi:.0f}%", ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK)
            ax.text(xi, 2, f"n={occ[g][2]}", ha="center", va="bottom", fontsize=10, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([glu_title(g).replace(" · ", "\n") for g in glus], fontsize=13)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of frames with K+ at the Glu")
    _clean(ax)
    fig.tight_layout()
    _finish(fig, title, sub, path)


def _fig_partners(glus, part, title, sub, path):
    n = len(glus)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, glus):
        x = np.arange(4)
        labels = []
        for i, p in enumerate(part[g]):
            if p is None:
                labels.append("—")
                continue
            labels.append(_one(p["col"]))
            if p["n_formed_frames"] == 0:
                ax.text(i, 4, "never\nbonded", ha="center", va="bottom", fontsize=10, color=MUTED)
            else:
                fm, br = p["formed"], p["broken"]
                for dx, v, c in ((-0.19, fm, FORMED), (0.19, br, BROKEN)):
                    if v[0] != v[0]:
                        ax.text(i + dx, 4, "too few\nframes", ha="center", va="bottom", fontsize=9, color=MUTED)
                    if v[0] == v[0]:
                        ax.bar(i + dx, v[0], width=0.36, color=c, zorder=2)
                        ax.errorbar(i + dx, v[0], yerr=v[1], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
                        ax.text(i + dx, v[0] + (v[1] if v[1] == v[1] else 0) + 2, f"{v[0]:.0f}",
                                ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13, fontweight="bold")
        ax.set_title(glu_title(g), color=GIRK1_C if is_girk1(g) else INK, pad=10)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        _clean(ax)
    axes[0].set_ylabel("% of frames with K+ at the Glu")
    from matplotlib.patches import Patch
    fig.tight_layout()
    H = fig.get_figheight()
    fig.legend(handles=[Patch(color=FORMED, label="H-bond formed"), Patch(color=BROKEN, label="H-bond broken")],
               loc="upper right", bbox_to_anchor=(0.995, 1 - 0.18 / H), frameon=False, ncol=2, fontsize=12)
    _finish(fig, title, sub, path, top_in=1.35)


def _fig_effect(glus, part, title, sub, path):
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ng = len(glus)
    h = 0.8 / ng
    j = 0
    colors = {}
    for g in glus:
        if is_girk1(g):
            colors[g] = GIRK1_C
        else:
            colors[g] = GIRK2_C[min(j, 3)]
            j += 1
    for i in range(4):
        for k, g in enumerate(glus):
            p = part[g][i]
            y = i + (k - (ng - 1) / 2) * h
            if p is None or p["n_formed_frames"] == 0:
                ax.text(1, y, "never bonded", va="center", ha="left", fontsize=10, color=MUTED)
                continue
            m, s, n = p["delta"]
            if m != m:
                ax.text(1, y, "too few frames", va="center", ha="left", fontsize=10, color=MUTED)
                continue
            ax.barh(y, m, height=h * 0.9, color=colors[g], zorder=2)
            ax.errorbar(m, y, xerr=s, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
            ax.text(m + (s if s == s else 0) * np.sign(m) + 2 * np.sign(m or 1), y, f"{m:+.0f}",
                    va="center", ha="left" if m >= 0 else "right", fontsize=10, color=INK)
    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.set_yticks(range(4))
    ax.set_yticklabels(POS_NAME, fontsize=13, fontweight="bold")
    ax.set_ylim(3.55, -0.85)
    ax.set_xlim(-100, 100)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xlabel("formed − broken  (percentage points)")
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(length=0)
    ax.text(-3, -0.68, "← bond keeps K+ away", fontsize=11, color=MUTED, ha="right", va="center")
    ax.text(3, -0.68, "bond goes with K+ →", fontsize=11, color=MUTED, ha="left", va="center")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[g], label=glu_title(g)) for g in glus], loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=len(glus), frameon=False, fontsize=11)
    fig.tight_layout()
    _finish(fig, title, sub, path, top_in=1.6)


# -----------------------------------------------------------------------------
# Compact table: rows = partner position, columns = Glu
# -----------------------------------------------------------------------------

def table_matrix(data, out_dir, title, k):
    """table_glu_partner_<k>.png/.csv: rows = the 4 partner positions (GIRK2 / GIRK1 residue),
    columns = the Glus; each cell = % of frames with K+ at the Glu when that H-bond is formed ->
    broken (all runs pooled), and how often the bond is there. Colour = formed - broken
    (blue: the bond keeps K+ away, orange: it goes with K+). Top row: K+ at the Glu, all frames."""
    from matplotlib.colors import to_rgb
    _style()
    kcol = KCOL[k]
    glus = sorted(data, key=lambda g: (0 if is_girk1(g) else 1, g))
    allf = {g: pd.concat(data[g].values(), ignore_index=True) for g in glus}
    rows, cells = [], {}
    for g in glus:
        df = allf[g]
        occ = 100.0 * (df[kcol] > 0).mean()
        cells[(g, "occ")] = occ
        rows.append(dict(glu=g, partner="all frames", pct_k_formed=np.nan, pct_k_broken=np.nan,
                         pct_k_all=occ, pct_frames_bonded=np.nan, n_frames=len(df)))
        for i, c in enumerate(partners_of(df)):
            if c is None:
                cells[(g, i)] = None
                continue
            f, b = df[df[c] == 1], df[df[c] == 0]
            v = dict(col=c, formed=100.0 * (f[kcol] > 0).mean() if len(f) else np.nan,
                     broken=100.0 * (b[kcol] > 0).mean() if len(b) else np.nan,
                     bonded=100.0 * len(f) / len(df), n_formed=len(f))
            cells[(g, i)] = v
            rows.append(dict(glu=g, partner=c, pct_k_formed=v["formed"], pct_k_broken=v["broken"],
                             pct_k_all=np.nan, pct_frames_bonded=v["bonded"], n_frames=len(df)))
    pd.DataFrame(rows).to_csv(out_dir / f"table_glu_partner_{k}.csv", index=False, float_format="%.2f")

    neg, pos, white = np.array(to_rgb("#2f5d9e")), np.array(to_rgb("#c8651b")), np.array([1.0, 1.0, 1.0])
    first_w, cw = 1.9, 1.35
    ncol, nrow = len(glus), 6                                   # header, occupancy, 4 partners
    heights = [0.9, 0.8, 1.15, 1.15, 1.15, 1.15]
    W, Htot = first_w + cw * ncol, sum(heights)
    fig, ax = plt.subplots(figsize=(W * 1.55, Htot * 1.0 + 1.6))
    ax.set_xlim(0, W)
    ax.set_ylim(Htot, 0)
    ax.axis("off")
    ys = np.cumsum([0] + heights)

    def box(x, y, w, h, fc, ec="#d9d6cf"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=0.8))

    navy = "#1f3b5c"
    box(0, ys[0], first_w, heights[0], navy)
    ax.text(first_w / 2, ys[0] + heights[0] / 2, "H-bond partner\n(GIRK2 / GIRK1)", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    for j, g in enumerate(glus):
        x = first_w + j * cw
        box(x, ys[0], cw, heights[0], GIRK1_C if is_girk1(g) else navy)
        ax.text(x + cw / 2, ys[0] + heights[0] / 2, glu_title(g).replace(" · ", "\n"), ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
    # occupancy row
    box(0, ys[1], first_w, heights[1], "#f4f2ec")
    ax.text(0.12, ys[1] + heights[1] / 2, "K+ at the Glu\n(all frames)", ha="left", va="center",
            fontsize=11, fontweight="bold", color=INK)
    for j, g in enumerate(glus):
        x = first_w + j * cw
        box(x, ys[1], cw, heights[1], "#f4f2ec")
        ax.text(x + cw / 2, ys[1] + heights[1] / 2, f"{cells[(g, 'occ')]:.1f}%", ha="center", va="center",
                fontsize=15, fontweight="bold", color=INK)
    # partner rows
    for i in range(4):
        y, h = ys[2 + i], heights[2 + i]
        box(0, y, first_w, h, "white")
        ax.text(0.12, y + h / 2, POS_NAME[i], ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
        for j, g in enumerate(glus):
            x = first_w + j * cw
            v = cells[(g, i)]
            if v is None or v["n_formed"] == 0:
                box(x, y, cw, h, "white")
                ax.text(x + cw / 2, y + h / 2, "never bonded", ha="center", va="center", fontsize=10, color=MUTED)
                continue
            d = v["formed"] - v["broken"]
            t = min(abs(d) / 60.0, 1.0) * 0.55
            fc = tuple(white - t * (white - (neg if d < 0 else pos)))
            box(x, y, cw, h, fc)
            ax.text(x + cw / 2, y + h * 0.36, f"{v['formed']:.1f} → {v['broken']:.1f}%", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=INK)
            ax.text(x + cw / 2, y + h * 0.72, f"bond present {v['bonded']:.0f}% of frames", ha="center",
                    va="center", fontsize=9, color=MUTED)
    fig.text(0.01, 0.985, f"{title}: K+ at each Glu with and without each H-bond", ha="left", va="top",
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.935, f"% of frames with {KDESC[k]}, H-bond formed → broken · all runs pooled",
             ha="left", va="top", fontsize=11, color=MUTED)
    fig.text(0.01, 0.025, "Blue: the H-bond keeps K+ away from the Glu.  Orange: the H-bond goes with K+ at the Glu.  "
             "Side chain–side chain H-bonds.", ha="left", va="bottom", fontsize=10, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08)
    fig.savefig(out_dir / f"table_glu_partner_{k}.png", dpi=200, facecolor="white")
    plt.close(fig)
