#!/usr/bin/env python3
"""
gate_openness.py
================
For every run, how much of the time was each gate too narrow to conduct?

Reads the three gate-distance files per run

    hbc_distances.csv        frame, mean, A_C, B_D
    upper_gl_distances.csv   frame, mean, A_C, B_D
    lower_gl_distances.csv   frame, mean, A_C, B_D

and the per-run permeation summary, and writes two tables.

THE THRESHOLD
-------------
Default 5.6 A. The published Kir/GIRK criterion is 5.69 A (Li et al. 2019):
a gate narrower than that cannot pass a K+ ion, and a channel counts as open
only when EVERY gate exceeds it. Set --threshold 5.69 to use it exactly.
The number refers to the MINIMUM atom-to-atom distance across the pore, not
a Calpha distance -- Calpha distances for these gates run ~11-15 A, so if
your input files hold Calpha distances this threshold is meaningless for
them. Check that your values sit in the 4-8 A range before trusting any of
this.

TWO TABLES, TWO DEFINITIONS OF "NARROW"
---------------------------------------
    gate_openness_mean.csv      a gate is narrow in a frame when its `mean`
                                column is below the threshold
    gate_openness_diagonal.csv  a gate is narrow when EITHER diagonal (A_C or
                                B_D) is below it, i.e. min(A_C, B_D) < thr

The second is the physical one: an ion passes the narrowest point, so a gate
with one diagonal at 4 A and the other at 8 A is shut even though the mean
reads 6 A. The two tables will differ, and by how much is itself a result --
it tells you how asymmetric the gates are.

COLUMNS (both tables, one row per run)
--------------------------------------
    n_frames                        frames present in the gate files
    frames_narrow_<gate>            frames that gate was below the threshold
    frac_narrow_<gate>              the same as a fraction
    frames_any_narrow               frames where AT LEAST ONE gate was narrow
                                    -- the channel could not conduct
    frames_all_open                 frames where every gate cleared it
    open_probability                frames_all_open / n_frames
    frames_narrow_<gate>_A_C        per-diagonal detail (diagonal table only)
    frames_narrow_<gate>_B_D
    n_permeations                   from the permeation summary
    plus every numeric column of the permeation summary, for reference

USAGE
-----
    python3 gate_openness.py \
        /media/ziyue/.../girk_analyser_results/G12 --channel G12

    # exact published threshold, and only some runs
    python3 gate_openness.py .../G12 --channel G12 --threshold 5.69 --runs 1-20

    # a different permeation column as `n_permeations`
    python3 gate_openness.py .../G12 --channel G12 --perm_col HBC_Permeations

OUTPUTS   <channel>/gate_openness/
----------------------------------
    gate_openness_mean_<CH>.csv
    gate_openness_diagonal_<CH>.csv
    gate_openness_summary_<CH>.csv        pooled over runs, both definitions
    gate_openness_<CH>_mean.png           closed fraction vs permeations
    gate_openness_<CH>_diagonal.png
    gate_openness_distributions_<CH>.png  distance histograms per gate
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_VERSION = "2026-10-06b  (auto-detects the permeation column)"

GATES = {"hbc": "hbc_distances.csv",
         "upper_gl": "upper_gl_distances.csv",
         "lower_gl": "lower_gl_distances.csv"}
GATE_ORDER = ["hbc", "upper_gl", "lower_gl"]
DIAGONALS = ["A_C", "B_D"]

PERM_DIR = "permeation_summary"
DEFAULT_THRESHOLD = 5.6
DEFAULT_PERM_COL = "Full_Permeations_Unique"
# Different channels' summaries use different column names, so when the
# requested one is absent the first of these that exists is used instead.
# Order = preference: a unique full-permeation count beats a per-frame rate.
PERM_COL_CANDIDATES = ["Full_Permeations_Unique", "Final_Permeations",
                       "HBC_Permeations", "Full_Permeations_Total",
                       "Permeations", "n_permeations"]

GATE_COLOR = {"hbc": "#C0392B", "upper_gl": "#2980B9", "lower_gl": "#27AE60"}
OUT_SUBDIR = "gate_openness"

# Distances outside this range almost certainly mean the file holds something
# other than a minimum atom-to-atom distance (Calpha distances sit at 11-15 A).
SANE_RANGE = (1.0, 12.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Discovery
# ─────────────────────────────────────────────────────────────────────────────

def expand_runs(spec):
    """"3", "3,5", "1-20", "RUN3" -> ["RUN3", ...]"""
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


def find_run_dirs(base, runs=None):
    """
    [(run_name, dir_holding_the_gate_files)].

    The gate files normally sit directly in RUNnn/; one extra level is checked
    so a subfolder layout still works. Bounded globs only -- never a full
    recursive walk, which is painfully slow on the external drive.
    """
    base = Path(base)
    if not base.is_dir():
        sys.exit(f"ERROR: not a directory: {base}")
    names = runs if runs else sorted(
        (d.name for d in base.glob("RUN*") if d.is_dir()),
        key=lambda n: (len(n), n))

    found = []
    for name in names:
        rd = base / name
        if not rd.is_dir():
            print(f"  [warn] {name}: no such directory")
            continue
        if (rd / GATES["hbc"]).is_file():
            found.append((name, rd))
            continue
        deeper = sorted(rd.glob(f"*/{GATES['hbc']}"))
        if deeper:
            found.append((name, deeper[0].parent))
        else:
            print(f"  [warn] {name}: no {GATES['hbc']} found")
    return found


def natural_key(name):
    digits = "".join(c for c in name if c.isdigit())
    return (int(digits) if digits else 0, name)


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gate(path):
    """frame, mean, A_C, B_D as float arrays. None if unusable."""
    path = Path(path)
    if not path.is_file():
        return None
    frames, cols = [], {k: [] for k in ["mean"] + DIAGONALS}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                fr = int(float(row["frame"]))
                vals = {k: float(row[k]) for k in cols}
            except (KeyError, TypeError, ValueError):
                continue
            frames.append(fr)
            for k, v in vals.items():
                cols[k].append(v)
    if not frames:
        return None
    return dict(frame=np.array(frames),
                **{k: np.array(v, dtype=float) for k, v in cols.items()})


def load_permeations(base, perm_col):
    """Run -> dict of every numeric column in the permeation summary."""
    candidates = (sorted((base / PERM_DIR).glob("*.csv"))
                  if (base / PERM_DIR).is_dir() else [])
    # A folder can hold several summaries; try the ones that actually carry a
    # permeation column first, so an unrelated CSV with a Run column cannot
    # win just by sorting earlier.
    def _rank(path):
        try:
            with open(path, newline="") as f:
                head = next(csv.reader(f), [])
        except OSError:
            return 2
        if perm_col in head:
            return 0
        return 1 if any(c in head for c in PERM_COL_CANDIDATES) else 2
    candidates.sort(key=_rank)

    for candidate in candidates:
        with open(candidate, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or "Run" not in rows[0]:
            continue
        out = {}
        for r in rows:
            name = str(r["Run"]).strip()
            rec = {}
            for k, v in r.items():
                if k == "Run":
                    continue
                try:
                    fv = float(v)
                    rec[k] = int(fv) if fv.is_integer() else round(fv, 6)
                except (TypeError, ValueError):
                    rec[k] = v
            out[name] = rec
        chosen = perm_col
        if chosen not in rows[0]:
            fallback = next((c for c in PERM_COL_CANDIDATES if c in rows[0]),
                            None)
            avail = ", ".join(k for k in rows[0] if k != "Run")
            if fallback:
                print(f"  [warn] '{perm_col}' not in {candidate.name} — "
                      f"using '{fallback}' instead.\n"
                      f"         available: {avail}\n"
                      f"         override with --perm_col <name>")
                chosen = fallback
            else:
                print(f"  [warn] '{perm_col}' not in {candidate.name} and no "
                      f"known alternative — n_permeations will be blank.\n"
                      f"         available: {avail}")
                chosen = None
        print(f"  Permeations from {candidate.relative_to(base)} "
              f"({len(out)} runs, column "
              f"{repr(chosen) if chosen else 'NONE'})")
        return out, chosen
    print(f"  [warn] no permeation summary CSV under {base / PERM_DIR} — "
          f"n_permeations will be blank")
    return {}, None


# ─────────────────────────────────────────────────────────────────────────────
#  Per-run computation
# ─────────────────────────────────────────────────────────────────────────────

def analyse_run(run, gate_dir, threshold):
    """
    (mean_row, diagonal_row, loaded_gates) for one run, or (None, None, None).

    mean definition      : gate narrow when its `mean` column < threshold
    diagonal definition  : gate narrow when min(A_C, B_D) < threshold
    """
    loaded = {}
    for gate, fname in GATES.items():
        d = load_gate(Path(gate_dir) / fname)
        if d is None:
            print(f"  [warn] {run}: missing or empty {fname} — run skipped")
            return None, None, None
        loaded[gate] = d

    lengths = {g: d["frame"].size for g, d in loaded.items()}
    n = min(lengths.values())
    if len(set(lengths.values())) > 1:
        print(f"  [warn] {run}: gate files differ in length {lengths} — "
              f"truncating to {n} frames")

    mean_row = dict(run=run, n_frames=n, threshold=threshold)
    diag_row = dict(run=run, n_frames=n, threshold=threshold)
    any_mean = np.zeros(n, dtype=bool)
    any_diag = np.zeros(n, dtype=bool)

    for gate in GATE_ORDER:
        d = loaded[gate]
        mean_v = d["mean"][:n]
        ac, bd = d["A_C"][:n], d["B_D"][:n]
        worst = np.minimum(ac, bd)          # the narrowest diagonal

        nm = mean_v < threshold
        nd = worst < threshold
        any_mean |= nm
        any_diag |= nd

        mean_row[f"frames_narrow_{gate}"] = int(nm.sum())
        mean_row[f"frac_narrow_{gate}"] = round(float(nm.mean()), 4)
        mean_row[f"mean_dist_{gate}"] = round(float(mean_v.mean()), 3)
        mean_row[f"min_dist_{gate}"] = round(float(mean_v.min()), 3)

        diag_row[f"frames_narrow_{gate}"] = int(nd.sum())
        diag_row[f"frac_narrow_{gate}"] = round(float(nd.mean()), 4)
        for name, arr in (("A_C", ac), ("B_D", bd)):
            diag_row[f"frames_narrow_{gate}_{name}"] = int((arr < threshold).sum())
            diag_row[f"frac_narrow_{gate}_{name}"] = round(
                float((arr < threshold).mean()), 4)
        diag_row[f"mean_dist_{gate}"] = round(float(worst.mean()), 3)
        diag_row[f"min_dist_{gate}"] = round(float(worst.min()), 3)
        # how often the two diagonals disagree about this gate being shut
        diag_row[f"frames_one_diag_only_{gate}"] = int(
            ((ac < threshold) ^ (bd < threshold)).sum())

    for row, mask in ((mean_row, any_mean), (diag_row, any_diag)):
        row["frames_any_narrow"] = int(mask.sum())
        row["frac_any_narrow"] = round(float(mask.mean()), 4)
        row["frames_all_open"] = int((~mask).sum())
        row["open_probability"] = round(float((~mask).mean()), 4)

    return mean_row, diag_row, loaded


def attach_permeations(row, perms, perm_col):
    rec = perms.get(row["run"], {})
    row["n_permeations"] = rec.get(perm_col, "") if perm_col else ""
    row["permeation_column"] = perm_col or ""
    for k, v in rec.items():
        if k != perm_col:
            row[f"perm_{k}"] = v
    return row


def sanity_check(loaded_by_run, threshold):
    """Warn if the distances look like Calpha distances rather than minima."""
    allv = []
    for loaded in loaded_by_run:
        for gate in GATE_ORDER:
            allv.append(loaded[gate]["mean"])
    if not allv:
        return
    v = np.concatenate(allv)
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    print(f"\n  distance sanity check: 1st-99th percentile = "
          f"{lo:.2f}-{hi:.2f} A  (threshold {threshold} A)")
    if lo > SANE_RANGE[1]:
        print("  [WARN] these look like Calpha distances (~11-15 A), not "
              "minimum atom-to-atom distances. The 5.69 A criterion does not "
              "apply to them — every frame will read as open.")
    elif hi < threshold:
        print("  [WARN] every distance is below the threshold — every frame "
              "will read as closed. Check the threshold and the units.")


# ─────────────────────────────────────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────────────────────────────────────

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


def pooled_summary(mean_rows, diag_rows, channel, threshold):
    """Totals across runs, both definitions side by side."""
    out = []
    for label, rows in (("mean", mean_rows), ("diagonal", diag_rows)):
        tot_frames = sum(r["n_frames"] for r in rows)
        for gate in GATE_ORDER:
            narrow = sum(r[f"frames_narrow_{gate}"] for r in rows)
            out.append(dict(channel_type=channel, definition=label,
                            threshold=threshold, scope=gate, n_runs=len(rows),
                            total_frames=tot_frames, frames_narrow=narrow,
                            frac_narrow=round(narrow / tot_frames, 4)
                            if tot_frames else ""))
        anyn = sum(r["frames_any_narrow"] for r in rows)
        out.append(dict(channel_type=channel, definition=label,
                        threshold=threshold, scope="any_gate_narrow",
                        n_runs=len(rows), total_frames=tot_frames,
                        frames_narrow=anyn,
                        frac_narrow=round(anyn / tot_frames, 4)
                        if tot_frames else ""))
    return out


def plot_vs_permeations(rows, out_path, title):
    pts = [(r["frac_any_narrow"] * 100, float(r["n_permeations"]), r["run"])
           for r in rows if str(r.get("n_permeations", "")).strip() != ""]
    if len(pts) < 2:
        print("    [skip] not enough runs with a permeation count")
        return
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])

    fig, ax = plt.subplots(figsize=(6.6, 5))
    ax.scatter(x, y, s=60, color="#2980B9", alpha=0.85, edgecolors="black",
               linewidths=0.6, zorder=3)
    for xi, yi, lb in pts:
        ax.annotate(lb, (xi, yi), fontsize=6.5, alpha=0.75,
                    xytext=(3, 3), textcoords="offset points")
    rho = float("nan")
    if len(x) >= 3 and np.unique(x).size > 1:
        try:
            from scipy import stats
            rho, p = stats.spearmanr(x, y)
            sub = f"Spearman ρ = {rho:.3f} (p = {p:.3g}),  n = {len(x)} runs"
        except Exception:
            sub = f"n = {len(x)} runs"
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "--", color="black", linewidth=1.2,
                alpha=0.7, zorder=2)
    else:
        sub = f"n = {len(x)} runs"
    ax.set_xlabel("% of frames with at least one gate narrow", fontsize=11)
    ax.set_ylabel("Number of permeations", fontsize=11)
    ax.set_title(f"{title}\n{sub}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")



def _sorted_by_perms(rows):
    """Runs ordered by permeation count, blanks last — the order every
    per-run figure uses, so they can be read side by side."""
    def key(r):
        v = str(r.get("n_permeations", "")).strip()
        return (0, float(v)) if v else (1, 0.0)
    return sorted(rows, key=key)


def _perm_tag(r):
    v = str(r.get("n_permeations", "")).strip()
    return f"{r['run']}  ({v})" if v else f"{r['run']}  (–)"


def plot_run_heatmap(rows, threshold, out_path, title, definition):
    """
    Runs x gates, cell = % of frames that gate was narrow, number printed in
    every cell. Rows ordered by permeation count so any trend down the
    column would be visible by eye.
    """
    rows = _sorted_by_perms(rows)
    cols = [f"frac_narrow_{g}" for g in GATE_ORDER] + ["frac_any_narrow"]
    labels = [g.upper().replace("_", " ") for g in GATE_ORDER] + ["ANY gate"]
    if definition == "diagonal":
        cols = ([f"frac_narrow_{g}_{dg}" for g in GATE_ORDER for dg in DIAGONALS]
                + [f"frac_narrow_{g}" for g in GATE_ORDER] + ["frac_any_narrow"])
        labels = ([f"{g.replace('_gl','GL').upper()}\n{dg}"
                   for g in GATE_ORDER for dg in DIAGONALS]
                  + [f"{g.upper().replace('_', ' ')}\n(worst)" for g in GATE_ORDER]
                  + ["ANY gate"])

    mat = np.array([[float(r.get(c, 0) or 0) * 100 for c in cols] for r in rows])
    h = max(4.0, 0.34 * len(rows) + 2.0)
    w = max(7.0, 0.95 * len(cols) + 3.0)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 60 else "black")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_perm_tag(r) for r in rows], fontsize=8)
    ax.set_ylabel("Run  (permeations)", fontsize=10)
    # separate the per-diagonal block from the per-gate block
    if definition == "diagonal":
        ax.axvline(len(GATE_ORDER) * len(DIAGONALS) - 0.5, color="black",
                   linewidth=2)
    ax.axvline(len(cols) - 1.5, color="black", linewidth=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(f"% of frames below {threshold} Å", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_run_bars(rows, threshold, out_path, title):
    """Horizontal grouped bars, one group per run, one bar per gate."""
    rows = _sorted_by_perms(rows)
    y = np.arange(len(rows))
    height = 0.8 / (len(GATE_ORDER) + 1)

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(rows) + 1.8)))
    for k, gate in enumerate(GATE_ORDER + ["any"]):
        col = "frac_any_narrow" if gate == "any" else f"frac_narrow_{gate}"
        vals = [float(r.get(col, 0) or 0) * 100 for r in rows]
        color = "#555555" if gate == "any" else GATE_COLOR[gate]
        pos = y + k * height - 0.4 + height / 2
        ax.barh(pos, vals, height * 0.9, color=color, alpha=0.88,
                edgecolor="black", linewidth=0.4,
                label="ANY gate" if gate == "any" else gate)
        for p, v in zip(pos, vals):
            if v >= 1:
                ax.text(v + 0.6, p, f"{v:.0f}", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels([_perm_tag(r) for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"% of frames below {threshold} Å", fontsize=11)
    ax.set_ylabel("Run  (permeations)", fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, ncol=4, loc="lower right")
    ax.grid(True, alpha=0.25, axis="x")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_diagonal_asymmetry(rows, threshold, out_path, title):
    """A_C against B_D per run, one panel per gate. Off-diagonal = asymmetric."""
    rows = _sorted_by_perms(rows)
    fig, axes = plt.subplots(1, len(GATE_ORDER), figsize=(14, 4.8))
    for ax, gate in zip(np.atleast_1d(axes), GATE_ORDER):
        a = np.array([float(r.get(f"frac_narrow_{gate}_A_C", 0) or 0) * 100
                      for r in rows])
        b = np.array([float(r.get(f"frac_narrow_{gate}_B_D", 0) or 0) * 100
                      for r in rows])
        top = max(1.0, a.max(), b.max()) * 1.12
        ax.plot([0, top], [0, top], "--", color="grey", linewidth=1,
                label="symmetric")
        ax.scatter(a, b, s=55, color=GATE_COLOR[gate], alpha=0.85,
                   edgecolors="black", linewidths=0.5, zorder=3)
        for ai, bi, r in zip(a, b, rows):
            ax.annotate(r["run"], (ai, bi), fontsize=6.5, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
        n_ac = int((a > b).sum())
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel("% frames A_C narrow", fontsize=10)
        ax.set_title(f"{gate}\nA_C limiting in {n_ac}/{len(rows)} runs",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("% frames B_D narrow", fontsize=10)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


def plot_distributions(loaded_by_run, threshold, out_path, title):
    fig, axes = plt.subplots(1, len(GATE_ORDER), figsize=(14, 4.4))
    for ax, gate in zip(np.atleast_1d(axes), GATE_ORDER):
        mean_v = np.concatenate([d[gate]["mean"] for d in loaded_by_run])
        worst = np.concatenate([np.minimum(d[gate]["A_C"], d[gate]["B_D"])
                                for d in loaded_by_run])
        bins = np.linspace(min(mean_v.min(), worst.min()),
                           max(mean_v.max(), worst.max()), 60)
        ax.hist(mean_v, bins=bins, alpha=0.55, color=GATE_COLOR[gate],
                edgecolor="none", label="mean of diagonals")
        ax.hist(worst, bins=bins, alpha=0.45, color="#444444",
                edgecolor="none", label="narrowest diagonal")
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.4,
                   label=f"threshold {threshold} Å")
        ax.set_title(gate, fontsize=11, fontweight="bold")
        ax.set_xlabel("Gate distance (Å)", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Frames", fontsize=10)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {Path(out_path).name}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", help="Channel folder holding RUN*/")
    ap.add_argument("-channel", "--channel", dest="channel", default="",
                    help="Channel label for file names and titles")
    ap.add_argument("-threshold", "--threshold", type=float,
                    default=DEFAULT_THRESHOLD,
                    help=f"Gate distance below which a gate is narrow, in A "
                         f"(default {DEFAULT_THRESHOLD}; published value 5.69)")
    ap.add_argument("-runs", "--runs", default="",
                    help="Only these runs, e.g. --runs 1-20 or --runs 3,5,14")
    ap.add_argument("-perm_col", "--perm_col", default=DEFAULT_PERM_COL,
                    help=f"Column of the permeation summary to use as "
                         f"n_permeations (default {DEFAULT_PERM_COL})")
    ap.add_argument("-out", "--out", default=None,
                    help=f"Output directory "
                         f"(default <base_dir>/{OUT_SUBDIR})")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    ch = args.channel or base.name

    print("=" * 70)
    print("GATE OPENNESS")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel: {ch}   threshold: {args.threshold} A")

    runs = expand_runs(args.runs) if args.runs else None
    run_dirs = find_run_dirs(base, runs)
    if not run_dirs:
        sys.exit(f"ERROR: no runs with gate files found under {base}")
    run_dirs.sort(key=lambda t: natural_key(t[0]))
    print(f"  Found {len(run_dirs)} run(s)")

    perms, perm_col = load_permeations(base, args.perm_col)

    mean_rows, diag_rows, loaded_all = [], [], []
    for run, gdir in run_dirs:
        m, d, loaded = analyse_run(run, gdir, args.threshold)
        if m is None:
            continue
        mean_rows.append(attach_permeations(m, perms, perm_col))
        diag_rows.append(attach_permeations(d, perms, perm_col))
        loaded_all.append(loaded)
        print(f"    {run:<8} {m['n_frames']:>6,} frames   "
              f"any-narrow: mean {m['frac_any_narrow']*100:5.1f}%  "
              f"diagonal {d['frac_any_narrow']*100:5.1f}%   "
              f"perms {m['n_permeations']}")
    if not mean_rows:
        sys.exit("ERROR: no runs could be analysed.")

    n_with = sum(1 for r in mean_rows
                 if str(r.get("n_permeations", "")).strip() != "")
    if n_with == 0:
        print(f"\n  [WARN] no run got a permeation count — the per-run labels "
              f"will read '(–)' and the permeation scatters are skipped.\n"
              f"         Pass --perm_col with one of the columns listed above.")
    elif n_with < len(mean_rows):
        missing = [r["run"] for r in mean_rows
                   if str(r.get("n_permeations", "")).strip() == ""]
        print(f"\n  [warn] no permeation count for: {', '.join(missing)}")

    sanity_check(loaded_all, args.threshold)

    out_dir = Path(args.out) if args.out else base / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    write_csv(mean_rows, out_dir / f"gate_openness_mean_{ch}.csv")
    write_csv(diag_rows, out_dir / f"gate_openness_diagonal_{ch}.csv")
    write_csv(pooled_summary(mean_rows, diag_rows, ch, args.threshold),
              out_dir / f"gate_openness_summary_{ch}.csv")

    for label, rows in (("mean", mean_rows), ("diagonal", diag_rows)):
        note = ("mean of the two diagonals" if label == "mean"
                else "narrowest diagonal")
        plot_run_heatmap(rows, args.threshold,
                         out_dir / f"gate_openness_heatmap_{ch}_{label}.png",
                         f"{ch} — % of frames each gate was below "
                         f"{args.threshold} Å ({note})", label)
        plot_run_bars(rows, args.threshold,
                      out_dir / f"gate_openness_bars_{ch}_{label}.png",
                      f"{ch} — narrow-gate time per run ({note})")
    plot_diagonal_asymmetry(diag_rows, args.threshold,
                            out_dir / f"gate_openness_asymmetry_{ch}.png",
                            f"{ch} — HBC/G-loop diagonal asymmetry "
                            f"(threshold {args.threshold} Å)")

    plot_vs_permeations(mean_rows, out_dir / f"gate_openness_{ch}_mean.png",
                        f"{ch} — permeations vs closed time (mean of diagonals)")
    plot_vs_permeations(diag_rows, out_dir / f"gate_openness_{ch}_diagonal.png",
                        f"{ch} — permeations vs closed time (narrowest diagonal)")
    plot_distributions(loaded_all, args.threshold,
                       out_dir / f"gate_openness_distributions_{ch}.png",
                       f"{ch} — gate distance distributions "
                       f"({len(loaded_all)} runs)")

    print(f"\n  {'definition':<10} {'scope':<18} {'frac narrow':>12}")
    for r in pooled_summary(mean_rows, diag_rows, ch, args.threshold):
        print(f"  {r['definition']:<10} {r['scope']:<18} "
              f"{r['frac_narrow']*100:>11.1f}%")

    print(f"\n✓ Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()

'''
 python3 gate_openness.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12/ \
--channel G12

 python3 gate_openness.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--channel G2

 python3 gate_openness.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G2/ \
--channel G2

 python3 gate_openness.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_D173N/ \
--channel D173N

 python3 gate_openness.py /media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/G12_E141Q/ \
--channel E141Q
'''