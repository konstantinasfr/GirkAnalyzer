"""
ion_ion_permeation_analysis.py
==============================
At the moment of permeation (frame = R2_end, last frame the permeating ion
is in the cavity), compute ion-ion distances and forces.

WHICH IONS ARE INCLUDED:
  ALL K+ ions that are inside channel1 (SF / selectivity filter) or
  channel2 (cavity) at that frame — no distance cutoff.
  The permeating ion itself is excluded.

DISTANCES from the permeating ion to the included ions:
  mean_dist_all      mean distance to ALL included ions (SF + cavity)
  mean_dist_sf       mean distance to SF ions only
  mean_dist_cavity   mean distance to cavity ions only
  min_dist_all       distance to the single closest ion (anywhere)
  dist_closest_above distance to the closest ion that is ABOVE the target
                     on the channel axis (i.e. closer to SF = the knock-on partner)

FORCES on the permeating ion (Coulomb, k=332 kcal·Å/(mol·e²)):
  Forces are vectors and are ADDITIVE — we sum the force vectors per group
  and report:
    magnitude  = |F_total|          (kcal/(mol·Å))
    axial      = dot(F_total, axis) (kcal/(mol·Å), positive = toward HBC)

  force_all_*        from ALL included ions (SF + cavity)
  force_sf_*         from SF ions only
  force_cavity_*     from cavity ions only
  force_closest_*    from the single closest ion above the target
                     (the direct knock-on partner)

Axis convention:
  origin = COM of lowest atom per sf_low_res residue  (d = 0)
  unit   = direction toward HBC COM                   (d > 0 = toward HBC)
  axial force > 0 → ion is pushed toward HBC → favors permeation

Outputs (saved to results_dir/ion_ion_permeation/):
  per_event.csv         one row per permeation event, all metrics
  summary.csv           mean ± std / min / max across all events
  debug_examples.txt    first 3 events with full ion listing (for verification)
  dist_bar.png          bar chart of distance metrics
  dist_hist.png         histograms + KDE for distance metrics
  force_mag_bar.png     bar chart of force magnitudes
  force_mag_hist.png    histograms + KDE for force magnitudes
  force_axial_bar.png   bar chart of axial forces
  force_axial_hist.png  histograms + KDE for axial forces

Called from main.py:
    from analysis.ion_ion_permeation_analysis import run_ion_ion_permeation_analysis
    run_ion_ion_permeation_analysis(
        u                   = u,
        permeation_events   = analyzer.permeation_events,
        channel1            = ch1,
        channel2            = ch2,
        sf_low_res_residues = sf_low_res_residues,
        hbc_residues        = hbc_residues,
        results_dir         = results_dir,
        channel_type        = args.channel_type,
    )
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Coulomb constant in MD units: kcal·Å / (mol·e²)
K_COULOMB = 332.0
Q_K       = 1.0   # K+ charge in elementary units


# ── geometry helpers ──────────────────────────────────────────────────────────

def _com(u, residues):
    return u.select_atoms(
        "resid " + " ".join(str(r) for r in residues)
    ).center_of_mass()


def _sf_lower_com(u, sf_low_res_residues):
    """COM of the lowest atom per SF residue — axis origin."""
    indices = []
    for resid in sf_low_res_residues:
        sel = u.select_atoms(f"resid {resid}")
        if sel.n_atoms == 0:
            continue
        indices.append(sel[int(np.argmin(sel.positions[:, 2]))].index)
    if not indices:
        return _com(u, sf_low_res_residues)
    return u.atoms[indices].center_of_mass()


def _make_axis(u, sf_low_res_residues, hbc_residues):
    """Return (origin, unit_vector) for the current frame."""
    origin = _sf_lower_com(u, sf_low_res_residues)
    end    = _com(u, hbc_residues)
    vec    = end - origin
    return origin, vec / np.linalg.norm(vec)


def _axial_proj(pos, origin, unit):
    """d=0 at SF origin, d>0 toward HBC."""
    return float(np.dot(pos - origin, unit))


# ── Coulomb force ─────────────────────────────────────────────────────────────

def _coulomb_force_vec(pos_target, pos_other):
    """
    Coulomb force vector ON the target ion FROM the other ion.
    F = K * q1*q2 / r^2 * r_hat
    r_hat points FROM other TO target (both K+, so repulsion away from other).
    Units: kcal / (mol·Å)
    """
    r_vec = pos_target - pos_other
    r     = np.linalg.norm(r_vec)
    if r < 0.1:
        return np.zeros(3)
    return K_COULOMB * Q_K * Q_K / r**2 * (r_vec / r)


# ── per-event analysis ────────────────────────────────────────────────────────

def _analyze_event(u, ev, channel1, channel2,
                   sf_low_res_residues, hbc_residues,
                   untracked_by_frame=None, debug=False):
    """
    Analyse one permeation event at R2_end (last frame in GLU/ASN cavity).

    Ions included:
      channel1 (SF)   → sf_ions
      channel2 (GLU/ASN cavity) → cavity_ions
      untracked_by_frame[r2_end] → also added to cavity_ions
        (ions that entered the cavity from a non-standard route,
         not captured by the cylinder check)

    ch3 is NOT included — the permeation happens at the end of ch2,
    so ch3 (ASN→HBC) is not relevant here.

    Returns (row_dict, debug_str | None)
    """
    ion_id = int(ev["ion_id"])
    r2_end = int(ev.get("R2_end", ev.get("end_2", -1)))
    if r2_end < 0:
        return None, None

    # ── go to permeation frame ────────────────────────────────────────────
    u.trajectory[r2_end]
    channel1.compute_geometry(1)
    channel2.compute_geometry(2)

    # ── axis for this frame ───────────────────────────────────────────────
    origin, unit = _make_axis(u, sf_low_res_residues, hbc_residues)

    # ── target ion ────────────────────────────────────────────────────────
    target_sel = u.select_atoms(f"resname K+ K and resid {ion_id}")
    if target_sel.n_atoms == 0:
        return None, None
    target_pos = target_sel.positions[0].copy()
    target_d   = _axial_proj(target_pos, origin, unit)

    # ── classify ALL other K+ ions via cylinder checks ────────────────────
    all_ions_sel = u.select_atoms("resname K+ K")
    sf_ions, cavity_ions = [], []
    seen_resids = {ion_id}

    for ion in all_ions_sel:
        if ion.resid == ion_id:
            continue
        pos  = ion.position.copy()
        dist = float(np.linalg.norm(pos - target_pos))
        d_ax = _axial_proj(pos, origin, unit)

        if channel1.is_within_cylinder(pos):
            sf_ions.append((ion.resid, pos, dist, d_ax))
            seen_resids.add(ion.resid)
        elif channel2.is_within_cylinder(pos):
            cavity_ions.append((ion.resid, pos, dist, d_ax))
            seen_resids.add(ion.resid)

    # ── add untracked ions not yet captured ───────────────────────────────
    n_untracked_added = 0
    if untracked_by_frame and r2_end in untracked_by_frame:
        for resid in untracked_by_frame[r2_end]:
            if resid in seen_resids:
                continue   # already captured by cylinder check
            ion_sel = u.select_atoms(f"resname K+ K and resid {resid}")
            if ion_sel.n_atoms == 0:
                continue
            pos  = ion_sel.positions[0].copy()
            dist = float(np.linalg.norm(pos - target_pos))
            d_ax = _axial_proj(pos, origin, unit)
            cavity_ions.append((resid, pos, dist, d_ax))
            seen_resids.add(resid)
            n_untracked_added += 1

    all_nearby = sf_ions + cavity_ions
    n_sf      = len(sf_ions)
    n_cavity  = len(cavity_ions)
    n_all     = len(all_nearby)

    # ── distance metrics ──────────────────────────────────────────────────
    def _mean_d(ions):
        return float(np.mean([d for _, _, d, _ in ions])) if ions else np.nan

    def _min_d(ions):
        return float(np.min([d for _, _, d, _ in ions])) if ions else np.nan

    mean_dist_all    = _mean_d(all_nearby)
    mean_dist_sf     = _mean_d(sf_ions)
    mean_dist_cavity = _mean_d(cavity_ions)
    min_dist_all     = _min_d(all_nearby)

    # "above" = lower d value = closer to SF = these push the target toward HBC
    above_ions = [(rid, pos, dist, d_ax) for rid, pos, dist, d_ax in all_nearby
                  if d_ax < target_d]
    if above_ions:
        # closest above = the direct knock-on partner
        closest_above = min(above_ions, key=lambda x: x[2])
        dist_closest_above = closest_above[2]
        f_closest = _coulomb_force_vec(target_pos, closest_above[1])
    else:
        dist_closest_above = np.nan
        f_closest = np.zeros(3)

    # ── force vectors (sum over group) ────────────────────────────────────
    def _total_force(ions):
        f = np.zeros(3)
        for _, pos, _, _ in ions:
            f += _coulomb_force_vec(target_pos, pos)
        return f

    f_all    = _total_force(all_nearby)
    f_sf     = _total_force(sf_ions)
    f_cavity = _total_force(cavity_ions)

    def _mag(f):   return float(np.linalg.norm(f))
    def _ax(f):    return float(np.dot(f, unit))
    def _r(v):     return round(v, 3) if not np.isnan(v) else None

    row = {
        "ion_id":              ion_id,
        "run":                 ev.get("run", ""),
        "r2_end":              r2_end,
        "n_sf":                n_sf,
        "n_cavity":            n_cavity,
        "n_untracked_added":   n_untracked_added,
        "n_all":               n_all,
        # distances
        "mean_dist_all":       _r(mean_dist_all),
        "mean_dist_sf":        _r(mean_dist_sf),
        "mean_dist_cavity":    _r(mean_dist_cavity),
        "min_dist_all":        _r(min_dist_all),
        "dist_closest_above":  _r(dist_closest_above),
        # force magnitudes (always a number, 0 if no ions)
        "force_all_mag":       round(_mag(f_all),     3),
        "force_sf_mag":        round(_mag(f_sf),      3),
        "force_cavity_mag":    round(_mag(f_cavity),  3),
        "force_closest_mag":   round(_mag(f_closest), 3),
        # axial forces (positive = toward HBC = favors permeation)
        "force_all_axial":     round(_ax(f_all),      3),
        "force_sf_axial":      round(_ax(f_sf),       3),
        "force_cavity_axial":  round(_ax(f_cavity),   3),
        "force_closest_axial": round(_ax(f_closest),  3),
    }

    # ── debug text ────────────────────────────────────────────────────────
    dbg = None
    if debug:
        lines = [
            f"{'─'*60}",
            f"  Ion {ion_id}  |  frame {r2_end}  |  axial d = {target_d:.2f} Å",
            f"  Target position: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})",
            f"",
            f"  SF ions (channel1)  — n = {n_sf}",
        ]
        for rid, pos, dist, d_ax in sorted(sf_ions, key=lambda x: x[2]):
            lines.append(f"    resid {rid:>6}  dist={dist:6.2f} Å  d_ax={d_ax:6.2f} Å"
                         f"  ({'above' if d_ax < target_d else 'below'} target)")
        lines += [f"", f"  Cavity ions (channel2+3)  — n = {n_cavity - n_untracked_added}"]
        for rid, pos, dist, d_ax in sorted(cavity_ions[:n_cavity - n_untracked_added],
                                           key=lambda x: x[2]):
            lines.append(f"    resid {rid:>6}  dist={dist:6.2f} Å  d_ax={d_ax:6.2f} Å"
                         f"  ({'above' if d_ax < target_d else 'below'} target)")
        lines += [f"", f"  Untracked ions added from CSV  — n = {n_untracked_added}"]
        for rid, pos, dist, d_ax in sorted(cavity_ions[n_cavity - n_untracked_added:],
                                           key=lambda x: x[2]):
            lines.append(f"    resid {rid:>6}  dist={dist:6.2f} Å  d_ax={d_ax:6.2f} Å"
                         f"  ({'above' if d_ax < target_d else 'below'} target)  [untracked]")
        lines += [
            f"",
            f"  Closest ion above  : resid {closest_above[0] if above_ions else 'none'}"
            f"  dist = {dist_closest_above:.2f} Å" if not np.isnan(dist_closest_above) else
            f"  Closest ion above  : none",
            f"",
            f"  force_all_axial  = {_ax(f_all):+.3f}  kcal/(mol·Å)",
            f"  force_sf_axial   = {_ax(f_sf):+.3f}  kcal/(mol·Å)",
            f"  force_cavity_axial = {_ax(f_cavity):+.3f}  kcal/(mol·Å)",
        ]
        dbg = "\n".join(lines)

    return row, dbg


# ── CSV saving ────────────────────────────────────────────────────────────────

FIELDS = [
    "ion_id", "run", "r2_end", "n_sf", "n_cavity", "n_untracked_added", "n_all",
    "mean_dist_all", "mean_dist_sf", "mean_dist_cavity",
    "min_dist_all", "dist_closest_above",
    "force_all_mag",     "force_sf_mag",     "force_cavity_mag",     "force_closest_mag",
    "force_all_axial",   "force_sf_axial",   "force_cavity_axial",   "force_closest_axial",
]


def _save_per_event(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"    Saved: {out_path}")


def _save_summary(rows, label, out_path):
    numeric = [f for f in FIELDS if f not in ("ion_id", "run", "r2_end")]
    summary = [
        {"metric": "system",   "mean": label,    "std": "", "min": "", "max": "", "n": len(rows)},
        {"metric": "n_events", "mean": len(rows), "std": "", "min": "", "max": "", "n": len(rows)},
    ]
    for col in numeric:
        vals = [r[col] for r in rows if r.get(col) is not None]
        if not vals:
            continue
        summary.append({
            "metric": col,
            "mean":   f"{np.mean(vals):.3f}",
            "std":    f"{np.std(vals):.3f}",
            "min":    f"{np.min(vals):.3f}",
            "max":    f"{np.max(vals):.3f}",
            "n":      len(vals),
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "min", "max", "n"])
        w.writeheader()
        w.writerows(summary)
    print(f"    Saved: {out_path}")


# ── plots ─────────────────────────────────────────────────────────────────────

def _vals(rows, col):
    return [r[col] for r in rows if r.get(col) is not None]


def _bar_plot(rows, metrics, names, label, ylabel, title, out_path, color):
    means, stds, all_vals = [], [], []
    for m in metrics:
        v = _vals(rows, m)
        means.append(np.mean(v) if v else 0)
        stds.append(np.std(v)   if v else 0)
        all_vals.append(v)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 2.5), 6))
    x   = np.arange(len(names))
    rng = np.random.default_rng(42)

    ax.bar(x, means, yerr=stds, capsize=8, width=0.55,
           color=color, alpha=0.75, edgecolor="black", linewidth=0.8,
           error_kw=dict(elinewidth=1.8, ecolor="black"), zorder=2)

    for i, v in enumerate(all_vals):
        jitter = rng.uniform(-0.18, 0.18, size=len(v))
        ax.scatter(i + jitter, v, color=color, alpha=0.25, s=22,
                   edgecolors="none", zorder=3)

    data_top = max((max(v) for v in all_vals if v), default=1)
    data_top = max(data_top, max(m + s for m, s in zip(means, stds)))
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s, f"{m:.2f}\n±{s:.2f}",
                ha="center", va="bottom", fontsize=13)

    ax.set_ylim(bottom=0, top=data_top * 1.40)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title(f"{title}\n{label}  (n = {len(rows)} permeation events)",
                 fontsize=15, pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


def _hist_plot(rows, metrics, names, label, title, out_path, color):
    from scipy.stats import gaussian_kde

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, m, name in zip(axes, metrics, names):
        v = np.array(_vals(rows, m))
        if len(v) == 0:
            ax.set_title(name, fontsize=14)
            continue

        ax.hist(v, bins=max(8, len(v) // 3), color=color, alpha=0.55,
                edgecolor="white", linewidth=0.5, density=True)

        if len(v) >= 3:
            try:
                kde = gaussian_kde(v, bw_method="scott")
                xg  = np.linspace(v.min(), v.max(), 200)
                ax.plot(xg, kde(xg), color=color, linewidth=2.2)
            except Exception:
                pass

        ax.axvline(np.mean(v), color="black", linewidth=1.4,
                   linestyle="--", label=f"mean = {np.mean(v):.2f}")
        ax.set_xlabel(name, fontsize=14)
        ax.set_ylabel("Density", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{title}\n{label}  (n = {len(rows)} events)",
                 fontsize=15, y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ── public entry point ────────────────────────────────────────────────────────

def run_ion_ion_permeation_analysis(
    u,
    permeation_events:   list,
    channel1,
    channel2,
    sf_low_res_residues: list,
    hbc_residues:        list,
    results_dir,
    channel_type: str = "G12",
):
    """
    Main entry point — call this from main.py.

    Parameters
    ----------
    u                   : MDAnalysis Universe
    permeation_events   : list of dicts from analyzer.permeation_events
    channel1            : Channel object for region 1 (SF)
    channel2            : Channel object for region 2 (GLU/ASN cavity)
    sf_low_res_residues : list of resids for SF bottom ring (axis origin, d=0)
    hbc_residues        : list of resids for HBC gate (axis end)
    results_dir         : Path to RUN-level results directory
    channel_type        : str, label for plots and CSVs

    Ions included at the permeation frame (R2_end):
      - ALL K+ inside channel1 (SF)
      - ALL K+ inside channel2 (GLU/ASN cavity)
      - Any untracked ions from untracked_cavity_ions/untracked_ions.csv
    ch3 is NOT included — permeation is defined as exit from ch2.
    """
    results_dir = Path(results_dir)
    out_dir     = results_dir / "ion_ion_permeation"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = channel_type

    print(f"\n  Ion-ion permeation analysis ({label})")
    print(f"  Events       : {len(permeation_events)}")
    print(f"  Frame        : R2_end (last frame the permeating ion is in the cavity)")
    print(f"  Ions included: ALL K+ inside channel1 (SF) and channel2 (GLU/ASN cavity)")
    print(f"                 + untracked ions from untracked_ions.csv if available")
    print(f"  No distance cutoff applied")
    print(f"  Output dir   : {out_dir}")

    if not permeation_events:
        print("  No permeation events — skipping.")
        return

    # ── load untracked ions record (if it exists) ─────────────────────────
    untracked_csv = results_dir / "untracked_cavity_ions" / "untracked_ions.csv"
    untracked_by_frame = {}   # {frame(int): set(resid(int))}
    if untracked_csv.exists():
        import csv as _csv
        with open(untracked_csv, newline="") as f:
            for row in _csv.DictReader(f):
                frame = int(float(row["frame"]))
                resid = int(float(row["ion_resid"]))
                untracked_by_frame.setdefault(frame, set()).add(resid)
        n_untracked_frames = len(untracked_by_frame)
        n_untracked_total  = sum(len(v) for v in untracked_by_frame.values())
        print(f"  Untracked ions : loaded {n_untracked_total} observations "
              f"across {n_untracked_frames} frames from {untracked_csv.name}")
    else:
        print(f"  Untracked ions : no untracked_ions.csv found — only cylinder-detected ions used")

    rows       = []
    debug_txts = []

    for idx, ev in enumerate(tqdm(permeation_events, desc="  analysing permeation frames")):
        want_debug = len(debug_txts) < 3
        row, dbg = _analyze_event(u, ev, channel1, channel2,
                                  sf_low_res_residues, hbc_residues,
                                  untracked_by_frame=untracked_by_frame,
                                  debug=want_debug)
        if row is not None:
            rows.append(row)
            if dbg:
                debug_txts.append(f"=== Event {idx+1} ===\n{dbg}")

    if not rows:
        print("  No valid events — skipping.")
        return

    print(f"  Valid events : {len(rows)}")
    n_events_with_untracked = sum(1 for r in rows if r.get("n_untracked_added", 0) > 0)
    total_untracked_added   = sum(r.get("n_untracked_added", 0) for r in rows)
    print(f"  Events with untracked ions added : {n_events_with_untracked} "
          f"({total_untracked_added} ions total)")

    # ── write debug examples ──────────────────────────────────────────────
    dbg_path = out_dir / "debug_examples.txt"
    with open(dbg_path, "w") as f:
        f.write(f"Ion-ion detection — first {len(debug_txts)} events\n")
        f.write(f"System: {label}\n\n")
        f.write("\n\n".join(debug_txts))
    print(f"\n  Debug examples written to: {dbg_path}")
    print("  Check this file to verify that SF and cavity ions are detected correctly.\n")

    # ── print debug to terminal too ───────────────────────────────────────
    for t in debug_txts:
        print(t)
    print()

    # ── terminal summary ──────────────────────────────────────────────────
    print("  DISTANCES (Å)  — mean across all events ± std:")
    for col, name in [
        ("mean_dist_all",      "Mean dist — all ions (SF+cavity)"),
        ("mean_dist_sf",       "Mean dist — SF ions only"),
        ("mean_dist_cavity",   "Mean dist — cavity ions only"),
        ("min_dist_all",       "Min dist  — closest ion (anywhere)"),
        ("dist_closest_above", "Dist to closest ion above target"),
    ]:
        v = _vals(rows, col)
        if v:
            print(f"    {name:<42}  {np.mean(v):6.2f} ± {np.std(v):.2f} Å")

    print("\n  FORCE MAGNITUDES (kcal/(mol·Å))  — mean ± std:")
    for col, name in [
        ("force_all_mag",     "Total force — all ions (SF+cavity)"),
        ("force_sf_mag",      "Total force — SF ions only"),
        ("force_cavity_mag",  "Total force — cavity ions only"),
        ("force_closest_mag", "Force from closest ion above"),
    ]:
        v = _vals(rows, col)
        if v:
            print(f"    {name:<42}  {np.mean(v):6.2f} ± {np.std(v):.2f}")

    print("\n  AXIAL FORCES (kcal/(mol·Å), +ve = toward HBC = favorable):")
    for col, name in [
        ("force_all_axial",     "Axial force — all ions (SF+cavity)"),
        ("force_sf_axial",      "Axial force — SF ions only"),
        ("force_cavity_axial",  "Axial force — cavity ions only"),
        ("force_closest_axial", "Axial force from closest ion above"),
    ]:
        v = _vals(rows, col)
        if v:
            print(f"    {name:<42}  {np.mean(v):+6.2f} ± {np.std(v):.2f}")
    print()

    # ── save CSVs ─────────────────────────────────────────────────────────
    _save_per_event(rows, out_dir / "per_event.csv")
    _save_summary(rows, label, out_dir / "summary.csv")

    # ── distance plots ────────────────────────────────────────────────────
    dist_m = ["mean_dist_all", "mean_dist_sf", "mean_dist_cavity",
              "min_dist_all", "dist_closest_above"]
    dist_n = ["Mean\n(all)", "Mean\n(SF)", "Mean\n(cavity)",
              "Min\n(closest)", "Closest\nabove"]
    _bar_plot(rows, dist_m, dist_n, label,
              "Distance (Å)",
              "Ion–ion distances at moment of permeation",
              out_dir / "dist_bar.png", color="#4878CF")
    _hist_plot(rows, dist_m, dist_n, label,
               "Distribution of ion–ion distances at moment of permeation",
               out_dir / "dist_hist.png", color="#4878CF")

    # ── force magnitude plots ─────────────────────────────────────────────
    fmag_m = ["force_all_mag", "force_sf_mag", "force_cavity_mag", "force_closest_mag"]
    fmag_n = ["Total\n(SF+cavity)", "SF\nonly", "Cavity\nonly", "Closest\nion above"]
    _bar_plot(rows, fmag_m, fmag_n, label,
              "Coulomb force magnitude (kcal/(mol·Å))",
              "Total Coulomb force magnitude on permeating ion",
              out_dir / "force_mag_bar.png", color="#C0392B")
    _hist_plot(rows, fmag_m, fmag_n, label,
               "Distribution of Coulomb force magnitude on permeating ion",
               out_dir / "force_mag_hist.png", color="#C0392B")

    # ── axial force plots ─────────────────────────────────────────────────
    fax_m = ["force_all_axial", "force_sf_axial", "force_cavity_axial", "force_closest_axial"]
    fax_n = ["Total\n(SF+cavity)", "SF\nonly", "Cavity\nonly", "Closest\nion above"]
    _bar_plot(rows, fax_m, fax_n, label,
              "Axial Coulomb force (kcal/(mol·Å))  [ + = toward HBC ]",
              "Axial ion–ion force on permeating ion at moment of permeation\n"
              "(positive = pushes toward HBC = favors permeation)",
              out_dir / "force_axial_bar.png", color="#E67E22")
    _hist_plot(rows, fax_m, fax_n, label,
               "Distribution of axial ion–ion force at moment of permeation\n"
               "(positive = pushes toward HBC)",
               out_dir / "force_axial_hist.png", color="#E67E22")

    print(f"  Done. All outputs in: {out_dir}")