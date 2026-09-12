"""
ion_cavity_analysis.py
======================
Single script that analyses ion permeation events (R1_end → R3_end) and
produces ALL axial-motion and cavity-contact results in one pass.

One loop per event, one folder for all outputs.

Axis:
  origin = COM of lowest atom per sf_low_res residue  (d = 0)
  end    = COM of hbc_residues                         (d = hbc_d)
  unit   = (end - origin) / |end - origin|

Per frame the ion is characterised by:
  d  -- axial projection onto SF→HBC axis

From d:
  COMPARTMENT (axial zone, no cutoff):
    GLU/ASN cavity   glu_d <= d < asn_d
    ASN/HBC cavity   asn_d <= d < hbc_d

  CLOSEST GROUP (no cutoff):
    whichever of GLU / ASN / HBC has the smallest min-distance to the ion

Outputs  (saved to results_dir/ion_cavity_analysis/):
  axial_traces.csv      frame-by-frame d for every event
  axial_summary.csv     per-event mean/std/min/max d
  contacts_per_event.csv  per-event zone + closest-group times (frames, ns, %)
  contacts_summary.csv    pooled mean +/- std across all events
  axial_traces.png      normalised trajectory plot (SF top, HBC bottom)
  axial_pmf.png         PMF along channel axis
  zones_bar_ns.png      compartment time in ns
  zones_bar_pct.png     compartment time in %
  closest_bar_ns.png    closest-group time in ns
  closest_bar_pct.png   closest-group time in %

Called from main.py:
    from analysis.ion_cavity_analysis import run_ion_cavity_analysis
    run_ion_cavity_analysis(
        u                   = u,
        permeation_events   = analyzer.permeation_events,
        sf_low_res_residues = sf_low_res_residues,
        hbc_residues        = hbc_residues,
        glu_residues        = glu_residues,
        asn_residues        = asn_residues,
        results_dir         = results_dir,
        frame_duration      = 0.075,
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


# ── physical constants ────────────────────────────────────────────────────────

kB   = 0.001987   # kcal / (mol * K)
TEMP = 310.0      # K
kT   = kB * TEMP


# ── zone / group labels ───────────────────────────────────────────────────────

ZONE_NAMES  = ["GLU/ASN cavity", "ASN/HBC cavity"]
ZONE_KEYS   = ["glu_asn",        "asn_hbc"]
ZONE_COLORS = ["#E67E22",        "#2980B9"]

CLOSE_NAMES  = ["Closest to GLU", "Closest to ASN", "Closest to HBC"]
CLOSE_KEYS   = ["close_glu",      "close_asn",      "close_hbc"]
CLOSE_COLORS = ["#C0392B",        "#E67E22",         "#2C3E50"]


# ── geometry ──────────────────────────────────────────────────────────────────

def _com(u, residues):
    return u.select_atoms(
        "resid " + " ".join(str(r) for r in residues)
    ).center_of_mass()


def _sf_lower_com(u, sf_low_res_residues):
    """COM of the lowest atom per SF residue."""
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
    origin = _sf_lower_com(u, sf_low_res_residues)
    end    = _com(u, hbc_residues)
    vec    = end - origin
    return origin, vec / np.linalg.norm(vec)


def _project(pos, origin, unit):
    return float(np.dot(pos - origin, unit))


def _ion_pos(u, ion_id):
    sel = u.select_atoms(f"resname K+ K and resid {ion_id}")
    return sel.positions[0].copy() if sel.n_atoms > 0 else None


def _min_dist(u, ion_pos, residues):
    """Minimum distance from ion_pos to any atom in any residue in the list."""
    min_d = np.inf
    for resid in residues:
        atoms = u.select_atoms(f"resid {resid}")
        if atoms.n_atoms == 0:
            continue
        d = float(np.min(np.linalg.norm(atoms.positions - ion_pos, axis=1)))
        if d < min_d:
            min_d = d
    return min_d


def _assign_zone(d, glu_d, asn_d, hbc_d):
    """Axial zone — returns key or None if outside cavity."""
    if glu_d <= d < asn_d:
        return "glu_asn"
    elif asn_d <= d < hbc_d:
        return "asn_hbc"
    return None


def _assign_closest(u, ion_pos, glu_residues, asn_residues, hbc_residues):
    """Assign to whichever group (glu/asn/hbc) is closest — no cutoff."""
    d_glu = _min_dist(u, ion_pos, glu_residues)
    d_asn = _min_dist(u, ion_pos, asn_residues)
    d_hbc = _min_dist(u, ion_pos, hbc_residues)
    m = min(d_glu, d_asn, d_hbc)
    if m == d_glu:   return "close_glu"
    elif m == d_asn: return "close_asn"
    else:            return "close_hbc"


# ── landmark computation ──────────────────────────────────────────────────────

def _compute_landmarks(u, all_frames, glu_residues, asn_residues,
                        sf_low_res_residues, hbc_residues):
    """Mean axial projection of GLU/ASN/HBC COM over sampled frames."""
    sample = all_frames[::max(1, len(all_frames) // 500)]
    glu_v, asn_v, hbc_v = [], [], []
    for f in tqdm(sample, desc="  computing landmarks", leave=False):
        u.trajectory[f]
        origin, unit = _make_axis(u, sf_low_res_residues, hbc_residues)
        glu_v.append(_project(_com(u, glu_residues), origin, unit))
        asn_v.append(_project(_com(u, asn_residues), origin, unit))
        hbc_v.append(_project(_com(u, hbc_residues), origin, unit))
    return float(np.mean(glu_v)), float(np.mean(asn_v)), float(np.mean(hbc_v))


# ── core loop — one pass per event ───────────────────────────────────────────

def _process_events(u, permeation_events, sf_low_res_residues, hbc_residues,
                    glu_residues, asn_residues,
                    frame_duration, hbc_oscillation_threshold=10.0):
    """
    Loop R1_end → R3_end for every event.

    At every frame:
      - axis (origin, unit) is recomputed from current atom positions
      - glu_d_frame, asn_d_frame, hbc_d_frame are recomputed from current COMs
        so zone and closest-group assignment reflect the actual frame geometry

    Returns traces (for plots), all_d, contact_rows, and
    mean landmark values (for plot dotted lines only).
    """
    traces        = []
    all_d         = []
    contact_rows  = []
    glu_d_all, asn_d_all, hbc_d_all = [], [], []   # for mean landmarks (plots only)

    for idx, ev in enumerate(tqdm(permeation_events, desc="  processing events")):
        ion_id = int(ev["ion_id"])
        r1_end = int(ev.get("R1_end", ev.get("end_1", -1)))
        r3_end = int(ev.get("R3_end", ev.get("end_3", -1)))

        if r1_end < 0 or r3_end < 0 or r3_end <= r1_end:
            continue

        total_frames = r3_end - r1_end + 1
        frames_d       = []
        d_vals         = []
        hbc_frame_vals = []   # per-frame hbc_d_frame values, for oscillation filter
        zone_counts    = {"glu_asn": 0, "asn_hbc": 0}
        close_counts   = {"close_glu": 0, "close_asn": 0, "close_hbc": 0}

        for frame in range(r1_end, r3_end + 1):
            u.trajectory[frame]
            origin, unit = _make_axis(u, sf_low_res_residues, hbc_residues)
            pos = _ion_pos(u, ion_id)
            if pos is None:
                continue

            # ── per-frame landmarks ───────────────────────────────────────
            glu_d_frame = _project(_com(u, glu_residues), origin, unit)
            asn_d_frame = _project(_com(u, asn_residues), origin, unit)
            hbc_d_frame = _project(_com(u, hbc_residues), origin, unit)

            # accumulate for mean (used only for plot dotted lines)
            glu_d_all.append(glu_d_frame)
            asn_d_all.append(asn_d_frame)
            hbc_d_all.append(hbc_d_frame)

            d = _project(pos, origin, unit)
            frames_d.append(frame)
            d_vals.append(d)
            hbc_frame_vals.append(hbc_d_frame)

            # compartment — uses this frame's actual boundaries
            zone = _assign_zone(d, glu_d_frame, asn_d_frame, hbc_d_frame)
            if zone:
                zone_counts[zone] += 1

            # closest group
            close_counts[_assign_closest(u, pos, glu_residues, asn_residues, hbc_residues)] += 1

        if len(d_vals) < 2:
            continue

        # ── oscillation filter ────────────────────────────────────────────
        # Skip events where the ion dips below HBC and then returns, if
        # those "premature" below-HBC frames exceed 10% of the event.
        #
        # Logic:
        #   1. Find last_cavity_idx = last frame where d < hbc_d_frame.
        #      Everything after this is the final exit — allowed.
        #   2. Count frames BEFORE last_cavity_idx where d >= hbc_d_frame.
        #      These are the oscillation frames (went below HBC, came back).
        #   3. If oscillation_frames / total_frames > 10% → skip.
        d_arr   = np.array(d_vals)
        hbc_arr = np.array(hbc_frame_vals)
        in_cavity = d_arr < hbc_arr

        if not np.any(in_cavity):
            continue   # ion never entered cavity

        last_cavity_idx = int(np.where(in_cavity)[0][-1])
        oscillation_frames = int(np.sum(d_arr[:last_cavity_idx] >= hbc_arr[:last_cavity_idx]))
        pct_oscillation = 100.0 * oscillation_frames / len(d_vals)

        if pct_oscillation > hbc_oscillation_threshold:
            print(f"    [SKIP] ion {ion_id} event {idx}: "
                  f"{pct_oscillation:.1f}% oscillation below HBC before final exit")
            continue

        all_d.extend(d_vals)

        # ── trace dict (for plots / axial CSVs) ──────────────────────────
        traces.append({
            "event_idx": idx,
            "ion_id":    ion_id,
            "run":       ev.get("run", ""),
            "r1_end":    r1_end,
            "r3_end":    r3_end,
            "frames":    frames_d,
            "d":         d_vals,
        })

        # ── contact row (for contacts CSVs) ──────────────────────────────
        row = {
            "ion_id":       ion_id,
            "run":          ev.get("run", ""),
            "r1_end":       r1_end,
            "r3_end":       r3_end,
            "total_frames": total_frames,
            "total_ns":     round(total_frames * frame_duration, 3),
        }
        for key in ZONE_KEYS:
            n = zone_counts[key]
            row[f"{key}_frames"] = n
            row[f"{key}_ns"]     = round(n * frame_duration, 3)
            row[f"{key}_pct"]    = round(100.0 * n / total_frames, 2)
        for key in CLOSE_KEYS:
            n = close_counts[key]
            row[f"{key}_frames"] = n
            row[f"{key}_ns"]     = round(n * frame_duration, 3)
            row[f"{key}_pct"]    = round(100.0 * n / total_frames, 2)
        contact_rows.append(row)

    # mean landmarks — used only for plot dotted lines
    mean_glu_d = float(np.mean(glu_d_all)) if glu_d_all else 0.0
    mean_asn_d = float(np.mean(asn_d_all)) if asn_d_all else 0.0
    mean_hbc_d = float(np.mean(hbc_d_all)) if hbc_d_all else 0.0

    return traces, np.array(all_d), contact_rows, mean_glu_d, mean_asn_d, mean_hbc_d


# ── CSV saving ────────────────────────────────────────────────────────────────

def _save_axial_traces_csv(traces, out_path, frame_duration):
    fields = ["event_idx", "ion_id", "frame", "t_ns", "d_A"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tr in traces:
            t0 = tr["r1_end"] * frame_duration
            for frame, d in zip(tr["frames"], tr["d"]):
                w.writerow({
                    "event_idx": tr["event_idx"],
                    "ion_id":    tr["ion_id"],
                    "frame":     frame,
                    "t_ns":      round(frame * frame_duration - t0, 4),
                    "d_A":       round(d, 3),
                })
    print(f"    Saved: {out_path}")


def _save_axial_summary_csv(traces, out_path):
    fields = ["event_idx", "ion_id", "run", "r1_end", "r3_end",
              "n_frames", "mean_d_A", "std_d_A", "min_d_A", "max_d_A"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tr in traces:
            d = np.array(tr["d"])
            w.writerow({
                "event_idx": tr["event_idx"],
                "ion_id":    tr["ion_id"],
                "run":       tr["run"],
                "r1_end":    tr["r1_end"],
                "r3_end":    tr["r3_end"],
                "n_frames":  len(d),
                "mean_d_A":  f"{np.mean(d):.2f}",
                "std_d_A":   f"{np.std(d):.2f}",
                "min_d_A":   f"{np.min(d):.2f}",
                "max_d_A":   f"{np.max(d):.2f}",
            })
    print(f"    Saved: {out_path}")


def _save_contacts_per_event(rows, out_path):
    if not rows:
        return
    fields = (
        ["ion_id", "run", "r1_end", "r3_end", "total_frames", "total_ns"]
        + [f"{k}_{s}" for k in ZONE_KEYS  for s in ("frames", "ns", "pct")]
        + [f"{k}_{s}" for k in CLOSE_KEYS for s in ("frames", "ns", "pct")]
    )
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"    Saved: {out_path}")


def _save_contacts_summary(rows, label, glu_d, asn_d, hbc_d, out_path):
    summary = [
        {"metric": "system",   "mean": label,     "std": "", "min": "", "max": "", "n": len(rows)},
        {"metric": "n_events", "mean": len(rows),  "std": "", "min": "", "max": "", "n": len(rows)},
        {"metric": "glu_d_A",  "mean": f"{glu_d:.2f}", "std": "", "min": "", "max": "", "n": ""},
        {"metric": "asn_d_A",  "mean": f"{asn_d:.2f}", "std": "", "min": "", "max": "", "n": ""},
        {"metric": "hbc_d_A",  "mean": f"{hbc_d:.2f}", "std": "", "min": "", "max": "", "n": ""},
    ]
    for key in ZONE_KEYS + CLOSE_KEYS:
        for suffix in ("ns", "pct"):
            col  = f"{key}_{suffix}"
            vals = [r[col] for r in rows]
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

def _plot_traces(traces, label, out_path, glu_d, asn_d, hbc_d):
    """Normalised trajectory plot — SF at top, HBC at bottom."""
    if not traces:
        return
    N = 200
    grid = np.linspace(0, 1, N)
    fig, ax = plt.subplots(figsize=(10, 6))
    mat = []
    for tr in traces:
        d_ev   = np.array(tr["d"])
        prog   = np.linspace(0, 1, len(d_ev))
        ax.plot(prog, d_ev, color="#4878CF", alpha=0.30, linewidth=0.9, zorder=1)
        mat.append(np.interp(grid, prog, d_ev))
    mat    = np.array(mat)
    mean_d = np.mean(mat, axis=0)
    std_d  = np.std(mat,  axis=0)
    ax.plot(grid, mean_d, color="#C0392B", linewidth=2.4, label="Mean", zorder=3)
    ax.fill_between(grid, mean_d - std_d, mean_d + std_d,
                    color="#C0392B", alpha=0.15, label="+/-1 std", zorder=2)
    ax.axhline(0,     color="#666666", linewidth=1.2, linestyle=":", alpha=0.9, label="SF (0 Å)")
    ax.axhline(glu_d, color="#C0392B", linewidth=1.2, linestyle=":", alpha=0.9, label=f"GLU ({glu_d:.1f} Å)")
    ax.axhline(asn_d, color="#E67E22", linewidth=1.2, linestyle=":", alpha=0.9, label=f"ASN ({asn_d:.1f} Å)")
    ax.axhline(hbc_d, color="#2C3E50", linewidth=1.2, linestyle=":", alpha=0.9, label=f"HBC ({hbc_d:.1f} Å)")
    ax.invert_yaxis()
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%\n(SF exit)", "25%", "50%", "75%", "100%\n(HBC exit)"], fontsize=11)
    ax.set_xlabel("Normalised permeation progress", fontsize=14)
    ax.set_ylabel("Axial position (Å from SF,  ↓ toward HBC)", fontsize=14)
    ax.set_title(f"{label} — Ion axial trajectory  (n = {len(traces)} events)\n"
                 f"SF = 0 Å (top)   HBC = {hbc_d:.1f} Å (bottom)", fontsize=14, pad=10)
    ax.legend(fontsize=10, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    Saved: {out_path}")


def _plot_pmf(all_d, label, out_path, glu_d, asn_d, hbc_d, n_bins=60):
    """PMF along channel axis, clipped to [0, hbc_d], lightly smoothed."""
    if len(all_d) == 0:
        return
    from scipy.ndimage import gaussian_filter1d
    clipped = all_d[(all_d >= 0) & (all_d <= hbc_d)]
    if len(clipped) < 10:
        clipped = all_d
    counts, edges = np.histogram(clipped, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth  = gaussian_filter1d(counts.astype(float), sigma=1.0)
    smooth  = np.where(smooth > 0, smooth, np.nan)
    pmf     = -kT * np.log(smooth)
    pmf    -= np.nanmin(pmf)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, pmf, color="#4878CF", linewidth=2.2, zorder=3)
    ax.fill_between(centers, 0, pmf, alpha=0.15, color="#4878CF", zorder=2)
    ax.axvline(0,     color="#666666", linewidth=1.2, linestyle=":", alpha=0.8, label="SF (d=0)")
    ax.axvline(glu_d, color="#C0392B", linewidth=1.2, linestyle=":", alpha=0.8, label=f"GLU ({glu_d:.1f} Å)")
    ax.axvline(asn_d, color="#E67E22", linewidth=1.2, linestyle=":", alpha=0.8, label=f"ASN ({asn_d:.1f} Å)")
    ax.axvline(hbc_d, color="#2C3E50", linewidth=1.2, linestyle=":", alpha=0.8, label=f"HBC ({hbc_d:.1f} Å)")
    ax.set_xlim(0, hbc_d * 1.02)
    ymax = np.nanmax(pmf[np.isfinite(pmf)]) * 1.05
    ax.set_ylim(bottom=0)
    ax.text(0,     ymax, "SF",  ha="center", va="bottom", fontsize=11, color="#666666")
    ax.text(hbc_d, ymax, "HBC", ha="center", va="bottom", fontsize=11, color="#2C3E50")
    ax.set_xlabel("Axial position (Å from SF,  SF left → HBC right)", fontsize=14)
    ax.set_ylabel("PMF (kcal/mol)", fontsize=14)
    ax.set_title(f"{label} — PMF along channel axis  (n_frames = {len(clipped)})", fontsize=14, pad=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    Saved: {out_path}")


def _bar_plot(rows, label, out_path, mode, keys, names, colors):
    ylabel = "Mean time (ns)" if mode == "ns" else "Mean time (%)"

    all_vals = []
    means, stds = [], []
    for key in keys:
        vals = [r[f"{key}_{mode}"] for r in rows]
        all_vals.append(vals)
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(names))

    ax.bar(x, means, yerr=stds, capsize=7, width=0.55,
           color=colors, alpha=0.80, edgecolor="black", linewidth=0.8,
           error_kw=dict(elinewidth=1.6, ecolor="black"), zorder=2)

    # jittered individual event dots — faint, shadow-like
    rng = np.random.default_rng(42)
    for i, (vals, col) in enumerate(zip(all_vals, colors)):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(i + jitter, vals, color=col, alpha=0.25, s=16,
                   edgecolors="none", zorder=3)

    # annotation: place text above the error bar top, then extend y-axis
    # so there is always room between annotation and title
    annotation_tops = []
    for i, (m, s) in enumerate(zip(means, stds)):
        top = m + s
        txt = ax.text(i, top, f"{m:.2f}\n±{s:.2f}",
                      ha="center", va="bottom", fontsize=11)
        annotation_tops.append(top)

    # set y upper limit = max(annotation top) * 1.25 to give title breathing room
    data_top = max(
        max(max(v) for v in all_vals if v),
        max(m + s for m, s in zip(means, stds))
    )
    ax.set_ylim(bottom=0, top=data_top * 1.35)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"{label}  (n = {len(rows)} events)", fontsize=14, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ── public entry point ────────────────────────────────────────────────────────

def run_ion_cavity_analysis(
    u,
    permeation_events:   list,
    sf_low_res_residues: list,
    hbc_residues:        list,
    glu_residues:        list,
    asn_residues:        list,
    results_dir,
    frame_duration:              float = 0.075,
    channel_type:                str   = "G12",
    pmf_bins:                    int   = 60,
    hbc_oscillation_threshold:   float = 10.0,
):
    """
    Main entry point — call this from main.py.

    Parameters
    ----------
    u                   : MDAnalysis Universe
    permeation_events   : list of dicts  (R1_end / R3_end / ion_id)
    sf_low_res_residues : list of resids for SF bottom ring (axis origin, d=0)
    hbc_residues        : list of resids for HBC gate (axis end)
    glu_residues        : list of resids for GLU (4 subunits)
    asn_residues        : list of resids for ASN (4 subunits)
    results_dir         : Path to RUN-level results directory
    frame_duration      : float, ns per frame (default 0.075)
    channel_type        : str, label for plots and CSVs
    pmf_bins            : int, histogram bins for PMF (default 60)
    hbc_oscillation_threshold : float, % of frames allowed to oscillate below
                                HBC before the event is excluded (default 10.0)
    """
    results_dir = Path(results_dir)
    out_dir = results_dir / "ion_cavity_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = channel_type

    print(f"\n  Ion cavity analysis ({label})")
    print(f"  Events        : {len(permeation_events)}")
    print(f"  Frame range   : R1_end --> R3_end")
    print(f"  Landmarks     : recomputed at every frame (no fixed mean)")
    print(f"  HBC filter    : skip if >{hbc_oscillation_threshold}% frames oscillate below HBC")
    print(f"  Output dir    : {out_dir}")

    if not permeation_events:
        print("  No permeation events — skipping.")
        return

    # ── single pass: axial traces + contact counts ────────────────────────────
    # Landmarks (glu_d, asn_d, hbc_d) are recomputed inside _process_events
    # at every frame so zone assignment always reflects the actual geometry.
    # The returned mean_glu_d / mean_asn_d / mean_hbc_d are used ONLY for
    # plot dotted lines (visual reference).
    traces, all_d, contact_rows, mean_glu_d, mean_asn_d, mean_hbc_d = _process_events(
        u, permeation_events,
        sf_low_res_residues, hbc_residues, glu_residues, asn_residues,
        frame_duration, hbc_oscillation_threshold
    )

    if not traces:
        print("  No valid events processed.")
        return

    print(f"  Input events  : {len(permeation_events)}")
    print(f"  Valid events  : {len(traces)}  (skipped {len(permeation_events) - len(traces)} with >{hbc_oscillation_threshold}% oscillation below HBC)")
    print(f"  Mean landmarks (for plot reference only):")
    print(f"    GLU COM: {mean_glu_d:.2f} Å")
    print(f"    ASN COM: {mean_asn_d:.2f} Å")
    print(f"    HBC COM: {mean_hbc_d:.2f} Å")
    print()

    # ── terminal summary ──────────────────────────────────────────────────────
    print("  COMPARTMENTS (axial position):")
    for key, name in zip(ZONE_KEYS, ZONE_NAMES):
        ns_v  = [r[f"{key}_ns"]  for r in contact_rows]
        pct_v = [r[f"{key}_pct"] for r in contact_rows]
        print(f"    {name:<22}  {np.mean(ns_v):6.2f} +/- {np.std(ns_v):.2f} ns  "
              f"({np.mean(pct_v):5.1f} +/- {np.std(pct_v):.1f} %)")
    print("  CLOSEST GROUP (no cutoff):")
    for key, name in zip(CLOSE_KEYS, CLOSE_NAMES):
        ns_v  = [r[f"{key}_ns"]  for r in contact_rows]
        pct_v = [r[f"{key}_pct"] for r in contact_rows]
        print(f"    {name:<22}  {np.mean(ns_v):6.2f} +/- {np.std(ns_v):.2f} ns  "
              f"({np.mean(pct_v):5.1f} +/- {np.std(pct_v):.1f} %)")
    print()

    # ── save CSVs ─────────────────────────────────────────────────────────────
    _save_axial_traces_csv(traces,  out_dir / "axial_traces.csv",  frame_duration)
    _save_axial_summary_csv(traces, out_dir / "axial_summary.csv")
    _save_contacts_per_event(contact_rows, out_dir / "contacts_per_event.csv")
    _save_contacts_summary(contact_rows, label, mean_glu_d, mean_asn_d, mean_hbc_d,
                           out_dir / "contacts_summary.csv")

    # ── plots ─────────────────────────────────────────────────────────────────
    _plot_traces(traces, label, out_dir / "axial_traces.png", mean_glu_d, mean_asn_d, mean_hbc_d)
    _plot_pmf(all_d, label, out_dir / "axial_pmf.png", mean_glu_d, mean_asn_d, mean_hbc_d, pmf_bins)

    _bar_plot(contact_rows, f"{label} — Cavity compartment time",
              out_dir / "zones_bar_ns.png",  mode="ns",
              keys=ZONE_KEYS, names=ZONE_NAMES, colors=ZONE_COLORS)
    _bar_plot(contact_rows, f"{label} — Cavity compartment time",
              out_dir / "zones_bar_pct.png", mode="pct",
              keys=ZONE_KEYS, names=ZONE_NAMES, colors=ZONE_COLORS)

    _bar_plot(contact_rows, f"{label} — Time closest to each residue group",
              out_dir / "closest_bar_ns.png",  mode="ns",
              keys=CLOSE_KEYS, names=CLOSE_NAMES, colors=CLOSE_COLORS)
    _bar_plot(contact_rows, f"{label} — Time closest to each residue group",
              out_dir / "closest_bar_pct.png", mode="pct",
              keys=CLOSE_KEYS, names=CLOSE_NAMES, colors=CLOSE_COLORS)

    print(f"  Done. All outputs in: {out_dir}")