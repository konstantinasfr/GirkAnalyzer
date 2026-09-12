"""
ion_residue_force_analysis.py
==============================
At the moment of permeation (frame = R2_end), compute the Coulomb force
on the permeating ion from the GLU and ASN residues lining the cavity,
using the partial charges from the AMBER prmtop file.

Force equation (per residue atom):
    F_vec = k * q_K * q_atom / r² * r_hat
    k     = 332.0  kcal·Å / (mol·e²)
    q_K   = +1.0 e  (K+ charge)
    q_atom = partial charge from prmtop (MDAnalysis u.atoms.charges)
    r_hat  = unit vector FROM atom TO permeating ion

Axial convention (same as ion_ion_permeation_analysis):
    origin = SF lower residues COM
    unit   = direction toward HBC
    axial force > 0 → pushes ion toward HBC → favors permeation
    axial force < 0 → pulls ion back toward SF → opposes permeation

    NOTE: GLU is negatively charged → attracts K+ → typically axial < 0
          (opposing permeation — the ion must overcome this attraction)

Groups computed:
    F_asn          force from ASN residues only
    F_glu          force from GLU residues only
    F_lig          force from ligand only (only present in G12_ML / G12_GAT)
    F_glu_asn      F_glu + F_asn  (full residue environment)
    F_total        F_glu + F_asn (+ F_lig if present) + F_all_ions
                   (ions read from ion_ion per_event.csv)

For each group: magnitude and axial component are reported.

Outputs (saved to results_dir/ion_residue_force/):
    per_event.csv         one row per permeation event
    summary.csv           mean ± std / min / max
    debug_examples.txt    first 3 events with full atom-level breakdown
    force_mag_bar.png     bar chart of force magnitudes
    force_mag_hist.png    histograms + KDE of force magnitudes
    force_axial_bar.png   bar chart of axial forces
    force_axial_hist.png  histograms + KDE of axial forces

Called from main.py:
    from analysis.ion_residue_force_analysis import run_ion_residue_force_analysis
    run_ion_residue_force_analysis(
        u                   = u,
        permeation_events   = analyzer.permeation_events,
        glu_residues        = glu_residues,
        asn_residues        = asn_residues,
        sf_low_res_residues = sf_low_res_residues,
        hbc_residues        = hbc_residues,
        results_dir         = results_dir,
        channel_type        = args.channel_type,
        lig_resname         = "LIG",   # None / omit for channels with no ligand
    )
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm

K_COULOMB = 332.0   # kcal·Å / (mol·e²)
Q_K       = 1.0     # K+ charge in elementary units


# ── geometry helpers (same as ion_ion_permeation_analysis) ───────────────────

def _com(u, residues):
    return u.select_atoms(
        "resid " + " ".join(str(r) for r in residues)
    ).center_of_mass()


def _sf_lower_com(u, sf_low_res_residues):
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


# ── Coulomb force from a residue group onto the target ion ───────────────────

def _force_from_residues(target_pos, residue_resids, u):
    """
    Total Coulomb force vector ON target_pos FROM all atoms of the given residues.
    Uses partial charges from prmtop via u.atoms.charges.
    Returns (force_vec, n_atoms_used, debug_lines)
    """
    f      = np.zeros(3)
    n_used = 0
    debug  = []

    for resid in residue_resids:
        atoms = u.select_atoms(f"resid {resid}")
        for atom in atoms:
            q = float(atom.charge)
            if abs(q) < 1e-6:
                continue          # skip uncharged atoms (no contribution)
            r_vec = target_pos - atom.position
            r     = np.linalg.norm(r_vec)
            if r < 0.1:
                continue
            f_vec = K_COULOMB * Q_K * q / r**2 * (r_vec / r)
            f    += f_vec
            n_used += 1
            debug.append(
                f"    resid {resid:>6}  atom {atom.name:<4}  "
                f"q={q:+.4f}e  r={r:5.2f}Å  "
                f"f_axial contribution noted"
            )

    return f, n_used, debug


def _force_from_selection(target_pos, atomgroup):
    """
    Same as _force_from_residues but takes an already-built AtomGroup directly
    (used for the ligand, which is selected by resname rather than a resid list).
    Returns (force_vec, n_atoms_used, debug_lines)
    """
    f      = np.zeros(3)
    n_used = 0
    debug  = []

    for atom in atomgroup:
        q = float(atom.charge)
        if abs(q) < 1e-6:
            continue
        r_vec = target_pos - atom.position
        r     = np.linalg.norm(r_vec)
        if r < 0.1:
            continue
        f_vec = K_COULOMB * Q_K * q / r**2 * (r_vec / r)
        f    += f_vec
        n_used += 1
        debug.append(
            f"    resname {atom.resname:>6}  atom {atom.name:<4}  "
            f"q={q:+.4f}e  r={r:5.2f}Å"
        )

    return f, n_used, debug


# ── load ion-ion forces from existing per_event.csv ─────────────────────────

def _load_ion_forces(ion_ion_csv_path):
    """
    Read ion_ion_permeation/per_event.csv.
    Returns dict: {(ion_id, r2_end): force_all_axial, force_all_mag}
    """
    result = {}
    if not Path(ion_ion_csv_path).exists():
        return result
    with open(ion_ion_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(float(row["ion_id"])), int(float(row["r2_end"])))
                result[key] = {
                    "force_all_axial": float(row["force_all_axial"]) if row.get("force_all_axial") else 0.0,
                    "force_all_mag":   float(row["force_all_mag"])   if row.get("force_all_mag")   else 0.0,
                }
            except (ValueError, KeyError):
                continue
    return result


# ── per-event analysis ────────────────────────────────────────────────────────

def _analyze_event(u, ev, glu_residues, asn_residues,
                   sf_low_res_residues, hbc_residues,
                   ion_forces, lig_resname=None, debug=False):
    """
    At R2_end, compute forces from GLU, ASN, and (optionally) LIG on the
    permeating ion. Merges with ion-ion forces from ion_forces dict.
    Returns (row_dict, debug_str | None)
    """
    ion_id = int(ev["ion_id"])
    r2_end = int(ev.get("R2_end", ev.get("end_2", -1)))
    if r2_end < 0:
        return None, None

    # go to permeation frame
    u.trajectory[r2_end]

    # axis
    origin, unit = _make_axis(u, sf_low_res_residues, hbc_residues)

    # permeating ion position
    target_sel = u.select_atoms(f"resname K+ K and resid {ion_id}")
    if target_sel.n_atoms == 0:
        return None, None
    target_pos = target_sel.positions[0].copy()

    # forces from residue groups
    f_asn, n_asn, dbg_asn = _force_from_residues(target_pos, asn_residues,  u)
    f_glu, n_glu, dbg_glu = _force_from_residues(target_pos, glu_residues,  u)
    f_res = f_asn + f_glu   # combined residue force (GLU + ASN)

    # ── ligand (optional — only present in some channel types) ─────────
    has_lig = False
    f_lig, n_lig, dbg_lig = np.zeros(3), 0, []
    if lig_resname:
        lig_sel = u.select_atoms(f"resname {lig_resname}")
        if lig_sel.n_atoms > 0:
            has_lig = True
            f_lig, n_lig, dbg_lig = _force_from_selection(target_pos, lig_sel)

    f_res_lig = f_res + f_lig   # GLU + ASN (+ LIG if present)

    # ion-ion forces from pre-computed CSV
    key = (ion_id, r2_end)
    ion_axial = ion_forces.get(key, {}).get("force_all_axial", None)
    ion_mag   = ion_forces.get(key, {}).get("force_all_mag",   None)

    # combined: residues + ions, axial only — WITHOUT ligand
    if ion_axial is not None:
        combined_axial_no_lig = float(np.dot(f_res, unit)) + ion_axial
    else:
        combined_axial_no_lig = None

    # combined: residues + ions, axial only — WITH ligand (if present)
    if ion_axial is not None:
        combined_axial = float(np.dot(f_res_lig, unit)) + ion_axial
    else:
        combined_axial = None

    def _mag(f):  return float(np.linalg.norm(f))
    def _ax(f):   return float(np.dot(f, unit))
    def _r(v):    return round(v, 4) if v is not None and not np.isnan(v) else None

    row = {
        "ion_id":  ion_id,
        "run":     ev.get("run", ""),
        "r2_end":  r2_end,
        "n_asn_atoms_used": n_asn,
        "n_glu_atoms_used": n_glu,
        "n_lig_atoms_used": n_lig,
        "lig_present": has_lig,
        # ASN forces
        "force_asn_mag":   round(_mag(f_asn), 4),
        "force_asn_axial": round(_ax(f_asn),  4),
        # GLU forces
        "force_glu_mag":   round(_mag(f_glu), 4),
        "force_glu_axial": round(_ax(f_glu),  4),
        # LIG forces (None if this channel has no ligand)
        "force_lig_mag":   round(_mag(f_lig), 4) if has_lig else None,
        "force_lig_axial": round(_ax(f_lig),  4) if has_lig else None,
        # GLU + ASN combined
        "force_glu_asn_mag":   round(_mag(f_res), 4),
        "force_glu_asn_axial": round(_ax(f_res),  4),
        # ion-ion (from pre-computed CSV)
        "force_ions_axial": _r(ion_axial),
        "force_ions_mag":   _r(ion_mag),
        # GLU + ASN + ions (axial only — excludes ligand even when present)
        "force_glu_asn_ions_axial": _r(combined_axial_no_lig),
        # GLU + ASN + LIG + ions (axial only — the full physical picture)
        "force_total_axial": _r(combined_axial),
    }

    # debug text
    dbg = None
    if debug:
        lines = [
            f"{'─'*60}",
            f"  Ion {ion_id}  |  frame {r2_end}",
            f"  Target pos: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})",
            f"",
            f"  ASN residues  ({n_asn} charged atoms used)",
        ] + dbg_asn + [
            f"  → F_asn_axial  = {_ax(f_asn):+.4f}  kcal/(mol·Å)  "
            f"({'toward HBC' if _ax(f_asn) > 0 else 'toward SF'})",
            f"",
            f"  GLU residues  ({n_glu} charged atoms used)",
        ] + dbg_glu + [
            f"  → F_glu_axial  = {_ax(f_glu):+.4f}  kcal/(mol·Å)  "
            f"({'toward HBC' if _ax(f_glu) > 0 else 'toward SF — opposes permeation'})",
        ]
        if has_lig:
            lines += [
                f"",
                f"  Ligand ({lig_resname}, {n_lig} charged atoms used)",
            ] + dbg_lig + [
                f"  → F_lig_axial  = {_ax(f_lig):+.4f}  kcal/(mol·Å)  "
                f"({'toward HBC' if _ax(f_lig) > 0 else 'toward SF'})",
            ]
        else:
            lines += [f"", f"  Ligand: not present in this system"]
        lines += [
            f"",
            f"  F_glu+asn_axial          = {_ax(f_res):+.4f}  kcal/(mol·Å)",
            f"  F_glu+asn+lig_axial      = {_ax(f_res_lig):+.4f}  kcal/(mol·Å)",
            f"  F_ions_axial             = {ion_axial:+.4f}  kcal/(mol·Å)" if ion_axial is not None
                else "  F_ions_axial             = not available",
            f"  F_glu+asn+ions_axial     = {combined_axial_no_lig:+.4f}  kcal/(mol·Å)" if combined_axial_no_lig is not None
                else "  F_glu+asn+ions_axial     = not available",
            f"  F_total_axial (+lig)     = {combined_axial:+.4f}  kcal/(mol·Å)" if combined_axial is not None
                else "  F_total_axial (+lig)     = not available",
        ]
        dbg = "\n".join(lines)

    return row, dbg


# ── CSV saving ────────────────────────────────────────────────────────────────

FIELDS = [
    "ion_id", "run", "r2_end",
    "n_asn_atoms_used", "n_glu_atoms_used", "n_lig_atoms_used", "lig_present",
    "force_asn_mag",      "force_asn_axial",
    "force_glu_mag",      "force_glu_axial",
    "force_lig_mag",      "force_lig_axial",
    "force_glu_asn_mag",  "force_glu_asn_axial",
    "force_ions_mag",     "force_ions_axial",
    "force_glu_asn_ions_axial",
    "force_total_axial",
]


def _save_per_event(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"    Saved: {out_path}")


def _save_summary(rows, label, out_path):
    numeric = [f for f in FIELDS if f not in ("ion_id", "run", "r2_end", "lig_present")]
    summary_rows = [
        {"metric": "system",   "mean": label,    "std": "", "min": "", "max": "", "n": len(rows)},
        {"metric": "n_events", "mean": len(rows), "std": "", "min": "", "max": "", "n": len(rows)},
    ]
    for col in numeric:
        vals = [r[col] for r in rows if r.get(col) is not None]
        if not vals:
            continue
        summary_rows.append({
            "metric": col,
            "mean":   f"{np.mean(vals):.4f}",
            "std":    f"{np.std(vals):.4f}",
            "min":    f"{np.min(vals):.4f}",
            "max":    f"{np.max(vals):.4f}",
            "n":      len(vals),
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "min", "max", "n"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"    Saved: {out_path}")


# ── plots ─────────────────────────────────────────────────────────────────────

FS_TITLE  = 20   # title font size
FS_LABEL  = 18   # axis label font size
FS_TICK   = 17   # tick label font size
FS_ANNOT  = 16   # bar annotation font size
FS_LEGEND = 16   # legend font size


def _vals(rows, col):
    return [r[col] for r in rows if r.get(col) is not None]


def _bar_plot(rows, metrics, names, label, ylabel, title, out_path, color):
    means, stds, all_vals = [], [], []
    for m in metrics:
        v = _vals(rows, m)
        means.append(np.mean(v) if v else 0)
        stds.append(np.std(v)   if v else 0)
        all_vals.append(v)

    fig, ax = plt.subplots(figsize=(max(12, len(metrics) * 3), 7))
    x   = np.arange(len(names))
    rng = np.random.default_rng(42)

    ax.bar(x, means, yerr=stds, capsize=10, width=0.55,
           color=color, alpha=0.75, edgecolor="black", linewidth=0.8,
           error_kw=dict(elinewidth=2.0, ecolor="black"), zorder=2)

    for i, v in enumerate(all_vals):
        jitter = rng.uniform(-0.18, 0.18, size=len(v))
        ax.scatter(i + jitter, v, color=color, alpha=0.25, s=28,
                   edgecolors="none", zorder=3)

    data_top = max((max(v) for v in all_vals if v), default=1)
    data_bot = min((min(v) for v in all_vals if v), default=0)
    data_top = max(data_top, max(m + s for m, s in zip(means, stds)))
    data_bot = min(data_bot, min(m - s for m, s in zip(means, stds)))

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s, f"{m:.3f}\n±{s:.3f}",
                ha="center", va="bottom", fontsize=FS_ANNOT)

    # allow negative y for axial force plots
    pad = (data_top - data_bot) * 0.40 if data_top != data_bot else 1
    ax.set_ylim(bottom=data_bot - pad * 0.1, top=data_top + pad)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=FS_TICK)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_title(f"{title}\n{label}  (n = {len(rows)} events)",
                 fontsize=FS_TITLE, pad=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


def _hist_plot(rows, metrics, names, label, suptitle, out_path, color):
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, m, name in zip(axes, metrics, names):
        v = np.array(_vals(rows, m))
        if len(v) == 0:
            ax.set_title(name, fontsize=FS_LABEL)
            continue

        ax.hist(v, bins=max(8, len(v) // 3), color=color, alpha=0.55,
                edgecolor="white", linewidth=0.5, density=True)

        if len(v) >= 3:
            try:
                kde = gaussian_kde(v, bw_method="scott")
                xg  = np.linspace(v.min(), v.max(), 200)
                ax.plot(xg, kde(xg), color=color, linewidth=2.5)
            except Exception:
                pass

        ax.axvline(np.mean(v), color="black", linewidth=1.6,
                   linestyle="--", label=f"mean = {np.mean(v):.3f}")
        ax.axvline(0, color="gray", linewidth=1.0, linestyle=":", alpha=0.6)
        ax.set_xlabel(name, fontsize=FS_LABEL)
        ax.set_ylabel("Density", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{suptitle}\n{label}  (n = {len(rows)} events)",
                 fontsize=FS_TITLE, y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ── public entry point ────────────────────────────────────────────────────────

def run_ion_residue_force_analysis(
    u,
    permeation_events:   list,
    glu_residues:        list,
    asn_residues:        list,
    sf_low_res_residues: list,
    hbc_residues:        list,
    results_dir,
    channel_type: str = "G12",
    lig_resname: str = None,   # e.g. "LIG" — only set for channels that have a bound ligand
):
    """
    Main entry point — call this from main.py.

    Parameters
    ----------
    u                   : MDAnalysis Universe (with prmtop charges loaded)
    permeation_events   : list of dicts from analyzer.permeation_events
    glu_residues        : list of GLU resids (4 residues)
    asn_residues        : list of ASN resids (4 residues)
    sf_low_res_residues : list of resids for SF bottom ring (axis origin)
    hbc_residues        : list of resids for HBC gate (axis end)
    results_dir         : Path to RUN-level results directory
    channel_type        : str, label for plots and CSVs
    lig_resname         : resname of bound ligand (e.g. "LIG"), or None.
                           Only G12_ML / G12_GAT currently have one — pass
                           None (or omit) for G2 / G12 and the ligand bars
                           are skipped automatically.

    Ion-ion forces are read automatically from:
        results_dir/ion_ion_permeation/per_event.csv
    """
    results_dir = Path(results_dir)
    out_dir     = results_dir / "ion_residue_force"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = channel_type

    print(f"\n  Ion-residue force analysis ({label})")
    print(f"  Events        : {len(permeation_events)}")
    print(f"  Frame         : R2_end (last frame in cavity)")
    print(f"  GLU residues  : {glu_residues}")
    print(f"  ASN residues  : {asn_residues}")
    print(f"  Ligand        : {lig_resname if lig_resname else 'none for this channel'}")
    print(f"  Charges from  : prmtop partial charges via MDAnalysis")
    print(f"  Output dir    : {out_dir}")

    if not permeation_events:
        print("  No permeation events — skipping.")
        return

    # load ion-ion forces from pre-computed CSV
    ion_ion_csv = results_dir / "ion_ion_permeation" / "per_event.csv"
    ion_forces  = _load_ion_forces(ion_ion_csv)
    if ion_forces:
        print(f"  Ion-ion forces: loaded {len(ion_forces)} events from {ion_ion_csv.name}")
    else:
        print(f"  Ion-ion forces: not found at {ion_ion_csv} — combined plot will be skipped")

    rows       = []
    debug_txts = []

    for idx, ev in enumerate(tqdm(permeation_events,
                                  desc="  computing residue forces")):
        want_debug = len(debug_txts) < 3
        row, dbg = _analyze_event(u, ev, glu_residues, asn_residues,
                                  sf_low_res_residues, hbc_residues,
                                  ion_forces, lig_resname=lig_resname,
                                  debug=want_debug)
        if row is not None:
            rows.append(row)
            if dbg:
                debug_txts.append(f"=== Event {idx+1} ===\n{dbg}")

    if not rows:
        print("  No valid events — skipping.")
        return

    print(f"  Valid events  : {len(rows)}")
    has_lig = any(r["lig_present"] for r in rows)

    # write debug
    dbg_path = out_dir / "debug_examples.txt"
    with open(dbg_path, "w") as f:
        f.write(f"Ion-residue force — first {len(debug_txts)} events\n")
        f.write(f"System: {label}\n\n")
        f.write("\n\n".join(debug_txts))
    print(f"\n  Debug examples: {dbg_path}")
    for t in debug_txts:
        print(t)
    print()

    # terminal summary
    print("  AXIAL FORCES  kcal/(mol·Å)  +ve = toward HBC:")
    summary_cols = [
        ("force_asn_axial",      "F_ASN only"),
        ("force_glu_axial",      "F_GLU only"),
    ]
    if has_lig:
        summary_cols.append(("force_lig_axial", "F_LIG only"))
    summary_cols += [
        ("force_glu_asn_axial",       "F_GLU + ASN"),
        ("force_ions_axial",         "F_ions only"),
        ("force_glu_asn_ions_axial", "F_GLU + ASN + ions"),
        ("force_total_axial",        "F_GLU + ASN + LIG + ions"),
    ]
    for col, name in summary_cols:
        v = _vals(rows, col)
        if v:
            print(f"    {name:<28}  {np.mean(v):+7.3f} ± {np.std(v):.3f}")
    print()

    # save CSVs
    _save_per_event(rows, out_dir / "per_event.csv")
    _save_summary(rows, label, out_dir / "summary.csv")

    # ── force magnitude plots ─────────────────────────────────────────────
    fmag_m = ["force_asn_mag", "force_glu_mag"]
    fmag_n = ["ASN\nonly", "GLU\nonly"]
    if has_lig:
        fmag_m.append("force_lig_mag")
        fmag_n.append("Ligand\nonly")
    fmag_m.append("force_ions_mag")
    fmag_n.append("All ions")

    _bar_plot(rows, fmag_m, fmag_n, label,
              "Force magnitude  (kcal/(mol·Å))",
              "Coulomb force magnitude on permeating ion at moment of permeation",
              out_dir / "force_mag_bar.png", color="#8E44AD")
    _hist_plot(rows, fmag_m, fmag_n, label,
               "Distribution of Coulomb force magnitude on permeating ion",
               out_dir / "force_mag_hist.png", color="#8E44AD")

    # ── axial force plots (the key physics) ──────────────────────────────
    fax_m = ["force_asn_axial", "force_glu_axial"]
    fax_n = ["ASN\nonly", "GLU\nonly"]
    if has_lig:
        fax_m.append("force_lig_axial")
        fax_n.append("Ligand\nonly")
    fax_m.append("force_ions_axial")
    fax_n.append("All ions")
    fax_m.append("force_glu_asn_ions_axial")
    fax_n.append("GLU + ASN\n+ ions")
    if has_lig:
        fax_m.append("force_total_axial")
        fax_n.append("GLU + ASN\n+ ligand + ions")

    _bar_plot(rows, fax_m, fax_n, label,
              "Axial force  (kcal/(mol·Å))   [ + = toward HBC ]",
              "Axial Coulomb force on permeating ion\n"
              "(positive = pushes toward HBC = favors permeation)",
              out_dir / "force_axial_bar.png", color="#E67E22")
    _hist_plot(rows, fax_m, fax_n, label,
               "Distribution of axial force on permeating ion\n"
               "(positive = toward HBC, negative = back toward SF)",
               out_dir / "force_axial_hist.png", color="#E67E22")

    print(f"  Done. All outputs in: {out_dir}")