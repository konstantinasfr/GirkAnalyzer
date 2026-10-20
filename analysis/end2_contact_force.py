#!/usr/bin/env python3
"""
end2_contact_force.py
=====================
At every end_2 frame -- the moment the permeating K+ leaves the GLU/ASN
cavity -- record WHICH acidic/amide residues are holding it, which are free,
and what z-force it feels from each source.

One row per permeation event, one CSV.

TWO DIFFERENT QUESTIONS, KEPT APART
-----------------------------------
A site can be filled by ANY ion in the cavity, or specifically by the ion
that is leaving. These are not the same thing and the answers differ a lot:
at end_2 the permeating ion has usually already let go and dropped past the
ASN ring, while the rings themselves stay loaded with OTHER ions. Both are
recorded.

  occ_*       filled by any cavity ion  <- the occupancy table
  contact_*   held by the permeating ion itself

WHAT IS MEASURED AT THAT FRAME
------------------------------
1. CONTACTS
   Minimum distance from an ion to ANY atom of each GLU and each ASN/ASP,
   contact if <= --cutoff (default 3.0 A). Minimum distance, not COM, is the
   right criterion for a coordination cutoff. Computed for the permeating
   ion and for every other ion in the cavity.

   NAMING, as requested: when BOTH the GLU and the ASN/ASP of the same
   subunit are taken, that pair is written as the SUBUNIT name alone (G1, 2a,
   2b, 2c). Otherwise each residue is written by its own name (E141.G1,
   D173.G1, E152.2b, N184.2b, ...). So a combination reads like

        G1 + E152.2b + N184.2a

   `occ_combination` is that string for the cavity as a whole,
   `contact_combination` for the permeating ion alone. Group on
   `occ_combination` to rebuild the filled/empty table.

2. WHICH ARE FREE
   `occ_free_residues` / `n_occ_free`, split by ring as `occ_free_E_ring` and
   `occ_free_DN_ring` -- the "is it the E or the D that is free" question.
   Per-residue 0/1 columns (`occ_E141.G1`, `contact_E141.G1`) plus
   `occ_by_E141.G1`, the ion id(s) sitting there, so statistics across events
   are a column mean rather than a re-parse of the combination string.

3. OTHER IONS IN THE CAVITY
   How many, how many of them are on a site, how many are free, how many sit
   behind the permeating ion and how many in front, which sites each one
   holds (`other_ions_sites`), and which single ion pushes hardest
   (`top_pusher_*`). A second CSV, end2_other_ions.csv, has one row per other
   cavity ion per event, with its position, its sites and its own force
   contribution.

4. Z-FORCE ON THE PERMEATING ION
   Coulomb only, from the prmtop partial charges, minimum-image, in
   kcal/mol/A:

        F_z = sum_j  332.0637 * q_i * q_j * dz_ij / r_ij^3

   decomposed into
        fz_ions                  every other K+ in the system
        fz_ions_cavity           only the K+ inside the GLU/ASN cavity
        fz_ions_cavity_bound     cavity K+ that are sitting on a site
        fz_ions_cavity_unbound   cavity K+ touching nothing
        fz_ions_above            cavity K+ behind it (away from the gate)
        fz_ions_below            cavity K+ in front of it
        fz_glu                   the four GLU residues (whole residue)
        fz_asn                   the four ASN/ASP residues
        fz_protein               glu + asn
        fz_contacting            only residues the permeating ion touches
        fz_noncontacting         the rest
        fz_total                 ions + protein
   plus one column per residue (`fz_E141.G1_out`, ...) and, in the second
   CSV, one row per cavity ion with its own `fz_out`.

   SIGN. `exit_sign` is +1 when the gate is at lower z, which it normally is.
   Every force column has a `_out` twin (`fz_ions_out`, ...) already
   multiplied by that sign, so POSITIVE ALWAYS MEANS PUSHED TOWARDS THE GATE
   and the columns can be averaged across runs without checking orientation.

   This is the electrostatic force only -- no vdW, no solvent screening, no
   reaction field. It is the term the repulsion argument is about, but it is
   not the total force on the ion, and the CSV should be described that way.

USAGE - from main.py
--------------------
    from analysis.end2_contact_force import run_end2_contact_force

    run_end2_contact_force(
        universe          = u,
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = channel_type,
        frame_duration    = frame_duration,
        run               = run_number,
        force_rerun       = False,
    )

USAGE - standalone
------------------
    python3 end2_contact_force.py \
        --top  /path/com_4fs.prmtop \
        --traj /path/05_ProdN.nc \
        --results_dir /path/girk_analyser_results/G12/RUN3 \
        --channel_type G12 --run 3
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# converter.py is the only project import, and it is optional: the inline
# fallback below covers the same four channel types.
try:
    from analysis.converter import to_pdb as _to_pdb
except ImportError:
    try:
        from converter import to_pdb as _to_pdb
    except ImportError:
        _to_pdb = None

SCRIPT_VERSION = "2026-10-14a"
MODULE_DIR = "end2_contact_force"
OUT_NAME = "end2_contact_force.csv"

COULOMB = 332.0637          # kcal*A / (mol * e^2)
ION_SELECTION = "resname K+ K"

# resid lists are index-aligned by subunit: glu[i] and asn[i] are the same
# subunit, which is what makes the "both -> subunit name" rule possible.
CHANNEL_CONFIG = {
    "G12":     dict(glu=[99, 424, 748, 1073], asn=[131, 456, 780, 1105],
                    hbc=[139, 464, 788, 1113], frame_duration=0.075),
    "G2":      dict(glu=[98, 426, 754, 1082], asn=[130, 458, 786, 1114],
                    hbc=[138, 466, 794, 1122], frame_duration=0.4),
    "G12_ML":  dict(glu=[99, 424, 749, 1074], asn=[131, 456, 781, 1106],
                    hbc=[139, 464, 789, 1114], frame_duration=0.075),
    "G12_GAT": dict(glu=[98, 422, 747, 1073], asn=[130, 454, 779, 1105],
                    hbc=[138, 462, 787, 1113], frame_duration=0.075),
}
CHANNEL_CONFIG["G12_E141Q"] = dict(CHANNEL_CONFIG["G12"])
CHANNEL_CONFIG["G12_D173N"] = dict(CHANNEL_CONFIG["G12"])


# ─────────────────────────────────────────────────────────────────────────────
#  Naming
# ─────────────────────────────────────────────────────────────────────────────

def _to_pdb_fallback(resid, channel_type):
    ct = str(channel_type).upper().strip()
    if ct in ("G2",):
        e1, e2, e3, e4, o = 328, 656, 984, 1312, 54
        if resid <= e1:  return f"{resid + o}.2a"
        if resid <= e2:  return f"{resid - e1 + o}.2c"
        if resid <= e3:  return f"{resid - e2 + o}.2b"
        if resid <= e4:  return f"{resid - e3 + o}.2d"
    elif ct in ("G12", "G12_E141Q", "G12_D173N"):
        e1, e2, e3, e4, o1, o = 326, 650, 975, 1300, 42, 54
        if resid <= e1:  return f"{resid + o1}.G1"
        if resid <= e2:  return f"{resid - e1 + o}.2b"
        if resid <= e3:  return f"{resid - e2 + o}.2a"
        if resid <= e4:  return f"{resid - e3 + o}.2c"
    elif ct in ("G12_ML", "ML"):
        e1, e2, e3, e4, o1, o = 326, 651, 976, 1300, 42, 53
        if resid <= e1:  return f"{resid + o1}.G1"
        if resid <= e2:  return f"{resid - e1 + 1 + o}.2c"
        if resid <= e3:  return f"{resid - e2 + 1 + o}.2a"
        if resid <= e4:  return f"{resid - e3 + 1 + o}.2b"
    elif ct in ("G12_GAT", "GAT"):
        e1, e2, e3, e4, o1, o = 324, 649, 974, 1300, 42, 54
        if resid <= e1:  return f"{resid + o}.2b"
        if resid <= e2:  return f"{resid - e1 + o}.2a"
        if resid <= e3:  return f"{resid - e2 + o}.2c"
        if resid <= e4:  return f"{resid - e3 + o1}.G1"
    return str(resid)


def to_pdb(resid, channel_type):
    if _to_pdb is not None:
        try:
            return _to_pdb(resid, channel_type)
        except Exception:
            pass
    return _to_pdb_fallback(resid, channel_type)


LETTER = {"GLU": "E", "GLH": "E", "ASP": "D", "ASH": "D", "ASN": "N",
          "GLN": "Q"}


def residue_label(u, resid, channel_type):
    """'E141.G1' — one-letter code from the topology, number/subunit from
    the converter, so a mutated residue is never mislabelled."""
    pdb = to_pdb(resid, channel_type)
    try:
        rn = u.select_atoms(f"resid {resid}").residues[0].resname.strip().upper()
    except Exception:
        rn = ""
    return f"{LETTER.get(rn, rn[:1] or '?')}{pdb}"


def subunit_of(label):
    """'E141.G1' -> 'G1'."""
    return label.split(".")[-1] if "." in label else ""


# ─────────────────────────────────────────────────────────────────────────────
#  Events
# ─────────────────────────────────────────────────────────────────────────────

def end2_events(permeation_events):
    """
    [(rank, ion_id, end_2)] sorted by end_3 so ion_rank matches
    permeation_ion_table.py. Events with no usable end_2 are dropped.
    """
    out = []
    for ev in permeation_events or []:
        ion = ev.get("ion_id")
        if ion is None:
            continue

        def g(*keys):
            for k in keys:
                if k in ev and ev[k] is not None:
                    try:
                        return int(ev[k])
                    except (TypeError, ValueError):
                        pass
            return None

        e2 = g("end_2", "R2_end")
        e3 = g("end_3", "R3_end")
        if e2 is None and e3 is None:
            continue
        out.append([None, int(ion), e2, e3])
    out.sort(key=lambda t: (t[3] if t[3] is not None else t[2]))
    for i, rec in enumerate(out, start=1):
        rec[0] = i
    return [(r[0], r[1], r[2], r[3]) for r in out]


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry and force
# ─────────────────────────────────────────────────────────────────────────────

def minimum_image(delta, box):
    if box is None:
        return delta
    L = np.asarray(box[:3], dtype=float)
    if np.any(L <= 0):
        return delta
    return delta - L * np.round(delta / L)


def min_distance(ion_pos, group_pos, box):
    d = minimum_image(group_pos - ion_pos, box)
    return float(np.sqrt(np.einsum("ij,ij->i", d, d)).min())


def coulomb_fz(ion_pos, q_ion, pos, charges, box):
    """
    z-component of the Coulomb force on the ion from the given atoms,
    kcal/mol/A. Positive = pushed towards +z.
    """
    if len(pos) == 0:
        return 0.0
    d = minimum_image(pos - ion_pos, box)          # vector ion -> atom
    r2 = np.einsum("ij,ij->i", d, d)
    ok = r2 > 1e-8
    if not ok.any():
        return 0.0
    r3 = r2[ok] ** 1.5
    # force ON the ion points away from a like charge, i.e. along -d
    fz = -COULOMB * q_ion * np.asarray(charges)[ok] * d[ok, 2] / r3
    return float(fz.sum())


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
#
#  MEDIAN FIRST. The force distributions are skewed and n is small, so a mean
#  is pulled a long way by one event -- in RUN3 a single +25.9 moves the
#  "E141 only" mean from 5.4 to 8.8 with n=6. Every panel therefore shows the
#  median as the solid line/bar, the mean as a red dashed marker beside it,
#  and the individual events as dots, so the reader can see what the summary
#  is standing on. The n is printed on every group for the same reason.
# ─────────────────────────────────────────────────────────────────────────────

ION_COLOR = "#C0392B"
GLU_COLOR = "#2980B9"
ASN_COLOR = "#27AE60"
NET_COLOR = "#8E44AD"
MEAN_STYLE = dict(color="red", linestyle="--", linewidth=1.6)


def _fv(row, key):
    v = row.get(key, "")
    if v == "" or v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _col(rows, key):
    a = np.array([_fv(r, key) for r in rows], dtype=float)
    return a[~np.isnan(a)]


def g1_state(row, e_label, d_label):
    """The four mutually exclusive G1 states. Overlapping categories
    (`E141 filled`, `both`, `at least one`) would count the same event more
    than once, so they are not used for any per-state statistic."""
    ev, dv = _fv(row, f"occ_{e_label}"), _fv(row, f"occ_{d_label}")
    e = 0 if np.isnan(ev) else int(ev)
    d = 0 if np.isnan(dv) else int(dv)
    if e and d:
        return "both"
    if e:
        return "E only"
    if d:
        return "D only"
    return "neither"


def _save(fig, out_path):
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"    Saved: {Path(out_path).name}")


def plot_site_occupancy(rows, labels, glu, asn, out_path, title):
    """Which sites are filled, by any ion vs by the permeating ion."""
    import matplotlib.pyplot as plt
    order = list(glu) + list(asn)
    names = [labels[r] for r in order]
    occ = [100 * np.mean([_fv(x, f"occ_{labels[r]}") for x in rows])
           for r in order]
    own = [100 * np.mean([_fv(x, f"contact_{labels[r]}") for x in rows])
           for r in order]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 0.2, occ, 0.4, color="#34495E", alpha=0.9,
           edgecolor="black", linewidth=0.6, label="filled by any cavity ion")
    ax.bar(x + 0.2, own, 0.4, color=ION_COLOR, alpha=0.9,
           edgecolor="black", linewidth=0.6, label="held by the permeating ion")
    for xi, v in zip(x - 0.2, occ):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8.5)
    ax.axvline(len(glu) - 0.5, color="black", linewidth=1.0, alpha=0.6)
    ax.text(len(glu) / 2 - 0.5, 103, "E ring", ha="center", fontsize=10,
            fontweight="bold")
    ax.text(len(glu) + len(asn) / 2 - 0.5, 103, "D/N ring", ha="center",
            fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9.5)
    ax.set_ylabel("% of permeation events", fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_title(f"{title}\n(n = {len(rows)} events)", fontsize=12,
                 fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_path)


FORCE_SPEC = [("fz_ions_cavity_out", "cavity\nions", ION_COLOR),
              ("fz_glu_out", "GLU\nring", GLU_COLOR),
              ("fz_asn_out", "ASN/ASP\nring", ASN_COLOR),
              ("fz_protein_out", "protein\n(both)", "#7F8C8D"),
              ("fz_total_out", "NET", NET_COLOR)]

YLABEL = ("z-force on the permeating K⁺ (kcal/mol/Å)\n"
          "+ = pushed towards the gate")


def _force_panel(ax, rows, ylim=None, label_fs=8.5, tick_fs=9.5):
    """
    One force-balance panel.

    The annotation on each box is median, mean and "% >0" -- the share of
    events in which that component pointed towards the gate. A median can be
    positive while the sign flips from event to event, so the percentage is
    what says whether the direction is consistent: the GLU ring sits at 0%
    (it pulls back every single time) while the ASN/ASP ring is near 50%.
    """
    data = [(_col(rows, k), lab, c) for k, lab, c in FORCE_SPEC]
    data = [d for d in data if d[0].size]
    if not data:
        ax.set_axis_off()
        return None

    bp = ax.boxplot([d for d, _, _ in data], widths=0.55, patch_artist=True,
                    showfliers=False, showmeans=False,
                    medianprops=dict(color="black", linewidth=2.2))
    for box, (_, _, c) in zip(bp["boxes"], data):
        box.set_facecolor(c)
        box.set_alpha(0.6)
    rng = np.random.default_rng(0)
    for i, (d, _, _) in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, d.size), d, s=16,
                   color="#222222", alpha=0.45, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [d.mean()] * 2, **MEAN_STYLE, zorder=4)
    ax.axhline(0, color="black", linewidth=1.1)

    lo, hi = (ylim if ylim else
              (min(d.min() for d, _, _ in data),
               max(d.max() for d, _, _ in data)))
    pad = 0.16 * (hi - lo)
    for i, (d, _, _) in enumerate(data, start=1):
        ax.text(i, hi + pad * 0.2,
                f"med {np.median(d):+.1f}\nmean {d.mean():+.1f}\n"
                f"{100 * (d > 0).mean():.0f}% >0",
                ha="center", va="bottom", fontsize=label_fs)
    ax.set_ylim(lo - pad * 0.3, hi + pad * 1.55)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([lab for _, lab, _ in data], fontsize=tick_fs)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return (lo, hi)


def _force_legend(ax):
    ax.plot([], [], **MEAN_STYLE, label="mean")
    ax.plot([], [], color="black", linewidth=2.2, label="median")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)


def plot_force_box(rows, out_path, title):
    """The force balance: who pushes the exiting ion, and which way."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.4))
    if _force_panel(ax, rows, label_fs=8.5, tick_fs=10) is None:
        plt.close(fig)
        return
    ax.set_ylabel(YLABEL, fontsize=10.5)
    _force_legend(ax)
    ax.set_title(f"{title}\n(n = {len(rows)} events, Coulomb only)",
                 fontsize=12, fontweight="bold")
    _save(fig, out_path)


def plot_force_box_by_state(rows, e_label, d_label, out_dir, tag, title):
    """
    The same force-balance panel, once per G1 state, on a SHARED y axis so
    the four are directly comparable, plus each state on its own for a slide
    that needs only one.

    The states are mutually exclusive -- both / E only / D only / neither --
    unlike "E141 filled", "D173 filled" and "at least one", which overlap and
    would put the same event into several panels.
    """
    import matplotlib.pyplot as plt
    order = ["both", "E only", "D only", "neither"]
    long = {"both": f"both {e_label} + {d_label}",
            "E only": f"{e_label} only",
            "D only": f"{d_label} only",
            "neither": "neither G1 site"}
    groups = {s: [x for x in rows if g1_state(x, e_label, d_label) == s]
              for s in order}
    order = [s for s in order if groups[s]]
    if not order:
        return

    allv = np.concatenate([_col(rows, k) for k, _, _ in FORCE_SPEC])
    ylim = (float(allv.min()), float(allv.max()))

    fig, axes = plt.subplots(1, len(order), figsize=(4.4 * len(order), 5.8),
                             sharey=True)
    for ax, s in zip(np.atleast_1d(axes), order):
        _force_panel(ax, groups[s], ylim=ylim, label_fs=7.5, tick_fs=8.5)
        ax.set_title(f"{long[s]}\nn = {len(groups[s])} "
                     f"({100 * len(groups[s]) / len(rows):.0f}%)",
                     fontsize=11, fontweight="bold")
    np.atleast_1d(axes)[0].set_ylabel(YLABEL, fontsize=10.5)
    _force_legend(np.atleast_1d(axes)[-1])
    fig.suptitle(f"{title}  (shared y axis; n = {len(rows)} events total)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir / f"force_balance_by_G1_state_{tag}.png")

    slug = {"both": "both", "E only": "E_only", "D only": "D_only",
            "neither": "neither"}
    for s in order:
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        if _force_panel(ax, groups[s], ylim=ylim, label_fs=8.5,
                        tick_fs=10) is None:
            plt.close(fig)
            continue
        ax.set_ylabel(YLABEL, fontsize=10.5)
        _force_legend(ax)
        ax.set_title(f"{title}\n{long[s]} — n = {len(groups[s])} events",
                     fontsize=12, fontweight="bold")
        _save(fig, out_dir / f"force_balance_{tag}_{slug[s]}.png")



def plot_g1_states(rows, e_label, d_label, out_path, title):
    """
    Force components across the four mutually exclusive G1 states.
    Medians, because with n = 3 and n = 6 in two of the cells a single event
    moves a mean several kcal/mol/A.
    """
    import matplotlib.pyplot as plt
    order = ["both", "E only", "D only", "neither"]
    groups = {s: [x for x in rows if g1_state(x, e_label, d_label) == s]
              for s in order}
    order = [s for s in order if groups[s]]
    spec = [("fz_ions_cavity_out", "cavity ions", ION_COLOR),
            ("fz_glu_out", "GLU", GLU_COLOR),
            ("fz_asn_out", "ASN/ASP", ASN_COLOR),
            ("fz_total_out", "NET", NET_COLOR)]
    x = np.arange(len(order))
    width = 0.8 / len(spec)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    rng = np.random.default_rng(0)
    for i, (key, lab, c) in enumerate(spec):
        pos = x + i * width - 0.4 + width / 2
        med = [np.median(_col(groups[s], key)) for s in order]
        ax.bar(pos, med, width * 0.88, color=c, alpha=0.85,
               edgecolor="black", linewidth=0.6, label=lab)
        for p, s in zip(pos, order):
            d = _col(groups[s], key)
            ax.scatter(p + rng.uniform(-width * 0.18, width * 0.18, d.size), d,
                       s=10, color="#222222", alpha=0.5, zorder=3)
            ax.plot([p - width * 0.4, p + width * 0.4],
                    [d.mean()] * 2, **MEAN_STYLE, zorder=4)
    ax.axhline(0, color="black", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n = {len(groups[s])})" for s in order],
                       fontsize=10)
    ax.set_xlabel(f"which G1 site is filled  ({e_label} / {d_label})",
                  fontsize=10.5)
    ax.set_ylabel("z-force on the permeating K⁺ (kcal/mol/Å)\n"
                  "+ = pushed towards the gate", fontsize=10.5)
    ax.plot([], [], **MEAN_STYLE, label="mean")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9,
              title="bars = median")
    ax.set_title(f"{title}\nstates are mutually exclusive — no event counted "
                 f"twice", fontsize=11.5, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_path)


def plot_combinations(rows, out_path, title, top_n=12):
    """How often each occupancy combination occurs, and the median net force
    beside it. A flat force column next to a very uneven count column is the
    point: which sites are filled does not set the push."""
    import matplotlib.pyplot as plt
    combos = {}
    for r in rows:
        combos.setdefault(r["occ_combination"], []).append(r)
    items = sorted(combos.items(), key=lambda kv: -len(kv[1]))[:top_n]
    names = [k if len(k) <= 34 else k[:32] + "…" for k, _ in items]
    counts = [len(v) for _, v in items]
    meds = [np.median(_col(v, "fz_total_out")) for _, v in items]
    y = np.arange(len(items))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.46 * len(items) + 2.6),
                                   sharey=True,
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    ax1.barh(y, counts, 0.7, color="#34495E", alpha=0.9, edgecolor="black",
             linewidth=0.5)
    for yi, c in zip(y, counts):
        ax1.text(c + 0.1, yi, f"{c}", va="center", fontsize=8.5)
    ax1.set_xlabel("events", fontsize=10)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8.5)
    ax1.set_xlim(0, max(counts) * 1.2)
    ax1.invert_xaxis()
    ax1.yaxis.tick_right()
    ax1.grid(True, alpha=0.25, axis="x")
    ax1.set_axisbelow(True)
    ax1.spines[["top", "left"]].set_visible(False)

    rng = np.random.default_rng(0)
    for yi, (_, v) in zip(y, items):
        d = _col(v, "fz_total_out")
        ax2.scatter(d, yi + rng.uniform(-0.16, 0.16, d.size), s=16,
                    color="#222222", alpha=0.5, zorder=3)
    # a "median" over one or two events is that event -- drawn hollow so it is
    # not read as a group result
    solid = [m if n >= 3 else 0.0 for m, n in zip(meds, counts)]
    faint = [m if n < 3 else 0.0 for m, n in zip(meds, counts)]
    ax2.barh(y, solid, 0.7, color=NET_COLOR, alpha=0.6, edgecolor="black",
             linewidth=0.5, zorder=2, label="median (n ≥ 3)")
    ax2.barh(y, faint, 0.7, facecolor="none", edgecolor=NET_COLOR,
             linewidth=1.0, linestyle=":", zorder=2, label="n < 3")
    ax2.tick_params(left=False)
    ax2.axvline(0, color="black", linewidth=1.1)
    ax2.axvline(np.median(_col(rows, "fz_total_out")), color="green",
                linestyle="-.", linewidth=1.5,
                label=f"overall median {np.median(_col(rows, 'fz_total_out')):+.1f}")
    ax2.set_xlabel("net z-force (kcal/mol/Å), + = towards the gate\n"
                   "bars = median, dots = individual events", fontsize=9.5)
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.grid(True, alpha=0.25, axis="x")
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(f"{title}  (n = {len(rows)} events)", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_path)


def plot_release_distance(rows, labels, all_resids, cutoff, out_path, title):
    """How far the permeating ion is from the nearest site when it leaves."""
    import matplotlib.pyplot as plt
    d = np.array([min(_fv(x, f"dmin_{labels[r]}") for r in all_resids)
                  for x in rows], dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(d, bins=min(20, max(6, d.size // 3)), color="#34495E", alpha=0.8,
            edgecolor="black", linewidth=0.5)
    ax.axvline(cutoff, color="black", linewidth=1.8,
               label=f"contact cutoff {cutoff:g} Å")
    ax.axvline(np.median(d), color="green", linestyle="-.", linewidth=1.8,
               label=f"median {np.median(d):.2f} Å")
    ax.axvline(d.mean(), **MEAN_STYLE, label=f"mean {d.mean():.2f} Å")
    n_free = int((d > cutoff).sum())
    ax.set_xlabel("distance from the permeating K⁺ to its nearest "
                  "GLU/ASN atom (Å)", fontsize=10)
    ax.set_ylabel("events", fontsize=10.5)
    ax.set_title(f"{title}\nholding nothing in {n_free}/{d.size} events",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_path)


def make_plots(rows, out_dir, labels, glu, asn, all_resids, cutoff, tag,
               event="end_2"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt   # noqa: F401
    except ImportError:
        print("  [warn] matplotlib not available — plots skipped")
        return
    e_label, d_label = labels[glu[0]], labels[asn[0]]
    ev = event
    # at end_3 the ion is past the gate, so "already let go" is no longer the
    # headline -- the distance panel is titled for what it actually shows
    rel_title = ("the exiting ion has already let go" if ev == "end_2"
                 else "distance to the rings once past the gate")
    print()
    plot_site_occupancy(rows, labels, glu, asn,
                        out_dir / f"site_occupancy_{tag}.png",
                        f"{tag} — which sites are filled at {ev}")
    plot_force_box(rows, out_dir / f"force_balance_{tag}.png",
                   f"{tag} — force balance on the permeating K⁺ at {ev}")
    plot_force_box_by_state(rows, e_label, d_label, out_dir, tag,
                            f"{tag} — force balance by G1 occupancy state "
                            f"at {ev}")
    plot_g1_states(rows, e_label, d_label,
                   out_dir / f"g1_states_{tag}.png",
                   f"{tag} — does G1 coverage change the force at {ev}?")
    plot_combinations(rows, out_dir / f"combinations_{tag}.png",
                      f"{tag} — occupancy combinations and the net force "
                      f"at {ev}")
    plot_release_distance(rows, labels, all_resids, cutoff,
                          out_dir / f"release_distance_{tag}.png",
                          f"{tag} — {rel_title}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main pass
# ─────────────────────────────────────────────────────────────────────────────

def run_end2_contact_force(universe, permeation_events, results_dir,
                           channel_type, glu_residues=None, asn_residues=None,
                           hbc_residues=None, frame_duration=None,
                           cutoff=3.0, radius=12.0, run="",
                           force_rerun=False, make_plots_too=True,
                           event="end_2", out_name=None):
    """
    One row per permeation event, measured at a single frame per event.

    event="end_2" is the moment the ion leaves the GLU/ASN cavity;
    event="end_3" is the moment it passes the HBC gate. The two answer
    different questions and produce separate files:

        end_2 -> end2_contact_force.csv, end2_other_ions.csv
        end_3 -> end3_contact_force.csv, end3_other_ions.csv

    At end_3 the ion is BELOW the gate, well outside the GLU/ASN rings, so
    its own contacts will be essentially zero and the forces smaller and
    longer-ranged. What stays meaningful there is the occupancy of the rings
    by the other ions -- which is what "which residues hold an ion at
    permeation" was asking -- and the push it still feels from the column
    above it as it clears the gate.

    Returns the rows, or None if skipped.
    """
    event = str(event).strip().lower().replace("-", "_")
    if event not in ("end_2", "end_3"):
        sys.exit(f"ERROR: event must be 'end_2' or 'end_3', got '{event}'")
    prefix = "end2" if event == "end_2" else "end3"
    if out_name is None:
        out_name = f"{prefix}_contact_force.csv"
    u = universe
    cfg = CHANNEL_CONFIG.get(channel_type, {})
    glu = list(glu_residues) if glu_residues else cfg.get("glu")
    asn = list(asn_residues) if asn_residues else cfg.get("asn")
    hbc = list(hbc_residues) if hbc_residues else cfg.get("hbc")
    if not (glu and asn and hbc):
        sys.exit(f"ERROR: no ring definitions for channel_type "
                 f"'{channel_type}' — pass glu/asn/hbc_residues explicitly.")
    if len(glu) != len(asn):
        sys.exit("ERROR: glu and asn lists must be the same length and "
                 "index-aligned by subunit.")
    if frame_duration is None:
        frame_duration = cfg.get("frame_duration", 0.075)

    out_dir = Path(results_dir) / MODULE_DIR
    out_csv = out_dir / out_name
    if out_csv.exists() and not force_rerun:
        print("\n" + "!" * 70)
        print(f"  SKIPPED: {out_csv} already exists.")
        print(f"  Nothing was recomputed. Pass force_rerun=True to overwrite.")
        print("!" * 70)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"{event.upper()} CONTACTS AND Z-FORCE")
    print("=" * 70)
    print(f"  version: {SCRIPT_VERSION}")
    print(f"  channel={channel_type}  run={run}  event={event}  "
          f"cutoff={cutoff} A  radius={radius} A")

    events = end2_events(permeation_events)
    events = [e for e in events
              if (e[2] if event == "end_2" else e[3]) is not None]
    print(f"  {len(events)} permeation events with a usable {event} frame")
    if not events:
        print("  Nothing to do.")
        return None

    # ---- selections, once ----
    ions = u.select_atoms(ION_SELECTION)
    if len(ions) == 0:
        sys.exit(f"ERROR: ion selection '{ION_SELECTION}' matched 0 atoms.")
    ion_resids = np.array([a.resid for a in ions])
    ion_charges = np.array(ions.charges, dtype=float)
    row_of_resid = {int(r): i for i, r in enumerate(ion_resids)}

    all_resids = list(glu) + list(asn)
    labels = {r: residue_label(u, r, channel_type) for r in all_resids}
    kind = {**{r: "E" for r in glu}, **{r: "DN" for r in asn}}
    groups = {}
    for r in all_resids:
        ag = u.select_atoms(f"resid {r}")
        if len(ag) == 0:
            sys.exit(f"ERROR: resid {r} matched 0 atoms — wrong channel_type?")
        groups[r] = ag
    # subunit -> (glu_resid, asn_resid), for the "both -> subunit name" rule
    pairs = [(g, a, subunit_of(labels[g])) for g, a in zip(glu, asn)]

    rings = {}
    for name, resids in (("glu", glu), ("asn", asn), ("hbc", hbc)):
        rings[name] = u.select_atoms("resid " + " ".join(str(x) for x in resids))

    ordered = [labels[r] for r in glu] + [labels[r] for r in asn]
    print("  residues tracked: " + ", ".join(ordered))

    traj_last = len(u.trajectory) - 1
    rows = []
    ion_rows_detail = []
    skipped = []

    try:
        from tqdm import tqdm
        iterator = tqdm(events, desc="end_2 frames", unit="event")
    except ImportError:
        iterator = events

    for rank, ion_id, frame_e2, frame_e3 in iterator:
        frame = frame_e2 if event == "end_2" else frame_e3
        if frame is None or frame < 0 or frame > traj_last:
            skipped.append((rank, ion_id, frame))
            continue
        i_row = row_of_resid.get(int(ion_id))
        if i_row is None:
            skipped.append((rank, ion_id, frame_e2))
            continue

        ts = u.trajectory[int(frame)]
        box = ts.dimensions

        z_glu = float(rings["glu"].center_of_mass()[2])
        z_asn = float(rings["asn"].center_of_mass()[2])
        z_hbc = float(rings["hbc"].center_of_mass()[2])
        axis_xy = np.mean([rings["glu"].center_of_mass()[:2],
                           rings["asn"].center_of_mass()[:2]], axis=0)
        # +1 when the gate is at lower z, so *_out columns are "towards gate"
        exit_sign = -1.0 if z_hbc < z_glu else 1.0
        z_lo, z_hi = min(z_glu, z_asn), max(z_glu, z_asn)

        ion_pos_all = ions.positions.copy()
        ion_pos = ion_pos_all[i_row]
        q_ion = float(ion_charges[i_row])

        # ---- contacts of the permeating ion ----
        dmin = {r: min_distance(ion_pos, groups[r].positions, box)
                for r in all_resids}
        in_contact = {r: dmin[r] <= cutoff for r in all_resids}

        # name the combination: a fully-held subunit collapses to its own name
        parts, used = [], set()
        for g, a, sub in pairs:
            if in_contact[g] and in_contact[a]:
                parts.append(sub)
                used.add(g)
                used.add(a)
        for r in all_resids:
            if in_contact[r] and r not in used:
                parts.append(labels[r])
        combination = " + ".join(parts) if parts else "(none)"

        contacted = [labels[r] for r in all_resids if in_contact[r]]
        free = [labels[r] for r in all_resids if not in_contact[r]]
        n_free_E = sum(1 for r in glu if not in_contact[r])
        n_free_DN = sum(1 for r in asn if not in_contact[r])

        # ---- other ions in the cavity: where each one sits, and what it does
        #      to the permeating ion ----
        others = np.array([k for k in range(len(ions)) if k != i_row])
        opos = ion_pos_all[others]
        dxy = opos[:, :2] - axis_xy
        in_cav = ((opos[:, 2] >= z_lo) & (opos[:, 2] <= z_hi)
                  & (np.einsum("ij,ij->i", dxy, dxy) <= radius * radius))
        cav_rows = others[in_cav]

        # per cavity ion: its own contacts and its own z-force contribution
        cav_info = []
        for k in cav_rows:
            p = ion_pos_all[k]
            dk = {r: min_distance(p, groups[r].positions, box)
                  for r in all_resids}
            touch = [r for r in all_resids if dk[r] <= cutoff]
            dvec = minimum_image(p - ion_pos, box)
            cav_info.append(dict(
                row=int(k), ion_id=int(ion_resids[k]),
                z=float(p[2]), z_rel_asn=float(p[2]) - z_asn,
                radial=float(np.linalg.norm(p[:2] - axis_xy)),
                dist_to_permeating=float(np.linalg.norm(dvec)),
                dz_to_permeating=float(dvec[2]),
                touching=touch, dmin=dk,
                fz=coulomb_fz(ion_pos, q_ion, p[None, :],
                              [float(ion_charges[k])], box)))

        bound = [c for c in cav_info if c["touching"]]
        unbound = [c for c in cav_info if not c["touching"]]
        above = [c for c in cav_info if c["dz_to_permeating"] * exit_sign < 0]
        below = [c for c in cav_info if c["dz_to_permeating"] * exit_sign > 0]

        # ---- which residues are OCCUPIED, by ANY ion ----
        # This is the professor's table: a site counts as filled whoever is
        # sitting on it. The permeating ion's own contacts stay in the
        # contact_* columns, so the two questions never get mixed up.
        occ_by = {}
        for r in all_resids:
            holders = [c["ion_id"] for c in cav_info if r in c["touching"]]
            if in_contact[r]:
                holders = [int(ion_id)] + holders
            occ_by[r] = holders
        occupied = {r: bool(occ_by[r]) for r in all_resids}

        oparts, oused = [], set()
        for g, a, sub in pairs:
            if occupied[g] and occupied[a]:
                oparts.append(sub)
                oused.add(g)
                oused.add(a)
        for r in all_resids:
            if occupied[r] and r not in oused:
                oparts.append(labels[r])
        occ_combination = " + ".join(oparts) if oparts else "(none)"
        occ_free = [labels[r] for r in all_resids if not occupied[r]]

        if cav_rows.size:
            dist_cav = np.array([c["dist_to_permeating"] for c in cav_info])
            nearest = float(dist_cav.min())
            mean_d = float(dist_cav.mean())
        else:
            nearest, mean_d = "", ""

        # ---- z-forces ----
        fz_ions = coulomb_fz(ion_pos, q_ion, opos, ion_charges[others], box)
        fz_ions_cav = sum(c["fz"] for c in cav_info)
        fz_cav_bound = sum(c["fz"] for c in bound)
        fz_cav_unbound = sum(c["fz"] for c in unbound)
        fz_cav_above = sum(c["fz"] for c in above)
        fz_cav_below = sum(c["fz"] for c in below)
        top = max(cav_info, key=lambda c: c["fz"] * exit_sign) if cav_info else None
        fz_res = {r: coulomb_fz(ion_pos, q_ion, groups[r].positions,
                                np.array(groups[r].charges, dtype=float), box)
                  for r in all_resids}
        fz_glu = sum(fz_res[r] for r in glu)
        fz_asn = sum(fz_res[r] for r in asn)
        fz_prot = fz_glu + fz_asn
        fz_con = sum(fz_res[r] for r in all_resids if in_contact[r])
        fz_free = sum(fz_res[r] for r in all_resids if not in_contact[r])
        fz_total = fz_ions + fz_prot

        row = dict(
            channel_type=channel_type, run=run, ion_rank=rank, ion_id=ion_id,
            event=event, frame_measured=frame,
            frame_end_2=frame_e2, frame_end_3=frame_e3,
            time_measured_ns=round(frame * frame_duration, 4),
            cutoff_A=cutoff, cavity_radius_A=radius, exit_sign=int(exit_sign),
            # contacts
            contact_combination=combination,
            n_contacts=len(contacted),
            n_contacts_E=sum(1 for r in glu if in_contact[r]),
            n_contacts_DN=sum(1 for r in asn if in_contact[r]),
            contacted_residues="|".join(contacted),
            free_residues="|".join(free),
            n_free=len(free), free_E_ring=n_free_E, free_DN_ring=n_free_DN,
            n_subunits_fully_held=sum(1 for g, a, _ in pairs
                                      if in_contact[g] and in_contact[a]),
            # WHICH SITES ARE FILLED, by any ion (the professor's table)
            occ_combination=occ_combination,
            n_occupied=sum(1 for r in all_resids if occupied[r]),
            n_occupied_E=sum(1 for r in glu if occupied[r]),
            n_occupied_DN=sum(1 for r in asn if occupied[r]),
            occ_free_residues="|".join(occ_free),
            n_occ_free=len(occ_free),
            occ_free_E_ring=sum(1 for r in glu if not occupied[r]),
            occ_free_DN_ring=sum(1 for r in asn if not occupied[r]),
            n_subunits_fully_occupied=sum(1 for g, a, _ in pairs
                                          if occupied[g] and occupied[a]),
            # other ions in the cavity
            n_other_ions_cavity=int(cav_rows.size),
            n_other_ions_in_contact=len(bound),
            n_other_ions_no_contact=len(unbound),
            n_other_ions_above=len(above), n_other_ions_below=len(below),
            other_ion_ids="|".join(str(c["ion_id"]) for c in cav_info),
            other_ions_bound_ids="|".join(str(c["ion_id"]) for c in bound),
            other_ions_sites="|".join(
                f"{c['ion_id']}:" + ",".join(labels[r] for r in c["touching"])
                for c in cav_info if c["touching"]),
            nearest_other_ion_A=(round(nearest, 3) if nearest != "" else ""),
            mean_other_ion_A=(round(mean_d, 3) if mean_d != "" else ""),
            top_pusher_ion_id=(top["ion_id"] if top else ""),
            top_pusher_fz_out=(round(top["fz"] * exit_sign, 4) if top else ""),
            top_pusher_dist_A=(round(top["dist_to_permeating"], 3)
                               if top else ""),
            top_pusher_bound=(int(bool(top["touching"])) if top else ""),
            # ion position
            ion_z=round(float(ion_pos[2]), 3),
            ion_z_rel_glu=round(float(ion_pos[2]) - z_glu, 3),
            ion_z_rel_asn=round(float(ion_pos[2]) - z_asn, 3),
            ion_radial_A=round(float(np.linalg.norm(ion_pos[:2] - axis_xy)), 3),
            # forces, raw then oriented towards the gate
            fz_ions=round(fz_ions, 4),
            fz_ions_cavity=round(fz_ions_cav, 4),
            fz_glu=round(fz_glu, 4), fz_asn=round(fz_asn, 4),
            fz_protein=round(fz_prot, 4),
            fz_contacting=round(fz_con, 4),
            fz_noncontacting=round(fz_free, 4),
            fz_total=round(fz_total, 4),
            fz_ions_cavity_bound=round(fz_cav_bound, 4),
            fz_ions_cavity_unbound=round(fz_cav_unbound, 4),
            fz_ions_out=round(fz_ions * exit_sign, 4),
            fz_ions_cavity_out=round(fz_ions_cav * exit_sign, 4),
            fz_ions_cavity_bound_out=round(fz_cav_bound * exit_sign, 4),
            fz_ions_cavity_unbound_out=round(fz_cav_unbound * exit_sign, 4),
            fz_ions_above_out=round(fz_cav_above * exit_sign, 4),
            fz_ions_below_out=round(fz_cav_below * exit_sign, 4),
            fz_glu_out=round(fz_glu * exit_sign, 4),
            fz_asn_out=round(fz_asn * exit_sign, 4),
            fz_protein_out=round(fz_prot * exit_sign, 4),
            fz_contacting_out=round(fz_con * exit_sign, 4),
            fz_noncontacting_out=round(fz_free * exit_sign, 4),
            fz_total_out=round(fz_total * exit_sign, 4),
        )
        for r in all_resids:
            lb = labels[r]
            row[f"occ_{lb}"] = int(occupied[r])
            row[f"occ_by_{lb}"] = "|".join(str(x) for x in occ_by[r])
            row[f"contact_{lb}"] = int(in_contact[r])
            row[f"dmin_{lb}"] = round(dmin[r], 3)
            row[f"fz_{lb}_out"] = round(fz_res[r] * exit_sign, 4)
        rows.append(row)

        # one row per OTHER cavity ion, for the per-ion plots
        for c in cav_info:
            d = dict(channel_type=channel_type, run=run, ion_rank=rank,
                     permeating_ion_id=ion_id, event=event,
                     frame_measured=frame,
                     other_ion_id=c["ion_id"],
                     bound=int(bool(c["touching"])),
                     sites="|".join(labels[r] for r in c["touching"]),
                     n_sites=len(c["touching"]),
                     z=round(c["z"], 3),
                     z_rel_asn=round(c["z_rel_asn"], 3),
                     radial_A=round(c["radial"], 3),
                     dist_to_permeating_A=round(c["dist_to_permeating"], 3),
                     dz_to_permeating=round(c["dz_to_permeating"], 3),
                     side=("below" if c["dz_to_permeating"] * exit_sign > 0
                           else "above"),
                     fz_out=round(c["fz"] * exit_sign, 4))
            for r in all_resids:
                d[f"dmin_{labels[r]}"] = round(c["dmin"][r], 3)
            ion_rows_detail.append(d)

    if skipped:
        print(f"  [warn] {len(skipped)} event(s) skipped (frame out of range "
              f"or ion not found): "
              + ", ".join(f"rank {a} ion {b} frame {c}" for a, b, c in skipped[:8])
              + (" ..." if len(skipped) > 8 else ""))
    if not rows:
        print("  No usable events — nothing written.")
        return None

    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"    Saved: {out_csv.name}  ({len(rows)} rows)")

    if ion_rows_detail:
        dcols = []
        for r in ion_rows_detail:
            for k in r:
                if k not in dcols:
                    dcols.append(k)
        detail_csv = out_dir / f"{prefix}_other_ions.csv"
        with open(detail_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=dcols, extrasaction="ignore")
            w.writeheader()
            w.writerows(ion_rows_detail)
        print(f"    Saved: {detail_csv.name}  ({len(ion_rows_detail)} rows, "
              f"one per other cavity ion per event)")

    # ---- console summary, so the run is readable without opening the CSV ----
    def table(key, title):
        combos = {}
        for r in rows:
            combos[r[key]] = combos.get(r[key], 0) + 1
        print(f"\n  {title}")
        print(f"  {'combination':<48}{'n':>4}{'%':>7}")
        for c, n in sorted(combos.items(), key=lambda kv: -kv[1]):
            print(f"  {c[:46]:<48}{n:>4}{100.0 * n / len(rows):>6.1f}%")

    table("occ_combination", "SITES FILLED BY ANY ION (the occupancy table)")
    table("contact_combination", "SITES HELD BY THE PERMEATING ION ITSELF")

    print(f"\n  {'residue':<12}{'filled %':>10}{'free %':>9}"
          f"{'by permeating %':>17}{'mean fz_out':>13}")
    for r in all_resids:
        lb = labels[r]
        occ = sum(x[f"occ_{lb}"] for x in rows) / len(rows)
        own = sum(x[f"contact_{lb}"] for x in rows) / len(rows)
        fz = np.mean([x[f"fz_{lb}_out"] for x in rows])
        print(f"  {lb:<12}{100 * occ:>9.1f}%{100 * (1 - occ):>8.1f}%"
              f"{100 * own:>16.1f}%{fz:>13.3f}")

    def m(key):
        v = [x[key] for x in rows if x[key] != ""]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n  mean over {len(rows)} events "
          f"(+ = towards the gate, kcal/mol/A):")
    print(f"    from other ions (all)        {m('fz_ions_out'):>9.3f}")
    print(f"    from cavity ions             {m('fz_ions_cavity_out'):>9.3f}")
    print(f"      of which, ions ON a site   {m('fz_ions_cavity_bound_out'):>9.3f}")
    print(f"      of which, free ions        {m('fz_ions_cavity_unbound_out'):>9.3f}")
    print(f"      from behind (above)        {m('fz_ions_above_out'):>9.3f}")
    print(f"      from in front (below)      {m('fz_ions_below_out'):>9.3f}")
    print(f"    from GLU ring                {m('fz_glu_out'):>9.3f}")
    print(f"    from ASN/ASP ring            {m('fz_asn_out'):>9.3f}")
    print(f"    from protein (both rings)    {m('fz_protein_out'):>9.3f}")
    print(f"    net                          {m('fz_total_out'):>9.3f}")
    print(f"\n    sites filled per event       {m('n_occupied'):>9.2f} of "
          f"{len(all_resids)}   (E ring free {m('occ_free_E_ring'):.2f}, "
          f"D/N ring free {m('occ_free_DN_ring'):.2f})")
    print(f"    other cavity ions            {m('n_other_ions_cavity'):>9.2f}"
          f"   (on a site {m('n_other_ions_in_contact'):.2f}, "
          f"free {m('n_other_ions_no_contact'):.2f})")
    print(f"    held by the permeating ion   {m('n_contacts'):>9.2f} of "
          f"{len(all_resids)}")
    if make_plots_too:
        tag = (f"{channel_type}" + (f"_RUN{run}" if str(run) != "" else "")
               + ("" if event == "end_2" else f"_{prefix}"))
        make_plots(rows, out_dir, labels, glu, asn, all_resids, cutoff, tag,
                   event)

    print(f"\n✓ Done. Output in {out_dir}/")
    return rows


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", required=True)
    ap.add_argument("--traj", required=True,
                    help="SOLVATED production trajectory (not protein.nc)")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--channel_type", required=True,
                    choices=list(CHANNEL_CONFIG))
    ap.add_argument("--run", default="")
    ap.add_argument("--cutoff", type=float, default=3.0,
                    help="Contact cutoff in A, minimum ion-to-residue-atom "
                         "distance (default 3.0)")
    ap.add_argument("--radius", type=float, default=12.0,
                    help="Cavity cylinder radius for counting other ions")
    ap.add_argument("--frame_duration", type=float, default=None)
    ap.add_argument("--glu", nargs="+", type=int, default=None)
    ap.add_argument("--asn", nargs="+", type=int, default=None)
    ap.add_argument("--hbc", nargs="+", type=int, default=None)
    ap.add_argument("--event", default="end_2",
                    choices=["end_2", "end_3", "both"],
                    help="Which frame to measure at (default end_2). "
                         "'both' runs each in turn into its own files.")
    ap.add_argument("--no_plots", action="store_true",
                    help="Write the CSVs only")
    ap.add_argument("--force_rerun", action="store_true")
    args = ap.parse_args()

    try:
        import MDAnalysis as mda
    except ImportError:
        sys.exit("ERROR: MDAnalysis required — activate the Amber conda env.")

    results_dir = Path(args.results_dir)

    # events: reuse whatever the pipeline already wrote for this run
    events = None
    try:
        from analysis.cavity_water_regions import load_events_from_disk
        events, _ = load_events_from_disk(results_dir)
    except Exception:
        try:
            from cavity_water_regions import load_events_from_disk
            events, _ = load_events_from_disk(results_dir)
        except Exception:
            events = None
    if not events:
        sys.exit("ERROR: could not load permeation events for this run "
                 "(cavity_water_regions.load_events_from_disk failed). "
                 "Call run_end2_contact_force() from main.py instead, passing "
                 "analyzer.permeation_events.")

    u = mda.Universe(args.top, args.traj)
    wanted = ["end_2", "end_3"] if args.event == "both" else [args.event]
    for ev in wanted:
        run_end2_contact_force(
            universe=u, permeation_events=events, results_dir=results_dir,
            channel_type=args.channel_type, glu_residues=args.glu,
            asn_residues=args.asn, hbc_residues=args.hbc,
            frame_duration=args.frame_duration, cutoff=args.cutoff,
            radius=args.radius, run=args.run, force_rerun=args.force_rerun,
            make_plots_too=not args.no_plots, event=ev)


if __name__ == "__main__":
    main()