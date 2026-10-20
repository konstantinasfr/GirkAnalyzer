#!/usr/bin/env python3
"""
analysis/pip_binding_quality.py
===============================
Per-PIP2 binding-quality summary: turns "is this PIP bound well or badly"
into four numbers per lipid, for all four sites at once.

Measured per PIP (labelled by pip_mapping, e.g. 2a / 2b / 2c / G1):

  1. CONTACT OCCUPANCY per protein residue
       fraction of frames with any heavy-atom pair within `cutoff` of the
       headgroup. Residues are auto-detected from the trajectory unless
       `track_residues` is given.

  2. SALT BRIDGES  (total, and split per phosphate: P4 / P5 / P1)
       phosphate oxygens <-> LYS NZ / ARG NE,NH1,NH2 / HIS(P) ND1,NE2 within
       `sb_cutoff`, counted as distinct cationic residues. This is the grip
       that actually holds the lipid -- a PIP can stay "in contact" while
       having lost every salt bridge.

  3. BOUND-STATE TIMESERIES -> frac_bound, longest bound stretch, number of
       unbinding events (debounced by `min_event` frames).

  4. HEADGROUP DRIFT: displacement of the phosphate COM from its starting
       position after CA superposition. Plateau = settled pose; a trace still
       climbing at the end = the lipid is leaving the site and the occupancies
       are averaging over two states.

Complements the existing PIP modules rather than replacing them:
pip_proximity_analysis gives per-residue traces, residue_pip_two_residues_proximity
gives the 199/64 pairs; this one gives the one-row-per-site verdict used to
decide whether a run is usable at all.

Interpretation (rules of thumb):
    good : >=2-3 residues with occupancy > 0.8, mean salt bridges >= 2,
           frac_bound > 0.9, 0-1 unbinding events, drift plateau < ~4 A
    poor : no residue above ~0.5, mean salt bridges < 1, drift climbing to the
           end of the run, many unbinding events

With a 3:1 stoichiometry the G1 interface is NOT equivalent to the three G2
ones, so the four sites are reported separately and never averaged.

OUTPUT (results_dir/pip_binding_quality/):
    pip_occupancy.csv            per-PIP x per-residue contact + salt-bridge occupancy
    pip_summary.csv              one row per PIP (the verdict table)
    pip_timeseries.csv           per-frame contacts / salt bridges / drift
    pip_interface_comparison.csv the four sites on a common residue axis
    pip_occupancy_heatmap.png
    pip_interface_comparison.png
    pip_binding_timeseries.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.analysis.align import rotation_matrix

try:
    from analysis.converter import to_pdb
except ImportError:  # standalone use
    def to_pdb(resid, channel_type):
        return str(resid)


CATION_SEL = ("(resname LYS LYN and name NZ) or "
              "(resname ARG and name NE NH1 NH2) or "
              "(resname HIP HIS HIE HID and name ND1 NE2)")

PROT_SEL = "protein and not name H*"


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _split_pdb(label):
    """'199.2a' -> ('199', '2a');  '42.G1' -> ('42', 'G1');  '1234' -> ('1234', '')"""
    if "." in label:
        num, sub = label.split(".", 1)
        return num, sub
    return label, ""


def _safe_box(ts):
    """protein.nc is stripped and often carries no box -- fall back to None."""
    dims = ts.dimensions
    if dims is None:
        return None
    if not np.all(np.isfinite(dims[:3])) or np.any(dims[:3] <= 0):
        return None
    return dims


def _phosphate_groups(residue):
    """
    {phosphorus name -> AtomGroup(P + its bonded oxygens)} for one PIP residue,
    e.g. {'P4': <...>, 'P5': <...>, 'P': <...>}. Uses prmtop bonds when present.
    """
    groups = {}
    p_atoms = residue.atoms.select_atoms("name P*")
    o_atoms = residue.atoms.select_atoms("name O*")
    for p in p_atoms:
        idx = [p.index]
        try:
            idx += [b.index for b in p.bonded_atoms if b.name.startswith("O")]
        except Exception:
            from MDAnalysis.lib.distances import distance_array
            d = distance_array(p.position[None, :], o_atoms.positions)[0]
            idx += [o_atoms[j].index for j in np.where(d < 1.8)[0]]
        groups[p.name] = residue.universe.atoms[sorted(set(idx))]
    return groups


def _debounce(b, min_len):
    """Drop runs shorter than min_len so single-frame flickers aren't events."""
    b = np.asarray(b, dtype=bool).copy()
    if min_len <= 1 or len(b) == 0:
        return b
    i = 0
    while i < len(b):
        j = i
        while j < len(b) and b[j] == b[i]:
            j += 1
        if (j - i) < min_len and i > 0:
            b[i:j] = b[i - 1]
        i = j
    return b


def _bound_stats(bound, min_event):
    b = _debounce(bound, min_event)
    frac = float(b.mean()) if len(b) else 0.0
    longest, run = 0, 0
    for v in b:
        run = run + 1 if v else 0
        longest = max(longest, run)
    n_events = int(np.sum((~b[1:]) & b[:-1])) if len(b) > 1 else 0
    return frac, longest, n_events


def _contacted_resids(positions, prot, cutoff, box):
    pairs = capped_distance(positions, prot.positions, max_cutoff=cutoff,
                            box=box, return_distances=False)
    if len(pairs) == 0:
        return set()
    return set(prot.resids[np.unique(pairs[:, 1])])


def _bridged_resids(hg, cations, cutoff, box):
    o = hg.select_atoms("name O*")
    if len(o) == 0 or len(cations) == 0:
        return set()
    pairs = capped_distance(o.positions, cations.positions, max_cutoff=cutoff,
                            box=box, return_distances=False)
    if len(pairs) == 0:
        return set()
    return set(cations.resids[np.unique(pairs[:, 1])])


# ─────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────

def run_pip_binding_quality(
    u,
    results_dir,
    channel_type,
    pip_resids,
    pip_mapping=None,
    start_frame=0,
    end_frame=None,
    cutoff=4.0,
    sb_cutoff=4.0,
    min_event=5,
    auto_thresh=0.10,
    track_residues=None,
    stride=1,
    force_rerun=False,
):
    out = Path(results_dir) / "pip_binding_quality"
    out.mkdir(parents=True, exist_ok=True)
    summary_csv = out / "pip_summary.csv"

    if summary_csv.is_file() and not force_rerun:
        print(f"[pip_binding_quality] {summary_csv.name} exists -- skipping "
              f"(force_rerun=True to recompute)")
        return pd.read_csv(summary_csv)

    pip_mapping = pip_mapping or {}
    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    # ---- selections ----
    lipids = []
    for resid in pip_resids:
        res = u.select_atoms(f"resid {resid}").residues
        if len(res) == 0:
            print(f"[pip_binding_quality] WARNING: resid {resid} matched nothing -- skipped")
            continue
        res = res[0]
        groups = _phosphate_groups(res)
        if not groups:
            print(f"[pip_binding_quality] WARNING: no P atoms in resid {resid} -- skipped")
            continue
        all_p = u.atoms[sorted({i for g in groups.values() for i in g.indices})]
        sub = pip_mapping.get(resid, "?")
        lipids.append(dict(resid=resid, subunit=sub, label=f"PIP{resid} ({sub})",
                           groups=groups, hg=all_p))

    if not lipids:
        raise RuntimeError("[pip_binding_quality] no PIP residues found -- check pip_resids")

    prot = u.select_atoms(PROT_SEL)
    cations = u.select_atoms(CATION_SEL)
    ca = u.select_atoms("protein and name CA")
    if len(prot) == 0 or len(ca) == 0:
        raise RuntimeError("[pip_binding_quality] empty protein selection")

    phos_names = sorted({n for l in lipids for n in l["groups"]})
    print(f"[pip_binding_quality] {len(lipids)} PIP lipids: "
          f"{', '.join(l['label'] for l in lipids)}")
    print(f"[pip_binding_quality] phosphates tracked: {', '.join(phos_names)} | "
          f"frames {start_frame}-{end_frame} | contact {cutoff} A, "
          f"salt bridge {sb_cutoff} A")

    track = set(track_residues) if track_residues else None

    # ---- frame loop ----
    ca_ref = ca.positions - ca.center_of_mass()
    hg_ref = None
    contact_counts = [dict() for _ in lipids]
    sb_counts = [dict() for _ in lipids]
    rows = []
    n_frames = 0

    for ts in tqdm(u.trajectory[start_frame:end_frame + 1:stride],
                   desc="pip_binding_quality"):
        box = _safe_box(ts)
        prot_com = ca.center_of_mass()
        R, _ = rotation_matrix(ca.positions - prot_com, ca_ref)
        if hg_ref is None:
            hg_ref = [R.dot(l["hg"].center_of_mass() - prot_com) for l in lipids]

        for i, l in enumerate(lipids):
            rs = _contacted_resids(l["hg"].positions, prot, cutoff, box)
            if track is not None:
                rs &= track
            for r in rs:
                contact_counts[i][r] = contact_counts[i].get(r, 0) + 1

            per_phos = {}
            sb_all = set()
            for pname, g in l["groups"].items():
                br = _bridged_resids(g, cations, sb_cutoff, box)
                if track is not None:
                    br &= track
                per_phos[pname] = len(br)
                sb_all |= br
            for r in sb_all:
                sb_counts[i][r] = sb_counts[i].get(r, 0) + 1

            drift = float(np.linalg.norm(
                R.dot(l["hg"].center_of_mass() - prot_com) - hg_ref[i]))

            row = dict(frame=ts.frame, pip=l["label"], subunit=l["subunit"],
                       n_contact_residues=len(rs), n_salt_bridges=len(sb_all),
                       headgroup_drift=round(drift, 3))
            for pname in phos_names:
                row[f"n_sb_{pname}"] = per_phos.get(pname, 0)
            rows.append(row)
        n_frames += 1

    df_ts = pd.DataFrame(rows)
    df_ts.to_csv(out / "pip_timeseries.csv", index=False)

    # ---- occupancy ----
    if track is None:
        track = set()
        for cc in contact_counts:
            track |= {r for r, n in cc.items() if n / n_frames >= auto_thresh}
        print(f"[pip_binding_quality] auto-detected {len(track)} contacting residues "
              f"(>= {auto_thresh:.0%} of frames)")
    resids = sorted(track)
    labels = {r: to_pdb(r, channel_type) for r in resids}

    occ_rows = []
    for i, l in enumerate(lipids):
        for r in resids:
            occ_rows.append(dict(
                pip=l["label"], subunit=l["subunit"], resid=r, residue=labels[r],
                contact_occupancy=round(contact_counts[i].get(r, 0) / n_frames, 4),
                saltbridge_occupancy=round(sb_counts[i].get(r, 0) / n_frames, 4)))
    df_occ = pd.DataFrame(occ_rows)
    df_occ["res_num"] = df_occ.residue.map(lambda s: _split_pdb(s)[0])
    df_occ["res_sub"] = df_occ.residue.map(lambda s: _split_pdb(s)[1])
    df_occ.to_csv(out / "pip_occupancy.csv", index=False)

    # ---- four sites on a common residue axis ----
    comp = (df_occ.groupby(["pip", "res_num"])
                  .contact_occupancy.max().unstack(fill_value=0.0))
    try:
        comp = comp[sorted(comp.columns, key=int)]
    except ValueError:
        pass
    comp.to_csv(out / "pip_interface_comparison.csv")

    # ---- per-PIP verdict ----
    sum_rows = []
    for l in lipids:
        sub_ts = df_ts[df_ts.pip == l["label"]].sort_values("frame")
        frac, longest, n_ev = _bound_stats(sub_ts.n_salt_bridges.values >= 1, min_event)
        occ = df_occ[df_occ.pip == l["label"]]
        strong = occ[occ.contact_occupancy > 0.8].sort_values(
            "contact_occupancy", ascending=False)
        row = dict(
            pip=l["label"], resid=l["resid"], subunit=l["subunit"],
            n_frames=n_frames,
            frac_bound=round(frac, 4),
            longest_bound_frames=longest,
            n_unbinding_events=n_ev,
            mean_salt_bridges=round(float(sub_ts.n_salt_bridges.mean()), 3),
            mean_contact_residues=round(float(sub_ts.n_contact_residues.mean()), 3),
            n_residues_occ_gt80=len(strong),
            top_residues=";".join(strong.residue.head(5)),
            drift_mean=round(float(sub_ts.headgroup_drift.mean()), 3),
            drift_final=round(float(sub_ts.headgroup_drift.values[-1]), 3),
            drift_max=round(float(sub_ts.headgroup_drift.max()), 3))
        for pname in phos_names:
            row[f"mean_sb_{pname}"] = round(float(sub_ts[f"n_sb_{pname}"].mean()), 3)
        sum_rows.append(row)
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(summary_csv, index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print("\n=== PIP binding summary (per site) ===")
        print(df_sum.to_string(index=False))
        print("\n=== Contact occupancy by equivalent residue ===")
        print(comp.round(2).to_string())
        spread = (comp.max(axis=0) - comp.min(axis=0)).sort_values(ascending=False)
        big = spread[spread > 0.3]
        if len(big):
            print("\nResidues where the four sites disagree most (max-min > 0.3):")
            print("  " + ", ".join(f"{r}: {v:.2f}" for r, v in big.head(8).items()))

    _plots(df_ts, df_occ, comp, resids, labels, out)
    print(f"\n[pip_binding_quality] wrote {out}")
    return df_sum


# ─────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────

def _heatmap(values, xlabels, ylabels, title, xaxis, cbar, path):
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(xlabels)),
                                    max(3, 0.55 * len(ylabels))))
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel(xaxis)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plots(df_ts, df_occ, comp, resids, labels, out):
    if resids:
        piv = df_occ.pivot(index="pip", columns="residue",
                           values="contact_occupancy")[[labels[r] for r in resids]]
        _heatmap(piv.values, list(piv.columns), list(piv.index),
                 "PIP contact occupancy", "residue (PDB numbering)",
                 "fraction of frames", out / "pip_occupancy_heatmap.png")

    if len(comp.columns):
        _heatmap(comp.values, list(comp.columns), list(comp.index),
                 "PIP sites on a common residue axis",
                 "residue (subunit collapsed)", "contact occupancy",
                 out / "pip_interface_comparison.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True)
    for name, sub in df_ts.groupby("pip"):
        sub = sub.sort_values("frame")
        ax1.plot(sub.frame, sub.n_salt_bridges, lw=0.9, alpha=0.85, label=name)
        ax2.plot(sub.frame, sub.headgroup_drift, lw=0.9, alpha=0.85, label=name)
    ax1.axhline(df_ts.n_salt_bridges.mean(), color="red", ls="--", lw=1.2,
                label="mean (all sites)")
    ax1.set_ylabel("salt bridges")
    ax1.set_title("PIP binding over time")
    ax2.set_ylabel("headgroup drift (Å)")
    ax2.set_xlabel("frame")
    for ax in (ax1, ax2):
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(out / "pip_binding_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()