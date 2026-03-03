"""
asn_axis_analysis.py
====================
For every frame of the MD trajectory, computes the perpendicular distance
of OD1 and ND2 (for each ASN residue) to the instantaneous pore axis.

The pore axis is defined every frame as the line connecting:
  - P_top : centre of mass of CA atoms of sf_residues
  - P_bot : centre of mass of CA atoms of hbc_residues

Perpendicular distance from point Q to the infinite line through P_top/P_bot:
  d = | (Q - P_top) × axis_unit | 

Outputs (saved to results_dir/asn_axis_analysis/):
  - asn_axis_distances.csv      : frame-by-frame distances + winner
  - hist_ASN<resid>.png         : distance histograms (OD1 vs ND2) per residue
  - winner_counts.png           : bar chart of winner counts per residue
  - summary.txt                 : text summary

Usage (called from main.py):
    from asn_axis_analysis import run_asn_axis_analysis
    run_asn_axis_analysis(u, sf_residues, hbc_residues, asn_residues,
                          results_dir, channel_type="G12",
                          start_frame=0, end_frame=None)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import MDAnalysis as mda
except ImportError:
    raise ImportError("MDAnalysis is required for asn_axis_analysis.py")


# ── PDB numbering helper (mirrors your existing converter) ────────────────────

def convert_to_pdb_numbering(residue_id, channel_type="G12"):
    if channel_type in ("G12", "G12_GAT"):
        asn_asp_mapping = {454: "184.A", 130: "184.B", 779: "184.C", 1105: "173.G1"}
        return asn_asp_mapping.get(residue_id, str(residue_id))
    elif channel_type == "G2":
        asn_mapping = {130: "184.A", 458: "184.C", 786: "184.B", 1114: "184.D"}
        return asn_mapping.get(residue_id, str(residue_id))
    elif channel_type == "G12_ML":
        asn_asp_mapping = {781: "184.A", 1106: "184.B", 456: "184.C", 131: "173.G1"}
        return asn_asp_mapping.get(residue_id, str(residue_id))
    return str(residue_id)


# ── Core geometry ─────────────────────────────────────────────────────────────

def perpendicular_distance_to_axis(point, axis_origin, axis_unit):
    """
    Perpendicular distance from `point` to the infinite line defined by
    `axis_origin` and direction `axis_unit` (unit vector).

    d = | (point - axis_origin) × axis_unit |
    """
    v = point - axis_origin
    cross = np.cross(v, axis_unit)
    return float(np.linalg.norm(cross))


def compute_axis(u, sf_residues, hbc_residues, use_ca=False):
    """
    Compute pore axis for the current frame.

    Parameters
    ----------
    use_ca : bool
        If True,  use only CA atoms to define ring centres.
        If False, use centre of mass of all atoms of each residue (default).

    Returns
    -------
    axis_origin : np.array [3]   (P_top, centre of sf ring)
    axis_unit   : np.array [3]   unit vector from P_top to P_bot
    """
    sf_resid_str  = " ".join(str(r) for r in sf_residues)
    hbc_resid_str = " ".join(str(r) for r in hbc_residues)

    if use_ca:
        sf_sel  = u.select_atoms(f"resid {sf_resid_str}  and name CA")
        hbc_sel = u.select_atoms(f"resid {hbc_resid_str} and name CA")
    else:
        sf_sel  = u.select_atoms(f"resid {sf_resid_str}")
        hbc_sel = u.select_atoms(f"resid {hbc_resid_str}")

    p_top = sf_sel.center_of_mass()   # top ring centre
    p_bot = hbc_sel.center_of_mass()  # bottom ring centre

    direction = p_bot - p_top
    length    = np.linalg.norm(direction)
    if length < 1e-6:
        raise ValueError("sf and hbc ring centers are coincident — check residue selection")

    axis_unit = direction / length
    return p_top, axis_unit


# ── Main analysis loop ────────────────────────────────────────────────────────

def run_asn_axis_analysis(
    u,
    sf_residues,
    hbc_residues,
    asn_residues,
    results_dir,
    channel_type = "G12",
    start_frame  = 0,
    end_frame    = None,
    use_ca       = False,
):
    """
    Run ASN axis proximity analysis over the full trajectory.

    Parameters
    ----------
    u            : MDAnalysis.Universe
    sf_residues  : list of int   residue IDs of selectivity filter ring
    hbc_residues : list of int   residue IDs of HBC ring
    asn_residues : list of int   residue IDs of ASN gate residues
    results_dir  : Path or str
    channel_type : str
    start_frame  : int
    end_frame    : int or None   (None = last frame)
    use_ca       : bool
        False (default) — axis defined by COM of ALL atoms of sf/hbc residues
        True            — axis defined by COM of CA atoms only
    """
    results_dir = Path(results_dir) / "asn_axis_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    n_frames = end_frame - start_frame + 1

    print(f"\n{'='*65}")
    print(f"ASN AXIS PROXIMITY ANALYSIS")
    print(f"{'='*65}")
    axis_method = "CA atoms only" if use_ca else "COM of all atoms (default)"
    print(f"  Frames       : {start_frame} → {end_frame}  ({n_frames} frames)")
    print(f"  SF residues  : {sf_residues}")
    print(f"  HBC residues : {hbc_residues}")
    print(f"  ASN residues : {asn_residues}")
    print(f"  Channel type : {channel_type}")
    print(f"  Axis method  : {axis_method}")
    print(f"  Output dir   : {results_dir}")

    # Pre-select atom groups so we don't re-select every frame
    pdb_labels = {r: convert_to_pdb_numbering(r, channel_type) for r in asn_residues}

    asn_atoms = {}
    for resid in asn_residues:
        od1 = u.select_atoms(f"resid {resid} and name OD1")
        nd2 = u.select_atoms(f"resid {resid} and name ND2")
        if len(od1) == 0 or len(nd2) == 0:
            print(f"  WARNING: OD1 or ND2 not found for ASN {resid} — skipping")
            continue
        asn_atoms[resid] = {"OD1": od1, "ND2": nd2}

    if not asn_atoms:
        print("  ERROR: No ASN atoms found. Aborting.")
        return

    # ── Frame loop ────────────────────────────────────────────────────────────
    records = []   # list of dicts, one per (frame × residue)

    for ts in tqdm(
        u.trajectory[start_frame : end_frame + 1],
        total=n_frames,
        desc="Computing ASN axis distances",
        unit="frame",
    ):
        frame = int(ts.frame)

        # Recompute axis this frame
        try:
            axis_origin, axis_unit = compute_axis(u, sf_residues, hbc_residues, use_ca=use_ca)
        except ValueError as e:
            print(f"  Frame {frame}: axis error — {e}")
            continue

        for resid, atoms in asn_atoms.items():
            pos_od1 = atoms["OD1"].positions[0]
            pos_nd2 = atoms["ND2"].positions[0]

            d_od1 = perpendicular_distance_to_axis(pos_od1, axis_origin, axis_unit)
            d_nd2 = perpendicular_distance_to_axis(pos_nd2, axis_origin, axis_unit)

            winner = "OD1" if d_od1 < d_nd2 else "ND2"

            records.append({
                "frame":       frame,
                "resid":       resid,
                "pdb_label":   pdb_labels[resid],
                "d_OD1":       d_od1,
                "d_ND2":       d_nd2,
                "closer_atom": winner,
                "delta":       d_od1 - d_nd2,   # positive = ND2 closer, negative = OD1 closer
            })

    if not records:
        print("  ERROR: No records collected. Aborting.")
        return

    df = pd.DataFrame(records)

    # Save CSV
    csv_path = results_dir / "asn_axis_distances.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}  ({len(df)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_distance_histograms(df, asn_residues, pdb_labels, results_dir)
    _plot_winner_counts(df, asn_residues, pdb_labels, results_dir)
    _plot_delta_histograms(df, asn_residues, pdb_labels, results_dir)
    _write_summary(df, asn_residues, pdb_labels, results_dir)

    print(f"\n  ✓ ASN axis analysis complete → {results_dir}/")
    return df


# ── Plot functions ────────────────────────────────────────────────────────────

TITLE_SIZE  = 18
LABEL_SIZE  = 15
TICK_SIZE   = 13
LEGEND_SIZE = 13

# Fixed display order for panels (PDB label → display position)
PANEL_ORDER = ["184.B", "184.C", "184.A", "173.G1"]

def _sort_residues(asn_residues, pdb_labels):
    """Sort residues by the fixed PANEL_ORDER for consistent plot layout."""
    labeled = [(r, pdb_labels[r]) for r in asn_residues if r in pdb_labels]
    def sort_key(item):
        try:
            return PANEL_ORDER.index(item[1])
        except ValueError:
            return 999  # unknown labels go last
    return [r for r, _ in sorted(labeled, key=sort_key)]


def _make_grid(n, ncols=2):
    """Create a grid of n axes with ncols columns."""
    import math
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 7 * nrows),
                              squeeze=False)
    ax_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for ax in ax_flat[n:]:
        ax.set_visible(False)
    return fig, ax_flat[:n]


def _plot_distance_histograms(df, asn_residues, pdb_labels, results_dir):
    """
    For each ASN: overlaid histogram of OD1 vs ND2 perpendicular distances.
    """
    valid = _sort_residues([r for r in asn_residues if r in df["resid"].values], pdb_labels)
    fig, axes = _make_grid(len(valid), ncols=2)

    for ax, resid in zip(axes, valid):
        sub = df[df["resid"] == resid]
        pdb = pdb_labels[resid]

        ax.hist(sub["d_OD1"], bins=60, alpha=0.6, color="#e74c3c",
                label="OD1", density=True)
        ax.hist(sub["d_ND2"], bins=60, alpha=0.6, color="#3498db",
                label="ND2", density=True)

        ax.axvline(sub["d_OD1"].mean(), color="#e74c3c", linestyle="--",
                   linewidth=2, label=f"OD1 mean={sub['d_OD1'].mean():.2f} Å")
        ax.axvline(sub["d_ND2"].mean(), color="#3498db", linestyle="--",
                   linewidth=2, label=f"ND2 mean={sub['d_ND2'].mean():.2f} Å")

        ax.set_title(f"ASN {pdb}  —  distance to pore axis",
                     fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Perpendicular distance to axis  (Å)", fontsize=LABEL_SIZE)
        ax.set_ylabel("Density", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("OD1 vs ND2 Distance to Pore Axis — All ASN Residues",
                 fontsize=TITLE_SIZE + 2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = results_dir / "hist_OD1_ND2_distances.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_winner_counts(df, asn_residues, pdb_labels, results_dir):
    """
    Bar chart: how many frames OD1 was closer vs ND2 was closer, per residue.
    """
    valid = _sort_residues([r for r in asn_residues if r in df["resid"].values], pdb_labels)
    fig, axes = _make_grid(len(valid), ncols=2)

    for ax, resid in zip(axes, valid):
        sub = df[df["resid"] == resid]
        pdb = pdb_labels[resid]
        total = len(sub)

        od1_count = (sub["closer_atom"] == "OD1").sum()
        nd2_count = (sub["closer_atom"] == "ND2").sum()

        bars = ax.bar(
            ["OD1 closer\n(O toward axis)", "ND2 closer\n(N toward axis)"],
            [od1_count, nd2_count],
            color=["#e74c3c", "#3498db"],
            alpha=0.85, edgecolor="black", linewidth=1.5,
        )
        for bar, count in zip(bars, [od1_count, nd2_count]):
            pct = 100 * count / total
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="bottom",
                    fontsize=TICK_SIZE, fontweight="bold")

        ax.set_title(f"ASN {pdb}  —  which atom faces the pore axis?",
                     fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_ylabel("Number of frames", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("ASN Amide Orientation Relative to Pore Axis",
                 fontsize=TITLE_SIZE + 2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = results_dir / "winner_counts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_delta_histograms(df, asn_residues, pdb_labels, results_dir):
    """
    Histogram of delta = d_OD1 - d_ND2 per residue.
    delta < 0 → OD1 closer to axis (oxygen faces pore)
    delta > 0 → ND2 closer to axis (nitrogen/HD faces pore)
    """
    valid = _sort_residues([r for r in asn_residues if r in df["resid"].values], pdb_labels)
    fig, axes = _make_grid(len(valid), ncols=2)

    for ax, resid in zip(axes, valid):
        sub = df[df["resid"] == resid]
        pdb = pdb_labels[resid]

        ax.hist(sub["delta"], bins=60, color="#8e44ad", alpha=0.75, density=True)
        ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="0 (equal)")
        ax.axvline(sub["delta"].mean(), color="red", linestyle="--",
                   linewidth=2, label=f"mean={sub['delta'].mean():.2f} Å")

        # Shade regions
        ax.axvspan(sub["delta"].min(), 0, alpha=0.07, color="#e74c3c",
                   label="OD1 closer (O→axis)")
        ax.axvspan(0, sub["delta"].max(), alpha=0.07, color="#3498db",
                   label="ND2 closer (N→axis)")

        ax.set_title(f"ASN {pdb}  —  Δd = d(OD1) − d(ND2)",
                     fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("Δd  (Å)     [ negative = OD1 faces axis ]",
                      fontsize=LABEL_SIZE)
        ax.set_ylabel("Density", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE - 1)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("ASN Amide Asymmetry  (Δd = d_OD1 − d_ND2)",
                 fontsize=TITLE_SIZE + 2, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = results_dir / "delta_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _write_summary(df, asn_residues, pdb_labels, results_dir):
    """Write a plain-text summary of the analysis."""
    path = results_dir / "summary.txt"
    lines = [
        "ASN AXIS PROXIMITY ANALYSIS — SUMMARY",
        "=" * 65,
        "",
        "Δd = d(OD1) − d(ND2)",
        "  Δd < 0 → OD1 is closer to axis (carbonyl O faces pore)",
        "  Δd > 0 → ND2 is closer to axis (amide N/HD faces pore)",
        "",
        "=" * 65,
        "",
    ]

    for resid in asn_residues:
        sub = df[df["resid"] == resid]
        if sub.empty:
            continue
        pdb = pdb_labels[resid]
        total = len(sub)
        od1_closer = (sub["closer_atom"] == "OD1").sum()
        nd2_closer = (sub["closer_atom"] == "ND2").sum()

        lines += [  
            f"ASN {pdb}  (resid {resid})",
            f"  Total frames      : {total}",
            f"  OD1 mean dist     : {sub['d_OD1'].mean():.3f} ± {sub['d_OD1'].std():.3f} Å",
            f"  ND2 mean dist     : {sub['d_ND2'].mean():.3f} ± {sub['d_ND2'].std():.3f} Å",
            f"  Mean Δd           : {sub['delta'].mean():.3f} ± {sub['delta'].std():.3f} Å",
            f"  OD1 closer        : {od1_closer:6d} frames ({100*od1_closer/total:.1f}%)",
            f"  ND2 closer        : {nd2_closer:6d} frames ({100*nd2_closer/total:.1f}%)",
            f"  Dominant orientation: {'OD1 faces pore (O inward)' if od1_closer > nd2_closer else 'ND2 faces pore (N/HD inward)'}",
            "",
        ]

    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")