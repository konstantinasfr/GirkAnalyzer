"""
neighbor_analysis.py
====================
Analyzes neighboring residue contacts around user-defined "focus" residues
(e.g. GLU subunits) across MD trajectory frames.

Usage
-----
python neighbor_analysis.py \
    --topology  channel.prmtop \
    --trajectory channel.nc \
    --focus      "resid 141 or resid 177" \   # MDAnalysis selection string
    --radius     5.0 \                         # Å cutoff for neighbor detection
    --start      0 \                           # first frame (0-indexed)
    --end        300 \                         # last frame (exclusive), None = all
    --output_dir results_neighbors

python3 neighbor_analysis.py \
    --topology  com.prmtop \
    --trajectory protein.nc \
    --focus      "resid 141 or resid 177" \   # MDAnalysis selection string
    --radius     5.0 \                         # Å cutoff for neighbor detection
    --start      0 \                           # first frame (0-indexed)
    --end        300 \                         # last frame (exclusive), None = all
    --output_dir results_neighbors

Requirements
------------
pip install MDAnalysis numpy pandas matplotlib seaborn
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import MDAnalysis as mda
from MDAnalysis.analysis import distances as mdadist
from MDAnalysis.lib.distances import calc_dihedrals

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

FS = 14   # global font size (matches your force_calculator.py preference)

def make_dir(path):
    os.makedirs(path, exist_ok=True)


def pdb_label(resname, resid):
    """Return a compact label like GLU141."""
    return f"{resname}{resid}"


# ──────────────────────────────────────────────────────────────────────────────
# Dihedral helpers
# ──────────────────────────────────────────────────────────────────────────────

CHI_ATOMS = {
    # chi1: N-CA-CB-XG
    "GLU": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD"]},
    "ASP": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","OD1"]},
    "ASN": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","OD1"]},
    "LYS": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD"]},
    "ARG": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD"]},
    "SER": {"chi1": ["N","CA","CB","OG"],  "chi2": None},
    "THR": {"chi1": ["N","CA","CB","OG1"], "chi2": None},
    "PHE": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD1"]},
    "TYR": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD1"]},
    "TRP": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD1"]},
    "HIS": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","ND1"]},
    "ILE": {"chi1": ["N","CA","CB","CG1"], "chi2": ["CA","CB","CG1","CD"]},
    "LEU": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD1"]},
    "MET": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","SD"]},
    "GLN": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD"]},
    "VAL": {"chi1": ["N","CA","CB","CG1"], "chi2": None},
    "CYS": {"chi1": ["N","CA","CB","SG"],  "chi2": None},
    "PRO": {"chi1": ["N","CA","CB","CG"],  "chi2": ["CA","CB","CG","CD"]},
}

def get_chi(residue, chi_name):
    """Return chi angle in degrees for a given MDAnalysis Residue, or NaN."""
    resname = residue.resname.upper()
    if resname not in CHI_ATOMS:
        return np.nan
    atom_names = CHI_ATOMS[resname].get(chi_name)
    if atom_names is None:
        return np.nan
    atoms = []
    for name in atom_names:
        sel = residue.atoms.select_atoms(f"name {name}")
        if len(sel) == 0:
            return np.nan
        atoms.append(sel.positions[0])
    a1, a2, a3, a4 = [p[np.newaxis, :] for p in atoms]
    angle = calc_dihedrals(a1, a2, a3, a4)
    return np.degrees(angle[0])


# ──────────────────────────────────────────────────────────────────────────────
# Hydrogen-bond detection (simple geometric criterion)
# ──────────────────────────────────────────────────────────────────────────────

HBOND_DONORS = {"N", "O", "S"}   # heavy-atom donors (must have H attached)
HBOND_ACCEPTORS = {"N", "O", "S"}

def count_hbonds_between(res_a, res_b, dist_cutoff=3.5, angle_cutoff=120.0):
    """
    Count hydrogen bonds between two residues using a simple D–H···A criterion.
    Returns integer count.
    """
    count = 0
    # collect donors (heavy atom + H) from res_a, acceptors from res_b and vice-versa
    for donor_res, acceptor_res in [(res_a, res_b), (res_b, res_a)]:
        for h_atom in donor_res.atoms:
            if h_atom.name.startswith("H"):
                # find bonded heavy atom
                bonded_heavy = [a for a in h_atom.bonded_atoms
                                if a.element.upper() in HBOND_DONORS]
                if not bonded_heavy:
                    continue
                D = bonded_heavy[0].position
                H = h_atom.position
                for acc_atom in acceptor_res.atoms:
                    if acc_atom.element.upper() not in HBOND_ACCEPTORS:
                        continue
                    A = acc_atom.position
                    dHA = np.linalg.norm(H - A)
                    if dHA > dist_cutoff:
                        continue
                    # D-H-A angle
                    v1 = D - H
                    v2 = A - H
                    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                    angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                    if angle >= angle_cutoff:
                        count += 1
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────────────

def auto_detect_neighbors(u, focus_resids, radius):
    """
    Across all frames find every residue that ever has AT LEAST ONE atom
    within `radius` Å of any atom of any focus residue.
    Returns sorted list of resids (excluding focus residues themselves).
    """
    print(f"  Auto-detecting neighbors within {radius} Å "
          f"(scanning all frames) ...")
    neighbor_set = set()
    focus_atoms = u.select_atoms(
        " or ".join([f"resid {r}" for r in focus_resids])
    )
    all_heavy = u.select_atoms("not (name H*) and protein")

    for ts in u.trajectory:
        pairs, _ = mdadist.capped_distance(
            focus_atoms.positions,
            all_heavy.positions,
            max_cutoff=radius,
            return_distances=False,
            box=u.dimensions
        )
        if len(pairs) > 0:
            for _, j in pairs:
                resid = all_heavy[j].resid
                if resid not in focus_resids:
                    neighbor_set.add(resid)

    return sorted(neighbor_set)


def analyze(topology, trajectory, focus_sel, radius,
            start, end, output_dir):

    make_dir(output_dir)
    u = mda.Universe(topology, trajectory)

    # ── resolve focus residues ──────────────────────────────────────────────
    focus_atoms_sel = u.select_atoms(focus_sel)
    if len(focus_atoms_sel) == 0:
        raise ValueError(f"Focus selection '{focus_sel}' returned 0 atoms. "
                         "Check your MDAnalysis selection syntax.")
    focus_resids = sorted(set(focus_atoms_sel.resids))
    focus_residues = [u.select_atoms(f"resid {r}").residues[0]
                      for r in focus_resids]

    print(f"\nFocus residues  : "
          + ", ".join([pdb_label(r.resname, r.resid) for r in focus_residues]))

    # ── frame range ────────────────────────────────────────────────────────
    n_frames_total = len(u.trajectory)
    start = start or 0
    end   = end   or n_frames_total
    end   = min(end, n_frames_total)
    frames = range(start, end)
    print(f"Frames          : {start} → {end-1}  ({len(frames)} frames)")

    # ── detect neighbors (full traj) ───────────────────────────────────────
    u.trajectory[start]   # reset
    neighbor_resids = auto_detect_neighbors(u, set(focus_resids), radius)
    neighbor_residues = [u.select_atoms(f"resid {r}").residues[0]
                         for r in neighbor_resids]

    print(f"\nNeighbors found : {len(neighbor_resids)}")
    for r in neighbor_residues:
        print(f"  resid {r.resid:5d}  {r.resname}")

    # write neighbor list
    with open(os.path.join(output_dir, "neighbor_list.txt"), "w") as fh:
        fh.write("# Residues found within {:.1f} Å of focus residues\n".format(radius))
        fh.write(f"# Focus: {', '.join([pdb_label(r.resname, r.resid) for r in focus_residues])}\n\n")
        fh.write(f"{'resid':>8}  {'resname':>8}\n")
        for r in neighbor_residues:
            fh.write(f"{r.resid:>8}  {r.resname:>8}\n")

    # ── per-frame loop ─────────────────────────────────────────────────────
    records = []

    for ts in u.trajectory[start:end]:
        frame = ts.frame
        row = {"frame": frame}

        # 1. min heavy-atom sidechain distance: focus × neighbor
        for fr in focus_residues:
            # sidechain heavy atoms only (exclude backbone N,CA,C,O)
            fr_sc = fr.atoms.select_atoms(
                "not name N CA C O H HA and not name H*"
            )
            if len(fr_sc) == 0:
                fr_sc = fr.atoms  # fallback

            for nr in neighbor_residues:
                nr_sc = nr.atoms.select_atoms(
                    "not name N CA C O H HA and not name H*"
                )
                if len(nr_sc) == 0:
                    nr_sc = nr.atoms

                # min distance
                d_mat = mdadist.distance_array(fr_sc.positions,
                                               nr_sc.positions,
                                               box=u.dimensions)
                min_d = d_mat.min()

                # Cα–Cα distance
                ca_f = fr.atoms.select_atoms("name CA")
                ca_n = nr.atoms.select_atoms("name CA")
                if len(ca_f) and len(ca_n):
                    ca_dist = np.linalg.norm(ca_f.positions[0] - ca_n.positions[0])
                else:
                    ca_dist = np.nan

                key = (f"F{fr.resid}_N{nr.resid}")
                row[f"dist_sc_{key}"]   = round(min_d,  3)
                row[f"dist_ca_{key}"]   = round(ca_dist, 3)

                # 2. hydrogen bonds
                try:
                    hb = count_hbonds_between(fr, nr)
                except Exception:
                    hb = 0
                row[f"hbonds_{key}"] = hb

        # 3. chi1, chi2 angles for each focus residue
        for fr in focus_residues:
            row[f"chi1_F{fr.resid}"] = round(get_chi(fr, "chi1"), 3) \
                if not np.isnan(get_chi(fr, "chi1")) else np.nan
            row[f"chi2_F{fr.resid}"] = round(get_chi(fr, "chi2"), 3) \
                if not np.isnan(get_chi(fr, "chi2")) else np.nan

        # 4. chi1, chi2 for each neighbor
        for nr in neighbor_residues:
            row[f"chi1_N{nr.resid}"] = round(get_chi(nr, "chi1"), 3) \
                if not np.isnan(get_chi(nr, "chi1")) else np.nan
            row[f"chi2_N{nr.resid}"] = round(get_chi(nr, "chi2"), 3) \
                if not np.isnan(get_chi(nr, "chi2")) else np.nan

        records.append(row)

        if (frame - start) % 50 == 0:
            print(f"  ... frame {frame}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "neighbor_timeseries.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved → {csv_path}")

    # ── plots ──────────────────────────────────────────────────────────────
    print("Generating plots ...")
    _plot_distances(df, focus_residues, neighbor_residues, output_dir)
    _plot_chi_angles(df, focus_residues, neighbor_residues, output_dir)
    _plot_hbonds(df, focus_residues, neighbor_residues, output_dir)
    _plot_focus_comparison(df, focus_residues, output_dir)

    print("\nDone. All output saved to:", output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "font.size":        FS,
        "axes.titlesize":   FS,
        "axes.labelsize":   FS,
        "xtick.labelsize":  FS - 1,
        "ytick.labelsize":  FS - 1,
        "legend.fontsize":  FS - 1,
    })


def _plot_distances(df, focus_res, neighbor_res, out_dir):
    """One figure per focus residue: min sidechain distance to each neighbor."""
    _style()
    for fr in focus_res:
        cols = [c for c in df.columns if c.startswith(f"dist_sc_F{fr.resid}_")]
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        for col in cols:
            nr_id = int(col.split("_N")[-1])
            nr = next((r for r in neighbor_res if r.resid == nr_id), None)
            label = pdb_label(nr.resname, nr.resid) if nr else str(nr_id)
            ax.plot(df["frame"], df[col], lw=1.2, label=label)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Min sidechain distance (Å)")
        ax.set_title(f"Sidechain distances to {pdb_label(fr.resname, fr.resid)}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=min(len(cols), 5), frameon=False)
        fig.tight_layout()
        fname = os.path.join(out_dir, f"dist_sc_F{fr.resid}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_chi_angles(df, focus_res, neighbor_res, out_dir):
    """Chi1 & chi2 over time for focus residues side by side."""
    _style()
    all_res = focus_res + neighbor_res
    for res in all_res:
        prefix  = "F" if res in focus_res else "N"
        c1_col  = f"chi1_{prefix}{res.resid}"
        c2_col  = f"chi2_{prefix}{res.resid}"
        has_c1  = c1_col in df.columns and df[c1_col].notna().any()
        has_c2  = c2_col in df.columns and df[c2_col].notna().any()
        if not (has_c1 or has_c2):
            continue

        n_panels = int(has_c1) + int(has_c2)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 3.5))
        if n_panels == 1:
            axes = [axes]
        idx = 0
        for chi, col, has in [("χ1", c1_col, has_c1), ("χ2", c2_col, has_c2)]:
            if not has:
                continue
            axes[idx].plot(df["frame"], df[col], lw=1.0, color="steelblue")
            axes[idx].set_xlabel("Frame")
            axes[idx].set_ylabel(f"{chi} (°)")
            axes[idx].set_ylim(-185, 185)
            axes[idx].yaxis.set_major_locator(ticker.MultipleLocator(60))
            axes[idx].set_title(f"{pdb_label(res.resname, res.resid)} {chi}")
            idx += 1
        fig.tight_layout()
        tag = "focus" if res in focus_res else "neigh"
        fname = os.path.join(out_dir, f"chi_{tag}_{res.resid}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_hbonds(df, focus_res, neighbor_res, out_dir):
    """Heatmap: mean H-bond count (focus × neighbor)."""
    _style()
    hb_cols = [c for c in df.columns if c.startswith("hbonds_")]
    if not hb_cols:
        return

    focus_ids  = [fr.resid for fr in focus_res]
    neigh_ids  = [nr.resid for nr in neighbor_res]
    matrix     = np.zeros((len(focus_ids), len(neigh_ids)))

    for i, fid in enumerate(focus_ids):
        for j, nid in enumerate(neigh_ids):
            col = f"hbonds_F{fid}_N{nid}"
            if col in df.columns:
                matrix[i, j] = df[col].mean()

    fig, ax = plt.subplots(figsize=(max(6, len(neigh_ids) * 0.6 + 2),
                                    max(3, len(focus_ids) * 0.6 + 1.5)))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[pdb_label(r.resname, r.resid) for r in neighbor_res],
        yticklabels=[pdb_label(r.resname, r.resid) for r in focus_res],
        cmap="YlOrRd", annot=True, fmt=".2f",
        linewidths=0.5, linecolor="lightgray",
        cbar_kws={"label": "Mean H-bonds / frame"}
    )
    ax.set_title("Average hydrogen bonds (focus × neighbor)")
    ax.set_xlabel("Neighbor residue")
    ax.set_ylabel("Focus residue")
    fig.tight_layout()
    fname = os.path.join(out_dir, "hbond_heatmap.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # also stacked area over time
    fig, axes = plt.subplots(len(focus_ids), 1,
                             figsize=(10, 3 * len(focus_ids)),
                             sharex=True)
    if len(focus_ids) == 1:
        axes = [axes]
    for ax, fid, fr in zip(axes, focus_ids, focus_res):
        for nr in neighbor_res:
            col = f"hbonds_F{fid}_N{nr.resid}"
            if col in df.columns:
                ax.plot(df["frame"], df[col], lw=1.2,
                        label=pdb_label(nr.resname, nr.resid))
        ax.set_ylabel("H-bonds")
        ax.set_title(f"H-bonds from {pdb_label(fr.resname, fr.resid)}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25),
                  ncol=min(len(neighbor_res), 5), frameon=False)
    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    fname = os.path.join(out_dir, "hbond_timeseries.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_focus_comparison(df, focus_res, out_dir):
    """
    Direct comparison of the two focus residues (assumed to be the two GLUs):
    overlay their chi1, chi2, and the distance to their respective neighbors
    so you can see the symmetry breaking at a glance.
    """
    if len(focus_res) < 2:
        return
    _style()

    angles = ["chi1", "chi2"]
    colors = ["#d62728", "#1f77b4"]   # red / blue

    for angle in angles:
        cols = [f"{angle}_F{fr.resid}" for fr in focus_res]
        valid = [c for c in cols if c in df.columns and df[c].notna().any()]
        if len(valid) < 2:
            continue
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for col, fr, color in zip(valid, focus_res, colors):
            ax.plot(df["frame"], df[col], lw=1.2, color=color,
                    label=pdb_label(fr.resname, fr.resid))
        ax.set_xlabel("Frame")
        ax.set_ylabel(f"{angle.replace('chi','χ')} (°)")
        ax.set_title(f"Focus residue comparison — {angle.replace('chi','χ')}")
        ax.set_ylim(-185, 185)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(60))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"compare_{angle}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Summary statistics table
    stats_rows = []
    for fr in focus_res:
        for angle in angles:
            col = f"{angle}_F{fr.resid}"
            if col not in df.columns:
                continue
            s = df[col].dropna()
            stats_rows.append({
                "residue": pdb_label(fr.resname, fr.resid),
                "angle":   angle,
                "mean":    round(s.mean(), 2),
                "std":     round(s.std(),  2),
                "min":     round(s.min(),  2),
                "max":     round(s.max(),  2),
            })
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(
            os.path.join(out_dir, "focus_angle_stats.csv"), index=False
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Neighboring residue contact analysis for MD trajectories"
    )
    p.add_argument("--topology",    required=True,
                   help="Topology file (.prmtop)")
    p.add_argument("--trajectory",  required=True,
                   help="Trajectory file (.nc)")
    p.add_argument("--focus",       required=True,
                   help='MDAnalysis selection for focus residues, '
                        'e.g. "resid 141 or resid 177"')
    p.add_argument("--radius",      type=float, default=5.0,
                   help="Neighbor detection cutoff in Å (default: 5.0)")
    p.add_argument("--start",       type=int,   default=0,
                   help="First frame (0-indexed, default: 0)")
    p.add_argument("--end",         type=int,   default=None,
                   help="Last frame exclusive (default: all frames)")
    p.add_argument("--output_dir",  default="neighbor_analysis_output",
                   help="Directory to write results (default: neighbor_analysis_output)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(
        topology    = args.topology,
        trajectory  = args.trajectory,
        focus_sel   = args.focus,
        radius      = args.radius,
        start       = args.start,
        end         = args.end,
        output_dir  = args.output_dir,
    )
