#!/usr/bin/env python3
"""
Ion Proximity Analysis — per-frame, per-residue

For every frame, for each of the 8 key residues (GLU x4, ASP x1, ASN x4),
checks whether any K+ ion is within `threshold` Angstroms of the residue's
CG atom (the gamma carbon, closest heavy atom to the side-chain tip).

Stores per frame:
    - min_distance  : float  — distance to nearest ion
    - has_ion       : bool   — True if min_distance <= threshold

If dihedral_raw_data.pkl exists in the dihedral_analysis/ folder of the
same RUN, creates a combined file (same pattern as ligand_distance_analysis).

Output (saved in results_dir/ion_proximity_analysis/):
    ion_proximity_raw_data.pkl          — per-residue arrays
    ion_proximity_report.txt            — human-readable full frame table
    combined_ion_dihedrals.pkl          — merged with dihedral angles
    combined_ion_dihedrals.txt          — human-readable combined table

Called from main.py for every channel type that has GLU/ASP residues.

Usage:
    from ion_proximity_analysis import run_ion_proximity_analysis
    run_ion_proximity_analysis(
        u, results_dir,
        residue_config,          # dict mapping label → resid
        start_frame, end_frame,
        threshold=3.5
    )

    residue_config example (G12_ML):
        {
            '152.B': 98,  '152.C': 422, '152.A': 747, '141.D': 1073,
            '184.B': 130, '184.C': 454, '184.A': 779, '173.D': 1105,
        }
"""

import pickle
import numpy as np
from pathlib import Path


# ION_SELECTION — adjust if your topology uses different resname
ION_SELECTION = 'resname K+ K POT'


def run_ion_proximity_analysis(u, results_dir, glu_residues, asn_residues,
                                channel_type='G12',
                                start_frame=0, end_frame=None,
                                threshold=3.5):
    """
    Compute per-frame ion proximity for GLU and ASN/ASP residues.

    Parameters
    ----------
    u               : MDAnalysis Universe
    results_dir     : Path  — RUN results directory (e.g. RUN1/)
    glu_residues    : list  — 4 GLU resids e.g. [98, 422, 747, 1073]
    asn_residues    : list  — 4 ASN/ASP resids e.g. [130, 454, 779, 1105]
    channel_type    : str   — e.g. 'G12', 'G12_ML' (for pdb_label mapping)
    start_frame     : int
    end_frame       : int or None
    threshold       : float  Angstroms (default 3.5)
    """
    results_dir = Path(results_dir) / 'ion_proximity_analysis'
    results_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    n_frames = end_frame - start_frame + 1

    # use convert_to_pdb_numbering from dihedral_analysis_module — single source of truth
    try:
        from analysis.dihedral_analysis_module import convert_to_pdb_numbering
    except ImportError:
        from dihedral_analysis_module import convert_to_pdb_numbering

    residue_config = {}
    for resid in glu_residues:
        lbl = convert_to_pdb_numbering(resid, channel_type)
        residue_config[lbl] = resid
    for resid in asn_residues:
        lbl = convert_to_pdb_numbering(resid, channel_type)
        residue_config[lbl] = resid

    print("\n" + "=" * 70)
    print(f"ION PROXIMITY ANALYSIS  (threshold = {threshold} Å)")
    print("=" * 70)

    # ── Select ion atoms ──────────────────────────────────────────────────────
    ions = u.select_atoms(ION_SELECTION)
    if len(ions) == 0:
        print(f"  [ERROR] No ions found with selection '{ION_SELECTION}'")
        print(f"  Available resnames: {sorted(set(u.atoms.resnames))}")
        return None
    print(f"  Found {len(ions)} K+ ions")

    # ── Select relevant sidechain atoms per residue type ─────────────────────
    # GLU: OE1, OE2  (both negatively charged oxygens)
    # ASP: OD1, OD2  (both negatively charged oxygens)
    # ASN: OD1       (carbonyl oxygen — partial negative; ND2 is a donor, not relevant for K+)
    SIDECHAIN_BY_RESNAME = {
        'GLU': 'OE1 OE2',
        'GLN': 'OE1',
        'ASP': 'OD1 OD2',
        'ASN': 'OD1 HD21 HD22',
    }

    res_sels = {}
    for pdb_label, resid in sorted(residue_config.items()):
        # first find the resname
        probe = u.select_atoms(f'resid {resid}')
        if len(probe) == 0:
            print(f"  [WARN] {pdb_label} (resid {resid}): residue not found")
            continue
        resname = probe[0].resname

        atom_names = SIDECHAIN_BY_RESNAME.get(resname)
        if atom_names is None:
            print(f"  [WARN] {pdb_label} (resid {resid}, {resname}): "
                  f"no sidechain rule defined — skipping")
            continue

        sel = u.select_atoms(f'resid {resid} and name {atom_names}')
        if len(sel) == 0:
            print(f"  [WARN] {pdb_label} (resid {resid}, {resname}): "
                  f"atoms '{atom_names}' not found — skipping")
            continue

        res_sels[pdb_label] = sel
        print(f"  Found {pdb_label}: resid={resid}  resname={resname}  "
              f"atoms: {' '.join(a.name for a in sel)}")

    if not res_sels:
        print("  [ERROR] No sidechain atoms found — check residue_config resids")
        return None

    labels = sorted(res_sels.keys())

    # ── Per-frame arrays ──────────────────────────────────────────────────────
    min_dist = {lbl: np.zeros(n_frames) for lbl in labels}
    has_ion  = {lbl: np.zeros(n_frames, dtype=bool) for lbl in labels}

    print(f"\n  Analysing {n_frames} frames...")
    for fi, ts in enumerate(u.trajectory[start_frame: end_frame + 1]):
        if fi % 500 == 0:
            print(f"    Frame {start_frame + fi}/{end_frame}", end='\r')

        ion_pos = ions.positions   # shape (N_ions, 3)

        for lbl in labels:
            res_pos = res_sels[lbl].positions   # shape (N_atoms, 3)
            # min distance from any ion to any atom of the residue
            d_min = float(np.min(
                np.linalg.norm(
                    ion_pos[:, np.newaxis, :] - res_pos[np.newaxis, :, :],
                    axis=2
                )
            ))
            min_dist[lbl][fi] = d_min
            has_ion[lbl][fi]  = d_min <= threshold

    print(f"\n  Done.")

    # ── Build results dict ────────────────────────────────────────────────────
    results = {
        'threshold':   threshold,
        'n_frames':    n_frames,
        'start_frame': start_frame,
        'end_frame':   end_frame,
        'labels':      labels,
        'residue_config': residue_config,
        'min_dist':    min_dist,   # {pdb_label: array}
        'has_ion':     has_ion,    # {pdb_label: bool array}
    }

    # ── Save pkl ──────────────────────────────────────────────────────────────
    pkl_path = results_dir / 'ion_proximity_raw_data.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  ✅ Saved: {pkl_path}")

    # ── Save human-readable report ────────────────────────────────────────────
    report_path = results_dir / 'ion_proximity_report.txt'
    with open(report_path, 'w') as f:
        f.write("ION PROXIMITY ANALYSIS\n")
        f.write(f"Threshold: {threshold} Å\n")
        f.write(f"Frames: {start_frame} → {end_frame}  ({n_frames} total)\n")
        f.write(f"Ion selection: {ION_SELECTION}\n")
        f.write("=" * 90 + "\n\n")

        # summary statistics
        f.write("SUMMARY (fraction of frames with ion within threshold)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Residue':<12} {'ResID':>6} {'Mean dist':>12} "
                f"{'Min dist':>10} {'Max dist':>10} "
                f"{'% with ion':>12}\n")
        f.write("-" * 60 + "\n")
        for lbl in labels:
            resid   = residue_config[lbl]
            md      = min_dist[lbl]
            pct_ion = np.mean(has_ion[lbl]) * 100
            f.write(f"{lbl:<12} {resid:>6} {np.mean(md):>12.3f} "
                    f"{np.min(md):>10.3f} {np.max(md):>10.3f} "
                    f"{pct_ion:>12.1f}%\n")

        # full frame table
        f.write("\n\nFULL FRAME TABLE\n")
        f.write("-" * 90 + "\n")

        # header
        dist_header = ''.join(f"{lbl+' dist':>14}" for lbl in labels)
        ion_header  = ''.join(f"{lbl+' ion':>10}" for lbl in labels)
        f.write(f"{'Frame':>8}{dist_header}{ion_header}\n")
        f.write("-" * 90 + "\n")

        for fi in range(n_frames):
            dist_row = ''.join(f"{min_dist[lbl][fi]:>14.3f}" for lbl in labels)
            ion_row  = ''.join(
                f"{'YES':>10}" if has_ion[lbl][fi] else f"{'no':>10}"
                for lbl in labels
            )
            f.write(f"{start_frame + fi:>8}{dist_row}{ion_row}\n")

    print(f"  ✅ Saved: {report_path}")

    # ── Combine with dihedrals if available ───────────────────────────────────
    dihedral_pkl = results_dir.parent / 'dihedral_analysis' / 'dihedral_raw_data.pkl'
    if dihedral_pkl.exists():
        print(f"  Found dihedral_raw_data.pkl — creating combined file...")
        _create_combined_file(results, min_dist, has_ion, labels,
                              dihedral_pkl, results_dir,
                              start_frame, n_frames, threshold)
    else:
        print(f"  [INFO] dihedral_raw_data.pkl not found at {dihedral_pkl}")
        print(f"         Combined file not created.")

    # print summary
    print(f"\n  Ion proximity summary (threshold={threshold} Å):")
    print(f"  {'Residue':<12} {'Mean dist':>12} {'% with ion':>12}")
    print(f"  {'-'*38}")
    for lbl in labels:
        pct = np.mean(has_ion[lbl]) * 100
        print(f"  {lbl:<12} {np.mean(min_dist[lbl]):>12.3f} {pct:>12.1f}%")

    return results


def _create_combined_file(results, min_dist, has_ion, labels,
                           dihedral_pkl, output_dir,
                           start_frame, n_frames, threshold):
    """
    Merge ion proximity data with dihedral angles from dihedral_raw_data.pkl.
    Saves combined pkl and human-readable txt.
    """
    with open(dihedral_pkl, 'rb') as f:
        dihedral_data = pickle.load(f)

    # ── Build combined dict ───────────────────────────────────────────────────
    combined = {
        'threshold':   threshold,
        'start_frame': start_frame,
        'n_frames':    n_frames,
        'labels':      labels,
        'min_dist':    min_dist,
        'has_ion':     has_ion,
        'dihedrals':   {}
    }

    residue_list = []
    for res_name, angles in dihedral_data.items():
        if not isinstance(angles, dict):
            continue
        pdb_label = angles.get('pdb_label', res_name)
        chi1 = np.array(angles['chi1'])[start_frame: start_frame + n_frames] \
               if angles.get('chi1') is not None else None
        chi2 = np.array(angles['chi2'])[start_frame: start_frame + n_frames] \
               if angles.get('chi2') is not None else None
        combined['dihedrals'][res_name] = {
            'pdb_label': pdb_label,
            'chi1': chi1,
            'chi2': chi2,
        }
        residue_list.append((res_name, pdb_label))

    pkl_out = output_dir / 'combined_ion_dihedrals.pkl'
    with open(pkl_out, 'wb') as f:
        pickle.dump(combined, f)
    print(f"  ✅ Saved: {pkl_out}")

    # ── Human-readable txt ────────────────────────────────────────────────────
    txt_out = output_dir / 'combined_ion_dihedrals.txt'
    with open(txt_out, 'w') as f:
        f.write("COMBINED ION PROXIMITY + DIHEDRAL ANGLES\n")
        f.write(f"Threshold: {threshold} Å  |  "
                f"Frames: {start_frame} → {start_frame + n_frames - 1} "
                f"({n_frames} total)\n")
        f.write("=" * 120 + "\n\n")

        # header — distances, yes/no, then all chi angles
        dist_hdr = ''.join(f"{lbl+'_dist':>14}" for lbl in labels)
        ion_hdr  = ''.join(f"{lbl+'_ion':>10}" for lbl in labels)
        dih_hdr  = ''.join(
            f"  {pdb:>7}_χ1  {pdb:>7}_χ2"
            for _, pdb in residue_list
        )
        f.write(f"{'Frame':>8}{dist_hdr}{ion_hdr}{dih_hdr}\n")
        f.write("-" * 120 + "\n")

        for fi in range(n_frames):
            dist_row = ''.join(f"{min_dist[lbl][fi]:>14.3f}" for lbl in labels)
            ion_row  = ''.join(
                f"{'YES':>10}" if has_ion[lbl][fi] else f"{'no':>10}"
                for lbl in labels
            )
            dih_row = ''
            for res_name, _ in residue_list:
                chi1 = combined['dihedrals'][res_name]['chi1']
                chi2 = combined['dihedrals'][res_name]['chi2']
                c1 = f"{chi1[fi]:>10.2f}" if chi1 is not None else f"{'N/A':>10}"
                c2 = f"{chi2[fi]:>10.2f}" if chi2 is not None else f"{'N/A':>10}"
                dih_row += f"{c1}{c2}"

            f.write(f"{start_frame + fi:>8}{dist_row}{ion_row}{dih_row}\n")

    print(f"  ✅ Saved: {txt_out}")