#!/usr/bin/env python3
"""
Ligand Distance Analysis — G12_ML only

Computes per-frame minimum distances between:
    LIG (resid 1305) atoms H28, H29
and two protein residues:
    GLU99  OE1/OE2  → saved in ligand_distance_analysis/GLU_G1/
    ASP131 OD1/OD2  → saved in ligand_distance_analysis/ASP_G1/

Each subfolder contains:
    ligand_distance_raw_data.pkl       — per-frame distance arrays
    ligand_distance_report.txt         — human-readable full frame table
    combined_distance_dihedrals.pkl    — distances + all dihedral angles
    combined_distance_dihedrals.txt    — human-readable combined table

Usage (from main.py when channel_type == 'G12_ML'):
    from ligand_distance_analysis import run_ligand_distance_analysis
    run_ligand_distance_analysis(u, results_dir, start_frame, end_frame)
"""

import pickle
import numpy as np
from pathlib import Path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_ligand_distance_analysis(u, results_dir, start_frame=0, end_frame=None):
    """
    Run distance analysis for both GLU99 and ASP131 and save to subfolders.
    """
    results_dir = Path(results_dir) / 'ligand_distance_analysis'
    results_dir.mkdir(parents=True, exist_ok=True)

    if end_frame is None:
        end_frame = len(u.trajectory) - 1

    print("\n" + "=" * 70)
    print("LIGAND DISTANCE ANALYSIS  —  G12_ML")
    print("=" * 70)

    # ── Find ligand atoms ─────────────────────────────────────────────────────
    lig_all = u.select_atoms('resname LIG1305')
    if len(lig_all) == 0:
        for rname in ['LIG', 'MOL', 'DRG', 'UNK', 'ML']:
            test = u.select_atoms(f'resname {rname}')
            if len(test) > 0:
                print(f"  [INFO] resname 'LIG1305' not found — using '{rname}' "
                      f"(resid 1305, {len(test)} atoms)")
                lig_all = test
                break
        if len(lig_all) == 0:
            resnames = sorted(set(u.atoms.resnames))
            print(f"  [ERROR] No ligand found. Resnames: {resnames}")
            return None

    print(f"\n  Ligand atoms ({len(lig_all)}):")
    for atom in lig_all:
        print(f"    {atom.name:<6} type={atom.type:<6} "
              f"resname={atom.resname} resid={atom.resid} index={atom.index}")

    # ── Define targets ────────────────────────────────────────────────────────
    targets = [
        {
            'label':   'GLU_G1',
            'resid':   99,
            'resname': 'GLU',
            'O_atoms': ['OE1', 'OE2'],
            'H_atoms': ['H28', 'H29'],
        },
        {
            'label':   'ASP_G1',
            'resid':   131,
            'resname': 'ASP',
            'O_atoms': ['OD1', 'OD2'],
            'H_atoms': ['H28', 'H29'],
        },
    ]

    dihedral_pkl = results_dir.parent / 'dihedral_analysis' / 'dihedral_raw_data.pkl'

    for target in targets:
        subfolder = results_dir / target['label']
        subfolder.mkdir(exist_ok=True)
        print(f"\n{'─'*70}")
        print(f"  → {target['label']}: "
              f"LIG H28/H29  to  "
              f"{target['resname']}{target['resid']} "
              f"{'/'.join(target['O_atoms'])}")
        print(f"{'─'*70}")
        _run_single_target(u, subfolder, target, start_frame, end_frame,
                           dihedral_pkl)

    print(f"\n{'='*70}")
    print(f"  Done. Results saved in subfolders:")
    print(f"    {results_dir}/GLU_G1/")
    print(f"    {results_dir}/ASP_G1/")
    print(f"{'='*70}")


# =============================================================================
# SINGLE TARGET
# =============================================================================

def _run_single_target(u, output_dir, target, start_frame, end_frame,
                       dihedral_pkl):
    resid   = target['resid']
    resname = target['resname']
    o_names = target['O_atoms']
    h_names = target['H_atoms']

    # ── Select atoms ──────────────────────────────────────────────────────────
    H_sels = {}
    for h in h_names:
        sel = u.select_atoms(f'resname LIG and resid 1305 and name {h}')
        if len(sel) == 0:
            print(f"  [ERROR] LIG:{h} not found"); return
        H_sels[h] = sel
        print(f"  Found LIG:{h}  index={sel[0].index}")

    O_sels = {}
    for o in o_names:
        sel = u.select_atoms(f'resid {resid} and name {o}')
        if len(sel) == 0:
            print(f"  [ERROR] {resname}{resid}:{o} not found"); return
        O_sels[o] = sel
        print(f"  Found {resname}{resid}:{o}  index={sel[0].index}")

    # ── Per-frame calculation ──────────────────────────────────────────────────
    n_frames  = end_frame - start_frame + 1
    pair_keys = [(h, o) for h in h_names for o in o_names]
    pair_arrs = {k: np.zeros(n_frames) for k in pair_keys}
    dist_min  = np.zeros(n_frames)
    pair_labels = [f'{h}-{o}' for h, o in pair_keys]

    print(f"  Computing {len(pair_keys)} pairs × {n_frames} frames...")
    for fi, ts in enumerate(u.trajectory[start_frame: end_frame + 1]):
        if fi % 1000 == 0:
            print(f"    Frame {start_frame + fi}/{end_frame}", end='\r')
        vals = []
        for (h, o) in pair_keys:
            d = float(np.linalg.norm(
                H_sels[h].positions[0] - O_sels[o].positions[0]))
            pair_arrs[(h, o)][fi] = d
            vals.append(d)
        dist_min[fi] = min(vals)
    print(f"\n  Done.")

    # ── Save pkl ──────────────────────────────────────────────────────────────
    results = {
        'label':        target['label'],
        'protein_res':  f"{resname}{resid}",
        'H_atoms':      h_names,
        'O_atoms':      o_names,
        'start_frame':  start_frame,
        'end_frame':    end_frame,
        'n_frames':     n_frames,
        'min_distance': dist_min,
    }
    for (h, o), arr in pair_arrs.items():
        results[f'{h}_{o}'] = arr

    pkl_path = output_dir / 'ligand_distance_raw_data.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  ✅ Saved: {pkl_path}")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = output_dir / 'ligand_distance_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"LIGAND DISTANCE ANALYSIS — G12_ML\n")
        f.write(f"LIG H28/H29  →  {resname}{resid} {'/'.join(o_names)}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Frames: {start_frame} → {end_frame}  ({n_frames} total)\n\n")

        f.write("STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Pair':<15} {'Mean':>8} {'Std':>8} {'Min':>8} "
                f"{'Max':>8} {'<3.5Å%':>8} {'<5.0Å%':>8}\n")
        f.write("-" * 70 + "\n")
        for lbl, (h, o) in zip(pair_labels, pair_keys):
            arr = pair_arrs[(h, o)]
            f.write(f"{lbl:<15} {np.mean(arr):>8.2f} {np.std(arr):>8.2f} "
                    f"{np.min(arr):>8.2f} {np.max(arr):>8.2f} "
                    f"{np.mean(arr<3.5)*100:>8.1f} {np.mean(arr<5.0)*100:>8.1f}\n")
        f.write(f"{'Min(all)':15} {np.mean(dist_min):>8.2f} {np.std(dist_min):>8.2f} "
                f"{np.min(dist_min):>8.2f} {np.max(dist_min):>8.2f} "
                f"{np.mean(dist_min<3.5)*100:>8.1f} "
                f"{np.mean(dist_min<5.0)*100:>8.1f}\n")

        f.write("\n\nFULL FRAME TABLE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Frame':>8} {'Min':>8}  " +
                ''.join(f"{l:>10}" for l in pair_labels) + "  Closest\n")
        f.write("-" * 70 + "\n")
        for i in range(n_frames):
            vals    = [pair_arrs[(h, o)][i] for h, o in pair_keys]
            closest = pair_labels[int(np.argmin(vals))]
            f.write(f"{start_frame+i:>8} {dist_min[i]:>8.3f}  " +
                    ''.join(f"{v:>10.3f}" for v in vals) +
                    f"  {closest}\n")

    print(f"  ✅ Saved: {report_path}")

    # ── Combined with dihedrals ───────────────────────────────────────────────
    if dihedral_pkl.exists():
        print(f"  Found dihedral_raw_data.pkl — creating combined file...")
        _create_combined_file(results, pair_arrs, pair_keys, pair_labels,
                              dist_min, dihedral_pkl, output_dir,
                              start_frame, n_frames)
    else:
        print(f"  [INFO] dihedral_raw_data.pkl not found at {dihedral_pkl}")


# =============================================================================
# COMBINED DISTANCE + DIHEDRAL FILE
# =============================================================================

def _create_combined_file(dist_results, pair_arrs, pair_keys, pair_labels,
                           dist_min, dihedral_pkl, output_dir,
                           start_frame, n_frames):
    with open(dihedral_pkl, 'rb') as f:
        dihedral_data = pickle.load(f)

    combined = {
        'start_frame':  start_frame,
        'n_frames':     n_frames,
        'min_distance': dist_min,
        'dihedrals':    {}
    }
    for (h, o), arr in pair_arrs.items():
        combined[f'{h}_{o}'] = arr

    residue_list = []
    for res_name, angles in dihedral_data.items():
        if not isinstance(angles, dict):
            continue
        pdb_label = angles.get('pdb_label', res_name)
        chi1 = np.array(angles['chi1'])[start_frame: start_frame + n_frames] \
               if angles.get('chi1') is not None else None
        chi2 = np.array(angles['chi2'])[start_frame: start_frame + n_frames] \
               if angles.get('chi2') is not None else None
        combined['dihedrals'][res_name] = {'pdb_label': pdb_label,
                                            'chi1': chi1, 'chi2': chi2}
        residue_list.append((res_name, pdb_label))

    pkl_out = output_dir / 'combined_distance_dihedrals.pkl'
    with open(pkl_out, 'wb') as f:
        pickle.dump(combined, f)
    print(f"  ✅ Saved: {pkl_out}")

    txt_out = output_dir / 'combined_distance_dihedrals.txt'
    with open(txt_out, 'w') as f:
        f.write(f"COMBINED DISTANCES + DIHEDRALS — G12_ML\n")
        f.write(f"LIG H28/H29 → {dist_results['protein_res']}\n")
        f.write("=" * 120 + "\n")
        f.write(f"Frames: {start_frame} → {start_frame+n_frames-1} "
                f"({n_frames} total)\n\n")

        dist_hdr = ''.join(f"{l:>10}" for l in pair_labels)
        res_hdr  = ''.join(f"  {pdb:>7}_χ1  {pdb:>7}_χ2"
                           for _, pdb in residue_list)
        f.write(f"{'Frame':>8} {'MinDist':>9}{dist_hdr}  "
                f"{'Closest':>10}{res_hdr}\n")
        f.write("-" * 120 + "\n")

        for i in range(n_frames):
            vals    = [pair_arrs[(h, o)][i] for h, o in pair_keys]
            closest = pair_labels[int(np.argmin(vals))]
            drow    = ''.join(f"{v:>10.3f}" for v in vals)
            arow    = ''
            for res_name, _ in residue_list:
                chi1 = combined['dihedrals'][res_name]['chi1']
                chi2 = combined['dihedrals'][res_name]['chi2']
                c1 = f"{chi1[i]:>10.2f}" if chi1 is not None else f"{'N/A':>10}"
                c2 = f"{chi2[i]:>10.2f}" if chi2 is not None else f"{'N/A':>10}"
                arow += f"{c1}{c2}"
            f.write(f"{start_frame+i:>8} {dist_min[i]:>9.3f}"
                    f"{drow}  {closest:>10}{arow}\n")

    print(f"  ✅ Saved: {txt_out}")