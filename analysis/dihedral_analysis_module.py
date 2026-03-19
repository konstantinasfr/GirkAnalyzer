#!/usr/bin/env python3
"""
Dihedral Analysis Module - Integrates with existing ion permeation analysis
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Dihedral
except ImportError:
    print("WARNING: MDAnalysis not found. Dihedral analysis will be skipped.")


def convert_to_pdb_numbering(residue_id, channel_type):
    """Converts a residue ID to PDB-style numbering."""
    if channel_type == "G2":
        glu_mapping = {98: "152.A", 426: "152.C", 754: "152.B", 1082: "152.D"}
        asn_mapping = {130: "184.A", 458: "184.C", 786: "184.B", 1114: "184.D"}
        if residue_id in glu_mapping:
            return glu_mapping[residue_id]
        if residue_id in asn_mapping:
            return asn_mapping[residue_id]
    elif channel_type == "G12" or channel_type == "G12_GAT":
        glu_mapping = {422: "152.A", 98: "152.B", 747: "152.C", 1073: "141.D"}
        asn_asp_mapping = {454: "184.A", 130: "184.B", 779: "184.C", 1105: "173.D"}
        if residue_id in glu_mapping:
            return glu_mapping[residue_id]
        if residue_id in asn_asp_mapping:
            return asn_asp_mapping[residue_id]
    elif channel_type == "G12_ML":
        glu_mapping = {749: "152.A", 1074: "152.B", 424: "152.C", 99: "141.D"}
        asn_asp_mapping = {781: "184.A", 1106: "184.B", 456: "184.C", 131: "173.D"}
        if residue_id in glu_mapping:
            return glu_mapping[residue_id]
        if residue_id in asn_asp_mapping:
            return asn_asp_mapping[residue_id]
    return str(residue_id)


def get_residue_lists(channel_type):
    """Returns residue lists based on channel type."""
    if channel_type == "G2":
        glu_residues = [98, 426, 754, 1082]
        asn_residues = [130, 458, 786, 1114]
        asp_residues = []
    elif channel_type == "G12" or channel_type == "G12_GAT":
        glu_residues = [98, 422, 747, 1073]
        asn_residues = [130, 454, 779]
        asp_residues = [1105]
    elif channel_type == "G12_ML":
        glu_residues = [99, 424, 749, 1074]
        asn_residues = [456, 781, 1106]
        asp_residues = [131]
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")
    
    return glu_residues, asn_residues, asp_residues


def analyze_residue_orientations(u, channel_type="G12"):
    """
    Analyze dihedral angles for GLU, ASN, and ASP residues.
    Includes extensive debug prints to verify correct residue/label mapping.
    """

    glu_residues, asn_residues, asp_residues = get_residue_lists(channel_type)

    print(f"\n{'='*70}")
    print(f"DIHEDRAL ANALYSIS - Channel type: {channel_type}")
    print(f"{'='*70}")
    print(f"Total frames: {len(u.trajectory)}")
    print(f"GLU residues: {glu_residues}")
    print(f"ASN residues: {asn_residues}")
    print(f"ASP residues: {asp_residues}")

    # ── DEBUG: verify converter gives correct labels ──────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  PDB LABEL MAPPING VERIFICATION")
    print(f"  {'─'*60}")
    print(f"  {'ResID':>6}  {'List':>5}  {'PDB Label':>12}  Check")
    print(f"  {'-'*50}")
    all_ok = True
    for resid in glu_residues:
        lbl = convert_to_pdb_numbering(resid, channel_type)
        prefix = lbl.split('.')[0] if '.' in lbl else '??'
        ok = prefix in ('152', '141')
        flag = '✅' if ok else '❌ WRONG — expected 152.x or 141.x'
        print(f"  {resid:>6}  {'GLU':>5}  {lbl:>12}  {flag}")
        if not ok: all_ok = False
    for resid in asn_residues:
        lbl = convert_to_pdb_numbering(resid, channel_type)
        prefix = lbl.split('.')[0] if '.' in lbl else '??'
        ok = prefix == '184'
        flag = '✅' if ok else '❌ WRONG — expected 184.x'
        print(f"  {resid:>6}  {'ASN':>5}  {lbl:>12}  {flag}")
        if not ok: all_ok = False
    for resid in asp_residues:
        lbl = convert_to_pdb_numbering(resid, channel_type)
        prefix = lbl.split('.')[0] if '.' in lbl else '??'
        ok = prefix == '173'
        flag = '✅' if ok else '❌ WRONG — expected 173.x'
        print(f"  {resid:>6}  {'ASP':>5}  {lbl:>12}  {flag}")
        if not ok: all_ok = False
    if all_ok:
        print(f"\n  ✅ ALL LABELS CORRECT")
    else:
        print(f"\n  ❌ LABEL ERRORS DETECTED — check convert_to_pdb_numbering and get_residue_lists!")
    print(f"  {'─'*60}\n")
    # ─────────────────────────────────────────────────────────────────────────

    results = {}

    def get_atom(resid, name):
        ag = u.select_atoms(f"resid {resid} and name {name}")
        return ag[0] if len(ag) else None

    # -------------------------------------------------------------------------
    # 1) GLU DIHEDRALS
    # -------------------------------------------------------------------------
    print("\n=== ANALYZING GLU RESIDUES ===")

    for resid in glu_residues:
        pdb_label = convert_to_pdb_numbering(resid, channel_type)
        prefix    = pdb_label.split('.')[0] if '.' in pdb_label else '??'
        print(f"GLU resid={resid}  pdb_label={pdb_label}  "
              f"{'✅' if prefix in ('152','141') else '❌ WRONG LABEL'}")

        # verify topology resname
        probe   = u.select_atoms(f"resid {resid}")
        resname = probe[0].resname if len(probe) > 0 else 'NOT FOUND'
        print(f"  topology resname = {resname}  "
              f"{'✅' if resname == 'GLU' else '⚠️  not GLU in topology'}")

        res_ag = u.select_atoms(f"resid {resid}").residues
        if len(res_ag) == 0:
            print(f"  [WARN] GLU {resid} not found")
            continue

        res = res_ag[0]

        chi1_sel = res.chi1_selection()
        if chi1_sel is not None and len(chi1_sel) == 4:
            dih1 = Dihedral([chi1_sel]).run()
            chi1 = dih1.results.angles[:, 0]
            print(f"  chi1: {len(chi1)} frames, mean={chi1.mean():.1f}°  ✅")
        else:
            print(f"  [WARN] χ1 missing")
            chi1 = None

        a_CA  = get_atom(resid, "CA")
        a_CB  = get_atom(resid, "CB")
        a_CG  = get_atom(resid, "CG")
        a_CD  = get_atom(resid, "CD")

        if all([a_CA, a_CB, a_CG, a_CD]):
            ag2  = u.atoms[[a_CA.ix, a_CB.ix, a_CG.ix, a_CD.ix]]
            dih2 = Dihedral([ag2]).run()
            chi2 = dih2.results.angles[:, 0]
            print(f"  chi2: {len(chi2)} frames, mean={chi2.mean():.1f}°  ✅")
        else:
            print(f"  [WARN] χ2 missing atoms  CA={a_CA} CB={a_CB} CG={a_CG} CD={a_CD}")
            chi2 = None

        a_OE1 = get_atom(resid, "OE1")
        if all([a_CB, a_CG, a_CD, a_OE1]):
            ag3  = u.atoms[[a_CB.ix, a_CG.ix, a_CD.ix, a_OE1.ix]]
            dih3 = Dihedral([ag3]).run()
            chi3 = dih3.results.angles[:, 0]
            print(f"  chi3: {len(chi3)} frames, mean={chi3.mean():.1f}°  ✅")
        else:
            print(f"  [WARN] χ3 missing atoms")
            chi3 = None

        results[f"GLU_{resid}"] = {
            "chi1": chi1, "chi2": chi2, "chi3": chi3,
            "pdb_label": pdb_label, "residue_type": "GLU"
        }

    # -------------------------------------------------------------------------
    # 2) ASN DIHEDRALS
    # -------------------------------------------------------------------------
    print("\n=== ANALYZING ASN RESIDUES ===")

    for resid in asn_residues:
        pdb_label = convert_to_pdb_numbering(resid, channel_type)
        prefix    = pdb_label.split('.')[0] if '.' in pdb_label else '??'
        print(f"ASN resid={resid}  pdb_label={pdb_label}  "
              f"{'✅' if prefix == '184' else '❌ WRONG LABEL — expected 184.x'}")

        probe   = u.select_atoms(f"resid {resid}")
        resname = probe[0].resname if len(probe) > 0 else 'NOT FOUND'
        print(f"  topology resname = {resname}  "
              f"{'✅' if resname == 'ASN' else '⚠️  not ASN in topology'}")

        if len(u.select_atoms(f"resid {resid}")) == 0:
            print(f"  [WARN] ASN {resid} not found")
            continue

        a_N   = get_atom(resid, "N")
        a_CA  = get_atom(resid, "CA")
        a_CB  = get_atom(resid, "CB")
        a_CG  = get_atom(resid, "CG")
        a_OD1 = get_atom(resid, "OD1")

        if all([a_N, a_CA, a_CB, a_CG]):
            ag1  = u.atoms[[a_N.ix, a_CA.ix, a_CB.ix, a_CG.ix]]
            dih1 = Dihedral([ag1]).run()
            chi1 = dih1.results.angles[:, 0]
            print(f"  chi1: {len(chi1)} frames, mean={chi1.mean():.1f}°  ✅")
        else:
            print(f"  [WARN] χ1 missing atoms  N={a_N} CA={a_CA} CB={a_CB} CG={a_CG}")
            chi1 = None

        if all([a_CA, a_CB, a_CG, a_OD1]):
            ag2  = u.atoms[[a_CA.ix, a_CB.ix, a_CG.ix, a_OD1.ix]]
            dih2 = Dihedral([ag2]).run()
            chi2 = dih2.results.angles[:, 0]
            print(f"  chi2: {len(chi2)} frames, mean={chi2.mean():.1f}°  ✅")
        else:
            print(f"  [WARN] χ2 missing atoms  CA={a_CA} CB={a_CB} CG={a_CG} OD1={a_OD1}")
            chi2 = None

        results[f"ASN_{resid}"] = {
            "chi1": chi1, "chi2": chi2,
            "pdb_label": pdb_label, "residue_type": "ASN"
        }

    # -------------------------------------------------------------------------
    # 3) ASP DIHEDRALS
    # -------------------------------------------------------------------------
    print("\n=== ANALYZING ASP RESIDUES ===")

    for resid in asp_residues:
        pdb_label = convert_to_pdb_numbering(resid, channel_type)
        prefix    = pdb_label.split('.')[0] if '.' in pdb_label else '??'
        print(f"ASP resid={resid}  pdb_label={pdb_label}  "
              f"{'✅' if prefix == '173' else '❌ WRONG LABEL — expected 173.x'}")

        probe   = u.select_atoms(f"resid {resid}")
        resname = probe[0].resname if len(probe) > 0 else 'NOT FOUND'
        print(f"  topology resname = {resname}  "
              f"{'✅' if resname == 'ASP' else '⚠️  not ASP in topology'}")

        if len(u.select_atoms(f"resid {resid}")) == 0:
            print(f"  [WARN] ASP {resid} not found")
            continue

        a_N   = get_atom(resid, "N")
        a_CA  = get_atom(resid, "CA")
        a_CB  = get_atom(resid, "CB")
        a_CG  = get_atom(resid, "CG")
        a_OD1 = get_atom(resid, "OD1")

        if all([a_N, a_CA, a_CB, a_CG]):
            ag1  = u.atoms[[a_N.ix, a_CA.ix, a_CB.ix, a_CG.ix]]
            dih1 = Dihedral([ag1]).run()
            chi1 = dih1.results.angles[:, 0]
            print(f"  chi1: {len(chi1)} frames, mean={chi1.mean():.1f}°  ✅")
        else:
            print(f"  [WARN] χ1 missing atoms  N={a_N} CA={a_CA} CB={a_CB} CG={a_CG}")
            chi1 = None

        if all([a_CA, a_CB, a_CG, a_OD1]):
            ag2  = u.atoms[[a_CA.ix, a_CB.ix, a_CG.ix, a_OD1.ix]]
            dih2 = Dihedral([ag2]).run()
            chi2 = dih2.results.angles[:, 0]
            print(f"  chi2: {len(chi2)} frames, mean={chi2.mean():.1f}°  ✅")
        else:
            print(f"  [WARN] χ2 missing atoms  CA={a_CA} CB={a_CB} CG={a_CG} OD1={a_OD1}")
            chi2 = None

        results[f"ASP_{resid}"] = {
            "chi1": chi1, "chi2": chi2,
            "pdb_label": pdb_label, "residue_type": "ASP"
        }

    # ── DEBUG: final summary ──────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  RESULTS SUMMARY")
    print(f"  {'─'*60}")
    print(f"  {'Key':>15}  {'PDB Label':>12}  {'Type':>5}  chi1  chi2")
    for key, val in results.items():
        c1 = '✅' if val['chi1'] is not None else '❌'
        c2 = '✅' if val['chi2'] is not None else '❌'
        print(f"  {key:>15}  {val['pdb_label']:>12}  "
              f"{val['residue_type']:>5}  {c1}    {c2}")
    print(f"  {'─'*60}\n")
    # ─────────────────────────────────────────────────────────────────────────

    return results
    """
    Analyze dihedral angles for GLU, ASN, and ASP residues.
    
    Parameters
    ----------
    u : MDAnalysis.Universe
        Universe object with topology and trajectory already loaded
    channel_type : str
        Channel type for PDB numbering conversion
    
    Returns
    -------
    results : dict
        Dictionary containing dihedral angles for each residue
    """
    
    glu_residues, asn_residues, asp_residues = get_residue_lists(channel_type)
    
    print(f"\n{'='*70}")
    print(f"DIHEDRAL ANALYSIS - Channel type: {channel_type}")
    print(f"{'='*70}")
    print(f"Total frames: {len(u.trajectory)}")
    print(f"GLU residues: {glu_residues}")
    print(f"ASN residues: {asn_residues}")
    print(f"ASP residues: {asp_residues}")
    
    results = {}
    
    def get_atom(resid, name):
        """Helper function to get atom from residue."""
        ag = u.select_atoms(f"resid {resid} and name {name}")
        return ag[0] if len(ag) else None
    
    # -------------------------------------------------------------------------
    # 1) GLU DIHEDRALS
    # -------------------------------------------------------------------------
    print("\n=== ANALYZING GLU RESIDUES ===")
    
    for resid in glu_residues:
        pdb_label = convert_to_pdb_numbering(resid, channel_type)
        print(f"GLU {resid} (PDB: {pdb_label})")
        
        res_ag = u.select_atoms(f"resid {resid}").residues
        if len(res_ag) == 0:
            print(f"  [WARN] GLU {resid} not found")
            continue
        
        res = res_ag[0]
        
        # χ1 (N-CA-CB-CG)
        chi1_sel = res.chi1_selection()
        if chi1_sel is not None and len(chi1_sel) == 4:
            dih1 = Dihedral([chi1_sel]).run()
            chi1 = dih1.results.angles[:, 0]
        else:
            print(f"  [WARN] χ1 missing")
            chi1 = None
        
        # χ2 (CA-CB-CG-CD)
        a_CA = get_atom(resid, "CA")
        a_CB = get_atom(resid, "CB")
        a_CG = get_atom(resid, "CG")
        a_CD = get_atom(resid, "CD")
        
        if all([a_CA, a_CB, a_CG, a_CD]):
            ag2 = u.atoms[[a_CA.ix, a_CB.ix, a_CG.ix, a_CD.ix]]
            dih2 = Dihedral([ag2]).run()
            chi2 = dih2.results.angles[:, 0]
        else:
            print(f"  [WARN] χ2 missing atoms")
            chi2 = None
        
        # χ3 (CB-CG-CD-OE1)
        a_OE1 = get_atom(resid, "OE1")
        if all([a_CB, a_CG, a_CD, a_OE1]):
            ag3 = u.atoms[[a_CB.ix, a_CG.ix, a_CD.ix, a_OE1.ix]]
            dih3 = Dihedral([ag3]).run()
            chi3 = dih3.results.angles[:, 0]
        else:
            print(f"  [WARN] χ3 missing atoms")
            chi3 = None
        
        results[f"GLU_{resid}"] = {
            "chi1": chi1,
            "chi2": chi2,
            "chi3": chi3,
            "pdb_label": pdb_label,
            "residue_type": "GLU"
        }
    
    # -------------------------------------------------------------------------
    # 2) ASN DIHEDRALS
    # -------------------------------------------------------------------------
    print("\n=== ANALYZING ASN RESIDUES ===")
    
    for resid in asn_residues:
        pdb_label = convert_to_pdb_numbering(resid, channel_type)
        print(f"ASN {resid} (PDB: {pdb_label})")
        
        if len(u.select_atoms(f"resid {resid}")) == 0:
            print(f"  [WARN] ASN {resid} not found")
            continue
        
        a_N = get_atom(resid, "N")
        a_CA = get_atom(resid, "CA")
        a_CB = get_atom(resid, "CB")
        a_CG = get_atom(resid, "CG")
        a_OD1 = get_atom(resid, "OD1")
        
        # χ1
        if all([a_N, a_CA, a_CB, a_CG]):
            ag1 = u.atoms[[a_N.ix, a_CA.ix, a_CB.ix, a_CG.ix]]
            dih1 = Dihedral([ag1]).run()
            chi1 = dih1.results.angles[:, 0]
        else:
            print(f"  [WARN] χ1 missing atoms")
            chi1 = None
        
        # χ2
        if all([a_CA, a_CB, a_CG, a_OD1]):
            ag2 = u.atoms[[a_CA.ix, a_CB.ix, a_CG.ix, a_OD1.ix]]
            dih2 = Dihedral([ag2]).run()
            chi2 = dih2.results.angles[:, 0]
        else:
            print(f"  [WARN] χ2 missing atoms")
            chi2 = None
        
        results[f"ASN_{resid}"] = {
            "chi1": chi1,
            "chi2": chi2,
            "pdb_label": pdb_label,
            "residue_type": "ASN"
        }
    
    # -------------------------------------------------------------------------
    # 3) ASP DIHEDRALS
    # -------------------------------------------------------------------------
    print("\n=== ANALYZING ASP RESIDUES ===")
    
    for resid in asp_residues:
        pdb_label = convert_to_pdb_numbering(resid, channel_type)
        print(f"ASP {resid} (PDB: {pdb_label})")
        
        if len(u.select_atoms(f"resid {resid}")) == 0:
            print(f"  [WARN] ASP {resid} not found")
            continue
        
        a_N = get_atom(resid, "N")
        a_CA = get_atom(resid, "CA")
        a_CB = get_atom(resid, "CB")
        a_CG = get_atom(resid, "CG")
        a_OD1 = get_atom(resid, "OD1")
        
        # χ1
        if all([a_N, a_CA, a_CB, a_CG]):
            ag1 = u.atoms[[a_N.ix, a_CA.ix, a_CB.ix, a_CG.ix]]
            dih1 = Dihedral([ag1]).run()
            chi1 = dih1.results.angles[:, 0]
        else:
            print(f"  [WARN] χ1 missing atoms")
            chi1 = None
        
        # χ2
        if all([a_CA, a_CB, a_CG, a_OD1]):
            ag2 = u.atoms[[a_CA.ix, a_CB.ix, a_CG.ix, a_OD1.ix]]
            dih2 = Dihedral([ag2]).run()
            chi2 = dih2.results.angles[:, 0]
        else:
            print(f"  [WARN] χ2 missing atoms")
            chi2 = None
        
        results[f"ASP_{resid}"] = {
            "chi1": chi1,
            "chi2": chi2,
            "pdb_label": pdb_label,
            "residue_type": "ASP"
        }
    
    return results


def detect_and_merge_peaks(hist, min_prom=0.05, min_height=0.02, distance=8, 
                           merge_bins=8, smooth_sigma=1, wrap_bins=8):
    """Detect peaks in circular histograms with smoothing and wraparound."""
    num_bins = len(hist)
    smoothed = gaussian_filter1d(hist, sigma=smooth_sigma)
    padded = np.concatenate([smoothed[-wrap_bins:], smoothed, smoothed[:wrap_bins]])
    
    raw_peaks, _ = find_peaks(
        padded,
        prominence=np.max(smoothed) * min_prom,
        height=np.max(smoothed) * min_height,
        distance=distance
    )
    
    shifted_peaks = raw_peaks - wrap_bins
    shifted_peaks = shifted_peaks[(shifted_peaks >= 0) & (shifted_peaks < num_bins)]
    
    if len(shifted_peaks) > 1:
        shifted_peaks = np.sort(shifted_peaks)
        merged = []
        i = 0
        while i < len(shifted_peaks):
            group = [shifted_peaks[i]]
            j = i + 1
            while j < len(shifted_peaks) and (shifted_peaks[j] - shifted_peaks[i]) <= merge_bins:
                group.append(shifted_peaks[j])
                j += 1
            merged.append(int(np.median(group)))
            i = j
        
        first = merged[0]
        last = merged[-1]
        if (first + num_bins - last) <= merge_bins:
            merged = merged[1:]
            merged[-1] = int(np.median([first, last]))
        
        shifted_peaks = np.array(merged)
    
    return np.array(shifted_peaks)


def find_peak_based_frames(results, output_dir, peak_tolerance=15):
    """Detect peaks and find matching frames."""
    print("\n=== PEAK-BASED FRAME ANALYSIS ===")
    peak_frames_dict = {}
    
    with open(f'{output_dir}/dihedral_peak_frames.txt', 'w') as f:
        f.write("PEAK-BASED FRAME ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        for res_name, angles in results.items():
            pdb_label = angles.get('pdb_label', res_name)
            print(f"\n{res_name} (PDB: {pdb_label}):")
            f.write(f"{res_name} (PDB: {pdb_label}):\n")
            
            chi1 = np.array(angles['chi1'])
            chi2 = np.array(angles['chi2'])
            
            chi1_hist, chi1_bins = np.histogram(chi1, bins=72, range=(-180, 180))
            chi2_hist, chi2_bins = np.histogram(chi2, bins=72, range=(-180, 180))
            chi1_centers = (chi1_bins[:-1] + chi1_bins[1:]) / 2
            chi2_centers = (chi2_bins[:-1] + chi2_bins[1:]) / 2
            
            chi1_peaks = detect_and_merge_peaks(chi1_hist)
            chi2_peaks = detect_and_merge_peaks(chi2_hist, merge_bins=5)
            
            angles.update({
                'chi1_hist': chi1_hist.tolist(),
                'chi2_hist': chi2_hist.tolist(),
                'chi1_centers': chi1_centers.tolist(),
                'chi2_centers': chi2_centers.tolist(),
                'chi1_peak_indices': chi1_peaks.tolist(),
                'chi2_peak_indices': chi2_peaks.tolist()
            })
            
            chi1_peak_angles = chi1_centers[chi1_peaks]
            chi2_peak_angles = chi2_centers[chi2_peaks]
            
            print(f"  χ1 peaks: {chi1_peak_angles}")
            print(f"  χ2 peaks: {chi2_peak_angles}")
            f.write(f"  χ1 peaks: {chi1_peak_angles}\n")
            f.write(f"  χ2 peaks: {chi2_peak_angles}\n")
            
            if 'chi3' in angles and angles['chi3'] is not None:
                chi3 = np.array(angles['chi3'])
                chi3_hist, chi3_bins = np.histogram(chi3, bins=72, range=(-180, 180))
                chi3_centers = (chi3_bins[:-1] + chi3_bins[1:]) / 2
                chi3_peaks = detect_and_merge_peaks(chi3_hist)
                
                angles.update({
                    'chi3_hist': chi3_hist.tolist(),
                    'chi3_centers': chi3_centers.tolist(),
                    'chi3_peak_indices': chi3_peaks.tolist()
                })
                
                chi3_peak_angles = chi3_centers[chi3_peaks]
                print(f"  χ3 peaks: {chi3_peak_angles}")
                f.write(f"  χ3 peaks: {chi3_peak_angles}\n")
            
            peak_combinations = {}
            for chi1_peak in chi1_peak_angles:
                for chi2_peak in chi2_peak_angles:
                    chi1_cond = (chi1 >= chi1_peak - peak_tolerance) & (chi1 <= chi1_peak + peak_tolerance)
                    chi2_cond = (chi2 >= chi2_peak - peak_tolerance) & (chi2 <= chi2_peak + peak_tolerance)
                    frames = np.where(chi1_cond & chi2_cond)[0]
                    
                    if len(frames) > 0:
                        combo_name = f"chi1_{chi1_peak:.0f}_chi2_{chi2_peak:.0f}"
                        peak_combinations[combo_name] = {
                            'chi1_peak': float(chi1_peak),
                            'chi2_peak': float(chi2_peak),
                            'frames': frames.tolist(),
                            'count': int(len(frames))
                        }
                        print(f"    {combo_name}: {len(frames)} frames")
                        f.write(f"    {combo_name}: {len(frames)} frames\n")
            
            peak_frames_dict[res_name] = {
                'pdb_label': pdb_label,
                'combinations': peak_combinations
            }
            f.write("\n")
    
    with open(f'{output_dir}/dihedral_peak_frames.json', 'w') as f:
        json.dump(peak_frames_dict, f, indent=2)
    
    print(f"\n✅ Peak analysis saved to: {output_dir}/dihedral_peak_frames.txt")
    print(f"✅ Peak frames saved to: {output_dir}/dihedral_peak_frames.json")
    
    return peak_frames_dict


def plot_dihedral_distributions(results, output_dir):
    """Plot dihedral distributions with peaks."""
    glu_results = {k: v for k, v in results.items() if v.get('residue_type') == 'GLU'}
    asn_asp_results = {k: v for k, v in results.items() if v.get('residue_type') in ['ASN', 'ASP']}
    
    if glu_results:
        n_res = len(glu_results)
        fig, axes = plt.subplots(n_res, 3, figsize=(15, 4*n_res))
        if n_res == 1:
            axes = axes.reshape(1, -1)
        
        for i, (res_name, angles) in enumerate(glu_results.items()):
            pdb_label = angles.get('pdb_label', res_name)
            
            for j, chi in enumerate(['chi1', 'chi2', 'chi3']):
                hist = np.array(angles.get(f'{chi}_hist', []))
                centers = np.array(angles.get(f'{chi}_centers', []))
                peaks = np.array(angles.get(f'{chi}_peak_indices', []))
                colors = ['skyblue', 'lightcoral', 'lightgreen']
                
                if len(hist) > 0:
                    axes[i, j].bar(centers, hist, width=5, alpha=0.7, color=colors[j])
                    if len(peaks) > 0:
                        axes[i, j].scatter(centers[peaks], hist[peaks], 
                                         color='red', s=80, marker='*', 
                                         label='Peaks', zorder=5)
                
                axes[i, j].set_title(f'{pdb_label} - {chi.upper()}', 
                                    fontsize=11, fontweight='bold')
                axes[i, j].set_xlabel('Angle (degrees)', fontsize=10)
                axes[i, j].set_ylabel('Count', fontsize=10)
                axes[i, j].set_xlim(-180, 180)
                axes[i, j].grid(True, alpha=0.3)
                if len(peaks) > 0:
                    axes[i, j].legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/dihedral_GLU_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/dihedral_GLU_distributions.png")
    
    if asn_asp_results:
        n_res = len(asn_asp_results)
        fig, axes = plt.subplots(n_res, 2, figsize=(10, 4*n_res))
        if n_res == 1:
            axes = axes.reshape(1, -1)
        
        for i, (res_name, angles) in enumerate(asn_asp_results.items()):
            pdb_label = angles.get('pdb_label', res_name)
            
            for j, chi in enumerate(['chi1', 'chi2']):
                hist = np.array(angles.get(f'{chi}_hist', []))
                centers = np.array(angles.get(f'{chi}_centers', []))
                peaks = np.array(angles.get(f'{chi}_peak_indices', []))
                colors = ['lightsteelblue', 'salmon']
                
                if len(hist) > 0:
                    axes[i, j].bar(centers, hist, width=5, alpha=0.7, color=colors[j])
                    if len(peaks) > 0:
                        axes[i, j].scatter(centers[peaks], hist[peaks], 
                                         color='red', s=80, marker='*', 
                                         label='Peaks', zorder=5)
                
                axes[i, j].set_title(f'{pdb_label} - {chi.upper()}', 
                                    fontsize=11, fontweight='bold')
                axes[i, j].set_xlabel('Angle (degrees)', fontsize=10)
                axes[i, j].set_ylabel('Count', fontsize=10)
                axes[i, j].set_xlim(-180, 180)
                axes[i, j].grid(True, alpha=0.3)
                if len(peaks) > 0:
                    axes[i, j].legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/dihedral_ASN_ASP_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/dihedral_ASN_ASP_distributions.png")


def analyze_chi2_buckets(results, output_dir):
    """
    Analyze chi1 angles in specific buckets and create bar plots.
    
    Buckets:
    - Bucket 1: -125° to -25° (Up conformation)
    - Bucket 2: -180° to -125° AND 125° to 180° (Down conformation)
    - Bucket 3: -25° to 125° (other orientations)
    """
    print("\n=== CHI1 BUCKET ANALYSIS ===")
    
    bucket_summary = {}
    
    with open(output_dir / 'chi1_bucket_analysis.txt', 'w') as f:
        f.write("CHI1 BUCKET ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write("Bucket definitions:\n")
        f.write("  Bucket 1 (Up): -125° to -25°\n")
        f.write("  Bucket 2 (Down): -180° to -125° AND 125° to 180°\n")
        f.write("  Bucket 3 (Other): -25° to 125°\n\n")
        f.write("="*70 + "\n\n")
        
        for res_name, angles in results.items():
            if angles['chi1'] is None:
                continue
                
            pdb_label = angles.get('pdb_label', res_name)
            chi1 = np.array(angles['chi1'])
            total_frames = len(chi1)
            
            # Define buckets
            bucket1_mask = (chi1 >= -125) & (chi1 <= -25)
            bucket2_mask = ((chi1 >= -180) & (chi1 <= -125)) | ((chi1 >= 125) & (chi1 <= 180))
            bucket3_mask = (chi1 > -25) & (chi1 < 125)
            
            bucket1_count = np.sum(bucket1_mask)
            bucket2_count = np.sum(bucket2_mask)
            bucket3_count = np.sum(bucket3_mask)
            
            bucket1_pct = (bucket1_count / total_frames) * 100
            bucket2_pct = (bucket2_count / total_frames) * 100
            bucket3_pct = (bucket3_count / total_frames) * 100
            
            # Store results
            bucket_summary[res_name] = {
                'pdb_label': pdb_label,
                'total_frames': total_frames,
                'bucket1': {'count': int(bucket1_count), 'percent': bucket1_pct},
                'bucket2': {'count': int(bucket2_count), 'percent': bucket2_pct},
                'bucket3': {'count': int(bucket3_count), 'percent': bucket3_pct}
            }
            
            # Write to file
            f.write(f"{res_name} (PDB: {pdb_label})\n")
            f.write("-"*70 + "\n")
            f.write(f"Total frames: {total_frames}\n\n")
            f.write(f"Bucket 1 (Up) [-125° to -25°]:  {bucket1_count:6d} frames ({bucket1_pct:5.1f}%)\n")
            f.write(f"Bucket 2 (Down) [-180° to -125° & 125° to 180°]: {bucket2_count:6d} frames ({bucket2_pct:5.1f}%)\n")
            f.write(f"Bucket 3 (Other) [-25° to 125°]:   {bucket3_count:6d} frames ({bucket3_pct:5.1f}%)\n")
            f.write("\n" + "="*70 + "\n\n")
            
            print(f"{res_name} (PDB: {pdb_label}):")
            print(f"  Bucket 1 (Up): {bucket1_count} frames ({bucket1_pct:.1f}%)")
            print(f"  Bucket 2 (Down): {bucket2_count} frames ({bucket2_pct:.1f}%)")
            print(f"  Bucket 3 (Other): {bucket3_count} frames ({bucket3_pct:.1f}%)")
    
    print(f"✅ Bucket analysis saved to: {output_dir}/chi1_bucket_analysis.txt")
    
    return bucket_summary


def plot_chi2_buckets(bucket_summary, output_dir):
    """Create 2x2 grid bar plots for chi1 bucket analysis."""
    print("\n=== CREATING CHI1 BUCKET PLOTS ===")
    
    # Separate ASN/ASP and GLU data
    asn_asp_data = {k: v for k, v in bucket_summary.items() 
                    if k.startswith('ASN') or k.startswith('ASP')}
    glu_data = {k: v for k, v in bucket_summary.items() 
                if k.startswith('GLU')}
    
    # =========================================================================
    # ASN/ASP BAR PLOTS
    # =========================================================================
    if asn_asp_data:
        # Order residues for grid: 184.B, 184.C, 184.A, 173.D
        pdb_order = ['184.B', '184.C', '184.A', '173.D']
        ordered_data = {}
        for pdb in pdb_order:
            for res_name, data in asn_asp_data.items():
                if data['pdb_label'] == pdb:
                    ordered_data[res_name] = data
                    break
        
        if len(ordered_data) != 4:
            print(f"Warning: Expected 4 ASN/ASP residues, got {len(ordered_data)}")
            ordered_data = asn_asp_data  # fallback
        
        residues_list = list(ordered_data.items())
        
        # Plot 1: ASN/ASP 2 buckets - counts
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            
            bucket_names = ['Up', 'Down']
            counts = [data['bucket1']['count'], data['bucket2']['count']]
            colors = ['#5DA5DA', '#FAA43A']
            
            bars = ax.bar(range(2), counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(2))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Number of Frames', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ASN_ASP_chi1_2buckets_counts_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/ASN_ASP_chi1_2buckets_counts_grid.png")
        
        # Plot 2: ASN/ASP 3 buckets - counts
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            
            bucket_names = ['Up', 'Down', 'Other']
            counts = [data['bucket1']['count'], data['bucket2']['count'], data['bucket3']['count']]
            colors = ['#5DA5DA', '#FAA43A', '#60BD68']
            
            bars = ax.bar(range(3), counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Number of Frames', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ASN_ASP_chi1_3buckets_counts_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/ASN_ASP_chi1_3buckets_counts_grid.png")
        
        # Plot 3: ASN/ASP 2 buckets - percentages
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            total = data['total_frames']
            
            bucket_names = ['Up', 'Down']
            percents = [(data['bucket1']['count']/total)*100, (data['bucket2']['count']/total)*100]
            colors = ['#5DA5DA', '#FAA43A']
            
            bars = ax.bar(range(2), percents, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(2))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, pct in zip(bars, percents):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.1f}%',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ASN_ASP_chi1_2buckets_percent_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/ASN_ASP_chi1_2buckets_percent_grid.png")
        
        # Plot 4: ASN/ASP 3 buckets - percentages
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            
            bucket_names = ['Up', 'Down', 'Other']
            percents = [data['bucket1']['percent'], data['bucket2']['percent'], data['bucket3']['percent']]
            colors = ['#5DA5DA', '#FAA43A', '#60BD68']
            
            bars = ax.bar(range(3), percents, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, pct in zip(bars, percents):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.1f}%',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ASN_ASP_chi1_3buckets_percent_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/ASN_ASP_chi1_3buckets_percent_grid.png")
    
    # =========================================================================
    # GLU BAR PLOTS
    # =========================================================================
    if glu_data:
        # Order residues for grid: 152.B, 152.C, 152.A, 141.D
        pdb_order_glu = ['152.B', '152.C', '152.A', '141.D']
        ordered_glu = {}
        for pdb in pdb_order_glu:
            for res_name, data in glu_data.items():
                if data['pdb_label'] == pdb:
                    ordered_glu[res_name] = data
                    break
        
        if len(ordered_glu) != 4:
            print(f"Warning: Expected 4 GLU residues, got {len(ordered_glu)}")
            ordered_glu = glu_data  # fallback
        
        residues_list_glu = list(ordered_glu.items())
        
        # Plot 1: GLU 2 buckets - counts
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list_glu[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            
            bucket_names = ['Up', 'Down']
            counts = [data['bucket1']['count'], data['bucket2']['count']]
            colors = ['#5DA5DA', '#FAA43A']
            
            bars = ax.bar(range(2), counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(2))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Number of Frames', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'GLU_chi1_2buckets_counts_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/GLU_chi1_2buckets_counts_grid.png")
        
        # Plot 2: GLU 3 buckets - counts
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list_glu[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            
            bucket_names = ['Up', 'Down', 'Other']
            counts = [data['bucket1']['count'], data['bucket2']['count'], data['bucket3']['count']]
            colors = ['#5DA5DA', '#FAA43A', '#60BD68']
            
            bars = ax.bar(range(3), counts, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Number of Frames', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'GLU_chi1_3buckets_counts_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/GLU_chi1_3buckets_counts_grid.png")
        
        # Plot 3: GLU 2 buckets - percentages
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list_glu[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            total = data['total_frames']
            
            bucket_names = ['Up', 'Down']
            percents = [(data['bucket1']['count']/total)*100, (data['bucket2']['count']/total)*100]
            colors = ['#5DA5DA', '#FAA43A']
            
            bars = ax.bar(range(2), percents, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(2))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, pct in zip(bars, percents):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.1f}%',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'GLU_chi1_2buckets_percent_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/GLU_chi1_2buckets_percent_grid.png")
        
        # Plot 4: GLU 3 buckets - percentages
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (res_name, data) in enumerate(residues_list_glu[:4]):
            ax = axes[idx]
            pdb_label = data['pdb_label']
            
            bucket_names = ['Up', 'Down', 'Other']
            percents = [data['bucket1']['percent'], data['bucket2']['percent'], data['bucket3']['percent']]
            colors = ['#5DA5DA', '#FAA43A', '#60BD68']
            
            bars = ax.bar(range(3), percents, color=colors, alpha=0.85, edgecolor='black', linewidth=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels(bucket_names, fontsize=20, fontweight='bold')
            ax.set_ylabel('Percentage (%)', fontsize=24, fontweight='bold')
            ax.set_title(f'{pdb_label}', fontsize=26, fontweight='bold', pad=10)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3, linewidth=1)
            ax.tick_params(axis='both', labelsize=20)
            
            for bar, pct in zip(bars, percents):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.1f}%',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'GLU_chi1_3buckets_percent_grid.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/GLU_chi1_3buckets_percent_grid.png")


def plot_simple_histograms(results, output_dir):
    """
    Create 2x2 grid histograms for chi1 and chi2 separately.
    For ASN/ASP: 184.B, 184.C, 184.A, 173.D
    For GLU: 152.B, 152.C, 152.A, 141.D
    """
    print("\n=== CREATING SIMPLE HISTOGRAM PLOTS ===")
    
    # Separate by residue type
    asn_asp_data = {k: v for k, v in results.items() 
                    if v.get('residue_type') in ['ASN', 'ASP']}
    glu_data = {k: v for k, v in results.items() 
                if v.get('residue_type') == 'GLU'}
    
    # =========================================================================
    # ASN/ASP Chi1 Grid (2x2)
    # =========================================================================
    if asn_asp_data:
        # Order: 184.B, 184.C, 184.A, 173.D
        pdb_order = ['184.B', '184.C', '184.A', '173.D']
        ordered_asn = {}
        for pdb in pdb_order:
            for res_name, angles in asn_asp_data.items():
                if angles.get('pdb_label') == pdb:
                    ordered_asn[res_name] = angles
                    break
        
        if len(ordered_asn) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            
            for idx, (res_name, angles) in enumerate(list(ordered_asn.items())[:4]):
                ax = axes[idx]
                pdb_label = angles.get('pdb_label', res_name)
                chi1 = np.array(angles['chi1'])
                
                hist, bins = np.histogram(chi1, bins=72, range=(-180, 180))
                centers = (bins[:-1] + bins[1:]) / 2
                
                ax.bar(centers, hist, width=5, alpha=0.85, color='steelblue', edgecolor='black', linewidth=0.8)
                ax.set_xlabel('χ1 Angle (degrees)', fontsize=26, fontweight='bold')
                ax.set_ylabel('Count', fontsize=26, fontweight='bold')
                ax.set_title(f'{pdb_label}', fontsize=28, fontweight='bold', pad=15)
                ax.set_xlim(-180, 180)
                ax.tick_params(axis='both', labelsize=22)
                ax.grid(True, alpha=0.3, linewidth=1)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'ASN_ASP_chi1_grid.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {output_dir}/ASN_ASP_chi1_grid.png")
    
    # =========================================================================
    # ASN/ASP Chi2 Grid (2x2)
    # =========================================================================
    if asn_asp_data:
        if len(ordered_asn) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            
            for idx, (res_name, angles) in enumerate(list(ordered_asn.items())[:4]):
                ax = axes[idx]
                pdb_label = angles.get('pdb_label', res_name)
                chi2 = np.array(angles['chi2'])
                
                hist, bins = np.histogram(chi2, bins=72, range=(-180, 180))
                centers = (bins[:-1] + bins[1:]) / 2
                
                ax.bar(centers, hist, width=5, alpha=0.85, color='coral', edgecolor='black', linewidth=0.8)
                ax.set_xlabel('χ2 Angle (degrees)', fontsize=26, fontweight='bold')
                ax.set_ylabel('Count', fontsize=26, fontweight='bold')
                ax.set_title(f'{pdb_label}', fontsize=28, fontweight='bold', pad=15)
                ax.set_xlim(-180, 180)
                ax.tick_params(axis='both', labelsize=22)
                ax.grid(True, alpha=0.3, linewidth=1)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'ASN_ASP_chi2_grid.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {output_dir}/ASN_ASP_chi2_grid.png")
    
    # =========================================================================
    # GLU Chi1 Grid (2x2)
    # =========================================================================
    if glu_data:
        # Order: 152.B, 152.C, 152.A, 141.D
        pdb_order_glu = ['152.B', '152.C', '152.A', '141.D']
        ordered_glu = {}
        for pdb in pdb_order_glu:
            for res_name, angles in glu_data.items():
                if angles.get('pdb_label') == pdb:
                    ordered_glu[res_name] = angles
                    break
        
        if len(ordered_glu) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            
            for idx, (res_name, angles) in enumerate(list(ordered_glu.items())[:4]):
                ax = axes[idx]
                pdb_label = angles.get('pdb_label', res_name)
                chi1 = np.array(angles['chi1'])
                
                hist, bins = np.histogram(chi1, bins=72, range=(-180, 180))
                centers = (bins[:-1] + bins[1:]) / 2
                
                ax.bar(centers, hist, width=5, alpha=0.85, color='steelblue', edgecolor='black', linewidth=0.8)
                ax.set_xlabel('χ1 Angle (degrees)', fontsize=26, fontweight='bold')
                ax.set_ylabel('Count', fontsize=26, fontweight='bold')
                ax.set_title(f'{pdb_label}', fontsize=28, fontweight='bold', pad=15)
                ax.set_xlim(-180, 180)
                ax.tick_params(axis='both', labelsize=22)
                ax.grid(True, alpha=0.3, linewidth=1)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'GLU_chi1_grid.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {output_dir}/GLU_chi1_grid.png")
    
    # =========================================================================
    # GLU Chi2 Grid (2x2)
    # =========================================================================
    if glu_data:
        if len(ordered_glu) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            axes = axes.flatten()
            
            for idx, (res_name, angles) in enumerate(list(ordered_glu.items())[:4]):
                ax = axes[idx]
                pdb_label = angles.get('pdb_label', res_name)
                chi2 = np.array(angles['chi2'])
                
                hist, bins = np.histogram(chi2, bins=72, range=(-180, 180))
                centers = (bins[:-1] + bins[1:]) / 2
                
                ax.bar(centers, hist, width=5, alpha=0.85, color='coral', edgecolor='black', linewidth=0.8)
                ax.set_xlabel('χ2 Angle (degrees)', fontsize=26, fontweight='bold')
                ax.set_ylabel('Count', fontsize=26, fontweight='bold')
                ax.set_title(f'{pdb_label}', fontsize=28, fontweight='bold', pad=15)
                ax.set_xlim(-180, 180)
                ax.tick_params(axis='both', labelsize=22)
                ax.grid(True, alpha=0.3, linewidth=1)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'GLU_chi2_grid.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {output_dir}/GLU_chi2_grid.png")


def run_dihedral_analysis(u, channel_type, results_dir):
    """
    Main function to run complete dihedral analysis.
    This is called from your main analysis script.
    
    Parameters
    ----------
    u : MDAnalysis.Universe
        Universe with topology and trajectory loaded
    channel_type : str
        Channel type ('G2', 'G12', 'G12_GAT', 'G12_ML')
    results_dir : Path or str
        Directory to save results
    
    Returns
    -------
    results : dict
        Dihedral angle data
    peak_frames : dict
        Peak-based frame information
    """
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("STARTING DIHEDRAL ORIENTATION ANALYSIS")
    print("="*70)
    
    try:
        # Step 1: Analyze dihedrals
        results = analyze_residue_orientations(u, channel_type)
        
        if not results:
            print("\nWARNING: No dihedral results obtained")
            return None, None
        
        # Step 2: Find peaks and frames
        peak_frames = find_peak_based_frames(results, results_dir)
        
        # Step 3: Plot distributions
        plot_dihedral_distributions(results, results_dir)
        
        # Step 4: NEW - Analyze chi2 buckets
        bucket_summary = analyze_chi2_buckets(results, results_dir)
        
        # Step 5: NEW - Plot chi2 buckets
        plot_chi2_buckets(bucket_summary, results_dir)
        
        # Step 6: NEW - Create simple 2x2 histograms
        plot_simple_histograms(results, results_dir)
        
        # Step 7: Save raw data
        with open(results_dir / 'dihedral_raw_data.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"✅ Raw data saved to: {results_dir}/dihedral_raw_data.pkl")
        
        print("\n" + "="*70)
        print("DIHEDRAL ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved in: {results_dir}/")
        print("  - dihedral_peak_frames.txt")
        print("  - dihedral_peak_frames.json")
        print("  - dihedral_GLU_distributions.png")
        print("  - dihedral_ASN_ASP_distributions.png")
        print("  - chi1_bucket_analysis.txt")
        print("  ASN/ASP χ1 bar plots (2x2 grids):")
        print("    - ASN_ASP_chi1_2buckets_counts_grid.png")
        print("    - ASN_ASP_chi1_3buckets_counts_grid.png")
        print("    - ASN_ASP_chi1_2buckets_percent_grid.png")
        print("    - ASN_ASP_chi1_3buckets_percent_grid.png")
        print("  GLU χ1 bar plots (2x2 grids):")
        print("    - GLU_chi1_2buckets_counts_grid.png")
        print("    - GLU_chi1_3buckets_counts_grid.png")
        print("    - GLU_chi1_2buckets_percent_grid.png")
        print("    - GLU_chi1_3buckets_percent_grid.png")
        print("  Histogram grids (2x2):")
        print("    - ASN_ASP_chi1_grid.png")
        print("    - ASN_ASP_chi2_grid.png")
        print("    - GLU_chi1_grid.png")
        print("    - GLU_chi2_grid.png")
        print("  - dihedral_raw_data.pkl")
        print("="*70 + "\n")
        
        return results, peak_frames
        
    except Exception as e:
        print(f"\nERROR in dihedral analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None