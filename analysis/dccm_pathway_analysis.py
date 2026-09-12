"""
analysis/dccm_pathway_analysis.py
===================================
Computes DCCM (Dynamic Cross-Correlation Matrix) for all protein residues
over a defined frame range, then finds allosteric communication pathways
between residue pairs of interest.

Steps:
  1. Build reverse lookup: PDB label -> raw resid (from MDAnalysis universe)
  2. Compute DCCM (skip if already saved)
  3. Compute average Cα distance matrix (skip if already saved)
  4. Build correlation graph
  5. For each residue pair: find shortest pathway, save HTML + CSV
  6. For each unique residue: save ranked correlation CSV vs all others

All residue labels in outputs use PDB format (e.g. "60.2b", "138.2b").

Called from main.py:

    from analysis.dccm_pathway_analysis import run_dccm_pathway_analysis

    run_dccm_pathway_analysis(
        u              = u,
        results_dir    = results_dir,
        channel_type   = args.channel_type,
        start_frame    = 0,
        end_frame      = 3000,
        residue_pairs  = [
            ("143.2b", "138.2b"),   # GLN197 -> PHE192
            ("60.2b",  "138.2b"),   # LYS64  -> PHE192
        ],
        corr_threshold = 0.3,
        dist_cutoff    = 6.0,
    )
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("  WARNING: plotly not installed -- 3D HTML visualization will be skipped.")


# -----------------------------------------------------------------------------
# Reverse lookup: PDB label -> raw resid
# -----------------------------------------------------------------------------

def _build_pdb_to_raw(u, channel_type: str) -> Dict[str, int]:
    """
    Loop all protein residues, apply to_pdb(), build reverse mapping.
    e.g. {"60.2b": 197, "138.2b": 192, ...}
    """
    try:
        from analysis.converter import to_pdb
    except ImportError:
        def to_pdb(resid, ct): return str(resid)

    protein   = u.select_atoms("protein and name CA")
    pdb_to_raw = {}
    raw_to_pdb = {}
    for res in protein.residues:
        pdb_label = to_pdb(res.resid, channel_type)
        pdb_to_raw[pdb_label] = res.resid
        raw_to_pdb[res.resid]  = pdb_label
    return pdb_to_raw, raw_to_pdb


# -----------------------------------------------------------------------------
# DCCM computation
# -----------------------------------------------------------------------------

def _compute_dccm(u, start_frame: int, end_frame: int) -> Tuple[np.ndarray, List[int]]:
    """Compute DCCM for all Cα atoms between start_frame and end_frame."""
    protein  = u.select_atoms("protein and name CA")
    n_atoms  = len(protein)
    n_frames = end_frame - start_frame + 1
    resids   = [res.resid for res in protein.residues]

    print(f"  Computing DCCM: {n_atoms} residues x {n_frames} frames...")

    pos_array = np.zeros((n_frames, n_atoms, 3))
    for i, ts in enumerate(tqdm(
        u.trajectory[start_frame:end_frame + 1],
        total=n_frames, desc="  Loading frames", leave=False
    )):
        pos_array[i] = protein.positions

    mean_structure = np.mean(pos_array, axis=0)
    fluctuation    = pos_array - mean_structure

    print("  Building covariance matrix...")
    covariance = np.zeros((n_atoms, n_atoms))
    for i in tqdm(range(n_atoms), desc="  Covariance", leave=False):
        for j in range(i, n_atoms):
            val = np.mean(np.sum(fluctuation[:, i, :] * fluctuation[:, j, :], axis=1))
            covariance[i, j] = val
            covariance[j, i] = val

    diag = np.diag(covariance)
    dccm = covariance / np.sqrt(np.outer(diag, diag))
    print(f"  DCCM shape: {dccm.shape}")
    return dccm, resids


# -----------------------------------------------------------------------------
# Average distance matrix
# -----------------------------------------------------------------------------

def _compute_avg_distance_matrix(u, start_frame: int, end_frame: int) -> np.ndarray:
    """Compute average pairwise Cα distance matrix over frame range."""
    protein  = u.select_atoms("protein and name CA")
    n_atoms  = len(protein)
    n_frames = end_frame - start_frame + 1
    dist_sum = np.zeros((n_atoms, n_atoms))

    print(f"  Computing average distance matrix ({n_atoms} x {n_atoms})...")
    for ts in tqdm(
        u.trajectory[start_frame:end_frame + 1],
        total=n_frames, desc="  Distance frames", leave=False
    ):
        pos = protein.positions
        for i in range(n_atoms):
            diff           = pos - pos[i]
            dists          = np.sqrt((diff ** 2).sum(axis=1))
            dist_sum[i]   += dists

    return dist_sum / n_frames


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------

def _build_graph(
    dccm:           np.ndarray,
    dist_matrix:    np.ndarray,
    resids:         List[int],
    corr_threshold: float,
    dist_cutoff:    float,
) -> nx.Graph:
    """Build correlation graph: edges where |DCCM| >= threshold AND distance <= cutoff."""
    G = nx.Graph()
    for resid in resids:
        G.add_node(resid)

    n = len(resids)
    for i in range(n):
        for j in range(i + 1, n):
            corr = abs(dccm[i, j])
            dist = dist_matrix[i, j]
            if corr >= corr_threshold and dist <= dist_cutoff:
                G.add_edge(resids[i], resids[j], weight=1.0 / corr)
    return G


# -----------------------------------------------------------------------------
# Pathway finding
# -----------------------------------------------------------------------------

def _find_pathway(
    G:       nx.Graph,
    dccm:    np.ndarray,
    resids:  List[int],
    src_raw: int,
    tgt_raw: int,
) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    """Find shortest path between src and tgt. Returns (path, correlations)."""
    if src_raw not in G or tgt_raw not in G:
        return None, None
    try:
        path = nx.shortest_path(G, source=src_raw, target=tgt_raw, weight="weight")
    except nx.NetworkXNoPath:
        return None, None

    idx_map = {r: i for i, r in enumerate(resids)}
    corrs   = []
    for k in range(len(path) - 1):
        i = idx_map[path[k]]
        j = idx_map[path[k + 1]]
        corrs.append(float(dccm[i, j]))

    return path, corrs


# -----------------------------------------------------------------------------
# 3D HTML visualization
# -----------------------------------------------------------------------------

def _save_3d_html(
    u:          object,
    G:          nx.Graph,
    path:       List[int],
    raw_to_pdb: Dict[int, str],
    out_path:   Path,
    frame:      int = 0,
):
    if not HAS_PLOTLY:
        print("    [3D viz] plotly not available -- skipping.")
        return

    u.trajectory[frame]
    protein  = u.select_atoms("protein and name CA")
    raw_pos  = {res.resid: tuple(atom.position)
                for res, atom in zip(protein.residues, protein.atoms)}

    center    = np.mean(list(raw_pos.values()), axis=0)
    positions = {node: np.array(raw_pos[node]) - center
                 for node in G.nodes if node in raw_pos}

    path_set   = set(path)
    path_edges = [(path[k], path[k + 1]) for k in range(len(path) - 1)]

    def edge_coords(edges):
        x, y, z = [], [], []
        for a, b in edges:
            if a in positions and b in positions:
                x += [positions[a][0], positions[b][0], None]
                y += [positions[a][1], positions[b][1], None]
                z += [positions[a][2], positions[b][2], None]
        return x, y, z

    other_edges = [e for e in G.edges if e not in path_edges and (e[1], e[0]) not in path_edges]

    px, py, pz = edge_coords(path_edges)
    ox, oy, oz = edge_coords(other_edges)

    traces = [
        go.Scatter3d(x=ox, y=oy, z=oz,
                     mode="lines", line=dict(width=1, color="lightgray"),
                     hoverinfo="skip", name="Other edges"),
        go.Scatter3d(x=px, y=py, z=pz,
                     mode="lines", line=dict(width=5, color="red"),
                     hoverinfo="skip", name="Pathway"),
    ]

    # nodes
    for node in G.nodes:
        if node not in positions:
            continue
        x, y, z = positions[node]
        label   = raw_to_pdb.get(node, str(node))
        is_path = node in path_set
        traces.append(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers",
            marker=dict(size=15 if is_path else 4,
                        color="red" if is_path else "gray"),
            text=[label],
            hoverinfo="text",
            showlegend=False,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Allosteric pathway: "
              f"{raw_to_pdb.get(path[0], path[0])} → "
              f"{raw_to_pdb.get(path[-1], path[-1])}",
        showlegend=True,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
    )
    fig.write_html(str(out_path))
    print(f"    [3D viz] Saved -> {out_path.name}")


# -----------------------------------------------------------------------------
# Ranked correlation CSV for a single residue
# -----------------------------------------------------------------------------

def _save_ranked_correlations(
    resid:      int,
    dccm:       np.ndarray,
    resids:     List[int],
    raw_to_pdb: Dict[int, str],
    out_path:   Path,
):
    idx     = resids.index(resid)
    row     = dccm[idx]
    pdb_lbl = raw_to_pdb.get(resid, str(resid))
    rows    = []
    for j, other_resid in enumerate(resids):
        if j == idx:
            continue
        rows.append({
            "residue":     raw_to_pdb.get(other_resid, str(other_resid)),
            f"dccm_{pdb_lbl}": round(float(row[j]), 4),
        })
    df = pd.DataFrame(rows).sort_values(f"dccm_{pdb_lbl}",
                                         key=abs, ascending=False)
    df.to_csv(out_path, index=False)
    print(f"    [ranked corr] Saved -> {out_path.name}")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def run_dccm_pathway_analysis(
    u,
    results_dir:    Path,
    channel_type:   str,
    start_frame:    int,
    end_frame:      int,
    residue_pairs:  List[Tuple[str, str]],   # PDB labels e.g. [("143.2b", "138.2b")]
    corr_threshold: float = 0.3,
    dist_cutoff:    float = 6.0,
):
    results_dir = Path(results_dir)
    out_dir     = results_dir / "dccm_pathway_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DCCM PATHWAY ANALYSIS")
    print(f"  results_dir    : {results_dir}")
    print(f"  frames         : {start_frame} -> {end_frame}")
    print(f"  corr_threshold : {corr_threshold}")
    print(f"  dist_cutoff    : {dist_cutoff} Å")
    print(f"  residue_pairs  : {residue_pairs}")
    print("=" * 70)

    # --- build PDB <-> raw lookup ---
    print("\n  Building PDB <-> raw resid lookup...")
    pdb_to_raw, raw_to_pdb = _build_pdb_to_raw(u, channel_type)

    # validate all requested residues
    all_requested_pdb = set()
    for src_pdb, tgt_pdb in residue_pairs:
        all_requested_pdb.add(src_pdb)
        all_requested_pdb.add(tgt_pdb)

    for pdb_lbl in all_requested_pdb:
        if pdb_lbl not in pdb_to_raw:
            print(f"  ERROR: '{pdb_lbl}' not found in protein residues.")
            print(f"  Available labels sample: {list(pdb_to_raw.keys())[:10]}")
            return

    # --- DCCM (skip if exists) ---
    dccm_path = out_dir / "dccm_matrix.pkl"
    resids_path = out_dir / "dccm_resids.pkl"

    if dccm_path.exists() and resids_path.exists():
        print(f"\n  dccm_matrix.pkl already exists -- loading...")
        with open(dccm_path, "rb") as f:
            dccm = pickle.load(f)
        with open(resids_path, "rb") as f:
            resids = pickle.load(f)
    else:
        dccm, resids = _compute_dccm(u, start_frame, end_frame)
        with open(dccm_path, "wb") as f:
            pickle.dump(dccm, f)
        with open(resids_path, "wb") as f:
            pickle.dump(resids, f)
        print(f"  Saved DCCM -> {dccm_path}")

    # --- average distance matrix (skip if exists) ---
    dist_path = out_dir / "avg_distance_matrix.pkl"

    if dist_path.exists():
        print(f"  avg_distance_matrix.pkl already exists -- loading...")
        with open(dist_path, "rb") as f:
            dist_matrix = pickle.load(f)
    else:
        dist_matrix = _compute_avg_distance_matrix(u, start_frame, end_frame)
        with open(dist_path, "wb") as f:
            pickle.dump(dist_matrix, f)
        print(f"  Saved distance matrix -> {dist_path}")

    # --- build graph ---
    print(f"\n  Building correlation graph "
          f"(corr >= {corr_threshold}, dist <= {dist_cutoff} Å)...")
    G = _build_graph(dccm, dist_matrix, resids, corr_threshold, dist_cutoff)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- ranked correlations for each unique residue ---
    ranked_dir = out_dir / "ranked_correlations"
    ranked_dir.mkdir(exist_ok=True)
    for pdb_lbl in sorted(all_requested_pdb):
        raw = pdb_to_raw[pdb_lbl]
        if raw not in resids:
            print(f"  WARNING: {pdb_lbl} (raw {raw}) not in DCCM resids -- skipping ranked CSV.")
            continue
        _save_ranked_correlations(
            resid      = raw,
            dccm       = dccm,
            resids     = resids,
            raw_to_pdb = raw_to_pdb,
            out_path   = ranked_dir / f"ranked_{pdb_lbl.replace('.', '_')}.csv",
        )

    # --- pathways ---
    for src_pdb, tgt_pdb in residue_pairs:
        pair_label = f"{src_pdb}_{tgt_pdb}".replace(".", "_")
        pair_dir   = out_dir / pair_label
        pair_dir.mkdir(exist_ok=True)

        print(f"\n  Pathway: {src_pdb} -> {tgt_pdb}")

        src_raw = pdb_to_raw[src_pdb]
        tgt_raw = pdb_to_raw[tgt_pdb]

        path, corrs = _find_pathway(G, dccm, resids, src_raw, tgt_raw)

        if path is None:
            print(f"    No path found between {src_pdb} and {tgt_pdb}.")
            print(f"    Try lowering corr_threshold or increasing dist_cutoff.")
            continue

        # convert path to PDB labels
        path_pdb = [raw_to_pdb.get(r, str(r)) for r in path]
        print(f"    Path ({len(path)} residues): {' -> '.join(path_pdb)}")

        # save pathway CSV
        pathway_csv = pair_dir / "pathway.csv"
        rows = []
        for k, (r, pdb) in enumerate(zip(path, path_pdb)):
            corr_to_next = corrs[k] if k < len(corrs) else None
            rows.append({
                "step":            k,
                "residue_pdb":     pdb,
                "residue_raw":     r,
                "dccm_to_next":    round(corr_to_next, 4) if corr_to_next is not None else "",
            })
        pd.DataFrame(rows).to_csv(pathway_csv, index=False)
        print(f"    Saved pathway CSV -> {pathway_csv.name}")

        # save 3D HTML
        if HAS_PLOTLY:
            _save_3d_html(
                u          = u,
                G          = G,
                path       = path,
                raw_to_pdb = raw_to_pdb,
                out_path   = pair_dir / "pathway_3d.html",
                frame      = start_frame,
            )

    print(f"\n✓ DCCM pathway analysis complete.")
    print(f"  Results in: {out_dir}")