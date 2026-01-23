import os
import gc
import numpy as np
from tqdm import tqdm
import MDAnalysis as mda
from openmm.app import AmberPrmtopFile, Simulation, NoCutoff
from openmm import Platform
from openmm.unit import kelvin, picoseconds
from openmm.openmm import LangevinIntegrator

def calculate_ionic_forces_all_frames(prmtop_path, nc_path, output_dir="./forces_cache_ions"):
    """
    Calculates and caches total forces from OpenMM only for K+ ions across all frames.

    Returns:
        force_data: dict - {frame: {resid: np.array([fx, fy, fz])}}
        atom_index_map: dict - {resid: atom_index}
    """
    os.makedirs(output_dir, exist_ok=True)
    force_file = os.path.join(output_dir, "ionic_forces_scaled_removeCMMotionF.npy")
    index_map_file = os.path.join(output_dir, "atom_index_map_scaled_removeCMMotionF.npy")

    if os.path.exists(force_file) and os.path.exists(index_map_file):
        print("Loading cached ionic forces...")
        return np.load(force_file, allow_pickle=True).item(), np.load(index_map_file, allow_pickle=True).item()

    u = mda.Universe(prmtop_path, nc_path)
    prmtop = AmberPrmtopFile(prmtop_path)
    # system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=None)
    system = prmtop.createSystem(
        nonbondedMethod=NoCutoff,
        constraints=None,
        removeCMMotion=False
    )
    integrator = LangevinIntegrator(300 * kelvin, 1 / picoseconds, 0.002 * picoseconds)
    platform = Platform.getPlatformByName("CUDA")
    simulation = Simulation(prmtop.topology, system, integrator, platform)

    atom_index_map = {}
    for atom in prmtop.topology.atoms():
        resid = atom.residue.id
        if resid.isdigit() and atom.residue.name == 'K+':
            atom_index_map[int(resid)] = atom.index

    force_data = {}
    for ts in tqdm(u.trajectory, desc="Calculating ionic forces per frame"):
        simulation.context.setPositions(ts.positions * 0.1)  # ✅ Å to nm
        state = simulation.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)
        forces = forces * 0.0239  # Convert forces from kJ/mol/nm to kcal/mol/Å

        frame_forces = {resid: np.array(forces[idx]) for resid, idx in atom_index_map.items()}
        force_data[ts.frame] = frame_forces

        gc.collect()  # ✅ free memory after every frame

    np.save(force_file, force_data)
    np.save(index_map_file, atom_index_map)
    return force_data, atom_index_map
