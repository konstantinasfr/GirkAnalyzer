# PermeationAnalyzer Class

The `PermeationAnalyzer` is a high-level orchestrator class for analyzing ion permeation events in molecular dynamics (MD) simulations. It coordinates multiple per-frame analyses and stores the results in a structured format.

---

## 🧠 Purpose

This class handles:
- Extracting ion positions across MD frames
- Calculating electrostatic forces from nearby ions
- Measuring radial distances relative to a defined channel axis
- Identifying nearby residues per ion per frame

It is designed to work with pre-extracted permeation events and MDAnalysis universes.

---

## ⚙️ Initialization

```python
analyzer = PermeationAnalyzer(
    ch2_permation_residues=...,    # List of permeation event dicts
    u=...,                         # MDAnalysis Universe
    start_frame=...,               # Frame index to begin analysis
    end_frame=...,                 # Frame index to end analysis
    min_results_per_frame=...,     # Minimum per-frame result count
    ch2=...,                       # Channel atoms for radial calc
    close_contacts_dict=...,       # Dict of closest residues per ion per frame
    cutoff=15.0,                   # Distance cutoff for force/residue analysis
    calculate_total_force=False,   # Whether to compute OpenMM forces
    prmtop_file=None,              # For OpenMM force calculation
    nc_file=None,                  # For OpenMM force calculation
    output_base_dir=None           # Optional directory for saving results
)


### 1. `analyze_forces(...)` – Electrostatic Interaction

[🔍 See full force analysis details →](force_analysis.md)

- Calculates the **Coulomb force** between the permeating ion and nearby ions.
- Computes:
  - Ionic force vector
  - Motion vector
  - Cosine similarity
  - Radial vs axial decomposition
