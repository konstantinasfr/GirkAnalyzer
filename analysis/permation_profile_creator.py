import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import wilcoxon
import json
import matplotlib.pyplot as plt
from collections import Counter
from analysis.calculate_openmm_forces import calculate_ionic_forces_all_frames
from analysis.force_analysis import analyze_forces
from analysis.radial_distance_analysis import analyze_radial_distances
from analysis.close_residues_analysis import analyze_close_residues, get_last_nth_frame_close_residues, plot_residue_counts, analyze_residue_combinations
from analysis.intervals_force_analysis import analyze_force_intervals
from collections import defaultdict
import gc

class PermeationAnalyzer:
    def __init__(self, ch2_permation_residues, ch1_permeation_events, ch2_permeation_events, u, start_frame, end_frame, min_results_per_frame,
                 ch2, close_contacts_dict, total_residue_comb_over_all_frames, glu_residues, asn_residues, sf_residues, cutoff=15.0, calculate_total_force=False,
                prmtop_file=None, nc_file=None,output_base_dir=None, calculate_intervals=False):
        """
        Initializes the analyzer with all necessary inputs and automatically
        runs the analysis, storing the results as attributes.
        """
        self.ch2_permation_residues = ch2_permation_residues
        self.ch1_permeation_events = ch1_permeation_events
        self.ch2_permeation_events = ch2_permeation_events
        self.u = u
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.min_results_per_frame = min_results_per_frame
        self.ch2 = ch2
        self.close_contacts_dict = close_contacts_dict
        self.total_residue_comb_over_all_frames = total_residue_comb_over_all_frames
        self.cutoff = cutoff
        self.calculate_total_force = calculate_total_force
        self.prmtop_file = prmtop_file
        self.nc_file = nc_file
        self.output_base_dir = output_base_dir
        self.glu_residues = glu_residues
        self.asn_residues = asn_residues
        self.sf_residues = sf_residues
        self.calculate_intervals = calculate_intervals
        self.force_results = []
        self.radial_distances_results = []
        self.close_residues_results = []
        self.close_residues_result_per_frame = {}
        self.force_intervals_results = []

    def _build_all_positions(self, ion_selection='resname K+ K'):
        """
        Extract ion positions from trajectory within a frame range.
        Returns:
            dict: {frame: {ion_id: np.array([x, y, z])}}
        """
        all_positions = {}
        ions = self.u.select_atoms(ion_selection)
        trajectory_slice = self.u.trajectory[self.start_frame:self.end_frame+1]

        for ts in tqdm(trajectory_slice, desc=f"Extracting positions ({self.start_frame}:{self.end_frame})"):
            frame_dict = {ion.resid: ion.position.copy() for ion in ions}
            all_positions[ts.frame] = frame_dict
        return all_positions

    def _build_residue_positions(self, residue_list, atom_names):
        """
        Store coordinates of specified atoms in residues across frames.
        Returns: {frame: {(resid, atom_name): np.array([x, y, z])}}
        """

        residue_pos = {}
        # it selects only the atoms that match both criteria per residue
        selection = "resid " + " ".join(str(r) for r in residue_list) + " and name " + " ".join(atom_names)
        atoms = self.u.select_atoms(selection)
        trajectory_slice = self.u.trajectory[self.start_frame:self.end_frame]

        for ts in tqdm(trajectory_slice, desc=f"Building residue positions ({self.start_frame}:{self.end_frame})"):
            frame_data = {}
            for atom in atoms:
                frame_data[(atom.resid, atom.name)] = atom.position.copy()
            residue_pos[ts.frame] = frame_data

        return residue_pos

    def _build_charge_map(self, pip2_resnames="PIP", unique_pip2_atom_names=None, residues_names=['ASN', 'GLU'], unique_res_atom_names=None, ion_selection='resname K+ K'):
        """
        Returns a charge map {(resid, atom_name): charge} for selected ions, GLU, ASN, and PIP2 atoms.

        - Ions get +1.0
        - GLU, ASN, and PIP2 atoms use their actual force field charges
        - More robust than using just atom names
        """
        charge_map = {}

        # --- Add ions (using resid to identify ion uniquely) ---
        ions = self.u.select_atoms(ion_selection)
        for ion in ions:
            charge_map[ion.resid] = 1.0  # atom index, not resid

        # --- Add residues atoms ---
        for resname in residues_names:
            res_atoms = self.u.select_atoms(f"resname {resname} and name {' '.join(str(item) for item in unique_res_atom_names)}")
            for atom in res_atoms:
                charge_map[(resname, atom.name)] = atom.charge


        # --- Add PIP2 atoms ---
        pip2_atoms = self.u.select_atoms(f"resname {pip2_resnames} and name {' '.join(str(item) for item in unique_pip2_atom_names)}")
        for atom in pip2_atoms:
            charge_map[(pip2_resnames,atom.name)] = atom.charge

        return charge_map

    def _get_previous_start_frame(self, events, target_ion_id):
        # Sort by start_frame to ensure logical order
        events = sorted(events, key=lambda x: x["start_frame"])

        for i, event in enumerate(events):
            if event["ion_id"] == target_ion_id:
                if i == 0:
                    return self.start_frame # No previous, return own
                    # return 0 # No previous, return own
                return events[i - 1]["start_frame"]
        
        return None  # target ion not found

    def _ion_id_exists_in_ch1(self, target_ion_id):
        """
        Checks if a given ion_id exists in a list of event dictionaries.
        """
        return any(event["ion_id"] == target_ion_id for event in self.ch1_permeation_events)

    def run_permeation_analysis(self):
        """
        Runs permeation analysis for all events in self.ch2_permation_residues.
        Returns:
        - force_results (list)
        - radial_distances_results (list)
        - close_residues_results (list)
        """
        ################## POSITION EXTRACTION ##################
        
        # Build positions and charge map once
        positions = self._build_all_positions()

        total_sf_residues = []
        for res in self.sf_residues:
            for i in range(7):
                total_sf_residues.append(res+i)


        residue_list=self.glu_residues + self.asn_residues + total_sf_residues

        residues_names = []
        for resid in residue_list:
            res = self.u.select_atoms(f"resid {resid}").residues[0]
            residues_names.append(res.resname)
            
        res_atoms = self.u.select_atoms(f"resname {' '.join(residues_names)}")
        # res_atom_names_by_resid = defaultdict(list)
        # for atom in res_atoms:
        #     res_atom_names_by_resid[atom.resid].append(atom.name)

        # Flatten unique atom names
        unique_res_atom_names = sorted(set(atom.name for atom in res_atoms))
        print("Unique atom names in residues:", unique_res_atom_names)
        
        residue_positions = self._build_residue_positions(
                                residue_list=residue_list,
                                atom_names=unique_res_atom_names
                            )
        
        possible_names = {"PIP2", "POPI", "POP2", "LPI2", "PIP"}  # Add more if needed
        actual_pip2_names = {res.resname for res in self.u.residues if res.resname in possible_names}

        print("Detected PIP2 residue names:", actual_pip2_names)

        #Collect all resids for PIP2 residues
        pip2_resids = sorted({int(res.resid) for res in self.u.residues if res.resname in actual_pip2_names})
        print("PIP2 residue IDs:", pip2_resids)
        #Find all atom names used in these PIP2 residues
        pip2_atoms = self.u.select_atoms(f"resname {' '.join(actual_pip2_names)}")
        # pip_atom_names_by_resid = defaultdict(list)
        # for atom in pip2_atoms:
        #     pip_atom_names_by_resid[atom.resid].append(atom.name)

        # Flatten unique atom names
        unique_pip2_atom_names = sorted(set(atom.name for atom in pip2_atoms))
        print("Unique atom names in PIP2 residues:", unique_pip2_atom_names)

        pip2_positions = self._build_residue_positions(
                                residue_list=pip2_resids,
                                atom_names=unique_pip2_atom_names,
                            )
        
        ################### CHARGE MAP BUILDING ###################
        # Get unique residue names for those resids
        charge_map = self._build_charge_map(list(actual_pip2_names)[0], unique_pip2_atom_names, residues_names, unique_res_atom_names)
        # print(charge_map)

        # Optional total force calculation via OpenMM
        total_force_data = None
        if self.calculate_total_force and self.prmtop_file and self.nc_file:
            print("Calculating total forces with OpenMM...")
            total_force_data, atom_index_map = calculate_ionic_forces_all_frames(
                self.prmtop_file, self.nc_file
            )

        for event in tqdm(self.ch2_permation_residues, desc="Permeation_profile_creator: Analyzing Permeation Events in Channel 2"):
            ion_id_to_find = event["permeated"]
            print(f"Analyzing ion {ion_id_to_find} in channel 2...")
            ch1_start_frame= self._get_previous_start_frame(self.ch2_permeation_events, ion_id_to_find)
            # ion already in the channel - More advanced
            if not self._ion_id_exists_in_ch1(ion_id_to_find):
                continue
            # ###!!!!!!!!!##!##!#!#!#!#!#!#!#!#!#!#!#!#
            # if ch1_start_frame == 0:
            #     continue



            if not (self.start_frame <= event["frame"] < self.end_frame):
                continue

            event_force_results = {
                "start_frame": event["start_frame"],
                "frame": event["frame"],
                "permeated_ion": event["permeated"],
                "analysis": {}
            }

            event_radial_distances_results = {
                "start_frame": event["start_frame"],
                "frame": event["frame"],
                "permeated_ion": event["permeated"],
                "analysis": {}
            }

            event_close_residues_results = {
                "start_frame": event["start_frame"],
                "frame": event["frame"],
                "permeated_ion": event["permeated"],
                "analysis": {}
            }

            event_force_intervals_results = {
                "start_frame": event["start_frame"],
                "frame": event["frame"],
                "permeated_ion": event["permeated"],
                "analysis": {}
            }

            # frames_to_check = list(range(event["start_frame"], event["frame"] + 1))
            frames_to_check = list(range(ch1_start_frame, event["frame"] + 1))

            for frame in tqdm(frames_to_check, desc=f"Analyzing frames for ion {ion_id_to_find}"):
                # # Skip if ion is near SF in this frame
                # residue_track = self.min_results_per_frame.get(event["permeated"], [])
                # is_sf = any(entry["frame"] == frame and entry["residue"] == "SF" for entry in residue_track)
                # if is_sf:
                #     continue

                if frame == self.end_frame:
                    continue


                # Force analysis
                frame_result = analyze_forces(
                        u=self.u,
                        positions=positions,
                        residue_positions = residue_positions,
                        pip2_positions=pip2_positions,
                        pip2_resids=pip2_resids,
                        unique_pip2_atom_names=unique_pip2_atom_names,
                        actual_pip2_names=list(actual_pip2_names)[0],
                        permeating_ion_id=event["permeated"],
                        frame=frame,
                        other_ions=positions.get(frame, {}).keys(),
                        charge_map=charge_map,
                        closest_residues_by_ion=self.min_results_per_frame,
                        glu_residues=self.glu_residues,        # <-- new
                        asn_residues=self.asn_residues,        # <-- new
                        total_sf_residues=total_sf_residues,          # <-- new
                        cutoff=self.cutoff,
                        calculate_total_force=self.calculate_total_force,
                        total_force_data=total_force_data
                    )
                event_force_results["analysis"][frame] = frame_result

                # # Radial distance analysis
                # radial_distances_result = analyze_radial_distances(
                #     positions=positions,
                #     permeating_ion_id=event["permeated"],
                #     frame=frame,
                #     channel=self.ch2
                # )
                # event_radial_distances_results["analysis"][frame] = radial_distances_result

                # Closest residues analysis
                close_residues_result = analyze_close_residues(
                    positions=positions,
                    permeating_ion_id=event["permeated"],
                    frame=frame,
                    other_ions=positions.get(frame, {}).keys(),
                    close_contacts_dict=self.close_contacts_dict,
                    cutoff=self.cutoff
                )
                event_close_residues_results["analysis"][frame] = close_residues_result

                if frame == event["frame"] and self.calculate_intervals:
                    # Force intervals analysis
                    force_intervals_result = analyze_force_intervals(
                        u=self.u,
                        positions=positions,
                        residue_positions=residue_positions,
                        permeating_ion_id=event["permeated"],
                        frame=frame,
                        charge_map=charge_map,
                        glu_residues=self.glu_residues,  # <-- new
                        asn_residues=self.asn_residues,  # <-- new
                        total_sf_residues=total_sf_residues,  # <-- new
                        cutoff=self.cutoff,
                        n_steps=20,  # Number of steps for interpolation
                        k=332.0  # Coulomb's constant in kJ/(mol*nm*e^2)
                    )
                    event_force_intervals_results["analysis"][frame] = force_intervals_result
                    del force_intervals_result

                # del frame_result, radial_distances_result, close_residues_result
                del frame_result, close_residues_result
                gc.collect()                

            # Append results per event
            self.force_results.append(event_force_results)

            self.radial_distances_results.append(event_radial_distances_results)
            self.close_residues_results.append(event_close_residues_results)
            if self.calculate_intervals:
                self.force_intervals_results.append(event_force_intervals_results)

            del event_force_results, event_radial_distances_results, event_close_residues_results
            gc.collect()


        return self.force_results, self.radial_distances_results, self.close_residues_results, self.force_intervals_results
    
    def closest_residues_comb_before_permeation(self, n=-1, use_pdb_format=False, sort_residues=True, channel_type="G2"):
        """
        Loop through all permeation events and apply get_last_nth_frame_close_residues.
        Saves both JSON and CSV outputs.
        """
        output_dir = os.path.join(self.output_base_dir, "closest_residues_comb")
        os.makedirs(output_dir, exist_ok=True)

        summary = {}

        for i, event in enumerate(self.close_residues_results):
            try:
                # Get closest residues for this permeation event
                frame_data = get_last_nth_frame_close_residues(
                    event,
                    n=n,
                    use_pdb_format=use_pdb_format,
                    sort_residues=sort_residues,
                    channel_type=channel_type
                )

                frame_key = list(frame_data.keys())[0]
                ion_dict = frame_data[frame_key]

                # ✅ Filter out 'SF' residues here (inside try)
                filtered_ion_dict = {ion_id: resid for ion_id, resid in ion_dict.items() if resid != "SF"}
                if filtered_ion_dict:
                    summary[frame_key] = filtered_ion_dict
                else:
                    print(f"⚠️ Skipping frame {frame_key} — only 'SF' residues found.")

            except Exception as e:
                print(f"Skipping event {i} due to error: {e}")

        # Save JSON summary
        with open(os.path.join(output_dir, f"closest_residues_n_{n}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Plot residue frequency bar chart
        plot_residue_counts(summary, output_dir, filename=f"residue_counts_{n}.png", exclude=(), duplicates=False)

        # (Optional) Residue combination analysis
        # analyze_residue_combinations(summary, output_dir, top_n_plot=20)



    def analyze_radial_significance(self):
        """
        Requires self.radial_distances_results to be set.
        """
        if self.radial_distances_results is None:
            raise ValueError("radial_distances_results not set in FrameAnalyzer.")

        self.output_dir = os.path.join(self.output_base_dir, "radial_analysis")
        os.makedirs(self.output_dir, exist_ok=True)

        results = []
        permeation_radials = []
        avg_nonpermeation_radials = []

        for event in tqdm(self.radial_distances_results, desc="Analyzing Radial Significance"):
            ion_id = str(event["permeated_ion"])
            permeation_frame = int(event["frame"])
            analysis = {int(k): v for k, v in event["analysis"].items()}

            frames = sorted(analysis.keys())
            radial_values = [analysis[f] for f in frames]

            if permeation_frame not in analysis:
                continue

            perm_radial = analysis[permeation_frame]
            non_perm_radials = [v for f, v in analysis.items() if f != permeation_frame]
            if not non_perm_radials:
                continue

            permeation_radials.append(perm_radial)
            avg_nonpermeation_radials.append(np.mean(non_perm_radials))

            count_extreme = sum(1 for v in non_perm_radials if v >= perm_radial)
            empirical_p = (count_extreme + 1) / (len(non_perm_radials) + 1)

            results.append({
                "ion_id": ion_id,
                "start_frame": event["start_frame"],
                "permeation_frame": permeation_frame,
                "permeation_radial": round(perm_radial, 3),
                "avg_nonpermeation_radial": round(np.mean(non_perm_radials), 3),
                "empirical_p": round(empirical_p, 3),
                "total_frames": len(frames),
            })

            self._plot_radial_traces(ion_id, frames, radial_values, perm_radial, permeation_frame, self.output_dir)

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, "radial_significance_results.csv"), index=False)

        self._plot_global_radial_distribution(permeation_radials, avg_nonpermeation_radials, self.output_dir)

        stat, p_value = wilcoxon(permeation_radials, avg_nonpermeation_radials)
        with open(os.path.join(self.output_dir, "wilcoxon_test_results.txt"), "w") as f:
            f.write(f"Wilcoxon signed-rank test result:\nStatistic = {stat}\nP-value = {p_value}\n")


    def analyze_cosine_significance(self, force_dir):
        """
        Requires self.force_results to be set.
        """
        if self.force_results is None:
            raise ValueError("force_results not set in FrameAnalyzer.")

        output_dir = os.path.join(force_dir, "cosine_analysis")
        os.makedirs(output_dir, exist_ok=True)

        results = []
        permeation_cosines = []
        avg_nonpermeation_cosines = []

        for event in tqdm(self.force_results, desc="Analyzing Cosine Significance"):
            ion_id = str(event["permeated_ion"])
            permeation_frame = event["frame"]
            analysis = event["analysis"]

            cosines = []
            frames = []
            permeation_cosine = None

            for frame_data in sorted(analysis.values(), key=lambda x: x["frame"]):
                frame = frame_data["frame"]
                cosine = frame_data["cosine_ionic_motion"]
                frames.append(frame)
                cosines.append(cosine)
                if frame == permeation_frame:
                    permeation_cosine = cosine

            if permeation_cosine is None or len(cosines) <= 1:
                continue

            permeation_cosines.append(permeation_cosine)
            avg_nonpermeation_cosines.append(np.mean([c for f, c in zip(frames, cosines) if f != permeation_frame]))

            count_extreme = sum(1 for c in cosines if c >= permeation_cosine)
            empirical_p = (count_extreme + 1) / (len(cosines) + 1)

            results.append({
                "ion_id": ion_id,
                "start_frame": event["start_frame"],
                "permeation_frame": permeation_frame,
                "permeation_cosine": round(permeation_cosine,3),
                "avg_nonpermeation_cosine": round(np.mean([c for f, c in zip(frames, cosines) if f != permeation_frame]),3),
                "empirical_p": round(empirical_p,3),
                "total_cosines": len(cosines),
            })

            self._plot_cosine_traces(ion_id, frames, cosines, permeation_cosine, permeation_frame, output_dir)

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, "cosine_significance_results.csv"), index=False)

        self._plot_global_cosine_distribution(permeation_cosines, avg_nonpermeation_cosines, output_dir)

        stat, p_value = wilcoxon(permeation_cosines, avg_nonpermeation_cosines)
        with open(os.path.join(output_dir, "wilcoxon_test_results.txt"), "w") as f:
            f.write(f"Wilcoxon signed-rank test result:\nStatistic = {stat}\nP-value = {p_value}\n")

        simplified_df = pd.DataFrame([
            {
                "ion_id": row["ion_id"],
                "avg_nonpermeation_cosine": round(row["avg_nonpermeation_cosine"],2),
                "permeation_cosine": round(row["permeation_cosine"],2)
            } for row in results
        ])
        simplified_df.to_csv(os.path.join(output_dir, "cosine_summary_table.csv"), index=False)

    # ---- Helper plotting methods ----
    def _plot_radial_traces(self, ion_id, frames, radials, perm_radial, perm_frame, output_dir):
        os.makedirs(f"{output_dir}/radial_trace_ion", exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(frames, radials, label="Radial Distance", color='blue')
        plt.axvline(perm_frame, color='red', linestyle='--', label='Permeation Frame')
        plt.axhline(perm_radial, color='green', linestyle='--', label='Permeation Radial')
        plt.title(f"Ion {ion_id} – Radial Distance Trace")
        plt.xlabel("Frame")
        plt.ylabel("Radial Distance (Å)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/radial_trace_ion/{ion_id}.png")
        plt.close()

        os.makedirs(f"{output_dir}/radial_histogram", exist_ok=True)
        plt.figure(figsize=(6, 4))
        non_perm = [r for r in radials if r != perm_radial]
        plt.hist(non_perm, bins=50, alpha=0.7, color='gray', edgecolor='black')
        plt.axvline(perm_radial, color='red', linestyle='--', linewidth=2, label='Permeation Radial')
        plt.title(f"Ion {ion_id} – Radial Histogram – Permeation: {perm_radial:.2f}")
        plt.xlabel("Radial Distance (Å)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/radial_histogram/{ion_id}.png")
        plt.close()

    def _plot_global_radial_distribution(self, permeation_radials, avg_nonperm_radials, output_dir):
        plt.figure()
        plt.hist(avg_nonperm_radials, bins=20, alpha=0.7, label='Avg Non-Permeation Radials')
        plt.axvline(np.mean(permeation_radials), color='red', linestyle='--', label='Mean Permeation Radial')
        plt.xlabel("Radial Distance (Å)")
        plt.ylabel("Frequency")
        plt.title("Radial Distance Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "radial_distribution_histogram.png"))
        plt.close()

    def _plot_cosine_traces(self, ion_id, frames, cosines, perm_cosine, perm_frame, output_dir):
        os.makedirs(f"{output_dir}/cosine_trace_ion", exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(frames, cosines, label="Cosine Ionic–Motion", color='blue')
        plt.axvline(perm_frame, color='red', linestyle='--', label='Permeation Frame')
        plt.axhline(perm_cosine, color='green', linestyle='--', label='Permeation Cosine')
        plt.title(f"Ion {ion_id} – Cosine Trajectory")
        plt.xlabel("Frame")
        plt.ylabel("Cosine(Ionic Force, Motion)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cosine_trace_ion/{ion_id}.png")
        plt.close()

        os.makedirs(f"{output_dir}/cosine_histogram", exist_ok=True)
        plt.figure(figsize=(6, 4))
        non_perm = [c for c in cosines if c != perm_cosine]
        plt.hist(non_perm, bins=50, alpha=0.7, color='gray', edgecolor='black')
        plt.axvline(perm_cosine, color='red', linestyle='--', linewidth=2, label='Permeation Cosine')
        plt.title(f"Ion {ion_id} – Cosine Histogram – Permeation: {perm_cosine:.2f}")
        plt.xlabel("Cosine Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cosine_histogram/{ion_id}.png")
        plt.close()


    def _plot_global_cosine_distribution(self, permeation_cosines, avg_nonperm_cosines, output_dir):
        plt.figure()
        plt.hist(avg_nonperm_cosines, bins=20, alpha=0.7, label='Avg Non-Permeation Cosines')
        plt.axvline(np.mean(permeation_cosines), color='red', linestyle='--', linewidth=2, label='Mean Permeation Cosine')
        plt.xlabel("Cosine Value")
        plt.ylabel("Frequency")
        plt.title("Cosine Distribution: Non-Permeation vs Permeation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cosine_distribution_histogram.png"))
        plt.close()
