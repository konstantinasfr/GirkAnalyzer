import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import copy

class IonPermeationAnalysis:
    def __init__(self, universe, ion_selection, start_frame, end_frame, channel1, channel2, channel3, channel4,
                 hbc_residues, hbc_diagonal_pairs, sf_low_res_residues, sf_low_res_diagonal_pairs, results_dir,count_ions=True):
        
        self.u = universe
        self.ion_selection = ion_selection
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3
        self.channel4 = channel4
        self.hbc_residues = hbc_residues
        self.hbc_diagonal_pairs = hbc_diagonal_pairs
        self.sf_low_res_residues = sf_low_res_residues
        self.sf_low_res_diagonal_pairs = sf_low_res_diagonal_pairs
        self.count_ions = count_ions
        
        # Track regions
        self.ion_region_tracking = {}
        self.ions_that_entered_region1 = set()
        
        # Store stages_reached and all_events for each ion
        self.ion_stages_reached = {}
        self.ion_all_events = {}
        
        # Permeation events derived from region tracking
        self.permeation_events = []
        
        self.results_dir = results_dir
        self.ions = self.u.select_atoms(self.ion_selection)

    def _determine_ion_region(self, ion_pos):
        """
        Determines which region (1-5) the ion is currently in, or 0 if outside all regions.
        Checks from region 1 to 5 in order.
        """
        channels = [self.channel1, self.channel2, self.channel3, self.channel4]
        
        for i, channel in enumerate(channels, start=1):
            if channel.is_within_cylinder(ion_pos):
                # ion_vec = ion_pos - channel.channel_center
                # ion_z = np.dot(ion_vec, channel.channel_axis)
                
                # upper_z = np.dot(channel.upper_center - channel.channel_center, channel.channel_axis)
                # lower_z = np.dot(channel.lower_center - channel.channel_center, channel.channel_axis)
                
                # # Check if ion is between upper and lower bounds
                # if ion_z <= upper_z and ion_z >= lower_z:
                    return i
        
        return 0  # Outside all regions

    def compute_constriction_point_diameters(self, frame, atoms, diagonal_pairs):
        """Computes the mean distance between pairs of residues."""
        distances = []
        for res1, res2 in diagonal_pairs:
            pos1 = atoms[res1].positions
            pos2 = atoms[res2].positions
            pairwise_dists = np.linalg.norm(pos1[:, None, :] - pos2[None, :, :], axis=2)
            dist = np.min(pairwise_dists)
            distances.append(dist)

        mean_diameter = np.mean(distances)
        consiction_point_diameters_dict = {
            "frame": int(frame),
            "mean": float(mean_diameter),
            "A_C": float(distances[0]),
            "B_D": float(distances[1])
        }
        return consiction_point_diameters_dict

    def ion_counter(self,ts):
        # Update channel geometries
        self.channel1.compute_geometry(1)
        self.channel2.compute_geometry(2)
        self.channel3.compute_geometry(3)
        self.channel4.compute_geometry(4)
        # self.channel5.compute_geometry(5)

        # SIMPLIFIED: Just track regions for each ion
        for ion in self.ions:

            ion_id = ion.resid
            pos = ion.position
            
            # Determine current region
            current_region = self._determine_ion_region(pos)
            # if ion_id == 1307:
            #     print(ts.frame, current_region)
            # Track ions that enter region 1
            if current_region == 1 or (ts.frame == self.start_frame and current_region == 2):
                if ion_id not in self.ions_that_entered_region1:
                    self.ions_that_entered_region1.add(ion_id)
                    self.ion_region_tracking[ion_id] = {ts.frame: current_region}
            
            # If this ion has ever been in region 1, track its region
            if ion_id in self.ions_that_entered_region1:
                self.ion_region_tracking[ion_id][ts.frame] = current_region
                        
    def run_analysis(self):
        print("Starting analysis...")

        self.hbc_diameters = []
        self.sf_low_res_diameters = []

        for ts in tqdm(self.u.trajectory[self.start_frame:self.end_frame+1],
                    total=(self.end_frame - self.start_frame + 1),
                    desc="Processing Frames", unit="frame"):
            
            # Compute diameters
            hbc_atoms = {resid: self.u.select_atoms(f"resid {resid}") for resid in self.hbc_residues}
            self.hbc_diameters.append(self.compute_constriction_point_diameters(ts.frame, hbc_atoms, self.hbc_diagonal_pairs))

            sf_low_res_atoms = {resid: self.u.select_atoms(f"resid {resid}") for resid in self.sf_low_res_residues}
            self.sf_low_res_diameters.append(self.compute_constriction_point_diameters(ts.frame, sf_low_res_atoms, self.sf_low_res_diagonal_pairs))

            if  self.count_ions:
                self.ion_counter(ts)

    def extract_permeation_events(self, stages_reached):
        ### assumption: ions always pass through region 4 before they leave
        all_events = []

        stage_seq = []
        prev_state = 0
        stage_timeline = {
            1:{"start": None, "end": None},
            2:{"start": None, "end": None},
            3:{"start": None, "end": None},
            4:{"start": None, "end": None},
        }
        # for j in [2,3]:
        #     if stage_timeline[j]["start"] is not None:
        #         stage_timeline[j]["start"] = None
        #         stage_timeline[j]["end"] = None
        def get_numbers_greater_than(lst, x):
            return [n for n in lst if n >= x]

        for e in stages_reached:
            s = e["state"]
            # print("all events")
            # print(all_events)
            if s == 0:
                if prev_state in [1,2,3]:
                    for i in range(1,5):
                        stage_timeline[i]["start"] = None
                        stage_timeline[i]["end"] = None
                    stage_seq = []
                else:
                    continue
            else:
                if s==1 and prev_state==0 and (1 in stage_seq and 4 in stage_seq):
                    
                    # print(stage_timeline.copy())
                    all_events.append(copy.deepcopy(stage_timeline))
                    for i in range(1,5):
                        stage_timeline[i]["start"] = None
                        stage_timeline[i]["end"] = None
                    stage_seq = []
                    # print("hello")
                    # print(all_events)
                    # print("hello")

                start, end = e["start"], e["end"]

                later_states = get_numbers_greater_than(stage_seq, s)

                for ls in later_states:
                    stage_timeline[ls]["start"] = None
                    stage_timeline[ls]["end"] = None
                    stage_seq.remove(ls)

                stage_timeline[s]["start"] = start
                stage_timeline[s]["end"] = end
                stage_seq.append(s)

        # print(stage_seq)
        if (1 in stage_seq and 4 in stage_seq and (2 in stage_seq or 3 in stage_seq)) and s==0:
            # print("lala")
            all_events.append(copy.deepcopy(stage_timeline))

        # print(all_events)
        return all_events

    def detect_permeation_events(self):
        """
        Analyzes ion_region_tracking to find all permeation events.
        A permeation is defined as: 1 -> 2 -> 3 -> 4 -> 5 -> 0 (exit)
        Handles multiple permeations by the same ion.
        """
        print("\nDetecting permeation events from region tracking...")

        for ion_id, region_seq in self.ion_region_tracking.items():
            stages_reached = []
            
            prev_frame = min(region_seq.keys())
            prev_region = 0
            start_frame = prev_frame

            for frame, region in region_seq.items():
                
                if prev_region != region:
                    if region == 1 and frame == self.start_frame:
                        stages_reached.append({"state": 0, "start": self.start_frame, "end": self.start_frame})
                    elif region == 2 and frame == self.start_frame:
                        stages_reached.append({"state": 0, "start": self.start_frame, "end": self.start_frame})
                        stages_reached.append({"state": 1, "start": self.start_frame, "end": self.start_frame})
                    else:
                        stages_reached.append({"state": prev_region, "start": start_frame, "end": frame-1})
                    start_frame = frame

                prev_region = region
                    
            stages_reached.append({"state": prev_region, "start": start_frame, "end": frame})

            # Store stages_reached for this ion
            self.ion_stages_reached[ion_id] = stages_reached

            print(ion_id)
            # Extract permeation events
            all_events = self.extract_permeation_events(stages_reached)
            
            # Store all_events for this ion
            self.ion_all_events[ion_id] = all_events

            # Add to flattened permeation_events list
            for event in all_events:
                
                self.permeation_events.append({
                    "ion_id": int(ion_id),
                    "start_1": int(event[1]["start"]),
                    "end_1": int(event[1]["end"]) if event[1]["end"] is not None else int(event[2]["start"]),
                    "start_2": int(event[2]["start"]) if event[2]["start"] is not None else int(event[1]["end"]),
                    "end_2": int(event[2]["end"]) if event[2]["end"] is not None else int(event[3]["start"]),
                    "start_3": int(event[3]["start"]) if event[3]["start"] is not None else int(event[2]["end"]),
                    "end_3": int(event[3]["end"]) if event[3]["end"] is not None else int(event[4]["start"]),
                    "start_4": int(event[4]["start"]) if event[4]["start"] is not None else int(event[3]["end"]),
                    "end_4": int(event[4]["end"]),
                    "total_time": int(event[4]["end"]) - int(event[1]["start"]),
                })
        
        # Sort by start frame
        self.permeation_events.sort(key=lambda x: x['start_1'])
        
        print(f"Found {len(self.permeation_events)} total permeation events")

    def save_ion_region_tracking(self):
        """Save the ion region tracking to a JSON file"""
        output_file = self.results_dir / "ion_region_tracking.json"
        
        # Convert to serializable format
        tracking_data = {
            str(ion_id): {str(frame): region for frame, region in regions.items()}
            for ion_id, regions in self.ion_region_tracking.items()
        }
        
        with open(output_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        print(f"Ion region tracking saved to: {output_file}")
        print(f"Total ions tracked: {len(self.ion_region_tracking)}")

    def save_stages_reached(self):
        """Save stages_reached for all ions"""
        output_file = self.results_dir / "ion_stages_reached.json"
        
        # Convert to serializable format
        stages_data = {
            str(ion_id): stages
            for ion_id, stages in self.ion_stages_reached.items()
        }
        
        with open(output_file, 'w') as f:
            json.dump(stages_data, f, indent=2)
        
        print(f"Ion stages reached saved to: {output_file}")

    def save_all_events(self):
        output_file = self.results_dir / "ion_all_events.json"

        events_data = {}
        for ion_id, events in self.ion_all_events.items():
            events_data[str(ion_id)] = []
            for ev in events:
                events_data[str(ion_id)].append({
                    "start_1": ev[1]["start"],
                    "end_1": ev[1]["end"],
                    "start_2": ev[2]["start"],
                    "end_2": ev[2]["end"],
                    "start_3": ev[3]["start"],
                    "end_3": ev[3]["end"],
                    "start_4": ev[4]["start"],
                    "end_4": ev[4]["end"],
                })

        with open(output_file, "w") as f:
            json.dump(events_data, f, indent=2)

        print(f"Ion all events saved to: {output_file}")

    def save_permeation_table(self):
        output_file = self.results_dir / "permeation_table.json"
        with open(output_file, "w") as f:
            json.dump(self.permeation_table, f, indent=2)
        print(f"Permeation summary table saved to: {output_file}")

    def print_results(self):
        """Print summary of permeation events"""
        
        if self.count_ions:
            # Detect permeations from region tracking
            self.detect_permeation_events()
            
            print(f"\n{'='*90}")
            print("PERMEATION EVENTS SUMMARY (Region Dwell Times)")
            print(f"{'='*90}")
            print(f"{'Ion':<8} {'R1_start':<10} {'R1_end':<10} {'R2_start':<10} {'R2_end':<10} "
                f"{'R3_start':<10} {'R3_end':<10} {'R4_start':<10} {'R4_end':<10} {'Total':<10}")
            print(f"{'-'*90}")

            for event in self.permeation_events:
                print(f"{event['ion_id']:<8} "
                    f"{event['start_1']:<10} {event['end_1']:<10} "
                    f"{event['start_2']:<10} {event['end_2']:<10} "
                    f"{event['start_3']:<10} {event['end_3']:<10} "
                    f"{event['start_4']:<10} {event['end_4']:<10} "
                    f"{event['total_time']:<10}")

            print(f"{'-'*90}")
            print(f"Total permeation events: {len(self.permeation_events)}")


            
            # Count permeations per ion
            from collections import Counter
            ion_counts = Counter([event['ion_id'] for event in self.permeation_events])
            multiple_permeations = {ion: count for ion, count in ion_counts.items() if count > 1}
            
            if multiple_permeations:
                print(f"\n{'='*50}")
                print(f"IONS WITH MULTIPLE PERMEATIONS:")
                print(f"{'='*50}")
                for ion_id, count in sorted(multiple_permeations.items()):
                    print(f"  Ion {ion_id}: {count} permeations")
                    # Show each event for this ion
                    ion_events = [e for e in self.permeation_events if e['ion_id'] == ion_id]
                    for i, evt in enumerate(ion_events, 1):
                        print(f"    Event {i}: frames {evt['start_1']}-{evt['end_4']}")
            
            # Save all results
            print(f"\n{'='*50}")
            print(f"SAVING RESULTS...")
            print(f"{'='*50}")

                    # Store table-style dict for saving
            self.permeation_table = []

            for event in self.permeation_events:
                self.permeation_table.append({
                    "ion_id": event["ion_id"],
                    "R1_start": event["start_1"],
                    "R1_end": event["end_1"],
                    "R2_start": event["start_2"],
                    "R2_end": event["end_2"],
                    "R3_start": event["start_3"],
                    "R3_end": event["end_3"],
                    "R4_start": event["start_4"],
                    "R4_end": event["end_4"],
                    "total_time": event["total_time"]
                })

            self.save_ion_region_tracking()
            self.save_stages_reached()
            self.save_permeation_table()

        # Plot diameters
        self.plot_residue_distances(self.hbc_diameters, self.results_dir, "hbc_pairs_distances.png", 
                                    "HBC Residue Pair Distances Over Time", "end_3")
        self.plot_residue_distances(self.sf_low_res_diameters, self.results_dir, "sf_pairs_distances.png", 
                                    "SF Residue Pair Distances Over Time", "end_1")

    def moving_average(self, values, window=50):
        """Simple moving average for smoothing."""
        return np.convolve(values, np.ones(window)/window, mode='same')

    def plot_residue_distances(self, data, output_dir="plots", filename_base="residue_distances",
                            title_base="Residue Pair Distances Over Time", frame_lines="R3_end"):
        """Plot residue distances over time with permeation event markers"""
        frames = [entry["frame"] for entry in data]
        mean_values = [entry["mean"] for entry in data]
        ac_values = [entry["A_C"] for entry in data]
        bd_values = [entry["B_D"] for entry in data]

        # Get frames to mark based on frame_lines parameter
        mark_frames = []
        if self.permeation_events:
            mark_frames = [entry[frame_lines] for entry in self.permeation_events]

        os.makedirs(output_dir, exist_ok=True)

        for smooth in [False, True]:
            suffix = "_smoothed" if smooth else "_raw"
            title = title_base + (" (Smoothed)" if smooth else "")
            filename = f"{filename_base}{suffix}.png"
            filepath = os.path.join(output_dir, filename)

            if smooth:
                mean_plot = self.moving_average(mean_values)
                ac_plot = self.moving_average(ac_values)
                bd_plot = self.moving_average(bd_values)
            else:
                mean_plot = mean_values
                ac_plot = ac_values
                bd_plot = bd_values

            plt.figure(figsize=(10, 6))
            plt.plot(frames, mean_plot, label="Mean", linewidth=2)
            plt.plot(frames, ac_plot, label="A_C", linestyle="--")
            plt.plot(frames, bd_plot, label="B_D", linestyle="--")

            for i, x in enumerate(mark_frames):
                label = "Permeation event" if i == 0 else None
                plt.axvline(x=x, linestyle="--", color="black", linewidth=0.8, alpha=0.6, label=label)

            plt.xlabel("Frame")
            plt.ylabel("Distance (Å)")
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300)
            plt.close()

            print(f"Plot saved to: {filepath}")