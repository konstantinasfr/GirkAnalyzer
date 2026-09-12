import MDAnalysis as mda
import numpy as np
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from analysis.converter import convert_to_pdb_numbering


class CavityOccupancyAnalysis:
    """
    Analyzes which residues/subunits have ions close to them at a given
    permeation stage frame, and counts total ions in the relevant channel
    region.

    stage="end_2" (default, unchanged behaviour):
        - frame key   : event['end_2']
        - channel     : channel2 (GLU/ASN cavity)
        - residues    : asn_residues + glu_residues
        - region code : 2 (used when reading ion_region_tracking.json)

    stage="end_3" (new):
        - frame key   : event['end_3']
        - channel     : channel3 (ASN -> HBC region)
        - residues    : hbc_residues
        - region code : 3

    Includes untracked ions loaded from untracked_cavity_ions/untracked_ions.csv.
    Also saves a per-frame ion count file (no trajectory scan needed).

    All outputs (JSON, CSV, plots) are written under a "cavity_occupancy"
    subfolder of the results_dir you pass in (e.g. results_dir/cavity_occupancy/),
    with further subfolders per stage: sf/, glu_asn/, asn_hbc/, glu_hbc/ (combined).
    You only need to pass the run's base results_dir — the subfolder is created
    automatically.
    """

    def __init__(self, universe, channel2, permeation_events, asn_residues, glu_residues,
                 subunit_groups, channel_type="G12", threshold=3.0, results_dir=".",
                 start_frame=0, end_frame=None, frame_duration=0.075,
                 stage="end_2", channel3=None, hbc_residues=None,
                 channel1=None, sf_residues=None, force_rerun=False):
        self.u              = universe
        self.permeation_events = permeation_events
        self.asn_residues   = asn_residues
        self.glu_residues   = glu_residues
        self.subunit_groups = subunit_groups
        self.channel_type   = channel_type
        self.threshold      = threshold
        # All cavity-occupancy outputs (JSON/CSV/plots) live under a
        # dedicated "cavity_occupancy" subfolder of the run's results_dir,
        # so callers just pass the base RUN*/ results_dir and don't need to
        # build the subfolder path themselves.
        self.results_dir    = Path(results_dir) / "cavity_occupancy"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results        = []
        self.start_frame    = start_frame
        self.end_frame      = end_frame if end_frame is not None else len(universe.trajectory) - 1
        self.frame_duration = frame_duration
        self.force_rerun    = force_rerun

        # ── Stage configuration ──
        # Controls which frame key, channel, residues and region code are used.
        # Defaults reproduce the original end_2 behaviour exactly.
        self.stage = stage
        if stage == "end_2":
            self.frame_key      = "end_2"
            self.channel        = channel2
            self.all_residues   = asn_residues + glu_residues
            self.region_code    = 2
            self.geometry_gate  = 2          # arg passed to channel.compute_geometry()
            self.stage_label    = "ion leaves GLU/ASN cavity"
            self.region_label   = "GLU/ASN cavity"
        elif stage == "end_3":
            if channel3 is None or hbc_residues is None:
                raise ValueError(
                    "stage='end_3' requires both channel3 and hbc_residues to be provided."
                )
            self.frame_key      = "end_3"
            self.channel        = channel3
            self.all_residues   = list(hbc_residues)
            self.region_code    = 3
            self.geometry_gate  = 3
            self.stage_label    = "ion leaves HBC gate"
            self.region_label   = "ASN\u2192HBC region"
        elif stage == "end_1":
            if channel1 is None or sf_residues is None:
                raise ValueError(
                    "stage='end_1' requires both channel1 and sf_residues to be provided."
                )
            self.frame_key      = "end_1"
            self.channel        = channel1
            self.all_residues   = list(sf_residues)
            self.region_code    = 1
            self.geometry_gate  = 1
            self.stage_label    = "ion leaves SF"
            self.region_label   = "selectivity filter (SF)"
        else:
            raise ValueError(f"Unknown stage '{stage}'. Use 'end_1', 'end_2' or 'end_3'.")

        # Subfolder for this stage's plots
        if stage == "end_2":
            self.stage_plot_dir = self.results_dir / "glu_asn"
        elif stage == "end_3":
            self.stage_plot_dir = self.results_dir / "asn_hbc"
        else:  # end_1
            self.stage_plot_dir = self.results_dir / "sf"
        self.stage_plot_dir.mkdir(parents=True, exist_ok=True)
        self.combined_plot_dir = self.results_dir / "glu_hbc"
        self.combined_plot_dir.mkdir(parents=True, exist_ok=True)

        # Keep channel2 attribute name available for backwards compatibility
        # with any external code that references analyzer.channel2 directly.
        self.channel2 = channel2

        # Build residue to subunit mapping
        self.residue_to_subunit = {}
        for subunit_name, residue_list in subunit_groups.items():
            for resid in residue_list:
                self.residue_to_subunit[resid] = subunit_name

        # Load untracked ions from CSV if available.
        # Search upward from results_dir because results_dir may be a
        # subdirectory (e.g. .../RUN5/cavity_occupancy) while the file
        # lives at .../RUN5/untracked_cavity_ions/untracked_ions.csv
        self.untracked_by_frame  = {}   # {frame(int): set(resid(int))}
        self.untracked_ions_path = None

        for candidate in [self.results_dir,
                          self.results_dir.parent,
                          self.results_dir.parent.parent]:
            p = candidate / "untracked_cavity_ions" / "untracked_ions.csv"
            if p.exists():
                self.untracked_ions_path = p
                break

        if self.untracked_ions_path:
            with open(self.untracked_ions_path, newline="") as f:
                for row in csv.DictReader(f):
                    frame = int(float(row["frame"]))
                    resid = int(float(row["ion_resid"]))
                    self.untracked_by_frame.setdefault(frame, set()).add(resid)
            n_obs = sum(len(v) for v in self.untracked_by_frame.values())
            print(f"  Loaded untracked ions: {n_obs} observations across "
                  f"{len(self.untracked_by_frame)} frames")
        else:
            print(f"  No untracked_ions.csv found — only cylinder-detected ions used")

    # ─────────────────────────────────────────────────────────────────────────
    # ORIGINAL METHODS (logic unchanged, now driven by stage config)
    # ─────────────────────────────────────────────────────────────────────────

    def find_ions_close_to_residues(self, frame, permeating_ion_id):
        """
        At a given frame, find which residues have ions within threshold distance.
        Excludes the permeating ion itself.
        Includes untracked ions alongside cylinder-detected ions.
        """
        self.u.trajectory[frame]
        self.channel.compute_geometry(self.geometry_gate)

        # Get all potassium ions EXCEPT the permeating one
        all_ions = self.u.select_atoms(f"resname K+ K and not resid {permeating_ion_id}")

        # Count ions in the relevant channel region via cylinder
        ions_in_cavity = []
        for ion in all_ions:
            if self.channel.is_within_cylinder(ion.position):
                ions_in_cavity.append(int(ion.resid))

        # Add untracked ions for this frame (exclude permeating ion)
        for resid in self.untracked_by_frame.get(frame, set()):
            if resid != int(permeating_ion_id) and resid not in ions_in_cavity:
                ions_in_cavity.append(resid)

        # For each residue of interest, find ions within threshold
        residue_ion_mapping = {}
        residue_pdb_mapping = {}

        # Build atom group for all cavity ions (for distance checks)
        if ions_in_cavity:
            sel_str = "resname K+ K and resid " + " ".join(str(r) for r in ions_in_cavity)
            cavity_ions_ag = self.u.select_atoms(sel_str)
        else:
            cavity_ions_ag = self.u.select_atoms("resname K+ K and resid -1")  # empty

        for resid in self.all_residues:
            res_atoms = self.u.select_atoms(f"resid {resid}")
            if len(res_atoms) == 0:
                continue

            pdb_num = convert_to_pdb_numbering(resid, self.channel_type)
            residue_pdb_mapping[resid] = pdb_num

            ions_close_to_this_residue = []
            for ion in cavity_ions_ag:
                ion_pos = ion.position
                distances = np.linalg.norm(res_atoms.positions - ion_pos, axis=1)
                min_dist = np.min(distances)
                if min_dist < self.threshold:
                    ions_close_to_this_residue.append(int(ion.resid))

            if ions_close_to_this_residue:
                residue_ion_mapping[pdb_num] = ions_close_to_this_residue

        residues_with_ions = list(residue_ion_mapping.keys())

        # Map to subunits
        subunit_ion_mapping = {}
        for resid, pdb_num in residue_pdb_mapping.items():
            if pdb_num in residue_ion_mapping:
                subunit = self.residue_to_subunit.get(resid)
                if subunit:
                    if subunit not in subunit_ion_mapping:
                        subunit_ion_mapping[subunit] = set()
                    subunit_ion_mapping[subunit].update(residue_ion_mapping[pdb_num])

        for subunit in subunit_ion_mapping:
            subunit_ion_mapping[subunit] = sorted(list(subunit_ion_mapping[subunit]))

        subunits_with_ions = list(subunit_ion_mapping.keys())

        return {
            'residues_with_ions':  residues_with_ions,
            'subunits_with_ions':  sorted(subunits_with_ions),
            'residue_ion_mapping': residue_ion_mapping,
            'subunit_ion_mapping': subunit_ion_mapping,
            'total_ions_in_cavity': len(ions_in_cavity),
            'ions_in_cavity':      ions_in_cavity
        }

    def _resolve_stage_frame(self, event):
        """
        Get the frame number for the configured stage from a permeation event.
        Falls back to 'R1_end' for stage='end_1', since some permeation_events
        store the SF-exit frame under that legacy key instead of 'end_1'.
        """
        if self.frame_key in event:
            return event[self.frame_key]
        if self.frame_key == "end_1" and "R1_end" in event:
            return event["R1_end"]
        raise KeyError(
            f"Event for ion {event.get('ion_id')} has no '{self.frame_key}' "
            f"(or 'R1_end' fallback) key."
        )

    def analyze_event(self, event):
        """Analyze cavity occupancy at the configured stage frame for one event."""
        ion_id     = event['ion_id']
        stage_frame = self._resolve_stage_frame(event)

        occupancy = self.find_ions_close_to_residues(stage_frame, ion_id)

        residue_combination = '_'.join([str(r) for r in sorted(occupancy['residues_with_ions'])])
        subunit_combination = '_'.join(sorted(occupancy['subunits_with_ions']))

        return {
            'ion_id':               int(ion_id),
            f'{self.frame_key}_frame': int(stage_frame),
            'residues_with_ions':   occupancy['residues_with_ions'],
            'residue_combination':  residue_combination,
            'subunits_with_ions':   occupancy['subunits_with_ions'],
            'subunit_combination':  subunit_combination,
            'residue_ion_mapping':  occupancy['residue_ion_mapping'],
            'subunit_ion_mapping':  occupancy['subunit_ion_mapping'],
            'total_ions_in_cavity': occupancy['total_ions_in_cavity'],
            'total_ions_including_permeating': occupancy['total_ions_in_cavity'] + 1,
            'ions_in_cavity':       occupancy['ions_in_cavity']
        }

    def run_analysis(self):
        """Run cavity occupancy analysis for all permeation events."""
        print("\n" + "="*80)
        print(f"CAVITY OCCUPANCY ANALYSIS (at {self.frame_key}, {self.stage_label})")
        print("="*80)
        print(f"Distance threshold: {self.threshold} Å")
        print(f"Analyzing {len(self.permeation_events)} permeation events...")

        for event in self.permeation_events:
            result = self.analyze_event(event)
            self.results.append(result)

            frame_field = f'{self.frame_key}_frame'
            print(f"\n{'='*60}")
            print(f"Ion {result['ion_id']} at {self.frame_key} (frame {result[frame_field]}):")
            print(f"{'='*60}")
            print(f"Total ions in {self.stage_label}: {result['total_ions_in_cavity']}")
            print(f"  Ion IDs: {result['ions_in_cavity']}")

            print(f"\nResidues with ions close (<{self.threshold}Å): {len(result['residues_with_ions'])}")
            for resid, ions in result['residue_ion_mapping'].items():
                print(f"  Residue {resid}: {len(ions)} ions - {ions}")
            print(f"Residue combination: {result['residue_combination']}")

            print(f"\nSubunits with ions close (<{self.threshold}Å): {len(result['subunits_with_ions'])}")
            for subunit, ions in result['subunit_ion_mapping'].items():
                print(f"  Subunit {subunit}: {len(ions)} ions - {ions}")
            print(f"Subunit combination: {result['subunit_combination']}")

    def save_results(self):
        """Save cavity occupancy results to JSON file."""
        output_file = self.results_dir / f"cavity_occupancy_at_{self.frame_key}.json"

        output_data = {
            'stage':                 self.frame_key,
            'threshold':             self.threshold,
            'all_residues_analyzed': self.all_residues,
            'subunit_groups':        self.subunit_groups,
            'results':               self.results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    def create_summary(self):
        """Create summary statistics of residue and subunit combinations."""
        if not self.results:
            print("No results to summarize.")
            return

        residue_combo_counts = Counter()
        for r in self.results:
            if r['residue_combination']:
                residue_combo_counts[r['residue_combination']] += 1

        subunit_combo_counts = Counter()
        for r in self.results:
            if r['subunit_combination']:
                subunit_combo_counts[r['subunit_combination']] += 1

        cavity_ion_counts = [r['total_ions_in_cavity'] for r in self.results]

        summary = {
            'stage':        self.frame_key,
            'total_events': len(self.results),
            'threshold':    self.threshold,
            'residue_combination_counts': dict(residue_combo_counts.most_common()),
            'subunit_combination_counts': dict(subunit_combo_counts.most_common()),
            'cavity_ion_stats': {
                'mean': float(np.mean(cavity_ion_counts)),
                'std':  float(np.std(cavity_ion_counts)),
                'min':  int(np.min(cavity_ion_counts)),
                'max':  int(np.max(cavity_ion_counts))
            }
        }

        output_file = self.results_dir / f"cavity_occupancy_summary_{self.frame_key}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*80)
        print(f"CAVITY OCCUPANCY SUMMARY ({self.frame_key})")
        print("="*80)
        print(f"Total events analyzed: {summary['total_events']}")

        print(f"\nMost common RESIDUE combinations:")
        for combo, count in list(residue_combo_counts.most_common(10)):
            print(f"  {combo}: {count} events")

        print(f"\nMost common SUBUNIT combinations:")
        for combo, count in list(subunit_combo_counts.most_common(10)):
            print(f"  {combo}: {count} events")

        print(f"\nIons in cavity statistics:")
        print(f"  Mean: {summary['cavity_ion_stats']['mean']:.1f} ± {summary['cavity_ion_stats']['std']:.1f}")
        print(f"  Range: {summary['cavity_ion_stats']['min']} - {summary['cavity_ion_stats']['max']}")

        print(f"\n✓ Summary saved to: {output_file}")
        print("="*80)

        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────────────────────────

    def _bar_plot(self, counts, title, ylabel, out_path, color, pct=False):
        """Helper: bar plot of ion count distribution (raw or %)."""
        from collections import Counter as _Counter
        freq   = _Counter(counts)
        items  = sorted(freq.items())
        ns     = [i[0] for i in items]
        vals   = [100. * i[1] / len(counts) for i in items] if pct \
                 else [i[1] for i in items]
        mean   = np.mean(counts)

        fig, ax = plt.subplots(figsize=(max(7, len(ns) * 1.6), 6))
        bars = ax.bar(ns, vals, color=color, alpha=0.75,
                      edgecolor="black", linewidth=1.2, width=0.7)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + (max(vals) * 0.02),
                    f"{val:.1f}%" if pct else f"{int(val)}",
                    ha="center", va="bottom", fontsize=15, fontweight="bold")

        ax.axvline(mean, color="red", linewidth=2.0, linestyle="--",
                   label=f"mean = {mean:.2f}")
        ax.set_xlabel("Number of K⁺ ions in cavity", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(title, fontsize=18, pad=14)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], fontsize=16)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(fontsize=14)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

        # stats box
        stats = (f"mean={mean:.2f}\nstd={np.std(counts):.2f}\n"
                 f"range [{min(counts)}–{max(counts)}]\nN={len(counts):,}")
        ax.text(0.02, 0.97, stats, transform=ax.transAxes,
                fontsize=12, va="top",
                bbox=dict(boxstyle="round", facecolor="white",
                          alpha=0.85, edgecolor="black", linewidth=1.2))
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    def plot_permeation_ion_counts(self):
        """
        Bar plots of ion count distribution at the configured stage frame.
        Saves 4 plots — excl/incl permeating ion x raw/%.
        Filenames are suffixed with the stage for end_3 so end_2 outputs
        are never overwritten.
        """
        if not self.results:
            return

        excl = [r['total_ions_in_cavity']              for r in self.results]
        incl = [r.get('total_ions_including_permeating',
                       r['total_ions_in_cavity'] + 1)  for r in self.results]
        run  = self.channel_type
        t    = self.threshold

        # Keep original filenames for end_2 (backwards compatible);
        # suffix filenames with the stage for end_3 and beyond.
        suffix = "" if self.stage == "end_2" else f"_{self.frame_key}"

        self._bar_plot(excl,
            f"K⁺ ions when {self.stage_label}\n(excl. permeating ion)",
            "Number of permeation events",
            self.stage_plot_dir / "ion_count_permeation_excl.png",
            color="#4878CF")

        self._bar_plot(excl,
            f"K⁺ ions when {self.stage_label} — %\n(excl. permeating ion)",
            "% of permeation events",
            self.stage_plot_dir / "ion_count_permeation_excl_pct.png",
            color="#4878CF", pct=True)

        self._bar_plot(incl,
            f"K⁺ ions when {self.stage_label}",
            "Number of permeation events",
            self.stage_plot_dir / "ion_count_permeation_incl.png",
            color="#E67E22")

        self._bar_plot(incl,
            f"K⁺ ions when {self.stage_label} — %",
            "% of permeation events",
            self.stage_plot_dir / "ion_count_permeation_incl_pct.png",
            color="#E67E22", pct=True)

    def plot_trajectory_ion_counts(self):
        """
        Bar plots of ion count distribution across all frames.
        Reads trajectory_ion_counts.csv (stage-specific) — must already exist.
        Saves 2 plots — raw counts + %.
        """
        suffix = "" if self.stage == "end_2" else f"_{self.frame_key}"
        out_csv = self.results_dir / f"trajectory_ion_counts{suffix}.csv"
        if not out_csv.exists():
            print(f"  {out_csv.name} not found — skipping trajectory plots.")
            return

        counts = []
        with open(out_csv, newline="") as f:
            for row in csv.DictReader(f):
                counts.append(int(row["n_total"]))

        if not counts:
            return

        run = self.channel_type

        self._bar_plot(counts,
            f"K⁺ ion occupancy — {self.stage_label}\nacross ALL simulation frames",
            "Number of frames",
            self.stage_plot_dir / "ion_count_all_frames.png",
            color="#27AE60")

        self._bar_plot(counts,
            f"K⁺ ion occupancy — {self.stage_label}\nacross ALL simulation frames — %",
            "% of all frames",
            self.stage_plot_dir / "ion_count_all_frames_pct.png",
            color="#27AE60", pct=True)

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: per-frame ion count (no trajectory scan — reads from files)
    # ─────────────────────────────────────────────────────────────────────────

    def save_trajectory_ion_counts(self):
        """
        Save the number of ions in the configured channel region for every
        frame of the trajectory.
        Uses ion_region_tracking.json (tracked) + untracked_ions.csv (untracked).
        NO trajectory scan needed.

        Output saved to results_dir (cavity_occupancy/):
          trajectory_ion_counts.csv        for stage="end_2" (unchanged name)
          trajectory_ion_counts_end_3.csv  for stage="end_3"

        Skips if file already exists.
        """
        suffix  = "" if self.stage == "end_2" else f"_{self.frame_key}"
        out_csv = self.results_dir / f"trajectory_ion_counts{suffix}.csv"

        if out_csv.exists() and not self.force_rerun:
            print(f"  {out_csv.name} already exists — skipping (use force_rerun=True to overwrite).")
            return

        print("\n" + "="*80)
        print(f"TRAJECTORY ION COUNTS ({self.stage_label})  [from files, no trajectory scan]")
        print("="*80)

        # find ion_region_tracking.json
        tracking_path = None
        for candidate in [self.results_dir,
                          self.results_dir.parent,
                          self.results_dir.parent.parent]:
            p = candidate / "ion_region_tracking.json"
            if p.exists():
                tracking_path = p
                break

        if tracking_path is None:
            print(f"  ERROR: ion_region_tracking.json not found — cannot build counts.")
            return

        print(f"  Loading {tracking_path.name} ...")
        with open(tracking_path) as f:
            raw = json.load(f)

        # build tracked-in-region per frame, using the configured region code
        tracked_in_region = {}
        for resid_str, frame_dict in raw.items():
            resid = int(resid_str)
            for fr_str, region in frame_dict.items():
                if region == self.region_code:
                    fr = int(fr_str)
                    if self.start_frame <= fr <= self.end_frame:
                        tracked_in_region.setdefault(fr, set()).add(resid)

        print(f"  Tracked:   {len(tracked_in_region)} frames with region-{self.region_code} ions")
        print(f"  Untracked: {len(self.untracked_by_frame)} frames with untracked ions")

        # write CSV
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "t_ns",
                                               "n_tracked", "n_untracked", "n_total"])
            w.writeheader()
            for fr in range(self.start_frame, self.end_frame + 1):
                n_tracked   = len(tracked_in_region.get(fr, set()))
                n_untracked = len(self.untracked_by_frame.get(fr, set()))
                w.writerow({
                    "frame":       fr,
                    "t_ns":        round(fr * self.frame_duration, 4),
                    "n_tracked":   n_tracked,
                    "n_untracked": n_untracked,
                    "n_total":     n_tracked + n_untracked,
                })

        print(f"  Saved: {out_csv}")
        print("="*80)


    # ─────────────────────────────────────────────────────────────────────────
    # COMBINED REGION 2+3 ION COUNTS
    # ─────────────────────────────────────────────────────────────────────────

    def save_combined_region_ion_counts(self):
        """
        Save the number of ions in the COMBINED region 2+3 (GLU/ASN + ASN->HBC)
        for every frame, reading ion_region_tracking.json.

        Output: trajectory_ion_counts_combined_r2r3.csv
        Columns: frame, t_ns, n_region2, n_region3, n_total_combined
        Skips if file already exists.
        """
        out_csv = self.results_dir / "trajectory_ion_counts_combined_r2r3.csv"
        if out_csv.exists() and not self.force_rerun:
            print(f"  {out_csv.name} already exists -- skipping (use force_rerun=True to overwrite).")
            return

        print("\n" + "="*80)
        print("COMBINED REGION 2+3 TRAJECTORY ION COUNTS  [from files, no trajectory scan]")
        print("="*80)

        tracking_path = None
        for candidate in [self.results_dir, self.results_dir.parent, self.results_dir.parent.parent]:
            p = candidate / "ion_region_tracking.json"
            if p.exists():
                tracking_path = p
                break

        if tracking_path is None:
            print("  ERROR: ion_region_tracking.json not found.")
            return

        with open(tracking_path) as f:
            raw = json.load(f)

        tracked_r2 = {}
        tracked_r3 = {}
        for resid_str, frame_dict in raw.items():
            resid = int(resid_str)
            for fr_str, region in frame_dict.items():
                fr = int(fr_str)
                if self.start_frame <= fr <= self.end_frame:
                    if region == 2:
                        tracked_r2.setdefault(fr, set()).add(resid)
                    elif region == 3:
                        tracked_r3.setdefault(fr, set()).add(resid)

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "t_ns", "n_region2", "n_region3", "n_total_combined"])
            w.writeheader()
            for fr in range(self.start_frame, self.end_frame + 1):
                n_r2 = len(tracked_r2.get(fr, set()))
                n_r3 = len(tracked_r3.get(fr, set()))
                w.writerow({
                    "frame": fr, "t_ns": round(fr * self.frame_duration, 4),
                    "n_region2": n_r2, "n_region3": n_r3, "n_total_combined": n_r2 + n_r3,
                })

        print(f"  Saved: {out_csv}")
        print("="*80)

        counts = []
        with open(out_csv, newline="") as f:
            for row in csv.DictReader(f):
                counts.append(int(row["n_total_combined"]))
        if counts:
            self._bar_plot(counts,
                f"K⁺ ions in GLU/HBC region\nacross ALL simulation frames",
                "Number of frames",
                self.combined_plot_dir / "ion_count_all_frames.png",
                color="#8E44AD")
            self._bar_plot(counts,
                f"K⁺ ions in GLU/HBC region\nacross ALL simulation frames — %",
                "% of all frames",
                self.combined_plot_dir / "ion_count_all_frames_pct.png",
                color="#8E44AD", pct=True)

    def save_combined_region_at_end3(self):
        """
        At each end_3 permeation frame, count how many ions are in the combined
        region 2+3, excluding the permeating ion, from ion_region_tracking.json.

        Output: combined_r2r3_at_end3.csv
        Columns: ion_id, end_3_frame, n_region2, n_region3, n_total_combined
        Skips if file already exists.
        """
        out_csv = self.results_dir / "combined_r2r3_at_end3.csv"
        if out_csv.exists() and not self.force_rerun:
            print(f"  {out_csv.name} already exists -- skipping (use force_rerun=True to overwrite).")
            return

        print("\n" + "="*80)
        print("COMBINED REGION 2+3 COUNTS AT END_3 FRAMES  [from ion_region_tracking.json]")
        print("="*80)

        tracking_path = None
        for candidate in [self.results_dir, self.results_dir.parent, self.results_dir.parent.parent]:
            p = candidate / "ion_region_tracking.json"
            if p.exists():
                tracking_path = p
                break

        if tracking_path is None:
            print("  ERROR: ion_region_tracking.json not found.")
            return

        with open(tracking_path) as f:
            raw = json.load(f)

        tracked_r2 = {}
        tracked_r3 = {}
        for resid_str, frame_dict in raw.items():
            resid = int(resid_str)
            for fr_str, region in frame_dict.items():
                fr = int(fr_str)
                if region == 2:
                    tracked_r2.setdefault(fr, set()).add(resid)
                elif region == 3:
                    tracked_r3.setdefault(fr, set()).add(resid)

        rows = []
        combined_excl = []
        combined_incl = []
        for event in self.permeation_events:
            ion_id = int(event["ion_id"])
            frame  = int(event["end_3"])
            n_r2 = len(tracked_r2.get(frame, set()) - {ion_id})
            n_r3 = len(tracked_r3.get(frame, set()) - {ion_id})
            n_excl = n_r2 + n_r3
            n_incl = n_excl + 1   # +1 for the permeating ion itself
            rows.append({
                "ion_id": ion_id, "end_3_frame": frame,
                "n_region2": n_r2, "n_region3": n_r3,
                "n_total_combined_excl": n_excl,
                "n_total_combined_incl": n_incl,
            })
            combined_excl.append(n_excl)
            combined_incl.append(n_incl)

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ion_id", "end_3_frame", "n_region2", "n_region3",
                "n_total_combined_excl", "n_total_combined_incl"])
            w.writeheader()
            w.writerows(rows)

        print(f"  Saved: {out_csv}")
        if combined_excl:
            print(f"  Mean GLU/HBC ions at end_3 (excl): {np.mean(combined_excl):.2f} +/- {np.std(combined_excl):.2f}")
            print(f"  Mean GLU/HBC ions at end_3 (incl): {np.mean(combined_incl):.2f} +/- {np.std(combined_incl):.2f}")
            print("="*80)
            # excl plots
            self._bar_plot(combined_excl,
                "K⁺ ions in GLU/HBC region when ion leaves HBC gate\n(excl. permeating ion)",
                "Number of permeation events",
                self.combined_plot_dir / "ion_count_at_end3_excl.png",
                color="#E74C3C")
            self._bar_plot(combined_excl,
                "K⁺ ions in GLU/HBC region when ion leaves HBC gate — %\n(excl. permeating ion)",
                "% of permeation events",
                self.combined_plot_dir / "ion_count_at_end3_excl_pct.png",
                color="#E74C3C", pct=True)
            # incl plots
            self._bar_plot(combined_incl,
                "K⁺ ions in GLU/HBC region when ion leaves HBC gate",
                "Number of permeation events",
                self.combined_plot_dir / "ion_count_at_end3_incl.png",
                color="#C0392B")
            self._bar_plot(combined_incl,
                "K⁺ ions in GLU/HBC region when ion leaves HBC gate — %",
                "% of permeation events",
                self.combined_plot_dir / "ion_count_at_end3_incl_pct.png",
                color="#C0392B", pct=True)


def run_cavity_occupancy_analysis(universe, channel2, permeation_events,
                                   asn_residues, glu_residues, subunit_groups,
                                   channel_type="G12", threshold=3.0, results_dir=".",
                                   start_frame=0, end_frame=None, frame_duration=0.075,
                                   channel3=None, hbc_residues=None,
                                   channel1=None, sf_residues=None, force_rerun=False):
    """
    Convenience function to run cavity occupancy analysis for UP TO THREE stages:
      - end_1 (selectivity filter, SF) — run automatically if channel1 and
                                          sf_residues are provided
      - end_2 (GLU/ASN cavity)         — always run
      - end_3 (ASN -> HBC gate)        — run automatically if channel3 and
                                          hbc_residues are provided

    All end_2 outputs use the original filenames (cavity_occupancy_at_end2.json,
    ion_count_permeation_excl.png, trajectory_ion_counts.csv, ...).
    All end_1 / end_3 outputs land in the same results_dir with an "_end_1" /
    "_end_3" suffix so nothing is overwritten.

    Parameters
    ----------
    universe           : MDAnalysis Universe
    channel2           : Channel object for region 2 (cavity)
    permeation_events  : list of permeation event dicts
    asn_residues       : list of ASN resids
    glu_residues       : list of GLU resids
    subunit_groups     : dict {subunit_name: [resids]}
    channel_type       : str
    threshold          : float, Å (default 3.0)
    results_dir        : Path or str — base RUN*/ results directory; outputs are
                          written to results_dir/cavity_occupancy/ automatically
    start_frame        : int (default 0)
    end_frame          : int (default last frame)
    frame_duration     : float, ns per frame (default 0.075)
    channel3           : Channel object for region 3 — needed to also run end_3
    hbc_residues       : list of HBC resids — needed to also run end_3
    channel1           : Channel object for region 1 — needed to also run end_1 (SF)
    sf_residues        : list of SF resids — needed to also run end_1 (SF)
    force_rerun        : bool — if True, recompute and overwrite existing CSV files

    Returns
    -------
    dict : {"end_1": analyzer_or_None, "end_2": analyzer, "end_3": analyzer_or_None}
    """
    analyzer_end2 = CavityOccupancyAnalysis(
        universe, channel2, permeation_events,
        asn_residues, glu_residues, subunit_groups,
        channel_type, threshold, results_dir,
        start_frame, end_frame, frame_duration,
        stage="end_2", force_rerun=force_rerun,
    )
    analyzer_end2.run_analysis()
    analyzer_end2.save_results()
    analyzer_end2.create_summary()
    analyzer_end2.plot_permeation_ion_counts()
    analyzer_end2.save_trajectory_ion_counts()
    analyzer_end2.plot_trajectory_ion_counts()

    analyzer_end3 = None
    if channel3 is not None and hbc_residues is not None:
        analyzer_end3 = CavityOccupancyAnalysis(
            universe, channel2, permeation_events,
            asn_residues, glu_residues, subunit_groups,
            channel_type, threshold, results_dir,
            start_frame, end_frame, frame_duration,
            stage="end_3", channel3=channel3, hbc_residues=hbc_residues,
            force_rerun=force_rerun,
        )
        analyzer_end3.run_analysis()
        analyzer_end3.save_results()
        analyzer_end3.create_summary()
        analyzer_end3.plot_permeation_ion_counts()
        analyzer_end3.save_trajectory_ion_counts()
        analyzer_end3.plot_trajectory_ion_counts()
    else:
        print("\n  NOTE: channel3 / hbc_residues not provided — skipping end_3 analysis.")

    analyzer_end1 = None
    if channel1 is not None and sf_residues is not None:
        analyzer_end1 = CavityOccupancyAnalysis(
            universe, channel2, permeation_events,
            asn_residues, glu_residues, subunit_groups,
            channel_type, threshold, results_dir,
            start_frame, end_frame, frame_duration,
            stage="end_1", channel1=channel1, sf_residues=sf_residues,
            force_rerun=force_rerun,
        )
        analyzer_end1.run_analysis()
        analyzer_end1.save_results()
        analyzer_end1.create_summary()
        analyzer_end1.plot_permeation_ion_counts()
        analyzer_end1.save_trajectory_ion_counts()
        analyzer_end1.plot_trajectory_ion_counts()
    else:
        print("\n  NOTE: channel1 / sf_residues not provided — skipping end_1 (SF) analysis.")

    # Combined region 2+3 counts — always run using end_2 analyzer
    # (it has access to all permeation_events including end_3 frames)
    analyzer_end2.save_combined_region_ion_counts()
    analyzer_end2.save_combined_region_at_end3()

    return {"end_1": analyzer_end1, "end_2": analyzer_end2, "end_3": analyzer_end3}