import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import warnings
from MDAnalysis.core.groups import AtomGroup
warnings.filterwarnings("ignore")
from pathlib import Path
from analysis.channels import Channel
from analysis.ion_analysis import IonPermeationAnalysis
from analysis.distance_calc import calculate_distances
from analysis.organizing_frames import cluster_frames_by_closest_residue, tracking_ion_distances, plot_ion_distance_traces
from analysis.organizing_frames import close_contact_residues_analysis, get_clean_ion_coexistence_table, analyze_and_plot_average_ions_per_residue
from analysis.frames_frequencies_plots import plot_top_intervals_by_frames
from analysis.analyze_ch2_permeation import analyze_ch2_permation_residues, count_residue_combinations_with_duplicates, find_all_pre_permeation_patterns
from analysis.analyze_ch2_permeation import count_last_residues,plot_last_residue_bar_chart, save_residue_combination_summary_to_excel
from analysis.force_analysis import collect_sorted_cosines_until_permeation
from analysis.force_analysis import extract_permeation_frames, extract_last_frame_analysis, extract_permeation_forces
import json
import pandas as pd
import os
from analysis.permation_profile_creator import PermeationAnalyzer
from analysis.close_residues_analysis import plot_residue_counts, analyze_residue_combinations, find_closest_residues_percentage
from analysis.close_residues_analysis import count_frames_residue_closest, extract_min_mean_distance_pairs, count_frames_pair_closest, plot_start_frame_residue_distribution
from analysis.significant_forces import significant_forces
from analysis.find_clean_stuck_frames import find_clean_stuck_frames
from analysis.force_extractor import analyze_frame_for_ion
from analysis.electric_field_analysis import run_field_analysis, plot_field_magnitudes_from_json, significance_field_analysis, generate_electric_field_heatmap_along_axis
from analysis.best_alignment import run_best_combo_per_ion_from_json
from analysis.find_closest_unentered_ion_to_upper_gate import find_closest_unentered_ion_to_upper_gate
from analysis.ions_sf_analysis import analyze_resid_changes_and_plot, plot_field_histograms_from_json, run_all_field_permeation_analyses

from analysis.residue_proximity_analysis import ResidueProximityAnalysis
from analysis.sf_alignment_analysis import run_sf_alignment_analysis

def main():
    parser = argparse.ArgumentParser(description="Run dual-channel ion permeation analysis.")
    parser.add_argument("--do_permeation_analysis", default=False)
    parser.add_argument("--do_electric_field_analysis", default=True)
    # parser.add_argument("--top_file", default="/media/konsfr/KINGSTON/trajectory/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="/media/konsfr/KINGSTON/trajectory/protein.nc")

    # parser.add_argument("--top_file", default="../com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="../protein.nc")
    # parser.add_argument("--channel_type", default="G4")

    # parser.add_argument("--top_file", default="../../G4-homotetramer/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="../../G4-homotetramer/protein.nc")

    # parser.add_argument("--top_file", default="../Rep0/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="../Rep0/protein.nc")

    # parser.add_argument("--top_file", default="../GIRK12_WT/RUN2/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="../GIRK12_WT/RUN2/protein.nc")
    

    # parser.add_argument("--top_file", default="../GIRK12_WT/RUN1/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="../GIRK12_WT/RUN1/protein.nc")

    # parser.add_argument("--top_file", default="/media/konsfr/KINGSTON/trajectory/Rep0/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="/media/konsfr/KINGSTON/trajectory/Rep0/GIRK_4kfm_NoCHL_Rep0_500ns.nc")
    # parser.add_argument("--channel_type", default="G12")
    parser.add_argument("--channel_type", default="G12")
    parser.add_argument("--run_number", default=1)
    # parser.add_argument("--channel_type", default="G2_FD")
    args = parser.parse_args()
    generate_electric_field_heatmap =  False
    data_path = "/home/data/Konstantina/ion-permeation-analyzer-results/version1"
    # data_path = "/media/konsfr/KINGSTON/trajectory/final_results/"
    run_number = args.run_number

    
    channel_type = args.channel_type
    print(f"Running analysis for channel type: {channel_type}")
    if args.channel_type == "G4":
        upper1 = [106, 431, 756, 1081]
        lower1 = [100, 425, 750, 1075]  #sf_residues
  
        upper2 = [100, 425, 750, 1075]  #sf_residues
        lower2 = [130, 455, 780, 1105]  #asn_residues

        upper3 = [130, 455, 780, 1105]
        lower3 = [138, 463, 788, 1113]

        upper4 = [138, 463, 788, 1113]  #hbc_residues
        lower4 = [265, 590, 915 ,1240]

        upper5 = [265, 590, 915 ,1240]
        lower5 = [259, 584, 909, 1234]

        hbc_residues = [138, 463, 788, 1113]
        hbc_diagonal_pairs = [(138, 788), (463, 1113)]

        sf_low_res_residues = [100, 425, 750, 1075]
        sf_low_res_diagonal_pairs = [(100, 750), (425, 1075)]

        glu_residues = [98, 423, 748, 1073]
        asn_residues = [130, 455, 780, 1105]
        sf_residues = [100, 425, 750, 1075]

        start_frame = 0
        # start_frame = 5550
        # start_frame = 6500
        end_frame = 6799

    elif args.channel_type == "G2":
        # upper1 = [106, 434, 762, 1090]
        upper1 = [104, 432, 760, 1088]
        lower1 = [100, 428, 756, 1084]

        upper2 = [100, 428, 756, 1084]
        lower2 = [130, 458, 786, 1114] #asn_residues

        upper3 = [130, 458, 786, 1114] #asn_residues
        lower3 = [138, 466, 794, 1122] #hbc_residues

        upper4 = [138, 466, 794, 1122] #hbc_residues
        lower4 = [265, 593, 921, 1249]

        upper5 = [265, 593, 921, 1249] #upper gloop
        lower5 = [259, 587, 915, 1243] #lower gloop

        hbc_residues = [138, 466, 794, 1122]
        hbc_diagonal_pairs = [(138, 466 ), (1122, 794)]

        glu_residues = [98, 426, 754, 1082]
        asn_residues = [130, 458, 786, 1114]
        subunit_groups = {
            'A': [98, 130],  # GLU and ASN/ASP of subunit A
            'B': [754, 786],
            'C': [426, 458],
            'D': [1082, 1114]
        }
        sf_residues = [100, 428, 756, 1084]

        sf_low_res_residues = [100, 428, 756, 1084]
        sf_low_res_diagonal_pairs = [(100, 428 ), (1084,  756)]

        start_frame = 0
        # start_frame = 800
        # start_frame = 5550
        # start_frame = 6500
        # end_frame = 1250
        end_frame = 6799

        top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/G2_S181P/RUN{run_number}/com.prmtop")
        traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/G2_S181P/RUN{run_number}/protein.nc")
        results_dir = Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/{args.channel_type}/RUN{run_number}")
        results_dir.mkdir(parents=True, exist_ok=True)


    # elif args.channel_type == "G2_FD":
    #     upper1 = [106, 434, 762, 1090]
    #     lower1 = [100, 428, 756, 1084]

    #     upper2 = [100, 428, 756, 1084]
    #     lower2 = [130, 458, 786, 1114] #asn_residues

    #     upper3 = [130, 458, 786, 1114] #asn_residues
    #     lower3 = [138, 466, 794, 1122] #hbc_residues

    #     upper4 = [138, 466, 794, 1122] #hbc_residues
    #     lower4 = [265, 593, 921, 1249]

    #     upper5 = [265, 593, 921, 1249] #upper gloop
    #     lower5 = [259, 587, 915, 1243] #lower gloop

    #     hbc_residues = [138, 466, 794, 1122]
    #     hbc_diagonal_pairs = [(138, 1122), (466,  794)]

    #     glu_residues = [98, 426, 754, 1082]
    #     asn_residues = [130, 458, 786, 1114]
    #     sf_residues = [100, 428, 756, 1084]

    #     sf_low_res_residues = [100, 428, 756, 1084]
    #     sf_low_res_diagonal_pairs = [(100, 1084 ), (428, 756)]

    #     start_frame = 0
    #     # start_frame = 800
    #     # start_frame = 5550
    #     # start_frame = 6500
    #     # end_frame = 1250

    #     top_file = Path(f"/home/data/Konstantina/GIRK2/GIRK2_FD_RUN{run_number}/com_4fs.prmtop")
    #     traj_file = Path(f"/home/data/Konstantina/GIRK2/GIRK2_FD_RUN{run_number}/protein.nc")
    #     results_dir = Path(f"{data_path}/results_G2_FD_CHL_RUN{run_number}")
    #     end_frame = 6799

    elif args.channel_type == "G12":
        # upper1 = [107, 431, 757, 1082]
        upper1 = [105, 429, 755, 1080]
        lower1 = [100, 424, 749, 1075]  #sf_residues
  
        upper2 = [100, 424, 749, 1075]  #sf_residues
        lower2 = [130,454,779,1105 ]  #asn_residues

        upper3 = [130,454,779,1105 ]
        lower3 = [138,462,787,1113]

        upper4 = [138,462,787,1113]  #hbc_residues
        lower4 = [265, 589, 914 ,1240]

        # upper5 = [265, 589, 914 ,1240]
        # lower5 = [259, 583, 908, 1234]

        hbc_residues = [138,462,787,1113]
        hbc_diagonal_pairs = [(138, 1113 ), (462, 787)]

        sf_low_res_residues = [101, 425, 750, 1076] 
        sf_low_res_diagonal_pairs = [(101, 1076 ), (425, 750)]

        glu_residues = [98,422,747,1073]
        asn_residues = [130,454,779,1105 ]
        subunit_groups = {
            'A': [422, 454],  # GLU and ASN/ASP of subunit A
            'B': [98, 130],
            'C': [747, 779],
            'D': [1073, 1105]
        }
        sf_residues = [101, 425, 750, 1076] 

        start_frame = 0
        # start_frame = 2500
        # end_frame = 1000
        # end_frame = 1250
        end_frame = 6799
        # end_frame = 4300

        top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/G12_S181P_S170P/RUN{run_number}/com.prmtop")
        traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/G12_S181P_S170P/RUN{run_number}/protein.nc")
        results_dir = Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/{args.channel_type}/RUN{run_number}")
        results_dir.mkdir(parents=True, exist_ok=True)

    elif args.channel_type == "G12_ML":
        # upper1 = [107, 431, 757, 1082]
        upper1 = [106, 431, 756, 1081]
        lower1 = [101, 426, 751, 1076]  #sf_residues
  
        upper2 = [101, 426, 751, 1076]  #sf_residues
        lower2 = [131,456,781,1106 ]  #asn_residues

        upper3 = [131,456,781,1106]
        lower3 = [139,464,789,1114]

        upper4 = [139,464,789,1114]  #hbc_residues
        lower4 = [266, 591, 916 ,1241]

        # upper5 = [265, 589, 914 ,1240]
        # lower5 = [259, 583, 908, 1234]

        hbc_residues = [139,464,789,1114]
        hbc_diagonal_pairs = [(139, 1114 ), (464, 789)]

        sf_low_res_residues = [101, 426, 751, 1076] 
        sf_low_res_diagonal_pairs = [(101, 1076 ), (426, 751)]

        glu_residues = [99,424,749,1074]
        asn_residues = [131,456,781,1106 ]
        subunit_groups = {
            'A': [781, 749],  # GLU and ASN/ASP of subunit A
            'B': [1106, 1074],
            'C': [424, 456],
            'D': [99, 131]
        }
        sf_residues = [101, 426, 751, 1076] 

        start_frame = 0
        # start_frame = 2500
        # end_frame = 1000
        # end_frame = 1250
        end_frame = 6799
        # end_frame = 4300

        top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_ML297/RUN{run_number}/com.prmtop")
        traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_ML297/RUN{run_number}/protein.nc")
        results_dir = Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/{args.channel_type}/RUN{run_number}")
        results_dir.mkdir(parents=True, exist_ok=True)
#########################################################################################


    from analysis.analyzer_utils import save_analyzer_state, load_analyzer_state, restore_analyzer_from_state
    # end_frame = 6562
    u = mda.Universe(top_file, traj_file)
    results_dir.mkdir(exist_ok=True)

    # start_frame = 5414
    # end_frame = 5553
    # start_frame = 5000
    # end_frame = 5500
    # start_frame = 6500
    start_frame = 0
    # start_frame = 2800
    # # start_frame = 6500
    end_frame = len(u.trajectory) - 1
    # end_frame = 4300
    print(f"Using end_frame = {end_frame} (last frame of trajectory)")
    
    ch1 = Channel(u, upper1, lower1, num=1, radius=2.5)
    ch2 = Channel(u, upper2, lower2, num=2, radius=15.0)
    ch3 = Channel(u, upper3, lower3, num=3, radius=15.0)
    ch4 = Channel(u, upper4, lower4, num=4, radius=10.0)

 # ========== TRY TO LOAD SAVED STATE ==========
    saved_state = load_analyzer_state(results_dir)
    
    if saved_state is not None:
        # Use saved state - SKIP trajectory analysis
        print("\n" + "="*80)
        print("USING SAVED ANALYZER STATE")
        print("="*80)
        
        analyzer = restore_analyzer_from_state(
            u, saved_state, results_dir, ch1, ch2, ch3, ch4,
            hbc_residues, hbc_diagonal_pairs,
            sf_low_res_residues, sf_low_res_diagonal_pairs
        )
        
    else:
        # No saved state - run full analysis
        print("\n" + "="*80)
        print("RUNNING FULL TRAJECTORY ANALYSIS")
        print("="*80)
        
        analyzer = IonPermeationAnalysis(
            u, ion_selection="resname K+ K", 
            start_frame=start_frame, end_frame=end_frame,
            channel1=ch1, channel2=ch2, channel3=ch3, channel4=ch4,
            hbc_residues=hbc_residues, hbc_diagonal_pairs=hbc_diagonal_pairs,
            sf_low_res_residues=sf_low_res_residues, 
            sf_low_res_diagonal_pairs=sf_low_res_diagonal_pairs,
            results_dir=results_dir,
            count_ions=True
        )
        
        analyzer.run_analysis()
        analyzer.print_results()
        
        # SAVE STATE for future runs
        save_analyzer_state(analyzer, results_dir)

    from analysis.occupancy_alignment import run_comprehensive_end2_analysis

        # ============= COMPREHENSIVE END_2 ANALYSIS =============
    if len(analyzer.permeation_events) > 0:
        print("\n" + "="*80)
        print("Running Comprehensive end_2 Analysis...")
        print("="*80)

        threshold=3
        comp_analyzer = run_comprehensive_end2_analysis(
            universe=u,
            channel1=ch1,
            channel2=ch2,
            permeation_events=analyzer.permeation_events,
            asn_residues=asn_residues,
            glu_residues=glu_residues,
            subunit_groups=subunit_groups,
            channel_type=channel_type,
            threshold=threshold,  # Distance threshold for "close to residue"
            results_dir=f"{results_dir}/occupancy_alignment/{threshold}"  # <-- Separate folder
        )
        
        print("\nComprehensive end_2 Analysis Complete!")

    # from analysis.cavity_occupancy_analysis import run_cavity_occupancy_analysis

    # cavity_analyzer = run_cavity_occupancy_analysis(
    #     universe=u,
    #     channel2=ch2,
    #     permeation_events=analyzer.permeation_events,
    #     asn_residues=asn_residues,
    #     glu_residues=glu_residues,
    #     subunit_groups=subunit_groups,
    #     channel_type=channel_type,
    #     threshold=3.0,
    #     results_dir=f"{results_dir}/cavity_occupancy"
    # )

    # if analyzer.count_ions and len(analyzer.permeation_events) > 0:
    #    sf_analyzer = run_sf_alignment_analysis(
    #        universe=u,
    #        channel1=ch1,
    #        channel2=ch2,
    #        permeation_events=analyzer.permeation_events,
    #        asn_residues=asn_residues,
    #        glu_residues=glu_residues,
    #        results_dir=results_dir
    #    )


    # from analysis.closest_residue_analysis import run_closest_residue_analysis

    # threshold=3
    # closest_res_analyzer = run_closest_residue_analysis(
    #     universe=u,
    #     permeation_events=analyzer.permeation_events,
    #     asn_residues=asn_residues,
    #     glu_residues=glu_residues,
    #     channel_type=channel_type,
    #     threshold=threshold,
    #     analysis_start_frame=start_frame,
    #     residue_types=['ASN'],  # Only ASN
    #     results_dir=f"{results_dir}/closest_res/{threshold}"
    # )  

    # from analysis.longest_duration_analysis import run_last_duration_analysis
    # threshold = 3.0
    # min_duration = 3

    # # Run 1: RESIDUE-LEVEL
    # run_last_duration_analysis(
    #     universe=u,
    #     permeation_events=analyzer.permeation_events,
    #     asn_residues=asn_residues,
    #     glu_residues=glu_residues,
    #     channel_type=channel_type,
    #     threshold=threshold,
    #     min_duration=min_duration,
    #     analysis_start_frame=start_frame,
    #     residue_types=['ASN', 'GLU'],
    #     subunit_groups=None,  # <-- Residue mode
    #     results_dir=f"{results_dir}/duration_residue/{threshold}_{min_duration}"
    # )

    # # Run 2: SUBUNIT-LEVEL
    # run_last_duration_analysis(
    #     universe=u,
    #     permeation_events=analyzer.permeation_events,
    #     asn_residues=asn_residues,
    #     glu_residues=glu_residues,
    #     channel_type=channel_type,
    #     threshold=threshold,
    #     min_duration=min_duration,
    #     analysis_start_frame=start_frame,
    #     residue_types=['ASN', 'GLU'],
    #     subunit_groups=subunit_groups,  # <-- Subunit mode
    #     results_dir=f"{results_dir}/duration_subunit/{threshold}_{min_duration}"
    # )


    # #  NEW: Add proximity analysis
    # print("\n" + "="*70)
    # print("RUNNING RESIDUE PROXIMITY ANALYSIS")
    # print("="*70)
    
    # proximity_analyzer = ResidueProximityAnalysis(
    #     universe=u,
    #     ion_selection="resname K+ K",
    #     glu_residues=glu_residues,
    #     asn_residues=asn_residues,
    #     start_frame=start_frame,
    #     end_frame=end_frame,
    #     results_dir=results_dir,
    #     channel_type=channel_type,  # NEW: pass the channel_type
    #     cutoff=3.0
    # )
    
    # proximity_analyzer.run_analysis()
    # proximity_analyzer.print_results()
    # proximity_analyzer.save_results()
    
    # # Generate timeline plots
    # print("\nGenerating timeline plots...")
    # proximity_analyzer.plot_individual_glu_timelines()
    # proximity_analyzer.plot_individual_asn_timelines()
    # proximity_analyzer.plot_all_residues_combined()
    # proximity_analyzer.plot_aggregate_comparison()
    
    # # Generate bar chart plots
    # print("\nGenerating bar chart plots...")
    # proximity_analyzer.plot_individual_residue_bar_chart()
    # proximity_analyzer.plot_all_residues_bar_chart()
    # proximity_analyzer.plot_aggregate_bar_chart()
    # proximity_analyzer.plot_percentage_comparison_bar_chart()

if __name__ == "__main__":
    main()
