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
        upper1 = [105, 433, 761, 1089]
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

        upper_gl_residues = [265, 593, 921, 1249  ]
        upper_gl_diagonal_pairs = [(265, 593 ), (921, 1249)]

        lower_gl_residues = [259, 587, 915, 1243 ]
        lower_gl_diagonal_pairs = [(259, 587 ), (915, 1243)]

        glu_residues = [98, 426, 754, 1082]
        glu_residues_pairs = [(98,426), (754,1082)]

        asn_residues = [130, 458, 786, 1114]
        asn_residues_pairs = [(130,458),(786,1114)]

        subunit_groups = {
            'A': [98, 130],  # GLU and ASN/ASP of subunit A
            'B': [754, 786],
            'C': [426, 458],
            'D': [1082, 1114]
        }
        sf_residues = [100, 428, 756, 1084]
        frame_duration = 0.4

        sf_low_res_residues = [100, 428, 756, 1084]
        sf_low_res_diagonal_pairs = [(100, 428 ), (1084,  756)]

        start_frame = 0
        # start_frame = 800
        # start_frame = 5550
        # start_frame = 6500
        # end_frame = 1250
        end_frame = 6799

        pip_mapping = {
            1313: "2b",
            1315: "2d",
            1314: "2c",
            1316: "2a"
        }

        pip_resids = [1313, 1314, 1315, 1316]
        mapping_json = "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/find_pip_residue_mapping/G2/A_B_C_D/pip_residue_mapping.json"

        top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK2_WT/RUN{run_number}/com_4fs.prmtop")
        traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK2_WT/RUN{run_number}/protein.nc")
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

    # elif args.channel_type == "G12":
    #     # upper1 = [107, 431, 757, 1082]
    #     upper1 = [105, 429, 755, 1080]
    #     lower1 = [100, 424, 749, 1075]  #sf_residues
  
    #     upper2 = [100, 424, 749, 1075]  #sf_residues
    #     lower2 = [130,454,779,1105 ]  #asn_residues

    #     upper3 = [130,454,779,1105 ]
    #     lower3 = [138,462,787,1113]

    #     upper4 = [138,462,787,1113]  #hbc_residues
    #     lower4 = [265, 589, 914 ,1240]

    #     # upper5 = [265, 589, 914 ,1240]
    #     # lower5 = [259, 583, 908, 1234]

    #     hbc_residues = [138,462,787,1113]
    #     hbc_diagonal_pairs = [(138, 1113 ), (462, 787)]

    #     upper_gl_residues = [265, 589, 914, 1240 ]
    #     upper_gl_diagonal_pairs = [(265, 1240 ), (589, 914)]

    #     lower_gl_residues = [259, 583, 908, 1234 ]
    #     lower_gl_diagonal_pairs = [(259, 1234 ), (583, 908)]

    #     sf_low_res_residues = [101, 425, 750, 1076] 
    #     sf_low_res_diagonal_pairs = [(101, 1076 ), (425, 750)]

    #     glu_residues = [98,422,747,1073]
    #     glu_residues_pairs = [(98,1073), (422,747)]

    #     asn_residues = [130,454,779,1105 ]
    #     asn_residues = [(130,1105), (454,779)]
        
        
    #     subunit_groups = {
    #         'A': [422, 454],  # GLU and ASN/ASP of subunit A
    #         'B': [98, 130],
    #         'C': [747, 779],
    #         'D': [1073, 1105]
    #     }
    #     sf_residues = [100, 424, 749, 1075]  

    #     start_frame = 0
    #     # start_frame = 2500
    #     # end_frame = 1000
    #     # end_frame = 1250
    #     end_frame = 6799
    #     # end_frame = 4300

    #     top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/G12_S181P_S170P/RUN{run_number}/com.prmtop")
    #     traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/G12_S181P_S170P/RUN{run_number}/protein.nc")
    #     results_dir = Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/{args.channel_type}/RUN{run_number}")
    #     results_dir.mkdir(parents=True, exist_ok=True)

    elif args.channel_type == "G12" and int(run_number)<21:
        upper1 = [106, 431, 755, 1080]
        lower1 = [101, 426, 750, 1075]  #sf_residues
  
        upper2 = [101, 426, 750, 1075]  #sf_residues
        lower2 = [131,456,780,1105 ]  #asn_residues

        upper3 = [131,456,780,1105 ]
        lower3 = [139,464,788,1113]

        upper4 = [139,464,788,1113]  #hbc_residues
        lower4 = [266,591,915,1240]
        res_324 = [920, 1245, 596, 271]   
        res_315 =[262, 587, 911,1236] 
        res_233 =[180, 505, 829, 1154] 
        res_199 = [795, 1120, 471,146]

        # upper5 = [265, 589, 914 ,1240]
        # lower5 = [259, 583, 908, 1234]

        hbc_residues = [139,464,788,1113]
        hbc_diagonal_pairs = [(788, 1113),(139, 464 )]

        upper_gl_residues = [266,591,915,1240]
        upper_gl_diagonal_pairs = [(1240, 915), (266, 591 )]

        lower_gl_residues = [260,585, 909, 1234 ]
        lower_gl_diagonal_pairs = [(909, 1234), (260, 585 )]

        sf_low_res_residues = [101, 426, 750, 1075] 
        sf_low_res_diagonal_pairs = [(750, 1075), (101, 426 )]

        glu_residues = [99,424,748,1073]
        glu_residues_pairs =[(748,1073), (99,424)]

        asn_residues = [131,456,780,1105 ]
        asn_residues_pairs = [(780,1105), (131,456)]

        frame_duration = 0.075

        subunit_groups = {
            'A': [780, 748],  # GLU and ASN/ASP of subunit A
            'B': [456, 424],
            'C': [1073, 1105],
            'D': [99, 131]
        }
        sf_residues = [101, 426, 750, 1075]  

        mapping_json = "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/find_pip_residue_mapping/G12/A_B_C/pip_residue_mapping.json"
        start_frame = 0
        # start_frame = 2500
        # end_frame = 1000
        # end_frame = 1250
        end_frame = 6799
        # end_frame = 3000
        pip_mapping = {
                    1301: "2b",
                    1302: "G1",
                    1303: "2c",
                    1304: "2a"
                }
        gly_residues = [127, 452, 776, 1101]

        pip_resids = [1301, 1302, 1303, 1304]

        top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN{run_number}/com_4fs.prmtop")
        traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN{run_number}/protein.nc")
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

        mapping_json = "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/find_pip_residue_mapping/G12_ML/A_B_C/pip_residue_mapping.json"
        # upper5 = [265, 589, 914 ,1240]
        # lower5 = [259, 583, 908, 1234]

        hbc_residues = [139,464,789,1114]
        hbc_diagonal_pairs = [(139, 1114 ), (464, 789)]

        upper_gl_residues = [266,591,916,1241]
        upper_gl_diagonal_pairs = [(266, 1241 ), (591, 916)]

        lower_gl_residues = [260,585, 910, 1235 ]
        lower_gl_diagonal_pairs = [(260, 1235 ), (585, 910)]

        sf_low_res_residues = [101, 426, 751, 1076] 
        sf_low_res_diagonal_pairs = [(101, 1076 ), (426, 751)]

        glu_residues = [99,424,749,1074]
        glu_residues_pairs =[(99,1074),(424,749)]

        asn_residues = [131,456,781,1106 ]
        asn_residues_pairs = [(131,1106),(456,781)]

        gly_residues = [127, 452, 777, 1102]

        subunit_groups = {
            'A': [781, 749],  # GLU and ASN/ASP of subunit A
            'B': [1106, 1074],
            'C': [424, 456],
            'D': [99, 131]
        }
        sf_residues = [101, 426, 751, 1076] 

        frame_duration = 0.075

        start_frame = 0
        # start_frame = 2500
        # end_frame = 1000
        # end_frame = 1250
        end_frame = 6799
        # end_frame = 4300

        pip_resids = [1301, 1302, 1303, 1304]

        pip_mapping = {
            1301: "G1",
            1302: "2b",
            1303: "2a",
            1304: "2c"
        }

        top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_ML297/RUN{run_number}/com.prmtop")
        traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_ML297/RUN{run_number}/protein.nc")
        results_dir = Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/{args.channel_type}/RUN{run_number}")
        results_dir.mkdir(parents=True, exist_ok=True)

    elif args.channel_type == "G12_GAT":
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

        mapping_json = "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/find_pip_residue_mapping/G12_GAT/A_B_C/pip_residue_mapping.json"

        hbc_residues = [138,462,787,1113]
        hbc_diagonal_pairs = [(138, 1113 ), (462, 787)]

        upper_gl_residues = [265, 589, 914, 1240 ]
        upper_gl_diagonal_pairs = [(265, 1240 ), (589, 914)]

        lower_gl_residues = [259, 583, 908, 1234 ]
        lower_gl_diagonal_pairs = [(259, 1234 ), (583, 908)]

        sf_low_res_residues = [101, 425, 750, 1076] 
        sf_low_res_diagonal_pairs = [(101, 1076 ), (425, 750)]

        glu_residues = [98,422,747,1073]
        glu_residues_pairs = [(98,1073), (422,747)]

        asn_residues = [130,454,779,1105 ]
        asn_residues_pairs = [(130,1105), (454,779)]

        pip_resids = [1301, 1302, 1303, 1304]

        gly_residues = [126, 450, 775, 1101]

        frame_duration = 0.075

        subunit_groups = {
            'A': [422, 454],  # GLU and ASN/ASP of subunit A
            'B': [98, 130],
            'C': [747, 779],
            'D': [1073, 1105]
        }
        sf_residues = [100, 424, 749, 1075]  

        start_frame = 0
        # start_frame = 2500
        # end_frame = 1000
        # end_frame = 1250
        end_frame = 6799
        # end_frame = 4300

        pip_mapping = {
            1301: "G1",
            1302: "2c",
            1303: "2a",
            1304: "2b"
        }

        top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_GAT1508/RUN{run_number}/com.prmtop")
        traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_GAT1508/RUN{run_number}/protein.nc")
        results_dir = Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/{args.channel_type}/RUN{run_number}")
        results_dir.mkdir(parents=True, exist_ok=True)

    elif args.channel_type == "G12" and int(run_number)>20:
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

        # mapping_json = "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/find_pip_residue_mapping/G12_GAT/A_B_C/pip_residue_mapping.json"

        hbc_residues = [138,462,787,1113]
        hbc_diagonal_pairs = [(138, 1113 ), (462, 787)]

        upper_gl_residues = [265, 589, 914, 1240 ]
        upper_gl_diagonal_pairs = [(265, 1240 ), (589, 914)]

        lower_gl_residues = [259, 583, 908, 1234 ]
        lower_gl_diagonal_pairs = [(259, 1234 ), (583, 908)]

        sf_low_res_residues = [101, 425, 750, 1076] 
        sf_low_res_diagonal_pairs = [(101, 1076 ), (425, 750)]

        glu_residues = [98,422,747,1073]
        glu_residues_pairs = [(98,1073), (422,747)]

        asn_residues = [130,454,779,1105 ]
        asn_residues_pairs = [(130,1105), (454,779)]

        pip_resids = [1301, 1302, 1303, 1304]

        gly_residues = [126, 450, 775, 1101]

        frame_duration = 0.075

        subunit_groups = {
            'A': [422, 454],  # GLU and ASN/ASP of subunit A
            'B': [98, 130],
            'C': [747, 779],
            'D': [1073, 1105]
        }
        sf_residues = [100, 424, 749, 1075]  

        start_frame = 0
        # start_frame = 2500
        # end_frame = 1000
        # end_frame = 1250
        end_frame = 6799
        # end_frame = 4300

        pip_mapping = {
            1301: "G1",
            1302: "2c",
            1303: "2a",
            1304: "2b"
        }

        top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN{run_number}/com_4fs.prmtop")
        traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_WT/RUN{run_number}/protein.nc")
        results_dir = Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/{args.channel_type}/RUN{run_number}")
        results_dir.mkdir(parents=True, exist_ok=True)



    elif args.channel_type == "G12_D173N":
        # upper1 = [107, 431, 757, 1082]
        upper1 = [105, 429, 755, 1080]
        lower1 = [100, 424, 749, 1075]  #sf_residues
  
        upper2 = [100, 424, 749, 1075]  #sf_residues
        lower2 = [130,454,779,1105 ]  #asn_residues

        upper3 = [130,454,779,1105 ]
        lower3 = [138,462,787,1113]

        upper4 = [138,462,787,1113]  #hbc_residues
        lower4 = [265, 589, 914 ,1240]

        pip_mapping = {
            1301: "G1",
            1302: "2c",
            1303: "2a",
            1304: "2b"
        }

        # mapping_json = "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/find_pip_residue_mapping/G12_GAT/A_B_C/pip_residue_mapping.json"

        hbc_residues = [138,462,787,1113]
        hbc_diagonal_pairs = [(138, 1113 ), (462, 787)]

        upper_gl_residues = [265, 589, 914, 1240 ]
        upper_gl_diagonal_pairs = [(265, 1240 ), (589, 914)]

        lower_gl_residues = [259, 583, 908, 1234 ]
        lower_gl_diagonal_pairs = [(259, 1234 ), (583, 908)]

        sf_low_res_residues = [101, 425, 750, 1076] 
        sf_low_res_diagonal_pairs = [(101, 1076 ), (425, 750)]

        glu_residues = [98,422,747,1073]
        glu_residues_pairs = [(98,1073), (422,747)]

        asn_residues = [130,454,779,1105 ]
        asn_residues_pairs = [(130,1105), (454,779)]

        pip_resids = [1301, 1302, 1303, 1304]

        gly_residues = [126, 450, 775, 1101]

        frame_duration = 0.075

        subunit_groups = {
            'A': [422, 454],  # GLU and ASN/ASP of subunit A
            'B': [98, 130],
            'C': [747, 779],
            'D': [1073, 1105]
        }
        sf_residues = [100, 424, 749, 1075]  

        start_frame = 0
        # start_frame = 2500
        # end_frame = 1000
        # end_frame = 1250
        end_frame = 6799
        # end_frame = 4300

        # pip_mapping = {
        #     1301: "G1",
        #     1302: "2c",
        #     1303: "2a",
        #     1304: "2b"
        # }

        top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_D173N/RUN{run_number}/com.prmtop")
        traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_D173N/RUN{run_number}/protein.nc")
        results_dir = Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/girk_analyser_results/{args.channel_type}/RUN{run_number}")
        results_dir.mkdir(parents=True, exist_ok=True)


    elif args.channel_type == "G12_E141Q":
        # upper1 = [107, 431, 757, 1082]
        upper1 = [105, 429, 755, 1080]
        lower1 = [100, 424, 749, 1075]  #sf_residues
  
        upper2 = [100, 424, 749, 1075]  #sf_residues
        lower2 = [130,454,779,1105 ]  #asn_residues

        upper3 = [130,454,779,1105 ]
        lower3 = [138,462,787,1113]

        upper4 = [138,462,787,1113]  #hbc_residues
        lower4 = [265, 589, 914 ,1240]

        pip_mapping = {
            1301: "G1",
            1302: "2c",
            1303: "2a",
            1304: "2b"
        }

        print("HELLO")
        # mapping_json = "/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/find_pip_residue_mapping/G12_GAT/A_B_C/pip_residue_mapping.json"

        hbc_residues = [138,462,787,1113]
        hbc_diagonal_pairs = [(138, 1113 ), (462, 787)]

        upper_gl_residues = [265, 589, 914, 1240 ]
        upper_gl_diagonal_pairs = [(265, 1240 ), (589, 914)]

        lower_gl_residues = [259, 583, 908, 1234 ]
        lower_gl_diagonal_pairs = [(259, 1234 ), (583, 908)]

        sf_low_res_residues = [101, 425, 750, 1076] 
        sf_low_res_diagonal_pairs = [(101, 1076 ), (425, 750)]

        glu_residues = [98,422,747,1073]
        glu_residues_pairs = [(98,1073), (422,747)]

        asn_residues = [130,454,779,1105 ]
        asn_residues_pairs = [(130,1105), (454,779)]

        pip_resids = [1301, 1302, 1303, 1304]

        gly_residues = [126, 450, 775, 1101]

        frame_duration = 0.075

        subunit_groups = {
            'A': [422, 454],  # GLU and ASN/ASP of subunit A
            'B': [98, 130],
            'C': [747, 779],
            'D': [1073, 1105]
        }
        sf_residues = [100, 424, 749, 1075]  

        start_frame = 0
        # start_frame = 2500
        # end_frame = 1000
        # end_frame = 1250
        end_frame = 6799
        # end_frame = 4300

        # pip_mapping = {
        #     1301: "G1",
        #     1302: "2c",
        #     1303: "2a",
        #     1304: "2b"
        # }

        top_file =    Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_E141Q/RUN{run_number}/com.prmtop")
        traj_file =   Path(f"/media/ziyue/328bfc27-c1a6-4fce-9199-95c389ecd48d/Konstantina/jaguar/GIRK12_E141Q/RUN{run_number}/protein.nc")
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
    # # start_frame = 6799

    end_frame = len(u.trajectory) - 1
    # end_frame = 3000
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
            sf_low_res_residues, sf_low_res_diagonal_pairs,
            upper_gl_residues, upper_gl_diagonal_pairs,
            lower_gl_residues, lower_gl_diagonal_pairs
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
            upper_gl_residues=upper_gl_residues,
            upper_gl_diagonal_pairs=upper_gl_diagonal_pairs,
            lower_gl_residues=lower_gl_residues,
            lower_gl_diagonal_pairs=lower_gl_diagonal_pairs,
            results_dir=results_dir,
            count_ions=True
        )
        
        analyzer.run_analysis()
        analyzer.print_results()
        
        # SAVE STATE for future runs
        save_analyzer_state(analyzer, results_dir)
    
    from analysis.end2_utils import get_end2_all_frames, get_end3_all_frames

    end2_all = get_end2_all_frames(results_dir, analyzer.permeation_events)
    end3_all = get_end3_all_frames(results_dir, analyzer.permeation_events)

    from analysis.pip_binding_quality import run_pip_binding_quality

    run_pip_binding_quality(
        u            = u,
        results_dir  = results_dir,
        channel_type = args.channel_type,
        pip_resids   = pip_resids,
        pip_mapping  = pip_mapping,
        start_frame  = start_frame,
        end_frame    = end_frame,
        cutoff       = 4.0,
        sb_cutoff    = 4.0,
        min_event    = 5,
        force_rerun  = False,
    )

    from analysis.end2_contact_force import run_end2_contact_force

    for ev in ("end_2", "end_3"):
        run_end2_contact_force(universe=u, permeation_events=analyzer.permeation_events,
                        results_dir=results_dir, channel_type=channel_type,
                        frame_duration=frame_duration, run=run_number, event=ev,
                        force_rerun=True)

    from analysis.cavity_mobility import run_cavity_mobility

    run_cavity_mobility(
        universe          = u,
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = channel_type,
        frame_duration    = frame_duration,
        start_frame       = start_frame,
        end_frame         = end_frame,
        run               = run_number,
        glu_residues      = glu_residues,
        asn_residues      = asn_residues,
        hbc_residues      = hbc_residues,
        force_rerun       = False,
    )

    from analysis.cavity_water_regions import run_cavity_water_regions

    run_cavity_water_regions(
        universe          = u,
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = channel_type,
        frame_duration    = frame_duration,
        start_frame       = start_frame,
        end_frame         = end_frame,
        radius            = 12.0,
        run               = run_number,
        glu_residues      = glu_residues,
        asn_residues      = asn_residues,
        hbc_residues      = hbc_residues,
        force_rerun       = False,
    )

    from analysis.ion_hydration_regions import run_ion_hydration_regions

    run_ion_hydration_regions(
        universe          = u,
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = channel_type,
        frame_duration    = frame_duration,
        start_frame       = start_frame,
        end_frame         = end_frame,
        shell             = 3.5,
        run               = run_number,
        glu_residues      = glu_residues,
        asn_residues      = asn_residues,
        hbc_residues      = hbc_residues,
        force_rerun       = False,
    )

    from analysis.permeation_ion_table import run_permeation_ion_table

    run_permeation_ion_table(
        universe          = u,
        permeation_events = analyzer.permeation_events,
        glu_residues      = glu_residues,
        asn_residues      = asn_residues,
        results_dir       = results_dir,
        channel_type      = channel_type,
        frame_duration    = frame_duration,
        start_frame       = start_frame,
        end_frame         = end_frame,
        cutoff            = 3.5,
        run               = run_number,
        force_rerun       = False,
    )

    from analysis.permeation_ion_stats import run_permeation_ion_stats

    run_permeation_ion_stats(
        results_dir    = results_dir,
        channel_type   = channel_type,
        frame_duration = frame_duration,
        run            = run_number,
        force_rerun    = False,
    )

    from analysis.permeation_correlation import run_permeation_correlation

    run_permeation_correlation(
        results_dir    = results_dir,
        channel_type   = channel_type,
        frame_duration = frame_duration,
        run            = run_number,
        force_rerun    = False,
    )

    from analysis.permeation_ion_combinations import run_permeation_combinations

    run_permeation_combinations(
        results_dir  = results_dir,
        channel_type = channel_type,
        run          = run_number,
        unit         = "frames",
        force_rerun  = False,
    )

    from analysis.glu_asn_cavity_split_analysis import run_cavity_split_analysis

    run_cavity_split_analysis(
        u                 = u,
        results_dir       = results_dir,
        channel_type      = args.channel_type,
        permeation_events = analyzer.permeation_events,
        frame_duration    = frame_duration,
        glu_resids      = glu_residues,
        asn_resids      = asn_residues,
        force_rerun       = True,
    )

    # from analysis.pip_199_64_permeation_corr import build_hbc_permeation_percentage_table

    # triples = [
    #     (1301, 471, 336),
    #     (1302, 146, 10),
    #     (1303, 1120, 985),
    #     (1304, 795, 660),
    # ]

    # build_hbc_permeation_percentage_table(
    #     u              = u,
    #     triples        = triples,
    #     pip_mapping    = pip_mapping,
    #     results_dir    = results_dir,
    #     run_number     = run_number,
    #     channel_type   = args.channel_type,
    #     hbc_frames     = end3_all,
    #     start_frame    = start_frame,
    #     end_frame      = end_frame,
    #     phosphate      = "P5",
    #     cutoff         = 4.0,
    # )
    # from analysis.glu_asn_geometry_analysis import run_glu_asn_geometry_analysis

    # run_glu_asn_geometry_analysis(
    #     u                   = u,
    #     glu_residues        = glu_residues,
    #     glu_residues_pairs  = glu_residues_pairs,
    #     asn_residues        = asn_residues,
    #     asn_residues_pairs  = asn_residues_pairs,
    #     sf_low_res_residues = sf_low_res_residues,
    #     hbc_residues        = hbc_residues,
    #     results_dir         = results_dir,
    #     channel_type        = args.channel_type,
    #     start_frame         = start_frame,
    #     end_frame           = end_frame,
    #     frame_duration      = frame_duration,
    # )

    from analysis.cavity_refilling_analysis import run_cavity_refilling_analysis

    run_cavity_refilling_analysis(
        permeation_events = analyzer.permeation_events,
        results_dir       = results_dir,
        channel_type      = args.channel_type,
        frame_duration    = frame_duration,
        end_frame         = end_frame,   # ← add this
        force_rerun       = True,
    )

    # # only these channel types have a bound ligand
    # # only these channel types have a bound ligand — each may use a different resname
    # LIGAND_RESNAME_BY_CHANNEL = {
    #     "G12_ML":  "LIG",
    #     "G12_GAT": "GAT",
    # }
    # lig_resname = LIGAND_RESNAME_BY_CHANNEL.get(args.channel_type, None)

    # from analysis.ion_residue_force_analysis import run_ion_residue_force_analysis

    # run_ion_residue_force_analysis(
    #     u                   = u,
    #     permeation_events   = analyzer.permeation_events,
    #     glu_residues        = glu_residues,
    #     asn_residues        = asn_residues,
    #     sf_low_res_residues = sf_low_res_residues,
    #     hbc_residues        = hbc_residues,
    #     results_dir         = results_dir,
    #     channel_type        = args.channel_type,
    #     lig_resname         = lig_resname,
    # )

#============== REMOVE
    # from analysis.network_gating_distance_analysis import run_network_distance_analysis

    # run_network_distance_analysis(
    #     u=u,
    #     res_313=res_313,
    #     res_315=res_315,
    #     res_233=res_233,
    #     res_199=res_199,
    #     results_dir=results_dir,
    #     channel_type=args.channel_type,
    #     run_number=run_number,
    #     start_frame=start_frame,
    #     end_frame=end_frame,
    #     smooth_window=50,
    #     threshold=4.0,
    #     force_rerun=False,
    # )

    # from analysis.network_199_315_324_233 import run_network_distance_analysis

    # run_network_distance_analysis(
    #     u=u,
    #     res_324=res_324,
    #     res_315=res_315,
    #     res_233=res_233,
    #     res_199=res_199,
    #     results_dir=results_dir,
    #     channel_type=args.channel_type,
    #     run_number=run_number,
    #     start_frame=start_frame,
    #     end_frame=end_frame,
    #     smooth_window=50,
    #     threshold=4.0,
    #     force_rerun=False,
    # )

    # from analysis.relay_state_classification import run_relay_state_classification

    # run_relay_state_classification(
    #     results_dir=results_dir,
    #     channel_type=args.channel_type,
    #     run_number=run_number,
    #     mode="functional_atoms",
    #     threshold=4.0,
    #     min_occupancy_pct=2.0,
    # )

    # from analysis.residence_time_stratified_analysis import run_residence_time_stratified_analysis


    # run_residence_time_stratified_analysis(
    #     permeation_events = analyzer.permeation_events,
    #     results_dir       = results_dir,
    #     channel_type      = args.channel_type,
    #     frame_duration    = frame_duration,
    #     force_rerun       = True,   # or True/False, matching however you pass this elsewhere
    # )

    # from analysis.pmf_seed_frame_finder import run_pmf_seed_frame_selection

    # run_pmf_seed_frame_selection(
    #     u              = u,
    #     upper1         = upper1,
    #     lower1         = lower1,
    #     results_dir    = results_dir,
    #     channel_type   = args.channel_type,
    #     frame_duration = frame_duration,
    #     force_rerun    = True,
    # )
    # from analysis.cavity_environment_analysis import run_cavity_environment_analysis
    # do_cavity_environment = True
    # force_rerun_cavity_env = True

    # if do_cavity_environment:
    #     run_cavity_environment_analysis(
    #         u                     = u,
    #         results_dir           = results_dir,
    #         gate_definitions      = {
    #             "glu": {"residues": glu_residues,
    #                     "diagonal_pairs": glu_residues_pairs},
    #             "asn": {"residues": asn_residues,
    #                     "diagonal_pairs": asn_residues_pairs},
    #             "hbc": {"residues": hbc_residues,
    #                     "diagonal_pairs": hbc_diagonal_pairs},
    #             "sf":  {"residues": sf_low_res_residues,
    #                     "diagonal_pairs": sf_low_res_diagonal_pairs},
    #         },
    #         cavity_upper_residues = sf_residues,
    #         cavity_lower_residues = hbc_residues,
    #         permeation_events     = analyzer.permeation_events,
    #         permeation_stage      = "end_3",
    #         start_frame           = start_frame,
    #         end_frame             = end_frame,
    #         frame_duration        = frame_duration,
    #         channel_type          = args.channel_type,
    #         run_number            = run_number,
    #         force_rerun           = force_rerun_cavity_env,
    #     )
    #=============== PIP ANALYSIS =============================================

    # from analysis.pip_pose_analysis import run_pip_pose_analysis

    # run_pip_pose_analysis(
    #     u            = u,
    #     mapping_json = mapping_json,
    #     results_dir  = results_dir,
    #     start_frame  = start_frame,
    #     end_frame    = end_frame,
    #     channel_type = channel_type,
    #     end2_all     = end2_all,   # from get_end2_all_frames()
    #     end3_all     = end3_all,   # from get_end3_all_frames()
    # )

    # from analysis.lys90_cooccurrence_analysis import run_lys90_cooccurrence_analysis

    # run_lys90_cooccurrence_analysis(
    #     u            = u,
    #     mapping_json = mapping_json,
    #     results_dir  = results_dir,
    #     start_frame  = start_frame,
    #     end_frame    = end_frame,
    #     channel_type = channel_type,
    #     end2_all     = end2_all,
    #     end3_all     = end3_all,
    # )

    # from analysis.pip_pocket_geometry import run_pip_pocket_geometry

    # run_pip_pocket_geometry(
    #     u            = u,
    #     mapping_json = mapping_json,
    #     results_dir  = results_dir,
    #     start_frame  = start_frame,
    #     end_frame    = end_frame,
    #     channel_type = args.channel_type,
    # )

    # from analysis.pip_headgroup_distances import run_pip_headgroup_distances

    # run_pip_headgroup_distances(
    #     u            = u,
    #     mapping_json = mapping_json,
    #     results_dir  = results_dir,
    #     start_frame  = start_frame,
    #     end_frame    = end_frame,
    #     channel_type = args.channel_type,
    #     end2_all     = end2_all,
    #     end3_all     = end3_all,
    # )

    # from analysis.pip_proximity_analysis import run_pip_proximity_analysis

    
 
    # run_pip_proximity_analysis(
    #     u                = u,
    #     results_dir      = results_dir,
    #     start_frame      = start_frame,
    #     end_frame        = end_frame,
    #     channel_type     = args.channel_type,
    #     pip_resids       = pip_resids,
    #     pip_mapping      = pip_mapping,
    #     threshold        = 4.0,
    #     prescreen_factor = 5.0,
    #     smoothing_window = 20,
    #     end2_all         = end2_all,
    #     end3_all         = end3_all,
    # )

    # from analysis.pip_gate_correlation import run_pip_gate_correlation

    # run_pip_gate_correlation(
    #     results_dir      = results_dir,
    #     prescreen_factor = 5.0, # Used to find the pip_proximity folder — it looks for pip_proximity_5/ in your results directory
    #     min_r            = 0.2, # The minimum absolute Spearman correlation |r| to include in the filtered output files
    # )

    # from analysis.pip_rmsd_analysis import run_pip_rmsd_analysis

    
    # run_pip_rmsd_analysis(
    #     u            = u,
    #     results_dir  = results_dir,
    #     start_frame  = start_frame,
    #     end_frame    = end_frame,
    #     channel_type = args.channel_type,
    #     pip_mapping  = pip_mapping,
    # )

    # from analysis.residue_pip_p4p5_proximity import run_p4p5_proximity

    # run_p4p5_proximity(
    #     u             = u,
    #     pairs         = [(1301, 471), (1302, 146), (1303, 1120), (1304, 795)],
    #     results_dir   = results_dir,
    #     start_frame   = start_frame,
    #     end_frame     = end_frame,
    #     channel_type  = args.channel_type,
    #     cutoff        = 4.0,
    #     force=False,   # default — use cache if valid, else compute
    #     gate_filters  = [
    #         {
    #             "name": "hbc",
    #             "csv": results_dir / "hbc_distances.csv",
    #             "thresholds": [5, 6, 7, 8, 9, 10, 11, 12],
    #         },
    #         {
    #             "name": "upper_gl",
    #             "csv": results_dir / "upper_gl_distances.csv",
    #             "thresholds": [5, 6, 7, 8, 9, 10, 11, 12],
    #         },
    #     ],
    # )
    # from analysis.residue_pip_two_residues_proximity import run_phosphate_two_residue_proximity

    # triples = [
    #     (1301, 471, 336),   # PIP1301 <-> 199.2b / 64.2b
    #     (1302, 146, 10),    # PIP1302 <-> 188.G1 / ARG at G1 copy
    #     (1303, 1120, 985),  # PIP1303 <-> 199.2c / 64.2c
    #     (1304, 795, 660),   # PIP1304 <-> 199.2a / 64.2a
    # ]

    # gate_filters = [
    #     {
    #         "name": "hbc",
    #         "csv": results_dir / "hbc_distances.csv",
    #         "thresholds": [5, 6, 7, 8, 9, 10, 11, 12],
    #     },
    #     {
    #         "name": "upper_gl",
    #         "csv": results_dir / "upper_gl_distances.csv",
    #         "thresholds": [5, 6, 7, 8, 9, 10, 11, 12],
    #     },
    # ]

    # # P5 vs residues 199/64
    # run_phosphate_two_residue_proximity(
    #     u                  = u,
    #     triples            = triples,
    #     results_dir        = results_dir,
    #     start_frame        = start_frame,
    #     end_frame          = end_frame,
    #     channel_type       = args.channel_type,
    #     phosphate          = "P5",
    #     cutoff             = 4.0,
    #     res_a_label        = "199",
    #     res_b_label        = "64",
    #     res_a_aa           = "LYS",
    #     res_b_aa           = "LYS",
    #     res_b_aa_overrides = {10: "ARG"},   # G1 copy of residue 64 is an Arg
    #     force              = False,
    #     gate_filters       = gate_filters,
    # )

    # # P4 vs residues 199/64 — same triples, written to a separate folder
    # run_phosphate_two_residue_proximity(
    #     u                  = u,
    #     triples            = triples,
    #     results_dir        = results_dir,
    #     start_frame        = start_frame,
    #     end_frame          = end_frame,
    #     channel_type       = args.channel_type,
    #     phosphate          = "P4",
    #     cutoff             = 4.0,
    #     res_a_label        = "199",
    #     res_b_label        = "64",
    #     res_a_aa           = "LYS",
    #     res_b_aa           = "LYS",
    #     res_b_aa_overrides = {10: "ARG"},
    #     force              = False,
    #     gate_filters       = gate_filters,
    # )
    #=============== PIP ANALYSIS END =============================================

    # from analysis.hbc_open_pip_analysis import run_hbc_open_pip_analysis

    # run_hbc_open_pip_analysis(
    #     results_dir      = results_dir,
    #     hbc_threshold    = 5.0,
    #     contact_cutoff   = 4.0,
    #     prescreen_factor = 5.0,
    # )

    # ===================== DCCM analysis =========================================
    # from analysis.dccm_pathway_analysis import run_dccm_pathway_analysis

    # run_dccm_pathway_analysis(
    #     u              = u,
    #     results_dir    = results_dir,
    #     channel_type   = args.channel_type,
    #     start_frame    = 1000,
    #     end_frame      = 6800,
    #     residue_pairs  = [
    #         ("192.2b", "192.2a"),
    #         ("192.2b", "192.2c"),
    #         ("192.2b", "181.G1"),

    #         ("90.2b", "192.2b"),

    #         ("197.2b", "192.2b"),
    #         ("197.2b", "192.2a"),
    #         ("197.2b", "192.2c"),
    #         ("197.2b", "181.G1"),

    #         ("194.2a", "192.2b"),
    #         ("194.2a", "192.2a"),
    #         ("194.2a", "192.2c"),
    #         ("194.2a", "181.G1"),
    #     ],
    #     corr_threshold = 0.7,
    #     dist_cutoff    = 6.0,
    # )

    # ===================== Kink analysis =========================================
    # from analysis.helix_bending_analysis import run_helix_bending_analysis

    # run_helix_bending_analysis(
    #     u                = u,
    #     results_dir      = results_dir,
    #     channel_type     = args.channel_type,
    #     gly_residues     = gly_residues,
    #     start_frame      = start_frame,
    #     end_frame        = end_frame,
    #     window           = 1,
    #     smoothing_window = 100,
    # )
    # # ============================ Neighboring Residues ================================
#     from analysis.residue_neighbors_analysis import run_residue_neighbors_analysis

# # G12 [262,587,911,1236], #[1122, 473, 797, 148],               #gln[1118, 469, 793, 144],      # [1122, 473, 797, 148], 
# # G2 gln
# # g12_gat [143,467,792,1118]
#     run_residue_neighbors_analysis(
#         u                = u,
#         results_dir      = results_dir,
#         channel_type     = args.channel_type,
#         resids           =  [471, 146, 1120, 795], #199 g12
#         start_frame      = start_frame,
#         end_frame        = end_frame,
#         threshold        = 4.0,
#         prescreen_factor = 3.0,
#         smoothing_window = 100,
#         min_frames       = 500,  # must be < 4Å for at least 500 frames (~7% of trajectory)
#     )

#############################################

    # from analysis.hbond_occupancy_gln197_lys199 import run_hbond_occupancy_gln197_lys199

    # run_hbond_occupancy_gln197_lys199(results_dir=results_dir)

    # # ======================= Hbonds when residue is close to P5 vs when it is far =========================
    # from analysis.pip_bound_free_hbond_analysis import run_pip_bound_free_hbond_analysis

    # run_pip_bound_free_hbond_analysis(
    #     results_dir      = results_dir,
    #     resid_query      = "197",
    #     pip_resids       = [1301, 1302, 1303, 1304],
    #     pip_mapping      = pip_mapping,
    #     threshold        = 4.0,
    #     prescreen_factor = 5.0,
    # )

    # ============================ chi_angle_analysis ==========================
    # from analysis.chi_angle_analysis import run_chi_angle_analysis

    # run_chi_angle_analysis(
    #     u                = u,
    #     results_dir      = results_dir,
    #     channel_type     = args.channel_type,
    #     resids           = [469],   # GLN197.2b
    #     start_frame      = start_frame,
    #     end_frame        = end_frame,
    #     smoothing_window = 100,
    # )

#  #  # ==================== Protein Rmsd ===========================
#     from analysis.protein_rmsd_analysis import run_protein_rmsd_analysis

#     run_protein_rmsd_analysis(
#         u                = u,
#         results_dir      = results_dir,
#         channel_type     = args.channel_type,
#         start_frame      = start_frame,
#         end_frame        = end_frame,
#         smoothing_window = 100,
#     )

    # ==================== Hbond detection ============================

    # from analysis.hbond_pair_analysis import run_hbond_pair_analysis

    # run_hbond_pair_analysis(
    #     u                = u,
    #     results_dir      = results_dir,
    #     channel_type     = args.channel_type,
    #     residue_pairs    = [(469, 591)],  # GLN197.2b ↔ MET319.2b
    #     start_frame      = start_frame,
    #     end_frame        = end_frame,
    #     smoothing_window = 100,
    #     hbond_dist       = 4,
    #     hbond_angle      = 120.0,
    # )

    # from analysis.hbc_rmsf_analysis import run_hbc_rmsf_analysis

    # run_hbc_rmsf_analysis(
    #     u              = u,
    #     hbc_residues   = hbc_residues,
    #     results_dir    = results_dir,
    #     channel_type   = args.channel_type,
    #     start_frame    = start_frame,
    #     end_frame      = end_frame,
    #     frame_duration = frame_duration,
    #     smoothing_window = 100,
    # )

    # from analysis.untracked_cavity_ions import run_untracked_cavity_ions

    # run_untracked_cavity_ions(
    #     u           = u,
    #     channel2    = ch2,
    #     results_dir = results_dir,
    #     start_frame = start_frame,
    #     end_frame   = end_frame,
    #     frame_duration = frame_duration,
    # )

    # from analysis.ion_ion_permeation_analysis import run_ion_ion_permeation_analysis

    # run_ion_ion_permeation_analysis(
    #     u                   = u,
    #     permeation_events   = analyzer.permeation_events,
    #     channel1            = ch1,
    #     channel2            = ch2,
    #     sf_low_res_residues = sf_low_res_residues,
    #     hbc_residues        = hbc_residues,
    #     results_dir         = results_dir,
    #     channel_type        = args.channel_type,
    # )

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
    #     results_dir=f"{results_dir}/cavity_occupancy",
    #     channel3=ch3,            # add this
    #     hbc_residues=hbc_residues,  # add this
    #     force_rerun=True,
    # )

    # from analysis.cavity_occupancy_analysis import run_cavity_occupancy_analysis
    # run_cavity_occupancy_analysis(
    #     u, ch2, analyzer.permeation_events,
    #     asn_residues, glu_residues, subunit_groups,
    #     channel_type=args.channel_type,
    #     results_dir=results_dir,   # base RUN*/ dir — subfolder created automatically
    #     start_frame=start_frame, end_frame=end_frame, frame_duration=frame_duration,
    #     channel3=ch3, hbc_residues=hbc_residues,
    #     channel1=ch1, sf_residues=sf_residues,
    # )

    # # from analysis.permeation_stage_times import run_permeation_stage_times_analysis
    # # run_permeation_stage_times_analysis(
    # #     permeation_events=analyzer.permeation_events,
    # #     results_dir=results_dir,
    # #     channel_type=args.channel_type,
    # #     frame_duration=frame_duration,
    # # )

    # from analysis.ion_residue_force_analysis import run_ion_residue_force_analysis

    # run_ion_residue_force_analysis(
    #     u                   = u,
    #     permeation_events   = analyzer.permeation_events,
    #     glu_residues        = glu_residues,
    #     asn_residues        = asn_residues,
    #     sf_low_res_residues = sf_low_res_residues,
    #     hbc_residues        = hbc_residues,
    #     results_dir         = results_dir,
    #     channel_type        = args.channel_type,
    # )

    # from analysis.ion_axial_motion import run_ion_cavity_analysis

    # run_ion_cavity_analysis(
    #     u                          = u,
    #     permeation_events          = analyzer.permeation_events,
    #     sf_low_res_residues        = sf_low_res_residues,
    #     hbc_residues               = hbc_residues,
    #     glu_residues               = glu_residues,
    #     asn_residues               = asn_residues,
    #     results_dir                = results_dir,
    #     frame_duration             = frame_duration,
    #     channel_type               = args.channel_type,
    #     hbc_oscillation_threshold  = 10.0,   # change this to adjust the filter
    # )

    # from analysis.cavity_water_count import run_cavity_water_count

    # run_cavity_water_count(
    #     u=u,
    #     channel2=ch2,
    #     permeation_events=analyzer.permeation_events,
    #     results_dir=results_dir,
    #     start_frame=start_frame,
    #     end_frame=end_frame,
    # )

    # from analysis.cavity_water_residence import run_cavity_water_residence

    # run_cavity_water_residence(
    #     u=u,
    #     channel2=ch2,
    #     results_dir=results_dir,
    #     start_frame=start_frame,
    #     end_frame=end_frame,
    #     water_radius=12.0,
    # )

    # from analysis.asn_axis_analysis import run_asn_axis_analysis

    # run_asn_axis_analysis(u, sf_residues, hbc_residues, asn_residues,
    #                   results_dir, channel_type=args.channel_type,
    #                   start_frame=start_frame, end_frame=end_frame,
    #                   use_ca=False)


    # from analysis.ion_force_correlation_analysis import run_force_correlation_analysis

    # force_analyzer = run_force_correlation_analysis(
    #     universe=u, channel1=ch1, channel2=ch2,
    #     permeation_events=analyzer.permeation_events,
    #     asn_residues=asn_residues, glu_residues=glu_residues,
    #     results_dir=results_dir / "force_correlation_analysis"
    # )

    # from analysis.dihedral_analysis_module import run_dihedral_analysis

    # # In your main(), after your ion permeation analysis:
    # dihedral_results, peak_frames = run_dihedral_analysis(
    #     u=u,                           # Your existing Universe
    #     channel_type=args.channel_type,        # Or args.channel_type
    #     results_dir=results_dir/ "dihedral_analysis"       # Your existing results directory
    # )

    # if args.channel_type in ["G12_ML", "G12_GAT"]:
    #     from analysis.ligand_distance_analysis import run_ligand_distance_analysis
    #     run_ligand_distance_analysis(
    #         u                  = u,
    #         results_dir        = results_dir,
    #         start_frame        = start_frame,
    #         end_frame          = end_frame,
    #         channel_type       = args.channel_type,
    #         frame_duration     = frame_duration,
    #         permeation_events  = analyzer.permeation_events,  # ← add this
    #     )

    # if args.channel_type in ["G12_ML", "G12_GAT"]:
    #     from analysis.ligand_cavity_correlation_analysis import (
    #         run_ligand_cavity_correlation_analysis
    #     )
    #     run_ligand_cavity_correlation_analysis(
    #         results_dir       = results_dir,
    #         permeation_events = analyzer.permeation_events,
    #         channel_type      = args.channel_type,
    #     )

#     from analysis.ion_proximity_analysis import run_ion_proximity_analysis

#     run_ion_proximity_analysis(
#         u=u,
#         results_dir=results_dir,
#         glu_residues=glu_residues,
#         asn_residues=asn_residues,
#         channel_type=args.channel_type,
#         start_frame=start_frame,
#         end_frame=end_frame,
#         threshold=4
#     )

# # ========== GEOMETRY ANALYSIS (AFTER MAIN ANALYSIS) ==========
#     if len(analyzer.permeation_events) > 0:
#         print("\n" + "="*80)
#         print("RUNNING RESIDUE GEOMETRY ANALYSIS")
#         print("="*80)
        
#         # Import the geometry analyzer
#         from analysis.analyze_angles import run_angle_analysis
        
        
#         all_residues_for_angles = asn_residues + glu_residues  # Add any others you want
    
#         run_angle_analysis(
#             topology_file=top_file,
#             trajectory_file=traj_file,
#             results_dir=results_dir,
#             residue_ids=all_residues_for_angles,
#             start_frame=start_frame,
#             end_frame=end_frame,
#             permeation_stage='end_2'  # Mark permeation events at end_2
#         )
        
#         print("\n" + "="*80)
#         print("GEOMETRY ANALYSIS COMPLETE!")
#         print("="*80)

#     from analysis.permeation_frame_significance import test_permeation_frame_significance

#         # The function will extract them from start_2 to end_2-1 for each ion
#     significance_results = test_permeation_frame_significance(
#         universe=u,
#         channel2=ch2,
#         all_residues=asn_residues + glu_residues,
#         permeation_events=analyzer.permeation_events,  # Must have 'ion_id', 'start_2', 'end_2'
#         results_dir=results_dir / "significance_analysis_end1",
#         channel_type=channel_type,
#         n_bootstrap=1000,
#         sample_size=50,
#         min_frames_method2=100,
#         use_end1=True  # Use end_1 instead of start_2!
#     )
    
#     print("\n✓ Statistical significance analysis complete!")

#     from analysis.occupancy_alignment import run_comprehensive_end2_analysis

#         # ============= COMPREHENSIVE END_2 ANALYSIS =============
#     if len(analyzer.permeation_events) > 0:
#         print("\n" + "="*80)
#         print("Running Comprehensive end_2 Analysis...")
#         print("="*80)

#         threshold=3
#         comp_analyzer = run_comprehensive_end2_analysis(
#             universe=u,
#             channel1=ch1,
#             channel2=ch2,
#             permeation_events=analyzer.permeation_events,
#             asn_residues=asn_residues,
#             glu_residues=glu_residues,
#             subunit_groups=subunit_groups,
#             channel_type=channel_type,
#             threshold=threshold,  # Distance threshold for "close to residue"
#             results_dir=f"{results_dir}/occupancy_alignment/{threshold}"  # <-- Separate folder
#         )
        
#         print("\nComprehensive end_2 Analysis Complete!")



#     if analyzer.count_ions and len(analyzer.permeation_events) > 0:
#        sf_analyzer = run_sf_alignment_analysis(
#            universe=u,
#            channel1=ch1,
#            channel2=ch2,
#            permeation_events=analyzer.permeation_events,
#            asn_residues=asn_residues,
#            glu_residues=glu_residues,
#            results_dir=results_dir
#        )


#     from analysis.closest_residue_analysis import run_closest_residue_analysis

#     threshold=3
#     closest_res_analyzer = run_closest_residue_analysis(
#         universe=u,
#         permeation_events=analyzer.permeation_events,
#         asn_residues=asn_residues,
#         glu_residues=glu_residues,
#         channel_type=channel_type,
#         threshold=threshold,
#         analysis_start_frame=start_frame,
#         residue_types=['ASN'],  # Only ASN
#         results_dir=f"{results_dir}/closest_res/{threshold}"
#     )  

#     from analysis.longest_duration_analysis import run_last_duration_analysis
#     threshold = 2.5
#     min_duration = 3

#     # Run 1: RESIDUE-LEVEL
#     run_last_duration_analysis(
#         universe=u,
#         permeation_events=analyzer.permeation_events,
#         asn_residues=asn_residues,
#         glu_residues=glu_residues,
#         channel_type=channel_type,
#         threshold=threshold,
#         min_duration=min_duration,
#         analysis_start_frame=start_frame,
#         residue_types=['ASN', 'GLU'],
#         subunit_groups=None,  # <-- Residue mode
#         results_dir=f"{results_dir}/duration_residue/{threshold}_{min_duration}"
#     )

#     # Run 2: SUBUNIT-LEVEL
#     run_last_duration_analysis(
#         universe=u,
#         permeation_events=analyzer.permeation_events,
#         asn_residues=asn_residues,
#         glu_residues=glu_residues,
#         channel_type=channel_type,
#         threshold=threshold,
#         min_duration=min_duration,
#         analysis_start_frame=start_frame,
#         residue_types=['ASN', 'GLU'],
#         subunit_groups=subunit_groups,  # <-- Subunit mode
#         results_dir=f"{results_dir}/duration_subunit/{threshold}_{min_duration}"
#     )


#     #  NEW: Add proximity analysis
#     print("\n" + "="*70)
#     print("RUNNING RESIDUE PROXIMITY ANALYSIS")
#     print("="*70)
    
#     proximity_analyzer = ResidueProximityAnalysis(
#         universe=u,
#         ion_selection="resname K+ K",
#         glu_residues=glu_residues,
#         asn_residues=asn_residues,
#         start_frame=start_frame,
#         end_frame=end_frame,
#         results_dir=results_dir,
#         channel_type=channel_type,  # NEW: pass the channel_type
#         cutoff=3.0
#     )
    
#     proximity_analyzer.run_analysis()
#     proximity_analyzer.print_results()
#     proximity_analyzer.save_results()
    
#     # Generate timeline plots
#     print("\nGenerating timeline plots...")
#     proximity_analyzer.plot_individual_glu_timelines()
#     proximity_analyzer.plot_individual_asn_timelines()
#     proximity_analyzer.plot_all_residues_combined()
#     proximity_analyzer.plot_aggregate_comparison()
    
#     # Generate bar chart plots
#     print("\nGenerating bar chart plots...")
#     proximity_analyzer.plot_individual_residue_bar_chart()
#     proximity_analyzer.plot_all_residues_bar_chart()
#     proximity_analyzer.plot_aggregate_bar_chart()
#     proximity_analyzer.plot_percentage_comparison_bar_chart()

if __name__ == "__main__":
    main()
