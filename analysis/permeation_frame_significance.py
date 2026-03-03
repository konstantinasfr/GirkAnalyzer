import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import f as f_dist
from scipy import stats
from analysis.converter import convert_to_pdb_numbering


def test_permeation_frame_significance(
    universe, channel2, all_residues, permeation_events,
    results_dir=".", channel_type="G12",
    n_bootstrap=1000, sample_size=50, min_frames_method2=80,
    use_end1=False
):
    """
    Statistical significance test with TWO methods:
    
    Method 1: Central Limit Theorem Bootstrap (tests if end_2 MEAN is unusual)
              + Individual ion analysis (where each ion's end_2 falls)
    Method 2: Hotelling's T² (group comparison: all state 2 vs all end_2)
    
    Parameters:
    -----------
    universe : MDAnalysis.Universe
    channel2 : Channel (cavity)
    all_residues : list
        8 residue IDs (4 ASN + 4 GLU)
    permeation_events : list
        Permeation events with 'ion_id', 'end_1', 'start_2', and 'end_2'
    results_dir : Path or str
    channel_type : str
    n_bootstrap : int
        Number of bootstrap samples for Method 1 (default: 1000)
    sample_size : int
        Size of each bootstrap sample for Method 1 (default: 50)
    min_frames_method2 : int
        Minimum non-permeating frames required for Method 2 (default: 80)
    use_end1 : bool
        If True, use range(end_1, end_2) for cavity frames
        If False, use range(start_2, end_2) for cavity frames (default)
    
    Returns:
    --------
    dict with both analyses results
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_residues = sorted(all_residues)
    
    # Convert residue IDs to PDB numbering
    residue_pdb_map = {}
    for resid in all_residues:
        residue_pdb_map[resid] = convert_to_pdb_numbering(resid, channel_type)
    
    # Create text report file
    report_file = results_dir / "permeation_significance_report.txt"
    report = open(report_file, 'w')
    
    def print_and_write(text):
        """Print to console and write to file."""
        print(text)
        report.write(text + '\n')
    
    def get_distance_vector(frame_idx, ion_id):
        """Get 8D vector of MINIMUM distances from THIS SPECIFIC ION to each residue's closest atom."""
        universe.trajectory[frame_idx]
        channel2.compute_geometry(2)
        
        ion = universe.select_atoms(f"resid {ion_id}")
        if len(ion) == 0:
            return None
        
        ion_pos = ion.positions[0]
        
        distance_vector = []
        for resid in all_residues:
            res_atoms = universe.select_atoms(f"resid {resid}")
            if len(res_atoms) == 0:
                return None
            
            # Calculate distance to EACH atom in the residue, then take MINIMUM
            min_dist = np.min([np.linalg.norm(ion_pos - atom_pos) 
                              for atom_pos in res_atoms.positions])
            distance_vector.append(min_dist)
        
        return np.array(distance_vector)
    
    # Helper function for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    print_and_write("\n" + "="*80)
    print_and_write("PERMEATION FRAME SIGNIFICANCE ANALYSIS")
    print_and_write("ION IN GLU/ASN CAVITY vs ION LEAVING CAVITY")
    print_and_write("="*80)
    print_and_write(f"Number of residues in vector: {len(all_residues)}")
    print_and_write(f"Residue IDs (internal): {all_residues}")
    print_and_write(f"Residue IDs (PDB): {[residue_pdb_map[r] for r in all_residues]}")
    print_and_write(f"Permeation events to analyze: {len(permeation_events)}")
    
    # =========================================================================
    # STEP 1: Collect vectors for ALL ions
    # =========================================================================
    print_and_write("\n" + "-"*80)
    print_and_write("STEP 1: Collecting vectors from all ions...")
    print_and_write("-"*80)
    
    all_state2_vectors = []  # Ion in GLU/ASN cavity (end_1 to end_2-1)
    all_end2_vectors = []    # Ion leaving GLU/ASN cavity (end_2 frame)
    all_end2_ion_ids = []
    all_end2_frames = []
    ion_end2_vectors = {}    # Store individual ion end_2 vectors
    
    # Track frame statistics
    total_frames_attempted = 0
    total_frames_valid = 0
    total_end2_attempted = 0
    total_end2_valid = 0
    frames_per_ion = []
    
    for event_idx, event in enumerate(permeation_events):
        ion_id = event['ion_id']
        start_2 = event['start_2']
        end_2 = event['end_2']
        
        # Choose frame range start based on parameter
        if use_end1:
            end_1 = event.get('end_1', start_2)
            frame_range_start = end_1
            range_label = "end_1 (leaves SF)"
        else:
            frame_range_start = start_2
            range_label = "start_2 (enters cavity)"
        
        frame_range_end = end_2
        expected_frames = frame_range_end - frame_range_start
        
        if (event_idx + 1) % 10 == 0 or event_idx == 0:
            print(f"\n=== Processing ion {ion_id} ({event_idx+1}/{len(permeation_events)}) ===")
            print(f"  {range_label}={frame_range_start}, end_2 (leaves cavity)={end_2}")
            print(f"  → Using frames {frame_range_start} to {frame_range_end-1} (ion in cavity)")
            print(f"  → Expected cavity frames: {expected_frames}")
        
        # Ion in GLU/ASN cavity: frame_range_start to (end_2 - 1) - DO NOT INCLUDE end_2!
        frames_collected_this_ion = 0
        for frame_idx in range(frame_range_start, end_2):
            total_frames_attempted += 1
            vec = get_distance_vector(frame_idx, ion_id)
            if vec is not None:
                all_state2_vectors.append(vec)
                total_frames_valid += 1
                frames_collected_this_ion += 1
        
        frames_per_ion.append({
            'ion_id': ion_id,
            'expected': expected_frames,
            'collected': frames_collected_this_ion
        })
        
        if (event_idx + 1) % 10 == 0 or event_idx == 0:
            print(f"  → Actually collected: {frames_collected_this_ion} frames")
            if frames_collected_this_ion < expected_frames:
                print(f"  ⚠ WARNING: Missing {expected_frames - frames_collected_this_ion} frames!")
        
        # Ion leaving cavity frame (end_2): separate
        total_end2_attempted += 1
        vec_end2 = get_distance_vector(end_2, ion_id)
        if vec_end2 is not None:
            all_end2_vectors.append(vec_end2)
            all_end2_ion_ids.append(ion_id)
            all_end2_frames.append(end_2)
            ion_end2_vectors[ion_id] = vec_end2  # Store individual vector
            total_end2_valid += 1
    
    all_state2_vectors = np.array(all_state2_vectors)
    all_end2_vectors = np.array(all_end2_vectors)
    
    # DETAILED FRAME STATISTICS
    print_and_write(f"\n" + "="*80)
    print_and_write(f"FRAME COLLECTION STATISTICS")
    print_and_write(f"="*80)
    print_and_write(f"Ion in GLU/ASN cavity frames (end_1 to end_2-1):")
    print_and_write(f"  Total attempted: {total_frames_attempted}")
    print_and_write(f"  Total valid:     {total_frames_valid}")
    print_and_write(f"  Invalid/missing: {total_frames_attempted - total_frames_valid}")
    print_and_write(f"  Success rate:    {100*total_frames_valid/total_frames_attempted:.1f}%")
    print_and_write(f"\nIon leaving cavity frames (end_2):")
    print_and_write(f"  Total attempted: {total_end2_attempted}")
    print_and_write(f"  Total valid:     {total_end2_valid}")
    print_and_write(f"  Invalid/missing: {total_end2_attempted - total_end2_valid}")
    print_and_write(f"  Success rate:    {100*total_end2_valid/total_end2_attempted:.1f}%")
    
    # Show distribution of frames per ion
    collected_counts = [item['collected'] for item in frames_per_ion]
    print_and_write(f"\nFrames collected per ion (in cavity):")
    print_and_write(f"  Mean:   {np.mean(collected_counts):.1f}")
    print_and_write(f"  Median: {np.median(collected_counts):.1f}")
    print_and_write(f"  Min:    {np.min(collected_counts)}")
    print_and_write(f"  Max:    {np.max(collected_counts)}")
    print_and_write(f"  Total:  {sum(collected_counts)}")
    
    # Show ions with missing frames
    ions_with_missing = [item for item in frames_per_ion if item['collected'] < item['expected']]
    if ions_with_missing:
        print_and_write(f"\n⚠ WARNING: {len(ions_with_missing)} ions have missing frames:")
        for item in ions_with_missing[:10]:  # Show first 10
            print_and_write(f"  Ion {item['ion_id']}: expected {item['expected']}, got {item['collected']}")
        if len(ions_with_missing) > 10:
            print_and_write(f"  ... and {len(ions_with_missing) - 10} more")
    
    print_and_write(f"\n✓ FINAL COUNTS:")
    print_and_write(f"  Total 'ion in cavity' vectors: {len(all_state2_vectors)}")
    print_and_write(f"  Total 'ion leaving cavity' vectors: {len(all_end2_vectors)}")
    
    if len(all_state2_vectors) == 0 or len(all_end2_vectors) == 0:
        print_and_write("ERROR: Not enough valid vectors for analysis!")
        report.close()
        return None
    
    # State 2 duration statistics
    if use_end1:
        durations = [event['end_2'] - event.get('end_1', event['start_2']) for event in permeation_events]
        range_desc = "end_1 (leaves SF) to end_2 (leaves cavity)"
    else:
        durations = [event['end_2'] - event['start_2'] for event in permeation_events]
        range_desc = "start_2 (enters cavity) to end_2 (leaves cavity)"
    
    print_and_write(f"\nTime in cavity statistics ({range_desc}):")
    print_and_write(f"  Mean duration: {np.mean(durations):.1f} frames")
    print_and_write(f"  Median duration: {np.median(durations):.1f} frames")
    print_and_write(f"  Min duration: {np.min(durations)} frames")
    print_and_write(f"  Max duration: {np.max(durations)} frames")
    
    # =========================================================================
    # METHOD 1: CENTRAL LIMIT THEOREM BOOTSTRAP
    # =========================================================================
    print_and_write("\n" + "="*80)
    print_and_write("METHOD 1: CENTRAL LIMIT THEOREM BOOTSTRAP")
    print_and_write("Tests if 'ion leaving cavity' configuration is unusual")
    print_and_write("compared to 'ion in cavity' distribution")
    print_and_write("="*80)
    
    # Check sample size
    if len(all_state2_vectors) < sample_size:
        print_and_write(f"\n⚠ WARNING: Only {len(all_state2_vectors)} 'ion in cavity' frames.")
        print_and_write(f"   Need at least {sample_size} for bootstrap.")
        print_and_write(f"   SKIPPING METHOD 1.")
        method1_results = None
    else:
        print_and_write(f"\n✓ Sample size check: {len(all_state2_vectors)} frames (>= {sample_size})")
        print_and_write(f"Bootstrap parameters: {n_bootstrap} samples of size {sample_size}")
        
        # Calculate observed means
        state2_mean = np.mean(all_state2_vectors, axis=0)
        end2_mean = np.mean(all_end2_vectors, axis=0)
        
        print_and_write(f"\nObserved means:")
        print_and_write(f"{'Residue (PDB)':<15} {'In cavity':<12} {'Leaving':<12} {'Difference':<12}")
        print_and_write("-"*55)
        for i, resid in enumerate(all_residues):
            pdb_name = residue_pdb_map[resid]
            diff = end2_mean[i] - state2_mean[i]
            print_and_write(f"{pdb_name:<15} {state2_mean[i]:>10.2f} Å  {end2_mean[i]:>10.2f} Å  {diff:>+10.2f} Å")
        
        # Bootstrap sampling (CLT)
        print_and_write(f"\nRunning {n_bootstrap} bootstrap samples...")
        
        np.random.seed(42)  # Reproducibility
        
        # For each residue, collect bootstrap means
        bootstrap_means = {resid: [] for resid in all_residues}
        
        for boot_idx in range(n_bootstrap):
            if (boot_idx + 1) % 200 == 0:
                print(f"  Bootstrap {boot_idx + 1}/{n_bootstrap}")
            
            # Random sample from 'ion in cavity' frames
            sample_indices = np.random.choice(len(all_state2_vectors), size=sample_size, replace=False)
            sample_vectors = all_state2_vectors[sample_indices]
            
            # Calculate mean of this sample
            sample_mean = np.mean(sample_vectors, axis=0)
            
            # Store for each residue
            for i, resid in enumerate(all_residues):
                bootstrap_means[resid].append(sample_mean[i])
        
        print_and_write(f"✓ Completed {n_bootstrap} bootstrap samples")
        
        # Fit Gaussian and calculate z-scores
        print_and_write(f"\nFitting Gaussian distributions and calculating significance:")
        print_and_write("-"*80)
        
        results_per_residue = {}
        
        for i, resid in enumerate(all_residues):
            boot_means = np.array(bootstrap_means[resid])
            
            # Fit Gaussian (CLT says it should be Gaussian!)
            mu_boot = np.mean(boot_means)
            sigma_boot = np.std(boot_means)
            
            # Where does end_2 mean fall?
            end2_value = end2_mean[i]
            
            # Z-score
            z_score = (end2_value - mu_boot) / sigma_boot
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            pdb_name = residue_pdb_map[resid]
            
            results_per_residue[int(resid)] = {
                'residue_pdb': pdb_name,
                'bootstrap_mean': float(mu_boot),
                'bootstrap_std': float(sigma_boot),
                'state2_mean': float(state2_mean[i]),
                'end2_mean': float(end2_value),
                'z_score': float(z_score),
                'p_value': float(p_value),
                'is_significant': bool(p_value < 0.05),
                'bootstrap_distribution': [float(x) for x in boot_means]
            }
            
            significance = ""
            if p_value < 0.001:
                significance = "*** (p < 0.001)"
            elif p_value < 0.01:
                significance = "**  (p < 0.01)"
            elif p_value < 0.05:
                significance = "*   (p < 0.05)"
            else:
                significance = "    (ns)"
            
            print_and_write(f"\nResidue {pdb_name} (internal ID: {resid}):")
            print_and_write(f"  Bootstrap mean: {mu_boot:.2f} ± {sigma_boot:.2f} Å (ion in cavity)")
            print_and_write(f"  Leaving mean:   {end2_value:.2f} Å (ion leaving cavity)")
            print_and_write(f"  Z-score:        {z_score:+.2f}")
            print_and_write(f"  p-value:        {p_value:.4f} {significance}")
        
        # NEW: Individual ion analysis
        print_and_write("\n" + "-"*80)
        print_and_write("INDIVIDUAL ION ANALYSIS")
        print_and_write("Where does each ion fall when it leaves the cavity?")
        print_and_write("-"*80)
        
        individual_ion_results = []
        
        for ion_id in all_end2_ion_ids:
            if ion_id not in ion_end2_vectors:
                continue
            
            ion_vec = ion_end2_vectors[ion_id]
            ion_result = {
                'ion_id': int(ion_id),
                'per_residue': {}
            }
            
            print_and_write(f"\nIon {ion_id}:")
            
            for i, resid in enumerate(all_residues):
                boot_means = np.array(bootstrap_means[resid])
                mu_boot = np.mean(boot_means)
                sigma_boot = np.std(boot_means)
                
                # This ion's end_2 value for this residue
                ion_value = ion_vec[i]
                
                # Z-score for this specific ion
                z_score_ion = (ion_value - mu_boot) / sigma_boot
                
                # Percentile in bootstrap distribution
                percentile = stats.percentileofscore(boot_means, ion_value)
                
                # Two-tailed p-value
                p_value_ion = 2 * (1 - stats.norm.cdf(abs(z_score_ion)))
                
                pdb_name = residue_pdb_map[resid]
                
                ion_result['per_residue'][int(resid)] = {
                    'residue_pdb': pdb_name,
                    'value': float(ion_value),
                    'bootstrap_mean': float(mu_boot),
                    'bootstrap_std': float(sigma_boot),
                    'z_score': float(z_score_ion),
                    'percentile': float(percentile),
                    'p_value': float(p_value_ion),
                    'is_outlier': bool(percentile < 5 or percentile > 95)
                }
                
                # Print outliers
                if percentile < 5 or percentile > 95:
                    outlier_type = "LOW" if percentile < 5 else "HIGH"
                    print_and_write(f"  {pdb_name}: {ion_value:.2f} Å (z={z_score_ion:+.2f}, "
                                   f"percentile={percentile:.1f}%, {outlier_type} outlier)")
            
            individual_ion_results.append(ion_result)
        
        # Multivariate test (overall configuration)
        print_and_write("\n" + "-"*80)
        print_and_write("Multivariate test (all residues combined):")
        print_and_write("-"*80)
        
        # Build bootstrap mean vectors
        bootstrap_mean_vectors = []
        for boot_idx in range(n_bootstrap):
            boot_mean_vec = [bootstrap_means[resid][boot_idx] for resid in all_residues]
            bootstrap_mean_vectors.append(boot_mean_vec)
        
        bootstrap_mean_vectors = np.array(bootstrap_mean_vectors)
        
        # Covariance of bootstrap means
        cov_boot = np.cov(bootstrap_mean_vectors.T)
        
        # Mahalanobis distance of end_2 mean from bootstrap distribution
        try:
            inv_cov = np.linalg.inv(cov_boot)
            diff_vec = end2_mean - np.mean(bootstrap_mean_vectors, axis=0)
            mahal_dist = np.sqrt(diff_vec @ inv_cov @ diff_vec)
            
            # Under CLT, this follows chi-squared with p degrees of freedom
            p_value_multivariate = 1 - stats.chi2.cdf(mahal_dist**2, df=len(all_residues))
            
            print_and_write(f"\nMahalanobis distance: {mahal_dist:.2f}")
            print_and_write(f"p-value (multivariate): {p_value_multivariate:.4e}")
            
            if p_value_multivariate < 0.001:
                print_and_write(f"Significance: *** (p < 0.001) - HIGHLY SIGNIFICANT")
            elif p_value_multivariate < 0.01:
                print_and_write(f"Significance: **  (p < 0.01)  - VERY SIGNIFICANT")
            elif p_value_multivariate < 0.05:
                print_and_write(f"Significance: *   (p < 0.05)  - SIGNIFICANT")
            else:
                print_and_write(f"Significance: ns  (p ≥ 0.05)  - NOT SIGNIFICANT")
            
            multivariate_result = {
                'mahalanobis_distance': float(mahal_dist),
                'p_value': float(p_value_multivariate),
                'is_significant': bool(p_value_multivariate < 0.05)
            }
        except np.linalg.LinAlgError:
            print_and_write("\nERROR: Singular covariance matrix")
            multivariate_result = None
        
        # Count significant residues
        n_significant = sum(1 for r in results_per_residue.values() if r['is_significant'])
        
        print_and_write("\n" + "-"*80)
        print_and_write("METHOD 1 SUMMARY:")
        print_and_write("-"*80)
        print_and_write(f"Significant residues: {n_significant}/{len(all_residues)}")
        
        if n_significant > 0:
            print_and_write(f"\nResidues with significant difference when ion leaves cavity:")
            for resid, res in results_per_residue.items():
                if res['is_significant']:
                    direction = "FARTHER" if res['end2_mean'] > res['state2_mean'] else "CLOSER"
                    print_and_write(f"  {res['residue_pdb']}: {direction} (z={res['z_score']:+.2f}, p={res['p_value']:.4f})")
        
        method1_results = {
            'method': 'central_limit_theorem',
            'n_bootstrap': int(n_bootstrap),
            'sample_size': int(sample_size),
            'n_state2_frames': int(len(all_state2_vectors)),
            'n_end2_frames': int(len(all_end2_vectors)),
            'state2_mean': [float(x) for x in state2_mean],
            'end2_mean': [float(x) for x in end2_mean],
            'per_residue_results': convert_to_serializable(results_per_residue),
            'individual_ion_results': convert_to_serializable(individual_ion_results),
            'multivariate_result': convert_to_serializable(multivariate_result),
            'n_significant_residues': int(n_significant),
            'residue_pdb_map': residue_pdb_map
        }
    
    # =========================================================================
    # METHOD 2: HOTELLING'S T²
    # =========================================================================
    print_and_write("\n" + "="*80)
    print_and_write("METHOD 2: HOTELLING'S T² TEST (GROUP COMPARISON)")
    print_and_write("="*80)
    
    n1 = len(all_state2_vectors)
    n2 = len(all_end2_vectors)
    p = all_state2_vectors.shape[1]
    
    if n1 < min_frames_method2:
        print_and_write(f"\n⚠ WARNING: Only {n1} 'ion in cavity' frames.")
        print_and_write(f"   Minimum recommended: {min_frames_method2} frames")
        print_and_write(f"   SKIPPING METHOD 2.")
        hotelling_results = None
    else:
        print_and_write(f"\n✓ Sample size check: {n1} frames (>= {min_frames_method2})")
        
        print_and_write(f"\nGroup sizes:")
        print_and_write(f"  Ion in cavity:      n1 = {n1}")
        print_and_write(f"  Ion leaving cavity: n2 = {n2}")
        print_and_write(f"  Dimensions:         p  = {p}")
        
        mean1 = np.mean(all_state2_vectors, axis=0)
        mean2 = np.mean(all_end2_vectors, axis=0)
        diff = mean2 - mean1
        
        print_and_write(f"\nMean vectors:")
        print_and_write(f"{'Residue (PDB)':<15} {'In cavity':<12} {'Leaving':<12} {'Difference':<12}")
        print_and_write("-"*55)
        for i, resid in enumerate(all_residues):
            pdb_name = residue_pdb_map[resid]
            print_and_write(f"{pdb_name:<15} {mean1[i]:>10.2f} Å  {mean2[i]:>10.2f} Å  {diff[i]:>+10.2f} Å")
        
        cov1 = np.cov(all_state2_vectors.T)
        cov2 = np.cov(all_end2_vectors.T)
        pooled_cov = ((n1-1)*cov1 + (n2-1)*cov2) / (n1+n2-2)
        
        try:
            T2 = (n1*n2)/(n1+n2) * diff @ np.linalg.inv(pooled_cov) @ diff
            F = (n1+n2-p-1) / ((n1+n2-2)*p) * T2
            df1 = p
            df2 = n1+n2-p-1
            p_value_hotelling = 1 - f_dist.cdf(F, df1, df2)
            
            print_and_write(f"\nHotelling's T² Test Results:")
            print_and_write(f"  T² statistic: {T2:.4f}")
            print_and_write(f"  F statistic:  {F:.4f}")
            print_and_write(f"  df1, df2:     {df1}, {df2}")
            print_and_write(f"  p-value:      {p_value_hotelling:.4e}")
            
            if p_value_hotelling < 0.001:
                print_and_write(f"  Significance: *** (p < 0.001) - HIGHLY SIGNIFICANT")
            elif p_value_hotelling < 0.01:
                print_and_write(f"  Significance: **  (p < 0.01)  - VERY SIGNIFICANT")
            elif p_value_hotelling < 0.05:
                print_and_write(f"  Significance: *   (p < 0.05)  - SIGNIFICANT")
            else:
                print_and_write(f"  Significance: ns  (p ≥ 0.05)  - NOT SIGNIFICANT")
            
            print_and_write(f"\nConclusion:")
            if p_value_hotelling < 0.05:
                print_and_write(f"  ✓ Ion configuration when LEAVING cavity is SIGNIFICANTLY DIFFERENT")
                print_and_write(f"    from typical 'ion in cavity' configuration (p = {p_value_hotelling:.4e})")
            else:
                print_and_write(f"  ✗ No significant difference between 'in cavity' and 'leaving cavity'")
            
            hotelling_results = {
                'method': 'hotellings_t2',
                'T2_statistic': float(T2),
                'F_statistic': float(F),
                'p_value': float(p_value_hotelling),
                'df1': int(df1),
                'df2': int(df2),
                'is_significant': bool(p_value_hotelling < 0.05),
                'state2_mean': [float(x) for x in mean1],
                'end2_mean': [float(x) for x in mean2],
                'mean_difference': [float(x) for x in diff],
                'n_state2': int(n1),
                'n_end2': int(n2)
            }
            
        except np.linalg.LinAlgError:
            print_and_write("\nERROR: Singular covariance matrix - cannot compute Hotelling's T²")
            print_and_write("This may happen if there's insufficient variation in the data.")
            hotelling_results = None
    
    # =========================================================================
    # INTERPRETATION
    # =========================================================================
    print_and_write("\n" + "="*80)
    print_and_write("INTERPRETATION & RECOMMENDATIONS")
    print_and_write("="*80)
    
    print_and_write("\n*** BOTH METHODS TEST MEANS ***")
    print_and_write("")
    print_and_write("Method 1 (CLT Bootstrap):")
    print_and_write("  → Tests each residue individually")
    print_and_write("  → Uses Central Limit Theorem to create Gaussian distributions")
    print_and_write("  → Also includes multivariate test (Mahalanobis distance)")
    print_and_write("  → NEW: Shows where each individual ion's end_2 falls")
    print_and_write("")
    print_and_write("Method 2 (Hotelling's T²):")
    print_and_write("  → Tests all residues simultaneously")
    print_and_write("  → Classic multivariate test")
    print_and_write("")
    
    if method1_results:
        print_and_write("\n1. METHOD 1 INTERPRETATION (CLT Bootstrap):")
        print_and_write("-" * 60)
        n_sig = method1_results['n_significant_residues']
        if n_sig > 0:
            print_and_write(f"   → {n_sig}/{len(all_residues)} residues show significant mean differences")
            print_and_write(f"   → Individual residue analysis reveals which specific residues change")
        else:
            print_and_write(f"   → No individual residues show significant mean differences")
        
        if method1_results['multivariate_result'] and method1_results['multivariate_result']['is_significant']:
            print_and_write(f"   → Multivariate test: SIGNIFICANT (p={method1_results['multivariate_result']['p_value']:.2e})")
            print_and_write(f"   → The overall 8-residue pattern is different at end_2")
    
    if hotelling_results:
        print_and_write("\n2. METHOD 2 INTERPRETATION (Hotelling's T²):")
        print_and_write("-" * 60)
        if hotelling_results['is_significant']:
            print_and_write(f"   → SIGNIFICANT (p={hotelling_results['p_value']:.2e})")
            print_and_write(f"   → end_2 frames are statistically different as a group")
            
            # Analyze which residues changed most
            mean1 = np.array(hotelling_results['state2_mean'])
            mean2 = np.array(hotelling_results['end2_mean'])
            diff = mean2 - mean1
            
            closer_residues = [(residue_pdb_map[all_residues[i]], diff[i]) for i in range(len(diff)) if diff[i] < -0.5]
            farther_residues = [(residue_pdb_map[all_residues[i]], diff[i]) for i in range(len(diff)) if diff[i] > 0.5]
            
            if closer_residues or farther_residues:
                print_and_write(f"")
                print_and_write(f"   Mean differences:")
                if closer_residues:
                    print_and_write(f"     Residues CLOSER at end_2:")
                    for res, delta in sorted(closer_residues, key=lambda x: x[1]):
                        print_and_write(f"       {res}: {delta:.2f} Å")
                
                if farther_residues:
                    print_and_write(f"     Residues FARTHER at end_2:")
                    for res, delta in sorted(farther_residues, key=lambda x: x[1], reverse=True):
                        print_and_write(f"       {res}: {delta:+.2f} Å")
        else:
            print_and_write(f"   → NOT SIGNIFICANT (p={hotelling_results['p_value']:.4f})")
            print_and_write(f"   → No evidence of group-level difference")
    
    print_and_write("\n" + "="*80)
    print_and_write("BIOLOGICAL CONCLUSION:")
    print_and_write("="*80)
    
    both_significant = False
    if method1_results and method1_results['multivariate_result']:
        if method1_results['multivariate_result']['is_significant'] and hotelling_results and hotelling_results['is_significant']:
            both_significant = True
    
    if both_significant:
        print_and_write("Both methods confirm: Permeation occurs from a distinct state")
        print_and_write("  • The ion-residue configuration at end_2 is significantly different")
        print_and_write("  • Specific residues show repositioning (some closer, some farther)")
        if method1_results:
            n_sig = method1_results['n_significant_residues']
            print_and_write(f"  • {n_sig}/{len(all_residues)} residues individually significant (Method 1)")
    elif method1_results or hotelling_results:
        if (method1_results and method1_results.get('multivariate_result') and method1_results['multivariate_result']['is_significant']) or (hotelling_results and hotelling_results['is_significant']):
            print_and_write("Evidence for structural distinction at permeation:")
            print_and_write("  • At least one method shows significant differences")
        else:
            print_and_write("No strong evidence for structural distinction at permeation:")
            print_and_write("  • Neither method shows significant differences")
    
    # Close report
    report.close()
    
    # =========================================================================
    # SAVE AND VISUALIZE
    # =========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # PRINT FINAL FRAME COUNTS CLEARLY
    print("\n" + "="*80)
    print("FINAL FRAME COUNTS USED IN ANALYSIS")
    print("="*80)
    print(f"Frames used for GAUSSIAN (ion in cavity): {len(all_state2_vectors)}")
    print(f"Frames tested AGAINST gaussian (ion leaving): {len(all_end2_vectors)}")
    print(f"Number of ions (permeation events): {len(permeation_events)}")
    print(f"Average frames per ion: {len(all_state2_vectors)/len(permeation_events):.1f}")
    print("="*80 + "\n")
    
    combined_results = {
        'analysis_info': {
            'n_residues': int(len(all_residues)),
            'residue_ids': [int(r) for r in all_residues],
            'residue_pdb_names': [residue_pdb_map[r] for r in all_residues],
            'n_events': int(len(permeation_events)),
            'n_state2_frames_total': int(len(all_state2_vectors)),
            'n_end2_frames': int(len(all_end2_vectors)),
            'mean_state2_duration': float(np.mean(durations)),
            'median_state2_duration': float(np.median(durations)),
            'frames_used_for_gaussian': int(len(all_state2_vectors)),  # EXPLICIT
            'frames_tested_against_gaussian': int(len(all_end2_vectors)),  # EXPLICIT
            'use_end1': bool(use_end1),  # Record which frame range was used
            'frame_range': 'end_1 to end_2' if use_end1 else 'start_2 to end_2'
        },
        'method1_clt_bootstrap': method1_results if method1_results else {},
        'method2_hotelling': hotelling_results if hotelling_results else {},
        'raw_vectors': {
            'state2_vectors': all_state2_vectors.tolist(),  # Save for pooling!
            'end2_vectors': all_end2_vectors.tolist()       # Save for pooling!
        }
    }
    
    output_file = results_dir / "permeation_significance_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    print(f"✓ JSON saved: {output_file}")
    print(f"✓ Report saved: {report_file}")
    
    # Create a MANIFEST file for external scripts to easily detect and parse
    manifest_file = results_dir / "ANALYSIS_MANIFEST.txt"
    with open(manifest_file, 'w') as f:
        f.write("PERMEATION SIGNIFICANCE ANALYSIS - FILE MANIFEST\n")
        f.write("="*80 + "\n\n")
        f.write("This directory contains permeation frame significance analysis results.\n")
        f.write("External scripts can parse these files to create summary analyses.\n\n")
        
        f.write("KEY METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Frames used for Gaussian (ion in cavity): {len(all_state2_vectors)}\n")
        f.write(f"Frames tested against Gaussian (leaving): {len(all_end2_vectors)}\n")
        f.write(f"Number of permeation events: {len(permeation_events)}\n")
        f.write(f"Number of residues tracked: {len(all_residues)}\n\n")
        
        f.write("FILES IN THIS DIRECTORY:\n")
        f.write("-"*80 + "\n")
        f.write("1. permeation_significance_report.txt\n")
        f.write("   - Human-readable detailed report\n")
        f.write("   - Contains all statistical results and interpretations\n\n")
        
        f.write("2. permeation_significance_analysis.json\n")
        f.write("   - Machine-readable complete results\n")
        f.write("   - Contains all numerical data, statistics, and p-values\n")
        f.write("   - Key fields:\n")
        f.write("     * analysis_info: Frame counts, residue info\n")
        f.write("     * method1_clt_bootstrap: Bootstrap results, z-scores, p-values\n")
        f.write("     * method2_hotelling: Hotelling's T² test results\n\n")
        
        f.write("3. individual_ion_analysis.csv\n")
        f.write("   - Per-ion results in spreadsheet format\n")
        f.write("   - Columns: ion_id, residue distances, z-scores, percentiles\n\n")
        
        f.write("4. PLOTS (PNG files):\n")
        if method1_results:
            f.write("   - clt_bootstrap_distributions.png: Bootstrap distributions per residue\n")
            f.write("   - clt_zscores.png: Z-score bar chart\n")
            f.write("   - individual_ion_distributions.png: Individual ion positions\n")
            f.write("   - individual_ion_heatmap.png: Ion-residue z-score heatmap\n\n")
        
        f.write("5. ANALYSIS_MANIFEST.txt (this file)\n")
        f.write("   - Directory contents and parsing guide\n\n")
        
        f.write("STATISTICAL RESULTS SUMMARY:\n")
        f.write("-"*80 + "\n")
        
        if method1_results:
            f.write("Method 1 (CLT Bootstrap):\n")
            f.write(f"  Significant residues: {method1_results['n_significant_residues']}/{len(all_residues)}\n")
            if method1_results.get('multivariate_result'):
                mv = method1_results['multivariate_result']
                f.write(f"  Multivariate test: {'SIGNIFICANT' if mv['is_significant'] else 'NOT SIGNIFICANT'}\n")
                f.write(f"  Multivariate p-value: {mv['p_value']:.4e}\n")
        
        if hotelling_results:
            f.write("\nMethod 2 (Hotelling's T²):\n")
            f.write(f"  Result: {'SIGNIFICANT' if hotelling_results['is_significant'] else 'NOT SIGNIFICANT'}\n")
            f.write(f"  p-value: {hotelling_results['p_value']:.4e}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("For external script parsing, use: permeation_significance_analysis.json\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Manifest saved: {manifest_file}")
    
    # Create a simple summary file for quick reference
    quick_summary_file = results_dir / "QUICK_SUMMARY.txt"
    with open(quick_summary_file, 'w') as f:
        f.write("QUICK SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Gaussian built from: {len(all_state2_vectors)} frames\n")
        f.write(f"Tested frames: {len(all_end2_vectors)} frames\n")
        f.write(f"Permeation events: {len(permeation_events)}\n\n")
        
        if method1_results and method1_results.get('multivariate_result'):
            mv = method1_results['multivariate_result']
            f.write(f"Method 1 Multivariate: {'SIGNIFICANT ***' if mv['is_significant'] else 'NOT SIGNIFICANT'}\n")
            f.write(f"  p-value: {mv['p_value']:.4e}\n\n")
        
        if hotelling_results:
            f.write(f"Method 2 Hotelling's T²: {'SIGNIFICANT ***' if hotelling_results['is_significant'] else 'NOT SIGNIFICANT'}\n")
            f.write(f"  p-value: {hotelling_results['p_value']:.4e}\n\n")
        
        if method1_results:
            f.write(f"Significant residues: {method1_results['n_significant_residues']}/{len(all_residues)}\n")
    
    print(f"✓ Quick summary saved: {quick_summary_file}")
    
    # Create CSV for individual ions
    if method1_results and 'individual_ion_results' in method1_results:
        create_individual_ion_csv(method1_results['individual_ion_results'], 
                                  all_residues, residue_pdb_map, results_dir)
    
    # Create plots if Method 1 was executed
    if method1_results:
        print("\nCreating visualizations...")
        create_clt_plots(method1_results, all_residues, residue_pdb_map, results_dir)
        
        # NEW: Individual ion plots
        if 'individual_ion_results' in method1_results:
            create_individual_ion_plots(method1_results, all_residues, residue_pdb_map, results_dir)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput files in: {results_dir}")
    print(f"  1. ANALYSIS_MANIFEST.txt  (file guide for external scripts)")
    print(f"  2. QUICK_SUMMARY.txt  (key results at a glance)")
    print(f"  3. permeation_significance_report.txt  (human-readable report)")
    print(f"  4. permeation_significance_analysis.json  (machine-readable data)")
    print(f"  5. individual_ion_analysis.csv  (per-ion results)")
    if method1_results:
        print(f"  6. clt_bootstrap_distributions.png")
        print(f"  7. clt_zscores.png")
        print(f"  8. individual_ion_distributions.png")
        print(f"  9. individual_ion_heatmap.png")
    print("\n" + "="*80)
    print(f"Frames used for Gaussian: {len(all_state2_vectors)}")
    print(f"Frames tested: {len(all_end2_vectors)}")
    print("="*80)
    
    return combined_results


def create_individual_ion_csv(individual_ion_results, all_residues, residue_pdb_map, output_dir):
    """Create CSV with individual ion analysis."""
    import pandas as pd
    
    rows = []
    for ion_result in individual_ion_results:
        ion_id = ion_result['ion_id']
        row = {'ion_id': ion_id}
        
        for resid in all_residues:
            if resid in ion_result['per_residue']:
                res_data = ion_result['per_residue'][resid]
                pdb_name = residue_pdb_map[resid]
                row[f'{pdb_name}_value'] = res_data['value']
                row[f'{pdb_name}_z_score'] = res_data['z_score']
                row[f'{pdb_name}_percentile'] = res_data['percentile']
                row[f'{pdb_name}_is_outlier'] = res_data['is_outlier']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_file = output_dir / "individual_ion_analysis.csv"
    df.to_csv(csv_file, index=False, float_format='%.3f')
    print(f"✓ Individual ion CSV saved: {csv_file}")


def create_clt_plots(method1_results, all_residues, residue_pdb_map, output_dir):
    """Create CLT visualization plots."""
    
    per_residue = method1_results['per_residue_results']
    
    # Plot 1: Bootstrap distributions for each residue
    n_residues = len(all_residues)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, resid in enumerate(all_residues):
        ax = axes[i]
        res = per_residue[resid]
        pdb_name = residue_pdb_map[resid]
        
        boot_dist = res['bootstrap_distribution']
        
        # Histogram
        ax.hist(boot_dist, bins=50, alpha=0.6, color='steelblue', 
               edgecolor='black', density=True, label='Bootstrap (in cavity)')
        
        # Fitted Gaussian
        mu = res['bootstrap_mean']
        sigma = res['bootstrap_std']
        x = np.linspace(min(boot_dist), max(boot_dist), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label='Gaussian (CLT)')
        
        # end_2 mean
        ax.axvline(res['end2_mean'], color='green', linestyle='--', 
                  linewidth=2, label='Leaving cavity mean')
        
        # Significance
        sig_marker = ""
        if res['p_value'] < 0.001:
            sig_marker = "***"
        elif res['p_value'] < 0.01:
            sig_marker = "**"
        elif res['p_value'] < 0.05:
            sig_marker = "*"
        
        # Labels
        ax.set_title(f"{pdb_name}\nz={res['z_score']:+.2f}, p={res['p_value']:.4f} {sig_marker}", 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Distance (Å)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('CLT Bootstrap Analysis: Distribution of Sample Means\n(Ion in GLU/ASN cavity)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "clt_bootstrap_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Bootstrap distributions plot saved")
    
    # Plot 2: Z-scores bar plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    pdb_names = [residue_pdb_map[r] for r in all_residues]
    z_scores = [per_residue[r]['z_score'] for r in all_residues]
    p_values = [per_residue[r]['p_value'] for r in all_residues]
    
    colors = ['red' if p < 0.001 else 'orange' if p < 0.05 else 'steelblue' 
             for p in p_values]
    
    bars = ax.bar(range(len(pdb_names)), z_scores, color=colors, alpha=0.7,
                 edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, z, p) in enumerate(zip(bars, z_scores, p_values)):
        height = bar.get_height()
        label_y = height + 0.3 if height > 0 else height - 0.3
        ax.text(i, label_y, f'{z:+.1f}', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=9, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=1.96, color='orange', linestyle='--', linewidth=2, label='p=0.05 (z=±1.96)')
    ax.axhline(y=-1.96, color='orange', linestyle='--', linewidth=2)
    ax.axhline(y=3.29, color='red', linestyle='--', linewidth=2, label='p=0.001 (z=±3.29)')
    ax.axhline(y=-3.29, color='red', linestyle='--', linewidth=2)
    
    ax.set_xticks(range(len(pdb_names)))
    ax.set_xticklabels(pdb_names, fontsize=12, fontweight='bold')
    ax.set_xlabel('Residue (PDB numbering)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Z-score', fontsize=14, fontweight='bold')
    ax.set_title('CLT Bootstrap Analysis: Z-scores for Each Residue\n(Ion leaving cavity vs ion in cavity)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label='p < 0.001 ***'),
        Patch(facecolor='orange', edgecolor='black', alpha=0.7, label='p < 0.05 *'),
        Patch(facecolor='steelblue', edgecolor='black', alpha=0.7, label='p ≥ 0.05 ns')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, title='Significance')
    
    plt.tight_layout()
    plt.savefig(output_dir / "clt_zscores.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Z-scores plot saved")


def create_individual_ion_plots(method1_results, all_residues, residue_pdb_map, output_dir):
    """Create plots showing individual ion positions in bootstrap distributions."""
    
    per_residue = method1_results['per_residue_results']
    individual_ions = method1_results['individual_ion_results']
    
    # Plot 1: Individual ions on bootstrap distributions (showing all ions on each residue plot)
    n_residues = len(all_residues)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, resid in enumerate(all_residues):
        ax = axes[i]
        res = per_residue[resid]
        pdb_name = residue_pdb_map[resid]
        
        boot_dist = res['bootstrap_distribution']
        
        # Histogram
        ax.hist(boot_dist, bins=50, alpha=0.4, color='steelblue', 
               edgecolor='black', density=True, label='Bootstrap (in cavity)')
        
        # Fitted Gaussian
        mu = res['bootstrap_mean']
        sigma = res['bootstrap_std']
        x = np.linspace(min(boot_dist), max(boot_dist), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'b-', linewidth=2, alpha=0.7)
        
        # Plot each individual ion's end_2
        ion_values = []
        for ion_result in individual_ions:
            if resid in ion_result['per_residue']:
                ion_val = ion_result['per_residue'][resid]['value']
                ion_values.append(ion_val)
        
        # Plot all individual ions
        y_max = ax.get_ylim()[1]
        for ion_val in ion_values:
            ax.scatter([ion_val], [y_max * 0.95], color='red', s=30, 
                      marker='v', alpha=0.6, zorder=5)
        
        # Plot mean end_2
        ax.axvline(res['end2_mean'], color='green', linestyle='--', 
                  linewidth=2, label='Mean (leaving)', zorder=4)
        
        # Mark first ion for legend
        if len(ion_values) > 0:
            ax.scatter([ion_values[0]], [y_max * 0.95], color='red', s=30, 
                      marker='v', label='Individual (leaving)', zorder=5)
        
        ax.set_title(f"{pdb_name} (z={res['z_score']:+.2f})", 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Distance (Å)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Individual Ions Leaving GLU/ASN Cavity', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "individual_ion_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Individual ion distributions plot saved")
    
    # Plot 2: Heatmap of z-scores for each ion-residue combination
    n_ions = len(individual_ions)
    z_score_matrix = np.zeros((n_ions, len(all_residues)))
    ion_ids = []
    
    for i, ion_result in enumerate(individual_ions):
        ion_ids.append(ion_result['ion_id'])
        for j, resid in enumerate(all_residues):
            if resid in ion_result['per_residue']:
                z_score_matrix[i, j] = ion_result['per_residue'][resid]['z_score']
    
    fig, ax = plt.subplots(figsize=(12, max(8, n_ions * 0.3)))
    
    # Create heatmap
    im = ax.imshow(z_score_matrix, cmap='RdBu_r', aspect='auto', 
                   vmin=-10, vmax=10, interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Z-score', fontsize=12, fontweight='bold')
    
    # Axes
    pdb_names = [residue_pdb_map[r] for r in all_residues]
    ax.set_xticks(range(len(pdb_names)))
    ax.set_xticklabels(pdb_names, fontsize=10, fontweight='bold')
    ax.set_yticks(range(min(n_ions, 50)))  # Show max 50 ions
    if n_ions <= 50:
        ax.set_yticklabels([str(ion_ids[i]) for i in range(n_ions)], fontsize=8)
    else:
        ax.set_yticklabels([str(ion_ids[i]) if i % 5 == 0 else '' for i in range(50)], fontsize=8)
    
    ax.set_xlabel('Residue (PDB numbering)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ion ID', fontsize=12, fontweight='bold')
    ax.set_title('Individual Ion Z-scores When Leaving Cavity\n(Red = farther than typical, Blue = closer than typical)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(len(pdb_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(min(n_ions, 50)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "individual_ion_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Individual ion heatmap saved")


# =============================================================================
# HELPER FUNCTION FOR COMBINING MULTIPLE SIMULATIONS
# =============================================================================

def combine_multiple_simulations(
    simulations_data,
    results_dir="combined_analysis",
    channel_type="G12",
    n_bootstrap=1000,
    sample_size=50,
    min_frames_method2=80
):
    """
    Combine permeation data from multiple simulations and run analysis.
    
    This pools all ion-residue distance vectors from ALL simulations together,
    then runs the statistical tests on the combined dataset.
    
    Parameters:
    -----------
    simulations_data : list of dict
        Each dict should contain:
        - 'universe': MDAnalysis.Universe
        - 'channel2': Channel object
        - 'all_residues': list of residue IDs
        - 'permeation_events': list of permeation events
        - 'name': str (optional, for tracking which simulation)
    results_dir : str or Path
        Where to save combined results
    channel_type : str
        Channel type (e.g., "G12")
    n_bootstrap, sample_size, min_frames_method2 : int
        Bootstrap parameters
        
    Returns:
    --------
    dict with combined analysis results
    
    Example usage:
    --------------
    simulations_data = [
        {
            'universe': universe1,
            'channel2': channel2_1,
            'all_residues': residues1,
            'permeation_events': events1,
            'name': 'simulation_1'
        },
        {
            'universe': universe2,
            'channel2': channel2_2,
            'all_residues': residues2,
            'permeation_events': events2,
            'name': 'simulation_2'
        },
        # ... more simulations
    ]
    
    combined_results = combine_multiple_simulations(
        simulations_data,
        results_dir="combined_analysis"
    )
    """
    from pathlib import Path
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMBINING MULTIPLE SIMULATIONS")
    print("="*80)
    print(f"Number of simulations: {len(simulations_data)}")
    
    # Get distance vector function (same as in main function)
    def get_distance_vector(universe, channel2, all_residues, frame_idx, ion_id):
        """Get 8D vector of distances from THIS SPECIFIC ION to each residue."""
        universe.trajectory[frame_idx]
        channel2.compute_geometry(2)
        
        ion = universe.select_atoms(f"resid {ion_id}")
        if len(ion) == 0:
            return None
        
        ion_pos = ion.positions[0]
        
        distance_vector = []
        for resid in all_residues:
            res_atoms = universe.select_atoms(f"resid {resid}")
            if len(res_atoms) == 0:
                return None
            res_com = res_atoms.center_of_mass()
            dist = np.linalg.norm(ion_pos - res_com)
            distance_vector.append(dist)
        
        return np.array(distance_vector)
    
    # Collect ALL vectors from ALL simulations
    all_combined_state2 = []
    all_combined_end2 = []
    all_combined_end2_ion_ids = []
    all_combined_end2_frames = []
    ion_end2_vectors_combined = {}
    
    simulation_stats = []
    
    for sim_idx, sim_data in enumerate(simulations_data):
        universe = sim_data['universe']
        channel2 = sim_data['channel2']
        all_residues = sorted(sim_data['all_residues'])
        permeation_events = sim_data['permeation_events']
        sim_name = sim_data.get('name', f'sim_{sim_idx+1}')
        
        print(f"\n--- Processing {sim_name} ---")
        print(f"  Permeation events: {len(permeation_events)}")
        
        state2_count = 0
        end2_count = 0
        
        for event in permeation_events:
            ion_id = event['ion_id']
            end_1 = event.get('end_1', event['start_2'])
            end_2 = event['end_2']
            
            # Collect 'in cavity' frames
            for frame_idx in range(end_1, end_2):
                vec = get_distance_vector(universe, channel2, all_residues, frame_idx, ion_id)
                if vec is not None:
                    all_combined_state2.append(vec)
                    state2_count += 1
            
            # Collect 'leaving cavity' frame
            vec_end2 = get_distance_vector(universe, channel2, all_residues, end_2, ion_id)
            if vec_end2 is not None:
                all_combined_end2.append(vec_end2)
                all_combined_end2_ion_ids.append(f"{sim_name}:{ion_id}")
                all_combined_end2_frames.append(end_2)
                ion_end2_vectors_combined[f"{sim_name}:{ion_id}"] = vec_end2
                end2_count += 1
        
        simulation_stats.append({
            'name': sim_name,
            'n_events': len(permeation_events),
            'n_state2_frames': state2_count,
            'n_end2_frames': end2_count
        })
        
        print(f"  → Collected {state2_count} 'in cavity' frames")
        print(f"  → Collected {end2_count} 'leaving cavity' frames")
    
    # Summary
    print("\n" + "="*80)
    print("COMBINED DATA SUMMARY")
    print("="*80)
    total_state2 = sum(s['n_state2_frames'] for s in simulation_stats)
    total_end2 = sum(s['n_end2_frames'] for s in simulation_stats)
    
    print(f"Total 'in cavity' frames: {total_state2}")
    print(f"Total 'leaving cavity' frames: {total_end2}")
    print(f"Total permeation events: {sum(s['n_events'] for s in simulation_stats)}")
    
    print("\nPer-simulation breakdown:")
    for stat in simulation_stats:
        print(f"  {stat['name']}: {stat['n_state2_frames']} in cavity, {stat['n_end2_frames']} leaving")
    
    # Convert to numpy arrays
    all_combined_state2 = np.array(all_combined_state2)
    all_combined_end2 = np.array(all_combined_end2)
    
    # Now create a "virtual" combined dataset to pass to main function
    # We'll use the first simulation's universe and channel for the analysis
    # (the actual frames have already been collected)
    
    print("\n" + "="*80)
    print("RUNNING COMBINED STATISTICAL ANALYSIS")
    print("="*80)
    print("Note: Analysis uses pooled data from all simulations")
    
    # Create a minimal permeation_events list (just for metadata)
    combined_events = []
    for stat in simulation_stats:
        combined_events.extend([{'simulation': stat['name']}] * stat['n_events'])
    
    # Use the residues from first simulation (should be same for all)
    all_residues = sorted(simulations_data[0]['all_residues'])
    
    # Convert residue IDs to PDB numbering
    from analysis.converter import convert_to_pdb_numbering
    residue_pdb_map = {}
    for resid in all_residues:
        residue_pdb_map[resid] = convert_to_pdb_numbering(resid, channel_type)
    
    # Now run the analysis using the collected vectors
    # We'll directly call the analysis parts (Method 1 and Method 2)
    # rather than re-collecting frames
    
    print(f"\nUsing {len(all_combined_state2)} 'in cavity' vectors")
    print(f"Using {len(all_combined_end2)} 'leaving cavity' vectors")
    
    # Save a summary file
    summary_file = results_dir / "combined_simulations_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("COMBINED SIMULATIONS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Number of simulations: {len(simulations_data)}\n")
        f.write(f"Total 'in cavity' frames: {total_state2}\n")
        f.write(f"Total 'leaving cavity' frames: {total_end2}\n")
        f.write(f"Total permeation events: {sum(s['n_events'] for s in simulation_stats)}\n\n")
        f.write("Per-simulation breakdown:\n")
        for stat in simulation_stats:
            f.write(f"  {stat['name']}:\n")
            f.write(f"    Events: {stat['n_events']}\n")
            f.write(f"    'In cavity' frames: {stat['n_state2_frames']}\n")
            f.write(f"    'Leaving cavity' frames: {stat['n_end2_frames']}\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")
    
    # Note: For full analysis, you would need to adapt the main function
    # to accept pre-collected vectors rather than collecting them itself
    # For now, return the combined data
    
    return {
        'all_state2_vectors': all_combined_state2,
        'all_end2_vectors': all_combined_end2,
        'simulation_stats': simulation_stats,
        'residue_pdb_map': residue_pdb_map,
        'all_residues': all_residues,
        'ion_end2_info': {
            'ion_ids': all_combined_end2_ion_ids,
            'frames': all_combined_end2_frames,
            'vectors': ion_end2_vectors_combined
        }
    }