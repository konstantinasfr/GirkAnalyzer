import pickle
from pathlib import Path


def save_analyzer_state(analyzer, results_dir):
    """
    Save the analyzer's permeation events and key data.
    
    Parameters:
    -----------
    analyzer : IonPermeationAnalysis
        The analyzer object after running analysis
    results_dir : Path
        Directory to save the state
    
    Returns:
    --------
    Path : Path to saved state file
    """
    state_file = Path(results_dir) / "analyzer_state.pkl"
    
    # Save the important data
    state = {
        'permeation_events': analyzer.permeation_events,
        'ion_region_tracking': analyzer.ion_region_tracking,
        'ion_stages_reached': analyzer.ion_stages_reached,
        'ion_all_events': analyzer.ion_all_events,
        'start_frame': analyzer.start_frame,
        'end_frame': analyzer.end_frame
    }
    
    with open(state_file, 'wb') as f:
        pickle.dump(state, f)
    
    print(f"\n✓ Analyzer state saved to: {state_file}")
    return state_file


def load_analyzer_state(results_dir):
    """
    Load previously saved analyzer state.
    
    Parameters:
    -----------
    results_dir : Path
        Directory where state was saved
        
    Returns:
    --------
    dict : The saved state, or None if file doesn't exist
    """
    state_file = Path(results_dir) / "analyzer_state.pkl"
    
    if not state_file.exists():
        return None
    
    with open(state_file, 'rb') as f:
        state = pickle.load(f)
    
    print(f"\n✓ Loaded analyzer state from: {state_file}")
    return state


def restore_analyzer_from_state(u, state, results_dir, ch1, ch2, ch3, ch4, 
                                hbc_residues, hbc_diagonal_pairs,
                                sf_low_res_residues, sf_low_res_diagonal_pairs):
    """
    Create an analyzer object from saved state without re-running analysis.
    
    Parameters:
    -----------
    u : MDAnalysis.Universe
        The universe object
    state : dict
        Loaded state dictionary
    results_dir : Path
        Results directory
    ch1, ch2, ch3, ch4 : Channel objects
    hbc_residues, hbc_diagonal_pairs : Lists
    sf_low_res_residues, sf_low_res_diagonal_pairs : Lists
        
    Returns:
    --------
    IonPermeationAnalysis : Analyzer with restored state
    """
    from analysis.ion_analysis import IonPermeationAnalysis
    
    # Create analyzer object
    analyzer = IonPermeationAnalysis(
        u, ion_selection="resname K+ K", 
        start_frame=state['start_frame'], 
        end_frame=state['end_frame'],
        channel1=ch1, channel2=ch2, channel3=ch3, channel4=ch4,
        hbc_residues=hbc_residues, hbc_diagonal_pairs=hbc_diagonal_pairs,
        sf_low_res_residues=sf_low_res_residues, 
        sf_low_res_diagonal_pairs=sf_low_res_diagonal_pairs,
        results_dir=results_dir,
        count_ions=False  # Don't count again
    )
    
    # Restore saved data
    analyzer.permeation_events = state['permeation_events']
    analyzer.ion_region_tracking = state['ion_region_tracking']
    analyzer.ion_stages_reached = state['ion_stages_reached']
    analyzer.ion_all_events = state['ion_all_events']
    
    print(f"✓ Analyzer restored with {len(analyzer.permeation_events)} permeation events")
    
    return analyzer