import json
from pathlib import Path


def get_end3_all_frames(results_dir, permeation_events):
    """
    For each ion that permeated, get all frames where it crossed
    the lower boundary of region 3 (transition from region 3 -> region 4).

    Saves result to 'end3_all_frames.json' in results_dir and reuses it
    on subsequent calls if the file already exists.

    Args:
        results_dir (str or Path): Directory where ion_region_tracking.json is saved
                                   and where end3_all_frames.json will be stored.
        permeation_events (list): List of permeation event dicts with 'ion_id' keys
                                  (i.e., analyzer.permeation_events).

    Returns:
        dict: {ion_id (int): [frame1, frame2, ...]}
              All frames where each permeating ion crossed from region 3 -> region 4,
              including bounces.
    """
    results_dir = Path(results_dir)
    output_file = results_dir / "end3_all_frames.json"

    # If already computed, load and return
    if output_file.exists():
        print(f"✅ Loading existing end3_all_frames from: {output_file}")
        with open(output_file, 'r') as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}

    # Otherwise compute from ion_region_tracking.json
    print("🔄 Computing end3_all_frames from ion_region_tracking.json...")
    ion_region_tracking_path = results_dir / "ion_region_tracking.json"

    if not ion_region_tracking_path.exists():
        raise FileNotFoundError(
            f"ion_region_tracking.json not found in {results_dir}. "
            "Please run the full analysis first."
        )

    with open(ion_region_tracking_path, 'r') as f:
        ion_region_tracking = json.load(f)

    permeated_ion_ids = set(event["ion_id"] for event in permeation_events)

    end3_all = {}
    for ion_id in permeated_ion_ids:
        region_seq = ion_region_tracking.get(str(ion_id), {})
        sorted_frames = sorted(int(f) for f in region_seq.keys())

        crossing_frames = []
        for i in range(1, len(sorted_frames)):
            prev_frame = sorted_frames[i - 1]
            curr_frame = sorted_frames[i]
            if region_seq[str(prev_frame)] == 3 and region_seq[str(curr_frame)] == 4:
                crossing_frames.append(curr_frame)

        end3_all[ion_id] = crossing_frames
        print(f"  Ion {ion_id}: {len(crossing_frames)} end3 crossing(s) → frames {crossing_frames}")

    # Save for future runs
    with open(output_file, 'w') as f:
        json.dump({str(k): v for k, v in end3_all.items()}, f, indent=2)
    print(f"✅ end3_all_frames saved to: {output_file}")

    return end3_all


def get_end2_all_frames(results_dir, permeation_events):
    """
    For each ion that permeated, get all frames where it crossed
    the lower boundary of region 2 (transition from region 2 -> region 3).

    Saves result to 'end2_all_frames.json' in results_dir and reuses it
    on subsequent calls if the file already exists.

    Args:
        results_dir (str or Path): Directory where ion_region_tracking.json is saved
                                   and where end2_all_frames.json will be stored.
        permeation_events (list): List of permeation event dicts with 'ion_id' keys
                                  (i.e., analyzer.permeation_events).

    Returns:
        dict: {ion_id (int): [frame1, frame2, ...]}
              All frames where each permeating ion crossed from region 2 -> region 3,
              including bounces.
    """
    results_dir = Path(results_dir)
    output_file = results_dir / "end2_all_frames.json"

    # If already computed, load and return
    if output_file.exists():
        print(f"✅ Loading existing end2_all_frames from: {output_file}")
        with open(output_file, 'r') as f:
            data = json.load(f)
        # Convert string keys back to int
        return {int(k): v for k, v in data.items()}

    # Otherwise compute from ion_region_tracking.json
    print("🔄 Computing end2_all_frames from ion_region_tracking.json...")
    ion_region_tracking_path = results_dir / "ion_region_tracking.json"

    if not ion_region_tracking_path.exists():
        raise FileNotFoundError(
            f"ion_region_tracking.json not found in {results_dir}. "
            "Please run the full analysis first."
        )

    with open(ion_region_tracking_path, 'r') as f:
        ion_region_tracking = json.load(f)

    permeated_ion_ids = set(event["ion_id"] for event in permeation_events)

    end2_all = {}
    for ion_id in permeated_ion_ids:
        region_seq = ion_region_tracking.get(str(ion_id), {})
        sorted_frames = sorted(int(f) for f in region_seq.keys())

        crossing_frames = []
        for i in range(1, len(sorted_frames)):
            prev_frame = sorted_frames[i - 1]
            curr_frame = sorted_frames[i]
            if region_seq[str(prev_frame)] == 2 and region_seq[str(curr_frame)] == 3:
                crossing_frames.append(curr_frame)

        end2_all[ion_id] = crossing_frames
        print(f"  Ion {ion_id}: {len(crossing_frames)} end2 crossing(s) → frames {crossing_frames}")

    # Save for future runs
    with open(output_file, 'w') as f:
        json.dump({str(k): v for k, v in end2_all.items()}, f, indent=2)
    print(f"✅ end2_all_frames saved to: {output_file}")

    return end2_all