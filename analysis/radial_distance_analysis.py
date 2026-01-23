import numpy as np

def analyze_radial_distances(positions, frame, permeating_ion_id, channel):
    ion_pos = positions.get(frame, {}).get(permeating_ion_id)
    channel_center = channel.channel_center
    channel_axis = channel.channel_axis

    # Step 1: Vector from channel center to ion
    rel_vector = ion_pos - channel_center

    # Step 2: Project that vector onto the channel axis
    proj_on_axis = np.dot(rel_vector, channel_axis) * channel_axis

    # Step 3: Subtract to get the radial component (perpendicular to axis)
    radial_vector = rel_vector - proj_on_axis

    # Step 4: Radial distance is just the norm of that perpendicular component
    radial_distance = np.linalg.norm(radial_vector)

    return radial_distance
