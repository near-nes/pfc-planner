import os
import glob
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import sys

# Add the controller path to sys.path to find the new modules
sys.path.insert(0, "/sim/controller/complete_control")

from complete_control.config.core_models import OracleData, SimulationParams
from complete_control.utils_common.generate_signals_minjerk import generate_trajectory_minjerk


INITIAL_ELBOW_ANGLE = 90  # Hardcoded initial angle for the arm in the 'start' images

def generate_minjerk_trajectory_for_angles(
    start_angle_deg: float,
    end_angle_deg: float,
    time_prep: float = 150.0,
    time_move: float = 500.0,
    time_post: float = 0.0,
    n_trials: int = 1,
) -> list:
    """
    Generates a minimum jerk trajectory using the project's simulation parameters.
    """
    # Create the necessary data structures to call the generation function.
    # We use default SimulationParams and override the angles.
    oracle_data = OracleData(init_joint_angle=start_angle_deg, tgt_joint_angle=end_angle_deg)
    sim_params = SimulationParams(oracle=oracle_data,
                                  time_prep=time_prep,
                                  time_move=time_move,
                                  time_post=time_post,
                                  n_trials=n_trials,
                                  frozen=False)
    # Generate the trajectory using the provided function
    trajectory = generate_trajectory_minjerk(sim_params)
    return trajectory.tolist()

def rad2deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * (180 / 3.141592653589793)

def extract_positions_from_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse filename to extract (phase, target_angle, color).
    """
    base = os.path.basename(filename)
    parts = base.replace('.bmp', '').split('_')
    if len(parts) != 3:
        return None, None, None
    try:
        phase, angle_str, color = parts
        return phase, int(angle_str), color
    except ValueError:
        return None, None, None

def load_task_mapping_from_file(txt_file: str) -> Dict[str, str]:
    """
    Load task mapping from text file.
    Example: lines containing 'blue' map to 'left', 'red' to 'right'.
    """
    mapping = {}
    try:
        with open(txt_file, 'r') as f:
            for line in f:
                if "blue" in line:
                    mapping['blue'] = 'left'
                elif "red" in line:
                    mapping['red'] = 'right'
    except FileNotFoundError:
        raise FileNotFoundError(f"Task mapping text file not found: {txt_file}")
    return mapping

def get_image_paths_and_labels(
    data_dir: str,
    initial_elbow_angle: int = INITIAL_ELBOW_ANGLE
) -> List[Dict[str, Any]]:
    """
    Parses 'start' image filenames, generates ground truth trajectories on the fly,
    and returns a list of dicts with task metadata.
    """
    image_data = []
    image_files = glob.glob(os.path.join(data_dir, 'start_*.bmp'))

    task_map_path = os.path.join(data_dir, 'task_diff.txt')
    task_mapping = load_task_mapping_from_file(task_map_path)

    for img_path in image_files:
        phase, target_final_angle, color = extract_positions_from_filename(img_path)
        if phase != 'start' or target_final_angle is None or color is None:
            continue

        # Generate the ground truth trajectory programmatically using min-jerk
        ground_truth = generate_minjerk_trajectory_for_angles(
            start_angle_deg=initial_elbow_angle,
            end_angle_deg=target_final_angle
        )
        # plot trajectory for debugging
        import matplotlib.pyplot as plt
        plt.plot(ground_truth)
        plt.title(f"Trajectory from {initial_elbow_angle} to {target_final_angle}")
        plt.xlabel("Time step")
        plt.ylabel("Elbow Angle (deg)")
        plt.savefig(f"trajectory_{initial_elbow_angle}_to_{target_final_angle}.png")
        plt.close()
        plt.clf()

        angle_diff = target_final_angle - initial_elbow_angle

        image_data.append({
            'image_path': img_path,
            'color': color,
            'initial_angle': initial_elbow_angle,
            'target_final_angle': target_final_angle,
            'angle_difference': angle_diff,
            'ground_truth_trajectory': ground_truth,
            'target_choice': task_mapping.get(color),
            'trajectory_len': len(ground_truth)
        })

    return image_data
