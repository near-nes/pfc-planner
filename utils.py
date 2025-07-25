import os
import glob
from typing import List, Tuple, Dict, Optional, Any

INITIAL_ELBOW_ANGLE = 90  # Hardcoded initial angle for the arm in the 'start' images

def rad2deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * (180 / 3.141592653589793)

def load_trajectory_txt(filename: str) -> List[float]:
    """Load trajectory from text file as a list of floats."""
    trajectory = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                value = float(line.strip())
                trajectory.append(value)
            except ValueError:
                continue  # Skip bad lines silently
    return trajectory

def extract_positions_from_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse filename to extract (phase, target_angle, color).
    Example: start_140_blue.bmp -> ('start', 140, 'blue')
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

def load_all_predefined_data_and_config(data_directory: str) -> Dict[str, Any]:
    """
    Loads predefined trajectories and task mappings from the given directory.
    Raises on error.
    """
    flexion_path = os.path.join(data_directory, 'trajectories_90_to_140.txt')
    extension_path = os.path.join(data_directory, 'trajectories_90_to_20.txt')
    task_map_path = os.path.join(data_directory, 'task_diff.txt')

    flexion_traj = load_trajectory_txt(flexion_path)
    extension_traj = load_trajectory_txt(extension_path)
    task_map = load_task_mapping_from_file(task_map_path)

    if not flexion_traj or not extension_traj:
        raise ValueError("One or both trajectory files could not be loaded or are empty.")

    if len(flexion_traj) != len(extension_traj):
        raise ValueError("Flexion and Extension trajectories have different lengths.")

    return {
        'FLEXION_TRAJECTORY_DATA': flexion_traj,
        'EXTENSION_TRAJECTORY_DATA': extension_traj,
        'TASK_MAPPING': task_map,
        'TRAJECTORY_LEN': len(flexion_traj),
        'INITIAL_ELBOW_ANGLE': INITIAL_ELBOW_ANGLE,
    }

def get_image_paths_and_labels(
    data_dir: str,
    flexion_trajectory_data: List[float],
    extension_trajectory_data: List[float],
    task_mapping: Dict[str, str],
    initial_elbow_angle: int = INITIAL_ELBOW_ANGLE
) -> List[Dict[str, Any]]:
    """
    Parses 'start' image filenames and returns a list of dicts with task metadata and ground truth trajectory.
    """
    image_data = []
    image_files = glob.glob(os.path.join(data_dir, 'start_*.bmp'))

    for img_path in image_files:
        phase, target_final_angle, color = extract_positions_from_filename(img_path)
        if phase != 'start' or target_final_angle is None or color is None:
            continue

        angle_diff = target_final_angle - initial_elbow_angle

        if target_final_angle == 140 and angle_diff == 50:
            ground_truth = flexion_trajectory_data
        elif target_final_angle == 20 and angle_diff == -70:
            ground_truth = extension_trajectory_data
        else:
            ground_truth = None

        if ground_truth is not None:
            image_data.append({
                'image_path': img_path,
                'color': color,
                'initial_angle': initial_elbow_angle,
                'target_final_angle': target_final_angle,
                'angle_difference': angle_diff,
                'ground_truth_trajectory': ground_truth,
                'target_choice': task_mapping.get(color)
            })

    return image_data
