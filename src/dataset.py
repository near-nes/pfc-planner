import os
import glob
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
from dataclasses import dataclass
from PIL import Image
import torch
from typing import Callable, Optional

# Import the generator from utils (utils still provides generate_minjerk_trajectory_for_angles)
from .utils import generate_minjerk_trajectory_for_angles

INITIAL_ELBOW_ANGLE = 90  # Hardcoded initial angle for the arm in the 'start' images (degrees)

def extract_positions_from_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse filename to extract (phase, target_angle, color).
    Expected filename format: "<phase>_<angle>_<color>.bmp" e.g. "start_120_blue.bmp"
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
    mapping: Dict[str, str] = {}
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

def deg2rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * (3.141592653589793 / 180.0)

def get_image_paths_and_labels(
    data_dir: str,
    initial_elbow_angle: int = INITIAL_ELBOW_ANGLE
) -> List[Dict[str, Any]]:
    """
    Parses 'start' image filenames, generates ground truth trajectories on the fly,
    and returns a list of dicts with task metadata. ground_truth_trajectory is returned in radians.
    """
    image_data: List[Dict[str, Any]] = []
    image_files = glob.glob(os.path.join(data_dir, 'start_*.bmp'))

    task_map_path = os.path.join(data_dir, 'task_diff.txt')
    task_mapping = load_task_mapping_from_file(task_map_path)

    for img_path in image_files:
        phase, target_final_angle, color = extract_positions_from_filename(img_path)
        if phase != 'start' or target_final_angle is None or color is None:
            continue

        # Generate the ground truth trajectory programmatically (degrees)
        ground_truth = generate_minjerk_trajectory_for_angles(
            start_angle_deg=initial_elbow_angle,
            end_angle_deg=target_final_angle
        )

        # Convert to radians for downstream use
        ground_truth_rad = deg2rad(np.array(ground_truth)).tolist()

        angle_diff = target_final_angle - initial_elbow_angle

        image_data.append({
            'image_path': img_path,
            'color': color,
            'initial_angle': initial_elbow_angle,
            'target_final_angle': target_final_angle,
            'angle_difference': angle_diff,
            'ground_truth_trajectory': ground_truth_rad,  # radians
            'target_choice': task_mapping.get(color),
            'trajectory_len': len(ground_truth_rad)
        })

    return image_data


class RobotArmDataset(torch.utils.data.Dataset):
    """
    Dataset holding image-based inputs and ground-truth trajectories/choices.

    Returns (image_tensor, target_trajectory_tensor, target_choice_idx_tensor)
    """
    def __init__(self, image_data: List[Dict[str, Any]], transform: Optional[Callable]=None):
        self.image_data = image_data
        self.transform = transform
        self.choice_to_idx = {'left': 0, 'right': 1}

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int):
        item = self.image_data[idx]
        image = Image.open(item['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Target for trajectory regression (the full sequence)
        target_trajectory = torch.tensor(item['ground_truth_trajectory'], dtype=torch.float)

        # Target for choice classification
        target_choice_idx = self.choice_to_idx[item['target_choice']]

        return image, target_trajectory, torch.tensor(target_choice_idx, dtype=torch.long)

    # Optionally you can return metadata or the original idx here to make evaluation
    # robust to shuffling (e.g., return image, traj, choice_idx, idx)
