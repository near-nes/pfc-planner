import os
import sys
import glob
import warnings
from typing import List, Tuple, Dict, Optional, Any, Callable

import numpy as np
from PIL import Image
import torch

from .config import PlannerParams
import structlog

_log: structlog.BoundLogger = structlog.get_logger("[pfc_planner.dataset]")

# Path to the external controller package
_CONTROLLER_PATH = os.environ.get("CONTROLLER_PATH", "/sim/controller/complete_control")

try:
    if not os.path.isdir(_CONTROLLER_PATH):
        raise ImportError(f"Controller path '{_CONTROLLER_PATH}' not found or not a directory.")

    sys.path.insert(0, _CONTROLLER_PATH)
    from complete_control.config.core_models import OracleData, SimulationParams
    from complete_control.utils_common.generate_signals_minjerk import generate_trajectory_minjerk
    _log.warning(f"Successfully imported controller from '{_CONTROLLER_PATH}'.")

except ImportError as e:
    _log.warning(f"Controller import failed: {e}. Using local fallback implementation.")
    # Assuming a local fallback implementation exists at .minjerk_fallback
    from .minjerk_fallback import (
        FallbackOracleData as OracleData,
        FallbackSimulationParams as SimulationParams,
        generate_minjerk_fallback as generate_trajectory_minjerk,
    )


class RobotArmDataset(torch.utils.data.Dataset):
    """
    Dataset for robot arm control, loading images and generating ground-truth trajectories.
    """
    def __init__(self, data_dir: str, params: PlannerParams, transform: Optional[Callable] = None):
        super().__init__()
        self.data_dir = data_dir
        self.params = params
        self.transform = transform
        self.choice_to_idx = {'left': 0, 'right': 1}
        self.task_data = self._load_all_task_data()

    def __len__(self) -> int:
        return len(self.task_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single data sample for the model.
        - Image tensor
        - Target trajectory tensor (in RADIANS)
        - Target choice index tensor
        """
        item = self.task_data[idx]

        # Load and transform image
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # The trajectory is already in radians, ready for the model
        target_trajectory = torch.tensor(item['ground_truth_trajectory_rad'], dtype=torch.float)

        # Target for choice classification
        target_choice_idx = self.choice_to_idx[item['target_choice']]
        target_choice = torch.tensor(target_choice_idx, dtype=torch.long)

        return image, target_trajectory, target_choice

    def _load_all_task_data(self) -> List[Dict[str, Any]]:
        """
        Parses all image filenames, generates trajectories, and returns a list of task metadata.
        """
        task_data = []
        image_files = glob.glob(os.path.join(self.data_dir, 'start_*.bmp'))
        task_map_path = os.path.join(self.data_dir, 'task_diff.txt')
        task_mapping = self._load_task_mapping(task_map_path)

        for img_path in image_files:
            phase, target_angle_deg, color = self._parse_filename(img_path)
            if phase != 'start' or target_angle_deg is None or color is None:
                continue

            # Generate ground truth trajectory, which will be in RADIANS
            trajectory_rad = self._generate_minjerk_trajectory_in_radians(
                start_angle_deg=self.params.initial_elbow_angle_deg,
                end_angle_deg=target_angle_deg
            )

            task_data.append({
                'image_path': img_path,
                'color': color,
                'initial_angle_deg': self.params.initial_elbow_angle_deg,
                'target_final_angle_deg': target_angle_deg,
                'ground_truth_trajectory_rad': trajectory_rad,  # Explicitly RADIANS
                'target_choice': task_mapping.get(color, 'unknown'),
            })

        # Ensure all trajectories have the same length as defined in params
        if task_data and len(task_data[0]['ground_truth_trajectory_rad']) != self.params.trajectory_length:
            _log.warning(
                f"Trajectory length mismatch! "
                f"Generated: {len(task_data[0]['ground_truth_trajectory_rad'])}, "
                f"Params: {self.params.trajectory_length}. "
                f"Please check simulation parameters in config.py."
            )

        return task_data

    @staticmethod
    def _parse_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Parses a filename like 'start_120_blue.bmp'."""
        base = os.path.basename(filename)
        parts = base.replace('.bmp', '').split('_')
        if len(parts) != 3:
            return None, None, None
        try:
            phase, angle_str, color = parts
            return phase, int(angle_str), color
        except (ValueError, IndexError):
            return None, None, None

    @staticmethod
    def _load_task_mapping(txt_file: str) -> Dict[str, str]:
        """Loads task mapping from 'blue' -> 'left', 'red' -> 'right'."""
        mapping = {}
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    if "blue" in line:
                        mapping['blue'] = 'left'
                    elif "red" in line:
                        mapping['red'] = 'right'
        except FileNotFoundError:
            raise FileNotFoundError(f"Task mapping file not found: {txt_file}")
        return mapping

    def _generate_minjerk_trajectory_in_radians(
        self, start_angle_deg: float, end_angle_deg: float
    ) -> List[float]:
        """
        Generates a min-jerk trajectory using the imported controller library
        based on simulation parameters from the params object.
        """
        oracle_data = OracleData(init_joint_angle=start_angle_deg, tgt_joint_angle=end_angle_deg)
        sim_params = SimulationParams(
            oracle=oracle_data,
            time_prep=self.params.time_prep,
            time_move=self.params.time_move,
            time_grasp=self.params.time_grasp,
            time_post=self.params.time_post,
            n_trials=1,
            frozen=False,
            resolution=self.params.resolution,
        )

        # The external `generate_trajectory_minjerk` function returns the trajectory in RADIANS
        trajectory_rad = generate_trajectory_minjerk(sim_params)

        # Ensure the result is a flat, 1D list of floats
        traj_arr = np.array(trajectory_rad).flatten()
        return traj_arr.tolist()
