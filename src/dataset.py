import os
import sys
import glob
from typing import List, Tuple, Dict, Optional, Any, Callable

import numpy as np
from PIL import Image
import torch

from .config import PlannerParams
import structlog

_log: structlog.BoundLogger = structlog.get_logger("[pfc_planner.dataset]")


class RobotArmDataset(torch.utils.data.Dataset):
    """
    Dataset for robot arm control, loading images and extracting start/final angles.
    """
    def __init__(self, data_dir: str, params: PlannerParams, transform: Optional[Callable] = None):
        super().__init__()
        self.data_dir = data_dir
        self.params = params
        self.transform = transform
        self.task_data = self._load_all_task_data()

    def __len__(self) -> int:
        return len(self.task_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single data sample for the model.
        - Image tensor
        - Target angles tensor [start_angle, final_angle] in RADIANS
        - Target choice index tensor
        """
        item = self.task_data[idx]

        # Load and transform image
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Convert angles from degrees to radians
        target_angles = torch.tensor([np.deg2rad(item['initial_angle_deg']), np.deg2rad(item['final_angle_deg'])], dtype=torch.float)

        # Use centralized labels from params
        target_choice_idx = self.params.choice_labels.index(item['target_choice'])
        target_choice = torch.tensor(target_choice_idx, dtype=torch.long)

        return image, target_angles, target_choice

    def _load_all_task_data(self) -> List[Dict[str, Any]]:
        """
        Parses all image filenames and extracts start/final angles.
        """
        task_data = []
        image_files = glob.glob(os.path.join(self.data_dir, '*.bmp'))
        task_map_path = os.path.join(self.data_dir, 'task_diff.txt')
        task_mapping = self._load_task_mapping(task_map_path)

        for img_path in image_files:
            start_angle, target_angle, color = self._parse_filename(img_path)

            if start_angle is None or target_angle is None:
                continue

            task_data.append({
                'image_path': img_path,
                'color': color,
                'initial_angle_deg': float(start_angle),
                'final_angle_deg': float(target_angle),
                'target_choice': task_mapping.get(color, 'unknown'),
            })

        _log.warning(f"Loaded {len(task_data)} task samples from {self.data_dir}")
        return task_data

    @staticmethod
    def _parse_filename(filename: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """Parses a filename like '90_120_blue.bmp'."""
        base = os.path.basename(filename)
        parts = base.replace('.bmp', '').split('_')
        if len(parts) != 3:
            return None, None, None
        try:
            return int(parts[0]), int(parts[1]), parts[2]
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
