"""
Centralized configuration for the PFC Planner project.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PlannerParams:
    """Parameters for planner models and training."""

    # --- Model Hyperparameters ---
    model_type: str = "gle"  # 'ann' or 'gle'
    num_choices: int = 2
    num_angle_outputs: int = 2  # start angle and final angle
    image_size: Tuple[int, int] = (100, 100)

    # For GLE model
    gle_tau: float = 1.0
    gle_beta: float = 1.0
    gle_update_steps: int = 10

    # --- Training Parameters ---
    learning_rate: float = 0.005
    num_epochs: int = 300
    batch_size: int = 64

    # --- Trajectory Generation Parameters (for minjerk) ---
    trajectory_generator_type: str = "minjerk"  # 'minjerk' or 'nn'
    time_prep: float = 650.0
    time_move: float = 500.0
    time_locked_with_feedback: float = 150.0
    time_grasp: float = 100.0
    time_post: float = 100.0
    resolution: float = 1.0

    # --- Tracking & Reproducibility ---
    git_commit: str = "N/A"  # To store the git commit hash
    seed: int = 0  # Random seed for reproducibility

    @property
    def trajectory_length(self) -> int:
        """Calculate trajectory length from time parameters."""
        total_time = (
            self.time_prep + self.time_move +
            self.time_locked_with_feedback +
            self.time_grasp + self.time_post
        )
        return int(total_time / self.resolution)


# Default parameters instance
default_params = PlannerParams()
