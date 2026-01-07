"""
Centralized configuration for the PFC Planner project.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PlannerParams:
    """Parameters for planner models, simulation, and training."""
    # --- Model Hyperparameters ---
    model_type: str = 'gle'  # 'ann' or 'gle'
    num_choices: int = 2
    image_size: Tuple[int, int] = (100, 100)
    # For GLE model
    gle_tau: float = 1.0
    gle_beta: float = 1.0

    # --- Simulation Parameters for Trajectory Generation ---
    initial_elbow_angle_deg: float = 90.0
    time_prep: float = 150.0
    time_move: float = 500.0
    time_grasp: float = 0.0
    time_post: float = 0.0
    resolution: float = 0.1  # Timestep for trajectory generation

    # --- Training Parameters ---
    learning_rate: float = 0.01
    num_epochs: int = 500
    gle_update_steps: int = 10

    # --- Derived/Calculated Parameters ---
    # This is calculated dynamically, but can be set if known.
    trajectory_length: int = 0

    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        total_time = self.time_prep + self.time_move + self.time_grasp + self.time_post
        self.trajectory_length = int(total_time / self.resolution)

# Default parameters instance
default_params = PlannerParams()
