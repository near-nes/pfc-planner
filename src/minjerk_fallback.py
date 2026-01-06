"""
Fallback implementations for minimum-jerk trajectory generation.

This module provides minimal, self-contained implementations of data structures
and trajectory generation logic, intended for use when the primary 'complete_control'
package is not available.
"""

from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class FallbackOracleData:
    """
    Minimal data structure for trajectory start and end points.
    Uses attribute names consistent with the primary 'complete_control' package.
    """
    init_joint_angle: float = 0.0
    tgt_joint_angle: float = 0.0

@dataclass
class FallbackSimulationParams:
    """Minimal simulation parameters for the fallback trajectory generator."""
    oracle: FallbackOracleData
    time_prep: float = 150.0
    time_move: float = 500.0
    time_grasp: float = 0.0
    time_post: float = 0.0
    n_trials: int = 1
    frozen: bool = False
    resolution: float = 0.1  # Time step for sampling

def generate_minjerk_fallback(sim_params: FallbackSimulationParams) -> np.ndarray:
    """
    Generate a simple min-jerk trajectory (in degrees) using fallback parameters.

    The trajectory is sampled at a time step defined by `resolution` and now
    includes a hold phase for `time_grasp`.
    """
    start = float(sim_params.oracle.init_joint_angle)
    end = float(sim_params.oracle.tgt_joint_angle)
    res = float(sim_params.resolution)

    # Convert durations to number of steps based on the resolution (`res`)
    prep_steps = max(0, int(round(sim_params.time_prep / res))) if sim_params.time_prep else 0
    move_steps = max(2, int(round(sim_params.time_move / res))) if sim_params.time_move else 2
    grasp_steps = max(0, int(round(sim_params.time_grasp / res))) if sim_params.time_grasp else 0
    post_steps = max(0, int(round(sim_params.time_post / res))) if sim_params.time_post else 0

    # Min-jerk polynomial blending for the movement segment
    t = np.linspace(0.0, 1.0, move_steps)
    blend = 10 * t**3 - 15 * t**4 + 6 * t**5
    move_segment = start + (end - start) * blend

    # Compose the full trajectory with all phases
    full_traj = np.concatenate([
        np.full(prep_steps, start, dtype=float),
        move_segment.astype(float),
        np.full(grasp_steps, end, dtype=float), # Hold at target angle for grasp duration
        np.full(post_steps, end, dtype=float)
    ])

    # If multiple trials are requested, repeat the trajectory for each trial
    if sim_params.n_trials > 1:
        full_traj = np.tile(full_traj[np.newaxis, :], (sim_params.n_trials, 1))

    return full_traj