"""
Fallback implementations for minimum-jerk trajectory generation.

This module provides minimal, self-contained implementations of data structures
and trajectory generation logic, intended for use when the primary 'complete_control'
package is not available. This implementation is aligned with the default parameters
and behavior of the primary SimulationParams model.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class FallbackOracleData:
    """
    Minimal data structure for trajectory start and end points.
    Angles are expected in DEGREES.
    """
    init_joint_angle: float = 0.0
    tgt_joint_angle: float = 0.0

@dataclass
class FallbackSimulationParams:
    """
    Minimal simulation parameters for the fallback trajectory generator, aligned
    with the defaults from the primary 'complete_control.config.core_models.SimulationParams'.
    """
    oracle: FallbackOracleData
    resolution: float = 1.0      # ms
    time_prep: float = 650.0     # ms
    time_move: float = 500.0     # ms
    time_grasp: float = 100.0    # ms
    time_post: float = 250.0     # ms
    n_trials: int = 1
    frozen: bool = False


def generate_minjerk_fallback(sim_params: FallbackSimulationParams) -> np.ndarray:
    """
    Generate a simple min-jerk trajectory (in RADIANS) using fallback parameters.

    This function mimics the behavior of the primary 'generate_trajectory_minjerk'
    function. It takes start/end angles in degrees, but computes and returns the
    trajectory in RADIANS.

    The trajectory consists of four phases:
    1. Prep: Hold at the start angle.
    2. Move: A minimum-jerk transition to the end angle.
    3. Grasp: Hold at the end angle.
    4. Post: Return to a neutral 0-radian position.
    """
    # Convert input degrees to radians for calculation, as this is the standard unit.
    start_rad = np.deg2rad(sim_params.oracle.init_joint_angle)
    end_rad = np.deg2rad(sim_params.oracle.tgt_joint_angle)
    res = sim_params.resolution

    # Convert durations to number of steps based on the resolution (`res`)
    prep_steps = max(0, int(round(sim_params.time_prep / res)))
    move_steps = max(2, int(round(sim_params.time_move / res))) # Need at least 2 for linspace
    grasp_steps = max(0, int(round(sim_params.time_grasp / res)))
    post_steps = max(0, int(round(sim_params.time_post / res)))

    # Min-jerk polynomial blending for the movement segment (in radians)
    t = np.linspace(0.0, 1.0, move_steps)
    blend = 10 * t**3 - 15 * t**4 + 6 * t**5
    move_segment = start_rad + (end_rad - start_rad) * blend

    full_traj = np.concatenate([
        np.full(prep_steps, start_rad),
        move_segment,
        np.full(grasp_steps, end_rad),
        np.zeros(post_steps)
    ])

    # If multiple trials are requested, tile the trajectory
    if sim_params.n_trials > 1:
        full_traj = np.tile(full_traj[np.newaxis, :], (sim_params.n_trials, 1))

    return full_traj
