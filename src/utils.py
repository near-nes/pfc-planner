import os
import sys
import warnings
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

# Path to the external controller package
_CONTROLLER_PATH = "/sim/controller/complete_control"

_USE_CONTROLLER = False
if os.path.isdir(_CONTROLLER_PATH):
    # Only insert if the directory actually exists
    sys.path.insert(0, _CONTROLLER_PATH)
    try:
        from complete_control.config.core_models import OracleData, SimulationParams
        from complete_control.utils_common.generate_signals_minjerk import generate_trajectory_minjerk
        _USE_CONTROLLER = True
    except Exception as e:  # ImportError or other issues during import
        warnings.warn(
            f"Controller modules found in {_CONTROLLER_PATH} but import failed: {e}. "
            "Falling back to local implementations."
        )
        _USE_CONTROLLER = False
else:
    warnings.warn(
        f"Controller path {_CONTROLLER_PATH} not found. Using fallback min-jerk implementation."
    )

# Minimal fallback implementations when the controller package is not available.
if not _USE_CONTROLLER:

    from dataclasses import dataclass

    @dataclass
    class OracleData:
        init_joint_angle: float = 0.0
        tgt_joint_angle: float = 0.0

    @dataclass
    class SimulationParams:
        oracle: OracleData
        time_prep: float = 150.0
        time_move: float = 500.0
        time_post: float = 0.0
        n_trials: int = 1
        frozen: bool = False
        dt: float = 0.1  # time step for sampling (same units as time_prep/time_move/time_post)

    def _generate_minjerk_fallback(sim_params: SimulationParams) -> np.ndarray:
        """
        Generate a simple min-jerk trajectory (degrees) for one trial using sampling step dt.
        time_prep/time_move/time_post are interpreted in the same units as dt (e.g., seconds).
        Returns degrees (fallback outputs are later converted to radians by the caller).
        """
        start = float(sim_params.oracle.init_joint_angle)
        end = float(sim_params.oracle.tgt_joint_angle)
        dt = float(getattr(sim_params, "dt", 0.1))

        # Convert durations to number of steps based on dt
        prep_steps = max(0, int(round(sim_params.time_prep / dt))) if sim_params.time_prep else 0
        move_steps = max(2, int(round(sim_params.time_move / dt))) if sim_params.time_move else 2
        post_steps = max(0, int(round(sim_params.time_post / dt))) if sim_params.time_post else 0

        # Min-jerk polynomial blending on [0,1]: 10*t^3 - 15*t^4 + 6*t^5
        t = np.linspace(0.0, 1.0, move_steps)
        blend = 10 * t**3 - 15 * t**4 + 6 * t**5
        move_segment = start + (end - start) * blend

        # Compose full trajectory: prep (constant start), move, post (constant end)
        full_traj = np.concatenate([
            np.full(prep_steps, start, dtype=float),
            move_segment.astype(float),
            np.full(post_steps, end, dtype=float)
        ])

        # If multiple trials requested, repeat the trajectory as rows
        if sim_params.n_trials and sim_params.n_trials > 1:
            # shape: (n_trials, timesteps)
            full_traj = np.tile(full_traj[np.newaxis, :], (sim_params.n_trials, 1))

        return full_traj

    # point generate_trajectory_minjerk name to fallback impl so the rest of the code can use the same name
    generate_trajectory_minjerk = _generate_minjerk_fallback

def generate_minjerk_trajectory_for_angles(
    start_angle_deg: float,
    end_angle_deg: float,
    time_prep: float = 150.0,
    time_move: float = 500.0,
    time_post: float = 0.0,
    n_trials: int = 1,
    dt: float = 0.1,
) -> list:
    """
    Generates a minimum jerk trajectory using the project's simulation parameters.
    Uses the external controller's implementation when available; otherwise uses
    the local fallback implementation. When using the fallback, dt controls the
    sampling timestep (default dt=0.1). time_prep/time_move/time_post are interpreted
    in the same time units as dt.

    The returned trajectory is a list of angles in degrees (float). If multiple
    trials are produced, the first trial is returned to preserve previous behavior.
    """
    oracle_data = OracleData(init_joint_angle=start_angle_deg, tgt_joint_angle=end_angle_deg)

    if _USE_CONTROLLER:
        # Use external SimulationParams signature (do not add dt attribute here)
        sim_params = SimulationParams(oracle=oracle_data,
                                      time_prep=time_prep,
                                      time_move=time_move,
                                      time_post=time_post,
                                      n_trials=n_trials,
                                      frozen=False)
        trajectory = generate_trajectory_minjerk(sim_params)
    else:
        # Use fallback SimulationParams which includes dt
        sim_params = SimulationParams(oracle=oracle_data,
                                      time_prep=time_prep,
                                      time_move=time_move,
                                      time_post=time_post,
                                      n_trials=n_trials,
                                      frozen=False,
                                      dt=dt)
        trajectory = generate_trajectory_minjerk(sim_params)

    # Normalise output to a numpy array (if multiple trials, take first trial)
    traj_arr = np.array(trajectory)
    if traj_arr.ndim == 2:
        traj_arr = traj_arr[0]

    return traj_arr.tolist()
