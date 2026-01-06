import os
import sys
import warnings
from typing import List
import numpy as np

# Path to the external controller package
_CONTROLLER_PATH = os.environ.get("CONTROLLER_PATH", "/sim/controller/complete_control")

try:
    # Attempt to import the primary controller package
    if not os.path.isdir(_CONTROLLER_PATH):
        raise ImportError(f"Controller path '{_CONTROLLER_PATH}' not found or not a directory.")

    sys.path.insert(0, _CONTROLLER_PATH)
    from complete_control.config.core_models import OracleData, SimulationParams
    from complete_control.utils_common.generate_signals_minjerk import generate_trajectory_minjerk
    warnings.warn(f"Successfully imported controller from '{_CONTROLLER_PATH}'.")

except ImportError as e:
    # If the import fails for any reason, fall back to the local implementation
    warnings.warn(f"Controller import failed: {e}. Using local fallback implementation.")
    from .minjerk_fallback import (
        FallbackOracleData as OracleData,
        FallbackSimulationParams as SimulationParams,
        generate_minjerk_fallback as generate_trajectory_minjerk,
    )

def generate_minjerk_trajectory_for_angles(
    start_angle_deg: float,
    end_angle_deg: float,
    time_prep: float = 150.0,
    time_move: float = 500.0,
    time_grasp: float = 0.0,
    time_post: float = 0.0,
    n_trials: int = 1,
    resolution: float = 0.1,
) -> List[float]:
    """
    Generates a minimum jerk trajectory using unified parameter interface.
    """
    oracle_data = OracleData(init_joint_angle=start_angle_deg, tgt_joint_angle=end_angle_deg)

    sim_params = SimulationParams(
        oracle=oracle_data,
        time_prep=time_prep,
        time_move=time_move,
        time_grasp=time_grasp,
        time_post=time_post,
        n_trials=n_trials,
        frozen=False,
        resolution=resolution,
    )

    trajectory = generate_trajectory_minjerk(sim_params)

    traj_arr = np.array(trajectory)
    if traj_arr.ndim == 2 and traj_arr.shape[0] > 0:
        traj_arr = traj_arr[0]

    return traj_arr.tolist()
