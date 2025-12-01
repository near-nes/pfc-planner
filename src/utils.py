import os
import glob
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import sys
import warnings

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

INITIAL_ELBOW_ANGLE = 90  # Hardcoded initial angle for the arm in the 'start' images (degrees)

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

    The returned trajectory is a list of angles in radians (float). If multiple
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

def rad2deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * (180 / 3.141592653589793)

def deg2rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * (3.141592653589793 / 180)

def extract_positions_from_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse filename to extract (phase, target_angle, color).
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

def get_image_paths_and_labels(
    data_dir: str,
    initial_elbow_angle: int = INITIAL_ELBOW_ANGLE
) -> List[Dict[str, Any]]:
    """
    Parses 'start' image filenames, generates ground truth trajectories on the fly,
    and returns a list of dicts with task metadata. ground_truth_trajectory is now
    returned in radians.
    """
    image_data = []
    image_files = glob.glob(os.path.join(data_dir, 'start_*.bmp'))

    task_map_path = os.path.join(data_dir, 'task_diff.txt')
    task_mapping = load_task_mapping_from_file(task_map_path)

    for img_path in image_files:
        phase, target_final_angle, color = extract_positions_from_filename(img_path)
        if phase != 'start' or target_final_angle is None or color is None:
            continue

        # Generate the ground truth trajectory programmatically using min-jerk (degrees)
        ground_truth = generate_minjerk_trajectory_for_angles(
            start_angle_deg=initial_elbow_angle,
            end_angle_deg=target_final_angle
        )
        # plot trajectory for debugging
        # import matplotlib.pyplot as plt
        # plt.plot(ground_truth)
        # plt.title(f"Trajectory from {initial_elbow_angle} to {target_final_angle}")
        # plt.xlabel("Time step")
        # plt.ylabel("Elbow Angle (deg)")
        # plt.savefig(f"submodules/pfc_planner/data/trajectory_{initial_elbow_angle}_to_{target_final_angle}.png")
        # plt.close()

        angle_diff = target_final_angle - initial_elbow_angle

        image_data.append({
            'image_path': img_path,
            'color': color,
            'initial_angle': initial_elbow_angle,
            'target_final_angle': target_final_angle,
            'angle_difference': angle_diff,
            'ground_truth_trajectory': deg2rad(np.array(ground_truth)).tolist(),  # Convert to radians
            'target_choice': task_mapping.get(color),
            'trajectory_len': len(ground_truth)
        })

    return image_data
