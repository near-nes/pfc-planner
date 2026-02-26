#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from .config import default_params
from .dataset import RobotArmDataset
from .factory import get_planner
from .training import get_project_root


def main():
    """Evaluate Planner Models for Robotic Arm."""
    parser = argparse.ArgumentParser(description="Evaluate Planner Models")
    parser.add_argument('--model', type=str, choices=['ann', 'gle'],
                        default=default_params.model_type, help="Vision model type")
    parser.add_argument('--traj-gen', type=str, choices=['minjerk', 'ann', 'gle'],
                        default='gle', help="Trajectory generator type")
    parser.add_argument('--plot-trajectories', action='store_true', help="Save trajectory plots")
    args = parser.parse_args()

    # Update configuration
    params = default_params
    params.model_type = args.model
    params.trajectory_generator_type = args.traj_gen

    # Get project context
    PROJECT_ROOT = get_project_root()
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    # Get fully configured planner (triggers training if models are missing)
    print(f"Initializing {args.model.upper()} planner with {args.traj_gen.upper()} generator...")
    planner = get_planner(params=params, project_root=PROJECT_ROOT)

    # Load evaluation dataset
    eval_dataset = RobotArmDataset(data_dir=str(DATA_DIR), params=params)
    if not eval_dataset.task_data:
        print(f"ERROR: No data found in {DATA_DIR}. Please generate data first.")
        return

    print(f"Evaluating {len(eval_dataset)} samples...")

    # Performance metrics
    metrics = {"choice": 0, "start": 0, "final": 0}
    history = {"true_start": [], "pred_start": [], "true_final": [], "pred_final": []}

    for item in eval_dataset.task_data:
        image_path = Path(item['image_path'])

        # Inference
        pred_start_rad, pred_final_rad, pred_choice = planner.image_to_angles(image_path)

        # Ground Truth
        true_start_rad = np.deg2rad(item['initial_angle_deg'])
        true_final_rad = np.deg2rad(item['final_angle_deg'])

        # Store for analysis
        history["true_start"].append(item['initial_angle_deg'])
        history["pred_start"].append(np.rad2deg(pred_start_rad))
        history["true_final"].append(item['final_angle_deg'])
        history["pred_final"].append(np.rad2deg(pred_final_rad))

        # Check accuracies (±5 degrees tolerance for angles)
        if pred_choice == item['target_choice']: metrics["choice"] += 1
        if np.isclose(pred_start_rad, true_start_rad, atol=np.deg2rad(5.0)): metrics["start"] += 1
        if np.isclose(pred_final_rad, true_final_rad, atol=np.deg2rad(5.0)): metrics["final"] += 1

    # Print results summary
    n = len(eval_dataset)
    print(f"\n--- Evaluation Results ({args.model.upper()} Vision + {args.traj_gen.upper()} Trajectory) ---")
    print(f"Choice Accuracy: {(metrics['choice']/n)*100:.2f}%")
    print(f"Start Angle Accuracy (±5°): {(metrics['start']/n)*100:.2f}%")
    print(f"Final Angle Accuracy (±5°): {(metrics['final']/n)*100:.2f}%")

if __name__ == '__main__':
    main()
