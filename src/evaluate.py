#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from .planners import ANNPlanner, GLEPlanner, ANNPlannerNet, GLEPlannerNet
from .dataset import RobotArmDataset
from .config import default_params

def main():
    """Main function to handle model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Planner Models for Robotic Arm")
    parser.add_argument('--model', type=str, choices=['ann', 'gle'], default=default_params.model_type, help="Model type to evaluate")
    args = parser.parse_args()

    # Update model type in params based on argument
    params = default_params
    params.model_type = args.model

    from .train import get_project_root
    PROJECT_ROOT = get_project_root()

    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    RESULTS_DIR.mkdir(exist_ok=True)

    eval_dataset = RobotArmDataset(data_dir=str(DATA_DIR), params=params)
    if not eval_dataset.task_data:
        sys.exit(f"ERROR: No data found in {DATA_DIR}. Exiting.")

    print(f"Loaded {len(eval_dataset)} samples. Trajectory length: {params.trajectory_length}")

    if args.model == 'ann':
        net = ANNPlannerNet(params=params)
        planner = ANNPlanner(params=params, net=net)
    else: # gle
        net = GLEPlannerNet(params=params)
        planner = GLEPlanner(params=params, net=net)

    model_path = MODELS_DIR / f"trained_{args.model}_planner.pth"
    try:
        planner.load_model(model_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Model not found at {model_path}. Please train it first.")

    correct_choices, correct_angles = 0, 0
    for item_metadata in eval_dataset.task_data:
        image_path = Path(item_metadata['image_path'])

        predicted_trajectory, predicted_choice = planner.image_to_trajectory(image_path)

        if predicted_choice == item_metadata['target_choice']:
            correct_choices += 1
        if np.isclose(predicted_trajectory[-1], item_metadata['ground_truth_trajectory_rad'][-1], atol=np.deg2rad(1.0)):
            correct_angles += 1

        plt.figure(figsize=(10, 6))
        plt.plot(np.rad2deg(item_metadata['ground_truth_trajectory_rad']), label='True', color='blue')
        plt.plot(np.rad2deg(predicted_trajectory), label='Predicted', color='red', linestyle='--')
        plt.title(f"Trajectory for {image_path.name}"); plt.xlabel("Time Step"); plt.ylabel("Angle (deg)")
        plt.legend(); plt.grid(True)
        plt.savefig(RESULTS_DIR / f"{image_path.stem}_trajectory.png")
        plt.close()

    choice_accuracy = (correct_choices / len(eval_dataset)) * 100
    angle_accuracy = (correct_angles / len(eval_dataset)) * 100
    print(f"\n--- Evaluation Complete ---")
    print(f"Choice Accuracy: {choice_accuracy:.2f}%")
    print(f"Final Angle Accuracy (within 1 degree): {angle_accuracy:.2f}%")
    print(f"Plots saved to '{RESULTS_DIR.resolve()}'")

if __name__ == '__main__':
    main()
