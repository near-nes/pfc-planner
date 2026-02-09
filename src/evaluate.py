#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .planners import ANNPlanner, GLEPlanner
from .dataset import RobotArmDataset
from .config import default_params
from .train import get_project_root
from .factory import get_planner


def main():
    """Main function to handle model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Planner Models for Robotic Arm")
    parser.add_argument('--model', type=str, choices=['ann', 'gle'], default=default_params.model_type, help="Model type to evaluate")
    args = parser.parse_args()

    PROJECT_ROOT = get_project_root()
    MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    # Update params with user's model choice
    current_params = default_params
    current_params.model_type = args.model

    # Use factory to get planner (pass project_root to avoid duplicate detection)
    print(f"Getting {args.model.upper()} planner...")
    planner = get_planner(params=current_params, model_dir=MODELS_DIR, project_root=PROJECT_ROOT)
    print("Planner loaded successfully.")

    # Load evaluation dataset
    eval_dataset = RobotArmDataset(data_dir=str(DATA_DIR), params=current_params)
    if not eval_dataset.task_data:
        sys.exit(f"ERROR: No data found in {DATA_DIR}. Exiting.")

    print(f"Loaded {len(eval_dataset)} samples. Trajectory length: {current_params.trajectory_length}")

    # Calculate comparison indices
    # Start index: time_prep + 1 step
    start_angle_idx = int(current_params.time_prep / current_params.resolution) + 1
    # Final index: - (time grasp + time post) - 1 step
    final_angle_idx = -int((current_params.time_grasp + current_params.time_post) / current_params.resolution) - 1

    print(f"Comparing angles at start index {start_angle_idx} and final index {final_angle_idx}.")

    # Evaluate
    correct_choices = 0
    correct_start_angles = 0
    correct_final_angles = 0

    for item_metadata in eval_dataset.task_data:
        image_path = Path(item_metadata['image_path'])
        predicted_trajectory, predicted_choice = planner.image_to_trajectory(image_path)

        if predicted_choice == item_metadata['target_choice']:
            correct_choices += 1

        # Check accuracy at start index (before reach)
        pred_start_angle = predicted_trajectory[start_angle_idx]
        true_start_angle = item_metadata['ground_truth_trajectory_rad'][start_angle_idx]
        if np.isclose(pred_start_angle, true_start_angle, atol=np.deg2rad(1.0)):
            correct_start_angles += 1

        # Check accuracy at final index (after reach)
        pred_final_angle = predicted_trajectory[final_angle_idx]
        true_final_angle = item_metadata['ground_truth_trajectory_rad'][final_angle_idx]
        if np.isclose(pred_final_angle, true_final_angle, atol=np.deg2rad(1.0)):
            correct_final_angles += 1

        # Plot trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(np.rad2deg(item_metadata['ground_truth_trajectory_rad']), label='True', color='blue')
        plt.plot(np.rad2deg(predicted_trajectory), label='Predicted', color='red', linestyle='--')

        # Vertical lines for comparison points
        plt.axvline(x=start_angle_idx, color='green', linestyle=':', label='Start Comparison Point')
        plt.axvline(x=len(predicted_trajectory) + final_angle_idx, color='orange', linestyle=':', label='Final Comparison Point')

        plt.title(f"Trajectory for {image_path.name}")
        plt.xlabel("Time Step")
        plt.ylabel("Angle (deg)")
        plt.legend()
        plt.grid(True)
        plt.savefig(RESULTS_DIR / f"{image_path.stem}_trajectory.png")
        plt.close()

    # Calculate percentages
    choice_accuracy = (correct_choices / len(eval_dataset)) * 100
    start_angle_accuracy = (correct_start_angles / len(eval_dataset)) * 100
    final_angle_accuracy = (correct_final_angles / len(eval_dataset)) * 100

    # Print results
    print(f"\n--- Evaluation Complete ---")
    print(f"Choice Accuracy: {choice_accuracy:.2f}%")
    print(f"Start Angle Accuracy (before reach): {start_angle_accuracy:.2f}%")
    print(f"Final Angle Accuracy (after reach): {final_angle_accuracy:.2f}%")
    print(f"Plots saved to '{RESULTS_DIR.resolve()}'")


if __name__ == '__main__':
    main()
