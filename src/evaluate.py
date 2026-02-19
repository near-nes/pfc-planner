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
from .trajectory_generators import get_trajectory_generator


def main():
    """Main function to handle model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Planner Models for Robotic Arm")
    parser.add_argument('--model', type=str, choices=['ann', 'gle'], default=default_params.model_type, help="Model type to evaluate")
    parser.add_argument('--traj-gen', type=str, choices=['minjerk', 'ann', 'gle'], default='gle', help="Trajectory generator type")
    parser.add_argument('--traj-model', type=str, default=None, help="Path to trajectory generator model (if using 'ann' or 'gle' type)")
    parser.add_argument('--plot-trajectories', action='store_true', help="Generate trajectory comparison plots")
    args = parser.parse_args()

    PROJECT_ROOT = get_project_root()
    MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    # Update params with user's model choice
    current_params = default_params
    current_params.model_type = args.model
    current_params.trajectory_generator_type = args.traj_gen

    # Use factory to get planner
    print(f"Getting {args.model.upper()} planner with {args.traj_gen} trajectory generator...")
    planner = get_planner(params=current_params, model_dir=MODELS_DIR, project_root=PROJECT_ROOT)

    # Override trajectory generator if specified
    if args.traj_gen in ['ann', 'gle']:
        if args.traj_model:
            traj_model_path = Path(args.traj_model)
        else:
            traj_model_path = MODELS_DIR / f"trained_{args.traj_gen}_trajectory_generator.pth"

        traj_gen = get_trajectory_generator(args.traj_gen, current_params, traj_model_path)
        planner.set_trajectory_generator(traj_gen)

    print(f"{args.traj_gen.upper()} trajectory generator loaded successfully.")
    print(f"{args.model.upper()} Planner loaded successfully.")

    # Load evaluation dataset
    eval_dataset = RobotArmDataset(data_dir=str(DATA_DIR), params=current_params)
    if not eval_dataset.task_data:
        sys.exit(f"ERROR: No data found in {DATA_DIR}. Exiting.")

    print(f"Loaded {len(eval_dataset)} samples.")

    # Evaluate
    correct_choices = 0
    correct_start_angles = 0
    correct_final_angles = 0

    all_true_start = []
    all_pred_start = []
    all_true_final = []
    all_pred_final = []

    for item_metadata in eval_dataset.task_data:
        image_path = Path(item_metadata['image_path'])

        # Get angle predictions from vision network
        pred_start_rad, pred_final_rad, predicted_choice = planner.image_to_angles(image_path)

        # Get ground truth angles from metadata
        true_start_rad = np.deg2rad(item_metadata['initial_angle_deg'])
        true_final_rad = np.deg2rad(item_metadata['final_angle_deg'])

        # Store for analysis
        all_true_start.append(item_metadata['initial_angle_deg'])
        all_pred_start.append(np.rad2deg(pred_start_rad))
        all_true_final.append(item_metadata['final_angle_deg'])
        all_pred_final.append(np.rad2deg(pred_final_rad))

        # Check choice accuracy
        if predicted_choice == item_metadata['target_choice']:
            correct_choices += 1

        # Check angle accuracy (within 5 degrees tolerance)
        if np.isclose(pred_start_rad, true_start_rad, atol=np.deg2rad(5.0)):
            correct_start_angles += 1
        if np.isclose(pred_final_rad, true_final_rad, atol=np.deg2rad(5.0)):
            correct_final_angles += 1

        # Optionally plot trajectories
        if args.plot_trajectories:
            # Generate full trajectories for comparison
            true_trajectory = planner.trajectory_generator.angles_to_trajectory(true_start_rad, true_final_rad)
            pred_trajectory, _ = planner.image_to_trajectory(image_path)

            plt.figure(figsize=(10, 6))
            plt.plot(np.rad2deg(true_trajectory), label='True', color='blue')
            plt.plot(np.rad2deg(pred_trajectory), label='Predicted', color='red', linestyle='--')
            plt.axhline(y=item_metadata['initial_angle_deg'], color='green', linestyle=':', alpha=0.5, label='True Start')
            plt.axhline(y=item_metadata['final_angle_deg'], color='purple', linestyle=':', alpha=0.5, label='True Final')
            plt.title(f"Trajectory for {image_path.name}")
            plt.xlabel("Time Step")
            plt.ylabel("Angle (deg)")
            plt.legend()
            plt.grid(True)
            plt.savefig(RESULTS_DIR / f"{image_path.stem}_trajectory.png")
            plt.close()

    # Calculate accuracies and errors
    n_samples = len(eval_dataset)
    choice_accuracy = (correct_choices / n_samples) * 100
    start_angle_accuracy = (correct_start_angles / n_samples) * 100
    final_angle_accuracy = (correct_final_angles / n_samples) * 100

    start_mae = np.mean(np.abs(np.array(all_pred_start) - np.array(all_true_start)))
    final_mae = np.mean(np.abs(np.array(all_pred_final) - np.array(all_true_final)))

    # Print results
    print(f"\n--- Evaluation Complete ---")
    print(f"Choice Accuracy: {choice_accuracy:.2f}%")
    print(f"Start Angle Accuracy (±5°): {start_angle_accuracy:.2f}%")
    print(f"Final Angle Accuracy (±5°): {final_angle_accuracy:.2f}%")
    print(f"Start Angle MAE: {start_mae:.2f}°")
    print(f"Final Angle MAE: {final_mae:.2f}°")

    # Create angle comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Start angle comparison
    axes[0].scatter(all_true_start, all_pred_start, alpha=0.6)
    axes[0].plot([min(all_true_start), max(all_true_start)],
                 [min(all_true_start), max(all_true_start)], 'r--', label='Perfect Prediction')
    axes[0].set_xlabel('True Start Angle (deg)')
    axes[0].set_ylabel('Predicted Start Angle (deg)')
    axes[0].set_title(f'Start Angle Prediction (MAE: {start_mae:.2f}°)')
    axes[0].legend()
    axes[0].grid(True)

    # Final angle comparison
    axes[1].scatter(all_true_final, all_pred_final, alpha=0.6)
    axes[1].plot([min(all_true_final), max(all_true_final)],
                 [min(all_true_final), max(all_true_final)], 'r--', label='Perfect Prediction')
    axes[1].set_xlabel('True Final Angle (deg)')
    axes[1].set_ylabel('Predicted Final Angle (deg)')
    axes[1].set_title(f'Final Angle Prediction (MAE: {final_mae:.2f}°)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{args.model}_{args.traj_gen}_angle_predictions.png", dpi=150)
    plt.close()

    print(f"Plots saved to '{RESULTS_DIR.resolve()}'")


if __name__ == '__main__':
    main()
