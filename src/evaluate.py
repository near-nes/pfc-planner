#!/usr/bin/env python3
import sys
import argparse
import json
import subprocess
import dataclasses
from dataclasses import asdict
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from .planners import ANNPlanner, GLEPlanner, ANNPlannerNet, GLEPlannerNet
from .dataset import RobotArmDataset
from .config import default_params, PlannerParams
from .train import get_project_root, get_git_commit_hash, run_training


def verify_or_get_params(model_type: str, models_dir: Path, current_params: PlannerParams, project_root: Path) -> PlannerParams:
    """
    Checks for a consistent, existing model configuration.
    - If critical params match, returns the params (even if git hash differs).
    - If critical params differ or files are missing, returns None to trigger retraining.
    """
    config_path = models_dir / f"trained_{model_type}_planner.json"
    model_path = models_dir / f"trained_{model_type}_planner.pth"

    if not config_path.exists() or not model_path.exists():
        print(f"WARNING: Model or config file not found. Model will be trained.")
        return None

    print(f"Loading and verifying configuration from {config_path}...")
    with open(config_path, 'r') as f:
        saved_params_dict = json.load(f)

    # Normalize image_size for comparison (JSON stores tuples as lists)
    if 'image_size' in saved_params_dict and isinstance(saved_params_dict['image_size'], list):
        saved_params_dict['image_size'] = tuple(saved_params_dict['image_size'])

    # Get the current parameters and git hash
    current_params.git_commit = get_git_commit_hash()
    current_params_dict = asdict(current_params)

    # Define which keys are just "metadata" and shouldn't trigger a retrain
    metadata_keys = {'git_commit'}

    critical_mismatch = False
    metadata_mismatch = False

    for key, current_val in current_params_dict.items():
        saved_val = saved_params_dict.get(key)
        if saved_val != current_val:
            if key in metadata_keys:
                metadata_mismatch = True
            else:
                print(f"  - CRITICAL Mismatch on '{key}': Saved='{saved_val}', Current='{current_val}'")
                critical_mismatch = True

    if critical_mismatch:
        print("\n" + "!"*80)
        print("CRITICAL CONFIGURATION MISMATCH: Retraining is required.")
        print("!"*80 + "\n")
        return None

    if metadata_mismatch:
        print("\n" + "-"*80)
        print("NOTE: Code version mismatch detected, but parameters are identical.")
        print(f"  - Model trained with: {saved_params_dict.get('git_commit', 'N/A')}")
        print(f"  - Current code version: {current_params_dict.get('git_commit', 'N/A')}")
        print("Skipping retraining and proceeding with evaluation.")
        print("-"*80 + "\n")

    print("Configuration verified successfully.")
    # Return the saved params to ensure evaluation uses exactly what the model was trained with
    return dataclasses.replace(current_params, **saved_params_dict)


def main():
    """Main function to handle model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Planner Models for Robotic Arm")
    parser.add_argument('--model', type=str, choices=['ann', 'gle'], default=default_params.model_type, help="Model type to evaluate")
    args = parser.parse_args()

    PROJECT_ROOT = get_project_root()
    MODELS_DIR = PROJECT_ROOT / "models"

    current_params = default_params
    current_params.model_type = args.model
    # git_commit will be set during verification

    # This function now handles all checks: existence and consistency
    verified_params = verify_or_get_params(args.model, MODELS_DIR, current_params, PROJECT_ROOT)

    if verified_params is None:
        print("Triggering model retraining...")
        # We need to ensure the git_commit is set on the params object used for training
        current_params.git_commit = get_git_commit_hash()
        run_training(current_params)
        print("\n--- Retraining complete. Proceeding with evaluation of the new model. ---")
        verified_params = current_params

    # --- Proceed with evaluation ---
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    eval_dataset = RobotArmDataset(data_dir=str(DATA_DIR), params=verified_params)
    if not eval_dataset.task_data:
        sys.exit(f"ERROR: No data found in {DATA_DIR}. Run imagedata_gen.py to generate data before evaluation.")

    print(f"Loaded {len(eval_dataset)} samples. Trajectory length: {verified_params.trajectory_length}")

    if args.model == 'ann':
        net = ANNPlannerNet(params=verified_params)
        planner = ANNPlanner(params=verified_params, net=net)
    else: # gle
        net = GLEPlannerNet(params=verified_params)
        planner = GLEPlanner(params=verified_params, net=net)

    model_path = MODELS_DIR / f"trained_{args.model}_planner.pth"
    planner.load_model(model_path)
    post_phase_steps = int((verified_params.time_grasp + verified_params.time_post) / verified_params.resolution)
    angle_comparison_index = -post_phase_steps - 1 if post_phase_steps > 0 else -1
    print(f"Comparing angles at index {angle_comparison_index} (end of movement phase).")

    correct_choices, correct_angles = 0, 0
    for item_metadata in eval_dataset.task_data:
        image_path = Path(item_metadata['image_path'])
        predicted_trajectory, predicted_choice = planner.image_to_trajectory(image_path)
        if predicted_choice == item_metadata['target_choice']:
            correct_choices += 1

        pred_angle = predicted_trajectory[angle_comparison_index]
        true_angle = item_metadata['ground_truth_trajectory_rad'][angle_comparison_index]
        if np.isclose(pred_angle, true_angle, atol=np.deg2rad(1.0)):
            correct_angles += 1

        # plt.figure(figsize=(10, 6))
        # plt.plot(np.rad2deg(item_metadata['ground_truth_trajectory_rad']), label='True', color='blue')
        # plt.plot(np.rad2deg(predicted_trajectory), label='Predicted', color='red', linestyle='--')
        # plt.axvline(x=len(predicted_trajectory) + angle_comparison_index, color='green', linestyle=':', label='Angle Comparison Point')
        # plt.title(f"Trajectory for {image_path.name}"); plt.xlabel("Time Step"); plt.ylabel("Angle (deg)")
        # plt.legend(); plt.grid(True)
        # plt.savefig(RESULTS_DIR / f"{image_path.stem}_trajectory.png")
        # plt.close()

    choice_accuracy = (correct_choices / len(eval_dataset)) * 100
    angle_accuracy = (correct_angles / len(eval_dataset)) * 100
    print(f"\n--- Evaluation Complete ---")
    print(f"Choice Accuracy: {choice_accuracy:.2f}%")
    print(f"Final Angle Accuracy (at end of movement): {angle_accuracy:.2f}%")
    print(f"Plots saved to '{RESULTS_DIR.resolve()}'")


if __name__ == '__main__':
    main()
