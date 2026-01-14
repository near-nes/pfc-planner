#!/usr/bin/env python3
import sys
import argparse
import json
import dataclasses
from dataclasses import asdict
from pathlib import Path
import subprocess

import torch
import numpy as np
import matplotlib.pyplot as plt

from .planners import ANNPlanner, GLEPlanner, ANNPlannerNet, GLEPlannerNet
from .dataset import RobotArmDataset
from .config import default_params, PlannerParams


def get_git_commit_hash() -> str:
    """Gets the current git commit hash."""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.PIPE
        ).decode('utf-8').strip()
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"

def load_and_verify_config(model_type: str, models_dir: Path, current_params: PlannerParams) -> PlannerParams:
    """Loads a saved config, verifies it against current params, and returns the loaded config."""
    config_path = models_dir / f"trained_{model_type}_planner.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    print(f"Loading and verifying configuration from {config_path}...")
    with open(config_path, 'r') as f:
        saved_params_dict = json.load(f)

    if 'image_size' in saved_params_dict and isinstance(saved_params_dict['image_size'], list):
        saved_params_dict['image_size'] = tuple(saved_params_dict['image_size'])

    # Set the current git commit on the params we are comparing against
    current_params.git_commit = get_git_commit_hash()
    current_params_dict = asdict(current_params)

    # Compare dictionaries
    if saved_params_dict != current_params_dict:
        print("\n" + "="*80)
        print("WARNING: Configuration mismatch detected!")
        print(f"  - Code version used for training: {saved_params_dict.get('git_commit', 'N/A')}")
        print(f"  - Current code version:           {current_params_dict.get('git_commit', 'N/A')}")
        print("Mismatched parameters:")
        for key in saved_params_dict:
            if key in current_params_dict and saved_params_dict[key] != current_params_dict[key]:
                print(f"  - '{key}': Saved='{saved_params_dict[key]}', Current='{current_params_dict[key]}'")
        print("="*80 + "\n")
        # Decide if you want to exit or just warn
        # sys.exit("Exiting due to configuration mismatch.")
    else:
        print("Configuration verified successfully.")

    # Return a PlannerParams instance from the saved config for the planner
    # We must remove any fields from the saved dict that are not in the dataclass definition
    valid_fields = {f.name for f in dataclasses.fields(PlannerParams)}
    filtered_dict = {k: v for k, v in saved_params_dict.items() if k in valid_fields}
    return PlannerParams(**filtered_dict)


def main():
    """Main function to handle model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Planner Models for Robotic Arm")
    parser.add_argument('--model', type=str, choices=['ann', 'gle'], default=default_params.model_type, help="Model type to evaluate")
    args = parser.parse_args()

    from .train import get_project_root
    PROJECT_ROOT = get_project_root()

    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    RESULTS_DIR.mkdir(exist_ok=True)

    # Temporarily set the model type on default_params for comparison
    # The final params for the planner will come from the loaded file
    params_for_comparison = default_params
    params_for_comparison.model_type = args.model

    try:
        # Load and verify the configuration first
        loaded_params = load_and_verify_config(args.model, MODELS_DIR, params_for_comparison)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}. Please train the model first.")
    except Exception as e:
        sys.exit(f"An unexpected error occurred during configuration verification: {e}")

    eval_dataset = RobotArmDataset(data_dir=str(DATA_DIR), params=loaded_params)
    if not eval_dataset.task_data:
        sys.exit(f"ERROR: No data found in {DATA_DIR}. Exiting.")

    print(f"Loaded {len(eval_dataset)} samples. Trajectory length: {loaded_params.trajectory_length}")

    if args.model == 'ann':
        net = ANNPlannerNet(params=loaded_params)
        planner = ANNPlanner(params=loaded_params, net=net)
    else: # gle
        net = GLEPlannerNet(params=loaded_params)
        planner = GLEPlanner(params=loaded_params, net=net)

    model_path = MODELS_DIR / f"trained_{args.model}_planner.pth"
    try:
        # Now, just load the weights since config is verified
        planner.load_model(model_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: Model file not found at {model_path}. Please train it first.")

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
