import sys
import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

from .planners import ANNPlannerNet, GLEPlannerNet
from .dataset import RobotArmDataset
from .config import PlannerParams, default_params

def get_project_root() -> Path:
    """
    Determines the project root directory by checking for a primary path
    and falling back to the current directory if not found.
    """
    primary_path = Path("submodules/pfc_planner")
    if primary_path.exists() and primary_path.is_dir():
        # Use the submodule path if it exists
        print(f"Using primary project path: {primary_path.resolve()}")
        return primary_path.resolve()
    else:
        # Fallback to the current directory for standalone execution
        print("WARNING: Primary project path not found. Using current directory as project root.")
        return Path(".").resolve()

def get_git_hash(path: Path) -> str:
    """Helper to get short hash and dirty status for a specific path."""
    try:
        # Get the short hash
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            cwd=path
        ).decode('utf-8').strip()

        # Check for uncommitted changes (dirty state)
        status = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'],
            stderr=subprocess.DEVNULL,
            cwd=path
        ).decode('utf-8').strip()

        if status:
            return f"{commit_hash}-dirty"
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: Could not determine git commit hash. Not a git repository or git is not installed.")
        return "N/A"

def get_git_commit_hash() -> str:
    """Gets the current git commit hashes for both the submodule and parent project."""
    pfc_root = get_project_root()
    pfc_hash = get_git_hash(pfc_root)

    # Try to find the parent project root
    # If pfc_root is '.../submodules/pfc_planner', the parent is 2 levels up
    if "submodules" in pfc_root.parts:
        controller_root = pfc_root.parent.parent
        controller_hash = get_git_hash(controller_root)
        return f"pfc:{pfc_hash} | controller:{controller_hash}"

    return f"pfc:{pfc_hash}"


def run_training(params: PlannerParams):
    """
    Runs the training process for a given set of parameters.
    """
    print(f"--- Starting Training for {params.model_type.upper()} Planner (Git commit: {params.git_commit}) ---")

    PROJECT_ROOT = get_project_root()
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RobotArmDataset(data_dir=str(DATA_DIR), params=params, transform=transforms.Compose([
        transforms.Resize(params.image_size), transforms.ToTensor()
    ]))

    if len(train_dataset) == 0:
        print(f"ERROR: No data found in {DATA_DIR}. Run imagedata_gen.py to generate data before evaluation.")
        return

    print(f"Loaded {len(train_dataset)} samples. Trajectory length: {params.trajectory_length}")

    # Use batch_size from params
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    if params.model_type == 'ann':
        net = ANNPlannerNet(params=params).to(device)
    else: # gle
        net = GLEPlannerNet(params=params).to(device)

    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)
    criterion_trajectory = nn.MSELoss()
    criterion_choice = nn.CrossEntropyLoss()

    loss_history, traj_loss_history, choice_loss_history = [], [], []

    print(f"\nStarting {params.model_type.upper()} training on device '{device}'...")
    for epoch in range(params.num_epochs):
        net.train()
        running_loss, running_traj_loss, running_choice_loss = 0.0, 0.0, 0.0

        for images, true_trajectory, target_choice_idx in train_loader:
            images, true_trajectory, target_choice_idx = images.to(device), true_trajectory.to(device), target_choice_idx.to(device)
            optimizer.zero_grad()

            if params.model_type == 'ann':
                output = net(images)
                trajectory_loss = criterion_trajectory(output[:, :params.trajectory_length], true_trajectory)
                choice_loss = criterion_choice(output[:, params.trajectory_length:], target_choice_idx)
                total_loss = trajectory_loss + choice_loss
                total_loss.backward()
                optimizer.step()
            else: # gle
                target = torch.cat((true_trajectory, torch.nn.functional.one_hot(target_choice_idx, num_classes=params.num_choices)), dim=1)
                for _ in range(params.gle_update_steps):
                    output = net(images, target, beta=params.gle_beta)
                optimizer.step()
                trajectory_loss = criterion_trajectory(output[:, :params.trajectory_length], true_trajectory)
                choice_loss = criterion_choice(output[:, params.trajectory_length:], target_choice_idx)
                total_loss = trajectory_loss + choice_loss

            running_loss += total_loss.item()
            running_traj_loss += trajectory_loss.item()
            running_choice_loss += choice_loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_traj_loss = running_traj_loss / len(train_loader)
        epoch_choice_loss = running_choice_loss / len(train_loader)

        loss_history.append(epoch_loss); traj_loss_history.append(epoch_traj_loss); choice_loss_history.append(epoch_choice_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1: >3}/{params.num_epochs} | Total Loss: {epoch_loss:.6f} | Traj Loss: {epoch_traj_loss:.6f} | Choice Loss: {epoch_choice_loss:.6f}")

    print("\n--- Training Finished ---")
    model_save_path = MODELS_DIR / f"trained_{params.model_type}_planner.pth"
    config_save_path = MODELS_DIR / f"trained_{params.model_type}_planner.json"

    # Save model weights
    torch.save(net.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save configuration to JSON
    with open(config_save_path, 'w') as f:
        json.dump(asdict(params), f, indent=4)
    print(f"Configuration saved to {config_save_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Total Loss'); plt.plot(traj_loss_history, label='Trajectory Loss'); plt.plot(choice_loss_history, label='Choice Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Training Loss for {params.model_type.upper()} Planner'); plt.legend(); plt.grid(True)
    plt.savefig(RESULTS_DIR / f'{params.model_type}_planner_training_loss.png')
    plt.close()
    print(f"Training plot saved to {RESULTS_DIR}")


def main():
    """Main function to handle training of a selected planner model."""
    parser = argparse.ArgumentParser(description="Train Planner Models for Robotic Arm")
    parser.add_argument('--model', type=str, choices=['ann', 'gle'], default=default_params.model_type, help="Model type to train")
    args = parser.parse_args()

    project_root = get_project_root()
    params = default_params
    params.model_type = args.model
    params.git_commit = get_git_commit_hash()

    run_training(params)

if __name__ == "__main__":
    main()
