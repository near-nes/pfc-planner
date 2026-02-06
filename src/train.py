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

from .planners import ANNVisionNet, GLEVisionNet
from .dataset import RobotArmDataset
from .config import PlannerParams, default_params
import structlog

_log: structlog.BoundLogger = structlog.get_logger("[pfc_planner]")


def get_project_root() -> Path:
    """
    Determines the project root directory by checking for a primary path
    and falling back to the current directory if not found.
    """
    primary_path = Path("submodules/pfc_planner")
    if primary_path.exists() and primary_path.is_dir():
        _log.debug(f"Using primary project path: {primary_path.resolve()}")
        return primary_path.resolve()
    else:
        _log.debug("WARNING: Primary project path not found. Using current directory as project root.")
        return Path(".").resolve()


def get_git_commit_hash(project_root: Path) -> str:
    """Gets the current git commit hash from the project root directory."""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'describe', '--always', '--dirty'],
            stderr=subprocess.PIPE,
            cwd=project_root
        ).decode('utf-8').strip()
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        _log.debug("WARNING: Could not determine git commit hash. Not a git repository or git is not installed.")
        return "N/A"


def run_training(params: PlannerParams, project_root: Path = None, model_dir: Path = None):
    """
    Runs the training process for a given set of parameters.
    Trains the vision network component of the planner.

    Args:
        params: PlannerParams with training configuration
        project_root: Override project root (default: auto-detect via get_project_root())
        model_dir: Override model directory
    """
    _log.debug(f"--- Starting Vision Network Training for {params.model_type.upper()} Planner (Git commit: {params.git_commit}) ---")

    if project_root is None:
        project_root = get_project_root()

    DATA_DIR = project_root / "data"
    MODELS_DIR = model_dir or project_root / "models"
    RESULTS_DIR = project_root / "results"

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RobotArmDataset(
        data_dir=str(DATA_DIR),
        params=params,
        transform=transforms.Compose([
            transforms.Resize(params.image_size),
            transforms.ToTensor()
        ])
    )

    if len(train_dataset) == 0:
        _log.debug(f"ERROR: No data found in {DATA_DIR}. Run imagedata_gen.py to generate data before training.")
        return

    _log.debug(f"Loaded {len(train_dataset)} samples.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    # Create vision network
    if params.model_type == 'ann':
        vision_net = ANNVisionNet(params=params).to(device)
    else:  # gle
        vision_net = GLEVisionNet(params=params).to(device)

    optimizer = optim.Adam(vision_net.parameters(), lr=params.learning_rate)
    criterion_angles = nn.MSELoss()
    criterion_choice = nn.CrossEntropyLoss()

    loss_history, angle_loss_history, choice_loss_history = [], [], []

    _log.debug(f"\nStarting {params.model_type.upper()} vision network training on device '{device}'...")

    for epoch in range(params.num_epochs):
        vision_net.train()
        running_loss, running_angle_loss, running_choice_loss = 0.0, 0.0, 0.0

        for images, true_angles, target_choice_idx in train_loader:
            images = images.to(device)
            true_angles = true_angles.to(device)
            target_choice_idx = target_choice_idx.to(device)

            optimizer.zero_grad()

            if params.model_type == 'ann':
                output = vision_net(images)
                angle_loss = criterion_angles(output[:, :params.num_angle_outputs], true_angles)
                choice_loss = criterion_choice(output[:, params.num_angle_outputs:], target_choice_idx)
                total_loss = angle_loss + choice_loss
                total_loss.backward()
                optimizer.step()
            else:  # gle
                target = torch.cat((
                    true_angles,
                    torch.nn.functional.one_hot(target_choice_idx, num_classes=params.num_choices).float()
                ), dim=1)

                for _ in range(params.gle_update_steps):
                    output = vision_net(images, target, beta=params.gle_beta)

                optimizer.step()
                angle_loss = criterion_angles(output[:, :params.num_angle_outputs], true_angles)
                choice_loss = criterion_choice(output[:, params.num_angle_outputs:], target_choice_idx)
                total_loss = angle_loss + choice_loss

            running_loss += total_loss.item()
            running_angle_loss += angle_loss.item()
            running_choice_loss += choice_loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_angle_loss = running_angle_loss / len(train_loader)
        epoch_choice_loss = running_choice_loss / len(train_loader)

        loss_history.append(epoch_loss)
        angle_loss_history.append(epoch_angle_loss)
        choice_loss_history.append(epoch_choice_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _log.debug(
                f"Epoch {epoch+1: >3}/{params.num_epochs} | "
                f"Total Loss: {epoch_loss:.6f} | "
                f"Angle Loss: {epoch_angle_loss:.6f} | "
                f"Choice Loss: {epoch_choice_loss:.6f}"
            )

    _log.debug("\n--- Training Finished ---")
    model_save_path = MODELS_DIR / f"trained_{params.model_type}_planner.pth"
    config_save_path = MODELS_DIR / f"trained_{params.model_type}_planner.json"

    # Save vision network weights
    torch.save(vision_net.state_dict(), model_save_path)
    _log.debug(f"Vision network saved to {model_save_path}")

    # Save configuration to JSON
    with open(config_save_path, 'w') as f:
        json.dump(asdict(params), f, indent=4)
    _log.debug(f"Configuration saved to {config_save_path}")

    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Total Loss')
    plt.plot(angle_loss_history, label='Angle Loss')
    plt.plot(choice_loss_history, label='Choice Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'Vision Network Training Loss for {params.model_type.upper()} Planner')
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS_DIR / f'{params.model_type}_planner_training_loss.png')
    plt.close()
    _log.debug(f"Training plot saved to {RESULTS_DIR}")


def main():
    """Main function to handle training of a selected planner model."""
    parser = argparse.ArgumentParser(description="Train Planner Vision Networks for Robotic Arm")
    parser.add_argument('--model', type=str, choices=['ann', 'gle'], default=default_params.model_type, help="Model type to train")
    args = parser.parse_args()

    project_root = get_project_root()
    params = default_params
    params.model_type = args.model
    params.git_commit = get_git_commit_hash(project_root)

    run_training(params, project_root)


if __name__ == "__main__":
    main()
