import json
import argparse
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from .planners import ANNVisionNet, GLEVisionNet
from .dataset import RobotArmDataset
from .config import PlannerParams, default_params
import structlog

_log: structlog.BoundLogger = structlog.get_logger("[pfc_planner.train_vision]")


def get_project_root() -> Path:
    """Determines the project root directory."""
    primary_path = Path("submodules/pfc_planner")
    if primary_path.exists() and primary_path.is_dir():
        return primary_path.resolve()
    return Path(".").resolve()


def get_git_commit_hash(project_root: Path) -> str:
    """Gets the current git commit hash."""
    try:
        return subprocess.check_output(
            ['git', 'describe', '--always', '--dirty'],
            stderr=subprocess.PIPE,
            cwd=project_root
        ).decode('utf-8').strip()
    except Exception:
        return "N/A"


def train_vision_network(params: PlannerParams, project_root: Path = None, model_dir: Path = None):
    """Trains the vision network component (Angle Regressor + Choice Classifier)."""
    _log.info(f"--- Vision Training Started: {params.model_type.upper()} ---")

    if project_root is None:
        project_root = get_project_root()

    DATA_DIR = project_root / "data"
    MODELS_DIR = model_dir or project_root / "models"
    RESULTS_DIR = project_root / "results"

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    torch.manual_seed(params.seed)
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
        raise FileNotFoundError(f"No data found in {DATA_DIR}. Please generate data first.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    # Initialize model
    vision_net = (ANNVisionNet(params) if params.model_type == 'ann'
                  else GLEVisionNet(params)).to(device)

    optimizer = optim.Adam(vision_net.parameters(), lr=params.learning_rate)
    criterion_angles = nn.MSELoss()
    criterion_choice = nn.CrossEntropyLoss()

    loss_history = []

    for epoch in range(params.num_epochs):
        vision_net.train()
        running_loss = 0.0

        for images, true_angles, target_choice_idx in train_loader:
            images, true_angles, target_choice_idx = images.to(device), true_angles.to(device), target_choice_idx.to(device)

            optimizer.zero_grad()
            if params.model_type == 'ann':
                output = vision_net(images)
                angle_loss = criterion_angles(output[:, :params.num_angle_outputs], true_angles)
                choice_loss = criterion_choice(output[:, params.num_angle_outputs:], target_choice_idx)
                total_loss = angle_loss + choice_loss
                total_loss.backward()
                optimizer.step()
            else:
                target = torch.cat((true_angles, torch.nn.functional.one_hot(target_choice_idx, params.num_choices).float()), dim=1)
                with torch.no_grad():
                    for _ in range(params.gle_update_steps):
                        output = vision_net(images, target, beta=params.gle_beta)
                optimizer.step()
                angle_loss = criterion_angles(output[:, :params.num_angle_outputs], true_angles)
                choice_loss = criterion_choice(output[:, params.num_angle_outputs:], target_choice_idx)
                total_loss = angle_loss + choice_loss

            running_loss += total_loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            _log.info(f"Epoch {epoch+1}/{params.num_epochs} | Total Loss: {epoch_loss:.6f} | Angle Loss: {angle_loss.item():.6f} | Choice Loss: {choice_loss.item():.6f}")

    # Persistence
    model_save_path = MODELS_DIR / f"trained_{params.model_type}_planner.pth"
    config_save_path = MODELS_DIR / f"trained_{params.model_type}_planner.json"
    torch.save(vision_net.state_dict(), model_save_path)
    with open(config_save_path, 'w') as f:
        json.dump(asdict(params), f, indent=4)

    _log.info(f"Vision network and configuration saved to {MODELS_DIR}")


def main():
    """Main function for vision network training."""
    parser = argparse.ArgumentParser(description="Train ANN or GLE-based Vision Network")
    parser.add_argument('--type', type=str, choices=['ann', 'gle'], default='gle', help="Vision network type")
    parser.add_argument('--epochs', type=int, default=default_params.num_epochs, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=default_params.batch_size, help="Batch size")
    parser.add_argument('--lr', type=float, default=default_params.learning_rate, help="Learning rate")
    args = parser.parse_args()

    # Use default params for vision training settings
    params = default_params
    params.model_type = args.type
    params.num_epochs = args.epochs
    params.batch_size = args.batch_size
    params.learning_rate = args.lr

    project_root = get_project_root()
    params.git_commit = get_git_commit_hash(project_root)

    train_vision_network(
        params=params,
        project_root=project_root
    )


if __name__ == "__main__":
    main()
