#!/usr/bin/env python3
"""
Train a neural network-based trajectory generator.

This script trains a NN to learn the mapping from (start_angle, final_angle) → full_trajectory
using min-jerk trajectories as ground truth.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from .config import PlannerParams, default_params
from .train import get_project_root, get_git_commit_hash
from .trajectory_generators import MLPTrajectoryGenerator, MinJerkTrajectoryGenerator
import structlog

_log: structlog.BoundLogger = structlog.get_logger("[pfc_planner.train_traj_gen]")


class TrajectoryDataset(Dataset):
    """Dataset for trajectory generation training."""

    def __init__(self, params: PlannerParams, num_samples: int = 10000, angle_range: tuple = (0, 180)):
        """
        Generate synthetic training data for trajectory generation.

        Args:
            params: PlannerParams with trajectory generation settings
            num_samples: Number of training samples to generate
            angle_range: Range of angles in degrees (min, max)
        """
        self.params = params
        self.num_samples = num_samples
        self.angle_range = angle_range

        # Use min-jerk generator to create ground truth
        self.minjerk_gen = MinJerkTrajectoryGenerator(params)

        # Pre-generate dataset
        _log.warning(f"Generating {num_samples} trajectory samples...")
        self.data = self._generate_samples()
        _log.warning(f"Dataset generation complete. Trajectory length: {self.minjerk_gen.trajectory_length}")

    def _generate_samples(self):
        """Generate random angle pairs and their corresponding trajectories."""
        samples = []

        for _ in range(self.num_samples):
            # Random start and final angles
            start_angle_deg = np.random.uniform(*self.angle_range)
            final_angle_deg = np.random.uniform(*self.angle_range)

            # Convert to radians
            start_angle_rad = np.deg2rad(start_angle_deg)
            final_angle_rad = np.deg2rad(final_angle_deg)

            # Generate ground truth trajectory using min-jerk
            trajectory = self.minjerk_gen.angles_to_trajectory(start_angle_rad, final_angle_rad)

            samples.append({
                'angles': np.array([start_angle_rad, final_angle_rad], dtype=np.float32),
                'trajectory': trajectory.astype(np.float32)
            })

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.from_numpy(sample['angles']),
            torch.from_numpy(sample['trajectory'])
        )


def train_trajectory_generator(
    params: PlannerParams,
    num_samples: int = 10000,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    angle_range: tuple = (0, 180),
    project_root: Path = None,
    model_dir: Path = None
):
    """
    Train the NN-based trajectory generator.

    Args:
        params: PlannerParams with trajectory generation settings
        num_samples: Number of training samples to generate
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        angle_range: Range of angles to sample from (degrees)
        project_root: Project root directory
        model_dir: Directory to save trained model
    """
    if project_root is None:
        project_root = get_project_root()
    if model_dir is None:
        model_dir = project_root / "models"

    model_dir.mkdir(exist_ok=True)
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.warning(f"Training trajectory generator on device: {device}")

    # Create dataset and dataloader
    dataset = TrajectoryDataset(params, num_samples, angle_range)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    nn_traj_gen = MLPTrajectoryGenerator(params)
    nn_traj_gen.net.to(device)
    nn_traj_gen.net.train()

    # Optimizer and loss
    optimizer = optim.Adam(nn_traj_gen.net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    loss_history = []

    _log.warning(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        running_loss = 0.0

        for angles, true_trajectory in dataloader:
            angles = angles.to(device)
            true_trajectory = true_trajectory.to(device)

            optimizer.zero_grad()

            # Forward pass
            predicted_trajectory = nn_traj_gen.net(angles)

            # Compute loss
            loss = criterion(predicted_trajectory, true_trajectory)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _log.warning(f"Epoch {epoch+1: >3}/{num_epochs} | Loss: {epoch_loss:.6f}")

    _log.warning("\n--- Training Complete ---")

    # Save model
    model_path = model_dir / "trajectory_generator_nn.pth"
    nn_traj_gen.save_model(model_path)
    _log.warning(f"Model saved to {model_path}")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Trajectory Generator Training Loss')
    plt.grid(True)
    plt.savefig(results_dir / 'trajectory_generator_training_loss.png')
    plt.close()
    _log.warning(f"Training plot saved to {results_dir}")

    # Test the model with a few examples
    _log.warning("\n--- Testing Trained Model ---")
    nn_traj_gen.net.eval()

    with torch.no_grad():
        # Test with a few specific angle pairs
        test_cases = [
            (90, 120),   # Small movement
            (45, 135),   # Medium movement
            (30, 150),   # Large movement
        ]

        for i, (start_deg, final_deg) in enumerate(test_cases):
            start_rad = np.deg2rad(start_deg)
            final_rad = np.deg2rad(final_deg)

            # Generate using trained NN
            pred_traj = nn_traj_gen.angles_to_trajectory(start_rad, final_rad)

            # Generate ground truth using min-jerk
            minjerk_gen = MinJerkTrajectoryGenerator(params)
            true_traj = minjerk_gen.angles_to_trajectory(start_rad, final_rad)

            # Calculate error
            mae = np.mean(np.abs(pred_traj - true_traj))
            _log.warning(f"Test {i+1}: {start_deg}° → {final_deg}° | MAE: {np.rad2deg(mae):.4f}°")

            # Plot comparison
            plt.figure(figsize=(10, 6))
            plt.plot(np.rad2deg(true_traj), label='Min-Jerk (Ground Truth)', color='blue')
            plt.plot(np.rad2deg(pred_traj), label='NN Prediction', color='red', linestyle='--')
            plt.axhline(y=start_deg, color='green', linestyle=':', alpha=0.5, label='Start Angle')
            plt.axhline(y=final_deg, color='purple', linestyle=':', alpha=0.5, label='Final Angle')
            plt.xlabel('Time Step')
            plt.ylabel('Angle (deg)')
            plt.title(f'Trajectory Comparison: {start_deg}° → {final_deg}°')
            plt.legend()
            plt.grid(True)
            plt.savefig(results_dir / f'traj_gen_test_{i+1}.png')
            plt.close()

    _log.warning(f"Test plots saved to {results_dir}")


def main():
    """Main function for trajectory generator training."""
    parser = argparse.ArgumentParser(description="Train NN-based Trajectory Generator")
    parser.add_argument('--samples', type=int, default=10000, help="Number of training samples")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--angle-min', type=int, default=0, help="Minimum angle (degrees)")
    parser.add_argument('--angle-max', type=int, default=180, help="Maximum angle (degrees)")
    args = parser.parse_args()

    # Use default params for trajectory generation settings
    params = default_params
    project_root = get_project_root()
    params.git_commit = get_git_commit_hash(project_root)

    train_trajectory_generator(
        params=params,
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        angle_range=(args.angle_min, args.angle_max),
        project_root=project_root
    )


if __name__ == "__main__":
    main()
