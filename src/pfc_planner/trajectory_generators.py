"""
Trajectory generation modules that convert start/final angles to full trajectories.
"""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from minjerk_dynamics import generate_trajectory

from .config import PlannerParams
from .gle.abstract_net import GLEAbstractNet
from .gle.dynamics import GLEDynamics
from .gle.layers import GLELinear
from .gle.utils import get_phi_and_derivative
import structlog

_log: structlog.BoundLogger = structlog.get_logger("[pfc_planner.trajectory_generators]")


class TrajectoryGeneratorBase(ABC):
    """Abstract base class for trajectory generation from angles."""

    @abstractmethod
    def angles_to_trajectory(self, start_angle_rad: float, final_angle_rad: float) -> np.ndarray:
        """
        Convert start and final angles to a full trajectory.

        Args:
            start_angle_rad: Starting angle in radians
            final_angle_rad: Final angle in radians

        Returns:
            numpy array of trajectory in radians
        """
        raise NotImplementedError

    def load_model(self, model_path: Path):
        """Default: do nothing (for generators like minjerk)."""
        pass

    def save_model(self, model_path: Path):
        """Default: do nothing."""
        pass


class MinJerkTrajectoryGenerator(TrajectoryGeneratorBase):
    """Generates trajectories using minimum-jerk profile."""

    def __init__(self, params: PlannerParams):
        self.params = params
        # Use calculated property from params for consistency
        self.trajectory_length = params.trajectory_length

    def angles_to_trajectory(self, start_angle_rad: float, final_angle_rad: float) -> np.ndarray:
        """Generate min-jerk trajectory from start to final angle."""
        return generate_trajectory(
            init_angle_rad=start_angle_rad,
            target_angle_rad=final_angle_rad,
            resolution_ms=self.params.resolution,
            time_prep_ms=self.params.time_prep,
            time_move_ms=self.params.time_move,
            time_locked_with_feedback_ms=self.params.time_locked_with_feedback,
            time_post_ms=self.params.time_grasp + self.params.time_post,
        )


class ANNTrajectoryGenerator(TrajectoryGeneratorBase):
    """Neural network-based trajectory generator."""

    def __init__(self, params: PlannerParams, model_path: Path = None):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get trajectory length from params property
        self.trajectory_length = params.trajectory_length

        # Create network
        self.net = ANNTrajectoryNet(
            input_size=2,  # start and final angles
            output_size=self.trajectory_length
        ).to(self.device)

        # Load pretrained model if provided
        if model_path is not None and model_path.exists():
            self.load_model(model_path)

    def angles_to_trajectory(self, start_angle_rad: float, final_angle_rad: float) -> np.ndarray:
        """Generate trajectory using neural network."""
        with torch.no_grad():
            angles = torch.tensor([[start_angle_rad, final_angle_rad]], dtype=torch.float32).to(self.device)
            trajectory = self.net(angles)
            return trajectory.squeeze(0).cpu().numpy()

    def save_model(self, model_path: Path):
        """Save the NN trajectory generator model."""
        _log.warning(f"Saving trajectory generator to {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), model_path)

    def load_model(self, model_path: Path):
        """Load the NN trajectory generator model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        _log.warning(f"Loading trajectory generator from {model_path}")
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()


class GLETrajectoryGenerator(TrajectoryGeneratorBase):
    """GLE-based trajectory generator."""

    def __init__(self, params: PlannerParams, model_path: Path = None):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get trajectory length from params property
        self.trajectory_length = params.trajectory_length

        # Create GLE network
        self.net = GLETrajectoryNet(
            params=params,
            input_size=2,  # start and final angles
            output_size=self.trajectory_length
        ).to(self.device)

        # Load pretrained model if provided
        if model_path is not None and model_path.exists():
            self.load_model(model_path)

    def angles_to_trajectory(self, start_angle_rad: float, final_angle_rad: float) -> np.ndarray:
        """Generate trajectory using GLE network."""
        with torch.no_grad():
            angles = torch.tensor([[start_angle_rad, final_angle_rad]], dtype=torch.float32).to(self.device)

            # Run multiple update steps for GLE convergence
            trajectory = None
            for _ in range(self.params.gle_update_steps):
                trajectory = self.net(angles)

            return trajectory.squeeze(0).cpu().numpy()

    def save_model(self, model_path: Path):
        """Save the GLE trajectory generator model."""
        _log.warning(f"Saving GLE trajectory generator to {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), model_path)

    def load_model(self, model_path: Path):
        """Load the GLE trajectory generator model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        _log.warning(f"Loading GLE trajectory generator from {model_path}")
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()


class ANNTrajectoryNet(nn.Module):
    """Neural network architecture for trajectory generation."""

    def __init__(self, input_size: int = 2, output_size: int = 1500, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, angles):
        """
        Args:
            angles: tensor of shape (batch_size, 2) containing [start_angle, final_angle]
        Returns:
            trajectory: tensor of shape (batch_size, trajectory_length)
        """
        return self.network(angles)


class GLETrajectoryNet(GLEAbstractNet, nn.Module):
    """GLE-based network architecture for trajectory generation."""

    def __init__(self, params: PlannerParams, input_size: int = 2, output_size: int = 1500, hidden_size: int = 256):
        super().__init__()
        self.params = params
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Get GLE parameters
        self.phi, self.phi_prime = get_phi_and_derivative("tanh")
        gle_tau = self.params.gle_tau

        # GLE layers (no convolutions, just fully connected)
        self.fc1 = GLELinear(input_size, hidden_size)
        self.fc2 = GLELinear(hidden_size, hidden_size)
        self.fc3 = GLELinear(hidden_size, output_size)

        # GLE dynamics for each layer
        self.fc1_dyn = GLEDynamics(self.fc1, tau_m=gle_tau, dt=1.0, phi=self.phi, phi_prime=self.phi_prime)
        self.fc2_dyn = GLEDynamics(self.fc2, tau_m=gle_tau, dt=1.0, phi=self.phi, phi_prime=self.phi_prime)
        self.fc3_dyn = GLEDynamics(self.fc3, tau_m=gle_tau, dt=1.0)  # No activation on output layer

    def compute_target_error(self, output, target, beta):
        """
        Compute target error for GLE learning.
        For trajectory generation, we simply compute MSE-style error.
        """
        e = torch.zeros_like(output)
        # Trajectory regression error
        e[:, :self.output_size] = (target[:, :self.output_size] - output[:, :self.output_size])
        return beta * e


def get_trajectory_generator(
    generator_type: str = "minjerk",
    params: PlannerParams = None,
    model_path: Path = None
) -> TrajectoryGeneratorBase:
    """
    Factory function to get a trajectory generator.

    Args:
        generator_type: Type of generator ('minjerk', 'ann', or 'gle')
        params: PlannerParams configuration
        model_path: Path to pretrained model (only for 'ann' or 'gle' types)

    Returns:
        TrajectoryGeneratorBase instance
    """
    if generator_type == "minjerk":
        return MinJerkTrajectoryGenerator(params)
    elif generator_type == "ann":
        return ANNTrajectoryGenerator(params, model_path)
    elif generator_type == "gle":
        return GLETrajectoryGenerator(params, model_path)
    else:
        raise ValueError(f"Unknown trajectory generator type: {generator_type}. Choose from 'minjerk', 'ann', or 'gle'.")
