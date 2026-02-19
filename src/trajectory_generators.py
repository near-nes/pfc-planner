"""
Trajectory generation modules that convert start/final angles to full trajectories.
"""
import sys
import os
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import PlannerParams
from .gle.abstract_net import GLEAbstractNet
from .gle.dynamics import GLEDynamics
from .gle.layers import GLELinear
from .gle.utils import get_phi_and_derivative
import structlog

_log: structlog.BoundLogger = structlog.get_logger("[pfc_planner.trajectory_generators]")

# Path to the external controller package
_CONTROLLER_PATH = os.environ.get("CONTROLLER_PATH", "/sim/controller/complete_control")

try:
    if not os.path.isdir(_CONTROLLER_PATH):
        raise ImportError(f"Controller path '{_CONTROLLER_PATH}' not found or not a directory.")

    sys.path.insert(0, _CONTROLLER_PATH)
    from complete_control.config.core_models import OracleData, SimulationParams
    from complete_control.utils_common.generate_signals_minjerk import generate_trajectory_minjerk
    _log.warning(f"Successfully imported controller from '{_CONTROLLER_PATH}'.")

except ImportError as e:
    _log.warning(f"Controller import failed: {e}. Using local fallback implementation.")
    from .minjerk_fallback import (
        FallbackOracleData as OracleData,
        FallbackSimulationParams as SimulationParams,
        generate_minjerk_fallback as generate_trajectory_minjerk,
    )


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


class MinJerkTrajectoryGenerator(TrajectoryGeneratorBase):
    """Generates trajectories using minimum-jerk profile."""

    def __init__(self, params: PlannerParams):
        self.params = params
        # Store simulation parameters for trajectory generation
        self.time_prep = getattr(params, 'time_prep', 650.0)
        self.time_move = getattr(params, 'time_move', 500.0)
        self.time_locked_with_feedback = getattr(params, 'time_locked_with_feedback', 150.0)
        self.time_grasp = getattr(params, 'time_grasp', 100.0)
        self.time_post = getattr(params, 'time_post', 100.0)
        self.resolution = getattr(params, 'resolution', 1.0)

        # Calculate expected trajectory length
        total_time = (
            self.time_prep + self.time_move +
            self.time_locked_with_feedback +
            self.time_grasp + self.time_post
        )
        self.trajectory_length = int(total_time / self.resolution)

    def angles_to_trajectory(self, start_angle_rad: float, final_angle_rad: float) -> np.ndarray:
        """Generate min-jerk trajectory from start to final angle."""
        # Convert to degrees for the controller
        start_angle_deg = np.rad2deg(start_angle_rad)
        final_angle_deg = np.rad2deg(final_angle_rad)

        oracle_data = OracleData(
            init_joint_angle=start_angle_deg,
            tgt_joint_angle=final_angle_deg
        )

        sim_params = SimulationParams(
            oracle=oracle_data,
            time_prep=self.time_prep,
            time_move=self.time_move,
            time_locked_with_feedback=self.time_locked_with_feedback,
            time_grasp=self.time_grasp,
            time_post=self.time_post,
            n_trials=1,
            frozen=False,
            resolution=self.resolution,
        )

        # Generate trajectory (returns in radians)
        trajectory_rad = generate_trajectory_minjerk(sim_params)
        traj_arr = np.array(trajectory_rad).flatten()

        return traj_arr


class ANNTrajectoryGenerator(TrajectoryGeneratorBase):
    """Neural network-based trajectory generator."""

    def __init__(self, params: PlannerParams, model_path: Path = None):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get trajectory length from params or use default
        self.trajectory_length = getattr(params, 'trajectory_length', 1500)

        # Create network
        self.net = ANNTrajectoryNet(
            input_size=2,  # start and final angles
            output_size=self.trajectory_length
        ).to(self.device)

        # Load pretrained model if provided
        if model_path is not None and model_path.exists():
            _log.warning(f"Loading trajectory generator from {model_path}")
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))

        self.net.eval()

    def angles_to_trajectory(self, start_angle_rad: float, final_angle_rad: float) -> np.ndarray:
        """Generate trajectory using neural network."""
        with torch.no_grad():
            angles = torch.tensor([start_angle_rad, final_angle_rad], dtype=torch.float32)
            angles = angles.unsqueeze(0).to(self.device)  # Add batch dimension
            trajectory = self.net(angles)
            return trajectory.squeeze(0).cpu().numpy()

    def save_model(self, model_path: Path):
        """Save the NN trajectory generator model."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), model_path)
        _log.warning(f"Trajectory generator saved to {model_path}")

    def load_model(self, model_path: Path):
        """Load the NN trajectory generator model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()
        _log.warning(f"Trajectory generator loaded from {model_path}")


class GLETrajectoryGenerator(TrajectoryGeneratorBase):
    """GLE-based trajectory generator."""

    def __init__(self, params: PlannerParams, model_path: Path = None):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get trajectory length from params or use default
        self.trajectory_length = getattr(params, 'trajectory_length', 1500)

        # Create GLE network
        self.net = GLETrajectoryNet(
            params=params,
            input_size=2,  # start and final angles
            output_size=self.trajectory_length
        ).to(self.device)

        # Load pretrained model if provided
        if model_path is not None and model_path.exists():
            _log.warning(f"Loading GLE trajectory generator from {model_path}")
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))

        self.net.eval()

    def angles_to_trajectory(self, start_angle_rad: float, final_angle_rad: float) -> np.ndarray:
        """Generate trajectory using GLE network."""
        with torch.no_grad():
            angles = torch.tensor([start_angle_rad, final_angle_rad], dtype=torch.float32)
            angles = angles.unsqueeze(0).to(self.device)  # Add batch dimension

            # Run multiple update steps for GLE convergence
            gle_update_steps = getattr(self.params, 'gle_update_steps', 10)
            for _ in range(gle_update_steps):
                trajectory = self.net(angles)

            return trajectory.squeeze(0).cpu().numpy()

    def save_model(self, model_path: Path):
        """Save the GLE trajectory generator model."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), model_path)
        _log.warning(f"GLE trajectory generator saved to {model_path}")

    def load_model(self, model_path: Path):
        """Load the GLE trajectory generator model."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()
        _log.warning(f"GLE trajectory generator loaded from {model_path}")


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
        gle_tau = getattr(params, 'gle_tau', 10.0)

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
        e[:, :self.output_size] = 0.01 * (target[:, :self.output_size] - output[:, :self.output_size])
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
        return NNTrajectoryGenerator(params, model_path)
    elif generator_type == "gle":
        return GLETrajectoryGenerator(params, model_path)
    else:
        raise ValueError(f"Unknown trajectory generator type: {generator_type}. Choose from 'minjerk', 'ann', or 'gle'.")
