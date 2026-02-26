import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from .gle.abstract_net import GLEAbstractNet
from .gle.dynamics import GLEDynamics
from .gle.layers import GLEConv, GLELinear
from .gle.utils import get_phi_and_derivative
from .config import PlannerParams
from .trajectory_generators import TrajectoryGeneratorBase, get_trajectory_generator

class VisionNet(ABC, nn.Module):
    """Abstract base class for vision networks that predict angles and choices from images."""

    @abstractmethod
    def forward(self, x, target=None, beta=1.0):
        """Forward pass returning concatenated [angles, choice_logits]."""
        raise NotImplementedError


class ANNVisionNet(VisionNet):
    """ANN-based vision network for angle and choice prediction."""

    def __init__(self, params: PlannerParams):
        super().__init__()
        self.params = params
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2), nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1), nn.Tanh(),
            nn.Flatten()
        )
        dummy_input = torch.rand(1, 3, *self.params.image_size)
        conv_output_size = self.conv_layers(dummy_input).size(1)

        # Output: 2 angles (start, final) + num_choices for classification
        self.angle_regressor = nn.Linear(conv_output_size, self.params.num_angle_outputs)
        self.choice_classifier = nn.Linear(conv_output_size, self.params.num_choices)

    def forward(self, x, target=None, beta=1.0):
        features = self.conv_layers(x)
        angles = self.angle_regressor(features)
        choice_logits = self.choice_classifier(features)
        return torch.cat((angles, choice_logits), dim=1)


class GLEVisionNet(GLEAbstractNet, VisionNet):
    """GLE-based vision network for angle and choice prediction."""

    def __init__(self, params: PlannerParams):
        super().__init__()
        self.params = params
        self.phi, self.phi_prime = get_phi_and_derivative("tanh")

        self.conv1 = GLEConv(3, 16, kernel_size=5, stride=4, padding=1)
        self.conv2 = GLEConv(16, 32, kernel_size=3, stride=4, padding=1)

        dummy_input = torch.rand(1, 3, *self.params.image_size)
        conv_output_size = self.conv2(self.conv1(dummy_input)).view(1, -1).size(1)

        self.fc1 = GLELinear(conv_output_size, 128)
        # Output: 2 angles + num_choices
        self.fc2 = GLELinear(128, self.params.num_angle_outputs + self.params.num_choices)

        self.conv1_dyn = GLEDynamics(self.conv1, tau_m=self.params.gle_tau, dt=1.0, phi=self.phi, phi_prime=self.phi_prime)
        self.conv2_dyn = GLEDynamics(self.conv2, tau_m=self.params.gle_tau, dt=1.0, phi=self.phi, phi_prime=self.phi_prime)
        self.fc1_dyn = GLEDynamics(self.fc1, tau_m=self.params.gle_tau, dt=1.0, phi=self.phi, phi_prime=self.phi_prime)
        self.fc2_dyn = GLEDynamics(self.fc2, tau_m=self.params.gle_tau, dt=1.0)

    def compute_target_error(self, output, target, beta):
        e = torch.zeros_like(output)
        # Error for angle prediction (first 2 outputs)
        e[:, :self.params.num_angle_outputs] = 0.01 * (target[:, :self.params.num_angle_outputs] - output[:, :self.params.num_angle_outputs])

        # Error for choice classification (remaining outputs)
        choice_probs = torch.softmax(output[:, self.params.num_angle_outputs:], dim=1)
        target_choice = target[:, self.params.num_angle_outputs:]
        e[:, self.params.num_angle_outputs:] = target_choice - choice_probs

        return beta * e


class Planner(ABC):
    """
    Base planner class composed of:
    1. Vision network: extracts angles and choice from images
    2. Trajectory generator: converts angles to full trajectories
    """

    def __init__(
        self,
        params: PlannerParams,
        vision_net: VisionNet,
        trajectory_generator: Optional[TrajectoryGeneratorBase] = None
    ):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vision network for angle and choice prediction
        self.vision_net = vision_net
        self.vision_net.to(self.device)

        self.image_transform = transforms.Compose([
            transforms.Resize(self.params.image_size),
            transforms.ToTensor(),
        ])
        self.model_loaded = False

        # Trajectory generator for converting angles to trajectories
        if trajectory_generator is None:
            self.trajectory_generator = get_trajectory_generator(
                generator_type=params.trajectory_generator_type,
                params=params
            )
        else:
            self.trajectory_generator = trajectory_generator

    def load_model(self, model_path: Path):
        """Load the vision network weights."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        print(f"Loading vision network from {model_path} to device '{self.device}'...")
        try:
            self.vision_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.vision_net.eval()
            self.model_loaded = True
        except RuntimeError as e:
            print(f"Error loading model. Architecture mismatch or corrupted file.")
            raise e

    def save_model(self, model_path: Path):
        """Save the vision network's state_dict."""
        print(f"Saving vision network to {model_path}...")
        os.makedirs(model_path.parent, exist_ok=True)
        torch.save(self.vision_net.state_dict(), model_path)

    def set_trajectory_generator(self, trajectory_generator: TrajectoryGeneratorBase):
        """Swap trajectory generator at runtime for flexibility."""
        self.trajectory_generator = trajectory_generator

    @abstractmethod
    def _run_vision_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Subclasses implement specific model inference (e.g., GLE iterations)."""
        raise NotImplementedError

    def image_to_angles(self, img_path: Path) -> Tuple[float, float, str]:
        """
        Extract angles and choice from image using vision network.

        Returns:
            start_angle_rad: Starting angle in radians
            final_angle_rad: Final angle in radians
            choice: 'left' or 'right'
        """
        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found at: {img_path}")
        if not self.model_loaded:
            raise RuntimeError("Model has not been loaded. Call `load_model()` first.")

        input_image = Image.open(img_path).convert("RGB")
        input_tensor = self.image_transform(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._run_vision_inference(input_tensor)

        # Extract angles (in radians)
        predicted_angles = output[:, :self.params.num_angle_outputs].squeeze(0).cpu().numpy()
        start_angle = float(predicted_angles[0])
        final_angle = float(predicted_angles[1])

        # Extract choice
        choice_logits = output[:, self.params.num_angle_outputs:]
        choice_idx = torch.argmax(choice_logits, dim=1).item()
        choice = self.params.choice_labels[choice_idx]

        return start_angle, final_angle, choice

    def image_to_trajectory(self, img_path: Path) -> Tuple[np.ndarray, str]:
        """
        Complete pipeline: image → angles → trajectory.
        This is the main interface used by the larger project.

        Returns:
            trajectory: numpy array of angles in radians over time
            choice: 'left' or 'right'
        """
        # Step 1: Vision network extracts angles and choice
        start_angle_rad, final_angle_rad, choice = self.image_to_angles(img_path)

        # Step 2: Trajectory generator converts angles to full trajectory
        trajectory = self.trajectory_generator.angles_to_trajectory(
            start_angle_rad, final_angle_rad
        )

        return trajectory, choice


class ANNPlanner(Planner):
    """Planner using ANN-based vision network."""

    def __init__(
        self,
        params: PlannerParams,
        trajectory_generator: Optional[TrajectoryGeneratorBase] = None
    ):
        vision_net = ANNVisionNet(params=params)
        super().__init__(params, vision_net, trajectory_generator)

    def _run_vision_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.vision_net(input_tensor)


class GLEPlanner(Planner):
    """Planner using GLE-based vision network."""

    def __init__(
        self,
        params: PlannerParams,
        trajectory_generator: Optional[TrajectoryGeneratorBase] = None
    ):
        vision_net = GLEVisionNet(params=params)
        super().__init__(params, vision_net, trajectory_generator)

    def _run_vision_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = None
        for _ in range(self.params.gle_update_steps):
            output = self.vision_net(input_tensor)
        return output


# Legacy aliases for backward compatibility
ANNPlannerNet = ANNVisionNet
GLEPlannerNet = GLEVisionNet
