import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Assuming these are available in your project structure
from .gle.abstract_net import GLEAbstractNet
from .gle.dynamics import GLEDynamics
from .gle.layers import GLEConv, GLELinear
from .gle.utils import get_phi_and_derivative

# --- Base Class and Network Definitions ---

class TrajectoryPlanner(ABC):
    """Abstract Base Class for trajectory and choice planning models."""
    def __init__(self, net: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net
        self.net.to(self.device)
        self.image_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])

    def load_model(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        print(f"Loading trained model from {model_path} to device '{self.device}'...")
        try:
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.net.eval()
        except RuntimeError as e:
            print(f"Error loading model. Architecture mismatch or corrupted file.")
            raise e

    def save_model(self, model_path: Path):
        """Saves the network's state_dict to the specified path."""
        print(f"Saving model to {model_path}...")
        os.makedirs(model_path.parent, exist_ok=True)
        torch.save(self.net.state_dict(), model_path)

    @abstractmethod
    def plan_from_image(self, image_path: Path, traj_len: int) -> Tuple[np.ndarray, str]:
        """Generates a trajectory and choice from a single input image."""
        raise NotImplementedError

class ANNPlannerNet(nn.Module):
    """The ANN network architecture."""
    def __init__(self, num_choices: int, trajectory_length: int):
        super().__init__()
        self.trajectory_length = trajectory_length
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2), nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1), nn.Tanh(),
            nn.Flatten()
        )
        dummy_input = torch.rand(1, 3, 100, 100)
        conv_output_size = self.conv_layers(dummy_input).size(1)
        self.trajectory_regressor = nn.Linear(conv_output_size, trajectory_length)
        self.choice_classifier = nn.Linear(conv_output_size, num_choices)

    def forward(self, x):
        features = self.conv_layers(x)
        return torch.cat((self.trajectory_regressor(features), self.choice_classifier(features)), dim=1)

class GLEPlannerNet(GLEAbstractNet, nn.Module):
    """The GLE network architecture."""
    def __init__(self, tau: float, dt: float, num_choices: int, trajectory_length: int):
        super().__init__()
        self.trajectory_length = trajectory_length
        self.phi, self.phi_prime = get_phi_and_derivative("tanh")
        self.conv1 = GLEConv(3, 16, kernel_size=5, stride=4, padding=1)
        self.conv2 = GLEConv(16, 32, kernel_size=3, stride=4, padding=1)
        dummy_input = torch.rand(1, 3, 100, 100)
        conv_output_size = self.conv2(self.conv1(dummy_input)).view(1, -1).size(1)
        self.fc = GLELinear(conv_output_size, trajectory_length + num_choices)
        self.conv1_dyn = GLEDynamics(self.conv1, tau_m=tau, dt=dt, phi=self.phi, phi_prime=self.phi_prime)
        self.conv2_dyn = GLEDynamics(self.conv2, tau_m=tau, dt=dt, phi=self.phi, phi_prime=self.phi_prime)
        self.fc_dyn = GLEDynamics(self.fc, tau_m=tau, dt=dt)
    
    # The 'forward' method is now inherited from GLEAbstractNet for inference.

    def compute_target_error(self, output, target, beta):
        e = torch.zeros_like(output)
        e[:, :self.trajectory_length] = 0.001 * (target[:, :self.trajectory_length] - output[:, :self.trajectory_length])
        choice_probs = torch.softmax(output[:, self.trajectory_length:], dim=1)
        target_choice = target[:, self.trajectory_length:]
        e[:, self.trajectory_length:] = target_choice - choice_probs
        return beta * e

# --- Concrete Planner Implementations ---

class ANNPlanner(TrajectoryPlanner):
    """Concrete implementation for the ANN model."""
    def plan_from_image(self, image_path: Path, traj_len: int) -> Tuple[np.ndarray, str]:
        input_image = Image.open(image_path).convert("RGB")
        input_tensor = self.image_transform(input_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.net(input_tensor)
        predicted_traj = output[:, :traj_len].squeeze(0).cpu().numpy()
        choice_logits = output[:, traj_len:]
        choice_idx = torch.argmax(choice_logits, dim=1).item()
        return predicted_traj, 'left' if choice_idx == 0 else 'right'

class GLEPlanner(TrajectoryPlanner):
    """Concrete implementation for the GLE model."""
    def plan_from_image(self, image_path: Path, traj_len: int) -> Tuple[np.ndarray, str]:
        input_image = Image.open(image_path).convert("RGB")
        input_tensor = self.image_transform(input_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            for _ in range(10):  # Run multiple steps for GLE convergence
                output = self.net(input_tensor)
        predicted_traj = output[:, :traj_len].squeeze(0).cpu().numpy()
        choice_logits = output[:, traj_len:]
        choice_idx = torch.argmax(choice_logits, dim=1).item()
        return predicted_traj, 'left' if choice_idx == 0 else 'right'
