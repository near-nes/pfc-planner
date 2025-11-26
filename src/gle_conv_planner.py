import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from ..gle.abstract_net import GLEAbstractNet
from ..gle.dynamics import GLEDynamics
from ..gle.layers import GLEConv, GLELinear
from ..gle.utils import get_phi_and_derivative
from .ann_planner import RobotArmDataset
from . import utils


class GLEConvPlanner(GLEAbstractNet, torch.nn.Module):
    def __init__(self, *, tau, dt, num_choices=2, trajectory_length=100):
        super().__init__()
        self.trajectory_length = trajectory_length
        self.num_choices = num_choices
        self.tau = tau
        self.dt = dt

        self.phi, self.phi_prime = get_phi_and_derivative("relu")

        # Convolutional layers
        self.conv1 = GLEConv(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = GLEConv(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = GLEConv(32, 64, kernel_size=3, stride=2, padding=1)

        # Calculate the flattened size after conv layers
        # Input: 100x100
        # After conv1 (stride 2): 50x50
        # After conv2 (stride 2): 25x25
        # After conv3 (stride 2): 13x13
        self.flattened_size = 64 * 13 * 13

        # Linear layer
        self.fc = GLELinear(self.flattened_size, trajectory_length + num_choices)

        # Dynamics for each layer
        self.conv1_dyn = GLEDynamics(
            self.conv1,
            tau_m=self.tau,
            dt=self.dt,
            phi=self.phi,
            phi_prime=self.phi_prime,
        )
        self.conv2_dyn = GLEDynamics(
            self.conv2,
            tau_m=self.tau,
            dt=self.dt,
            phi=self.phi,
            phi_prime=self.phi_prime,
        )
        self.conv3_dyn = GLEDynamics(
            self.conv3,
            tau_m=self.tau,
            dt=self.dt,
            phi=self.phi,
            phi_prime=self.phi_prime,
        )
        self.fc_dyn = GLEDynamics(self.fc, tau_m=self.tau, dt=self.dt)

    def compute_target_error(self, output, target, beta):
        e = torch.zeros_like(output)
        # MSE for trajectory part
        e[:, : self.trajectory_length] = (
            target[:, : self.trajectory_length] - output[:, : self.trajectory_length]
        )
        # CE for choice part
        choice_probs = torch.softmax(output[:, self.trajectory_length :], dim=1)
        target_choice = target[:, self.trajectory_length :]
        e[:, self.trajectory_length :] = target_choice - choice_probs
        return beta * e


if __name__ == "__main__":
    print("Starting GLE Convolutional Planner for Robotic Arm...")
    EXPERIMENT_DIR = "submodules/pfc_planner"
    DATA_DIR = os.path.join(EXPERIMENT_DIR, "data/")
    print("Using data from:", DATA_DIR)

    all_image_data = utils.get_image_paths_and_labels(DATA_DIR)
    print(f"Loaded {len(all_image_data)} distinct data samples for training.")
    if not all_image_data:
        print("No image data found. Please check DATA_DIR.")
        sys.exit(1)

    TRAJECTORY_LEN = all_image_data[0]['trajectory_len']
    print(f"Detected trajectory length: {TRAJECTORY_LEN}")

    # Image transform without flattening
    image_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RobotArmDataset(all_image_data, transform=image_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(all_image_data), shuffle=True
    )

    num_choices = 2
    model = GLEConvPlanner(tau=1.0, dt=0.01, num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)

    UPDATE_STEPS = 20
    criterion_trajectory = nn.MSELoss()
    criterion_choice = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500
    print("\nStarting online training for GLEConvPlanner...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, true_trajectory, target_choice_idx) in enumerate(train_loader):
            optimizer.zero_grad()

            target = torch.cat(
                (true_trajectory, torch.nn.functional.one_hot(target_choice_idx, num_classes=num_choices)),
                dim=1,
            )
            
            # The input `images` are now 4D tensors [batch, channels, height, width]
            with torch.no_grad():
                for _ in range(UPDATE_STEPS):
                    output = model(images, target, beta=1.0)
                optimizer.step()
                
            predicted_trajectory = output[:, :TRAJECTORY_LEN]
            choice_logits = output[:, TRAJECTORY_LEN:]

            loss_trajectory = criterion_trajectory(predicted_trajectory, true_trajectory)
            loss_choice = criterion_choice(choice_logits, target_choice_idx)
            total_loss = loss_trajectory + loss_choice
            running_loss += total_loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}")

    print("\nTraining finished.")

    MODEL_SAVE_PATH = os.path.join(EXPERIMENT_DIR, "models/trained_gle_conv_planner.pth")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    from .evaluate import evaluate_model
    # Note: evaluate_model needs to handle un-flattened images if it uses the model directly
    # Creating a new loader with flattened images for evaluation consistency with other models
    eval_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    eval_dataset = RobotArmDataset(all_image_data, transform=eval_transform)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=len(all_image_data))
    
    # We need to adapt the evaluation if the model expects 4D input
    # For now, assuming evaluate_model can take the model and a loader
    print("Skipping evaluation for GLEConvPlanner as evaluate.py may need adjustments for 4D input.")