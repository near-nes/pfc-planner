import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from . import utils
from .ann_planner import RobotArmDataset
from ..gle.abstract_net import GLEAbstractNet
from ..gle.dynamics import GLEDynamics
from ..gle.layers import GLELinear
from ..gle.utils import get_phi_and_derivative


class GLEPlanner(GLEAbstractNet, torch.nn.Module):
    def __init__(self, *, tau, dt, num_choices=2, trajectory_length=100):
        super().__init__()
        self.trajectory_length = trajectory_length
        self.num_choices = num_choices

        self.tau = tau
        self.dt = dt

        self.phi, self.phi_prime = get_phi_and_derivative("relu")

        self.input_size = 100 * 100 * 3
        self.output_size = 10

        self.first = GLELinear(self.input_size, 300)
        self.hidden = GLELinear(300, 100)
        self.last = GLELinear(100, trajectory_length + num_choices)

        self.first_dyn = GLEDynamics(
            self.first,
            tau_m=self.tau,
            dt=self.dt,
            phi=self.phi,
            phi_prime=self.phi_prime,
        )
        self.hidden_dyn = GLEDynamics(
            self.hidden,
            tau_m=self.tau,
            dt=self.dt,
            phi=self.phi,
            phi_prime=self.phi_prime,
        )
        self.last_dyn = GLEDynamics(self.last, tau_m=self.tau, dt=self.dt)

    def compute_target_error(self, output, target, beta):
        e = torch.zeros_like(output)
        # MSE for trajectory part
        e[:, : self.trajectory_length] = (
            target[:, : self.trajectory_length] - output[:, : self.trajectory_length]
        )
        # CE for choice part
        # convert logits to probabilities
        choice_probs = torch.softmax(output[:, self.trajectory_length :], dim=1)
        target_choice = target[
            :, self.trajectory_length :
        ]  # This is the one-hot encoded choice part
        # error for choice part
        e[:, self.trajectory_length :] = target_choice - choice_probs
        # Scale the error by beta
        return beta * e


if __name__ == "__main__":
    print("Starting GLE Planner for Robotic Arm...")
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    # Define your data directory relative to where you run this script
    EXPERIMENT_DIR = "submodules/pfc_planner"  # Make sure this path is correct
    DATA_DIR = os.path.join(EXPERIMENT_DIR, "data/")
    print("Using data from:", DATA_DIR)

    # Load image data
    all_image_data = utils.get_image_paths_and_labels(DATA_DIR)

    print(f"Loaded {len(all_image_data)} distinct data samples for training.")
    if not all_image_data:
        print("No image data found. Please check DATA_DIR and filename patterns.")
        sys.exit(1)

    # Dynamically get trajectory length from the first data sample
    TRAJECTORY_LEN = all_image_data[0]['trajectory_len']
    print(f"Detected trajectory length: {TRAJECTORY_LEN}")

    image_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten the image to a vector
    ])

    train_dataset = RobotArmDataset(all_image_data, transform=image_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(all_image_data), shuffle=True
    )  # Use full batch for this small dataset

    num_choices = 2
    # Pass trajectory_length to the model
    model = GLEPlanner(tau=1.0, dt=0.01, num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)

    UPDATE_STEPS = 10
    criterion_trajectory = nn.MSELoss()
    # CrossEntropyLoss for the choice classification
    criterion_choice = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500
    print("\nStarting online training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, true_trajectory, target_choice_idx) in enumerate(train_loader):
            optimizer.zero_grad()

            # one-hot encoding for target choice index
            target = torch.cat(
                (true_trajectory, torch.nn.functional.one_hot(target_choice_idx, num_classes=num_choices)),
                dim=1,
            )

            # Model outputs predicted_trajectory (tensor of shape [batch_size, TRAJECTORY_LEN]) and choice_logits (tensor)
            with torch.no_grad():
                for _ in range(UPDATE_STEPS):
                    output = model(images, target, beta=1.0)
                optimizer.step()

            predicted_trajectory = output[
                :, :TRAJECTORY_LEN
            ]  # First part is the trajectory
            choice_logits = output[
                :, TRAJECTORY_LEN:
            ]  # Second part is the choice logits

            loss_trajectory = criterion_trajectory(predicted_trajectory, true_trajectory)
            loss_choice = criterion_choice(choice_logits, target_choice_idx)
            total_loss = loss_trajectory + loss_choice

            running_loss += total_loss.item()

        # Print loss less frequently due to high epoch count
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}")

    print("\nTraining finished.")

    MODEL_SAVE_PATH = os.path.join(EXPERIMENT_DIR, "models/trained_gle_planner.pth")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    from .evaluate import evaluate_model
    evaluate_model(model, train_loader, all_image_data, path_prefix=EXPERIMENT_DIR)
