import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

from .gle.abstract_net import GLEAbstractNet
from .gle.dynamics import GLEDynamics
from .gle.layers import GLEConv, GLELinear
from .gle.utils import get_phi_and_derivative
from .dataset import RobotArmDataset


class GLEPlanner(GLEAbstractNet, torch.nn.Module):
    def __init__(self, *, tau, dt, num_choices=2, trajectory_length=100):
        super().__init__()
        self.trajectory_length = trajectory_length
        self.num_choices = num_choices
        self.tau = tau
        self.dt = dt

        self.phi, self.phi_prime = get_phi_and_derivative("tanh")

        # Convolutional layers
        self.conv1 = GLEConv(3, 16, kernel_size=5, stride=4, padding=1)
        self.conv2 = GLEConv(16, 32, kernel_size=3, stride=4, padding=1)

        # Calculate input features for the linear layers
        self._dummy_input_shape = (1, 3, 100, 100) # Assuming 100x100 images
        dummy_input = torch.rand(self._dummy_input_shape)
        self.conv_output_size = self.conv2(self.conv1(dummy_input)).view(1, -1).size(1)

        # Linear layer
        self.fc = GLELinear(self.conv_output_size, trajectory_length + num_choices)

        # Dynamics for each layer
        self.conv1_dyn = GLEDynamics(self.conv1, tau_m=self.tau, dt=self.dt, phi=self.phi, phi_prime=self.phi_prime)
        self.conv2_dyn = GLEDynamics(self.conv2, tau_m=self.tau, dt=self.dt, phi=self.phi, phi_prime=self.phi_prime)
        self.fc_dyn = GLEDynamics(self.fc, tau_m=self.tau, dt=self.dt)

    def compute_target_error(self, output, target, beta):
        e = torch.zeros_like(output)
        # MSE for trajectory part
        e[:, : self.trajectory_length] = 0.001 * (target[:, : self.trajectory_length] - output[:, : self.trajectory_length])
        # CE for choice part
        # convert logits to probabilities
        choice_probs = torch.softmax(output[:, self.trajectory_length :], dim=1)
        target_choice = target[:, self.trajectory_length :]
        # error for choice part
        e[:, self.trajectory_length :] = target_choice - choice_probs
        # Scale the error by beta
        return beta * e


if __name__ == "__main__":
    print("Starting GLE Planner for Robotic Arm...")
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    EXPERIMENT_DIR = "submodules/pfc_planner"
    if not os.path.exists(EXPERIMENT_DIR):
        EXPERIMENT_DIR = "./"  # local testing fallback
    DATA_DIR = os.path.join(EXPERIMENT_DIR, "data/")
    print("Using data from:", os.path.abspath(DATA_DIR))

    image_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    train_dataset = RobotArmDataset(data_dir=DATA_DIR, transform=image_transform)
    
    if len(train_dataset) == 0:
        print("No image data found. Please check DATA_DIR.")
        sys.exit(1)
        
    print(f"Loaded {len(train_dataset)} distinct data samples for training.")

    TRAJECTORY_LEN = len(train_dataset[0][1])
    print(f"Detected trajectory length: {TRAJECTORY_LEN}")

    all_image_data = train_dataset.task_data

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    num_choices = 2
    # Pass trajectory_length to the model
    model = GLEPlanner(tau=1.0, dt=0.1, num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)

    # MSELoss for the trajectory regression (comparing sequences)
    criterion_trajectory = nn.MSELoss()
    # CrossEntropyLoss for the choice classification
    criterion_choice = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    UPDATE_STEPS = 10
    num_epochs = 500
    print("\nStarting online training for GLEPlanner...")
    loss_history = []
    trajectory_loss_history = []
    choice_loss_history = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_trajectory_loss = 0.0
        running_choice_loss = 0.0
        for i, (images, true_trajectory, target_choice_idx) in enumerate(train_loader):
            optimizer.zero_grad()

            # one-hot encoding for target choice index
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
            running_trajectory_loss += loss_trajectory.item()
            running_choice_loss += loss_choice.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}, Trajectory Loss: {running_trajectory_loss/len(train_loader):.6f}, Choice Loss: {running_choice_loss/len(train_loader):.6f}")
        loss_history.append(running_loss / len(train_loader))
        trajectory_loss_history.append(running_trajectory_loss / len(train_loader))
        choice_loss_history.append(running_choice_loss / len(train_loader))

    print("\nTraining finished.")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Total Loss')
    plt.plot(trajectory_loss_history, label='Trajectory Loss')
    plt.plot(choice_loss_history, label='Choice Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History for GLEPlanner')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(EXPERIMENT_DIR, 'results/gle_planner_training_loss.png'), dpi=300)
    plt.close()

    MODEL_SAVE_PATH = os.path.join(EXPERIMENT_DIR, "models/trained_gle_planner.pth")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    from .evaluate import evaluate_model
    eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    evaluate_model(model, eval_loader, all_image_data, path_prefix=EXPERIMENT_DIR)
