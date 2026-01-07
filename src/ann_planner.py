import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

from .dataset import RobotArmDataset

class ANNPlanner(nn.Module):
    def __init__(self, num_choices, trajectory_length):
        super(ANNPlanner, self).__init__()
        self.num_choices = num_choices
        self.trajectory_length = trajectory_length
        # Simplified CNN for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2), # Output: 16x25x25
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1), # Output: 32x7x7
            nn.Tanh(),
            nn.Flatten()
        )
        # Calculate input features for the linear layers
        self._dummy_input_shape = (1, 3, 100, 100) # Assuming 100x100 images
        dummy_input = torch.rand(self._dummy_input_shape)
        conv_output_size = self.conv_layers(dummy_input).size(1)

        # Output for *entire trajectory* (regression of a vector)
        # This layer will output 'trajectory_length' number of values
        self.trajectory_regressor = nn.Linear(conv_output_size, trajectory_length)

        # Output for choice (left/right - classification)
        self.choice_classifier = nn.Linear(conv_output_size, num_choices)

    def forward(self, x):
        features = self.conv_layers(x)
        predicted_trajectory = self.trajectory_regressor(features)
        choice_logits = self.choice_classifier(features)
        # to align with GLE we return a single tensor
        return torch.cat((predicted_trajectory, choice_logits), dim=1)

if __name__ == "__main__":
    print("Starting ANN Planner for Robotic Arm...")
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    # Define your data directory relative to where you run this script
    EXPERIMENT_DIR = "submodules/pfc_planner"
    DATA_DIR = os.path.join(EXPERIMENT_DIR, "data/")
    print("Using data from:", DATA_DIR)

    image_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    train_dataset = RobotArmDataset(data_dir=DATA_DIR, transform=image_transform)

    if len(train_dataset) == 0:
        print("No image data found. Please check DATA_DIR and filename patterns.")
        sys.exit(1)

    print(f"Loaded {len(train_dataset)} distinct data samples for training.")

    # The trajectory is the second element of the tuple returned by __getitem__
    TRAJECTORY_LEN = len(train_dataset[0][1])
    print(f"Detected trajectory length: {TRAJECTORY_LEN}")

    all_image_data = train_dataset.task_data

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    num_choices = 2
    # Pass trajectory_length to the model
    model = ANNPlanner(num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)

    # MSELoss for the trajectory regression (comparing sequences)
    criterion_trajectory = nn.MSELoss()
    # CrossEntropyLoss for the choice classification
    criterion_choice = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 500  # More epochs might be needed for sequence regression
    print("\nStarting offline ANN training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_trajectory_loss = 0.0
        running_choice_loss = 0.0
        for i, (images, true_trajectory, target_choice_idx) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(images)
            predicted_trajectory = output[:, :TRAJECTORY_LEN]  # First part is the trajectory
            choice_logits = output[:, TRAJECTORY_LEN:]  # Second part is the choice logits

            loss_trajectory = criterion_trajectory(predicted_trajectory, true_trajectory)
            loss_choice = criterion_choice(choice_logits, target_choice_idx)
            total_loss = loss_trajectory + loss_choice
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_trajectory_loss += loss_trajectory.item()
            running_choice_loss += loss_choice.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}, Trajectory Loss: {running_trajectory_loss/len(train_loader):.6f}, Choice Loss: {running_choice_loss/len(train_loader):.6f}")

    print("\nTraining finished.")

    MODEL_SAVE_PATH = os.path.join(EXPERIMENT_DIR, "models/trained_ann_planner.pth")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    from .evaluate import evaluate_model
    eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    evaluate_model(model, eval_loader, all_image_data, path_prefix=EXPERIMENT_DIR)
