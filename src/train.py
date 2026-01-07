import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

from .planners import ANNPlannerNet, GLEPlannerNet
from .dataset import RobotArmDataset

def get_project_root() -> Path:
    """
    Determines the project root directory by checking for a primary path
    and falling back to the current directory if not found.
    """
    primary_path = Path("submodules/pfc_planner")
    if primary_path.exists() and primary_path.is_dir():
        # Use the submodule path if it exists
        print(f"Using primary project path: {primary_path.resolve()}")
        return primary_path.resolve()
    else:
        # Fallback to the current directory for standalone execution
        print("WARNING: Primary project path not found. Using current directory as project root.")
        return Path(".").resolve()

def main():
    """Main function to handle training of a selected planner model."""
    parser = argparse.ArgumentParser(description="Train Planner Models for Robotic Arm")
    parser.add_argument('--model', type=str, choices=['ann', 'gle'], default='gle', help="Model type to train")
    args = parser.parse_args()

    print(f"--- Starting Training for {args.model.upper()} Planner ---")

    PROJECT_ROOT = get_project_root()
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RobotArmDataset(data_dir=str(DATA_DIR), transform=transforms.Compose([
        transforms.Resize((100, 100)), transforms.ToTensor()
    ]))

    if len(train_dataset) == 0: sys.exit(f"ERROR: No data found in {DATA_DIR}. Exiting.")

    TRAJECTORY_LEN = len(train_dataset[0][1])
    print(f"Loaded {len(train_dataset)} samples. Trajectory length: {TRAJECTORY_LEN}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    
    num_choices = 2
    if args.model == 'ann':
        net = ANNPlannerNet(num_choices=num_choices, trajectory_length=TRAJECTORY_LEN).to(device)
    else: # gle
        net = GLEPlannerNet(tau=1.0, dt=0.1, num_choices=num_choices, trajectory_length=TRAJECTORY_LEN).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    criterion_trajectory = nn.MSELoss()
    criterion_choice = nn.CrossEntropyLoss()

    num_epochs, gle_update_steps = 500, 10
    loss_history, traj_loss_history, choice_loss_history = [], [], []

    print(f"\nStarting {args.model.upper()} training on device '{device}'...")
    for epoch in range(num_epochs):
        net.train()
        running_loss, running_traj_loss, running_choice_loss = 0.0, 0.0, 0.0
        
        for images, true_trajectory, target_choice_idx in train_loader:
            images, true_trajectory, target_choice_idx = images.to(device), true_trajectory.to(device), target_choice_idx.to(device)
            optimizer.zero_grad()
            
            if args.model == 'ann':
                output = net(images)
                trajectory_loss = criterion_trajectory(output[:, :TRAJECTORY_LEN], true_trajectory)
                choice_loss = criterion_choice(output[:, TRAJECTORY_LEN:], target_choice_idx)
                total_loss = trajectory_loss + choice_loss
                total_loss.backward();
                optimizer.step()
            else: # gle
                target = torch.cat((true_trajectory, torch.nn.functional.one_hot(target_choice_idx, num_classes=num_choices)), dim=1)
                for _ in range(gle_update_steps):
                    output = net(images, target, beta=1.0)
                optimizer.step()
                trajectory_loss = criterion_trajectory(output[:, :TRAJECTORY_LEN], true_trajectory)
                choice_loss = criterion_choice(output[:, TRAJECTORY_LEN:], target_choice_idx)
                total_loss = trajectory_loss + choice_loss

            running_loss += total_loss.item();
            running_traj_loss += trajectory_loss.item();
            running_choice_loss += choice_loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_traj_loss = running_traj_loss / len(train_loader)
        epoch_choice_loss = running_choice_loss / len(train_loader)
        
        loss_history.append(epoch_loss); traj_loss_history.append(epoch_traj_loss); choice_loss_history.append(epoch_choice_loss)
        
        # --- MODIFIED PRINT STATEMENT ---
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1: >3}/{num_epochs} | Total Loss: {epoch_loss:.6f} | Traj Loss: {epoch_traj_loss:.6f} | Choice Loss: {epoch_choice_loss:.6f}")
        # --- END OF MODIFICATION ---

    print("\n--- Training Finished ---")
    model_save_path = MODELS_DIR / f"trained_{args.model}_planner.pth"
    torch.save(net.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Total Loss'); plt.plot(traj_loss_history, label='Trajectory Loss'); plt.plot(choice_loss_history, label='Choice Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Training Loss for {args.model.upper()} Planner'); plt.legend(); plt.grid(True)
    plt.savefig(RESULTS_DIR / f'{args.model}_planner_training_loss.png')
    plt.close()
    print(f"Training plot saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
