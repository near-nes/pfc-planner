#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from .ann_planner import ANNPlanner
from .gle_planner import GLEPlanner
from .dataset import RobotArmDataset

def evaluate_model(model, eval_loader, all_image_data, path_prefix=""):
    """
    Evaluates the model's performance on choice and trajectory prediction.
    """
    print(f"\n--- Evaluating Model: {model.__class__.__name__} ---")
    model.eval()
    correct_choices = 0
    correct_angles = 0

    # Create the results directory for saving plots
    results_dir = os.path.join(path_prefix, "results")
    os.makedirs(results_dir, exist_ok=True)

    with torch.no_grad():
        # We assume one large batch for evaluation, as in the main script
        images, true_trajectory, true_choice_idx = next(iter(eval_loader))

        # Determine trajectory length from the data itself
        TRAJECTORY_LEN = true_trajectory.shape[1]

        for k in range(len(all_image_data)):
            # Get single items for evaluation
            single_image = images[k].unsqueeze(0)
            single_true_trajectory = true_trajectory[k]
            single_true_choice_idx = true_choice_idx[k]

            # Get the corresponding metadata for this item
            item_metadata = all_image_data[k]

            # For GLE models, multiple forward passes might be needed for the state to settle
            if isinstance(model, GLEPlanner):
                for _ in range(10): # Settle the network state
                    output = model(single_image, beta=0.0) # In inference, beta is 0
            else: # For ANN, a single forward pass is enough
                output = model(single_image)

            # Split the model output into trajectory and choice predictions
            predicted_trajectory_tensor = output[:, :TRAJECTORY_LEN]
            pred_choice_logits = output[:, TRAJECTORY_LEN:]

            predicted_trajectory = predicted_trajectory_tensor.squeeze(0).cpu().numpy()
            _, predicted_choice_idx = torch.max(pred_choice_logits, 1)

            correct_choices += (predicted_choice_idx.item() == single_true_choice_idx.item())
            # Check if the final predicted angle is within 1 degree of the true final angle
            is_close = np.isclose(
                predicted_trajectory[-1],
                single_true_trajectory[-1].item(),
                atol=np.deg2rad(1.0) # tolerance of 1 degree, converted to radians
            )
            correct_angles += is_close

            # Plotting the trajectories
            plt.figure(figsize=(10, 6))
            # Convert trajectories from radians to degrees for plotting
            plt.plot(np.rad2deg(single_true_trajectory.cpu().numpy()), label='True Trajectory', color='blue', linewidth=2)
            plt.plot(np.rad2deg(predicted_trajectory), label='Predicted Trajectory', color='red', linestyle='--')

            # Use new dictionary keys for metadata
            plot_title = f"Trajectory for {os.path.basename(item_metadata['image_path'])}"
            plt.title(plot_title)
            plt.xlabel("Time Step")
            plt.ylabel("Elbow Angle (degrees)")
            plt.legend()
            plt.grid(True)

            # Save the plot
            plot_filename = f"{os.path.basename(item_metadata['image_path']).removesuffix('.bmp')}_trajectory.png"
            plt.savefig(os.path.join(results_dir, plot_filename))
            plt.close()

    choice_accuracy = correct_choices / len(all_image_data)
    angle_accuracy = correct_angles / len(all_image_data)
    print(f"Overall Choice Prediction Accuracy: {choice_accuracy*100:.2f}%")
    print(f"Final Angle Accuracy (within 1 degree): {angle_accuracy*100:.2f}%")
    print(f"Plots saved to '{results_dir}'")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Planner Models for Robotic Arm")
    parser.add_argument('--model', type=str, choices=['ann', 'gle'], default='gle', help="Model type to evaluate")
    args = parser.parse_args()

    print("Initializing evaluation for Robotic Arm Planners...")
    EXPERIMENT_DIR = "submodules/pfc_planner"
    DATA_DIR = os.path.join(EXPERIMENT_DIR, "data/")
    print(f"Using data from: {DATA_DIR}")

    image_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    # The dataset now handles loading and processing internally.
    eval_dataset = RobotArmDataset(data_dir=DATA_DIR, transform=image_transform)

    if not eval_dataset.task_data:
        print("No image data found. Please check the DATA_DIR path and filename patterns.")
        sys.exit(1)

    all_image_data = eval_dataset.task_data
    TRAJECTORY_LEN = len(all_image_data[0]['ground_truth_trajectory_rad'])

    print(f"Loaded {len(all_image_data)} data samples.")
    print(f"Detected trajectory length: {TRAJECTORY_LEN}")

    # IMPORTANT: shuffle=False is critical for aligning plots with filenames.
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=len(all_image_data), shuffle=False)

    num_choices = 2

    # Load the specified model
    if args.model == 'gle':
        model = GLEPlanner(tau=1.0, dt=0.01, num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)
        model_path = os.path.join(EXPERIMENT_DIR, 'models/trained_gle_planner.pth')
    elif args.model == 'ann':
        model = ANNPlanner(num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)
        model_path = os.path.join(EXPERIMENT_DIR, 'models/trained_ann_planner.pth')

    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}.")
        print("Please ensure the model is trained and saved correctly before evaluation.")
        sys.exit(1)

    # Run the evaluation
    evaluate_model(model, eval_loader, all_image_data, path_prefix=EXPERIMENT_DIR)

    print("\nEvaluation finished.")
