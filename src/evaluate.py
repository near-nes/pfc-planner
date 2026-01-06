#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from torchvision import transforms

from . import utils
from .ann_planner import ANNPlanner
from .gle_conv_planner import GLEConvPlanner
from .gle_planner import GLEPlanner
from .dataset import RobotArmDataset, get_image_paths_and_labels

def evaluate_model(model, train_loader, all_image_data, path_prefix=""):
    # print("\n--- Demonstration of Inference ---")
    model.eval()
    correct_choices = 0
    correct_angles = 0
    with torch.no_grad():
        # print("Evaluating on training data:")
        for i, (images, true_trajectory, true_choice_idx) in enumerate(train_loader):
            # compute where this batch starts in the original all_image_data (works only when shuffle=False)
            batch_size = images.shape[0]
            batch_start = i * batch_size

            for k in range(batch_size):
                single_image = images[k].unsqueeze(0)  # Get one image from batch and add batch dim
                single_true_trajectory = true_trajectory[k]
                single_true_choice_idx = true_choice_idx[k]

                # Get the corresponding original item from all_image_data
                original_item_data = all_image_data[batch_start + k]

                # Warm-up / multiple forward passes (if needed)
                for _ in range(20):
                    output = model(single_image)

                # Split output into trajectory and choice logits
                predicted_trajectory_tensor = output[:, :single_true_trajectory.shape[0]]
                pred_choice_logits = output[:, single_true_trajectory.shape[0]:]

                predicted_trajectory = predicted_trajectory_tensor.squeeze(0).cpu().numpy()  # Remove batch dim, to numpy

                _, predicted_choice_idx = torch.max(pred_choice_logits, 1)
                predicted_choice = 'left' if predicted_choice_idx.item() == 0 else 'right'

                true_choice = 'left' if single_true_choice_idx.item() == 0 else 'right'
                correct_choices += (predicted_choice_idx.item() == single_true_choice_idx.item())

                correct_angles += np.isclose(predicted_trajectory[-1], single_true_trajectory[-1].item(), atol=np.deg2rad(1.0))

                # Print detailed results
                # print(f"\n--- Input Image: {os.path.basename(original_item_data['image_path'])} ---")
                # print(f"Initial Angle (Hardcoded): {original_item_data['initial_angle']}°")
                # print(f"Target Final Angle (from filename): {original_item_data['target_final_angle']}°")
                # print(f"Calculated Angle Difference: {original_item_data['angle_difference']}°")
                # print(f"True Choice: {true_choice}")
                # print(f"Predicted Choice (Left/Right): {predicted_choice}")

                # Compare predicted and true trajectories (convert from radians to degrees for readable printing)
                # print(f"True Trajectory (last 5 points): {np.rad2deg(single_true_trajectory.cpu().numpy()[-5:]).tolist()}")
                # print(f"Predicted Trajectory (last 5 points): {np.rad2deg(predicted_trajectory[-5:]).tolist()}")

                # Optional: Plot the trajectories to visualize the fit (display in degrees)
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(np.rad2deg(single_true_trajectory.cpu().numpy()), label='True Trajectory')
                plt.plot(np.rad2deg(predicted_trajectory), label='Predicted Trajectory')
                plt.title(f"Trajectory for {os.path.basename(original_item_data['image_path'])}")
                plt.xlabel("Time Step")
                plt.ylabel("Elbow Angle (deg)")
                plt.legend()
                os.makedirs(os.path.join(path_prefix, "results"), exist_ok=True)
                plt.savefig(os.path.join(path_prefix, f"results/{os.path.basename(original_item_data['image_path']).removesuffix('.bmp')}_trajectory.png"))
                # plt.show()
                plt.close()  # Close the plot to free memory

        choice_accuracy = correct_choices / len(all_image_data)
        angle_accuracy = correct_angles / len(all_image_data)
        print(f"\nOverall Choice Prediction Accuracy: {choice_accuracy*100:.2f}%")
        print(f"Overall Final Angle Prediction Accuracy (within 1 degree): {angle_accuracy*100:.2f}%")

if __name__ == '__main__':
    # get which model to evaluate from command line args
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Planner Models for Robotic Arm")
    parser.add_argument('--model', type=str, choices=['ann', 'gle', 'gle_conv'], default='gle_conv', help="Model type to evaluate")
    args = parser.parse_args()

    print("Evaluating Planner models for Robotic Arm...")
    # Define your data directory relative to where you run this script
    EXPERIMENT_DIR = "submodules/pfc_planner"  # Make sure this path is correct
    DATA_DIR = os.path.join(EXPERIMENT_DIR, "data/")

    # Load image data
    all_image_data = get_image_paths_and_labels(DATA_DIR)

    print(f"Loaded {len(all_image_data)} distinct data samples for training.")
    if not all_image_data:
        print("No image data found. Please check DATA_DIR and filename patterns.")
        sys.exit(1)

    # Dynamically get trajectory length from the first data sample
    TRAJECTORY_LEN = all_image_data[0]['trajectory_len']
    print(f"Detected trajectory length: {TRAJECTORY_LEN}")

    image_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    train_dataset = RobotArmDataset(all_image_data, transform=image_transform)

    # IMPORTANT: Do not shuffle during evaluation if you rely on all_image_data ordering.
    # During training you should use shuffle=True; here we evaluate on a fixed order.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(all_image_data), shuffle=False)  # Use full batch for this small dataset

    num_choices = 2

    if args.model == 'gle_conv':
        model = GLEConvPlanner(tau=1.0, dt=0.01, num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)
        try:
            model.load_state_dict(torch.load(os.path.join(EXPERIMENT_DIR, 'models/trained_gle_conv_planner.pth')))
        except FileNotFoundError:
            print("GLE model file not found. Please ensure the model is trained and saved correctly.")
            sys.exit(1)
    elif args.model == 'gle':
        from .gle_planner import GLEPlanner
        model = GLEPlanner(tau=1.0, dt=0.01, num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)
        try:
            model.load_state_dict(torch.load(os.path.join(EXPERIMENT_DIR, 'models/trained_gle_planner.pth')))
        except FileNotFoundError:
            print("GLE model file not found. Please ensure the model is trained and saved correctly.")
            sys.exit(1)
    elif args.model == 'ann':
        model = ANNPlanner(num_choices=num_choices, trajectory_length=TRAJECTORY_LEN)
        try:
            model.load_state_dict(torch.load(os.path.join(EXPERIMENT_DIR, 'models/trained_ann_planner.pth')))
        except FileNotFoundError:
            print("ANN model file not found. Please ensure the model is trained and saved correctly.")
            sys.exit(1)

    print(f"Evaluating model: {model.__class__.__name__}")
    evaluate_model(model, train_loader, all_image_data, path_prefix=EXPERIMENT_DIR)
    print(f"Finished evaluating model: {model.__class__.__name__}\n")
