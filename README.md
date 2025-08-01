# Robotic Arm Trajectory Planner

This project implements a neural network-based planner for a simplified 1-Degree-of-Freedom (DoF) robotic arm (elbow joint), designed to generate smooth trajectories and make task-related choices based on visual input.

## Project Overview

The core of this project is a Planner module, implemented using PyTorch, which takes a 2D image of the environment as input and produces:

1. Elbow Joint Trajectory: A smooth, sigmoidal trajectory for the elbow joint over a predefined duration.
2. Multiclass Choice: A "left" or "right" decision based on the color of a target object in the environment.

This simplified version focuses on movements from a 90° starting position to either a 20° (extension) or 140° (flexion) target angle, based on the color of a blue or red target ball.

Currently, there are two planners implemented:
- **ANNPlanner**: An ANN-based planner that uses a CNN + MLP architecture to process the input image and generate the trajectory and decision. This serves as a baseline for the GLEPlanner.
- **GLEPlanner**: A planner that uses GLE dynamics and learning to generate the trajectory and make decisions.

## How to use

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Run the planner with the following command:

```bash
python gle_planner.py
```

Alternatively, use the provided Jupyter Notebook for interactive exploration, for example:

```bash
jupyter notebook gle_planner.ipynb
```
