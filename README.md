# Robotic Arm Trajectory Planner

This project implements a neural network-based planner for a simplified 1-Degree-of-Freedom (DoF) robotic arm (elbow joint), designed to generate smooth trajectories and make task-related choices based on visual input.

## Project Overview

The core of this project is a Planner module, implemented using PyTorch, which takes a 2D image of the environment as input and produces:

1. Elbow Joint Trajectory: A smooth, sigmoidal (minimum jerk) trajectory for the elbow joint over a predefined duration.
2. Multiclass Choice: A "left" or "right" decision based on the color of a target object in the environment.

Currently, there are two planners implemented:
- **ANNPlanner**: An ANN-based planner that uses a CNN + MLP architecture to process the input image and generate the trajectory and decision. This serves as a baseline for the GLEPlanner.
- **GLEPlanner**: A planner with similar architecture than the ANN planner that uses GLE dynamics and learning to generate the trajectory and make decisions.

## How to use

### Inside the Docker Container
The code in this repository is supposed to be run from within [near-nes/controller](https://github.com/near-nes/controller) docker container. All commands should be run from the `submodules/pfc_planner` directory.

First, generate the required image dataset:
```bash
python imagedata_gen.py
```

Then train the GLEPlanner components (vision network and trajectory generator):
```bash
python -m pfc_planner.train_vision --type gle
python -m pfc_planner.train_trajectory --type gle
```

To test the pretrained and saved GLEPlanner model:
```bash
python -m pfc_planner.evaluate --model gle --traj-gen gle
```

### Locally
The code can also be run outside of the docker container.

Install the package in a virtual environment and activate it:

```bash
uv venv
uv pip install -e .
source .venv/bin/activate
```

If you want to run the included notebook (`planner_demo.ipynb`), install with the `notebook` extra:

```bash
uv pip install -e ".[notebook]"
```

Generate the required image dataset:
```bash
python imagedata_gen.py
```

Then train the GLEPlanner components (vision network and trajectory generator):
```bash
python -m pfc_planner.train_vision --type gle
python -m pfc_planner.train_trajectory --type gle
```

To test the pretrained and saved GLEPlanner model:
```bash
python -m pfc_planner.evaluate --model gle --traj-gen gle
```
