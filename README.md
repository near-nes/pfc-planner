# Robotic Arm Trajectory Planner

This project implements a neural network-based planner for a simplified 1-Degree-of-Freedom (DoF) robotic arm (elbow joint), designed to generate smooth trajectories and make task-related choices based on visual input.

## Project Overview

The core of this project is a Planner module, implemented using PyTorch, which takes a 2D image of the environment as input and produces:

1. Elbow Joint Trajectory: A smooth, sigmoidal trajectory for the elbow joint over a predefined duration.
2. Multiclass Choice: A "left" or "right" decision based on the color of a target object in the environment.

This simplified version focuses on movements from a 90° starting position to either a 20° (extension) or 140° (flexion) target angle, based on the color of a blue or red target ball.

Currently, there are two planners implemented:
- **ANNPlanner**: An ANN-based planner that uses a CNN + MLP architecture to process the input image and generate the trajectory and decision. This serves as a baseline for the GLEPlanner.
- **GLEPlanner**: A planner with similar architecture than the ANN planner taht uses GLE dynamics and learning to generate the trajectory and make decisions.

## How to use

### Inside the Docker Container
The code in this repository is supposed to be run from within [near-nes/controller](https://github.com/near-nes/controller) docker container's project root `/sim/controller` via:

```bash
python -m submodules.pfc_planner.src.train --model gle
```

To test the pretrained and saved GLEPlanner model, run:

```bash
python -m submodules.pfc_planner.src.evaluate --model gle
```

### Locally
The code can also be run outside of the docker container using a fallback implementation for the minjerk trajectory generator.

Install requirements locally in a virtual environment and activate it:

```bash
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

Then run train the GLEPlanner outside of the docker container with:
```bash
python -m src.train --model gle
```

To test the pretrained and saved GLEPlanner model locally, run:
```bash
python -m src.evaluate --model gle
```
