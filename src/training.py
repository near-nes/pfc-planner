"""Shared training module for vision and trajectory models."""

from .train_vision import train_vision_network, get_project_root, get_git_commit_hash
from .train_trajectory import train_trajectory_generator

__all__ = [
    "train_vision_network",
    "train_trajectory_generator",
    "get_project_root",
    "get_git_commit_hash"
]
