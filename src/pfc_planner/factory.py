"""Factory function to get or train planners."""

import json
from dataclasses import asdict
from pathlib import Path

import structlog

from .config import PlannerParams
from .planners import ANNPlanner, GLEPlanner
from .training import (
    train_vision_network,
    train_trajectory_generator,
    get_project_root,
    get_git_commit_hash
)
from .trajectory_generators import get_trajectory_generator

_log: structlog.stdlib.BoundLogger = structlog.get_logger("pfc_planner.factory")


def get_planner(
    params: PlannerParams,
    model_dir: Path = None,
    project_root: Path = None,
    skip_cache: bool = False,
):
    """
    Get a fully configured planner, training components if necessary.

    Args:
        params: PlannerParams with required configuration.
        model_dir: Directory where models are stored.
        project_root: Root directory of the project.
        skip_cache: If True, force retraining of components.
    """
    if project_root is None:
        project_root = get_project_root()
    if model_dir is None:
        model_dir = project_root / "models"

    params.git_commit = get_git_commit_hash(project_root)

    # 1. Ensure Vision Network is ready
    vision_model_path = model_dir / f"trained_{params.model_type}_planner.pth"
    vision_config_path = model_dir / f"trained_{params.model_type}_planner.json"

    vision_values_to_check = [
        "model_type",
        "image_size",
        "num_choices",
        "num_angle_outputs",
    ]
    if skip_cache or not _is_model_valid(
        params, vision_model_path, vision_config_path, vision_values_to_check
    ):
        _log.warning("Vision model missing or invalid, starting training...", model_type=params.model_type)
        train_vision_network(params, project_root, model_dir)

    # 2. Ensure Trajectory Generator is ready (if learning-based)
    traj_gen_type = params.trajectory_generator_type
    traj_model_path = None
    if traj_gen_type in ["ann", "gle"]:
        traj_model_path = model_dir / f"trained_{traj_gen_type}_trajectory_generator.pth"
        traj_values_to_check = [
            "time_prep",
            "time_move",
            "time_locked_with_feedback",
            "time_grasp",
            "time_post",
            "resolution",
        ]
        if skip_cache or not _is_model_valid(
            params, traj_model_path, vision_config_path, traj_values_to_check
        ):
            _log.warning("Trajectory generator model missing, starting training...", type=traj_gen_type)
            train_trajectory_generator(params, generator_type=traj_gen_type, project_root=project_root, model_dir=model_dir)

    # 3. Assemble and return the planner
    return _assemble_planner(params, vision_model_path, traj_model_path)


def _is_model_valid(
    params: PlannerParams,
    model_path: Path,
    config_path: Path,
    values_to_check: list[str],
) -> bool:
    """Check if the existing vision model matches requested parameters."""
    if not (model_path.exists() and config_path.exists()):
        return False

    try:
        with open(config_path) as f:
            saved = json.load(f)

        requested = asdict(params)
        # Check critical parameters that would require a different architecture
        for k in values_to_check:
            val = requested.get(k)
            # Normalize tuples (from dataclass) to lists (from JSON)
            if isinstance(val, tuple):
                val = list(val)
            if val != saved.get(k):
                _log.debug(f"Parameter mismatch for {k}: {val} vs {saved.get(k)}")
                return False
        return True
    except Exception as e:
        _log.error(f"Error checking model validity: {e}")
        return False


def _assemble_planner(params: PlannerParams, vision_model_path: Path, traj_model_path: Path = None):
    """Assembles the planner with its components."""
    # Initialize trajectory generator
    traj_gen = get_trajectory_generator(
        params.trajectory_generator_type,
        params,
        traj_model_path
    )

    # Initialize planner with the vision architecture
    if params.model_type == "gle":
        planner = GLEPlanner(params=params, trajectory_generator=traj_gen)
    else:
        planner = ANNPlanner(params=params, trajectory_generator=traj_gen)

    planner.load_model(vision_model_path)
    return planner
