"""Factory function to get or train planners."""

import json
from dataclasses import asdict
from pathlib import Path

import structlog

from .config import PlannerParams
from .planners import ANNPlanner, GLEPlanner
from .train import get_git_commit_hash, get_project_root, run_training

_log: structlog.stdlib.BoundLogger = structlog.get_logger("pfc_planner.factory")


def get_planner(
    params: PlannerParams,
    model_dir: Path = None,
    project_root: Path = None,
    skip_cache: bool = False,
):
    """
    Get a planner, training it if necessary.

    A planner consists of:
    1. Vision network: for extracting angles and choice from images
    2. Trajectory generator: for converting angles to full trajectories

    Args:
        params: PlannerParams with required configuration (including model_type)
        model_dir: Override default model directory (default: submodule's models/)
        project_root: Override project root (default: auto-detect via get_project_root())
        skip_cache: If True, force retraining even if matching model exists (default: False)
    """
    if project_root is None:
        project_root = get_project_root()
    if model_dir is None:
        model_dir = project_root / "models"

    params.git_commit = get_git_commit_hash(project_root)

    model_type = params.model_type
    model_path = model_dir / f"trained_{model_type}_planner.pth"
    config_path = model_dir / f"trained_{model_type}_planner.json"

    if not skip_cache and model_path.exists() and config_path.exists():
        with open(config_path) as f:
            saved = json.load(f)

        # Normalize image_size to tuple for comparison (JSON serializes tuples as lists)
        if "image_size" in saved and isinstance(saved["image_size"], list):
            saved["image_size"] = tuple(saved["image_size"])

        requested = asdict(params)
        diffs = {
            k: (v, saved.get(k)) for k, v in requested.items() if v != saved.get(k)
        }
        if "git_commit" in diffs:
            (current, train) = diffs.pop("git_commit")
            _log.warning(
                f"planner git commit changed from training",
                training_hash=train,
                current_hash=current,
            )

        if diffs:
            diff_strs = [f"{k}: {v[0]} vs {v[1]}" for k, v in diffs.items()]
            _log.warning(
                "Critical parameter mismatch, retraining required",
                differences=diff_strs,
            )
        else:
            # No differences, load existing model
            return _load_planner(params, model_path)
    elif skip_cache:
        _log.warning(
            "Cache skipped, forcing retraining",
            model_type=model_type,
        )

    _log.warning(
        "No matching model found, starting training",
        model_type=model_type,
        model_path=str(model_path),
    )
    run_training(params, project_root, model_dir)
    return _load_planner(params, model_path)


def _load_planner(params: PlannerParams, model_path: Path):
    """Load a planner with its vision network and trajectory generator."""
    if params.model_type == "gle":
        planner = GLEPlanner(params=params)
    else:
        planner = ANNPlanner(params=params)

    planner.load_model(model_path)
    return planner
