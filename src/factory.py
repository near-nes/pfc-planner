"""Factory function to get or train planners."""

import json
from dataclasses import asdict
from pathlib import Path

import structlog

from .config import PlannerParams
from .planners import ANNPlanner, ANNPlannerNet, GLEPlanner, GLEPlannerNet
from .train import get_git_commit_hash, get_project_root, run_training

_log: structlog.stdlib.BoundLogger = structlog.get_logger("pfc_planner.factory")


def get_planner(params: PlannerParams, model_dir: Path = None):
    """
    Get a planner, training it if necessary.

    Args:
        params: PlannerParams with required configuration (including model_type)
        model_dir: Override default model directory (default: submodule's models/)
    """
    project_root = get_project_root()
    if model_dir is None:
        model_dir = project_root / "models"

    current_git_commit = get_git_commit_hash(project_root)

    model_type = params.model_type
    model_path = model_dir / f"trained_{model_type}_planner.pth"
    config_path = model_dir / f"trained_{model_type}_planner.json"

    if model_path.exists() and config_path.exists():
        with open(config_path) as f:
            saved = json.load(f)

        critical_match = (
            saved.get("model_type") == params.model_type
            and saved.get("git_commit") == current_git_commit
            and saved.get("resolution") == params.resolution
            and saved.get("time_prep") == params.time_prep
            and saved.get("time_move") == params.time_move
            and saved.get("time_locked_with_feedback")
            == params.time_locked_with_feedback
            and saved.get("time_grasp") == params.time_grasp
            and saved.get("time_post") == params.time_post
            and saved.get("num_choices") == params.num_choices
            and tuple(saved.get("image_size", [])) == params.image_size
        )

        # Log all differences
        requested = asdict(params)
        diffs = {
            k: (v, saved.get(k)) for k, v in requested.items() if v != saved.get(k)
        }
        if diffs:
            diff_strs = [f"{k}: {v[0]} vs {v[1]}" for k, v in diffs.items()]
            if critical_match:
                _log.info(
                    "Parameter differences (requested vs saved), using saved model",
                    differences=diff_strs,
                )
            else:
                _log.warning(
                    "Critical parameter mismatch, retraining required",
                    differences=diff_strs,
                )

        if critical_match:
            return _load_planner(params, model_path)

    _log.warning(
        "No matching model found, starting training",
        model_type=model_type,
        model_path=str(model_path),
    )
    params.git_commit = current_git_commit
    run_training(params)
    return _load_planner(params, model_path)


def _load_planner(params: PlannerParams, model_path: Path):
    if params.model_type == "gle":
        net = GLEPlannerNet(params=params)
        planner = GLEPlanner(params=params, net=net)
    else:
        net = ANNPlannerNet(params=params)
        planner = ANNPlanner(params=params, net=net)

    planner.load_model(model_path)
    return planner
