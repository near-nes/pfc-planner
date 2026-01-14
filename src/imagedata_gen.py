import datetime
import os
import sys
import time
from pathlib import Path

import pybullet

sys.path.insert(0, "/sim/controller/complete_control")
os.environ["RUNS_PATH"] = str((Path(__file__).parent / "runs").absolute())

from complete_control.config.core_models import (
    OracleData,
    SimulationParams,
    TargetColor,
)
from complete_control.config.paths import RunPaths
from complete_control.config.plant_config import PlantConfig
from complete_control.plant.robotic_plant import RoboticPlant

if __name__ == "__main__":
    joint_angle = 90
    for target in range(0, 151, 10):
        for color, color_label in zip(
            (TargetColor.BLUE_LEFT, TargetColor.RED_RIGHT),
            ("blue", "red"),
        ):
            image_path = f"./data/start_{target}_{color_label}.bmp"
            run_paths = RunPaths.from_run_id(
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            config = PlantConfig.from_runpaths(
                run_paths,
                simulation=SimulationParams(
                    oracle=OracleData(
                        init_joint_angle=joint_angle,
                        tgt_joint_angle=target,
                        target_color=color,
                    )
                ),
            )
            plant = RoboticPlant(config, pybullet)
            plant._capture_state_and_save(image_path)
            plant.p.resetSimulation()
            plant.p.disconnect()
