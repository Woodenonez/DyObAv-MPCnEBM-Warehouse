"""Adapter from a resolved launch to the existing scene loops.

The scenes own the pixel/world transform, 0.2-second control timestep, 20-step
prediction horizon, current-plus-future obstacle slots, and robot diameter /
effective human radius conventions. This adapter does not alter those inputs.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import TYPE_CHECKING

from .config import ResolvedExperiment, validate_launch
from .records import EpisodeTrajectories, capture_trajectories, prepare_run_dir, write_metadata, write_trajectories

if TYPE_CHECKING:
    from main_base import MainBase as WarehouseMainBase
    from main_base_zos import MainBase as HospitalMainBase


def create_simulation(config: ResolvedExperiment) -> WarehouseMainBase | HospitalMainBase:
    """Construct, load, and prepare the selected original scene."""
    validate_launch(config)
    if config.output_dir is not None and config.output_dir.exists():
        raise FileExistsError(config.output_dir)

    if config.seed is not None:
        import numpy as np
        import torch

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    if config.predictor == "sgan" or (config.device == "cuda" and config.predictor in ("enll", "nll", "bce", "kld")):
        import torch

        if config.predictor == "sgan" and (config.device == "cpu" or not torch.cuda.is_available()):
            raise ValueError("SGAN requires CUDA; select another predictor on this machine")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested for learned inference but is unavailable")

    if config.save_video is not None:
        if config.save_video.exists():
            raise FileExistsError(config.save_video)
        if not config.save_video.parent.is_dir():
            raise FileNotFoundError(config.save_video.parent)

    if config.scene == "warehouse":
        from main_base import MainBase

        simulation: WarehouseMainBase | HospitalMainBase = MainBase(
            scenario_index=config.scenario, evaluation=config.evaluation,
            visualization=config.visualization, verbose=config.verbose,
            map_only=config.map_only,
            save_video_name=str(config.save_video) if config.save_video is not None else None,
        )
    else:
        from main_base_zos import MainBase

        simulation = MainBase(
            scenario_index=config.scenario, evaluation=config.evaluation,
            map_only=config.map_only,
            save_video_name=str(config.save_video) if config.save_video is not None else None,
            verbose=config.verbose, visualization=config.visualization,
        )

    simulation.load(
        None if config.planner == "none" else config.planner,
        str(config.planner_config) if config.planner_config is not None else "none",
        config.tracker, str(config.tracker_config), str(config.robot_config),
        str(config.human_config), str(config.transform_config),
    )
    if config.tracker == "mpc":
        build_directory: Path = Path(simulation.config_tracker.build_directory)
        simulation.config_tracker.build_directory = str(
            build_directory if build_directory.is_absolute() else config.project_root / build_directory
        )
    simulation.prepare(
        str(config.graph_path), str(config.map_path),
        None if config.predictor == "none" else config.predictor,
        str(config.predictor_config) if config.predictor_config is not None else "none",
        ref_image_path=str(config.ref_image_path), model_suffix=config.model_suffix,
        pedestrian_model="None" if config.scene == "warehouse" and config.pedestrian == "none" else config.pedestrian,
        device=config.device,
    )
    prepare_run_dir(config)
    return simulation


def run_experiment(config: ResolvedExperiment) -> None:
    """Run the original scene loop and optional evaluation report."""
    simulation: WarehouseMainBase | HospitalMainBase = create_simulation(config)
    run_dir: Path | None = config.output_dir
    episodes: list[EpisodeTrajectories] = []
    if run_dir is not None:
        write_metadata(config, simulation, run_dir)

    def capture_episode(scene: WarehouseMainBase | HospitalMainBase, repeat_index: int) -> None:
        episodes.append(capture_trajectories(scene, repeat_index))

    simulation.run_once(
        repeat=config.repeat, time_step_out=config.duration, extra_debug_panel=None,
        auto_run=config.auto_run, on_episode_end=capture_episode if run_dir is not None else None,
    )
    if run_dir is not None:
        write_trajectories(episodes, run_dir)
    if config.evaluation:
        simulation.report(save_dir=str(run_dir) if run_dir is not None else "./")
    planner: str | None = None if config.planner == "none" else config.planner
    predictor: str | None = None if config.predictor == "none" else config.predictor
    pedestrian: str = "None" if config.scene == "warehouse" and config.pedestrian == "none" else config.pedestrian
    print(f"Finish with predictor: {predictor}, tracker: {config.tracker}, planner: {planner}, scenario: {config.scenario}, human model: {pedestrian}.")
