"""Pure scene presets and launch validation."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal


Scene = Literal["warehouse", "hospital"]
Predictor = Literal["enll", "nll", "bce", "kld", "cvm", "sgan", "none"]
Tracker = Literal["mpc", "rpp"]
Planner = Literal["none", "teb"]
Pedestrian = Literal["none", "sf", "minisf"]
Device = Literal["auto", "cpu", "cuda"]


@dataclass
class ExperimentArgs:
    scene: Scene = "warehouse"
    scenario: int | None = None
    predictor: Predictor = "enll"
    tracker: Tracker = "mpc"
    planner: Planner = "none"
    pedestrian: Literal["default", "none", "sf", "minisf"] = "default"
    duration: float | None = None
    repeat: int = 1
    evaluation: bool = False
    visualization: Literal["default", "on", "off"] = "default"
    view: Literal["default", "full", "map"] = "default"
    auto_run: bool = True
    verbose: bool = False
    seed: int | None = None
    device: Device = "auto"
    tracker_config: Path | None = None
    predictor_config: Path | None = None
    model_suffix: str | None = None
    output_dir: Path | None = None
    save_video: Path | None = None


@dataclass(frozen=True)
class ResolvedExperiment:
    scene: Scene
    scenario: int
    predictor: Predictor
    tracker: Tracker
    planner: Planner
    pedestrian: Pedestrian
    duration: float
    repeat: int
    evaluation: bool
    visualization: bool
    map_only: bool
    auto_run: bool
    verbose: bool
    seed: int | None
    device: Device
    project_root: Path
    tracker_config: Path
    planner_config: Path | None
    predictor_config: Path | None
    robot_config: Path
    human_config: Path
    transform_config: Path
    graph_path: Path
    map_path: Path
    ref_image_path: Path
    model_suffix: str
    output_dir: Path | None
    save_video: Path | None


def _root_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def resolve_experiment(args: ExperimentArgs) -> ResolvedExperiment:
    root: Path = Path(__file__).resolve().parents[2]
    scene: Scene = args.scene
    prefix: str = "wsd" if scene == "warehouse" else "zpd"
    dataset: str = "warehouse_sim_original" if scene == "warehouse" else "zospital_sim_original"
    predictor_config: Path | None = None
    if args.predictor == "enll":
        predictor_config = root / "config" / f"{prefix}_1t20_poselu_enll_train.yaml"
    elif args.predictor in ("nll", "bce", "kld") and not (scene == "hospital" and args.predictor == "nll"):
        predictor_config = root / "config" / f"{prefix}_1t20_none_{args.predictor}_train.yaml"
    elif args.predictor == "sgan":
        predictor_config = root / "model/sgan_eth12"
    if args.predictor_config is not None:
        predictor_config = _root_path(root, args.predictor_config)

    config: ResolvedExperiment = ResolvedExperiment(
        scene=scene,
        scenario=args.scenario if args.scenario is not None else (1 if scene == "warehouse" else 2),
        predictor=args.predictor,
        tracker=args.tracker,
        planner=args.planner,
        pedestrian=("none" if scene == "warehouse" else "minisf") if args.pedestrian == "default" else args.pedestrian,
        duration=args.duration if args.duration is not None else (30.0 if scene == "warehouse" else 200.0),
        repeat=args.repeat,
        evaluation=args.evaluation,
        visualization=(not args.evaluation) if args.visualization == "default" else args.visualization == "on",
        map_only=(scene == "hospital") if args.view == "default" else args.view == "map",
        auto_run=args.auto_run,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
        project_root=root,
        tracker_config=_root_path(root, args.tracker_config) if args.tracker_config is not None else root / "config" / ("mpc_fast.yaml" if args.tracker == "mpc" else "rpp.yaml"),
        planner_config=root / "config/teb.yaml" if args.planner == "teb" else None,
        predictor_config=predictor_config,
        robot_config=root / "config/spec_robot.yaml",
        human_config=root / "config/spec_human.yaml",
        transform_config=root / "config" / ("global_setting_warehouse.yaml" if scene == "warehouse" else "global_setting_zospital.yaml"),
        graph_path=root / "data" / dataset / "graph.json",
        map_path=root / "data" / dataset / "map.json",
        ref_image_path=root / "data" / dataset / "background.png",
        model_suffix=args.model_suffix if args.model_suffix is not None else ("0" if scene == "warehouse" else "1"),
        output_dir=_root_path(root, args.output_dir) if args.output_dir is not None else None,
        save_video=_root_path(root, args.save_video) if args.save_video is not None else None,
    )
    validate_launch(config)
    return config


def validate_launch(config: ResolvedExperiment) -> None:
    """Reject incompatible launch selections before loading numerical modules."""
    valid_scenarios: tuple[int, ...] = (1, 2, 3, 4, 10, 11, 12, 13, 20) if config.scene == "warehouse" else (1, 2)
    if config.scenario not in valid_scenarios:
        raise ValueError(f"Invalid {config.scene} scenario: {config.scenario}")
    if config.repeat < 1:
        raise ValueError("repeat must be at least 1")
    if not math.isfinite(config.duration) or config.duration <= 0:
        raise ValueError("duration must be finite and positive")
    if config.scene == "hospital" and config.predictor == "nll" and config.predictor_config is None:
        raise ValueError("Hospital NLL has no preset; supply --predictor-config")
    if config.save_video is not None:
        if not config.visualization:
            raise ValueError("video requires visualization")
        if config.repeat != 1:
            raise ValueError("video requires repeat=1")
