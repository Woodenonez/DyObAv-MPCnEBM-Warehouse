"""Optional, episode-level records for an experiment run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np

from .config import ResolvedExperiment

if TYPE_CHECKING:
    from main_base import MainBase as WarehouseMainBase
    from main_base_zos import MainBase as HospitalMainBase


@dataclass(frozen=True)
class EpisodeTrajectories:
    repeat_index: int
    robot_states: dict[int, np.ndarray]
    robot_actions: dict[int, np.ndarray]
    human_states: dict[int, np.ndarray]


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    sha256: str


@dataclass(frozen=True)
class GitState:
    commit: str | None
    dirty: bool | None


@dataclass(frozen=True)
class MetadataRecord:
    resolved_args: dict[str, str | int | float | bool | None]
    selected_configs: dict[str, dict[str, str]]
    seed: int | None
    git: GitState
    versions: dict[str, str | None]
    scene: str
    neural_device: dict[str, str | None]
    checkpoint: ArtifactRecord | None
    solver_artifact: ArtifactRecord | None
    trajectory_notes: str


def prepare_run_dir(config: ResolvedExperiment) -> Path | None:
    """Create the exact fresh directory requested for recording, if any."""
    if config.output_dir is None:
        return None
    config.output_dir.mkdir(parents=True, exist_ok=False)
    return config.output_dir


def _artifact(path: Path | None) -> ArtifactRecord | None:
    if path is None:
        return None
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return ArtifactRecord(path=str(path.resolve()), sha256=digest.hexdigest())


def _git_state(root: Path) -> GitState:
    commit: subprocess.CompletedProcess[str] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False,
    )
    status: subprocess.CompletedProcess[str] = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False,
    )
    return GitState(
        commit=commit.stdout.strip() if commit.returncode == 0 else None,
        dirty=bool(status.stdout) if status.returncode == 0 else None,
    )


def write_metadata(
    config: ResolvedExperiment,
    simulation: WarehouseMainBase | HospitalMainBase,
    run_dir: Path,
) -> None:
    """Write the resolved launch and artifacts actually loaded by this scene."""
    predictor = simulation.motion_predictor
    checkpoint_path: Path | None = None
    actual_device: str | None = None
    if config.predictor in ("enll", "nll", "bce", "kld"):
        checkpoint_path = Path(predictor.network_loader.save_path)
        actual_device = predictor.actual_device
    elif config.predictor == "sgan":
        # SGAN loads the selected checkpoint directly during construction.
        checkpoint_path = config.predictor_config
        actual_device = str(predictor.device)

    solver_path: Path | None = None
    if config.tracker == "mpc":
        controller = simulation.robot_manager.get_controller(simulation.robot_ids[0])
        built_solver = getattr(controller, "built_solver", None)
        if built_solver is not None:
            solver_path = Path(built_solver.__file__)

    selected_configs: dict[str, dict[str, str]] = {}
    for name in ("tracker_config", "planner_config", "predictor_config", "robot_config", "human_config", "transform_config"):
        path: Path | None = getattr(config, name)
        if path is not None and path.suffix.lower() in (".yaml", ".yml"):
            selected_configs[name] = {"path": str(path), "text": path.read_text(encoding="utf-8")}

    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("numpy", "torch", "scipy", "casadi", "opengen"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None

    resolved_args: dict[str, str | int | float | bool | None] = {
        key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
    }
    metadata: MetadataRecord = MetadataRecord(
        resolved_args=resolved_args,
        selected_configs=selected_configs,
        seed=config.seed,
        git=_git_state(config.project_root),
        versions=versions,
        scene=config.scene,
        neural_device={"requested": config.device if actual_device is not None else None, "actual": actual_device},
        checkpoint=_artifact(checkpoint_path),
        solver_artifact=_artifact(solver_path),
        trajectory_notes="Controller actions are stored as recorded; with action_steps > 1 their count need not align with physical states.",
    )
    (run_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")


def capture_trajectories(
    simulation: WarehouseMainBase | HospitalMainBase, repeat_index: int,
) -> EpisodeTrajectories:
    """Copy physical states and controller actions before the scene resets."""
    return EpisodeTrajectories(
        repeat_index=repeat_index,
        robot_states={rid: np.asarray(simulation.robot_manager.get_robot(rid).past_traj).copy() for rid in simulation.robot_ids},
        robot_actions={rid: np.asarray(simulation.robot_manager.get_controller(rid).past_actions).copy() for rid in simulation.robot_ids},
        human_states={index: np.asarray(human.past_traj).copy() for index, human in enumerate(simulation.humans)},
    )


def write_trajectories(episodes: list[EpisodeTrajectories], run_dir: Path) -> None:
    """Store each available history under its episode and object key."""
    arrays: dict[str, np.ndarray] = {}
    for episode in episodes:
        prefix = f"repeat_{episode.repeat_index:03d}"
        for rid, states in episode.robot_states.items():
            arrays[f"{prefix}_robot_{rid}_states"] = states
        for rid, actions in episode.robot_actions.items():
            arrays[f"{prefix}_robot_{rid}_actions"] = actions
        for index, states in episode.human_states.items():
            arrays[f"{prefix}_human_{index}_states"] = states
    np.savez_compressed(run_dir / "trajectories.npz", **arrays)
