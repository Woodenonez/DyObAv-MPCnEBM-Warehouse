"""Typed entry point for building the existing PANOC solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs import CircularRobotSpecification, MpcConfiguration


ROOT: Path = Path(__file__).resolve().parents[1]


@dataclass
class BuildArgs:
    mpc_config: Path = Path("config/mpc_fast.yaml")
    robot_config: Path = Path("config/spec_robot.yaml")
    build_directory: Path | None = None
    use_tcp: bool = False


def _root_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def return_cfg_path(fname: str) -> str:
    return str(ROOT / "config" / fname)


def load_mpc_config(fname: str) -> MpcConfiguration:
    """Load the MPC configuration."""
    from configs import MpcConfiguration

    return MpcConfiguration.from_yaml(return_cfg_path(fname))

def load_robot_spec(fname: str) -> CircularRobotSpecification:
    """Load the robot specification."""
    from configs import CircularRobotSpecification

    return CircularRobotSpecification.from_yaml(return_cfg_path(fname))

def build_solver(args: BuildArgs) -> None:
    from basic_motion_model import motion_model
    from configs import CircularRobotSpecification, MpcConfiguration
    from pkg_tracker_mpc.casadi_build import builder_panoc

    config_mpc: MpcConfiguration = MpcConfiguration.from_yaml(str(_root_path(args.mpc_config)))
    config_robot: CircularRobotSpecification = CircularRobotSpecification.from_yaml(str(_root_path(args.robot_config)))
    build_directory: Path = _root_path(args.build_directory) if args.build_directory is not None else _root_path(Path(config_mpc.build_directory))
    config_mpc.build_directory = str(build_directory)
    mpc_module: builder_panoc.PanocBuilder = builder_panoc.PanocBuilder(config_mpc, config_robot)
    mpc_module.load_motion_model(motion_model.unicycle_model)
    mpc_module.build(test=False, use_tcp=args.use_tcp)


def main() -> None:
    import tyro

    args: BuildArgs = tyro.cli(BuildArgs)
    build_solver(args)


if __name__ == "__main__":
    main()
