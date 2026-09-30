"""Typed command-line entry point for the existing scene simulations."""

from collections.abc import Sequence
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment.config import ExperimentArgs, Scene, resolve_experiment


def parse_args(argv: Sequence[str] | None = None, *, default_scene: Scene = "warehouse") -> ExperimentArgs:
    import tyro

    return tyro.cli(ExperimentArgs, args=argv, default=ExperimentArgs(scene=default_scene))


def main(argv: Sequence[str] | None = None, *, default_scene: Scene = "warehouse") -> None:
    args: ExperimentArgs = parse_args(argv, default_scene=default_scene)
    from experiment.runner import run_experiment

    run_experiment(resolve_experiment(args))


if __name__ == "__main__":
    main()
