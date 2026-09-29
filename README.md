# Dynamic Obstacle Avoidance with EBM and MPC

This project combines one-shot energy-based multimodal motion prediction with model predictive control for collision-free indoor robot navigation. It includes warehouse and hospital simulation scenes. The method is described in the [RA-L paper](https://ieeexplore.ieee.org/document/11021381).

```bibtex
@ARTICLE{ze_2025_ebmmpc,
  author={Zhang, Ze and Hess, Georg and Hu, Junjie and Dean, Emmanuel and Svensson, Lennart and Åkesson, Knut},
  journal={IEEE Robotics and Automation Letters},
  title={Future-Oriented Navigation: Dynamic Obstacle Avoidance With One-Shot Energy-Based Multimodal Motion Prediction},
  year={2025}, volume={10}, number={8}, pages={8043-8050},
  doi={10.1109/LRA.2025.3575969}
}
```

![Simulation example](doc/cover.png "Example")

## Setup

The recommended setup uses the committed `uv.lock` with Python 3.11. Install [uv](https://docs.astral.sh/uv/) and the existing [OpEn/PANOC prerequisites](https://alphaville.github.io/optimization-engine/docs/installation), then run from the repository root:

```bash
uv sync --locked
uv run --locked python src/build_solver.py
```

The first command installs the locked Python environment. The second builds the native MPC solver used by the scenes. Solver artifacts are generated locally and are not rebuilt automatically when a simulation starts. Rebuild after changing solver dimensions, prediction horizon, structural constraints, or statically compiled cost terms. Runtime-passed weights and reference states do not by themselves require a rebuild. `src/build_solver.py` uses the active `builder_panoc`; the older PANOC builder variants remain in the source tree.

Conda is an alternative Python environment route. It has not been runtime-verified here; the commands below describe the supported installation path, not a passing environment check. Install the same OpEn/PANOC prerequisites first.

```bash
conda env create -f environment.yml
conda activate dyobav
python -m pip install -r requirements-conda.txt
python -m pip install --no-deps -e .
python src/build_solver.py
```

`requirements-conda.txt` is exported with pins and hashes from `uv.lock`. Regenerate it from the repository root after a deliberate lockfile change with:

```bash
uv export --locked --no-dev --no-emit-project --format requirements.txt --output-file requirements-conda.txt
```

The old `requirements.txt` is a historical environment file and is not the setup path for this checkout.

## Run a simulation

Run these commands from the repository root after building the solver:

```bash
uv run --locked python src/run_simulation.py --scene warehouse
uv run --locked python src/run_simulation.py --scene hospital
```

The command defaults to warehouse. Hospital uses its own scene preset. Both default to ENLL prediction, MPC tracking, one repetition, and automatic device selection. `--device auto` uses CUDA when available and otherwise CPU for EBM inference; `--device cpu` explicitly selects CPU, and `--device cuda` requests CUDA and errors if unavailable. The supplied EBM checkpoints support CPU inference. SGAN remains CUDA-only. This host is CPU-only, so CUDA execution has not been tested here.

Common options include `--seed`, `--evaluation`, `--duration`, `--repeat`, `--output-dir`, `--save-video`, `--visualization {default,on,off}`, and `--view {default,full,map}`. See `--help` for all typed choices and config overrides. Examples:

```bash
# Reproducible evaluated run. The output directory must not already exist.
uv run --locked python src/run_simulation.py --scene warehouse --evaluation --seed 0 --output-dir runs/warehouse-eval-001

# Short, headless hospital run.
uv run --locked python src/run_simulation.py --scene hospital --duration 10 --visualization off

# Warehouse map-only view, or explicitly request a full hospital view.
uv run --locked python src/run_simulation.py --scene warehouse --view map
uv run --locked python src/run_simulation.py --scene hospital --view full

# Save one video; video recording requires visualization and --repeat 1.
mkdir -p runs
uv run --locked python src/run_simulation.py --scene warehouse --save-video runs/warehouse.avi
```

Relative configuration overrides, output directories, and video paths are resolved from the current checkout root. A requested `--output-dir` is created as a new directory and must not exist already. A video target must also be new, and its parent directory must already exist, even when `--output-dir` is requested; create the parent first with `mkdir -p`. Without `--output-dir`, no run-record files are added. With it, the run writes `metadata.json` and compressed `trajectories.npz`, including all repetitions, physical agent state histories, controller action histories as recorded, and selected configuration/artifact metadata. Controller action counts may not align with physical state counts when `action_steps > 1`.

Existing entry filenames remain available: `python src/main_base.py` launches the warehouse preset and `python src/main_base_zos.py` launches the hospital preset. The typed solver builder also supports options; see `uv run --locked python src/build_solver.py --help`.

## Code map and checks

`src/run_simulation.py` and `src/experiment/` provide the thin typed experiment interface. The existing `src/main_base.py` and `src/main_base_zos.py` scene classes and numerical prediction/control packages remain in place. `src/build_solver.py` wraps the existing native builder. `tests/test_experiment.py` contains the three lightweight interface tests; `scripts/smoke.py` exercises both real scenes. Files under `test/` are interactive demos, not an automated discovery suite.

```bash
PYTHONPATH=src uv run --locked python -m unittest discover -s tests -p 'test_experiment.py' -v
PYTHONPATH=src uv run --locked python scripts/smoke.py
```

The ROS 2 Gazebo implementation is maintained on the [ROS 2 branch](https://github.com/Woodenonez/DyObAv-MPCnEBM-Warehouse/tree/ros2-version) and in the [ROS 2 repository](https://github.com/Woodenonez/DyObAv-MPCnEBM-Warehouse-ROS2). Demo videos: [ROS 2/Gazebo](https://youtu.be/j4n2mt0KdMY) and [Python simulation](https://youtu.be/nNLAS4Hfgtk).
