import os
from pathlib import Path
import subprocess
import sys
import unittest

from experiment.config import ExperimentArgs, resolve_experiment


ROOT = Path(__file__).resolve().parents[1]


class ExperimentInterfaceTests(unittest.TestCase):
    def test_scene_defaults(self) -> None:
        for scene, scenario, duration, suffix, dataset, map_only, pedestrian in (
            ("warehouse", 1, 30.0, "0", "warehouse_sim_original", False, "none"),
            ("hospital", 2, 200.0, "1", "zospital_sim_original", True, "minisf"),
        ):
            with self.subTest(scene=scene):
                config = resolve_experiment(ExperimentArgs(scene=scene))
                self.assertEqual((config.scenario, config.duration, config.model_suffix), (scenario, duration, suffix))
                self.assertEqual((config.predictor, config.tracker, config.planner), ("enll", "mpc", "none"))
                self.assertEqual((config.pedestrian, config.visualization, config.map_only), (pedestrian, True, map_only))
                self.assertEqual(config.project_root, ROOT)
                self.assertEqual(config.tracker_config, ROOT / "config/mpc_fast.yaml")
                self.assertEqual(config.graph_path, ROOT / "data" / dataset / "graph.json")
                self.assertEqual(config.device, "auto")

    def test_explicit_overrides(self) -> None:
        for scene in ("warehouse", "hospital"):
            with self.subTest(scene=scene):
                config = resolve_experiment(ExperimentArgs(
                    scene=scene, scenario=1, predictor="cvm", tracker="rpp", planner="teb",
                    pedestrian="sf", duration=1.0, repeat=2, evaluation=True,
                    visualization="off", view="full", seed=7, device="cpu",
                    tracker_config=Path("custom/tracker.yaml"),
                    predictor_config=Path("custom/predictor.yaml"),
                    model_suffix="42", output_dir=Path("runs/example"),
                ))
                self.assertEqual((config.scenario, config.duration, config.repeat, config.seed), (1, 1.0, 2, 7))
                self.assertEqual((config.pedestrian, config.visualization, config.map_only, config.device), ("sf", False, False, "cpu"))
                self.assertEqual(config.tracker_config, ROOT / "custom/tracker.yaml")
                self.assertEqual(config.predictor_config, ROOT / "custom/predictor.yaml")
                self.assertEqual(config.output_dir, ROOT / "runs/example")
                self.assertEqual(config.model_suffix, "42")
                self.assertIsNone(resolve_experiment(ExperimentArgs(scene=scene)).seed)

        rejected = (
            ExperimentArgs(repeat=0), ExperimentArgs(duration=0),
            ExperimentArgs(duration=float("inf")), ExperimentArgs(scenario=999),
            ExperimentArgs(scene="hospital", predictor="nll"),
            ExperimentArgs(save_video=Path("video.mp4"), visualization="off"),
            ExperimentArgs(save_video=Path("video.mp4"), repeat=2),
        )
        for args in rejected:
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_experiment(args)
        self.assertEqual(
            resolve_experiment(ExperimentArgs(scene="hospital", predictor="nll", predictor_config=Path("custom/nll.yaml"))).predictor,
            "nll",
        )

    def test_help_is_lightweight(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        for filename in ("run_simulation.py", "main_base.py", "main_base_zos.py", "build_solver.py"):
            with self.subTest(filename=filename):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "src" / filename), "--help"],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--help", result.stdout)
                self.assertNotIn("Preparing:", result.stdout)
                self.assertNotIn("Loading complete!", result.stdout)


if __name__ == "__main__":
    unittest.main()
