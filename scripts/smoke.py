"""Run short, headless ENLL/MPC checks on the two default scenes."""

import hashlib
import json
from pathlib import Path
import random
import sys
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from main_base import MainBase as WarehouseBase
from main_base_zos import MainBase as HospitalBase


def _identity(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def run_smoke(output_dir: Path, seed: int = 0, steps: int = 10,
              solver_directory: Path | None = None) -> None:
    """Save real prediction and MPC traces for warehouse 1 and hospital 2."""
    if steps < 1:
        raise ValueError('steps must be positive')
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    solver_directory = (solver_directory or ROOT / 'mpc_solver').resolve()
    solver_path = solver_directory / 'mpc_fast' / 'mpc_fast.so'
    if not solver_path.is_file():
        raise FileNotFoundError(f'MPC native solver missing: {solver_path}')

    results: dict[str, dict] = {}
    scenes = (
        ('warehouse1', WarehouseBase, 1, 'warehouse_sim_original',
         'global_setting_warehouse.yaml', 'wsd_1t20_poselu_enll_train.yaml',
         'wsd_poselu_enll_0', '0', None),
        ('hospital2', HospitalBase, 2, 'zospital_sim_original',
         'global_setting_zospital.yaml', 'zpd_1t20_poselu_enll_train.yaml',
         'zospital_poselu_enll_1', '1', 'zospital'),
    )
    for name, base_class, scenario, data_name, tf_name, ebm_name, checkpoint_name, suffix, extra_key in scenes:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        data = ROOT / 'data' / data_name
        config = ROOT / 'config'
        if name == 'warehouse1':
            base = base_class(scenario_index=scenario, evaluation=True, visualization=False, verbose=False)
        else:
            base = base_class(scenario_index=scenario, evaluation=True, map_only=True, verbose=False)
        base.load(None, 'none', 'mpc', str(config / 'mpc_fast.yaml'),
                  str(config / 'spec_robot.yaml'), str(config / 'spec_human.yaml'),
                  str(config / tf_name))
        base.config_tracker.build_directory = str(solver_directory)
        base.prepare(str(data / 'graph.json'), str(data / 'map.json'), 'enll',
                     str(config / ebm_name), str(data / 'background.png'), suffix,
                     pedestrian_model='minisf' if name == 'hospital2' else 'None')
        base._prepare_evaluation(repeat=1)

        # This is the physical initial history used by run_motion_prediction.
        history = [base.tf_img2real(point, False) for point in base.humans[0].past_traj]
        predictor = base.motion_predictor
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            _, _, probability = predictor.get_network_output(
                history, rescale=base.config_tf.scale2nn, extra_key=extra_key)
            np.save(output_dir / f'{name}-first-probability.npy', probability.numpy())
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        trace: list[dict] = []
        for tick in range(steps):
            before = {rid: base.robot_manager.get_robot(rid).state.tolist() for rid in base.robot_ids}
            base.run_one_step(current_time=tick * base.config_tracker.ts, num_repeat=0)
            robots = {}
            for rid in base.robot_ids:
                controller = base.robot_manager.get_controller(rid)
                robots[rid] = {
                    'before': before[rid],
                    'after': base.robot_manager.get_robot(rid).state.tolist(),
                    'action': controller.past_actions[-1].tolist() if controller.past_actions else None,
                }
            trace.append({'tick': tick, 'time': tick * base.config_tracker.ts,
                          'robots': robots, 'humans': [human.state.tolist() for human in base.humans]})
            if any(base.evaluator.eval_results[rid]['collision_results'][0] for rid in base.robot_ids) or \
                    all(base.evaluator.eval_results[rid]['complete_results'][0] for rid in base.robot_ids):
                break

        flags: dict[int, dict[str, bool]] = {rid: {
            'collision': base.evaluator.eval_results[rid]['collision_results'][0],
            'complete': base.evaluator.eval_results[rid]['complete_results'][0],
        } for rid in base.robot_ids}
        result: dict = {
            'scene': name, 'scenario': scenario, 'seed': seed, 'requested_steps': steps,
            'ticks_completed': len(trace), 'tracker': 'mpc_fast', 'predictor': 'enll',
            'pedestrian_model': base.pedestrian_model,
            'settings': {'time_step': base.config_tracker.ts,
                         'horizon': base.config_tracker.N_hor,
                         'max_solver_time_us': base.config_tracker.max_solver_time,
                         'solver_type': base.config_tracker.solver_type,
                         'predictor_config': str(config / ebm_name)},
            'requested_device': predictor.requested_device, 'actual_device': predictor.actual_device,
            'checkpoint': _identity(ROOT / 'model' / checkpoint_name),
            'native_solver': _identity(solver_path), 'termination': flags, 'trace': trace,
        }
        (output_dir / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
        results[name] = {'ticks_completed': len(trace), 'termination': flags}

    if any(item['ticks_completed'] < 1 for item in results.values()):
        raise RuntimeError('Each scene must complete at least one real tick')
    (output_dir / 'success.json').write_text(json.dumps(results, indent=2) + '\n')


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output_dir = ROOT / 'runs' / f'{stamp}-{uuid4().hex[:8]}'
    run_smoke(output_dir)
    print(f'Smoke traces: {output_dir}')


if __name__ == '__main__':
    main()
