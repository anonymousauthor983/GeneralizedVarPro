"""Run the fixed MRI reproduction schedule, with exclusive output ownership."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
METHODS = ['projected', 'joint_prox', 'alternating']


def schedule(args):
    for seed in args.seeds:
        for mu in [1, 0]:
            for acceleration in args.accelerations:
                for power in args.powers:
                    if mu == 0 and power != 1:
                        continue
                    family = [4, 8].index(acceleration) * 3 + [1, 2, .5].index(power)
                    for method in args.methods:
                        if mu == 0 and method == 'projected':
                            continue
                        for index in args.lambda_indices:
                            yield seed, mu, acceleration, power, method, index, family


def validate_result(path, seed, mu, acceleration, power, method, lam):
    result = json.loads(path.read_text())
    expected = dict(seed=seed, mu=mu, R=acceleration, power=power,
                    method=method, lambda_train=lam, epochs=8, updates=3840)
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Result configuration mismatch: {path}')
    evaluations = result['all_inference_results']
    if len(evaluations) != 1 or evaluations[0]['lambda_test'] != lam:
        raise ValueError(f'Inference lambda mismatch: {path}')
    evaluation = evaluations[0]
    if len(evaluation['per_image']) != 480 or len(evaluation['solver_states']) != 480:
        raise ValueError(f'Incomplete validation: {path}')
    if not all(math.isfinite(value) for row in evaluation['per_image'] for value in row.values()):
        raise ValueError(f'Nonfinite validation metric: {path}')
    keys = ['active_stationarity', 'coordinate_residual'] if power == .5 else ['kkt']
    for state in evaluation['solver_states']:
        if any(not math.isfinite(state[key]) or state[key] > 1e-8 for key in keys):
            raise ValueError(f'Invalid solver residual: {path}')
        if power == .5 and state['active_eigen_ratio'] < 0:
            raise ValueError(f'Invalid active curvature: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='runs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--powers', type=float, nargs='+', choices=[1, 2, .5], default=[1, 2, .5])
    parser.add_argument('--accelerations', type=int, nargs='+', choices=[4, 8], default=[4, 8])
    parser.add_argument('--lambda-indices', type=int, nargs='+', choices=[0, 1, 2], default=[0, 1, 2])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error('Seeds must be nonnegative.')
    for key in ['seeds', 'powers', 'accelerations', 'lambda_indices', 'methods']:
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            parser.error(f'Duplicate values in {key}.')
    if args.dry_run:
        for seed, mu, acceleration, power, method, index, family in schedule(args):
            print(json.dumps(dict(seed=seed, mu=mu, R=acceleration, power=power,
                                  method=method, lambda_index=index)))
        return
    data = Path(args.data).resolve()
    for name in ['manifest.json', 'train.npy', 'val.npy']:
        if not (data / name).is_file():
            parser.error(f'Missing prepared data file: {data / name}')
    import torch
    if not torch.cuda.is_available():
        parser.error('Training requires a CUDA GPU.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.launch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another launcher is using this output directory.')
        for seed, mu, acceleration, power, method, index, family in schedule(args):
            work = output / f'seed{seed}' / f'mu{mu}'
            work.mkdir(parents=True, exist_ok=True)
            config = json.loads((ROOT / 'base_config.json').read_text())
            config.update(data=str(data), seed=seed, mu=float(mu))
            config_path = work / 'config.json'
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError(f'Existing configuration differs: {config_path}')
            config_path.write_text(json.dumps(config, indent=2))
            calibration = ROOT / 'calibration' / f'CALIBRATION_{family}.json'
            values = json.loads(calibration.read_text())
            if not values['passed']:
                raise ValueError(f'Invalid calibration: {calibration}')
            lam = values['lambdas'][index]
            target = work / calibration.name
            if target.exists() and json.loads(target.read_text()) != values:
                raise ValueError(f'Existing calibration differs: {target}')
            shutil.copy2(calibration, target)
            result = work / 'results' / f'r{acceleration}_p{power:g}_{method}_l{index}' / 'result.json'
            if result.exists():
                validate_result(result, seed, mu, acceleration, power, method, lam)
                continue
            task = family * 9 + METHODS.index(method) * 3 + index
            (work / 'logs').mkdir(exist_ok=True)
            with (work / 'logs' / f'task{task}.out').open('a') as log:
                subprocess.run([sys.executable, str(ROOT / 'run.py'), '--task', str(task), '--resume'],
                               cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
            validate_result(result, seed, mu, acceleration, power, method, lam)


if __name__ == '__main__':
    main()
