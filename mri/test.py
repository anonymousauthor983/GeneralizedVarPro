"""Run isolated numerical and packaging checks without training."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text())
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    train = manifest['splits']['train']['examples']
    val = manifest['splits']['val']['examples']
    assert len(train) == 1920 and len(val) == 480
    assert not {r['volume'] for r in train} & {r['volume'] for r in val}
    for family in range(6):
        calibration = json.loads((ROOT / 'calibration' / f'CALIBRATION_{family}.json').read_text())
        assert calibration['passed'] and len(calibration['lambdas']) == 3
    with tempfile.TemporaryDirectory(prefix='mri-tests-') as directory:
        plan = subprocess.check_output([sys.executable, str(ROOT / 'reproduce.py'),
                                        '--data', 'unused', '--dry-run'], cwd=directory, text=True)
        rows = [json.loads(line) for line in plan.splitlines()]
        assert len(rows) == 330 and len({json.dumps(row, sort_keys=True) for row in rows}) == 330
        for script in ['check_powers.py', 'check_repair.py', 'preflight.py', 'check_margin.py', 'smoke_test.py']:
            # Import torch first to avoid conflicting OpenMP import order in older local environments.
            command = 'import sys,torch,runpy;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2],run_name="__main__")'
            result = subprocess.run([sys.executable, '-c', command, str(ROOT), str(ROOT / script)],
                                    cwd=directory, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                sys.stderr.write(result.stdout)
                raise SystemExit(f'FAILED: {script}')
    print('All checks passed.')


if __name__ == '__main__':
    main()
