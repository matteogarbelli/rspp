"""Manifest-driven reproduction; archived inputs are never overwritten."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'reproduction_manifest.json').read_text())
OUT = ROOT / 'outputs'


def run_module(module, args=(), cwd=ROOT):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)
    env['MPLCONFIGDIR'] = str(OUT / 'matplotlib')
    subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=cwd,
                   env=env, check=True)


def log(action, started, result):
    OUT.mkdir(exist_ok=True)
    with (OUT / 'runs.jsonl').open('a') as f:
        f.write(json.dumps({'action': action, 'manifest_sha256': hashlib.sha256(
            (ROOT / 'reproduction_manifest.json').read_bytes()).hexdigest(),
            'elapsed_seconds': time.perf_counter()-started, 'result': result})+'\n')


def kernels():
    source = ROOT / 'experiments/rspp/kernels.c'
    subprocess.run(['cc', '-O3', '-shared', '-fPIC', '-o',
                    str(source.with_name('libkernels.so')), str(source), '-lm'], check=True)


def verify():
    manifest = json.loads((ROOT / 'FILES.sha256.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    kernels()
    subprocess.run([sys.executable, str(ROOT/'checks/audit.py')], cwd=ROOT, check=True)
    return {'checksummed_files': len(manifest), 'status': 'pass'}


def smoke():
    kernels()
    from experiments.rspp.algorithm import run
    from experiments.rspp.instance import RSPPPair
    spec = CFG['smoke']
    pair = RSPPPair.load(ROOT/'data/rsppbench'/f"{spec['instance']}.json")
    result = run(pair, arm=spec['arm'], seed=spec['seed'], cfg={'max_gen': spec['max_gen']})
    OUT.mkdir(exist_ok=True)
    (OUT/'smoke.json').write_text(json.dumps(result, indent=2)+'\n')
    assert result['feasible'], 'Smoke run failed hard feasibility'
    return {'status': 'pass', 'objectives': result['objectives'], 'n_evals': result['n_evals']}


def bench():
    destination = OUT/'regenerated_instances'
    run_module('experiments.rsppbench.generate', ['--out', destination,
        '--seeds', *CFG['benchmark']['seeds']])
    archived = json.loads((ROOT/CFG['instance_manifest']).read_text())['suite']
    for entry in archived:
        actual = hashlib.sha256((destination/entry['file']).read_bytes()).hexdigest()
        assert actual == entry['sha256'], entry['name']
    return {'status': 'pass', 'byte_identical_instances': len(archived)}


def replay():
    kernels()
    from experiments.rspp.algorithm import run
    from experiments.rspp.instance import RSPPPair
    import numpy as np
    spec = CFG['replay']
    with gzip.open(ROOT/CFG['archive'], 'rt') as f:
        expected = next(row for line in f if all((row := json.loads(line)).get(k) == v
                        for k,v in spec.items()))
    pair = RSPPPair.load(ROOT/'data/rsppbench'/f"{spec['instance']}.json")
    actual = run(pair, arm=spec['arm'], seed=spec['seed'], cfg=expected['config'])
    error = float(np.max(np.abs(np.asarray(actual['objectives'])-expected['objectives'])))
    assert error <= CFG['validation']['absolute_tolerance'], error
    for key in ['feasible', 'n_evals', 'routes', 'front']:
        assert actual[key] == expected[key], key
    (OUT/'replay.json').write_text(json.dumps(actual, indent=2)+'\n')
    return {'status': 'pass', 'objective_max_abs_error': error,
            'matching_fields': ['feasible','n_evals','routes','front'], 'case': spec}


def figures():
    # Stage only research inputs. The original analysis writes to relative paths.
    stage = OUT/'reanalysis'
    (stage/'submission_cor').mkdir(parents=True, exist_ok=True)
    for relative in ['data/rsppbench', 'experiments/results']:
        shutil.copytree(ROOT/relative, stage/relative, dirs_exist_ok=True)
    run_module('experiments.analysis.figures', cwd=stage)
    def macros(file):
        return dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}', file.read_text()))
    expected = macros(ROOT/'reference/results_macros.tex')
    actual = macros(stage/'submission_cor/results_macros.tex')
    differences = {k: [v, actual.get(k)] for k,v in expected.items()
                   if k != 'NREFS' and v != actual.get(k)}
    assert not differences, differences
    for name in ['tab_bench.tex', 'tab_indicators.tex', 'tab_family.tex', 'tab_main.tex']:
        assert (stage/'submission_cor'/name).read_bytes() == (ROOT/'reference'/name).read_bytes(), name
    return {'status': 'pass', 'matching_experimental_macros': len(expected)-('NREFS' in expected),
            'matching_tables': 4, 'bibliography_count_excluded': True}


def campaign(kind):
    spec = CFG[kind]
    destination = OUT/'rerun/campaign.jsonl'
    # A fresh campaign is separate from the bundled archive, including its environment.
    run_module('experiments.campaign.run', ['--bench', ROOT/'data/rsppbench',
        '--out', destination, '--seeds', spec['seeds'], '--procs', spec['processes'],
        '--studies', *spec['studies']])
    return {'status': 'complete', 'output': str(destination.relative_to(ROOT))}


def selftest():
    kernels()
    from experiments.rspp import selftest as st
    result = {'kernel_max_abs': st.check_kernels(),
              'metric_axioms': st.check_metric_axioms(),
              'vertex_preservation': st.check_vertex_preservation(),
              'fine_reference_comparison': st.check_resampling_bound(),
              'corner_cutting_comparison': st.check_naive_underreport(),
              'interpretation': 'Fine discrete references approximate continuous distance; the count below a reference is not a certified count below continuous distance.'}
    (OUT/'selftest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['verify','kernels','smoke','replay','bench','figures','campaign','timing','selftest'])
    action = p.parse_args().action
    OUT.mkdir(exist_ok=True)
    started = time.perf_counter()
    try:
        result = campaign(action) if action in ('campaign','timing') else globals()[action]()
    except Exception as exc:
        log(action, started, {'status': 'failed', 'error': str(exc)})
        raise
    log(action, started, result)
    print(json.dumps(result, indent=2))
