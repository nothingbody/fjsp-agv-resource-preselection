"""Portable entry point; frozen scientific functions, no legacy job manager."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
sys.dont_write_bytecode = True

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance', required=True, type=Path)
    p.add_argument('--family', choices=['primary','score','knn','pool','nsga2','time','model'], default='primary')
    p.add_argument('--arm', default='de_proxy')
    p.add_argument('--policy', choices=['physical','random','exact'], default='physical')
    p.add_argument('--pool-size', type=int, choices=[2,4], default=4)
    p.add_argument('--model', choices=['original','first_rightshift','no_same_machine_service'], default='original')
    p.add_argument('--dataset', default='study30_synthetic_v1')
    p.add_argument('--name', help='Frozen instance name; defaults to input basename')
    p.add_argument('--seed', type=int, default=909302001)
    p.add_argument('--evaluations', type=int, default=10000)
    p.add_argument('--seconds', type=float, default=120.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists; choose a new path.')
    manifest = json.loads((ROOT/'CODE_MANIFEST.json').read_text())
    for rel, entry in manifest['files'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() != entry['public_sha256']:
            raise RuntimeError('Source differs from released version: '+rel)
    E = ROOT/'code/engine'
    for d in ('fixed_evaluation_component_ablation_20260909',
              'fixed_evaluation_simple_score_20260909',
              'fixed_evaluation_knn_preselection_20260909',
              'enhancement_v3_8_four_families_20260913',
              'equal_walltime_harness_engineering_20260909'):
        sys.path.insert(0, str(E/d))
    import common30
    inst = common30.s24.restore_instance(json.loads(a.instance.read_text()))
    name = a.name or a.instance.stem
    ends = tuple(sorted({x for x in (2000,5000,a.evaluations) if x <= a.evaluations}))
    task = dict(dataset=a.dataset, name=name, seed=a.seed, policy=a.policy, pool_size=a.pool_size)
    if a.family in ('primary','score','knn'):
        module = __import__({'primary':'common30','score':'common31','knn':'common32'}[a.family])
        result = module.run_solver(inst,a.dataset,name,a.seed,a.arm,a.evaluations,ends)
    elif a.family == 'pool':
        from pool_core38 import solve_pool
        result = solve_pool(inst,task,a.evaluations,ends)
    elif a.family == 'nsga2':
        from nsga_core38 import solve_nsga
        result = solve_nsga(inst,task,a.evaluations,ends)
    elif a.family == 'time':
        from timed_solver26 import solve
        result = solve(inst,a.dataset,name,a.seed,a.arm,a.seconds,(a.seconds,))
    else:
        sys.path.insert(0,str(E/'model_semantics_full_validation_20260913/source'))
        import models
        with models.semantics(a.model):
            result = common30.run_solver(inst,a.dataset,name,a.seed,a.arm,a.evaluations,ends)
    import numpy as np
    def encode(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result,f,default=encode,allow_nan=False)
    print(json.dumps(dict(status='PASS', family=a.family, output=str(a.output),
                         scope='new execution, not automatically an exact replay of published cells')))

if __name__ == '__main__': main()
