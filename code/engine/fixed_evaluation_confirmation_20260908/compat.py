"""Use frozen algorithm functions without importing unused neural components."""
from __future__ import annotations
import ast
import importlib
from pathlib import Path
import sys
import types
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'lib/base'
sys.path.insert(0,str(BASE))
# No scientific module is replaced. Only an eager convenience __init__ that
# imports torch/branching is bypassed. Reference-mode validation tests normal import.
if 'src.algorithm.aos2' not in sys.modules:
    import src.algorithm
    package=types.ModuleType('src.algorithm.aos2')
    package.__path__=[str(BASE/'src/algorithm/aos2')]
    package.__package__='src.algorithm.aos2'
    sys.modules[package.__name__]=package
    setattr(src.algorithm,'aos2',package)
from src.algorithm.aos2 import finegrained as fine
from src.algorithm.aos2.rng_ledger import RNGLedger,legacy_initialization_seed
from src.algorithm.aos2.credits import objective_array,fixed_reference_hv
from src.algorithm.nsga3.selection import non_dominated_sort,generate_reference_points
from src.algorithm.nsga3.encoding import (Chromosome,spt_chromosome,energy_chromosome,
    balance_chromosome,random_chromosome)
from src.algorithm.nsga3 import decoding
from data.loader import load_benchmark_set

names={'_largest_remainder_quotas','population_initialization_quotas','initialize_population',
       'crowding_distance','_objective_key','stable_unique_solutions','update_archive','update_generation_archive'}
tree=ast.parse((BASE/'experiments/run_revision_aos.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(functions)==len(names)
exec(compile(ast.Module(body=functions,type_ignores=[]),'frozen_legacy_helpers','exec'),globals())

ORIGINAL_INTRA=fine.build_intra_state


class LightUCB(fine.DelayedExposureUCB):
    def build_base_state(self,*args,**kwargs):
        return np.empty(0,dtype=float)


def set_diagnostic_light(enabled):
    fine.build_intra_state=(lambda *args,**kwargs:(np.empty(0,dtype=float),[])) if enabled else ORIGINAL_INTRA


class CountCalls:
    def __enter__(self):
        self.original_evaluate,self.original_decode=decoding.evaluate,decoding.decode
        self.evaluations,self.decodes=0,0
        def ev(*a,**kw):
            self.evaluations+=1
            return self.original_evaluate(*a,**kw)
        def de(*a,**kw):
            self.decodes+=1
            return self.original_decode(*a,**kw)
        decoding.evaluate,decoding.decode=ev,de
        return self
    def __exit__(self,*args):
        decoding.evaluate,decoding.decode=self.original_evaluate,self.original_decode


def instance(dataset,name):
    return dict(load_benchmark_set(str(BASE/'data/benchmarks'/dataset),num_agv=3))[name]
