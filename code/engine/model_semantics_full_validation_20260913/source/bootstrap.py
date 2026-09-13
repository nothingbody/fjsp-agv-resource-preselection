"""Only import frozen dependencies; bytecode writes disabled by launcher."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'fixed_evaluation_confirmation_20260908'
STUDY=ROOT.parent/'fixed_evaluation_component_ablation_20260909'
sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_legacy():
    lock=json.loads((ROOT/'LEGACY_SOURCES.json').read_text())['dependencies']
    for n,h in lock.items():assert sha(ROOT.parent/n)==h,n
    return len(lock)
def restore(record):
    from src.problem.instance import FJSPAGVInstance
    import numpy as np
    p=STUDY/record['input_file']
    assert sha(p)==record['input_sha256']
    d=json.loads(p.read_text())
    inst=FJSPAGVInstance(d['num_jobs'],d['num_machines'],d['num_agv'],d['num_operations'])
    for k,v in d.items():
        if k=='processing_times':v={(j,o,m):t for j,o,m,t in v}
        elif k=='compatible_machines':v={(j,o):ms for j,o,ms in v}
        elif k in ('distance_matrix','machine_proc_power','machine_idle_power','machine_setup_power','machine_setup_time','layout_coordinates'):
            v=None if v is None else np.asarray(v,dtype=float)
        elif k in ('agv_load_power_params','agv_empty_power_params'):v=tuple(v)
        setattr(inst,k,v)
    return inst
