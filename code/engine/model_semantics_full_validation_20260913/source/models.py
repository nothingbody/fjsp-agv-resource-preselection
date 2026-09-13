"""Three separate decoder semantics. No combined repair is tested here."""
import bootstrap
import copy
from contextlib import contextmanager
import numpy as np
import compat,online_core
from surrogate import EncodingFeatures as OriginalFeatures
from nohop_decoder import decode as nohop_decode
from nohop_features import EncodingFeatures as NoHopFeatures
ORIGINAL_DECODE=compat.decoding.decode
MODES=('original','first_rightshift','no_same_machine_service')

def first_rightshift(inst,s):
    """Maximize each first-operation start with other ops and all trips fixed.
    This is a limited timing repair, not an optimal energy scheduler.
    """
    out=copy.copy(s)
    out.op_start=s.op_start.copy();out.op_end=s.op_end.copy()
    out.machine_schedule={m:list(ops) for m,ops in s.machine_schedule.items()}
    for m,ops in out.machine_schedule.items():
        if len(ops)<2:continue
        j,o,start,end=ops[0];nj,no,ns,ne=ops[1]
        setup=float(inst.machine_setup_time[m]) if inst.machine_setup_time is not None and j!=nj else 0.
        latest_end=ns-setup
        if o+1<inst.num_operations[j]:
            succ=(j,o+1)
            # Moving this completion must not invalidate an already fixed pickup.
            bound=out.transport_start[succ] if succ in out.transport_start else out.op_start[succ]
            latest_end=min(latest_end,bound)
        latest_start=latest_end-(end-start)
        if latest_start>start+1e-10:
            out.op_start[j,o]=float(latest_start)
            out.op_end[j,o]=float(latest_end)
            ops[0]=(j,o,float(latest_start),float(latest_end))
    return out

@contextmanager
def semantics(mode):
    assert mode in MODES
    prev_decode,prev_features=compat.decoding.decode,online_core.EncodingFeatures
    try:
        if mode=='original':compat.decoding.decode=ORIGINAL_DECODE
        elif mode=='first_rightshift':
            compat.decoding.decode=lambda c:first_rightshift(c.instance,ORIGINAL_DECODE(c))
        else:compat.decoding.decode=nohop_decode
        online_core.EncodingFeatures=NoHopFeatures if mode=='no_same_machine_service' else OriginalFeatures
        yield
    finally:
        compat.decoding.decode,online_core.EncodingFeatures=prev_decode,prev_features

def features(inst,scale,mode):
    return (NoHopFeatures if mode=='no_same_machine_service' else OriginalFeatures)(inst,scale)
