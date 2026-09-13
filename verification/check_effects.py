"""Dependency-free independent arithmetic check of the supplied base effects.

python check_effects.py
This checks supplied summaries, not the original optimizers or genuine external
replication. No local installations, network connections, or original paths needed.
"""
from pathlib import Path
import itertools
import json
import math
import statistics

root = Path(__file__).resolve().parent
d = json.loads((root/'sensitivity_results.json').read_text())
checked=[]
for r in d['rows']:
    a = r['values'];n=len(a)
    mean=statistics.mean(a)
    assert math.isclose(mean,r['mean'],abs_tol=1e-12)
    assert len(set(map(abs,a)))==n and all(a)
    # Wilcoxon: independently enumerate subsets of ranks 1..n (no SciPy).
    order=sorted(range(n),key=lambda i:abs(a[i]))
    wplus=sum(rank for rank,i in enumerate(order,1) if a[i]>0)
    total=n*(n+1)//2
    observed=min(wplus,total-wplus)
    counts=[0]*(total+1);counts[0]=1
    for rank in range(1,n+1):
        for s in range(total,rank-1,-1):counts[s]+=counts[s-rank]
    pw=min(1.,2*sum(counts[:observed+1])/2**n)
    assert abs(pw-r['wilcoxon_raw'])<1e-12
    # Enumerate sign changes in the original-valued effects, not in ranks.
    ps=sum(abs(statistics.mean(v*s for v,s in zip(a,signs)))>=abs(mean)-1e-14
           for signs in itertools.product((-1,1),repeat=n))/2**n
    assert abs(ps-r['signflip_raw_recomputed'])<1e-12
    checked.append(dict(group=r['group'],label=r['label'],mean=mean,signflip_raw=ps,wilcoxon_raw=pw))
for g in {r['group'] for r in d['rows']}:
    rows=[r for r in d['rows'] if r['group']==g]
    for raw,adjusted in [('wilcoxon_raw','wilcoxon_holm'),('signflip_raw_recomputed','signflip_holm_recomputed')]:
        last=0.
        for i,r in enumerate(sorted(rows,key=lambda r:r[raw])):
            last=max(last,min(1.,(len(rows)-i)*r[raw]))
            assert abs(last-r[adjusted])<1e-12
print(json.dumps(dict(status='PASS',checked_contrasts=len(checked),
                     checks='means, exact sign flips, exact Wilcoxon ranks, Holm families',
                     scope='arithmetic verification of supplied effects; not optimizer replication')))
