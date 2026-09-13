"""Recalculate matched HV effects and all 1,920 pool metrics using Python stdlib.

Uses supplied saved objective-derived metrics, not new optimization. Run from any
directory with Python >=3.10: python /path/to/check_records.py
"""
from pathlib import Path
import collections
import gzip
import hashlib
import itertools
import json
import math
import statistics as st

ROOT=Path(__file__).resolve().parent
def read(name): return json.loads(gzip.decompress((ROOT/'data'/name).read_bytes()))
manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
for name,m in manifest['files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==m['sha256'],name

def matched(rows,first,second,metric,expected):
    cells={}
    for r in rows:
        if r['arm'] in (first,second):
            key=(r['base_id'],r['regime'],r['seed'])
            assert r['arm'] not in cells.setdefault(key,{})
            cells[key][r['arm']]=r[metric]
    assert len(cells)==240 and all(len(d)==2 for d in cells.values())
    bases=collections.defaultdict(list)
    for k,d in cells.items():bases[k[0]].append(d[first]-d[second])
    assert len(bases)==16 and all(len(v)==15 for v in bases.values())
    effect=st.mean(st.mean(v) for v in bases.values())
    assert abs(effect-expected)<1e-12,(effect,expected)
    return dict(cells=240,bases=16,mean=effect)

primary=[r for r in read('primary_run_metrics.json.gz') if r['evaluations']==10000]
a=matched(primary,'de_proxy','de_random4','hv_normalized',0.02478467563209115)
enh=read('enhancement_endpoint_metrics.json.gz')
b=matched([r for r in enh if r['family']=='equal_time' and r['endpoint']==120],
          'physical4','random4','hv_1_1',0.0158864981439758)
c=matched([r for r in enh if r['family']=='nsga2' and r['endpoint']==10000],
          'physical4','random4','hv_1_1',0.01584829771115471)

def tau_b(x,y):
    concordant=discordant=only_x=only_y=0
    for i,j in itertools.combinations(range(len(x)),2):
        dx=(x[i]>x[j])-(x[i]<x[j]);dy=(y[i]>y[j])-(y[i]<y[j])
        if not dx and not dy:continue
        if not dx:only_x+=1
        elif not dy:only_y+=1
        elif dx==dy:concordant+=1
        else:discordant+=1
    n=concordant+discordant
    den=math.sqrt((n+only_x)*(n+only_y))
    return (concordant-discordant)/den if den else None

pools=read('audited_pools.json.gz');assert len(pools)==1920
summary=collections.defaultdict(list)
for r in pools:
    x,y=r['proxy_g'],r['exact_g'];i=r['selected']
    assert len(x)==len(y)==4 and x[i]==min(x)
    top=float(y[i]<=min(y)+1e-12)
    regret=y[i]-min(y)
    tau=tau_b(x,y)
    assert abs(top-float(r['top1']))<1e-12
    assert abs(regret-r['regret'])<1e-12
    assert tau is not None and abs(tau-r['kendall_tau_b'])<1e-12
    den=st.mean(y)-min(y);assert den>0
    capture=(st.mean(y)-y[i])/den
    assert abs(capture-r['captured_local_value'])<1e-10
    summary[r['base_id']].append((top,tau,regret,capture))
assert len(summary)==16 and all(len(v)==120 for v in summary.values())
means=[st.mean(st.mean(row[j] for row in v) for v in summary.values()) for j in range(4)]
print(json.dumps(dict(status='PASS',primary=a,equal_time=b,nsga2=c,pools_checked=1920,
                     pool_means=dict(zip(['top1','tau_b','regret','captured_fraction'],means)),
                     scope='saved run-metric aggregation and independent pool arithmetic, not optimizer replication')))
