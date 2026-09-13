from pathlib import Path
import gzip,hashlib,itertools,json,math,os,random,sys,time
ROOT=Path(__file__).resolve().parent/'equal_time'
B=ROOT.parent.parent
H=B/'equal_walltime_harness_engineering_20260909'
SCIENCE=B/'baseline_calibration_and_transport_20260908'
PY=B/'fixed_evaluation_confirmation_20260908/.venv/bin/python'
PROBE=B/'fixed_evaluation_confirmation_20260908/calibration_worker.py'
PROBE_SHA='b46e11a3aed851f9e191b1b17d9f43607655124d5352d68e3f943690fd0ea940'
CPUS=tuple(range(64,80,2))
ARMS=('de_proxy','de_random4','ucb_w500_c1')
STRATA=('small_low','small_high','medium_low','medium_high')
BUDGET=120.
CHECKPOINTS=(60.,120.)
PROTOCOL='enhancement38_matched_elapsed_time_v1'
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(v):return hashlib.sha256(canonical(v)).hexdigest()


def read(p):return json.loads(Path(p).read_text())


def put(p,v):
    with Path(p).open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def progress(p,v):
    temp=Path(str(p)+'.tmp')
    temp.write_bytes(canonical(v))
    temp.replace(p)


def packed(p,v):
    with Path(p).open('xb') as out:
        with gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0,compresslevel=5) as z:
            z.write(canonical(v))


def unpack(p):
    with gzip.open(p,'rt') as f:return json.load(f)


def probe_summary(raw):
    assert raw['cpu'] in CPUS and len(raw['samples'])==3
    rates=[]
    for s in raw['samples']:
        used=float(s['cpu_seconds']);wall=float(s['wall_seconds'])
        rate=float(s['iterations_per_cpu_second'])
        assert all(math.isfinite(v) and v>0 for v in (used,wall,rate))
        assert abs(s['ratio']-used/wall)<1e-12
        assert s['ratio']>=.95,'Probe CPU availability below .95'
        rates.append(rate)
    return sum(rates)/3


def paired_resource(pre,post,reports):
    a=probe_summary(pre);b=probe_summary(post)
    assert pre['cpu']==post['cpu']
    assert .85<=b/a<=1.15,'Probe rate changed by more than 15%'
    ratios=[r['cpu_seconds']/r['search_wall_seconds'] for r in reports]
    assert len(ratios)==3 and all(math.isfinite(x) and .90<=x<=1.05 for x in ratios)
    assert max(ratios)-min(ratios)<=.05,'Paired CPU availability differs by more than .05'
    assert len({r['initial_identity'] for r in reports})==1
    assert len({tuple(r['reference']) for r in reports})==1
    assert len({tuple(r['wide_reference']) for r in reports})==1
    return dict(pre_rate=a,post_rate=b,post_pre=b/a,cpu_wall_ratios=ratios)


def input_lock():
    f=read(ROOT/'FREEZE.json')
    assert f['formal_runs_before_freeze']==0
    for path,h in f['files'].items():assert sha(path)==h,('Frozen source/input changed',path)
    tasks=read(ROOT/'TRIPLETS.json')
    assert len(tasks)==240 and len({t['id'] for t in tasks})==240
    assert {t['cpu'] for t in tasks}==set(CPUS)
    for t in tasks:assert sha(t['input_file'])==t['input_sha256']
    return sha(ROOT/'FREEZE.json')
