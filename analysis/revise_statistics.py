"""Retrospective uncertainty sensitivity; read saved effects, never run optimizers.

Run on the compute server. Original estimates, intervals and test families remain
unchanged. Creates self-contained inputs for a separate stdlib checker.
"""
from pathlib import Path
import hashlib
import json
import collections
import numpy as np
import scipy
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reproduction'
OUT.mkdir(exist_ok=True)
hashes = {}

def read(name):
    p = ROOT / 'evidence' / name
    b = p.read_bytes()
    hashes[name] = hashlib.sha256(b).hexdigest()
    return json.loads(b)

rows = []
def add(group, label, vals, ci, expected, raw=None, holm=None):
    if isinstance(vals, dict):
        ids = sorted(vals)
        a = [vals[k] for k in ids]
    else:
        ids, a = [f'base_{i+1:02d}' for i in range(len(vals))], vals
    assert len(a) == 16 and abs(float(np.mean(a))-expected) < 1e-12
    rows.append(dict(group=group, label=label, base_ids=ids, values=a,
                     mean=expected, original_stratified_ci95=ci,
                     original_signflip_raw=raw, original_signflip_holm=holm))

orig = read('original_PRIMARY_ANALYSIS.json')
for r in orig['primary']:
    add('Component (5)', r['comparator'], r['base_effects'],
        r['bootstrap95_unadjusted'], r['mean_absolute_normalized_hv_difference'],
        r['raw_signflip_p'], r['holm_p_five'])
sem = read('SEMANTICS_RESULTS.json')
for r in sem['primary_five_contrasts']:
    add('Model (5)', r['name'], r['base_effects'], r['bootstrap95_pointwise'],
        r['mean'], holm=r['holm_p_five'])
enh = read('enhancement_RESULTS.json')
for r in enh['comparisons']:
    ep = 120 if r['family'] == 'equal_time' else 10000
    if r['endpoint'] == ep and r['metric'] == 'hv_1_1':
        add(r['family'], r['contrast'], r['base_effects'], r['ci95_unadjusted'],
            r['mean'], r['p_raw'])
for study in (31,32):
    for r in read(f'score{study}_PRIMARY_ANALYSIS.json')['primary']:
        add(f'Score study {study} (2)',r['comparator'],r['base_effects'],
            r['bootstrap95_unadjusted'],r['mean_absolute_normalized_hv_difference'],
            r['raw_signflip_p'],r['holm_p_two'])
assert len(rows) == 23
signs = ((np.arange(65536)[:, None] >> np.arange(16)) & 1)*2-1
for i, r in enumerate(rows):
    a = np.array(r['values'], dtype=float)
    r['bootstrap_seed'] = 2026091400 + i
    rng = np.random.default_rng(r['bootstrap_seed'])
    samples = a[rng.integers(0, 16, size=(100000, 16))].mean(axis=1)
    r['unstratified_ci95'] = np.quantile(samples, [0.025, 0.975]).tolist()
    r['signflip_raw_recomputed'] = float(np.mean(np.abs(signs @ a / 16) >= abs(a.mean())-1e-14))
    if r['original_signflip_raw'] is not None:
        assert abs(r['signflip_raw_recomputed']-r['original_signflip_raw']) < 1e-12
    assert np.all(a != 0) and len(set(abs(a))) == 16, 'Exact Wilcoxon requires no zeros/ties here'
    r['wilcoxon_raw'] = float(wilcoxon(a, alternative='two-sided', method='exact').pvalue)
    r['positive'] = int(sum(a > 0))
for family in {r['group'] for r in rows}:
    subset = [r for r in rows if r['group'] == family]
    for raw, adjusted in [('wilcoxon_raw','wilcoxon_holm'),('signflip_raw_recomputed','signflip_holm_recomputed')]:
        last = 0.
        for i, r in enumerate(sorted(subset, key=lambda r:r[raw])):
            last = max(last, min(1., (len(subset)-i)*r[raw]))
            r[adjusted] = last
            if adjusted == 'signflip_holm_recomputed' and r['original_signflip_holm'] is not None:
                assert abs(last-r['original_signflip_holm']) < 1e-12
report = dict(status='PASS', new_searches=0, retrospective=True,
              original_primary_analysis_unchanged=True, bootstrap_resamples=100000,
              versions=dict(numpy=np.__version__,scipy=scipy.__version__), input_sha256=hashes, rows=rows)
(OUT/'sensitivity_results.json').write_text(json.dumps(report, indent=2))
(ROOT/'evidence/UNCERTAINTY_SENSITIVITY.json').write_text(json.dumps(report, indent=2))

names = {'de_random4':'Full--random', 'ucb_w500_c1':'Full--UCB (framework)',
 'proxy_no_transport':'Full--no-transport','proxy_no_energy':'Full--no-energy',
 'proxy_no_load':'Full--no-load','physical2_minus_exact2':'Physical--exact, K=2',
 'physical4_minus_exact4':'Physical--exact, K=4','physical2_minus_physical4':'Physical: K=2 minus K=4',
 'exact2_minus_exact4':'Exact: K=2 minus K=4','selection_effect_k2_minus_k4':'Selection-effect difference',
 'physical4_minus_random4':'Physical--random','physical4_minus_ucb':'Physical--UCB (framework)',
 'original_full_minus_random':'Original: full--random',
 'first_rightshift_full_minus_random':'Rightshift: full--random',
 'no_same_machine_service_full_minus_random':'No-service: full--random',
 'first_rightshift_change_in_matched_effect':'Rightshift change',
 'no_same_machine_service_change_in_matched_effect':'No-service change',
 'de_weighted_sum':'Full--weighted sum','de_knn':'Full--three-neighbor predictor',
 'de_knn3':'Full--three-neighbor predictor'}
def fmt(x):
    s = f'{x:.3g}'
    if 'e' in s:
        a,b=s.split('e');return rf'${a}\times10^{{{int(b)}}}$'
    return s
def ci(xs): return '['+', '.join(fmt(x) for x in xs)+']'
lines = [r'\begin{table}[H]',r'\centering\scriptsize',
 r'\caption{Retrospective uncertainty sensitivities for component, model, pool, backbone, time and alternative-score primary HV contrasts. Original stratified intervals are retained. Unstratified intervals resample sixteen bases with replacement 100,000 times. Wilcoxon columns are raw and Holm-adjusted exact signed-rank probabilities, with the original comparison-family sizes. No zero differences or tied absolute ranks occur. These are sensitivity analyses, not replacement primary tests or independent confirmations.}\label{tab:uncertainty}',
 r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',
 r'Family / comparison & Mean & Stratified 95\% & Unstratified 95\% & W raw $p$ & W Holm $p$\\',r'\midrule']
last=None
for r in rows:
    if r['group'] != last:
        lines.append(r'\multicolumn{6}{l}{\textit{'+r['group'].replace('_',' ')+r'}}\\')
        last=r['group']
    label=names.get(r['label'],r['label'].replace('_',' '))
    lines.append(' & '.join([label,fmt(r['mean']),ci(r['original_stratified_ci95']),
                             ci(r['unstratified_ci95']),fmt(r['wilcoxon_raw']),fmt(r['wilcoxon_holm'])])+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(ROOT/'tables/uncertainty_sensitivity.tex').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status='PASS', contrasts=len(rows), output=str(OUT),
                     labels=[r['label'] for r in rows])))
