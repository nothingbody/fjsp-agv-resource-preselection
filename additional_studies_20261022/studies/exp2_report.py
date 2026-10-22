"""Collect exp2 analysis outputs into one Markdown report (numbers only; interpretation is written separately).

  python exp2_report.py --out out > RESULTS_TABLES.md
"""
import argparse
import json
from pathlib import Path


def fmt(x, d=6):
    return f'{x:+.{d}f}'


def table(items, title):
    lines = [f'### {title}', '', '| 比较 | 端点 | 指标 | 均值差 | 95% 分层区间 | 正/零/负 | 符号翻转 p | Holm p |',
             '|---|---|---|---:|---|---|---:|---:|']
    for c in items:
        lo, hi = c['ci95_stratified']
        ph = c.get('p_holm')
        lines.append(f"| {c['contrast']} | {c['endpoint']} | {c['metric']} | {fmt(c['mean'])} | [{fmt(lo)}, {fmt(hi)}] | "
                     f"{c['positive']}/{c['zero']}/{c['negative']} | {c['p_signflip']:.4g} | {'' if ph is None else f'{ph:.4g}'} |")
    return '\n'.join(lines) + '\n'


def describe(d, title, keys):
    lines = [f'### {title}', '', '| 方法 | ' + ' | '.join(keys) + ' |', '|---|' + '---:|' * len(keys)]
    for arm, v in d.items():
        lines.append(f'| {arm} | ' + ' | '.join(f'{v[k]:.4f}' if isinstance(v.get(k), float) else str(v.get(k, '')) for k in keys) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='out')
    a = p.parse_args()
    out = Path(a.out)
    parts = ['# exp2 结果表（自动生成）', '']
    sel = Path('SELECTION.json')
    if sel.exists():
        s = json.loads(sel.read_text())
        parts.append('## B1 调参选择\n')
        for fam in ('nsga2', 'moead'):
            parts.append(f"- {fam}: {s[fam]['params']}，平均秩 {s[fam]['mean_rank']:.2f}，平均 HV {s[fam]['mean_hv']:.4f}")
            parts.append('')
            parts.append('| 排名 | 参数 | 平均秩 | 平均 HV |')
            parts.append('|---:|---|---:|---:|')
            for i, r in enumerate(s[fam + '_table'], 1):
                parts.append(f"| {i} | {r['params']} | {r['mean_rank']:.2f} | {r['mean_hv']:.4f} |")
            parts.append('')
    for study in ('B2e', 'B2t', 'Ce', 'Ct', 'A'):
        f = out / study / f'analysis_{study}.json'
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        parts.append(f'## {study}（{d["runs"]} 次运行）\n')
        if d.get('primary'):
            parts.append(table(d['primary'], '主族'))
        parts.append(table(d['secondary'], '次要比较'))
        keys = ['runs', 'hv', 'evaluations', 'eval_share', 'score_share', 'cpu_ratio', 'runs_cpu_ratio_below_0_9', 'makespan_min']
        if study == 'A':
            keys.insert(2, 'hv_test')
        parts.append(describe(d['describe'], '各方法均值（主端点）', keys))
        if 'reproduction_vs_B2e' in d:
            parts.append(f"复现门槛（与 B2e 逐位比较）：{d['reproduction_vs_B2e']}\n")
    f = out / 'B3' / 'analysis_B3.json'
    if f.exists():
        d = json.loads(f.read_text())
        parts.append(f"## B3 CP-SAT 下界（{d['instances']} 个快照，状态 {d['status']}）\n")
        for study in ('B2e', 'B2t'):
            parts.append(f'### {study}：最终存档最小完工时间相对下界的差距\n')
            parts.append('| 方法 | 平均差距 | 中位差距 | small_low | small_high | medium_low | medium_high |')
            parts.append('|---|---:|---:|---:|---:|---:|---:|')
            for arm, v in d[study].items():
                b = v['by_stratum']
                parts.append(f"| {arm} | {v['mean_gap']:.2%} | {v['median_gap']:.2%} | " +
                             ' | '.join(f"{b.get(s, float('nan')):.2%}" for s in ('small_low', 'small_high', 'medium_low', 'medium_high')) + ' |')
            parts.append('')
    print('\n'.join(parts))


if __name__ == '__main__':
    main()
