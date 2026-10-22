"""Draw the evaluation-cost figure from the archived S3 analysis.

  python make_fig_evaluation_cost.py ANALYSIS_S3_JSON
Writes fig4_evaluation_cost.svg, .png and fig4_evaluation_cost_source.csv next to this script.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COSTS = (0, 10, 25, 50)
SERIES = [('a', 'nsga2_pre-nsga2', 'NSGA-II+RBP − NSGA-II', '#1f5a8c', 'o'),
          ('b', 'always-nsga2', 'RBP − NSGA-II', '#1f5a8c', 'o'),
          ('b', 'always-moead', 'RBP − MOEA/D', '#8c8c8c', 's')]


def main():
    a = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    rows = {c['contrast']: c for c in a['primary'] + a['secondary']}
    table = []
    for panel, key, label, _, _ in SERIES:
        for d in COSTS:
            c = rows[f'{key}@{d}ms']
            table.append(dict(panel=panel, contrast=label, added_ms=d, mean=c['mean'],
                              ci_low=c['ci95_stratified'][0], ci_high=c['ci95_stratified'][1],
                              positive=c['positive'], negative=c['negative']))
    with open(HERE / 'fig4_evaluation_cost_source.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'svg.fonttype': 'none',
                         'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 62 / 25.4))
    for ax, panel, title in zip(axes, 'ab', ('Plug-in gain', 'RBP versus standard algorithms')):
        offs = {'NSGA-II+RBP − NSGA-II': 0, 'RBP − NSGA-II': -0.6, 'RBP − MOEA/D': 0.6}
        for p, key, label, color, marker in SERIES:
            if p != panel:
                continue
            pts = [r for r in table if r['contrast'] == label]
            x = [r['added_ms'] + offs[label] for r in pts]
            y = [r['mean'] for r in pts]
            err = [[r['mean'] - r['ci_low'] for r in pts], [r['ci_high'] - r['mean'] for r in pts]]
            ax.errorbar(x, y, yerr=err, color=color, marker=marker, ms=3.5, lw=1.0, capsize=2,
                        elinewidth=0.8, label=label)
        ax.axhline(0, color='#444444', lw=0.6, ls=(0, (3, 2)))
        ax.set_xticks(COSTS)
        ax.set_xlabel('Added cost per exact evaluation (ms)')
        ax.set_ylabel('Normalized HV difference at 120 s')
        ax.set_title(title, fontsize=7.5, fontweight='bold', loc='left')
        ax.text(-0.13, 1.04, panel, transform=ax.transAxes, fontsize=9, fontweight='bold')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        if panel == 'b':
            ax.legend(frameon=False, loc='lower right')
            ax.set_ylim(0, 0.05)
    fig.text(0.5, 0.005, 'Positive values favor the first-named method; bars are 95% stratified-bootstrap intervals over 16 bases',
             ha='center', fontsize=6.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(HERE / 'fig4_evaluation_cost.svg')
    fig.savefig(HERE / 'fig4_evaluation_cost.png', dpi=300)


if __name__ == '__main__':
    main()
