"""Cumulative and maximum-driver deviation of the archived returned plans."""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..rspp.instance import RSPPPair
from ..rspp.solution import Evaluator
from .style import ARM_COLOR, ARM_LABEL, save

ARMS = ("cost_only", "sum_lambda", "sum_lambda_x4", "sum_lambda_x16")


def analyse(rows, figure_dir):
    """Recompute driver terms; compare matched feasible seed IDs per instance.

    These are descriptive diagnostics of the returned plans. Their driver
    assignments and orientations remain those chosen by the campaign.
    """
    selected = [r for r in rows if r['study'] == 'main' and r['arm'] in ARMS]
    if not all(any(r['arm'] == a for r in selected) for a in ARMS):
        return None
    assert all(r.get('routes') is not None for r in selected), 'Driver plot requires retained main routes'
    keys = [(r['instance'], r['arm'], r['seed']) for r in selected]
    assert len(keys) == len(set(keys)), 'Duplicate driver-comparison cell'
    evaluators, records, worst_error = {}, [], 0.0
    for r in selected:
        step = float(r['config'].get('resample_step', 250.0))
        assert step == 250.0 and r['config'].get('metric', 'frechet') == 'frechet'
        i = r['instance']
        if i not in evaluators:
            pair = RSPPPair.load(f'data/rsppbench/{i}.json')
            evaluators[i] = Evaluator(pair)
        ev = evaluators[i]
        dev = [ev.route_deviation(k, route) for k, route in enumerate(r['routes'])]
        total, maximum = sum(dev), max(dev, default=0.0)
        error = abs(total - r['objectives'][2])
        worst_error = max(worst_error, error)
        assert error < 1e-6, (i, r['arm'], r['seed'], error)
        records.append(dict(instance=i, arm=r['arm'], seed=r['seed'], feasible=r['feasible'],
            S_m=total, max_driver_m=maximum,
            largest_share_pct=100 * maximum / total if total > 0 else np.nan,
            n_matched=len(ev.base_curves), driver_deviations_m=json.dumps(dev)))
    df = pd.DataFrame(records)
    comparisons = []
    baseline = df[df.arm == 'cost_only']
    for arm in ARMS[1:]:
        paired = df[df.arm == arm].merge(baseline, on=['instance', 'seed'], suffixes=('', '_base'), validate='one_to_one')
        paired = paired[paired.feasible & paired.feasible_base]
        for i, g in paired.groupby('instance'):
            assert g.S_m_base.median() > 0 and g.max_driver_m_base.median() > 0
            comparisons.append(dict(instance=i, arm=arm, n_seeds=len(g),
                total_reduction_pct=100 * (1 - g.S_m.median() / g.S_m_base.median()),
                worst_driver_reduction_pct=100 * (1 - g.max_driver_m.median() / g.max_driver_m_base.median()),
                largest_share_pct=g.largest_share_pct.median(),
                seeds_worst_driver_worse=int((g.max_driver_m > g.max_driver_m_base + 1e-6).sum())))
    comp = pd.DataFrame(comparisons)
    os.makedirs(figure_dir, exist_ok=True)
    df.to_csv(f'{figure_dir}/driver_terms.csv', index=False)
    comp.to_csv(f'{figure_dir}/driver_comparison.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for arm, marker in zip(ARMS[1:], ['o', 's', '^']):
        g = comp[comp.arm == arm]
        axes[0].scatter(g.total_reduction_pct, g.worst_driver_reduction_pct,
                       s=22, c=ARM_COLOR[arm], marker=marker, label=ARM_LABEL[arm], alpha=.8)
    axes[0].axhline(0, color='black', lw=.7, ls=':')
    axes[0].axvline(0, color='black', lw=.7, ls=':')
    axes[0].set_xlabel('total deviation reduction (%)')
    axes[0].set_ylabel('worst driver deviation reduction (%)')
    axes[0].legend(frameon=False, fontsize=7, loc='upper left')
    rng = np.random.default_rng(20261008)
    for x, arm in enumerate(ARMS):
        vals = df[(df.arm == arm) & df.feasible].groupby('instance').largest_share_pct.median().dropna().values
        bp = axes[1].boxplot([vals], positions=[x], widths=.5, patch_artist=True, showfliers=False,
                            medianprops=dict(color='black'))
        bp['boxes'][0].set_facecolor(ARM_COLOR[arm]); bp['boxes'][0].set_alpha(.5)
        axes[1].scatter(x + rng.uniform(-.12, .12, len(vals)), vals, s=9, color=ARM_COLOR[arm], zorder=3)
    axes[1].set_xticks(range(4), ['Cost-only', '$c=1$', '$c=4$', '$c=16$'])
    axes[1].set_ylabel(r"largest driver's share of $S$ (%)")
    axes[1].set_ylim(0, 103)
    fig.tight_layout()
    save(fig, f'{figure_dir}/fig_driver_burden')
    summary = dict(records=len(df), instances=df.instance.nunique(),
                   seed_counts=sorted(comp.n_seeds.unique().tolist()),
                   objective_recomputation_max_error_m=worst_error,
                   arms={a: dict(median_max_reduction_pct=float(comp[comp.arm == a].worst_driver_reduction_pct.median()),
                       total_improves_max_worsens=int(((comp.arm == a) & (comp.total_reduction_pct > 0)
                                                     & (comp.worst_driver_reduction_pct < 0)).sum()))
                         for a in ARMS[1:]})
    with open(f'{figure_dir}/driver_burden_summary.json', 'w') as f:
        json.dump(summary, f, indent=2); f.write('\n')
    print(f"  verified {len(df)} driver-deviation totals (max error {worst_error:.2e} m)")
    return summary
