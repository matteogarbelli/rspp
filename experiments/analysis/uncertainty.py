"""Paired-seed bootstrap uncertainty for the fixed main-study instance suite.

Each bootstrap block is one seed across all instances and all four arms.
This retains matching and any dependence induced by reuse of a seed. The
intervals concern stochastic-run variability conditional on the fixed suite;
they do not quantify variation over new baselines or operational populations.
"""
from pathlib import Path
import json

import numpy as np

ARMS = ["cost_only", "sum_lambda", "sum_lambda_x4", "sum_lambda_x16"]
FACTORS = [1, 4, 16]
BOOTSTRAP_SEED = 20261008
REPLICATES = 10000
BATCH = 125
SEEDS = list(range(30))


def analyse(rows, destination, tex_destination):
    records = [r for r in rows if r["study"] == "main" and r["arm"] in ARMS]
    indexed = {(r["arm"], r["instance"], r["seed"]): r for r in records}
    instances = sorted({r["instance"] for r in records})
    assert len(instances) == 27 and len(indexed) == len(records) == 3240
    values = np.empty((4, 27, 30, 3))
    minimum_distance = np.empty((27, 30))
    for a, arm in enumerate(ARMS):
        for i, instance in enumerate(instances):
            for s, seed in enumerate(SEEDS):
                record = indexed[arm, instance, seed]
                assert record["feasible"]
                values[a, i, s] = record["objectives"]
                if arm == "cost_only":
                    minimum_distance[i, s] = min(p[0] for p in record["front"])

    def statistic(medians, front_minima):
        # Preserve the publication statistic: median of within-instance ratios.
        change_s = 100*(1-medians[1:, ..., 2]/medians[0, ..., 2])
        change_distance = 100*(medians[1:, ..., 0]/front_minima-1)
        change_spread = 100*(medians[1:, ..., 1]/medians[0, ..., 1]-1)
        return np.median(np.stack([change_s, change_distance, change_spread], axis=-1), axis=1)

    assert np.all(values[0, :, :, 1:].mean(axis=1) > 0)
    point = statistic(np.median(values, axis=2), np.median(minimum_distance, axis=1))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for start in range(0, REPLICATES, BATCH):
        ids = rng.integers(0, len(SEEDS), size=(min(BATCH, REPLICATES-start), len(SEEDS)))
        # shape: arms, instances, bootstrap replicates, seed draws, objectives
        medians = np.median(values[:, :, ids, :], axis=3)
        minima = np.median(minimum_distance[:, ids], axis=2)
        samples.append(statistic(medians, minima))
    bootstrap = np.concatenate(samples, axis=1)
    assert bootstrap.shape == (3, REPLICATES, 3) and np.isfinite(bootstrap).all()
    intervals = np.percentile(bootstrap, [2.5, 97.5], axis=1).transpose(1, 2, 0)

    destination, tex_destination = Path(destination), Path(tex_destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary = dict(phase="post-processing of completed production; no new search",
                   bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=REPLICATES,
                   confidence_level=.95, interval_method="percentile paired-seed block bootstrap",
                   target="stochastic-run variability conditional on the fixed 27-instance suite",
                   block="one matched seed across all instances and arms", seeds=SEEDS,
                   records=len(records), instances=instances,
                   criteria=["deviation reduction pct", "distance increase over cost-only front minimum pct",
                             "workload spread increase pct"],
                   rows=[dict(c=c, estimates=point[j].tolist(), intervals=intervals[j].tolist())
                         for j, c in enumerate(FACTORS)])
    (destination/"uncertainty_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez_compressed(destination/"uncertainty_bootstrap.npz", statistics=bootstrap,
                        estimates=point, intervals=intervals)
    lines = [r"\begin{tabular}{rrrr}", r"\toprule",
             r"$c$ & $\Delta S$ & $\Delta f_1^{\min}$ & $\Delta f_2$ \\", r"\midrule"]
    for j, c in enumerate(FACTORS):
        entries = [f"{point[j, k]:.1f} [{intervals[j, k, 0]:.1f}, {intervals[j, k, 1]:.1f}]"
                   for k in range(3)]
        lines.append(str(c)+" & "+" & ".join(entries)+r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (tex_destination/"tab_uncertainty.tex").write_text("\n".join(lines)+"\n")
    return summary
