"""A route and its optimal discrete Frechet coupling from the reported archive.

This illustrates the metric; it is not an additional optimisation experiment.
The full dynamic programme is checked against the production two-row kernel.
"""
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from ..rspp import kernels
from ..rspp.geometry import resample, route_polyline
from ..rspp.instance import RSPPPair
from ..rspp.solution import Evaluator
from .style import apply, save, BLUE, VERMILLION

INSTANCE = "C100-7v-INS8-s1"
ARM = "cost_only"
SEED = 0
STEP_M = 250.0
ATOL_M = 1e-9


def optimal_coupling(p, q):
    distances = np.linalg.norm(p[:, None] - q[None, :], axis=2)
    costs = np.empty_like(distances)
    predecessor = np.zeros(distances.shape + (2,), dtype=np.int32)
    for i in range(len(p)):
        for j in range(len(q)):
            if i == j == 0:
                costs[i, j] = distances[i, j]
                continue
            candidates = [(i-1, j-1), (i-1, j), (i, j-1)]
            candidates = [(a, b) for a, b in candidates if a >= 0 and b >= 0]
            a, b = min(candidates, key=lambda x: costs[x])
            costs[i, j] = max(distances[i, j], costs[a, b])
            predecessor[i, j] = (a, b)
    path = [(len(p)-1, len(q)-1)]
    while path[-1] != (0, 0):
        path.append(tuple(predecessor[path[-1]]))
    path = np.array(path[::-1])
    kernel_distance = float(kernels.frechet(p, q))
    path_distance = float(distances[path[:, 0], path[:, 1]].max())
    error = max(abs(float(costs[-1, -1])-kernel_distance),
                abs(path_distance-kernel_distance))
    assert error <= ATOL_M, error
    increments = np.diff(path, axis=0)
    assert all(tuple(step) in {(1, 0), (0, 1), (1, 1)} for step in increments)
    return distances, path, kernel_distance, error


def analyse(rows, destination):
    record = next(r for r in rows if r["study"] == "main"
                  and r["instance"] == INSTANCE and r["arm"] == ARM
                  and r["seed"] == SEED)
    assert record["feasible"]
    pair = RSPPPair.load(f"data/rsppbench/{INSTANCE}.json")
    evaluator = Evaluator(pair, resample_step=STEP_M)
    driver = max(evaluator.base_curves,
                 key=lambda k: evaluator.route_deviation(k, record["routes"][k]))
    p = evaluator.base_curves[driver]
    q = resample(route_polyline(record["routes"][driver], pair.disrupted.coords), STEP_M)
    distances, path, delta, error = optimal_coupling(p, q)
    bottleneck = path[np.argmax(distances[path[:, 0], path[:, 1]])]

    apply()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    axes[0].plot(p[:, 0]/1000, p[:, 1]/1000, "--", color=BLUE, label="baseline")
    axes[0].plot(q[:, 0]/1000, q[:, 1]/1000, color=VERMILLION, label="revised")
    segment = np.array([p[bottleneck[0]], q[bottleneck[1]]])/1000
    axes[0].plot(segment[:, 0], segment[:, 1], ":", color="black", lw=1,
                 label="bottleneck pair")
    depot = pair.disrupted.coords[0]/1000
    axes[0].scatter([depot[0]], [depot[1]], marker="s", color="black", s=18,
                    label="depot")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("east (km)")
    axes[0].set_ylabel("north (km)")
    axes[0].legend(frameon=False, fontsize=7)
    im = axes[1].imshow(distances/1000, origin="lower", aspect="auto", cmap="viridis",
                         extent=[-.5, len(q)-.5, -.5, len(p)-.5])
    axes[1].plot(path[:, 1], path[:, 0], color="white", lw=1.5)
    axes[1].set_xlabel("revised sample index")
    axes[1].set_ylabel("baseline sample index")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], label="pairwise distance (km)", fraction=.05, pad=.04)
    fig.tight_layout()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    save(fig, str(destination/"fig_frechet_coupling"))
    np.savez_compressed(destination/"frechet_coupling.npz", baseline=p, revised=q,
                        distances_m=distances, coupling=path, bottleneck=bottleneck)
    summary = dict(instance=INSTANCE, arm=ARM, seed=SEED, driver=int(driver),
                   driver_selection="largest deviation in this returned plan",
                   subdivision_step_m=STEP_M, baseline_samples=len(p), revised_samples=len(q),
                   deviation_m=delta, maximum_absolute_error_m=error,
                   absolute_tolerance_m=ATOL_M, monotone_coupling=True)
    (destination/"frechet_coupling_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary
