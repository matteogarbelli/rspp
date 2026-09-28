"""Front quality indicators and effect-size / significance machinery."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..rspp.moea import epsilon_indicator, hypervolume, igd_plus, nondominated, spacing

__all__ = ["normalise_bounds", "front_indicators", "vargha_delaney",
           "friedman_nemenyi", "friedman_control", "wilcoxon_pair", "holm",
           "per_instance_comparison"]


def normalise_bounds(all_fronts):
    """Componentwise min/max over every point of every front on one instance."""
    pts = np.vstack([np.asarray(f, float) for f in all_fronts if len(f)])
    return pts.min(axis=0), pts.max(axis=0)


def _norm(front, lo, hi):
    span = np.where(hi - lo <= 0, 1.0, hi - lo)
    return (np.asarray(front, float) - lo) / span


def reference_front(all_fronts, lo, hi):
    pts = np.vstack([_norm(f, lo, hi) for f in all_fronts if len(f)])
    pts = np.unique(np.round(pts, 9), axis=0)
    return pts[nondominated(pts)]


def front_indicators(front, lo, hi, ref_front, ref_point=1.1):
    """Hypervolume, IGD+, additive epsilon and spacing on normalised objectives.

    Duplicate objective vectors are removed first.  A population that has
    converged onto a handful of distinct solutions reports a nominally large
    non-dominated set -- duplicates never dominate one another -- so counting
    them would overstate the diversity of the returned trade-off surface, and
    would make the spacing indicator identically zero.
    """
    if front is None or len(front) == 0:
        return dict(hv=0.0, igdp=float("inf"), eps=float("inf"), spacing=0.0,
                    n_front=0, n_raw=0)
    A = _norm(front, lo, hi)
    A = A[nondominated(A)]
    n_raw = len(A)
    A = np.unique(np.round(A, 9), axis=0)
    rp = np.full(A.shape[1], ref_point)
    return dict(hv=hypervolume(A, rp), igdp=igd_plus(A, ref_front),
                eps=epsilon_indicator(A, ref_front), spacing=spacing(A),
                n_front=len(A), n_raw=n_raw)


def vargha_delaney(a, b):
    """A_12: probability that a random draw from `a` exceeds one from `b`.

    0.5 means no effect. Reported for the *pair as ordered*; callers decide
    which direction counts as better.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    r = stats.rankdata(np.concatenate([a, b]))
    r1 = r[:n1].sum()
    return float((r1 / n1 - (n1 + 1) / 2) / n2)


def magnitude(a12):
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def wilcoxon_pair(a, b):
    """Seed-paired Wilcoxon signed-rank test plus the effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return dict(p=1.0, a12=0.5, effect="negligible")
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    a12 = vargha_delaney(a, b)
    return dict(p=p, a12=a12, effect=magnitude(a12))


def holm(pvalues):
    """Holm step-down adjustment; returns adjusted p-values in input order."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


# Nemenyi critical values q_alpha for alpha = 0.05, indexed by number of methods
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


def friedman_control(matrix, control, higher_is_better=False):
    """Holm-adjusted 1xN post-hoc against a control method.

    Nemenyi's critical difference is designed for all-pairs comparison and is
    conservative to the point of uselessness when the question is "how does the
    proposed method compare against each baseline".  For that question the
    standard procedure is a control comparison on the average ranks
    (Demsar 2006; Garcia & Herrera 2008): the rank difference is normalised by
    sqrt(k(k+1)/(6N)) and the resulting normal p-values are Holm-adjusted across
    the k-1 comparisons.

    Returns (avg_ranks, friedman_p, {method_index: adjusted_p}).
    """
    M = np.asarray(matrix, float)
    n, k = M.shape
    scores = -M if higher_is_better else M
    ranks = np.apply_along_axis(stats.rankdata, 1, scores)
    avg = ranks.mean(axis=0)
    try:
        fp = float(stats.friedmanchisquare(*[M[:, j] for j in range(k)]).pvalue)
    except ValueError:
        fp = 1.0
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    others = [j for j in range(k) if j != control]
    raw = [2.0 * stats.norm.sf(abs(avg[j] - avg[control]) / se) for j in others]
    adj = holm(raw)
    return avg, fp, {j: float(p) for j, p in zip(others, adj)}


def per_instance_comparison(pairs):
    """Seed-paired comparison run *within* each instance, then aggregated.

    ``pairs`` is an iterable of (a, b) arrays, one per instance, already
    seed-aligned and filtered to feasible runs.

    Pooling raw objective values across instances of different size is not a
    valid alternative: the rank statistic is then dominated by the difference
    between instances rather than between methods, which drives the effect size
    towards 0.5 no matter how consistently one method wins.  The comparison
    therefore happens inside each instance, and only the summaries are pooled.

    Returns dict with the median A_12 across instances, the win/tie/loss counts
    against the control at alpha = 0.05 with Holm correction across instances,
    and the number of instances compared.
    """
    a12s, ps, signs = [], [], []
    for a, b in pairs:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) < 5 or len(a) != len(b):
            continue
        w = wilcoxon_pair(a, b)
        a12s.append(w["a12"])
        ps.append(w["p"])
        signs.append(np.sign(np.median(a) - np.median(b)))
    if not a12s:
        return dict(a12=float("nan"), effect="---", win=0, tie=0, loss=0,
                    n_cmp=0, p_min=float("nan"))
    adj = holm(ps)
    win = sum(1 for p, s in zip(adj, signs) if p < 0.05 and s > 0)
    loss = sum(1 for p, s in zip(adj, signs) if p < 0.05 and s < 0)
    med = float(np.median(a12s))
    return dict(a12=med, effect=magnitude(med), win=win, loss=loss,
                tie=len(a12s) - win - loss, n_cmp=len(a12s),
                p_min=float(np.min(adj)))


def friedman_nemenyi(matrix, higher_is_better=False):
    """Friedman test over a (n_datasets x n_methods) matrix of scores.

    Returns average ranks (rank 1 = best), the Friedman p-value, and the
    Nemenyi critical difference at alpha = 0.05.
    """
    M = np.asarray(matrix, float)
    n, k = M.shape
    scores = -M if higher_is_better else M
    ranks = np.apply_along_axis(stats.rankdata, 1, scores)
    avg = ranks.mean(axis=0)
    try:
        p = float(stats.friedmanchisquare(*[M[:, j] for j in range(k)]).pvalue)
    except ValueError:
        p = 1.0
    cd = _Q05.get(k, 3.3) * np.sqrt(k * (k + 1) / (6.0 * n))
    return avg, p, float(cd)
