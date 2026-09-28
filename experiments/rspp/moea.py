"""Multi-objective machinery: constrained dominance, NSGA-III, NSGA-II, MOEA/D.

Everything here is written out explicitly rather than delegated to a library
because two of the paper's claims are claims *about the selection operator*:
that a generation-wide positive multiplier on an objective cannot steer it
(Proposition 1), and that a corridor constraint can (Proposition 2).  Those are
only checkable if the operator is inspectable and instrumentable.
"""
from __future__ import annotations

import itertools

import numpy as np


# ------------------------------------------------------------ reference points
def uniform_reference_points(n_obj, p):
    """Das-Dennis simplex-lattice points: all non-negative integer vectors
    summing to p, scaled to the unit simplex."""
    pts = []
    for c in itertools.combinations(range(p + n_obj - 1), n_obj - 1):
        prev, row = -1, []
        for x in c:
            row.append(x - prev - 1)
            prev = x
        row.append(p + n_obj - 2 - prev)
        pts.append(row)
    return np.array(pts, dtype=float) / p


# -------------------------------------------------------- constrained dominance
def dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))


def constrained_fronts(F, V, tol=1e-12):
    """Non-dominated sorting under the Constraint Dominance Principle (Deb 2002).

    x constraint-dominates y iff
      (i)   both feasible and x Pareto-dominates y; or
      (ii)  x feasible and y infeasible; or
      (iii) both infeasible and V(x) < V(y).

    Feasible individuals are sorted into Pareto fronts; infeasible ones follow,
    ordered by violation magnitude.  This partition is *not* invariant under a
    positive rescaling of any objective, which is exactly why the corridor can
    steer selection where a scalar weight cannot.
    """
    F = np.asarray(F, dtype=float)
    V = np.asarray(V, dtype=float)
    feas = np.flatnonzero(V <= tol)
    infeas = np.flatnonzero(V > tol)

    fronts = []
    if len(feas):
        sub = F[feas]
        # full domination matrix in one vectorised pass: dom[i, j] <=> i dominates j
        le = (sub[:, None, :] <= sub[None, :, :]).all(-1)
        lt = (sub[:, None, :] < sub[None, :, :]).any(-1)
        dom = le & lt
        dom_count = dom.sum(axis=0)
        remaining = np.ones(len(sub), dtype=bool)
        while remaining.any():
            cur = np.flatnonzero(remaining & (dom_count == 0))
            if len(cur) == 0:            # numerical safety net; should not trigger
                cur = np.flatnonzero(remaining)
            fronts.append([int(feas[i]) for i in cur])
            remaining[cur] = False
            dom_count = dom_count - dom[cur].sum(axis=0)
            dom_count[cur] = 1           # keep settled rows out of the next pick

    if len(infeas):
        order = infeas[np.argsort(V[infeas], kind="stable")]
        group, last = [], None
        for i in order:
            if last is not None and V[i] > last + tol:
                fronts.append(group)
                group = []
            group.append(int(i))
            last = V[i]
        if group:
            fronts.append(group)
    return fronts


# ------------------------------------------------------------------- NSGA-III
class Nsga3State:
    """Carries the running ideal point across generations, as NSGA-III requires."""

    def __init__(self, n_obj):
        self.ideal = np.full(n_obj, np.inf)

    def update(self, F):
        self.ideal = np.minimum(self.ideal, np.asarray(F, float).min(axis=0))


def _normalize(F, state):
    """Translate by the ideal point and scale by the hyperplane intercepts."""
    F = np.asarray(F, dtype=float)
    state.update(F)
    T = F - state.ideal
    n_obj = F.shape[1]
    # extreme points via the achievement scalarizing function
    w = np.eye(n_obj) + 1e-6
    ext = np.empty((n_obj, n_obj))
    for i in range(n_obj):
        asf = (T / w[i]).max(axis=1)
        ext[i] = T[int(np.argmin(asf))]
    try:
        b = np.linalg.solve(ext, np.ones(n_obj))
        with np.errstate(divide="ignore", invalid="ignore"):
            intercepts = 1.0 / b
        if not np.all(np.isfinite(intercepts)) or np.any(intercepts <= 1e-9):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        intercepts = T.max(axis=0)
    intercepts = np.where(intercepts <= 1e-9, 1e-9, intercepts)
    return T / intercepts


def _associate(Fn, ref):
    """Perpendicular-distance association of points to reference directions."""
    norm = np.linalg.norm(ref, axis=1, keepdims=True)
    unit = ref / np.where(norm == 0, 1.0, norm)
    proj = Fn @ unit.T                      # (n_pts, n_ref)
    d2 = (Fn ** 2).sum(axis=1)[:, None] - proj ** 2
    d = np.sqrt(np.maximum(d2, 0.0))
    niche = np.argmin(d, axis=1)
    return niche, d[np.arange(len(Fn)), niche]


def nsga3_survivors(F, V, n_survive, ref, state, rng):
    """NSGA-III environmental selection under constrained dominance."""
    fronts = constrained_fronts(F, V)
    chosen, last = [], None
    for fr in fronts:
        if len(chosen) + len(fr) <= n_survive:
            chosen.extend(fr)
        else:
            last = fr
            break
    if last is None or len(chosen) == n_survive:
        return chosen[:n_survive]

    k = n_survive - len(chosen)
    pool = chosen + last
    Fn = _normalize(np.asarray(F)[pool], state)
    niche, dist = _associate(Fn, ref)
    n_chosen = len(chosen)
    counts = np.zeros(len(ref), dtype=int)
    for i in range(n_chosen):
        counts[niche[i]] += 1

    cand = list(range(n_chosen, len(pool)))
    picked = []
    while len(picked) < k:
        avail_niches = set(niche[i] for i in cand)
        if not avail_niches:
            break
        min_count = min(counts[j] for j in avail_niches)
        tied = [j for j in avail_niches if counts[j] == min_count]
        j = int(tied[rng.integers(len(tied))])
        members = [i for i in cand if niche[i] == j]
        if counts[j] == 0:
            pick = min(members, key=lambda i: dist[i])
        else:
            pick = members[int(rng.integers(len(members)))]
        picked.append(pick)
        cand.remove(pick)
        counts[j] += 1
    return chosen + [pool[i] for i in picked]


# -------------------------------------------------------------------- NSGA-II
def _crowding(F):
    F = np.asarray(F, float)
    n, m = F.shape
    cd = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        cd[order[0]] = cd[order[-1]] = np.inf
        span = F[order[-1], j] - F[order[0], j]
        if span <= 0:
            continue
        cd[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return cd


def nsga2_survivors(F, V, n_survive, rng):
    fronts = constrained_fronts(F, V)
    chosen = []
    for fr in fronts:
        if len(chosen) + len(fr) <= n_survive:
            chosen.extend(fr)
            continue
        k = n_survive - len(chosen)
        cd = _crowding(np.asarray(F)[fr])
        order = np.argsort(-cd, kind="stable")
        chosen.extend([fr[i] for i in order[:k]])
        break
    return chosen[:n_survive]


# -------------------------------------------------------------------- MOEA/D
def tchebycheff(f, w, ideal):
    w = np.where(w <= 1e-9, 1e-9, w)
    return float(np.max(w * np.abs(np.asarray(f, float) - ideal)))


def moead_neighbourhoods(weights, t_size):
    d = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=-1)
    return np.argsort(d, axis=1)[:, :t_size]


# ------------------------------------------------------------------ indicators
def hypervolume(front, ref_point):
    """Exact hypervolume for 2 or 3 objectives (minimisation) by dimension sweep."""
    P = np.asarray(front, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    P = P[np.all(P <= ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.lexsort(tuple(P[:, i] for i in range(P.shape[1] - 1, -1, -1)))]
    if P.shape[1] == 2:
        hv, prev_y = 0.0, ref[1]
        for x, y in P:
            if y < prev_y:
                hv += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return float(hv)
    if P.shape[1] == 3:
        hv = 0.0
        zs = np.unique(P[:, 2])
        zs = np.append(zs, ref[2])
        for a in range(len(zs) - 1):
            slab = P[P[:, 2] <= zs[a]][:, :2]
            if len(slab) == 0:
                continue
            hv += hypervolume(slab, ref[:2]) * (zs[a + 1] - zs[a])
        return float(hv)
    raise ValueError("hypervolume implemented for 2 or 3 objectives")


def igd_plus(front, reference):
    """IGD+ (Ishibuchi et al. 2015): Pareto-compliant inverted generational distance."""
    A = np.asarray(front, float)
    Z = np.asarray(reference, float)
    if len(A) == 0:
        return float("inf")
    d = np.sqrt((np.maximum(A[None, :, :] - Z[:, None, :], 0.0) ** 2).sum(-1))
    return float(d.min(axis=1).mean())


def epsilon_indicator(front, reference):
    """Additive unary epsilon indicator against a reference front."""
    A = np.asarray(front, float)
    Z = np.asarray(reference, float)
    if len(A) == 0:
        return float("inf")
    return float(np.max(np.min(np.max(A[None, :, :] - Z[:, None, :], axis=2), axis=1)))


def spacing(front):
    A = np.asarray(front, float)
    if len(A) < 2:
        return 0.0
    d = np.abs(A[:, None, :] - A[None, :, :]).sum(-1)
    np.fill_diagonal(d, np.inf)
    dmin = d.min(axis=1)
    return float(np.sqrt(((dmin.mean() - dmin) ** 2).sum() / (len(A) - 1)))


def nondominated(F):
    """Indices of the non-dominated subset, in O(n * |front|) time and O(n) memory.

    Points are visited in ascending order of objective sum. Dominance implies a
    strictly smaller sum, so every point that could dominate the current one has
    already been seen, and a single pass against the running front is exact.
    The pairwise-matrix formulation is quadratic in *memory* and becomes
    unusable on the pooled fronts used to build a reference set: 22,000 points
    would need a 1.5 GB boolean array per comparison.
    """
    F = np.asarray(F, dtype=float)
    n = len(F)
    if n == 0:
        return np.array([], dtype=int)
    if n <= 512:                      # small case: the vectorised form is faster
        le = (F[:, None, :] <= F[None, :, :]).all(-1)
        lt = (F[:, None, :] < F[None, :, :]).any(-1)
        return np.flatnonzero(~(le & lt).any(axis=0))

    order = np.argsort(F.sum(axis=1), kind="stable")
    keep, front = [], np.empty((0, F.shape[1]), dtype=float)
    for i in order:
        p = F[i]
        if len(front):
            if np.any(np.all(front <= p, axis=1) & np.any(front < p, axis=1)):
                continue
        keep.append(int(i))
        front = np.vstack([front, p])
    return np.array(sorted(keep), dtype=int)


def front_index(F, V):
    """Front number of every individual under constrained dominance.

    Deterministic: no RNG enters constrained non-dominated sorting.  This is
    the quantity Propositions 1 and 2 are about -- a mechanism that cannot move
    an individual between fronts cannot steer the search through selection
    pressure, whatever it does to the numerical objective values.
    """
    rank = np.empty(len(F), dtype=int)
    for r, fr in enumerate(constrained_fronts(F, V)):
        for i in fr:
            rank[i] = r
    return rank
