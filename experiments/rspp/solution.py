"""Solution encoding, decoding, and evaluation for the RSPP.

Encoding.  A solution is a pair ``(perm, breaks)``: ``perm`` is a permutation of
the customers to be served and ``breaks`` is a sorted vector of K-1 cut
positions, so route k is ``perm[breaks[k-1]:breaks[k]]``.  Routes are indexed by
*vehicle*, not by position, because route matching against the baseline is by
driver identity -- the driver keeps their own route, which is the whole point of
the problem.

Objectives (all minimised, all in natural units):
    f1  total travel distance                      metres
    f2  workload imbalance, max_k T_k - min_k T_k  seconds
    f3  cumulative trajectory deviation S(X)       metres

Constraints, aggregated into a single normalised violation for the Constraint
Dominance Principle:
    capacity      sum_k max(0, load_k - Q) / Q
    duration      sum_k max(0, T_k - T_max) / T_max
    corridor      max(0, S(X) - tau(t)) / max(tau(t), 1)
"""
from __future__ import annotations

import numpy as np

from . import kernels
from .geometry import resample, route_polyline


def decode(perm, breaks, n_vehicles):
    """Split a permutation at ``breaks`` into ``n_vehicles`` routes."""
    cuts = [0] + list(breaks) + [len(perm)]
    return [list(perm[cuts[k]:cuts[k + 1]]) for k in range(n_vehicles)]


def random_solution(inst, rng, serve=None):
    """Uniformly random permutation with random balanced cut points."""
    cust = np.array(serve if serve is not None else inst.customers())
    perm = rng.permutation(cust)
    k_act = int(inst.available.sum())
    breaks = _balanced_breaks(len(perm), k_act, inst.n_vehicles, inst.available, rng)
    return list(map(int, perm)), breaks


def _balanced_breaks(n, k_act, n_vehicles, available, rng):
    """Cut positions giving work only to available vehicles."""
    if k_act <= 0:
        return [0] * (n_vehicles - 1)
    pts = sorted(rng.choice(np.arange(1, max(2, n)), size=max(0, k_act - 1),
                            replace=False).tolist()) if n > k_act > 1 else \
        [int(round(i * n / k_act)) for i in range(1, k_act)]
    sizes = np.diff([0] + list(pts) + [n])
    breaks, s = [], 0
    it = iter(sizes)
    for k in range(n_vehicles - 1):
        if available[k]:
            s += int(next(it, 0))
        breaks.append(s)
    return breaks


class Evaluator:
    """Evaluates solutions against a fixed RSPP pair.

    Per-vehicle memoisation of trajectory deviation is the single most
    important optimisation here: S is separable across vehicles, so a route
    that survives a generation unchanged costs nothing to re-score.
    """

    def __init__(self, pair, metric="frechet", resample_step=250.0,
                 cache_size=60_000):
        self.pair = pair
        self.inst = pair.disrupted
        self.metric_name = metric
        self.metric = kernels.METRICS[metric]
        self.resample_step = resample_step
        self._cache = {}
        self._cache_size = cache_size
        self.n_evals = 0

        # The matching set: vehicles that had a route before the disruption and
        # are still available after it.  A withdrawn vehicle is excluded --
        # there is no "path ahead" left for that driver to preserve, and
        # charging its abandonment to every candidate solution would add a
        # constant to S that inflates the denominator of every reported
        # reduction without changing any ranking.
        base = pair.base
        self.base_curves = {}
        for k, route in enumerate(pair.baseline_routes):
            if len(route) == 0 or not self.inst.available[k]:
                continue
            curve = route_polyline(route, base.coords)
            self.base_curves[k] = resample(curve, resample_step)

    # -------------------------------------------------------------- pieces
    def route_distance(self, route):
        if not route:
            return 0.0
        D = self.inst.dist
        idx = [0] + list(route) + [0]
        return float(D[idx[:-1], idx[1:]].sum())

    def route_duration(self, route):
        if not route:
            return 0.0
        return (self.route_distance(route) / self.inst.speed
                + float(self.inst.service[list(route)].sum()))

    def route_load(self, route):
        if not route:
            return 0.0
        return float(self.inst.demand[list(route)].sum())

    def route_deviation(self, k, route):
        """Trajectory deviation of vehicle k's route from its baseline route.

        Vehicles with no baseline route (newly commissioned) and vehicles that
        were withdrawn contribute nothing: there is no pair to compare.  A
        vehicle that had a baseline route but is now empty is charged the full
        deviation of abandoning it, measured against the depot-only curve.
        """
        if k not in self.base_curves:
            return 0.0
        key = (k, tuple(route))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if not route:
            cur = np.repeat(self.inst.coords[:1], 2, axis=0)
        else:
            cur = resample(route_polyline(route, self.inst.coords),
                           self.resample_step)
        val = float(self.metric(cur, self.base_curves[k]))
        if len(self._cache) < self._cache_size:
            self._cache[key] = val
        return val

    # ---------------------------------------------------------- evaluation
    def evaluate(self, routes, tau=None):
        """Return ``(f1, f2, f3, violation, info)`` for a decoded solution."""
        self.n_evals += 1
        inst = self.inst
        dists, durs, loads, devs = [], [], [], []
        for k, route in enumerate(routes):
            dists.append(self.route_distance(route))
            durs.append(self.route_duration(route))
            loads.append(self.route_load(route))
            devs.append(self.route_deviation(k, route))

        f1 = float(np.sum(dists))
        # Workload spread is taken over vehicles that are available *and*
        # carrying work.  An available-but-empty vehicle is excluded rather than
        # entered as a zero-duration minimum, because a crew member with no work
        # is an allocation decision and not a zero-length shift.  The guard has
        # a degenerate corner -- with fewer than two working vehicles the spread
        # is nil by definition -- which the duration cap makes unreachable at
        # every size in the suite, and which is monitored by ``n_active``.
        active = [d for k, d in enumerate(durs)
                  if inst.available[k] and len(routes[k]) > 0]
        f2 = float(max(active) - min(active)) if len(active) > 1 else 0.0
        f3 = float(np.sum(devs))

        cap = sum(max(0.0, L - inst.capacity) for L in loads) / inst.capacity
        dur = sum(max(0.0, T - inst.max_duration) for T in durs) / inst.max_duration
        # work assigned to a withdrawn vehicle is a hard infeasibility
        unavail = sum(len(routes[k]) for k in range(inst.n_vehicles)
                      if not inst.available[k])
        viol = cap + dur + float(unavail)

        corridor = 0.0
        if tau is not None:
            corridor = max(0.0, f3 - tau) / max(tau, 1.0)
            viol += corridor

        return f1, f2, f3, viol, {"cap": cap, "dur": dur, "corridor": corridor,
                                  "unavail": unavail, "n_active": len(active)}
