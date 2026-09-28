"""Constructive and improvement heuristics.

Two uses:
  * building X_baseline for an RSPP-Bench pair (sweep + local search), which
    must be deterministic given the instance seed so the benchmark is
    reproducible;
  * the ``greedy`` comparison arm -- cheapest-feasible-insertion repair of the
    existing plan, which is what dispatchers actually do by hand and what most
    commercial systems do on a disruption.
"""
from __future__ import annotations

import numpy as np


def _route_cost(route, D):
    if not route:
        return 0.0
    idx = [0] + list(route) + [0]
    return float(D[idx[:-1], idx[1:]].sum())


def _duration(route, inst):
    if not route:
        return 0.0
    return _route_cost(route, inst.dist) / inst.speed + float(inst.service[list(route)].sum())


def _load(route, inst):
    return float(inst.demand[list(route)].sum()) if route else 0.0


def sweep_construct(inst, serve=None):
    """Angular sweep assignment into exactly ``K`` available routes."""
    cust = np.array(serve if serve is not None else inst.customers())
    rel = inst.coords[cust] - inst.coords[0]
    order = cust[np.argsort(np.arctan2(rel[:, 1], rel[:, 0]))]
    avail = [k for k in range(inst.n_vehicles) if inst.available[k]]
    routes = [[] for _ in range(inst.n_vehicles)]
    if not avail:
        return routes
    # fill each available vehicle up to its capacity share before moving on
    share = len(order) / len(avail)
    for i, c in enumerate(order):
        k = avail[min(int(i / share), len(avail) - 1)]
        routes[k].append(int(c))
    return routes


def two_opt(route, D, max_pass=8):
    """First-improvement 2-opt on a single route (depot-anchored)."""
    if len(route) < 4:
        return route
    best = list(route)
    for _ in range(max_pass):
        improved = False
        idx = [0] + best + [0]
        for i in range(1, len(idx) - 2):
            for j in range(i + 1, len(idx) - 1):
                a, b, c, d = idx[i - 1], idx[i], idx[j], idx[j + 1]
                if a == c or b == d:
                    continue
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-9:
                    idx[i:j + 1] = idx[i:j + 1][::-1]
                    best = idx[1:-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def or_opt(routes, inst, max_pass=3):
    """Inter-route relocation of segments of length 1-3, respecting capacity/duration."""
    D = inst.dist
    routes = [list(r) for r in routes]
    for _ in range(max_pass):
        improved = False
        for ka in range(len(routes)):
            if not routes[ka]:
                continue
            for seg_len in (1, 2, 3):
                for i in range(len(routes[ka]) - seg_len + 1):
                    seg = routes[ka][i:i + seg_len]
                    rest = routes[ka][:i] + routes[ka][i + seg_len:]
                    gain_out = _route_cost(routes[ka], D) - _route_cost(rest, D)
                    if gain_out <= 1e-9:
                        continue
                    for kb in range(len(routes)):
                        if kb == ka or not inst.available[kb]:
                            continue
                        if _load(routes[kb], inst) + _load(seg, inst) > inst.capacity:
                            continue
                        base_b = _route_cost(routes[kb], D)
                        best_pos, best_add = None, np.inf
                        for j in range(len(routes[kb]) + 1):
                            cand = routes[kb][:j] + seg + routes[kb][j:]
                            add = _route_cost(cand, D) - base_b
                            if add < best_add:
                                best_add, best_pos = add, j
                        if best_pos is None or best_add >= gain_out - 1e-9:
                            continue
                        cand_b = routes[kb][:best_pos] + seg + routes[kb][best_pos:]
                        if _duration(cand_b, inst) > inst.max_duration:
                            continue
                        routes[ka], routes[kb] = rest, cand_b
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return routes


def planner_objective(routes, inst):
    """Distance plus the distance-equivalent of the workload spread.

    Multiplying the duration spread (seconds) by the vehicle speed (m/s) puts
    both terms in metres, so the two criteria the replanner will later face are
    traded off on a common scale rather than one being sacrificed to the other.
    """
    D = inst.dist
    dist = sum(_route_cost(r, D) for r in routes)
    dur = [_duration(r, inst) for k, r in enumerate(routes)
           if inst.available[k] and r]
    spread = (max(dur) - min(dur)) if len(dur) > 1 else 0.0
    return dist + inst.speed * spread


def balance_descent(routes, inst, max_moves=400):
    """First-improvement relocation descent on the planner objective.

    Evaluated incrementally: a relocation touches exactly two routes, so only
    their length and duration are recomputed, and the spread is a scan over K
    cached durations.  The naive version recomputes every route on every
    candidate move and is unusable beyond about fifty requests.
    """
    D = inst.dist
    routes = [list(r) for r in routes]
    dist = [_route_cost(r, D) for r in routes]
    dur = [_duration(r, inst) for r in routes]

    def spread_of(dvec):
        act = [dvec[k] for k in range(inst.n_vehicles)
               if inst.available[k] and routes[k]]
        return (max(act) - min(act)) if len(act) > 1 else 0.0

    cur = sum(dist) + inst.speed * spread_of(dur)
    for _ in range(max_moves):
        best = None
        for ka in range(inst.n_vehicles):
            if not routes[ka]:
                continue
            for i in range(len(routes[ka])):
                c = int(routes[ka][i])
                rest = routes[ka][:i] + routes[ka][i + 1:]
                d_rest = _route_cost(rest, D)
                t_rest = d_rest / inst.speed + float(inst.service[rest].sum()) \
                    if rest else 0.0
                for kb in range(inst.n_vehicles):
                    if kb == ka or not inst.available[kb]:
                        continue
                    if _load(routes[kb], inst) + inst.demand[c] > inst.capacity:
                        continue
                    for j in range(len(routes[kb]) + 1):
                        cand = routes[kb][:j] + [c] + routes[kb][j:]
                        d_cand = _route_cost(cand, D)
                        t_cand = (d_cand / inst.speed
                                  + float(inst.service[cand].sum()))
                        if t_cand > inst.max_duration:
                            continue
                        nd = list(dur); nd[ka], nd[kb] = t_rest, t_cand
                        old_ka, old_kb = routes[ka], routes[kb]
                        routes[ka], routes[kb] = rest, cand
                        val = (sum(dist) - dist[ka] - dist[kb] + d_rest + d_cand
                               + inst.speed * spread_of(nd))
                        routes[ka], routes[kb] = old_ka, old_kb
                        if val < cur - 1e-6 and (best is None or val < best[0]):
                            best = (val, ka, kb, rest, cand, d_rest, d_cand,
                                    t_rest, t_cand)
        if best is None:
            break
        cur, ka, kb, rest, cand, d_rest, d_cand, t_rest, t_cand = best
        routes[ka], routes[kb] = rest, cand
        dist[ka], dist[kb] = d_rest, d_cand
        dur[ka], dur[kb] = t_rest, t_cand
    return routes


def build_baseline(inst, passes=4):
    """Deterministic baseline plan: a local optimum of the planner objective.

    The plan in force is meant to be a *good* plan -- that is the premise of the
    whole problem -- so it is driven to a local optimum in travel distance and
    workload spread jointly, exactly the two criteria the replanner will later
    face.  A baseline that is merely distance-good would make stability and
    equity artificially adversarial, because any solution staying near it would
    inherit its inequity, and the measured price of stability would really be
    the price of repairing the baseline.
    """
    routes = sweep_construct(inst)
    for _ in range(passes):
        routes = [two_opt(r, inst.dist) for r in routes]
        routes = or_opt(routes, inst)
        routes = balance_descent(routes, inst)
    return [two_opt(r, inst.dist) for r in routes]


def greedy_reinsertion(pair):
    """Repair the baseline plan by cheapest feasible insertion of new work.

    Customers no longer present are dropped; customers on withdrawn vehicles
    and genuinely new customers are inserted at their cheapest feasible
    position.  No global re-optimisation is performed -- this arm represents
    manual dispatcher repair.
    """
    inst = pair.disrupted
    D = inst.dist
    present = set(int(c) for c in inst.customers())
    routes = []
    orphans = []
    for k, route in enumerate(pair.baseline_routes):
        kept = [c for c in route if c in present]
        if not inst.available[k]:
            orphans.extend(kept)
            kept = []
        routes.append(kept)
    placed = set(c for r in routes for c in r)
    orphans.extend(sorted(present - placed - set(orphans)))

    for c in orphans:
        best = (np.inf, None, None)
        for k in range(inst.n_vehicles):
            if not inst.available[k]:
                continue
            if _load(routes[k], inst) + inst.demand[c] > inst.capacity:
                continue
            base = _route_cost(routes[k], D)
            for j in range(len(routes[k]) + 1):
                cand = routes[k][:j] + [int(c)] + routes[k][j:]
                if _duration(cand, inst) > inst.max_duration:
                    continue
                add = _route_cost(cand, D) - base
                if add < best[0]:
                    best = (add, k, j)
        if best[1] is None:  # no feasible slot: force onto the emptiest vehicle
            k = int(np.argmin([_load(r, inst) if inst.available[i] else np.inf
                               for i, r in enumerate(routes)]))
            routes[k].append(int(c))
        else:
            _, k, j = best
            routes[k] = routes[k][:j] + [int(c)] + routes[k][j:]
    return routes
