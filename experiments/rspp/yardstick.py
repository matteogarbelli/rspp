"""Distance yardstick: a short plan for the disrupted instance, found by OR-Tools.

This is not a comparison arm.  It locates the distance end of the trade-off,
against which the distance premium of every method can be read, and it ignores
deviation and workload spread.

Settings, all fixed:
- guided local search, stopped after a fixed number of solutions, so the
  result is deterministic for a given OR-Tools version (the time limit is a
  safety cap only, and a run that reaches it is flagged);
- warm start from the greedy repair of the plan in force when that repair is
  hard-feasible, otherwise the cheapest-arc construction;
- capacity, duration and availability as hard constraints, and every available
  vehicle serving at least one request (fixed crews);
- integer data: distances in decimetres, arc durations rounded up to
  deciseconds, and demands doubled, since surged demands are multiples of one
  half.

Requires the ``ortools`` package, which the rest of the code does not use.
"""
from __future__ import annotations

import time

import numpy as np

from .algorithm import relabel
from .heuristics import greedy_reinsertion
from .solution import Evaluator

SCALE = 10.0            # decimetres and deciseconds
SOLUTION_LIMIT = 500
TIME_CAP_S = 600


def solve(pair, solution_limit=SOLUTION_LIMIT, time_cap_s=TIME_CAP_S):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    import ortools

    inst = pair.disrupted
    n, K, D = inst.n_customers, inst.n_vehicles, inst.dist
    man = pywrapcp.RoutingIndexManager(n + 1, K, 0)
    rt = pywrapcp.RoutingModel(man)

    def dist_cb(i, j):
        return int(round(SCALE * D[man.IndexToNode(i), man.IndexToNode(j)]))
    rt.SetArcCostEvaluatorOfAllVehicles(rt.RegisterTransitCallback(dist_cb))

    dem = [int(round(2.0 * x)) for x in inst.demand]
    if any(abs(d - 2.0 * x) > 1e-9 for d, x in zip(dem, inst.demand)):
        raise ValueError("demand is not a multiple of one half")
    qcb = rt.RegisterUnaryTransitCallback(lambda i: dem[man.IndexToNode(i)])
    rt.AddDimensionWithVehicleCapacity(qcb, 0, [int(np.floor(2.0 * inst.capacity))] * K,
                                       True, "load")

    def time_cb(i, j):
        a, b = man.IndexToNode(i), man.IndexToNode(j)
        return int(np.ceil(SCALE * (D[a, b] / inst.speed + inst.service[a])))
    rt.AddDimension(rt.RegisterTransitCallback(time_cb), 0,
                    int(np.floor(SCALE * inst.max_duration)), True, "time")

    avail = [k for k in range(K) if inst.available[k]]
    for node in range(1, n + 1):
        rt.SetAllowedVehiclesForIndex(avail, man.NodeToIndex(node))
    ccb = rt.RegisterUnaryTransitCallback(lambda i: 0 if man.IndexToNode(i) == 0 else 1)
    rt.AddDimension(ccb, 0, n, True, "count")
    count = rt.GetDimensionOrDie("count")
    for k in avail:
        count.CumulVar(rt.End(k)).SetMin(1)

    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    prm.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    prm.solution_limit = solution_limit
    prm.time_limit.seconds = time_cap_s
    rt.CloseModelWithParameters(prm)

    ev = Evaluator(pair)
    repair = greedy_reinsertion(pair)
    warm = None
    if ev.evaluate(repair)[3] <= 1e-12:
        warm = rt.ReadAssignmentFromRoutes([list(r) for r in repair], True)

    t0 = time.perf_counter()
    sol = (rt.SolveFromAssignmentWithParameters(warm, prm) if warm is not None
           else rt.SolveWithParameters(prm))
    runtime = time.perf_counter() - t0
    if sol is None:
        return dict(feasible=False, warm_start=warm is not None, runtime=runtime,
                    ortools=ortools.__version__, solution_limit=solution_limit)

    routes = []
    for k in range(K):
        idx, r = sol.Value(rt.NextVar(rt.Start(k))), []
        while not rt.IsEnd(idx):
            r.append(man.IndexToNode(idx))
            idx = sol.Value(rt.NextVar(idx))
        routes.append(r)

    f1, f2, f3, viol, info = ev.evaluate(routes)
    rl = relabel(routes, ev, inst)
    f3_rl = float(np.sum([ev.route_deviation(k, r) for k, r in enumerate(rl)]))
    n_solutions = rt.solver().Solutions()
    return dict(
        feasible=bool(viol <= 1e-12), violation=float(viol),
        objectives=[float(f1), float(f2), float(f3_rl)], objectives_raw=[float(f1), float(f2), float(f3)],
        routes=rl, n_active=int(info["n_active"]), warm_start=warm is not None,
        n_solutions=int(n_solutions), hit_time_cap=bool(n_solutions < solution_limit),
        runtime=runtime, ortools=ortools.__version__, solution_limit=solution_limit,
        time_cap_s=time_cap_s)
