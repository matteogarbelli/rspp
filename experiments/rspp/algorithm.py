"""The similarity-preserving evolutionary algorithm and its comparison arms.

Arms
----
greedy           cheapest-feasible-insertion repair of the baseline (no search)
cost_only        NSGA-III on (f1, f2); similarity measured but not optimised
sim_obj          NSGA-III on (f1, f2, f3); similarity as a plain objective
sim_lambda       NSGA-III on (f1, f2, lambda(t)*f3), population-adaptive weight (Eq. 9)
eps_static       NSGA-III on (f1, f2, f3) + a *static* epsilon-constraint on
                 deviation, the bound fixed at the deviation of the greedy
                 repair of the plan in force.  Exogenous to the population and
                 to the schedule, so it isolates "similarity moved from the
                 objective side to the constraint side" from the two things the
                 corridor adds on top: population calibration and tightening.
corridor         NSGA-III on (f1, f2, f3) + adaptive similarity corridor (tightening bound)
corridor_fixed   as above with a constant scheduled width (constant bound)
corridor_nsga2   the tightening bound carried by NSGA-II
corridor_moead   the tightening bound carried by MOEA/D (Tchebycheff)
sum_lambda       NSGA-III on (f1 + lambda(t)*f3, f2): deviation summed into distance
                 with the population-adaptive weight of Eq. (9)  [proposed]
sum_lambda_nsga2 the summed objectives carried by NSGA-II
sum_lambda_moead the summed objectives carried by MOEA/D (normalised Tchebycheff)
sum_lambda_x4    the summed objectives with the weight scaled to 4 * lambda(t)
sum_lambda_x16   the summed objectives with the weight scaled to 16 * lambda(t)
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import moea
from .heuristics import greedy_reinsertion, two_opt
from .solution import Evaluator, decode, random_solution

# --------------------------------------------------------------------- config
DEFAULTS = dict(
    pop_size=92,
    max_gen=300,
    p_cx=0.9,
    p_mut=0.25,
    p_ls=0.10,
    ref_p=12,             # Das-Dennis divisions -> H = 91 for 3 objectives
    tau_q0=0.90,          # quantile of the initial S distribution -> tau_0
    tau_qmin=0.10,        # non-vacuity floor: >=10% of the pool stays inside (Prop. 3)
    schedule="linear",    # linear (DCT) | gaussian (DCMOEA) | fixed
    gauss_k=3.0,
    metric="frechet",
    resample_step=250.0,
    init="grafted",       # grafted | random
    lambda_eps=0.01,
    lambda_scale=1.0,     # c in c * lambda(t); summed arms only (rate variants)
)


# ----------------------------------------------------------------- encoding ops
def repair_breaks(breaks, n, available):
    """Force a valid, monotone cut vector that leaves withdrawn vehicles empty."""
    K = len(available)
    b = sorted(min(max(int(x), 0), n) for x in breaks)
    out, prev = [], 0
    for k in range(K - 1):
        v = max(b[k] if k < len(b) else n, prev)
        if not available[k]:
            v = prev
        out.append(v)
        prev = v
    if not available[K - 1] and out:
        out[-1] = n                      # last vehicle withdrawn -> empty tail
        for k in range(K - 2, -1, -1):   # restore monotonicity backwards
            if not available[k]:
                out[k] = out[k + 1] if k + 1 < len(out) else n
    return out


def order_crossover(p1, p2, rng):
    """Order crossover (OX) preserving relative order of the second parent."""
    n = len(p1)
    if n < 3:
        return list(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    child = [None] * n
    child[a:b] = p1[a:b]
    taken = set(p1[a:b])
    fill = [g for g in p2 if g not in taken]
    it = iter(fill)
    for i in range(n):
        if child[i] is None:
            child[i] = next(it)
    return child


def mutate(perm, breaks, rng, n_vehicles, available):
    perm = list(perm)
    n = len(perm)
    if n >= 2:
        kind = rng.integers(3)
        i, j = sorted(rng.choice(n, 2, replace=False))
        if kind == 0:                                   # swap
            perm[i], perm[j] = perm[j], perm[i]
        elif kind == 1:                                 # inversion
            perm[i:j + 1] = perm[i:j + 1][::-1]
        else:                                           # relocate
            g = perm.pop(j)
            perm.insert(i, g)
    breaks = list(breaks)
    if breaks and rng.random() < 0.5:
        k = int(rng.integers(len(breaks)))
        breaks[k] += int(rng.integers(-3, 4))
    return perm, repair_breaks(breaks, n, available)


def local_search(perm, breaks, inst, rng):
    """2-opt on one randomly chosen non-trivial route (Lamarckian: written back)."""
    routes = decode(perm, breaks, inst.n_vehicles)
    cand = [k for k, r in enumerate(routes) if len(r) >= 4]
    if not cand:
        return perm, breaks
    k = int(cand[rng.integers(len(cand))])
    routes[k] = two_opt(routes[k], inst.dist, max_pass=3)
    new_perm, new_breaks, pos = [], [], 0
    for kk, r in enumerate(routes):
        new_perm.extend(r)
        pos += len(r)
        if kk < inst.n_vehicles - 1:
            new_breaks.append(pos)
    return new_perm, new_breaks


def encode_routes(routes, n_vehicles):
    """Inverse of ``decode``: the concatenated permutation and its cut vector."""
    perm, breaks, pos = [], [], 0
    for k, r in enumerate(routes):
        perm.extend(r)
        pos += len(r)
        if k < n_vehicles - 1:
            breaks.append(pos)
    return perm, breaks


def relabel(routes, ev, inst):
    """Route orientations and driver assignment that minimise the deviation S.

    Reversing a route, or exchanging routes between available vehicles, leaves
    travel distance, route durations, loads and availability unchanged: costs
    are symmetric Euclidean distances, the fleet is homogeneous and there are
    no time windows.  S changes, because routes are matched to the plan in
    force by driver identity and the Frechet distance follows traversal order.
    The minimum of S over these operations is a linear assignment between the
    available vehicles and the plan's routes, each entry taking the better of
    the two orientations.  The input labelling is returned unchanged unless the
    assignment lowers S, so a plan already at the minimum is not touched.
    """
    avail = [k for k in range(inst.n_vehicles) if inst.available[k]]
    plan = [list(routes[k]) for k in avail]
    m = len(avail)
    cost = np.zeros((m, m))
    flip = np.zeros((m, m), dtype=bool)
    for a, k in enumerate(avail):
        for j, r in enumerate(plan):
            fwd = ev.route_deviation(k, r)
            if len(r) > 1:
                rev = ev.route_deviation(k, r[::-1])
                if rev < fwd:
                    cost[a, j], flip[a, j] = rev, True
                    continue
            cost[a, j] = fwd
    current = float(sum(ev.route_deviation(k, plan[a]) for a, k in enumerate(avail)))
    rows, cols = linear_sum_assignment(cost)
    if float(cost[rows, cols].sum()) >= current - 1e-9:
        return [list(r) for r in routes]
    out = [list(r) for r in routes]
    for a, j in zip(rows, cols):
        out[avail[a]] = plan[j][::-1] if flip[a, j] else list(plan[j])
    return out


def relabel_population(pop, F, ev, inst):
    """Apply ``relabel`` to every member; only the deviation column can change.

    Members whose labelling is already optimal keep their stored objective
    vector bit for bit.  A changed member's deviation is recomputed with the
    same per-vehicle sum as ``Evaluator.evaluate``.  Returns the new population,
    the new objective matrix and the number of members changed.
    """
    memo, out, F = {}, [], np.array(F, dtype=float, copy=True)
    n_changed = 0
    for i, (p, b) in enumerate(pop):
        routes = decode(p, b, inst.n_vehicles)
        key = tuple(tuple(r) for r in routes)
        if key not in memo:
            rl = relabel(routes, ev, inst)
            if rl == routes:
                memo[key] = None
            else:
                s = float(np.sum([ev.route_deviation(k, r) for k, r in enumerate(rl)]))
                memo[key] = (encode_routes(rl, inst.n_vehicles), s)
        hit = memo[key]
        if hit is None:
            out.append((p, b))
            continue
        out.append((list(hit[0][0]), list(hit[0][1])))
        F[i, 2] = hit[1]
        n_changed += 1
    return out, F, n_changed


# ------------------------------------------------------------- initialisation
def init_population(pair, ev, cfg, rng):
    inst = pair.disrupted
    N = cfg["pop_size"]
    pop = []
    if cfg["init"] == "grafted":
        seed_routes = greedy_reinsertion(pair)
        perm, breaks, pos = [], [], 0
        for k, r in enumerate(seed_routes):
            perm.extend(r)
            pos += len(r)
            if k < inst.n_vehicles - 1:
                breaks.append(pos)
        breaks = repair_breaks(breaks, len(perm), inst.available)
        n_exist = max(1, int(0.20 * N))
        n_part = int(0.50 * N)
        for _ in range(n_exist):
            pop.append((list(perm), list(breaks)))
        for _ in range(n_part):                     # locally perturbed grafts
            p, b = list(perm), list(breaks)
            for _ in range(int(rng.integers(1, 4))):
                p, b = mutate(p, b, rng, inst.n_vehicles, inst.available)
            pop.append((p, b))
    while len(pop) < N:
        pop.append(random_solution(inst, rng))
    return pop[:N]


# ------------------------------------------------------------------ schedules
def tau_schedule(tau0, gen, max_gen, cfg):
    frac = gen / max(1, max_gen)
    kind = cfg["schedule"]
    if kind == "fixed":
        # Constant scheduled width tau_0/2, the continuous-time mean of the
        # linear schedule.  The quantile floor still applies, so the effective
        # width remains population-dependent (Section 5.1).
        return 0.5 * tau0
    if kind == "gaussian":                       # Zeng et al. 2017 (DCMOEA)
        return tau0 * float(np.exp(-((cfg["gauss_k"] * frac) ** 2)))
    return tau0 * max(0.0, 1.0 - frac)           # Hobbie et al. 2021 (DCT)


def lambda_weight(S_feasible, diam, cfg):
    """Population-adaptive scalar weight lambda(t) of Eq. (9)."""
    f_max = float(np.max(S_feasible)) if len(S_feasible) else 0.0
    return diam / max(f_max, cfg["lambda_eps"] * diam)


# --------------------------------------------------------------------- driver
def _asf_pick(F, feas_mask, cols=(0, 1, 2)):
    """Final decision rule: minimax compromise over the hard-feasible set.

    Each objective is min-max normalised over the hard-feasible members, and
    the member minimising the largest normalised objective is returned; ties
    follow population order (Section 6.4).  ``cols`` selects the objectives.
    The reported results use all three for every arm, including ``cost_only``,
    which does not optimise deviation.  The choice restricted to the objectives
    each arm optimises is recorded as ``knee_alt`` (Table 2, "Per-arm minimax
    choice").
    """
    F = np.asarray(F, float)
    idx = np.flatnonzero(feas_mask)
    if len(idx) == 0:
        return None
    sub = F[idx][:, list(cols)]
    lo, hi = sub.min(axis=0), sub.max(axis=0)
    span = np.where(hi - lo <= 0, 1.0, hi - lo)
    return int(idx[int(np.argmin(((sub - lo) / span).max(axis=1)))])


# Eleven weights over ten decades (Appendix B), applied in situ so that
# Proposition 1 is checked on the populations that arise during optimisation.
LAMBDA_SWEEP = np.logspace(-4, 6, 11)


def run(pair, arm="corridor", seed=0, cfg=None, log_every=5, measure_influence=True):
    """Run one arm on one RSPP pair with one seed. Returns a result dict."""
    cfg = {**DEFAULTS, **(cfg or {})}
    rng = np.random.default_rng(seed)
    inst = pair.disrupted
    t_start = time.perf_counter()
    c_start = time.process_time()

    if arm == "greedy":
        routes = greedy_reinsertion(pair)
        ev = Evaluator(pair, cfg["metric"], cfg["resample_step"])
        f1, f2, f3, viol, info = ev.evaluate(routes)
        objectives_raw = [f1, f2, f3]
        # Output step shared by every arm (Section 6.4); only S can change.
        rl = relabel(routes, ev, inst)
        n_relabelled = int(rl != routes)
        if n_relabelled:
            routes = rl
            f3 = float(np.sum([ev.route_deviation(k, r) for k, r in enumerate(routes)]))
        return dict(arm=arm, seed=seed, instance=pair.name, objectives=[f1, f2, f3],
                    objectives_raw=objectives_raw, n_relabelled=n_relabelled,
                    knee_alt=[f1, f2, f3], violation=viol, feasible=viol <= 1e-12,
                    routes=routes, n_active=info["n_active"],
                    front=[[f1, f2, f3]], history=[], n_evals=1,
                    runtime=time.perf_counter() - t_start,
                    cpu_time=time.process_time() - c_start, stop="heuristic",
                    config={k: cfg[k] for k in ("metric", "resample_step")})

    ev = Evaluator(pair, cfg["metric"], cfg["resample_step"])
    from .geometry import diameter
    diam = diameter(inst.coords)

    use_sum = arm.startswith("sum_lambda")      # deviation summed into distance
    use_sim = arm != "cost_only" and not use_sum  # deviation as its own objective
    use_corridor = arm.startswith("corridor")
    use_eps = arm == "eps_static"
    use_bound = use_corridor or use_eps
    use_lambda = arm == "sim_lambda"
    n_obj = 3 if use_sim else 2
    if arm == "corridor_fixed":
        cfg = {**cfg, "schedule": "fixed"}
    if use_sum and "_x" in arm:          # rate variants: sum_lambda_x4 -> 4 * lambda(t)
        cfg = {**cfg, "lambda_scale": float(arm.rsplit("_x", 1)[1])}
    alg = ("nsga2" if arm.endswith("_nsga2")
           else "moead" if arm.endswith("_moead") else "nsga3")

    ref = moea.uniform_reference_points(n_obj, cfg["ref_p"] if n_obj == 3 else 91)
    state = moea.Nsga3State(n_obj)
    N = cfg["pop_size"]

    pop = init_population(pair, ev, cfg, rng)
    raw = [ev.evaluate(decode(p, b, inst.n_vehicles)) for p, b in pop]
    F_all = np.array([[r[0], r[1], r[2]] for r in raw])
    V_base = np.array([r[3] for r in raw])

    if use_corridor:
        tau0 = float(np.quantile(F_all[:, 2], cfg["tau_q0"]))
    elif use_eps:
        # Fixed before the first generation and never revised: the deviation of
        # the greedy repair of the plan in force, Eq. (15).
        tau0 = float(ev.evaluate(greedy_reinsertion(pair))[2])
    else:
        tau0 = None
    history = []

    if alg == "moead":
        weights = ref.copy()
        nbrs = moea.moead_neighbourhoods(weights, min(20, len(weights)))
        sel = rng.choice(len(pop), len(weights), replace=True)
        pop = [pop[i] for i in sel]
        F_all, V_base = F_all[sel], V_base[sel]
        N = len(weights)

    # ---- objective vector shown to the selector, and the violation it sees
    def obj_matrix(F, lam_val):
        if use_sum:
            return np.column_stack([F[:, 0] + lam_val * F[:, 2], F[:, 1]])
        if not use_sim:
            return F[:, :2]
        if use_lambda:
            out = F.copy()
            out[:, 2] = out[:, 2] * lam_val
            return out
        return F

    def viol_vector(F, V, tau_val):
        if tau_val is None:
            return V
        return V + np.maximum(0.0, F[:, 2] - tau_val) / max(tau_val, 1.0)

    max_gen = cfg["max_gen"]
    stop = "max_gen"
    for gen in range(1, max_gen + 1):
        # ---- corridor / weight state for this generation
        tau = None
        if use_corridor:
            tau = max(tau_schedule(tau0, gen, max_gen, cfg),
                      float(np.quantile(F_all[:, 2], cfg["tau_qmin"])))
        elif use_eps:
            tau = tau0        # static: no schedule, no non-vacuity floor
        lam = 1.0
        if use_lambda:
            feas_S = F_all[V_base <= 1e-12, 2]
            lam = lambda_weight(feas_S, diam, cfg)
        elif use_sum:
            # Eq. (9) takes S_max over the hard-feasible members.  While there
            # are none it is taken over the whole population: the empty-set
            # value 1/lambda_eps would make the summed objective almost
            # proportional to deviation in the generation in which the first
            # feasible plans appear.
            feas_S = F_all[V_base <= 1e-12, 2]
            lam = cfg["lambda_scale"] * lambda_weight(
                feas_S if len(feas_S) else F_all[:, 2], diam, cfg)
            # The summed objective changes scale with lambda(t), so the NSGA-III
            # ideal point is taken afresh from each generation's pool.
            state = moea.Nsga3State(n_obj)

        # ---- variation
        offspring = []
        while len(offspring) < N:
            i, j = rng.integers(len(pop), size=2)
            (p1, b1), (p2, b2) = pop[i], pop[j]
            if rng.random() < cfg["p_cx"] and len(p1) == len(p2):
                child_p = order_crossover(p1, p2, rng)
                child_b = [b1[k] if rng.random() < 0.5 else b2[k]
                           for k in range(len(b1))]
                child_b = repair_breaks(child_b, len(child_p), inst.available)
            else:
                child_p, child_b = list(p1), list(b1)
            if rng.random() < cfg["p_mut"]:
                child_p, child_b = mutate(child_p, child_b, rng,
                                          inst.n_vehicles, inst.available)
            if rng.random() < cfg["p_ls"]:
                child_p, child_b = local_search(child_p, child_b, inst, rng)
                child_b = repair_breaks(child_b, len(child_p), inst.available)
            offspring.append((child_p, child_b))

        raw_off = [ev.evaluate(decode(p, b, inst.n_vehicles)) for p, b in offspring]
        F_off = np.array([[r[0], r[1], r[2]] for r in raw_off])
        V_off = np.array([r[3] for r in raw_off])

        comb = pop + offspring
        F_comb = np.vstack([F_all, F_off])
        V_comb = np.concatenate([V_base, V_off])

        sched_active = None
        if use_eps:
            tau = tau0
        if use_corridor:  # re-floor tau on the combined pool (non-vacuity)
            tau_sched = tau_schedule(tau0, gen, max_gen, cfg)
            tau_floor = float(np.quantile(F_comb[:, 2], cfg["tau_qmin"]))
            tau = max(tau_sched, tau_floor)
            # Which of the two terms in Eq. (12) is binding, recorded directly
            # rather than inferred from where the width crosses the population.
            sched_active = bool(tau_sched >= tau_floor)

        Fsel = obj_matrix(F_comb, lam)
        Vsel = viol_vector(F_comb, V_comb, tau)

        # ---- environmental selection
        if alg == "nsga3":
            rng_state = rng.bit_generator.state
            surv = moea.nsga3_survivors(Fsel, Vsel, N, ref, state, rng)
        elif alg == "nsga2":
            rng_state = rng.bit_generator.state
            surv = moea.nsga2_survivors(Fsel, Vsel, N, rng)
        else:  # MOEA/D: neighbourhood replacement, feasibility first
            rng_state = rng.bit_generator.state
            if use_sum:
                # Normalised Tchebycheff: the summed objective changes scale with
                # lambda(t), so the ideal and worst values are those of the
                # current pool in the selection objectives.
                ideal, nadir = Fsel.min(axis=0), Fsel.max(axis=0)
            else:
                state.update(F_comb[:, :n_obj])
                ideal, nadir = state.ideal, None
            surv = list(range(N))
            for si in range(N):
                child = N + si
                for nb in nbrs[si]:
                    cur = surv[nb]
                    vc, vk = Vsel[child], Vsel[cur]
                    if vc < vk - 1e-12:
                        surv[nb] = child
                    elif abs(vc - vk) <= 1e-12:
                        gc = moea.tchebycheff(Fsel[child][:n_obj], weights[nb], ideal, nadir)
                        gk = moea.tchebycheff(Fsel[cur][:n_obj], weights[nb], ideal, nadir)
                        if gc < gk:
                            surv[nb] = child

        # ---- in-situ measurement of the selection influence (Sections 4.3, 9.1)
        #
        # Two survivor-set numbers, not used in the article:
        #   flip_treat  same population, same RNG stream, mechanism on vs off
        #   flip_noise  same population, mechanism on both times, RNG reseeded
        # Both selections run from a freshly initialised normalisation state.
        # The article reports rank_change below, which involves neither the
        # RNG nor normalisation.
        flip_treat = flip_noise = rank_change = sweep_rank_change = np.nan
        if (measure_influence and alg == "nsga3" and (use_bound or use_lambda or use_sum)
                and (gen % log_every == 0 or gen == max_gen)):
            saved = rng.bit_generator.state

            def _sel(Fm, Vv, rstate):
                rng.bit_generator.state = rstate
                return moea.nsga3_survivors(Fm, Vv, N, ref, moea.Nsga3State(n_obj), rng)

            if use_bound:                          # off = drop the corridor term
                F_on, V_on, F_off, V_off_ = Fsel, Vsel, Fsel, V_comb
            elif use_sum:                          # off = no deviation term, lambda = 0
                F_on, V_on = Fsel, Vsel
                F_off, V_off_ = obj_matrix(F_comb, 0.0), Vsel
            else:                                  # off = set the multiplier to 1
                F_on, V_on = Fsel, Vsel
                F_off, V_off_ = obj_matrix(F_comb, 1.0), Vsel

            a = _sel(F_on, V_on, rng_state)
            b = _sel(F_off, V_off_, rng_state)
            c = _sel(F_on, V_on, rng.bit_generator.state)   # same treatment, new RNG
            flip_treat = len(set(a) ^ set(b)) / (2.0 * N)
            flip_noise = len(set(a) ^ set(c)) / (2.0 * N)
            rng.bit_generator.state = saved
            # RNG-free: does the mechanism move anyone between fronts at all?
            idx_on = moea.front_index(F_on, V_on)
            rank_change = float(np.mean(idx_on != moea.front_index(F_off, V_off_)))

            # For the weighted arms, the on/off contrast above tests a single
            # lambda.  Proposition 1 is a statement about every positive
            # multiplier, so sweep ten decades on the identical population and
            # record the worst case against lambda = 1.  On a separate objective
            # this is zero in exact arithmetic, and a non-zero value comes from
            # floating-point rounding (Section 4.1).  Summed into distance, the
            # same sweep measures how far the ranking depends on the weight.
            if use_lambda or use_sum:
                idx_unit = moea.front_index(obj_matrix(F_comb, 1.0), Vsel)
                worst = 0.0
                for lam_s in LAMBDA_SWEEP:
                    worst = max(worst, float(np.mean(
                        moea.front_index(obj_matrix(F_comb, float(lam_s)), Vsel)
                        != idx_unit)))
                sweep_rank_change = worst

        pop = [comb[i] for i in surv]
        F_all = F_comb[surv]
        V_base = V_comb[surv]

        if gen % log_every == 0 or gen == max_gen:
            feas = V_base <= 1e-12
            history.append(dict(
                gen=gen,
                tau=None if tau is None else float(tau),
                sched_active=sched_active,
                lam=float(lam),
                S_med=float(np.median(F_all[:, 2])),
                S_q10=float(np.quantile(F_all[:, 2], 0.10)),
                S_q90=float(np.quantile(F_all[:, 2], 0.90)),
                dist_med=float(np.median(F_all[:, 0])),
                corridor_feasible=(float(np.mean(F_all[:, 2] <= tau))
                                   if tau is not None else None),
                feasible_frac=float(np.mean(feas)),
                n_active_med=float(np.median(
                    [sum(1 for r in decode(p, b, inst.n_vehicles) if r)
                     for p, b in pop])),
                flip_treat=float(flip_treat) if flip_treat == flip_treat else None,
                flip_noise=float(flip_noise) if flip_noise == flip_noise else None,
                rank_change=float(rank_change) if rank_change == rank_change else None,
                sweep_rank_change=(float(sweep_rank_change)
                                   if sweep_rank_change == sweep_rank_change else None),
                outside_corridor=(float(np.mean(F_all[:, 2] > tau))
                                  if tau is not None else None),
            ))

    # ---------------------------------------------------------------- output
    # Objectives of the plan the search itself selects, before the output step.
    # Kept so that a rerun can be checked against earlier databases, run for run.
    feas = V_base <= 1e-12
    pick_raw = _asf_pick(F_all, feas)
    objectives_raw = (None if pick_raw is None
                      else [float(x) for x in F_all[pick_raw]])

    # Output step shared by every arm (Section 6.4): each member of the final
    # population takes the route orientations and driver assignment that
    # minimise S.  Distance, durations, loads and availability are unchanged,
    # so only the deviation column of the objective matrix can move.
    pop, F_all, n_relabelled = relabel_population(pop, F_all, ev, inst)

    nd = moea.nondominated(F_all[feas]) if feas.any() else np.array([], dtype=int)
    front = F_all[feas][nd].tolist() if len(nd) else []
    pick = _asf_pick(F_all, feas)
    # The same rule restricted to the objectives this arm optimises.  Reported,
    # not used for the headline results (Table 2, "Per-arm minimax choice").
    if use_sum:
        # The summed arm optimises (f1 + lambda S, f2): the rule is applied to
        # those two objectives at the weight of the last generation.
        alt = _asf_pick(obj_matrix(F_all, lam), feas, cols=(0, 1))
    else:
        alt = _asf_pick(F_all, feas, cols=(0, 1, 2) if use_sim else (0, 1))
    if pick is None:
        pick = int(np.argmin(V_base))
    p, b = pop[pick]

    # Re-measure the chosen plan under every trajectory metric, so that arms
    # which *optimised* different metrics can still be compared on one yardstick.
    chosen_routes = decode(p, b, inst.n_vehicles)
    deviation_all = {}
    for mname in ("frechet", "dtw", "hausdorff", "lcss"):
        ev_m = Evaluator(pair, mname, cfg["resample_step"])
        deviation_all[mname] = float(ev_m.evaluate(chosen_routes)[2])

    return dict(
        arm=arm, seed=seed, instance=pair.name,
        objectives=[float(x) for x in F_all[pick]],
        objectives_raw=objectives_raw, n_relabelled=int(n_relabelled),
        knee_alt=None if alt is None else [float(x) for x in F_all[alt]],
        n_active=int(sum(1 for r in chosen_routes if r)),
        violation=float(V_base[pick]), feasible=bool(feas[pick]),
        routes=chosen_routes, deviation_all=deviation_all,
        front=front, history=history, n_evals=ev.n_evals,
        runtime=time.perf_counter() - t_start,
        cpu_time=time.process_time() - c_start, stop=stop,
        tau0=tau0, diam=diam,
        config={k: cfg[k] for k in ("pop_size", "max_gen", "metric", "schedule",
                                    "init", "resample_step", "ref_p", "lambda_scale")},
    )
