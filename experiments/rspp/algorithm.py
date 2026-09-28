"""The similarity-preserving evolutionary algorithm and its comparison arms.

Arms
----
greedy           cheapest-feasible-insertion repair of the baseline (no search)
cost_only        NSGA-III on (f1, f2); similarity measured but not optimised
sim_obj          NSGA-III on (f1, f2, f3); similarity as a plain objective
sim_lambda       NSGA-III on (f1, f2, lambda(t)*f3)  -- the previous method
eps_static       NSGA-III on (f1, f2, f3) + a *static* epsilon-constraint on
                 deviation, the bound fixed at the deviation of the greedy
                 repair of the plan in force.  Exogenous to the population and
                 to the schedule, so it isolates "similarity moved from the
                 objective side to the constraint side" from the two things the
                 corridor adds on top: population calibration and tightening.
corridor         NSGA-III on (f1, f2, f3) + adaptive similarity corridor  [proposed]
corridor_fixed   as above with a static corridor (no tightening)          [ablation]
corridor_nsga2   proposed mechanism carried by NSGA-II
corridor_moead   proposed mechanism carried by MOEA/D (Tchebycheff)
"""
from __future__ import annotations

import time

import numpy as np

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
    tau_qmin=0.10,        # non-vacuity floor: >=10% of the population stays inside
    schedule="linear",    # linear (DCT) | gaussian (DCMOEA) | fixed
    gauss_k=3.0,
    metric="frechet",
    resample_step=250.0,
    init="grafted",       # grafted | random
    lambda_eps=0.01,
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
        # Held at the time-average of the linear schedule, so that this arm
        # isolates *tightening* from *tightness*: it spends the same total
        # corridor budget without ever narrowing.  Holding it at tau_0 instead
        # would leave the corridor permanently slack and make the arm a
        # duplicate of the plain deviation objective.
        return 0.5 * tau0
    if kind == "gaussian":                       # Jiao et al. 2019 (DCMOEA)
        return tau0 * float(np.exp(-((cfg["gauss_k"] * frac) ** 2)))
    return tau0 * max(0.0, 1.0 - frac)           # Hobbie et al. 2021 (DCT)


def lambda_weight(S_feasible, diam, cfg):
    """The previous method's population-adaptive scalar weight, verbatim."""
    f_max = float(np.max(S_feasible)) if len(S_feasible) else 0.0
    return diam / max(f_max, cfg["lambda_eps"] * diam)


# --------------------------------------------------------------------- driver
def _asf_pick(F, feas_mask, cols=(0, 1, 2)):
    """Final decision rule: the achievement-scalarizing knee of the feasible set.

    ``cols`` selects the objectives the scalarizing function ranges over.  The
    reported results use all three for *every* arm, including ``cost_only``,
    which never optimises deviation: that hands the baseline a deviation-sensitive
    final pick and therefore makes the reported advantage of the corridor a
    lower bound.  The alternative -- each arm's knee ranging only over the
    objectives it optimises -- is recorded alongside as ``knee_alt`` so the
    choice is visible rather than buried.
    """
    F = np.asarray(F, float)
    idx = np.flatnonzero(feas_mask)
    if len(idx) == 0:
        return None
    sub = F[idx][:, list(cols)]
    lo, hi = sub.min(axis=0), sub.max(axis=0)
    span = np.where(hi - lo <= 0, 1.0, hi - lo)
    return int(idx[int(np.argmin(((sub - lo) / span).max(axis=1)))])


# Ten decades, the range swept by the standalone gate experiment, reused in situ so
# that Proposition 1 is checked against the populations that actually arise
# during optimisation rather than only against synthetic ones.
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
        return dict(arm=arm, seed=seed, instance=pair.name, objectives=[f1, f2, f3],
                    knee_alt=[f1, f2, f3], violation=viol, feasible=viol <= 1e-12,
                    routes=routes, n_active=info["n_active"],
                    front=[[f1, f2, f3]], history=[], n_evals=1,
                    runtime=time.perf_counter() - t_start,
                    cpu_time=time.process_time() - c_start, stop="heuristic",
                    config={k: cfg[k] for k in ("metric", "resample_step")})

    ev = Evaluator(pair, cfg["metric"], cfg["resample_step"])
    from .geometry import diameter
    diam = diameter(inst.coords)

    use_sim = arm != "cost_only"
    use_corridor = arm.startswith("corridor")
    use_eps = arm == "eps_static"
    use_bound = use_corridor or use_eps
    use_lambda = arm == "sim_lambda"
    n_obj = 3 if use_sim else 2
    if arm == "corridor_fixed":
        cfg = {**cfg, "schedule": "fixed"}
    alg = ("nsga2" if arm == "corridor_nsga2"
           else "moead" if arm == "corridor_moead" else "nsga3")

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
        # the greedy repair of the plan in force.  A dispatcher can compute it
        # without running the search, which is what makes it a legitimate
        # exogenous epsilon rather than a value read off the corridor.
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
            # Which of the two terms in Eq. (8) is binding.  Reported directly
            # rather than inferred from where the width crosses the population,
            # because "the schedule has handed over to the non-vacuity floor" is
            # a statement about the max in Eq. (8) and nothing else.
            sched_active = bool(tau_sched >= tau_floor)

        # ---- objective vector actually shown to the selector
        def obj_matrix(F, lam_val):
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
            state.update(F_comb[:, :n_obj])
            surv = list(range(N))
            for si in range(N):
                child = N + si
                for nb in nbrs[si]:
                    cur = surv[nb]
                    vc, vk = Vsel[child], Vsel[cur]
                    if vc < vk - 1e-12:
                        surv[nb] = child
                    elif abs(vc - vk) <= 1e-12:
                        gc = moea.tchebycheff(Fsel[child][:n_obj], weights[nb], state.ideal)
                        gk = moea.tchebycheff(Fsel[cur][:n_obj], weights[nb], state.ideal)
                        if gc < gk:
                            surv[nb] = child

        # ---- in-situ measurement of the selection influence (Reviewer #4.3)
        #
        # Two numbers, and the second is what makes the first interpretable:
        #   flip_treat  same population, same RNG stream, mechanism on vs off
        #   flip_noise  same population, mechanism on both times, RNG reseeded
        # Both selections run from a *freshly initialised* normalisation state
        # so that the comparison isolates this generation's decision rather than
        # inheriting an ideal point that the treatment itself has been moving.
        flip_treat = flip_noise = rank_change = sweep_rank_change = np.nan
        if (measure_influence and alg == "nsga3" and (use_bound or use_lambda)
                and (gen % log_every == 0 or gen == max_gen)):
            saved = rng.bit_generator.state

            def _sel(Fm, Vv, rstate):
                rng.bit_generator.state = rstate
                return moea.nsga3_survivors(Fm, Vv, N, ref, moea.Nsga3State(n_obj), rng)

            if use_bound:                          # off = drop the corridor term
                F_on, V_on, F_off, V_off_ = Fsel, Vsel, Fsel, V_comb
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

            # For the weighted arm, the on/off contrast above tests a single
            # lambda.  Proposition 1 is a statement about *every* positive
            # multiplier, so sweep ten decades on the identical population and
            # record the worst case.  If the proposition holds, this is zero for
            # every lambda; anything non-zero is the floating-point channel of
            # Corollary 1 and is what the reported residue consists of.
            if use_lambda:
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
    feas = V_base <= 1e-12
    nd = moea.nondominated(F_all[feas]) if feas.any() else np.array([], dtype=int)
    front = F_all[feas][nd].tolist() if len(nd) else []
    pick = _asf_pick(F_all, feas)
    # The same rule restricted to the objectives this arm actually optimises.
    # Reported, not used: it is what the comparison would look like if each arm
    # were allowed its own decision space, and for `cost_only` it is materially
    # worse on deviation -- which is precisely why the three-objective rule is
    # the conservative one to headline.
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
        knee_alt=None if alt is None else [float(x) for x in F_all[alt]],
        n_active=int(sum(1 for r in chosen_routes if r)),
        violation=float(V_base[pick]), feasible=bool(feas[pick]),
        routes=chosen_routes, deviation_all=deviation_all,
        front=front, history=history, n_evals=ev.n_evals,
        runtime=time.perf_counter() - t_start,
        cpu_time=time.process_time() - c_start, stop=stop,
        tau0=tau0, diam=diam,
        config={k: cfg[k] for k in ("pop_size", "max_gen", "metric", "schedule",
                                    "init", "resample_step", "ref_p")},
    )
