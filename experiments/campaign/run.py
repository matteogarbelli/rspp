"""Staged experimental campaign over RSPP-Bench.

Every result row carries the seed, the arm, the instance, the full config and
the stopping criterion, so that any figure in the paper regenerates from the
stored rows alone.  Results are appended as JSON Lines; a run that is already
present (same instance/arm/seed/study) is skipped, so the campaign is
restartable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

from ..rspp import algorithm
from ..rspp.instance import RSPPPair

MAIN_ARMS = ["greedy", "cost_only", "sim_obj", "sim_lambda", "eps_static",
             "corridor", "corridor_fixed", "corridor_nsga2", "corridor_moead"]
METRICS = ["frechet", "dtw", "hausdorff", "lcss"]


def _trim(res, keep_routes):
    if not keep_routes:
        res.pop("routes", None)
    return res


def _one(job):
    bench_dir, inst_name, arm, seed, study, cfg, keep_routes = job
    pair = RSPPPair.load(os.path.join(bench_dir, inst_name + ".json"))
    t0 = time.perf_counter()
    try:
        res = algorithm.run(pair, arm=arm, seed=seed, cfg=cfg)
    except Exception as exc:            # a crashed cell must not lose the campaign
        import traceback
        return {"study": study, "instance": inst_name, "arm": arm, "seed": seed,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:]}
    res["study"] = study
    res["wall"] = time.perf_counter() - t0
    res["n_customers"] = pair.disrupted.n_customers
    res["n_vehicles"] = pair.disrupted.n_vehicles
    res["family"] = pair.disruption["family"]
    return _trim(res, keep_routes)


# Sensitivity and metric studies run on a representative subset; the main
# comparison uses all 27 instances.  The subset is a Latin square over the
# design: one instance for every (family, size) pair, with each layout appearing
# exactly once per family and once per size.  Nine cells therefore cover every
# level of every factor, which the previous four-instance subset did not -- it
# contained no demand-surge instance at all, so no ablation ever saw the family
# with the largest effect.
SUBSET = [
    # INS            WDR                SRG
    "C50-4v-INS5-s1",   "R50-4v-WDR0-s1",   "RC50-4v-SRG6-s1",
    "R100-7v-INS8-s1",  "RC100-7v-WDR0-s1", "C100-7v-SRG12-s1",
    "RC200-12v-INS15-s1", "C200-12v-WDR0-s1", "R200-12v-SRG24-s1",
]


SCALE_ARMS = ["cost_only", "corridor", "corridor_moead"]


def build_jobs(bench_dir, instances, seeds, studies, scale_instances=()):
    jobs = []
    subset = [i for i in instances if i in SUBSET] or instances[:2]
    for inst in instances:
        for s in seeds:
            if "main" in studies:
                for arm in MAIN_ARMS:
                    jobs.append((bench_dir, inst, arm, s, "main", {}, s == seeds[0]))
    for inst in subset:
        for s in seeds:
            if "budget" in studies:
                # Does the generation budget suffice?  Run time alone cannot say
                # whether 300 generations is enough at the largest size; halving
                # and doubling it can.
                for g in (150, 600):
                    jobs.append((bench_dir, inst, "corridor", s, f"budget:{g}",
                                 {"max_gen": g}, False))
            if "timing" in studies:
                # Uncontended timing: run this study with --procs 1 so the
                # reported cost is not wall time under seven-way contention.
                jobs.append((bench_dir, inst, "corridor", s, "timing", {}, False))
    for inst in scale_instances:
        if "scale" in studies:
            for s in seeds:
                for arm in SCALE_ARMS:
                    jobs.append((bench_dir, inst, arm, s, "scale", {}, False))
    for inst in subset:
        for s in seeds:
            if "metric" in studies:
                for m in METRICS:
                    jobs.append((bench_dir, inst, "corridor", s, f"metricv2:{m}",
                                 {"metric": m}, False))
            if "init" in studies:
                jobs.append((bench_dir, inst, "corridor", s, "init:random",
                             {"init": "random"}, False))
            if "schedule" in studies:
                jobs.append((bench_dir, inst, "corridor", s, "schedule:gaussian",
                             {"schedule": "gaussian"}, False))
            if "calib" in studies:
                for q in (0.25, 0.50, 0.75):
                    jobs.append((bench_dir, inst, "corridor", s,
                                 f"calib:q{int(q*100)}", {"tau_q0": q}, False))
            if "qmin" in studies:
                # The non-vacuity floor is the parameter the guarantee of
                # Proposition 3 is stated in terms of, so its own sensitivity
                # cannot be left to the calibration study: q_0 sets where the
                # corridor starts, q_min sets how far it is allowed to close.
                for q in (0.01, 0.05, 0.20):
                    jobs.append((bench_dir, inst, "corridor", s,
                                 f"qmin:q{int(q*100)}", {"tau_qmin": q}, False))
            if "resample" in studies:
                for step in (100.0, 500.0, 1000.0):
                    jobs.append((bench_dir, inst, "corridor", s,
                                 f"resample:{int(step)}",
                                 {"resample_step": step}, False))
    return jobs


def _write_environment(path, procs, studies):
    """Snapshot the execution environment alongside the results database."""
    import platform
    import subprocess

    def _sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or None
        except Exception:
            return None

    import numpy
    import scipy
    env = dict(
        cpu=_sysctl("machdep.cpu.brand_string") or platform.processor(),
        cores=os.cpu_count(),
        memory_gb=(round(int(_sysctl("hw.memsize")) / 2 ** 30)
                   if (_sysctl("hw.memsize") or "").isdigit() else None),
        system=platform.system(), release=platform.release(),
        mac_ver=platform.mac_ver()[0] or None,
        python=platform.python_version(), numpy=numpy.__version__,
        scipy=scipy.__version__, procs=procs, studies=list(studies),
    )
    prev = {}
    if os.path.exists(path):
        try:
            prev = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            prev = {}
    # One entry per study group, so a study run at a different concurrency (the
    # uncontended timing study, notably) records its own procs count.
    prev.update({s: env for s in studies})
    prev["_last"] = env
    with open(path, "w") as fh:
        json.dump(prev, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/rsppbench")
    ap.add_argument("--out", default="experiments/results/campaign.jsonl")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--studies", nargs="+", default=["main"])
    ap.add_argument("--instances", nargs="*", default=None)
    a = ap.parse_args()

    with open(os.path.join(a.bench, "MANIFEST.json")) as fh:
        manifest = json.load(fh)["suite"]
    # The oversized pairs exist only to extend the range of the cost
    # measurement; they are not part of the comparison suite.
    scale_instances = [m["name"] for m in manifest if m.get("scale_only")]
    instances = a.instances or [m["name"] for m in manifest
                                if not m.get("scale_only")]
    seeds = list(range(a.seeds))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out) or os.path.exists(a.out + ".gz"):
        import gzip
        opener = (gzip.open(a.out + ".gz", "rt")
                  if not os.path.exists(a.out) else open(a.out))
        with opener as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r.get("study"), r.get("instance"), r.get("arm"),
                              r.get("seed")))
                except json.JSONDecodeError:
                    continue

    jobs = [j for j in build_jobs(a.bench, instances, seeds, a.studies,
                                  scale_instances)
            if (j[4], j[1], j[2], j[3]) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already present), "
          f"{a.procs} processes", flush=True)

    # Record the machine that ran the campaign, next to the campaign.  Deriving
    # it later from whichever machine happened to draw the figures is how a
    # paper ends up describing the wrong computer.
    _write_environment(os.path.join(os.path.dirname(a.out), "environment.json"),
                       a.procs, a.studies)

    t0 = time.perf_counter()
    with open(a.out, "a") as fh, Pool(a.procs) as pool:
        for i, res in enumerate(pool.imap_unordered(_one, jobs, chunksize=1), 1):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if i % 20 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                eta = el / i * (len(jobs) - i)
                print(f"  {i}/{len(jobs)}  elapsed {el/60:.1f} min  "
                      f"eta {eta/60:.1f} min", flush=True)
    print("campaign complete", flush=True)


if __name__ == "__main__":
    main()
