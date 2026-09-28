"""RSPP-Bench: a public, seeded benchmark of baseline-vs-disrupted instance pairs.

Public routing benchmarks are single-shot: they give a problem, not a problem
*and the plan that was already running when it changed*.  Evaluating a stability-
preserving method needs the pair.  RSPP-Bench supplies it, and supplies it
reproducibly: every instance is a pure function of its seed, so the whole suite
regenerates byte-identically from the manifest on a clean checkout.

Naming:  <layout><n>-<K>v-<family><k>-s<seed>
  layout  C clustered | R uniform | RC mixed          (after Solomon's taxonomy)
  family  INS new requests | WDR vehicle withdrawal | SRG demand surge

Usage:  python -m experiments.rsppbench.generate --out data/rsppbench
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from ..rspp.heuristics import build_baseline
from ..rspp.instance import Instance, RSPPPair, make_instance


# ---------------------------------------------------------------- disruptions
def disrupt_insert(base, baseline_routes, rng, k_new):
    """k new service requests arrive mid-shift, at fresh locations."""
    extent = base.meta["extent"]
    new_pts = rng.uniform(-extent, extent, (k_new, 2))
    coords = np.vstack([base.coords, new_pts])
    demand = np.concatenate([base.demand, rng.integers(1, 20, k_new).astype(float)])
    service = np.concatenate([base.service, rng.integers(300, 1200, k_new).astype(float)])
    inst = Instance(name=base.name + "+ins", coords=coords, demand=demand,
                    service=service, capacity=base.capacity,
                    n_vehicles=base.n_vehicles, max_duration=base.max_duration,
                    speed=base.speed, available=base.available.copy(),
                    meta=dict(base.meta))
    ids = list(range(len(base.coords), len(coords)))
    return inst, {"family": "INS", "k_new": int(k_new), "new_customers": ids}


def disrupt_withdraw(base, baseline_routes, rng, k_new=0):
    """One vehicle is lost; its customers must be absorbed by the rest."""
    busy = [k for k, r in enumerate(baseline_routes) if len(r) > 0]
    victim = int(busy[rng.integers(len(busy))]) if busy else 0
    if victim == base.n_vehicles - 1 and len(busy) > 1:
        victim = int(busy[0])                       # keep the tail vehicle usable
    avail = base.available.copy()
    avail[victim] = False
    inst = Instance(name=base.name + "+wdr", coords=base.coords.copy(),
                    demand=base.demand.copy(), service=base.service.copy(),
                    capacity=base.capacity, n_vehicles=base.n_vehicles,
                    max_duration=base.max_duration, speed=base.speed,
                    available=avail, meta=dict(base.meta))
    return inst, {"family": "WDR", "withdrawn_vehicle": victim,
                  "orphaned": [int(c) for c in baseline_routes[victim]]}


def disrupt_surge(base, baseline_routes, rng, k_new):
    """Demand at k already-scheduled customers is revised sharply upward."""
    n = base.n_customers
    picked = rng.choice(np.arange(1, n + 1), size=min(k_new, n), replace=False)
    demand = base.demand.copy()
    demand[picked] *= 2.5
    service = base.service.copy()
    service[picked] *= 1.5
    inst = Instance(name=base.name + "+srg", coords=base.coords.copy(),
                    demand=demand, service=service, capacity=base.capacity,
                    n_vehicles=base.n_vehicles, max_duration=base.max_duration,
                    speed=base.speed, available=base.available.copy(),
                    meta=dict(base.meta))
    return inst, {"family": "SRG", "surged_customers": [int(c) for c in picked]}


FAMILIES = {"INS": disrupt_insert, "WDR": disrupt_withdraw, "SRG": disrupt_surge}


def make_pair(layout, n, n_vehicles, family, k_new, seed):
    base = make_instance(f"{layout}{n}", n, n_vehicles, seed, layout=layout)
    baseline_routes = build_baseline(base)
    rng = np.random.default_rng(seed + 10_000)
    inst, manifest = FAMILIES[family](base, baseline_routes, rng, k_new)
    manifest["k_new"] = int(k_new)
    name = f"{layout}{n}-{n_vehicles}v-{family}{k_new}-s{seed}"
    return RSPPPair(name=name, base=base, disrupted=inst,
                    baseline_routes=baseline_routes, disruption=manifest, seed=seed)


# ------------------------------------------------------------------- the suite
#
# The suite is a full factorial: every spatial layout crossed with every
# disruption family at every size.  An earlier version sampled 8 of these 27
# cells, which left the demand-surge family represented by a single instance and
# made every family comparison inseparable from layout and size.  A claim of the
# form "the gain is largest on demand surge" needs the surge family to appear at
# every layout and every size, or it is a statement about one instance.
SIZES = [(50, 4), (100, 7), (200, 12)]          # (customers, crew)
LAYOUTS_ORDER = ["C", "R", "RC"]

# Disruption magnitude scales with the instance so that the *relative* severity
# is comparable across sizes: ~10% of the request set for INS, ~12% for SRG.
K_NEW = {"INS": {50: 5, 100: 8, 200: 15},
         "SRG": {50: 6, 100: 12, 200: 24},
         "WDR": {50: 0, 100: 0, 200: 0}}       # WDR magnitude is one vehicle

SUITE = [(layout, n, K, fam, K_NEW[fam][n])
         for n, K in SIZES
         for fam in ("INS", "WDR", "SRG")
         for layout in LAYOUTS_ORDER]

# Larger pairs used only by the scalability study.  They are deliberately not
# part of the main comparison: the point of the suite is to isolate the handling
# of stability, and these exist to extend the size range of the cost measurement
# past the point where a fit over three sizes would be meaningless.
#
# The size ceiling is set by baseline *construction*, not by the solver: the
# balancing descent that drives the plan in force to a local optimum of the
# planner objective costs about 2 s at n = 100 and 27 s at n = 200, so a
# thousand-customer pair would take hours to generate and its baseline would be
# the only expensive thing in the suite.  Four doublings of n is the range this
# construction supports.
SCALE_SUITE = [
    ("C",  400, 20, "INS", 30),
    ("R",  400, 20, "INS", 30),
    ("RC", 400, 20, "INS", 30),
]


def build_suite(out_dir, seeds=(1,), include_scale=True, skip_existing=False):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    suite = list(SUITE) + (list(SCALE_SUITE) if include_scale else [])
    scale_names = {f"{l}{n}-{K}v-{f}{k}" for l, n, K, f, k in SCALE_SUITE}
    for layout, n, K, fam, k_new in suite:
        for s in seeds:
            name = f"{layout}{n}-{K}v-{fam}{k_new}-s{s}"
            path = os.path.join(out_dir, name + ".json")
            # Instances are pure functions of their seed, so a file already on
            # disk is byte-identical to the one this call would write.  Skipping
            # it keeps a partially built suite resumable, which matters because
            # driving the plan in force to a local optimum is the expensive part
            # of generation at the larger sizes.
            if skip_existing and os.path.exists(path):
                pair = RSPPPair.load(path)
            else:
                pair = make_pair(layout, n, K, fam, k_new, s)
                pair.save(path)
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            scale_only = f"{layout}{n}-{K}v-{fam}{k_new}" in scale_names
            manifest.append(dict(name=pair.name, file=os.path.basename(path),
                                 layout=layout, n_customers=n, n_vehicles=K,
                                 family=fam, k_new=k_new, seed=s,
                                 scale_only=scale_only,
                                 sha256=digest,
                                 disruption=pair.disruption))
            tag = "  [scale]" if scale_only else ""
            print(f"  {pair.name:28s} n={n:3d} K={K:2d} {fam}  "
                  f"sha={digest[:12]}{tag}", flush=True)
            # Written after every instance, so a suite still being built is
            # already usable for the cells it contains.
            _write_manifest(out_dir, manifest)
    return manifest


def _write_manifest(out_dir, manifest):
    with open(os.path.join(out_dir, "MANIFEST.json"), "w") as fh:
        json.dump({"suite": manifest, "generator": "experiments.rsppbench.generate",
                   "version": "2.0"}, fh, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rsppbench")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse instance files already on disk "
                         "(they are pure functions of their seed)")
    ap.add_argument("--no-scale", action="store_true")
    a = ap.parse_args()
    print(f"Building RSPP-Bench into {a.out}")
    build_suite(a.out, a.seeds, include_scale=not a.no_scale,
                skip_existing=a.skip_existing)
    print("done")
