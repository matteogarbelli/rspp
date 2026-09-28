"""Load the campaign results database into tidy frames."""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd


def _open(path):
    """Open a results database, transparently handling gzip.

    The uncompressed campaign database is ~75 MB, which is awkward to version;
    the gzipped form is a fifth of that and is what ships with the release.
    """
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    return open(path)


# Arms whose per-generation history any figure actually reads.  The mechanism
# measurement needs the corridor and the weighted arm; the convergence plot adds
# the two objective-only arms.  Nothing reads the history of the rest.
_HISTORY_ARMS = {"sim_lambda", "corridor", "cost_only", "sim_obj"}


def _summarise_history(r):
    """Reduce a run's per-generation log to the scalars any figure reads.

    Attached before the history is dropped, so that the corridor trajectory of
    a sensitivity run remains reportable without holding 300 generations of
    logs for 12,000 runs in memory.
    """
    hist = r.get("history") or []
    if not hist:
        return
    taus = [h["tau"] for h in hist if h.get("tau") is not None]
    sched = [h["sched_active"] for h in hist if h.get("sched_active") is not None]
    cfeas = [h["corridor_feasible"] for h in hist
             if h.get("corridor_feasible") is not None]
    if taus:
        r["tau_final"] = taus[-1]
        r["tau_min"] = min(taus)
    if sched:
        r["sched_frac"] = sum(sched) / len(sched)
    if cfeas:
        r["corr_feas_final"] = cfeas[-1]


def _prune(r):
    """Drop payload no figure reads, in place, before the row is retained.

    The full campaign is ~275 MB of JSON Lines and expands by roughly an order
    of magnitude as Python objects, which does not fit in memory alongside the
    frame built from it.  Almost all of that bulk is per-generation history and
    returned fronts on rows that are only ever consulted through their summary
    columns, so it is cheaper to discard it at parse time than to hold it.
    """
    _summarise_history(r)
    if r.get("study") != "main":
        # Sensitivity, budget, scale and timing studies are read only through
        # the tidy frame: knee objectives, runtime, feasibility, tau0 and the
        # corridor-trajectory summary attached just above.
        for k in ("history", "front", "routes"):
            r.pop(k, None)
        return
    if r.get("arm") not in _HISTORY_ARMS:
        r.pop("history", None)
    if r.get("seed") != 0:
        r.pop("routes", None)          # only the map figure reads geometry


def load(path="experiments/results/campaign.jsonl", prune=True):
    rows, errors = [], []
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                errors.append(r)
                continue
            if prune:
                _prune(r)
            rows.append(r)
    return rows, errors


def to_frame(rows):
    recs = []
    for r in rows:
        f1, f2, f3 = r["objectives"]
        hist = r.get("history", [])
        def hmean(key):
            vals = [h[key] for h in hist if h.get(key) is not None]
            return float(np.mean(vals)) if vals else np.nan
        front = r.get("front") or []
        # The cheapest plan this run could have returned.  The knee is a
        # compromise across three objectives, so comparing knee distances
        # conflates the mechanism with the decision rule; the front minimum is
        # what the arm actually achieved on distance.
        f1_front_min = float(min(p[0] for p in front)) if front else np.nan
        alt = r.get("knee_alt")
        recs.append(dict(
            study=r["study"], instance=r["instance"], arm=r["arm"], seed=r["seed"],
            n_customers=r.get("n_customers"), n_vehicles=r.get("n_vehicles"),
            family=r.get("family"),
            f1=f1, f2=f2, f3=f3, violation=r["violation"], feasible=r["feasible"],
            f1_front_min=f1_front_min,
            alt_f1=alt[0] if alt else np.nan, alt_f3=alt[2] if alt else np.nan,
            runtime=r.get("runtime", np.nan),
            cpu_time=r.get("cpu_time", np.nan),
            n_evals=r.get("n_evals", np.nan), n_active=r.get("n_active", np.nan),
            n_front=len(front),
            tau0=r.get("tau0"), diam=r.get("diam"),
            tau_final=r.get("tau_final"), tau_min=r.get("tau_min"),
            sched_frac=r.get("sched_frac"),
            corr_feas_final=r.get("corr_feas_final"),
            rank_change=hmean("rank_change"),
            sweep_rank_change=hmean("sweep_rank_change"),
            flip_treat=hmean("flip_treat"), flip_noise=hmean("flip_noise"),
            dev_frechet=(r.get("deviation_all") or {}).get("frechet"),
            dev_dtw=(r.get("deviation_all") or {}).get("dtw"),
            dev_hausdorff=(r.get("deviation_all") or {}).get("hausdorff"),
            dev_lcss=(r.get("deviation_all") or {}).get("lcss"),
        ))
    return pd.DataFrame(recs)


def fronts_by_instance(rows, study="main"):
    out = {}
    for r in rows:
        if r["study"] != study:
            continue
        out.setdefault(r["instance"], []).append(r)
    return out
