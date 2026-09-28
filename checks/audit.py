"""Reproduce review checks without changing the solver or production results.

Run from the repository root:
    python3 checks/audit.py
Inputs are fixed in the adjacent manifest. Results are append-only.
"""
import collections
import csv
import gzip
import hashlib
import json
import pathlib
import platform
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from experiments.rspp import kernels, moea, selftest
from experiments.rspp.algorithm import _asf_pick
from experiments.rspp.geometry import resample, resample_fixed
from experiments.rspp.instance import RSPPPair
from experiments.rspp.solution import Evaluator
from experiments.campaign.run import build_jobs

MANIFEST = pathlib.Path(__file__).resolve().parent / "manifest.json"
OUT = ROOT / "outputs/audit"
OUT.mkdir(parents=True, exist_ok=True)
CFG = json.loads(MANIFEST.read_text())
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN = STAMP + "-" + hashlib.sha256(MANIFEST.read_bytes()).hexdigest()[:8]
LOG = OUT / "runs.csv"
FIELDS = ["run_id", "timestamp", "git_commit", "claim_ref", "parameters", "metrics",
          "tolerance", "passed", "seed", "n_repeats", "runtime_s", "artifact"]
REPORT = {}

def revision():
    head = (ROOT / ".git/HEAD")
    if not head.exists():
        return "unavailable"
    h = head.read_text().strip()
    if not h.startswith("ref: "):
        return h
    ref = ROOT / ".git" / h[5:]
    return ref.read_text().strip() if ref.exists() else "unresolved ref: " + h[5:]

def record(name, start, metrics, passed=True, repeats=1):
    REPORT[name] = {"passed": bool(passed), **metrics}
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(dict(run_id=RUN, timestamp=STAMP, git_commit=revision(), claim_ref=name,
                        parameters=json.dumps(CFG, sort_keys=True), metrics=json.dumps(metrics),
                        tolerance=CFG["absolute_tolerance"], passed=bool(passed),
                        seed=CFG["seed"], n_repeats=repeats,
                        runtime_s=round(time.perf_counter()-start, 3), artifact=RUN+".json"))
    (OUT / (RUN+".json")).write_text(json.dumps(REPORT, indent=2))
    display = {k: (len(v) if isinstance(v, list) and len(v)>20 else v) for k,v in metrics.items()}
    print(name, "PASS" if passed else "REVIEW", json.dumps(display), flush=True)

t = time.perf_counter()
assert kernels._LIB is not None, "Compile kernels before this audit; avoid quadratic Python runs."
rng = np.random.default_rng(CFG["seed"])
worst = dict.fromkeys(("frechet", "dtw", "hausdorff", "lcss"), 0.0)
for _ in range(CFG["kernel_trials"]):
    p, q = (rng.normal(0, 500, (int(rng.integers(1, 30)), 2)) for k in range(2))
    for key in worst:
        if key == "lcss":
            a = kernels.lcss(p, q, eps=250.0, delta=10)
            b = kernels._lcss_numpy(p, q, 250.0, 10)
        else:
            a = kernels.METRICS[key](p, q)
            b = kernels.PY_REFERENCE[key](p, q)
        worst[key] = max(worst[key], abs(a-b))
record("all_four_kernels", t, worst, max(worst.values()) < CFG["absolute_tolerance"], CFG["kernel_trials"])

t = time.perf_counter()
rank_errors = quantile_errors = 0
for _ in range(CFG["invariance_trials"]):
    F = rng.integers(0, 10000, (30, 3)).astype(float)
    V = rng.choice([0.0, 0.0, 0.1, 0.3], 30)
    idx = moea.front_index(F, V)
    for scale in (0.125, 2.0, 128.0):
        scaled = F.copy(); scaled[:, 2] *= scale
        rank_errors += int(not np.array_equal(idx, moea.front_index(scaled, V)))
    n = int(rng.integers(1, 200)); q = float(rng.random())
    S = rng.integers(0, 1000, n)
    quantile_errors += int(np.sum(S <= np.quantile(S, q)) < int(np.floor(q*(n-1)))+1)
record("dominance_and_quantile_guarantees", t,
       dict(rank_errors=rank_errors, quantile_errors=quantile_errors), rank_errors+quantile_errors == 0,
       CFG["invariance_trials"])

t = time.perf_counter()
errors = []
for _ in range(CFG["geometric_trials"]):
    P = rng.normal(0, 100, (8, 2)); Q = rng.normal(0, 100, (7, 2))
    a = float(rng.uniform(0.2, 5)); shift = rng.normal(0, 10, 2)
    R = np.array([[0., -1.], [1., 0.]])
    d = kernels.frechet(resample(P, 40), resample(Q, 40))
    ds = kernels.frechet(resample(a*P@R+shift, a*40), resample(a*Q@R+shift, a*40))
    errors.append(abs(ds-a*d))
record("geometric_scale_equivariance", t, dict(max_abs_error=max(errors)), max(errors)<1e-9, len(errors))

t = time.perf_counter()
entries = json.loads((ROOT / CFG["instance_manifest"]).read_text())["suite"]
bad_hash, bad_base, base_ids = [], [], set()
for e in entries:
    path = ROOT / "data/rsppbench" / e["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != e["sha256"]:
        bad_hash.append(e["name"])
    pair = RSPPPair.load(path)
    base_pair = RSPPPair(pair.name, pair.base, pair.base, pair.baseline_routes, {}, pair.seed)
    v = Evaluator(base_pair).evaluate(pair.baseline_routes)[3]
    if v > 1e-12:
        bad_base.append([pair.name, v])
    if not e["scale_only"]:
        base_ids.add(hashlib.sha256(json.dumps(pair.base.to_dict(), sort_keys=True).encode()).hexdigest())
record("instance_hashes_and_baselines", t, dict(n_instances=len(entries), bad_hashes=bad_hash,
       infeasible_baselines=bad_base, unique_main_baselines=len(base_ids)), not bad_hash and not bad_base)

t = time.perf_counter()
counts = collections.Counter(); keys = set(); duplicates=[]; malformed=[]; rows=[]
recalc_err = 0.0; recalc_count=0; scheduled_mismatch=0
opener = gzip.open if CFG["database"].endswith(".gz") else open
with opener(ROOT / CFG["database"], "rt") as f:
    for line_no, line in enumerate(f, 1):
        try:
            r = json.loads(line)
        except ValueError:
            malformed.append(line_no); continue
        if "error" in r:
            malformed.append(line_no); continue
        key = (r["study"], r["instance"], r["arm"], r["seed"])
        if key in keys: duplicates.append(key)
        keys.add(key); counts[r["study"]]+=1
        if r["study"] == "main":
            front=np.asarray(r.get("front") or []).reshape(-1, 3)
            rows.append(dict(instance=r["instance"], arm=r["arm"], seed=r["seed"],
                             feasible=r["feasible"], f1=r["objectives"][0], f2=r["objectives"][1],
                             f3=r["objectives"][2], fmin=front[:,0].min() if len(front) else np.nan,
                             n_dist=len(np.unique(np.round(front, 6), axis=0)),
                             n_evals=r["n_evals"], tau0=r.get("tau0")))
            if r.get("routes") is not None:
                pair=RSPPPair.load(ROOT/"data/rsppbench"/(r["instance"]+".json"))
                vals=Evaluator(pair).evaluate(r["routes"])
                recalc_err=max(recalc_err, float(np.max(np.abs(np.array(vals[:4])-np.array(r["objectives"]+[r["violation"]])))))
                recalc_count+=1
df=pd.DataFrame(rows)
main=[e["name"] for e in entries if not e["scale_only"]]
scale=[e["name"] for e in entries if e["scale_only"]]
jobs=build_jobs("data/rsppbench",main,list(range(30)),
                ["main","metric","init","schedule","calib","resample","budget","scale","qmin"],scale)
jobs+=build_jobs("data/rsppbench",main,list(range(5)),["timing"],scale)
expected={(j[4],j[1],j[2],j[3]) for j in jobs}
record("campaign_integrity",t,dict(counts=dict(counts),total=sum(counts.values()),duplicates=duplicates,
       malformed=malformed,missing_cells=sorted(expected-keys),extra_cells=sorted(keys-expected),
       stored_plans_recomputed=recalc_count,max_objective_violation_error=recalc_err),
       not duplicates and not malformed and expected==keys and recalc_err<1e-9)

t=time.perf_counter()
feas=df[df.feasible]; med=feas.groupby(["instance","arm"])[["f1","f2","f3","fmin"]].median()
b=med.xs("cost_only",level="arm"); c=med.xs("corridor",level="arm")
metrics=dict(deviation_reduction_pct=float((100*(1-c.f3/b.f3)).median()),
             distance_vs_min_pct=float((100*(c.f1/b.fmin-1)).median()),
             workload_increase_pct=float((100*(c.f2/b.f2-1)).median()),
             corridor_feasible_pct=float(100*df[df.arm=="corridor"].feasible.mean()),
             distinct_vectors_raw_rounded_median=float(df[df.arm=="corridor"].n_dist.median()))
record("headline_results",t,metrics)

t=time.perf_counter()
# A stored ideal from old units is not scaled when the current objective is scaled.
F=np.array([[1.,8.,6.],[3.,3.,3.],[8.,1.,1.],[4.,4.,4.]])
s1=moea.Nsga3State(3); s1.ideal=np.array([0.,0.,0.5])
s2=moea.Nsga3State(3); s2.ideal=np.array([0.,0.,0.5])
G=F.copy(); G[:,2]*=10
norm_error=float(np.max(np.abs(moea._normalize(F,s1)-moea._normalize(G,s2))))
record("adaptive_weight_normalisation_state",t,dict(max_normalised_difference=norm_error,
       interpretation="Expected counterexample: retained ideal coordinates allow adaptive-weight normalisation effects."),
       norm_error>0)

if CFG["run_existing_selftests"]:
    for name, fn in [("existing_kernel_checks",selftest.check_kernels),
                     ("existing_metric_axioms",selftest.check_metric_axioms),
                     ("existing_vertex_preservation",selftest.check_vertex_preservation),
                     ("existing_resampling_reference_check",selftest.check_resampling_bound),
                     ("existing_uniform_resampling_comparison",selftest.check_naive_underreport)]:
        t=time.perf_counter()
        try:
            value=fn()
            record(name,t,dict(result=value, note="Fine discrete references are not exact continuous distances."))
        except Exception as e:
            record(name,t,dict(error=repr(e)),False)
print("Audit artifact:",OUT/(RUN+".json"),flush=True)

assert all(r["passed"] for r in REPORT.values()), "Validation failed; inspect the JSON report."
