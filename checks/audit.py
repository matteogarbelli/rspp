"""Validate every archived study cell and recompute all stored main plans."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.campaign.run import build_jobs
from experiments.rspp.instance import RSPPPair
from experiments.rspp.solution import Evaluator
from experiments.rspp import kernels,moea
CFG=json.loads((ROOT/'reproduction_manifest.json').read_text())
OUT=ROOT/'outputs/audit';OUT.mkdir(parents=True,exist_ok=True)
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
START=time.perf_counter();REPORT={}
assert kernels._LIB is not None,'Compile trajectory kernels first.'
subprocess.run([sys.executable,str(ROOT/'checks/validate_formalism.py')],cwd=ROOT,check=True)
suite=json.loads((ROOT/CFG['instance_manifest']).read_text())['suite']
main=[e['name'] for e in suite if not e['scale_only']];scale=[e['name'] for e in suite if e['scale_only']]
jobs=[]
for group in ['campaign','timing']:
    spec=CFG[group]
    jobs+=build_jobs('data/rsppbench',main,range(spec['seeds']),spec['studies'],scale,study_arm=spec['arm'])
expected={(j[4],j[1],j[2],j[3]) for j in jobs}
assert len(expected)==len(jobs)==CFG['expected_rows']
bad_hash=[];bad_base=[];baseline_ids=set();pairs={};evaluators={}
for e in suite:
    path=ROOT/'data/rsppbench'/e['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=e['sha256']:bad_hash.append(e['name'])
    pair=RSPPPair.load(path);pairs[e['name']]=pair
    bp=RSPPPair(pair.name,pair.base,pair.base,pair.baseline_routes,{},pair.seed)
    v=Evaluator(bp).evaluate(pair.baseline_routes)[3]
    if v>1e-12:bad_base.append([e['name'],v])
    if not e['scale_only']:
        baseline_ids.add(hashlib.sha256(json.dumps(pair.base.to_dict(),sort_keys=True).encode()).hexdigest())
REPORT['instances']=dict(passed=not bad_hash and not bad_base,count=len(suite),bad_hashes=bad_hash,infeasible_baselines=bad_base,unique_main_baselines=len(baseline_ids))

keys=set();dupes=[];malformed=[];bad=[];counts=Counter();recomputed=0;max_error=0.
with gzip.open(ROOT/CFG['archive'],'rt') as f:
    for line_no,line in enumerate(f,1):
        try:r=json.loads(line)
        except ValueError:malformed.append(line_no);continue
        if 'error' in r:malformed.append(line_no);continue
        key=(r['study'],r['instance'],r['arm'],r['seed'])
        if key in keys:dupes.append(key)
        keys.add(key);counts[r['study']]+=1
        vals=np.asarray(r['objectives'],float)
        if not np.isfinite(vals).all() or (vals < -1e-9).any():bad.append(['objectives',key])
        if r['feasible']!=(r['violation']<=1e-12):bad.append(['feasibility',key])
        if r['study']!='main':continue
        ne=1 if r['arm']=='greedy' else (27693 if r['arm']=='eps_static' else 27692)
        if r['n_evals']!=ne:bad.append(['evaluations',key])
        routes=r.get('routes')
        if routes is None:bad.append(['missing_main_plan',key]);continue
        pair=pairs[r['instance']]
        if sorted(i for route in routes for i in route)!=list(pair.disrupted.customers()):bad.append(['coverage',key])
        ek=(r['instance'],r['config'].get('metric','frechet'),r['config'].get('resample_step',250.))
        if ek not in evaluators:evaluators[ek]=Evaluator(pair,ek[1],ek[2],cache_size=1000)
        ev=evaluators[ek];ev._cache.clear()
        error=float(np.max(np.abs(np.asarray(ev.evaluate(routes)[:4])-np.r_[vals,r['violation']])))
        max_error=max(max_error,error);recomputed+=1
        if error>CFG['validation']['absolute_tolerance']:bad.append(['recomputation',key,error])
        if recomputed%1000==0:print('Recomputed',recomputed,'main-study plans',flush=True)
REPORT['campaign']=dict(passed=not dupes and not malformed and not bad and keys==expected,
    total=sum(counts.values()),counts=dict(counts),duplicate_cells=dupes,malformed_or_error_rows=malformed,
    invalid_records=bad,missing_cells=sorted(expected-keys),extra_cells=sorted(keys-expected),
    stored_main_plans_recomputed=recomputed,max_objective_violation_error=max_error)

# A running ideal point in unscaled units allows a separate objective weight
# to affect normalisation despite exact dominance invariance.
F=np.array([[1.,8.,6.],[3.,3.,3.],[8.,1.,1.],[4.,4.,4.]])
s1=moea.Nsga3State(3);s1.ideal=np.array([0.,0.,0.5])
s2=moea.Nsga3State(3);s2.ideal=np.array([0.,0.,0.5])
G=F.copy();G[:,2]*=10
error=float(np.max(np.abs(moea._normalize(F,s1)-moea._normalize(G,s2))))
REPORT['normalisation_state_counterexample']=dict(passed=error>0,max_normalised_difference=error)
REPORT['passed']=all(r['passed'] for r in REPORT.values())
REPORT['runtime_s']=time.perf_counter()-START
artifact=OUT/(STAMP+'.json');artifact.write_text(json.dumps(REPORT,indent=2)+'\n')
log=OUT/'runs.csv';new=not log.exists()
with log.open('a',newline='') as f:
    fields=['timestamp_utc','manifest_sha256','passed','recomputed_main_plans','max_error','runtime_s','artifact']
    w=csv.DictWriter(f,fieldnames=fields)
    if new:w.writeheader()
    w.writerow(dict(timestamp_utc=STAMP,manifest_sha256=hashlib.sha256((ROOT/'reproduction_manifest.json').read_bytes()).hexdigest(),
        passed=REPORT['passed'],recomputed_main_plans=recomputed,max_error=max_error,runtime_s=REPORT['runtime_s'],artifact=artifact.name))
print(json.dumps(REPORT,indent=2),flush=True)
assert REPORT['passed'],'Audit failed; inspect saved JSON.'
