"""Small independent checks driven by validation_manifest.json.

No production source, result or compiled kernel is modified. Fine discrete
references are used as intervals, never as exact continuous distances.
"""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from experiments.rspp import kernels, moea, selftest
from experiments.rspp.algorithm import relabel, lambda_weight, _asf_pick, DEFAULTS
from experiments.rspp.geometry import resample, diameter
from experiments.rspp.instance import Instance, RSPPPair
from experiments.rspp.solution import Evaluator
from experiments.analysis import indicators as ind

CFG=json.loads((HERE/'validation_manifest.json').read_text())
RNG=np.random.default_rng(CFG['seed'])
ATOL=CFG['absolute_tolerance']
OUT=ROOT/'outputs/formalism'; OUT.mkdir(parents=True,exist_ok=True)
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN=STAMP+'-'+hashlib.sha256((HERE/'validation_manifest.json').read_bytes()).hexdigest()[:8]
REPORT={}
FIELDS=['run_id','timestamp','git_commit','claim_ref','parameters','metrics','tolerance','passed','seed','n_repeats','runtime_s','artifact']
def revision():
    head=ROOT/'.git/HEAD'
    if not head.exists():return 'unavailable'
    h=head.read_text().strip()
    if h.startswith('ref: '):
        p=ROOT/'.git'/h[5:]
        return p.read_text().strip() if p.exists() else h
    return h

def record(name, start, metrics, passed, repeats):
    REPORT[name]={'passed':bool(passed),**metrics}
    log=OUT/'runs.csv'; new=not log.exists()
    with log.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if new:w.writeheader()
        w.writerow(dict(run_id=RUN,timestamp=STAMP,git_commit=revision(),claim_ref=name,
                        parameters=json.dumps(CFG,sort_keys=True),metrics=json.dumps(metrics),
                        tolerance=ATOL,passed=bool(passed),seed=CFG['seed'],n_repeats=repeats,
                        runtime_s=round(time.perf_counter()-start,4),artifact=RUN+'.json'))
    (OUT/(RUN+'.json')).write_text(json.dumps(REPORT,indent=2)+'\n')
    print(name,'PASS' if passed else 'FAIL',json.dumps(metrics),flush=True)

# Existing checks that do not write their legacy production self-test file.
t=time.perf_counter()
w=selftest.check_kernels(); ok=selftest.check_metric_axioms()
sp=selftest.check_vertex_preservation()
record('existing_lightweight_selftests',t,dict(kernel_error=w,max_spacing=sp),ok and w<ATOL,20)

t=time.perf_counter(); worst={k:0. for k in ('frechet','dtw','hausdorff','lcss')}
for _ in range(CFG['kernel_trials']):
    P=RNG.normal(0,100,(int(RNG.integers(1,25)),2))
    Q=RNG.normal(0,100,(int(RNG.integers(1,25)),2))
    for k in worst:
        a=kernels.METRICS[k](P,Q)
        b=kernels._lcss_numpy(P,Q,250.,25) if k=='lcss' else kernels.PY_REFERENCE[k](P,Q)
        worst[k]=max(worst[k],abs(a-b))
record('four_kernel_reference_comparisons',t,worst,max(worst.values())<ATOL,CFG['kernel_trials'])

t=time.perf_counter(); rank_bad=rate_bad=quant_bad=outside_bad=0
for _ in range(CFG['algebra_trials']):
    F=RNG.integers(0,100,(24,3)).astype(float)
    V=RNG.choice([0.,0.,0.1,0.4],24)
    r=moea.front_index(F,V)
    for a in (0.125,2.,128.):
        H=F.copy();H[:,2]*=a
        rank_bad+=not np.array_equal(r,moea.front_index(H,V))
        G=np.column_stack([F[:,0]+a*F[:,2],F[:,1]])
        for i in range(len(F)):
            for j in range(len(F)):
                if V[i] or V[j] or i==j:continue
                lhs=F[i,0]-F[j,0];rhs=a*(F[j,2]-F[i,2])
                expect=F[i,1]<=F[j,1] and (lhs<rhs or (lhs==rhs and F[i,1]<F[j,1]))
                rate_bad+=moea.dominates(G[i],G[j])!=expect
                if moea.dominates(F[i],F[j]):rate_bad+=not moea.dominates(G[i],G[j])
    n=int(RNG.integers(1,100));S=RNG.integers(0,20,n)
    for q in (0.,0.01,0.1,0.5,1.,float(RNG.random())):
        quant_bad+=np.sum(S<=np.quantile(S,q))<int(np.floor(q*(n-1)))+1
    # One feasible corridor member, then a strictly increasing S order.
    S=np.arange(10,dtype=float)*10; FF=np.column_stack([RNG.random(10),RNG.random(10),S])
    VV=np.maximum(0.,S-5)/5
    selected=moea.nsga3_survivors(FF,VV,5,moea.uniform_reference_points(3,2),moea.Nsga3State(3),RNG)
    outside_bad+=set(selected)!=set(range(5))
record('dominance_exchange_quantile_outside',t,dict(rank_errors=int(rank_bad),exchange_errors=int(rate_bad),quantile_errors=int(quant_bad),outside_errors=int(outside_bad)),rank_bad+rate_bad+quant_bad+outside_bad==0,CFG['algebra_trials'])

# The exact continuous distance for collinear out-and-back curves is |a-b|.
t=time.perf_counter(); worst_bound=0.; bad=0; compare_bad=0
for _ in range(CFG['resampling_trials']):
    S=[]; exact=[]
    for plan in range(2):
        ds=[];es=[]
        for k in range(3):
            a,b=RNG.uniform(10,100,2)
            P=np.array([[0.,0.],[a,0.],[0.,0.]])
            Q=np.array([[0.,0.],[b,0.],[0.,0.]])
            h=float(RNG.choice(CFG['subdivision_steps']))
            d=kernels.frechet(resample(P,h),resample(Q,h))
            e=abs(a-b);ds.append(d);es.append(e)
            bad+=not (-ATOL<=d-e<=h+ATOL)
            worst_bound=max(worst_bound,(d-e)/h)
        S.append(sum(ds));exact.append(sum(es))
    # Use the largest allowed step to bound the summed errors for both plans.
    h=max(CFG['subdivision_steps']);rho=float(RNG.uniform(0.01,100.))
    error=rho*((S[0]-S[1])-(exact[0]-exact[1]))
    compare_bad+=abs(error)>rho*3*h+ATOL
record('known_continuous_resampling_and_penalty_bound',t,dict(bound_errors=int(bad),penalty_comparison_errors=int(compare_bad),max_fraction_of_h=float(worst_bound)),bad+compare_bad==0,CFG['resampling_trials'])

t=time.perf_counter(); overlap_bad=0; smallest_gap=float('inf')
for _ in range(CFG['resampling_trials']):
    P=RNG.normal(0,30,(5,2));Q=RNG.normal(0,30,(6,2))
    href=CFG['reference_step'];fine=kernels.frechet(resample(P,href),resample(Q,href))
    for h in CFG['subdivision_steps']:
        coarse=kernels.frechet(resample(P,h),resample(Q,h))
        low=max(0.,fine-href,coarse-h);hi=min(fine,coarse)
        overlap_bad+=low>hi+ATOL
        smallest_gap=min(smallest_gap,coarse-fine)
record('reference_intervals_without_monotonicity_assumption',t,dict(interval_failures=int(overlap_bad),min_coarse_minus_fine=float(smallest_gap),reference_step=CFG['reference_step']),overlap_bad==0,CFG['resampling_trials'])

t=time.perf_counter(); worst_s=worst_l=0.
for _ in range(CFG['geometric_trials']):
    P=RNG.normal(0,100,(7,2));Q=RNG.normal(0,100,(6,2));a=float(RNG.uniform(.1,10))
    R=np.array([[0.,-1.],[1.,0.]]);b=RNG.normal(0,10,2)
    d=kernels.frechet(resample(P,20),resample(Q,20))
    ds=kernels.frechet(resample(a*P@R+b,a*20),resample(a*Q@R+b,a*20))
    worst_s=max(worst_s,abs(ds-a*d))
    vals=RNG.uniform(0,100,12);D=diameter(P)
    l=lambda_weight(vals,D,DEFAULTS);ls=lambda_weight(a*vals,a*D,DEFAULTS)
    worst_l=max(worst_l,abs(ls-l))
record('fixed_population_scale_equivariance',t,dict(distance_error=worst_s,weight_error=worst_l),max(worst_s,worst_l)<ATOL,CFG['geometric_trials'])

# Exhaustive assignment/reversal oracle includes a withdrawn and a baseline-idle driver.
t=time.perf_counter(); worst_s=worst_pres=0.;increases=0
for _ in range(CFG['relabel_trials']):
    coords=np.vstack([np.zeros((1,2)),RNG.normal(0,100,(7,2))])
    base=Instance('audit',coords,np.r_[0.,np.ones(7)],np.r_[0.,RNG.integers(1,5,7)],100,4,10000)
    dis=Instance('audit',coords,base.demand,base.service,100,4,10000,available=[1,1,0,1])
    pair=RSPPPair('audit',base,dis,[[1,2],[3,4],[5,6,7],[]],{},CFG['seed'])
    ev=Evaluator(pair,resample_step=20)
    perm=RNG.permutation(np.arange(1,8)).tolist();cuts=sorted(RNG.integers(0,8,2))
    plan=[perm[:cuts[0]],perm[cuts[0]:cuts[1]],perm[cuts[1]:]]
    routes=[plan[0],plan[1],[],plan[2]];old=ev.evaluate(routes);out=relabel(routes,ev,dis);new=ev.evaluate(out)
    optimum=float('inf')
    for pi in itertools.permutations(range(3)):
        for flip in itertools.product((False,True),repeat=3):
            rr=[[],[],[],[]]
            for k,j,o in zip((0,1,3),pi,flip):rr[k]=plan[j][::-1] if o else plan[j]
            optimum=min(optimum,ev.evaluate(rr)[2])
    worst_s=max(worst_s,abs(new[2]-optimum))
    worst_pres=max(worst_pres,max(abs(old[j]-new[j]) for j in (0,1,3)))
    increases+=new[2]>old[2]+ATOL
record('relabel_against_exhaustive_oracle',t,dict(optimality_error=worst_s,preserved_objective_violation_error=worst_pres,increases=int(increases)),worst_s<ATOL and worst_pres<ATOL and increases==0,CFG['relabel_trials'])

# Regression: an infeasible heuristic point cannot change reference or bounds.
t=time.perf_counter()
valid=[dict(arm='valid',feasible=True,front=[[2.,3.,4.],[3.,2.,4.]])]
bad=[dict(arm='greedy',feasible=False,front=[[0.,0.,0.]])]
one=ind.instance_front_indicators(valid);two=ind.instance_front_indicators(valid+bad)
empty=two['greedy'][0]
preserved=one['valid']==two['valid']
a=np.full(30,100000.1);b=np.full(30,100000.)
w=ind.wilcoxon_pair(a,b)
F=np.array([[2.,2.,2.],[2.,1.,1.],[1.,3.,3.]])
pick=_asf_pick(F,np.ones(3,dtype=bool))
weak_only=pick==0 and moea.dominates(F[1],F[pick])
record('feasibility_statistics_and_choice_regressions',t,dict(valid_indicators_unchanged=preserved,infeasible_hv=empty['hv'],infeasible_vector_count=empty['n_front'],infeasible_igd_infinite=bool(np.isinf(empty['igdp'])),close_but_distinct_a12=w['a12'],close_but_distinct_p=w['p'],minimax_can_choose_dominated_point=bool(weak_only)),preserved and empty['hv']==0 and empty['n_front']==0 and np.isinf(empty['igdp']) and w['a12']==1. and w['p']<.05 and weak_only,1)

assert all(v['passed'] for v in REPORT.values()),'See validation report for failed checks.'
print('Report:',OUT/(RUN+'.json'))
