# Synthetic instances and experiment records

## Instances

Each instance contains `name`, `base`, `disrupted`, `baseline_routes`,
`disruption` and `seed`. Vertex 0 is the depot. Positive integers identify
requests; route sequences are indexed by vehicle identity. Coordinates and
distance are in metres, durations in seconds, and speed in metres per second.
`data/rsppbench/MANIFEST.json` records each file's SHA-256 digest and factors.
`scale_only` identifies the three oversized pairs outside the 27 main instances.
The main pairs share nine baselines across their disruption families.

## Reported archive

`experiments/results/campaign.jsonl.gz` is gzip-compressed JSON Lines. The
unique cell key is `(study, instance, arm, seed)`. Its 12,735 records consist of:

| Study | Records | Settings |
|---|---:|---|
| `main` | 9,720 | 27 instances, 12 methods, 30 seeds |
| `metricv2:{frechet,dtw,hausdorff,lcss}` | 1,080 | Nine-instance subset, four measures, 30 seeds |
| `init:random` | 270 | Random initialisation on the subset |
| `resample:{100,500,1000}` | 810 | Subdivision steps in metres on the subset |
| `budget:{150,600}` | 540 | Generation budgets on the subset |
| `scale` | 270 | Three oversized instances, cost-only, summed deviation and tightening bound |
| `timing` | 45 | Nine-instance subset, five seeds, one worker |

Sensitivity and timing use `sum_lambda` at c=1. Greedy repetitions are
deterministic. The archive is an exact extraction of the reported study cells
from the completed production database; the extraction and source hashes are
recorded in `archive_manifest.json`. Earlier exploratory studies and variants
are not included in this reported archive.

| Field | Meaning |
|---|---|
| `study`, `instance`, `arm`, `seed` | Unique cell key |
| `family`, `n_customers`, `n_vehicles` | Disruption family (`INS`, `WDR`, `SRG`), disrupted request count and fleet size |
| `objectives` | Selected plan: distance (m), active-route duration spread (s), cumulative deviation (m) |
| `objectives_raw` | Three-objective compromise selected before relabelling; it can be a different plan |
| `n_relabelled` | Number of final-population members whose deviation decreases after relabelling |
| `violation`, `feasible` | Hard routing violation and flag (violation <= 1e-12); these fields do not certify a geometric bound |
| `front` | Hard-feasible non-dominated final vectors, with duplicates retained; the greedy heuristic also stores its single vector when infeasible, which the analysis treats as an empty front |
| `routes` | Selected routes, stored for every main-study run |
| `knee_alt` | Historical name for the alternative minimax choice using the objectives optimized by that method |
| `deviation_all` | Selected plan measured under the four measures at the run's subdivision step |
| `tau0`, `diam` | Initial bound width and disrupted-instance diameter (m), where applicable |
| `config` | Population size, generation budget, measure, schedule, initialisation, subdivision step, reference directions and fixed preference factor c (stored as `lambda_scale`); remaining defaults are in `algorithm.py` |
| `n_evals`, `stop` | Search evaluation count and stopping rule |
| `n_active` | Selected routes serving requests on available vehicles |
| `runtime`, `wall`, `cpu_time` | Machine- and load-dependent timings |
| `history` | Generation diagnostics listed below |

Older comparison configurations omit the unused `lambda_scale` field. It is
explicit in every summed-arm record. Greedy configurations contain only the
measure and subdivision step.

The default compromise minimizes the maximum of the three min-max-normalized
objectives over hard-feasible final members. Ties follow population order. This
rule guarantees weak efficiency within that population. It can select a
dominated tied vector. Objective summaries condition on hard feasibility.

## History

| Field | Meaning |
|---|---|
| `gen` | Generation |
| `lam` | Weight used in that generation; for summed arms this is the actual exchange rate rho_t=c*lambda(t), already including the fixed preference factor c |
| `tau`, `sched_active` | Width and whether its scheduled term is binding |
| `S_med`, `S_q10`, `S_q90`, `dist_med` | Population deviation median/quantiles and distance median (m) |
| `corridor_feasible`, `outside_corridor` | Fractions inside and outside the geometric width |
| `feasible_frac`, `n_active_med` | Hard-feasible fraction and active-vehicle median |
| `rank_change` | Fraction of pool front indices changed by toggling the deviation term |
| `sweep_rank_change` | Largest rank-change fraction in the eleven-weight sweep |
| `flip_treat`, `flip_noise` | Survivor-set diagnostics under fresh normalisation states; not used for main claims |

## Analysis conventions

Only hard-feasible fronts determine normalization and the pooled reference.
Failed runs contribute zero hypervolume, spacing and vector counts, and infinite
IGD+ and additive epsilon. The table's IGD+ and epsilon medians use finite values;
its other front-quality medians include failures. Duplicate vectors are removed
after normalization and rounding to nine decimals. Wilcoxon tests use matched
feasible seeds within instances. A12 is oriented as the comparison method's
probability of larger deviation than summed deviation at c=1, counting ties
by half; values above 0.5 favor the latter.

`environment.json` records the actual campaign environments and source digests.
Its seven launch records retain the production arguments, dates and source
hashes, with the executable path made repository-relative. The JSON object
`yardstick_provenance.json` contains two historical records in its `launches`
array, with the same path treatment. The source used separate JSON lines;
`provenance.json` records that format conversion and both source and public
file hashes. These transformations affect metadata only.
`yardstick.jsonl` stores distance-only OR-Tools references, which impose active
routes and rounded duration constraints and are not optimality certificates.
`selftest.json` preserves the campaign-time test output; `true_m` is a fine
discrete reference rather than an exact continuous Fréchet distance.

## Final analysis outputs

`frechet_coupling.npz` stores the baseline and revised sample coordinates,
Euclidean distance matrix in metres, optimal monotone coupling and bottleneck
pair. The companion JSON specifies instance, seed, driver and kernel error.
`uncertainty_bootstrap.npz` stores the 10,000 resampled statistics, point
estimates and percentile intervals. Its JSON specifies block construction,
seed and the fixed-suite interpretation. These files are regenerated in
`outputs/reanalysis/experiments/figures/`; the archive inputs are unchanged.
The generated confidence-interval table is checked against
`reference/tab_uncertainty.tex` by `make figures`.
