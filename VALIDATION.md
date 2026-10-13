# Validation record

## Author annotation revision — 8 October 2026

The manuscript now focuses on geometric continuity after a disruption and its
travel-distance and workload costs. The coupling heatmap is excluded from the
article, and ancillary lemmas, geographic scaling claims and explicit research
questions have been removed. The analysis code, production archive and all
556 experimental point estimates are unchanged. Ten existing external figures
remain in the manuscript; the verified coupling figure is retained as an
optional analysis diagnostic. Five generated tables, including the bootstrap
interval table, remain unchanged. The validation records below describe the
versions checked at their respective stages; the eleven-figure check includes
the now optional coupling diagnostic.


## Final editorial revision — 8 October 2026

The final analysis adds an explanatory discrete Fréchet coupling heatmap and
paired-seed bootstrap intervals for the headline trade-offs. The optimisation
implementation and all production records are unchanged. The preceding
validation rounds are retained below as historical evidence.

An isolated copy of the 91-file inventory passed `reproduce.py verify` and
`reproduce.py figures` in the previously tested clean scientific environment.
The copy contained no Git history or compiled trajectory library. Verification
compiled its own kernel, checked all input hashes and 12,735 expected records,
and independently recomputed all 9,720 main-study plans. Reanalysis reproduced
all 556 existing experimental macros, all five generated tables and the driver
comparison. The bibliography count is excluded from numerical comparisons.

The coupling illustration uses C100-7v-INS8-s1, cost-only seed 0, the driver
with the largest deviation in that plan, at h=250 m. A full dynamic programme
and its monotone path agree with the production kernel within 1.819e-12 m
(tolerance 1e-9 m). This illustrates the metric; it is not additional search
performance evidence. Historical industrial population-fitness images are
not part of the current manuscript.

The bootstrap uses 10,000 replicates and seed 20261008. One seed block contains
all 27 instances and all four arms, preserving matching and any dependence
induced by seed reuse. Each replicate recomputes the original median-of-ratios
statistic. The intervals describe stochastic-run variability conditional on
the fixed suite; new baselines and operating regions are outside their scope.
At c=1,4,16 the deviation-reduction intervals are [5.2,11.4], [12.2,19.3] and
[17.8,26.1] percent. They do not establish the benefit of population adaptation
over a fixed exchange rate. That comparison and dedicated modern routing
baselines remain absent.

Private/public comparison verifies decoded pixels for all eleven external
manuscript figures and exact bootstrap statistics. Command times were 26.589 s
for verification and 48.157 s for reanalysis. After execution, the validation record, data dictionary and reference
bibliography count were updated, with corresponding provenance and inventory
hashes. Scientific code, data and numerical reference results are unchanged.

## Previous clean-install validation round

## Clean publication check — 8 October 2026

An isolated copy of the 88-file release inventory was tested without Git,
previous outputs or compiled kernels. A fresh Python 3.11.6 virtual environment
installed the pinned requirements from public PyPI, with local package-index
settings disabled. All commands completed successfully:

| Check | Result |
|---|---|
| `make verify` | 88 file hashes, all 30 instances and baseline feasibility, all 12,735 expected cells, mathematical checks and all 9,720 stored main plans pass; maximum objective/violation error 1.74623e-10, tolerance 1e-9 |
| `make replay` | Summed deviation c=1, C50-4v-INS5-s1, seed 0, 300 generations: zero objective error; exact routes, front, feasibility and evaluation count |
| `make figures` | All 556 experimental macros, four tables and the driver-comparison CSV match their references |
| Figure comparison | All ten manuscript PNGs have identical decoded pixels to the reviewed figures; compressed bytes differ with the fresh image-library installation |
| Driver outputs | Both CSVs and the summary JSON are byte-identical to the manuscript analysis |
| Publication scan | No credential patterns, industrial endpoints or local filesystem paths in the release inventory; no credential or industrial-endpoint patterns in the repository's single existing commit |

Measured runtimes were 32.184 s for verification, 27.044 s for replay and
76.333 s for reanalysis on the same production machine. The test installed
the four pinned scientific packages and their public dependencies; no private
package was required. Instance generation was not repeated because the
generator sources and instance hashes are unchanged.

The production archive remains unchanged. Two metadata files now replace
absolute executable paths with repository-relative paths. Historical
distance-reference provenance is a valid JSON object with a two-launch array.
`provenance.json` retains original source hashes, public hashes and the exact
transformation identifiers. These changes do not affect the numerical code,
launch settings or experiment records.

The companion repository is public, but the reviewed October revision remains
a local preparation until committed and pushed by the author. Citation metadata
identifies both MIT and CC BY 4.0 and omits an unpublished release date.

## Driver diagnostic update — 8 October 2026

The publication analysis now recomputes each driver's Fréchet term for the
3,240 returned main-study plans from cost-only and summed-deviation NSGA-III
at c=1,4,16. Each arm contains 27 instances and 30 feasible seeds. The maximum
disagreement between the recomputed sum and its archived objective is
1.45519e-11 m, below the declared 1e-6 m post-processing tolerance.

The standalone `make figures` passed after this addition: all 556 experimental
macros, four tables and the reference driver-comparison CSV match. All ten
manuscript PNGs are byte-identical to the private manuscript analysis. The
current release inventory contains 88 files and is checked separately after
documentation edits. Numerical source hashes for the optimisation, instance
generator, geometry and trajectory kernels remain unchanged.

The seven new macros describe maximum-driver deviation and sample counts;
all 549 previous experimental macros and all four tables retain their values.
The summed model uses a fixed dimensionless preference factor c and the
adaptive exchange rate rho_t=c*lambda(t). It weights cumulative deviation and
does not impose a bound on any driver's deviation. The diagnostic is
descriptive and includes the campaign's final compromise selection and route
relabelling. No production optimisation was rerun for this update.

The broader numerical checks below record the 7 October review. Those checks
were not repeated for this analysis and documentation addition.

## Full review — 7 October 2026

Validated on 7 October 2026 on the production machine: Apple M2 (8 cores,
8 GB RAM), macOS 27.0.1, Python 3.11.6, NumPy 2.4.3, SciPy 1.16.1,
pandas 2.3.2 and Matplotlib 3.8.4. C kernels were compiled locally.

| Check | Command | Result |
|---|---|---|
| Release integrity | `make verify` | 85 files passed the initial audit; the final 86-file inventory also includes the checkout line-ending configuration and is verified separately |
| Instance integrity | `make verify` | All 30 hashes and baseline feasibility checks passed; nine distinct main baseline problems |
| Reported campaign | `make verify` | 12,735 records; exact expected cells; no duplicates, malformed rows, error rows or missing cells |
| Stored main-study plans | `make verify` | All 9,720 plans recomputed; maximum objective/violation error 1.74623e-10, tolerance 1e-9 |
| Four trajectory kernels | `make verify` | 100 C-versus-NumPy trials per kernel; maximum errors: Fréchet/Hausdorff 5.68434e-14, DTW 2.84217e-14, LCSS 0 |
| Dominance, exchange rate, quantile count and outside-bound ordering | `make verify` | Zero errors in 100 trials each |
| Continuous oracle and penalty bound | `make verify` | 100 collinear out-and-back curve pairs with known continuous distance; no bound failures; maximum error 0.397524 h |
| Sampling reference intervals | `make verify` | 100 trials, no interval failures; a coarse-minus-fine value of -0.002755 m confirms refinement need not be monotone |
| Fixed-population scale equivariance | `make verify` | Distance error 9.09495e-13; weight error 8.88178e-16 |
| Route relabelling | `make verify` | 100 exhaustive permutation/reversal oracles; zero assignment optimality error and no deviation increases; preserved-objective/violation error 2.27374e-13 |
| Statistical and front-reference regressions | `make verify` | Failed fronts leave feasible normalization/reference unchanged; close-but-distinct samples give A12=1 and p=4.32046e-8; dominated minimax tie reproduced |
| Normalization-state counterexample | `make verify` | Reproduced: maximum normalized difference 0.0687548 |
| Five-generation execution | `make smoke` | Hard-feasible, 552 evaluations |
| 300-generation replay | `make replay` | Summed deviation c=1, C50-4v-INS5-s1, seed 0: zero objective error; exact front, routes, feasibility and evaluation count |
| Full archive reanalysis | `make figures` | All 549 experimental macros and four tables match; bibliography count excluded; all nine manuscript PNGs byte-identical to the inspected figures |
| Legacy geometric self-test | `make selftest` | All checks pass; numerical values exactly match campaign-time selftest.json; fine-reference excess at most 0.308875 h |
| Completed compressed-archive restart | `reference/restart_regression.json` | Zero remaining jobs; no plain database created; archive and environment unchanged |

`outputs/runs.jsonl`, `outputs/audit/` and `outputs/formalism/` record these
local commands and outcomes. `reference/production_audit.json` records the
strict audit of all 19,080 raw production records and extraction of the
12,735 reported cells. The main-study audit found 95 selected minimax
compromises strictly dominated in their own stored front under exact float
comparison; minimax ties only guarantee weak efficiency, as stated in the
paper. The records were preserved.

## Source provenance

The summed optimisation implementation is unchanged since its 6–7 October
production launches. Earlier comparison results were retained after the
addition of the summed methods, which preserves the comparison branches.
After production, the self-test was corrected to use fine-reference
intervals, and the runner gained an early no-op return for a completed archive.
Before/after hashes are in `reference/post_campaign_source_changes.json`.
Analysis now excludes infeasible fronts from normalization, distinguishes
close-but-unequal statistical samples and prints small positive p-values with
adequate precision. A p-value previously displayed as 0.000 is now 0.0004;
the statistical calculation is unchanged by this display correction.

The full audit, smoke, replay and self-test passed before this final p-value
display adjustment. Full reanalysis passed afterwards. The final file
inventory is checked separately after documentation edits; repeating the
optimization checks for these presentation-only changes is unnecessary.

## Historical instance regeneration

All 30 instances were regenerated byte-for-byte on 28 September 2026
(`make bench`, approximately 31 minutes). Generator, geometry and kernel
sources are byte-identical to the previously validated release, and all
instance hashes were checked again in this review. The complete generation
run was not repeated.

## Scope

The production campaign was not rerun during this review. Recomputed plans,
independent property checks and one exact replay establish numerical
consistency, not global optimality or general operational performance.
Across-instance statistical tests are exploratory because disruptions share
baselines. Kernel random checks do not prove metric axioms. Fine discrete
references approximate continuous distance. The exact continuous oracle is
restricted to the test family stated above. Timings depend on platform and load.
