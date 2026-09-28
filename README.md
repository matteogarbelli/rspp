# RSPP reproducibility package

Research materials for *The Route Similarity Preservation Problem: an adaptive
geometric corridor for vehicle routing replanning*, Matteo Garbelli.

This folder is prepared for a separate public repository. It contains the
standalone synthetic research implementation, generator, 30 instance files,
instance manifest, 12,195 archived result records and analysis scripts. It
contains no files copied from the HPA repositories, industrial datasets,
deployment configuration, authentication utilities or private repository history.
The research implementation runs independently of those systems. An older isolated
selection diagnostic depended on HPA and is excluded, together with its results.
The shortened manuscript uses the reproducible in-campaign weight measurements.

## Quick start

Use Python 3.11, a C compiler and Make (or invoke `python reproduce.py` directly).
The original campaign used Python 3.11.6. Dependency installation requires
internet access; the research commands run locally after installation.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make verify
make smoke
make replay
make figures
```

`verify` checks file integrity, all four numerical kernels, mathematical
invariants, campaign completeness and all 243 stored main-study plans. It also
reproduces the headline statistics and the documented normalisation-state
counterexample. `smoke` runs five generations on one instance; it is an execution
check, not a reproduction of a production result. `figures` regenerates the
figures, tables and numerical macros from the archived campaign and compares
them with the bundled reference values. The bibliography count and excluded historical diagnostic are outside this comparison.
`replay` reruns one archived main-study cell at the original 300-generation
budget and checks its objectives, front, selected routes and feasibility.

Every wrapper command reads `reproduction_manifest.json`, writes to `outputs/`
and appends its outcome to `outputs/runs.jsonl`. The archived inputs remain
unchanged. A checksum inventory is in `FILES.sha256.json`; source-copy hashes
are recorded in `provenance.json`.

## Recompute the experiments

```sh
make bench       # regenerate all 30 instances and compare their SHA-256 hashes
make campaign    # full production settings, 30 seeds, 7 workers
make timing      # separate single-worker timing study; use an idle machine
make selftest    # additional geometric checks, written outside the archive
```

The complete campaign is substantially more expensive than archive reanalysis.
Its results and machine snapshot go to `outputs/rerun/`. Repeating a campaign
command resumes completed cells. Run `campaign` and `timing` sequentially.
Wall-clock measurements depend on hardware and load. Matching seeds and versions
improve numerical reproducibility; bitwise agreement across platforms is not
guaranteed. `make bench` checks deterministic instance bytes on the local setup.

## Contents

| Path | Contents |
|---|---|
| `experiments/rspp/` | Reference solver, geometry, routing operators and C kernels |
| `experiments/rsppbench/` | Seeded synthetic instance generator |
| `experiments/campaign/` | Main study and sensitivity campaign definitions |
| `experiments/analysis/` | Statistics, figures, tables and LaTeX macros |
| `data/rsppbench/` | 27 main cases, 3 scale cases and their manifest |
| `experiments/results/` | Compressed results and archived environment/self-test metadata |
| `reference/` | Expected numerical macros and tables |
| `checks/` | Independent validation and its manifest |

The 27 main cases share nine baseline problems across three disruption families.
Main study arms comprise greedy repair, cost-only search, a deviation objective,
adaptive weighting, a repair-derived static bound, and four corridor variants.
The complete archive includes the floor-quantile and timing studies.

## Interpretation

The research solver reproduces the implementation used for the reported
experiments. In particular, it retains the adaptive-weight normalisation state
and repair behaviour discussed in the manuscript. These are documented
limitations, not changes made after the campaign. The legacy self-test uses fine
discrete references; its field `true_m` denotes an approximation to continuous
Fréchet distance. Use `make selftest` for correctly qualified output.

Releasing these materials makes the experiments inspectable and rerunnable. It
does not resolve the shared-baseline dependence, comparator limitations or
absence of road-network and mid-shift validation discussed in the paper.

## Release status

Prepared locally; no public URL or DOI has yet been assigned. Licensing is pending
the author's choice. Industrial HPA code remains outside this package. See
`RELEASE_STATUS.md` for checks and the remaining publication steps.
