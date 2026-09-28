# Synthetic inputs and result records

Each `data/rsppbench/*.json` contains `name`, `base`, `disrupted`,
`baseline_routes`, `disruption` and `seed`. Vertex 0 is the depot; positive
integer vertices identify requests. Coordinates and travel distances are in
metres, service and route durations in seconds, and speed in metres/second.
Routes list customer indices in visit order and are indexed by vehicle identity.
The instance manifest stores a SHA-256 digest, layout, baseline size, fleet size,
family, seed and disruption description for each file. `scale_only` distinguishes
the three oversized cases from the 27 main cases.

`experiments/results/campaign.jsonl.gz` is gzip-compressed JSON Lines. Its unique
cell key is `(study, instance, arm, seed)`. The 12,195 rows include 7,290 main
records, 270 records for each of 18 sensitivity/scale configurations and 45
single-worker timing records. Main records comprise 27 instances × 9 arms × 30
seeds; deterministic greedy repetitions return the same solution.

| Field | Meaning |
|---|---|
| `objectives` | Selected plan's distance (m), active-route duration spread (s), deviation (m) |
| `violation`, `feasible` | Aggregate hard violation and hard-feasibility flag |
| `front` | Hard-feasible non-dominated objective vectors; duplicates may remain |
| `routes` | Selected route lists, retained for seed 0 of main-study cells |
| `knee_alt` | Legacy field name for the alternative per-arm minimax choice |
| `config` | Effective run settings; defaults are declared in `algorithm.py` |
| `history` | Logged population, corridor and selection diagnostics |
| `n_evals`, `stop` | Evaluation count and stopping criterion |
| `n_active` | Number of available vehicles with non-empty selected routes |
| `runtime`, `wall`, `cpu_time` | Timing measurements; machine/load dependent |

Some fields apply only to particular arms. The analysis conditions objective
summaries on hard feasibility. The default compromise minimises the largest
min–max-normalised objective over hard-feasible final candidates, using all
three objectives for every arm. It is not necessarily a geometric knee.

`environment.json` describes the historical campaign machine and dependency
versions by study. It contains hardware specifications, not a host/user name.
`selftest.json` preserves the historical numerical checks; its `true_m` value is
a fine discrete reference with a stated approximation error, not an exact
continuous Fréchet distance.
