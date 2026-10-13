"""Run the distance yardstick of ``rspp.yardstick`` on the comparison instances.

One deterministic run per instance.  Rows are appended as JSON Lines and a run
already present is skipped; the launch is recorded next to the output with the
same provenance fields as the campaign.
"""
from __future__ import annotations

import argparse
import json
import os

from ..rspp import yardstick
from ..rspp.instance import RSPPPair
from .run import _provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/rsppbench")
    ap.add_argument("--out", default="experiments/results/v2/yardstick.jsonl")
    a = ap.parse_args()

    with open(os.path.join(a.bench, "MANIFEST.json")) as fh:
        suite = json.load(fh)["suite"]
    instances = [m["name"] for m in suite if not m.get("scale_only")]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            done = {json.loads(line)["instance"] for line in fh if line.strip()}
    with open(os.path.join(os.path.dirname(a.out), "yardstick_provenance.json"), "a") as fh:
        fh.write(json.dumps(_provenance(["yardstick"], 1)) + "\n")

    with open(a.out, "a") as fh:
        for name in instances:
            if name in done:
                continue
            pair = RSPPPair.load(os.path.join(a.bench, name + ".json"))
            res = yardstick.solve(pair)
            res["instance"] = name
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(f"  {name}: feasible {res['feasible']}  "
                  f"f1 {res.get('objectives', [float('nan')])[0] / 1000:.1f} km  "
                  f"solutions {res.get('n_solutions')}  {res['runtime']:.1f}s"
                  + ("  TIME CAP REACHED" if res.get("hit_time_cap") else ""), flush=True)
    print("yardstick complete", flush=True)


if __name__ == "__main__":
    main()
