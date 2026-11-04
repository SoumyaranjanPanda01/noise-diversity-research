"""Runs the full condition x seed matrix (resumable). Each run is a fresh subprocess.

Order: untuned baseline -> primary comparison (clean, singles, Mixed-3@0.5) -> sweeps
-> NEFTune / augmentation baselines, seed-major within each stage, so the primary
result is available as early as possible.

    python scripts/run_all.py --dry-run           # print plan + status
    python scripts/run_all.py                     # run everything not yet complete
    python scripts/run_all.py --stage primary --seeds 0
    python scripts/run_all.py --shard 0/2         # split the plan across 2 machines
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.conditions import CONDITIONS  # noqa: E402

RUNS_DIR = Path("results/runs")
STAGES = {
    "primary": ["clean", "single", "mixed3"],
    "sweep": ["sweep"],
    "baselines": ["neftune", "augment"],
}


def build_plan(seeds: list[int], stages: list[str]) -> list[tuple[str, int | None]]:
    plan = [("untuned", None)]
    for stage in stages:
        cats = STAGES[stage]
        for seed in seeds:
            plan += [(c.name, seed) for c in CONDITIONS.values() if c.category in cats]
    return plan


def run_id(cond: str, seed: int | None) -> str:
    return cond.replace("/", "__") + ("" if seed is None else f"__s{seed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stage", choices=list(STAGES), nargs="+", default=list(STAGES))
    ap.add_argument("--shard", default=None, help="i/n: run every n-th plan entry starting at i")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("passthrough", nargs=argparse.REMAINDER, help="extra args for train.py after --")
    args = ap.parse_args()

    plan = build_plan(args.seeds, args.stage)
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        plan = plan[i::n]
    done = [p for p in plan if (RUNS_DIR / run_id(*p) / "metrics.json").exists()]
    print(f"plan: {len(plan)} runs, {len(done)} complete, {len(plan) - len(done)} remaining")
    if args.dry_run:
        for p in plan:
            print(("  [x] " if p in done else "  [ ] ") + run_id(*p))
        return

    extra = [a for a in args.passthrough if a != "--"]
    failures = []
    for k, (cond, seed) in enumerate(plan, 1):
        rid = run_id(cond, seed)
        if (RUNS_DIR / rid / "metrics.json").exists():
            continue
        (RUNS_DIR / rid).mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "scripts/train.py", "--condition", cond] + \
              ([] if seed is None else ["--seed", str(seed)]) + extra
        print(f"[{k}/{len(plan)}] {rid}", flush=True)
        t0 = time.time()
        with open(RUNS_DIR / rid / "stdout.log", "w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(f"    -> {status} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            failures.append(rid)
    with open(RUNS_DIR / "_last_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"plan": [run_id(*p) for p in plan], "failures": failures,
                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
    if failures:
        print("failures:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
