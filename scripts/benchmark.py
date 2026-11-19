"""M22: run the generalization benchmark (every policy x every condition, 100 seeded episodes each).

    python scripts/benchmark.py                 # main matrix (78 cells, ~40 min on 2 cores)
    python scripts/benchmark.py --sweeps        # severity sweeps (jitter, offset, latency)
    python scripts/benchmark.py --image         # the image policy (renders two cameras per step; slow)
    python scripts/benchmark.py --extras        # footnote measurements used in docs/results_stage7.md
    python scripts/benchmark.py --camera --n 50 # cube and target from the cameras (~3 h on 2 cores)
    python scripts/benchmark.py --only nominal moved --n 20     # a quick look

Results accumulate in outputs/benchmark/results.json (finished cells are skipped; --force reruns).
Plot and tables: scripts/benchmark_report.py.
"""

import argparse
import json
import time
from pathlib import Path

from ari import benchmark as B

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", action="store_true")
    p.add_argument("--image", action="store_true")
    p.add_argument("--extras", action="store_true")
    p.add_argument("--camera", action="store_true", help="the camera loop: cube and target from images (50 episodes per cell)")
    p.add_argument("--only", nargs="*", default=None, help="condition names")
    p.add_argument("--policies", nargs="*", default=None, help="policy specs")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", type=Path, default=Path("outputs/benchmark/results.json"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(args.out.read_text()) if args.out.exists() else {}
    if args.extras:
        cells = [("reactive#4cm", "large"), ("reactive#4cm", "small"), ("reactive", "heavy@0.2"),
                 (f"{B.M}/bc_dagger.pt", "heavy@0.2")]
        # the action interface before the latency fix, at every latency strength
        cells += [(s, f"latency_legacy@{k}") for k in (1, 2, 3, 4) for s in B.POLICIES]
    elif args.camera:
        cells = [(s, f"cam:{c}") for c in B.CAMERA_CONDITIONS for s in B.POLICIES]
    elif args.image:
        cells = [(B.IMAGE_POLICY[0], c) for c in B.IMAGE_CONDITIONS]
    elif args.sweeps:
        # The strength already in the main matrix is not run twice (benchmark_report.py reuses it).
        cells = [(s, f"{axis}@{v}") for axis, (attr, _, values) in B.SWEEPS.items() for v in values
                 if v != getattr(B.CONDITIONS[axis], attr) for s in B.POLICIES]
    else:
        cells = [(s, c) for c in B.CONDITIONS for s in B.POLICIES]
    if args.only:
        cells = [x for x in cells if x[1] in args.only]
    if args.policies:
        cells = [x for x in cells if x[0] in args.policies]

    for spec, cond in cells:
        key = f"{spec}|{cond}"
        if key in store and store[key]["n"] >= args.n and not args.force:
            continue
        t0 = time.time()
        r = B.run_cell(spec, cond, range(args.n))
        store[key] = r
        # one cell per line: readable diffs, small file
        args.out.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}"
                                               for k, v in store.items()) + "\n}\n")
        print(f"{cond:18s} {Path(spec).stem:18s} {100 * r['success_rate']:5.1f}%  (CI {100 * r['ci95'][0]:.0f}-{100 * r['ci95'][1]:.0f})  "
              f"{r['failures'] or ''}  [{time.time() - t0:.0f}s]", flush=True)
