"""M14: evaluate policies on the fixed evaluation set (seeds 0-99).

Run:  python scripts/evaluate_bc.py reactive scripted outputs/models/bc_dagger.pt
      python scripts/evaluate_bc.py --disturb reactive scripted outputs/models/bc_dagger.pt

--disturb: 1 s into each episode, while the gripper is on its way, the cube is moved 5-10 cm.
Results are appended to outputs/eval/bc_results.json.
"""

import argparse
import json
from pathlib import Path

from ari.learning.evaluate import evaluate, summary_line

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("policies", nargs="+", help='"scripted", "reactive", or a .pt file')
    p.add_argument("--disturb", action="store_true")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("outputs/eval/bc_results.json"))
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(args.out.read_text()) if args.out.exists() else {}
    for spec in args.policies:
        r = evaluate(spec, range(args.n), disturb=args.disturb, workers=args.workers)
        print(summary_line(r), flush=True)
        store[f"{spec}|{'disturbed' if args.disturb else 'nominal'}"] = r
        args.out.write_text(json.dumps(store, indent=1))
