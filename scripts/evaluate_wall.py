"""M8: compare policies on the wall scene (same 100 seeded episodes for each).

Policies:
  naive      Stage 2 scripted policy, unchanged: carries the cube at 15 cm, straight through the wall.
  lift-high  Scripted policy that knows the wall height and carries the cube 12 cm above it.
             The obvious hand-written fix; no collision checking.
  planned    Collision-checked: posture selection + RRT-Connect (PlannedPickPlace).

"Clean success" = cube placed AND nothing touched the wall during the episode.
A success that scraped the wall would break things on a real robot.

Run:  python scripts/evaluate_wall.py --policy planned
      python scripts/evaluate_wall.py --policy planned --margin 0.01
"""

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ari.envs.pick_place import PickPlaceEnv, SceneConfig
from ari.skills.pick_place import ScriptedPickPlace, run_episode
from ari.skills.planned_pick_place import PlannedPickPlace

parser = argparse.ArgumentParser()
parser.add_argument("--policy", choices=["naive", "lift-high", "planned"], required=True)
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--margin", type=float, default=0.02, help="planner clearance (m)")
parser.add_argument("--outdir", type=Path, default=Path("outputs/eval"))
args = parser.parse_args()

env = PickPlaceEnv(SceneConfig.with_wall())


def make_policy(seed):
    ep = env.sample_episode(seed)
    if args.policy == "naive":
        return ScriptedPickPlace()
    if args.policy == "lift-high":
        return ScriptedPickPlace(carry_height=ep.wall_height + 0.12)
    return PlannedPickPlace(margin=args.margin, plan_seed=seed)


t0 = time.time()
results = [run_episode(env, make_policy(s), s) for s in range(args.n)]
wall = time.time() - t0


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return 100 * (c - h), 100 * (c + h)


n = args.n
succ = sum(r.success for r in results)
clean = sum(r.success and r.wall_contact_steps == 0 for r in results)
touched = sum(r.wall_contact_steps > 0 for r in results)
plan_t = np.array([r.plan_time for r in results])
fails = Counter(r.failure for r in results if not r.success)

label = args.policy + (f" (margin {args.margin * 100:.0f} cm)" if args.policy == "planned" else "")
print(f"Wall scene | {label} | {n} episodes in {wall:.0f}s")
print(f"  clean success (placed, no wall contact): {clean}%  CI {wilson(clean, n)[0]:.1f}-{wilson(clean, n)[1]:.1f}%")
print(f"  success (ignoring contact)             : {succ}%")
print(f"  episodes touching the wall             : {touched}")
print(f"  max wall penetration                   : {1000 * max(r.max_wall_penetration for r in results):.1f} mm")
print(f"  failures                               : {dict(fails) if fails else 'none'}")
if args.policy == "planned":
    print(f"  planning time                          : median {np.median(plan_t):.2f}s, max {plan_t.max():.2f}s")
print(f"  mean episode time (successes)          : {np.mean([r.duration for r in results if r.success]):.1f}s")

name = f"wall_{args.policy}" + (f"_m{args.margin}" if args.policy == "planned" else "")
args.outdir.mkdir(parents=True, exist_ok=True)
(args.outdir / f"{name}.json").write_text(json.dumps({
    "config": {k: str(v) for k, v in vars(args).items()},
    "clean_success": clean / n, "success": succ / n, "touched_wall": touched,
    "failures": dict(fails), "median_plan_time": float(np.median(plan_t)),
    "episodes": [asdict(r) for r in results],
}, indent=1, default=float))
