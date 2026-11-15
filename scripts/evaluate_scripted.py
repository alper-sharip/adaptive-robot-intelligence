"""M9: evaluate the scripted pick-and-place baseline on a fixed, seeded episode set.

Evaluation seeds are 0..N-1. Later, demonstrations for imitation learning will be
collected from a DIFFERENT seed range (10000+), so learned policies are never
evaluated on episodes they were trained on.

Run:  python scripts/evaluate_scripted.py                          # 100 episodes, defaults
      python scripts/evaluate_scripted.py --friction 0.3           # slippery cube
      python scripts/evaluate_scripted.py --gripper-stiffness 10   # stronger grip
      python scripts/evaluate_scripted.py --max-speed 0.6          # faster motions
Results are written to outputs/eval/<name>.json so experiments can be compared later.
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

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--gripper-stiffness", type=float, default=1.0, help="multiplier on Menagerie gripper kp")
parser.add_argument("--pyramidal", action="store_true", help="MuJoCo default friction cone (shows creep)")
parser.add_argument("--max-speed", type=float, default=0.25)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--outdir", type=Path, default=Path("outputs/eval"))
args = parser.parse_args()

cfg = SceneConfig(cube_friction=args.friction, gripper_stiffness_scale=args.gripper_stiffness,
                  elliptic_cone=not args.pyramidal, impratio=1.0 if args.pyramidal else 10.0)
env = PickPlaceEnv(cfg)
policy = ScriptedPickPlace(max_speed=args.max_speed)

t0 = time.time()
results = [run_episode(env, policy, seed) for seed in range(args.n)]
wall = time.time() - t0

succ = np.array([r.success for r in results])
err = np.array([r.placement_error for r in results]) * 1000
slip = np.array([r.slip for r in results]) * 1000
dur = np.array([r.duration for r in results])
failures = Counter(r.failure for r in results if not r.success)

# 95% confidence interval for the success rate (Wilson score interval). With 100
# episodes, 95% vs 98% is NOT a meaningful difference; this interval tells you so.
n, p, z = args.n, succ.mean(), 1.96
centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)

print(f"Scripted baseline | friction={args.friction} gripper_stiffness=x{args.gripper_stiffness} "
      f"max_speed={args.max_speed} cone={'pyramidal' if args.pyramidal else 'elliptic'} | {n} episodes in {wall:.0f}s")
print(f"  success rate      : {100 * p:.1f}%  (95% CI {100 * (centre - half):.1f}-{100 * (centre + half):.1f}%)")
print(f"  placement error   : median {np.median(err[succ]) if succ.any() else float('nan'):.1f} mm (successes)")
print(f"  slip while carried: median {np.median(slip):.1f} mm, max {slip.max():.1f} mm")
print(f"  episode time      : mean {dur.mean():.1f} s")
print(f"  failures          : {dict(failures) if failures else 'none'}")
if failures:
    print(f"  failed seeds      : {[r.seed for r in results if not r.success]}")

name = args.name or f"scripted_f{args.friction}_g{args.gripper_stiffness}_v{args.max_speed}" + ("_pyr" if args.pyramidal else "")
args.outdir.mkdir(parents=True, exist_ok=True)
summary = {
    "config": vars(args) | {"outdir": str(args.outdir)},
    "success_rate": float(p),
    "success_ci95": [float(centre - half), float(centre + half)],
    "median_placement_error_mm": float(np.median(err[succ])) if succ.any() else None,
    "median_slip_mm": float(np.median(slip)),
    "mean_episode_time_s": float(dur.mean()),
    "failures": dict(failures),
    "episodes": [asdict(r) | {"phase_log": r.phase_log} for r in results],
}
(args.outdir / f"{name}.json").write_text(json.dumps(summary, indent=1, default=float))
print(f"Saved {args.outdir / f'{name}.json'}")
