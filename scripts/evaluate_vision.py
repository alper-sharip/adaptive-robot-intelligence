"""M12: does the robot still succeed when it has to SEE the cube instead of being told?

Same 100 seeded episodes as the ground-truth evaluations, so results compare directly with
docs/results_stage2.md (open scene, scripted) and docs/results_m8.md (wall scene, planned).

Run:  python scripts/evaluate_vision.py --scene open --noise realistic
      python scripts/evaluate_vision.py --scene wall --noise poor
      python scripts/evaluate_vision.py --scene open --perception gt   # ground-truth reference
"""

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ari.envs.pick_place import PickPlaceEnv, SceneConfig
from ari.perception.estimator import VisionPolicy
from ari.skills.pick_place import ScriptedPickPlace, run_episode
from ari.skills.planned_pick_place import PlannedPickPlace

parser = argparse.ArgumentParser()
parser.add_argument("--scene", choices=["open", "wall"], required=True)
parser.add_argument("--perception", choices=["vision", "gt"], default="vision")
parser.add_argument("--noise", choices=["none", "realistic", "poor"], default="none")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--outdir", type=Path, default=Path("outputs/eval"))
args = parser.parse_args()

env = PickPlaceEnv(SceneConfig.with_wall() if args.scene == "wall" else SceneConfig())


def make_policy(seed):
    base = PlannedPickPlace(plan_seed=seed) if args.scene == "wall" else ScriptedPickPlace()
    if args.perception == "gt":
        return base
    return VisionPolicy(base, noise=args.noise, noise_seed=seed + 100_000)


t0 = time.time()
results, perr = [], []
for s in range(args.n):
    pol = make_policy(s)
    results.append(run_episode(env, pol, s))
    if args.perception == "vision" and hasattr(pol, "errors"):
        perr.append(asdict(pol.errors))
wall = time.time() - t0

n = args.n
ok = [r.success and r.wall_contact_steps == 0 for r in results]
k = sum(ok)
p = k / n
z = 1.96
c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
fails = Counter(r.failure or "wall_contact" for r, o in zip(results, ok) if not o)
err = np.array([r.placement_error for r, o in zip(results, ok) if o]) * 1000

label = f"{args.scene} scene | {'planned' if args.scene == 'wall' else 'scripted'} | " + (
    "ground truth" if args.perception == "gt" else f"vision, noise={args.noise}")
print(f"{label} | {n} episodes in {wall:.0f}s")
print(f"  clean success     : {100 * p:.0f}%  (95% CI {100 * (c - h):.1f}-{100 * (c + h):.1f})")
print(f"  placement error   : median {np.median(err):.1f} mm" if len(err) else "  placement error   : -")
print(f"  failures          : {dict(fails) if fails else 'none'}")
summary = {"config": {k: str(v) for k, v in vars(args).items()}, "clean_success": p,
           "ci95": [c - h, c + h], "median_placement_error_mm": float(np.median(err)) if len(err) else None,
           "failures": dict(fails)}
if perr:
    P = {key: np.array([e[key] for e in perr]) for key in perr[0]}
    summary["perception"] = {
        "cube_xy_mm": [float(np.median(P["cube_xy"]) * 1000), float(P["cube_xy"].max() * 1000)],
        "cube_z_bias_mm": float(P["cube_z"].mean() * 1000),
        "cube_yaw_deg": [float(np.degrees(np.median(P["cube_yaw"]))), float(np.degrees(P["cube_yaw"].max()))],
        "target_xy_mm": [float(np.median(P["target_xy"]) * 1000), float(P["target_xy"].max() * 1000)],
        "detections": len(perr),
    }
    q = summary["perception"]
    print(f"  perception (median / max): cube xy {q['cube_xy_mm'][0]:.2f} / {q['cube_xy_mm'][1]:.2f} mm, "
          f"z bias {q['cube_z_bias_mm']:+.2f} mm, yaw {q['cube_yaw_deg'][0]:.2f} / {q['cube_yaw_deg'][1]:.2f} deg, "
          f"target {q['target_xy_mm'][0]:.2f} / {q['target_xy_mm'][1]:.2f} mm, detected {q['detections']}/{n}")
args.outdir.mkdir(parents=True, exist_ok=True)
name = f"vision_{args.scene}_{args.perception}" + (f"_{args.noise}" if args.perception == "vision" else "")
summary["episodes"] = [asdict(r) for r in results]
(args.outdir / f"{name}.json").write_text(json.dumps(summary, indent=1, default=float))
