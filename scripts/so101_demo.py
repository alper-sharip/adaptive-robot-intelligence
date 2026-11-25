"""SO-101 in simulation: calibrate the one-sided gripper, run scripted pick-and-place, save a GIF.

    python scripts/so101_demo.py --episodes 100 --gif outputs/so101/so101_pick_place.gif
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import ari  # noqa: F401  (headless rendering setup before mujoco)

import imageio.v2 as imageio
import mujoco
import numpy as np

from ari.so101.env import SO101Env
from ari.so101.pick_place import ScriptedPickPlace


def summarize(results):
    ok = [r for r in results if r.success]
    e = np.array([r.placement_error_mm for r in ok]) if ok else np.array([np.nan])
    return dict(success=len(ok), episodes=len(results),
                failures=dict(Counter(r.failure for r in results if not r.success)),
                placement_median_mm=float(np.median(e)), placement_p95_mm=float(np.percentile(e, 95)),
                placement_max_mm=float(np.max(e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--gif", default="outputs/so101/so101_pick_place.gif")
    ap.add_argument("--gif-seeds", type=int, nargs="+", default=[3])
    ap.add_argument("--out", default="outputs/so101/demo.json")
    a = ap.parse_args()

    env = SO101Env()
    cal = ScriptedPickPlace.calibrate(env)
    print(f"gripper calibration (tool y, z): held {np.round(1000 * cal.held, 1)} mm, "
          f"after release {np.round(1000 * cal.placed, 1)} mm")
    report = {"held_offset_mm": (1000 * cal.held).tolist(), "placed_offset_mm": (1000 * cal.placed).tolist()}
    for label, c in (("uncalibrated", None), ("calibrated", cal)):
        pp = ScriptedPickPlace(env, calibration=c)
        report[label] = summarize([pp.run(s) for s in range(a.episodes)])
        print(label, report[label])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))

    if a.gif:
        renderer = mujoco.Renderer(env.model, 300, 400)
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
            renderer.scene.flags[flag] = 0        # shadow-map stripes otherwise
        frames, every = [], int(round(1 / 12 / env.model.opt.timestep))   # 12 fps
        count = [0]

        def grab(e):
            count[0] += 1
            if count[0] % every == 0:
                renderer.update_scene(e.data, camera="front_cam")
                frames.append(renderer.render().copy())

        pp = ScriptedPickPlace(env, calibration=cal)
        for s in a.gif_seeds:
            r = pp.run(s, frame_callback=grab)
            print(f"gif seed {s}: success={r.success} error={r.placement_error_mm:.1f} mm")
        imageio.mimsave(a.gif, frames, duration=1 / 12, loop=0)
        print("wrote", a.gif, len(frames), "frames")


if __name__ == "__main__":
    main()
