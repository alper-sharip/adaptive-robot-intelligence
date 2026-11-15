"""M14: watch a policy run one episode, optionally with the cube moved mid-episode.

Run:  python scripts/bc_demo.py outputs/models/bc_reactive_500.pt --seed 3 --disturb
      python scripts/bc_demo.py scripted --seed 3 --disturb     # the clock-driven expert, for contrast
Saves a GIF to outputs/.
"""

import argparse
from pathlib import Path

from ari.envs.pick_place import PickPlaceEnv  # isort: skip  (ari first: it selects the headless renderer)

import mujoco
from PIL import Image

from ari.learning import demos
from ari.learning.evaluate import CubeShift, make_policy

p = argparse.ArgumentParser()
p.add_argument("policy", help='"scripted", "reactive", or a .pt file')
p.add_argument("--seed", type=int, default=3)
p.add_argument("--disturb", action="store_true")
p.add_argument("--max-time", type=float, default=14.0)
p.add_argument("--joint-interp", action="store_true", help="executor used by RL policies (Stage 5)")
p.add_argument("--out", type=Path, default=None)
args = p.parse_args()

env = PickPlaceEnv()
policy = make_policy(args.policy, env)
renderer = mujoco.Renderer(env.model, height=240, width=320)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.4, 0.0, 0.25]
cam.distance, cam.azimuth, cam.elevation = 1.5, 150, -25
frames = []


def on_frame(env):
    if len(frames) * 0.1 <= env.data.time:      # 10 frames per simulated second
        renderer.update_scene(env.data, camera=cam)
        frames.append(Image.fromarray(renderer.render()))


res = demos.rollout(env, policy, args.seed, max_time=args.max_time,
                    disturbance=CubeShift(args.seed) if args.disturb else None, frame_callback=on_frame,
                    joint_interp=args.joint_interp)
print(f"{args.policy} seed {args.seed}{' (cube moved at 1 s)' if args.disturb else ''}: "
      f"success={res.success} {res.failure} placement error {1000 * res.placement_error:.1f} mm, {res.duration:.1f} s")
name = Path(args.policy).stem + f"_seed{args.seed}" + ("_disturbed" if args.disturb else "")
out = args.out or Path("outputs") / f"{name}.gif"
palette = frames[len(frames) // 2].quantize(colors=64)
gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
gif[0].save(out, save_all=True, append_images=gif[1:], duration=100, loop=0)
print(f"Saved {out}")
