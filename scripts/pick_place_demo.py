"""M7: run one scripted pick-and-place episode and record it.

Saves:
  outputs/pick_place_seed<N>.gif        animation
  outputs/pick_place_seed<N>_strip.png  one frame per phase
  outputs/pick_place_seed<N>_trace.png  cube height, TCP speed, gripper width over time

Run:  python scripts/pick_place_demo.py --seed 0
      MUJOCO_GL=egl python scripts/pick_place_demo.py   (headless Linux / Colab)
Try:  --max-speed 0.5, --grasp-depth 0.015 (grasp higher on the cube), --friction 0.3
Wall: --wall --policy planned   (collision-aware, M8)
      --wall --policy naive     (Stage 2 policy, drives through the wall)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

from ari.envs.pick_place import PickPlaceEnv, SceneConfig
from ari.skills.pick_place import ScriptedPickPlace, run_episode
from ari.skills.planned_pick_place import PlannedPickPlace

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-speed", type=float, default=0.25)
parser.add_argument("--grasp-depth", type=float, default=0.0)
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--wall", action="store_true", help="add the wall obstacle (M8)")
parser.add_argument("--policy", choices=["scripted", "naive", "planned"], default="scripted")
parser.add_argument("--outdir", type=Path, default=Path("outputs"))
args = parser.parse_args()

cfg_kw = dict(cube_friction=args.friction)
env = PickPlaceEnv(SceneConfig.with_wall(**cfg_kw) if args.wall else SceneConfig(**cfg_kw))
if args.policy == "planned":
    policy = PlannedPickPlace(max_speed=args.max_speed, grasp_depth=args.grasp_depth, plan_seed=args.seed)
else:
    policy = ScriptedPickPlace(max_speed=args.max_speed, grasp_depth=args.grasp_depth)

renderer = mujoco.Renderer(env.model, height=360, width=480)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.4, 0.0, 0.25]
cam.distance, cam.azimuth, cam.elevation = 1.5, 150, -25

frames, phase_first_frame, trace = [], {}, []


def on_frame(env, phase):
    renderer.update_scene(env.data, camera=cam)
    img = renderer.render()
    frames.append(img)
    phase_first_frame.setdefault(phase, img)
    trace.append((env.data.time, env.data.xpos[env.cube_body][2], env.finger_width(),
                  np.linalg.norm(env.data.cvel[env.hand_body][3:]), phase))


res = run_episode(env, policy, args.seed, frame_callback=on_frame, frame_every=10)
print(f"seed {res.seed}: success={res.success} failure='{res.failure}' "
      f"placement error={1000 * res.placement_error:.1f} mm, {res.duration:.2f} s, "
      f"wall contact steps={res.wall_contact_steps}, planning={res.plan_time:.2f} s")
print("phase end times:", res.phase_log)

args.outdir.mkdir(parents=True, exist_ok=True)
stem = args.outdir / (f"pick_place_seed{args.seed}" if not args.wall else f"wall_{args.policy}_seed{args.seed}")
# GIF at half resolution with a shared 64-colour palette keeps the file small enough for GitHub.
gif = [Image.fromarray(f).resize((320, 240)) for f in frames]
palette = gif[len(gif) // 2].quantize(colors=64)
gif = [g.quantize(palette=palette, dither=Image.Dither.NONE) for g in gif]
gif[0].save(f"{stem}.gif", save_all=True, append_images=gif[1:], duration=50, loop=0)

names = list(phase_first_frame)
fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 2.6))
for ax, n in zip(axes, names):
    ax.imshow(phase_first_frame[n])
    ax.set_title(n, fontsize=10)
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{stem}_strip.png", dpi=90)

t, cz, fw, v, ph = zip(*trace)
fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
axes[0].plot(t, np.array(cz) * 100)
axes[0].set_ylabel("cube height (cm)")
axes[1].plot(t, np.array(fw) * 1000)
axes[1].set_ylabel("finger gap (mm)")
axes[2].plot(t, v)
axes[2].set_ylabel("hand speed (m/s)")
axes[2].set_xlabel("time (s)")
for ax in axes:
    for name, t_end in res.phase_log:
        ax.axvline(t_end, color="gray", lw=0.6, ls=":")
for name, t_end in res.phase_log:
    axes[0].text(t_end, axes[0].get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=7)
fig.suptitle(f"Scripted pick-and-place, seed {args.seed}")
fig.tight_layout()
fig.savefig(f"{stem}_trace.png", dpi=100)
print(f"Saved {stem}.gif, {stem}_strip.png, {stem}_trace.png")
