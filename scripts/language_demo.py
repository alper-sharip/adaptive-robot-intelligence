"""Stage 6: type an instruction, watch the robot plan and carry it out.

Run:  python scripts/language_demo.py "put the yellow cube on the green target, then the red cube goes on the blue spot"
      python scripts/language_demo.py --seed 7 --faults            # a generated task with injected mishaps
      python scripts/language_demo.py "put the red cube on the red target and then move the blue cube to the red target"
                                                                    # refused: the red target would be used twice

Prints the plan, any validation problems, and what happened; saves a GIF to outputs/.
"""

import argparse
from pathlib import Path

from ari.envs.multi_object import MultiObjectEnv  # isort: skip  (ari first: it selects the headless renderer)

import mujoco
from PIL import Image

from ari.language import evaluate as lev
from ari.language import parser
from ari.learning.evaluate import make_policy
from ari.planning.task_planner import Faults, TaskExecutor, make_plan, validate_plan

p = argparse.ArgumentParser()
p.add_argument("instruction", nargs="?", default=None, help="omit to use the evaluation task for --seed")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--skill", default="outputs/models/bc_dagger.pt", help='"scripted", "reactive", or a .pt file')
p.add_argument("--fallback", default="reactive", help='skill used after a failed attempt ("none" to disable)')
p.add_argument("--faults", action="store_true", help="knock the cube out of the gripper / move a placed cube")
p.add_argument("--no-monitor", action="store_true")
p.add_argument("--out", type=Path, default=None)
args = p.parse_args()

env = MultiObjectEnv()
env.reset(args.seed)
text = args.instruction or lev.m20_task(args.seed)[0]
steps = make_plan(text, parser.load("outputs/models/language_parser.pt"))
print(f'instruction: "{text}"')
for k, s in enumerate(steps):
    print(f"  step {k + 1}: {s.cube} cube -> {s.target} target   (confidence {s.confidence:.2f})   from: \"{s.text}\"")
problems = validate_plan(steps, env)
if problems:
    print("REFUSED before moving:")
    for msg in problems:
        print("  -", msg)
    raise SystemExit(0)

renderer = mujoco.Renderer(env.model, height=300, width=400)
renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0          # small coloured objects read better
renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.48, 0.02, 0.05]
cam.distance, cam.azimuth, cam.elevation = 0.95, 180, -50          # from the front, looking down at the table
frames = []


def on_frame(env):
    if len(frames) * 0.1 <= env.data.time:      # 10 frames per simulated second
        renderer.update_scene(env.data, camera=cam)
        frames.append(Image.fromarray(renderer.render()))


executor = TaskExecutor(env, make_policy(args.skill, env), monitor=not args.no_monitor,
                        fallback=None if args.fallback == "none" else make_policy(args.fallback, env),
                        frame_callback=on_frame)
faults = Faults(args.seed, len(steps)) if args.faults else None
res = executor.run(steps, faults)
print(f"success={res.success}  skill executions={res.attempts}  repairs={res.repairs}  "
      f"injected faults={res.faults}  simulated time {res.sim_time:.1f} s")
out = args.out or Path("outputs") / f"language_seed{args.seed}{'_faults' if args.faults else ''}.gif"
palette = frames[len(frames) // 2].quantize(colors=64)
gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames[::2]]
gif[0].save(out, save_all=True, append_images=gif[1:], duration=200, loop=0)
print(f"Saved {out}")
