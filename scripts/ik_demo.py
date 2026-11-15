"""M4: move the gripper to Cartesian targets with your own IK, then check in simulation.

For each target:
  1. solve_ik() finds joint angles (pure math, no physics).
  2. Those angles are sent to the PD servos and the physics runs for 1.5 s.
  3. We measure where the TCP really ended up. The gap between (1) and (3) is
     what gravity sag and servo error cost you.

Run:  python scripts/ik_demo.py            # prints a table, saves a filmstrip image
      python scripts/ik_demo.py --no-render
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from ari.control.joint import arm_actuator_ids, run_targets
from ari.robot.ik import Q_HOME, solve_ik
from ari.robot.kinematics import tcp_pose
from ari.robot.model import TCP_OFFSET, arm_qpos_indices, load_panda, reset_to_home
from ari.robot.transforms import make_T, rot_x

parser = argparse.ArgumentParser()
parser.add_argument("--no-render", action="store_true")
parser.add_argument("--gravity-comp", action="store_true", help="add gravity feedforward torque")
parser.add_argument("--out", type=Path, default=Path("outputs/ik_demo.png"))
args = parser.parse_args()

# Gripper pointing straight down: hand z-axis = -world z. That is a 180 deg rotation about x.
DOWN = rot_x(np.pi)
TARGETS = [
    (0.5, 0.0, 0.4),
    (0.5, 0.3, 0.2),
    (0.4, -0.3, 0.15),
    (0.3, 0.0, 0.6),
    (0.6, 0.0, 0.1),
]

model, data = load_panda()
reset_to_home(model, data)
qi = arm_qpos_indices(model)
hand = model.body("hand").id

renderer = None
if not args.no_render:
    renderer = mujoco.Renderer(model, height=360, width=480)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.4, 0.0, 0.3]
    cam.distance, cam.azimuth, cam.elevation = 1.8, 140, -20
frames = []

print(f"{'target (m)':>22s} {'IK ok':>6s} {'iters':>5s} {'IK err(mm)':>10s} {'sim err(mm)':>11s}")
q_prev = Q_HOME
for p in TARGETS:
    T_target = make_T(DOWN, np.array(p))
    res = solve_ik(T_target, q_init=q_prev)
    q_prev = res.q

    # Execute: hold the IK solution as the servo target, keep current gripper command.
    tr = run_targets(model, data, lambda t, q=res.q: q, duration=1.5, gravity_comp=args.gravity_comp)
    T_hand = make_T(data.xmat[hand].reshape(3, 3), data.xpos[hand])
    tcp_sim = (T_hand @ np.array([0, 0, TCP_OFFSET, 1.0]))[:3]
    sim_err = np.linalg.norm(tcp_sim - np.array(p))
    print(f"{str(p):>22s} {str(res.success):>6s} {res.iterations:5d} "
          f"{1000 * res.pos_error:10.3f} {1000 * sim_err:11.2f}")

    if renderer is not None:
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

if frames:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 3.4))
    for ax, img, p in zip(axes, frames, TARGETS):
        ax.imshow(img)
        ax.set_title(f"target {p}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"\nSaved filmstrip to {args.out}")
