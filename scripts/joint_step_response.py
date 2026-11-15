"""M2: step response of each arm joint under the built-in PD servos.

Each joint in turn is commanded to jump +0.3 rad from the home posture. We record
the response, compute rise time, overshoot, settling time and steady-state error,
and plot it.

Run:  python scripts/joint_step_response.py                 # Menagerie gains
      python scripts/joint_step_response.py --kp-scale 0.1  # 10x softer springs
      python scripts/joint_step_response.py --kd-scale 0.2  # less damping

Things to try and explain:
  * Lower kp: why does the steady-state error grow, and why mostly for joints 2 and 4?
  * Lower kd: when does overshoot appear?
  * Very high kp: watch the torque plot. What limits the response now?
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ari.control.joint import get_pd_gains, run_targets, set_pd_gains, step_metrics
from ari.robot.model import load_panda, reset_to_home

STEP = 0.3        # rad
SETTLE = 0.5      # s at home before the step
DURATION = 2.5    # s total

parser = argparse.ArgumentParser()
parser.add_argument("--kp-scale", type=float, default=1.0)
parser.add_argument("--kd-scale", type=float, default=1.0)
parser.add_argument("--gravity-comp", action="store_true")
parser.add_argument("--out", type=Path, default=Path("outputs/joint_step_response.png"))
args = parser.parse_args()

model, data = load_panda()
kp0, kd0 = get_pd_gains(model)
set_pd_gains(model, kp0 * args.kp_scale, kd0 * args.kd_scale)
kp, kd = get_pd_gains(model)
print(f"kp = {kp}\nkd = {kd}\n")

fig, axes = plt.subplots(7, 2, figsize=(11, 14), sharex=True)
print(f"{'joint':6s} {'rise(s)':>8s} {'overshoot':>10s} {'settle(s)':>10s} {'ss_err(mrad)':>13s} {'max|tau|':>9s}")
for j in range(7):
    reset_to_home(model, data)
    q_home = data.ctrl[:7].copy()

    def target(t, j=j):
        q = q_home.copy()
        if t >= SETTLE:
            q[j] += STEP
        return q

    tr = run_targets(model, data, target, DURATION, gravity_comp=args.gravity_comp)
    after = tr.t >= SETTLE
    q_before = tr.q[~after, j][-1]   # where the joint actually sat before the step (gravity sag!)
    m = step_metrics(tr.t[after], tr.q[after, j], q_before, q_home[j] + STEP)
    print(f"{j + 1:<6d} {m.rise_time:8.3f} {100 * m.overshoot:9.1f}% {m.settling_time:10.3f} "
          f"{1000 * m.steady_state_error:13.2f} {np.abs(tr.tau[:, j]).max():9.1f}")

    axes[j, 0].plot(tr.t, tr.q_target[:, j], "k--", lw=1, label="target")
    axes[j, 0].plot(tr.t, tr.q[:, j], lw=1.5, label="measured")
    axes[j, 0].set_ylabel(f"joint {j + 1}\n(rad)")
    axes[j, 1].plot(tr.t, tr.tau[:, j], color="tab:red", lw=1.2)
    axes[j, 1].set_ylabel("torque (Nm)")
axes[0, 0].legend(loc="lower right", fontsize=8)
axes[0, 0].set_title("Joint angle: commanded vs measured")
axes[0, 1].set_title("Actuator torque")
axes[-1, 0].set_xlabel("time (s)")
axes[-1, 1].set_xlabel("time (s)")
fig.suptitle(f"Panda joint step response (kp x{args.kp_scale}, kd x{args.kd_scale})")
fig.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=110)
print(f"\nSaved plot to {args.out}")
