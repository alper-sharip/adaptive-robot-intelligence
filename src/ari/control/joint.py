"""Joint-space position control of the Panda.

How the Menagerie Panda's arm actuators work (see scripts/inspect_model.py):
each one is a *position servo*. You set `data.ctrl[i]` to a target angle and
MuJoCo applies a torque

    tau_i = kp * (target_i - q_i) - kd * qdot_i        (clipped to the force limit)

which is a PD controller. kp is the stiffness (torque per radian of error),
kd the damping (torque per rad/s). With no gravity compensation, the arm sags
until the spring term balances gravity:  steady-state error ~= gravity_torque / kp.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ari.robot.model import ARM_ACTUATORS, arm_dof_indices, arm_qpos_indices


def arm_actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.array([model.actuator(name).id for name in ARM_ACTUATORS])


def set_pd_gains(model: mujoco.MjModel, kp: np.ndarray, kd: np.ndarray) -> None:
    """Override the servo gains in the model (for experiments).

    MuJoCo's general actuator computes force = gain * ctrl + bias0 + bias1*q + bias2*qdot.
    A PD servo is gain = kp, bias1 = -kp, bias2 = -kd.
    """
    ids = arm_actuator_ids(model)
    kp = np.broadcast_to(kp, (7,))
    kd = np.broadcast_to(kd, (7,))
    model.actuator_gainprm[ids, 0] = kp
    model.actuator_biasprm[ids, 1] = -kp
    model.actuator_biasprm[ids, 2] = -kd


def get_pd_gains(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    ids = arm_actuator_ids(model)
    return model.actuator_gainprm[ids, 0].copy(), -model.actuator_biasprm[ids, 2].copy()


@dataclass
class Trace:
    """Recorded time series from a simulation run."""
    t: np.ndarray          # (T,)
    q: np.ndarray          # (T, 7) measured joint angles
    q_target: np.ndarray   # (T, 7) commanded joint angles
    tau: np.ndarray        # (T, 7) actuator torques actually applied


def run_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_fn,
    duration: float,
    gravity_comp: bool = False,
) -> Trace:
    """Simulate for `duration` seconds, commanding target_fn(t) -> (7,) joint angles.

    gravity_comp: add a feedforward torque equal to MuJoCo's bias force
    (gravity + Coriolis/centrifugal, data.qfrc_bias) on the arm joints. The PD
    servo then only has to correct errors instead of also holding up the arm,
    which removes the steady-state sag. (Real Panda controllers do this too.)
    """
    act = arm_actuator_ids(model)
    qi, vi = arm_qpos_indices(model), arm_dof_indices(model)
    n = int(round(duration / model.opt.timestep))
    t = np.zeros(n)
    q = np.zeros((n, 7))
    qt = np.zeros((n, 7))
    tau = np.zeros((n, 7))
    for k in range(n):
        target = target_fn(data.time)
        data.ctrl[act] = target
        if gravity_comp:
            # qfrc_bias is from the previous step's state; one step of lag at 500 Hz is negligible.
            data.qfrc_applied[vi] = data.qfrc_bias[vi]
        mujoco.mj_step(model, data)
        t[k] = data.time
        q[k] = data.qpos[qi]
        qt[k] = target
        tau[k] = data.actuator_force[act]
    return Trace(t, q, qt, tau)


@dataclass
class StepMetrics:
    rise_time: float          # s, time from 10% to 90% of the step
    overshoot: float          # fraction of step size beyond the target
    settling_time: float      # s, after which error stays within 2% of step size
    steady_state_error: float # rad, mean error over the final 10% of the run


def step_metrics(t: np.ndarray, q: np.ndarray, q0: float, q1: float) -> StepMetrics:
    """Classic step-response numbers for one joint moving from q0 to q1."""
    step = q1 - q0
    frac = (q - q0) / step
    t10 = t[np.argmax(frac >= 0.1)] if np.any(frac >= 0.1) else np.nan
    t90 = t[np.argmax(frac >= 0.9)] if np.any(frac >= 0.9) else np.nan
    overshoot = max(0.0, frac.max() - 1.0)
    outside = np.abs(frac - 1.0) > 0.02
    settling = t[np.where(outside)[0][-1]] - t[0] if outside.any() else 0.0
    tail = q[int(0.9 * len(q)):]
    return StepMetrics(t90 - t10, overshoot, settling, float(np.mean(q1 - tail)))
