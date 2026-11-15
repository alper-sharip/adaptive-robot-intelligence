"""M5: Cartesian (task-space) control built on IK + joint servos.

Layers, from the top:
    desired TCP pose (from a trajectory or, later, a learned policy)
      -> IK, warm-started from the previous solution          (control rate, 100 Hz)
      -> joint position targets to the PD servos + gravity feedforward
      -> MuJoCo physics                                       (500 Hz)

Why warm-start from the previous *commanded* joints instead of the measured ones?
Consecutive targets on a smooth trajectory are close, so the previous solution
is an excellent initial guess (IK converges in 1-3 iterations), and the commanded
posture stays continuous, which avoids the arm jumping between IK solutions (remember
the 7-DoF redundancy).
"""

from __future__ import annotations

import mujoco
import numpy as np

from ari.robot.ik import Q_HOME, solve_ik
from ari.robot.model import arm_dof_indices

GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0


class CartesianController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, control_dt: float = 0.01):
        self.model, self.data = model, data
        self.arm_dofs = arm_dof_indices(model)
        self.substeps = max(1, int(round(control_dt / model.opt.timestep)))
        self.control_dt = self.substeps * model.opt.timestep
        self.q_cmd = Q_HOME.copy()
        self.gripper_cmd = GRIPPER_OPEN
        self.ik_failures = 0

    def reset(self, q: np.ndarray = Q_HOME) -> None:
        self.q_cmd = np.array(q, dtype=float)
        self.gripper_cmd = GRIPPER_OPEN
        self.ik_failures = 0

    def set_pose_target(self, T: np.ndarray) -> bool:
        res = solve_ik(T, q_init=self.q_cmd, max_iters=30, q_preferred=Q_HOME)
        if not res.success:
            self.ik_failures += 1
        self.q_cmd = res.q  # use best effort even if not fully converged
        return res.success

    def set_gripper(self, closed: bool) -> None:
        self.gripper_cmd = GRIPPER_CLOSED if closed else GRIPPER_OPEN

    def step(self) -> None:
        """Advance one control period (several physics steps)."""
        d = self.data
        d.ctrl[:7] = self.q_cmd
        d.ctrl[7] = self.gripper_cmd
        for _ in range(self.substeps):
            d.qfrc_applied[self.arm_dofs] = d.qfrc_bias[self.arm_dofs]
            mujoco.mj_step(self.model, d)
