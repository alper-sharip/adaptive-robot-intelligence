"""Interactive slider control of the Panda, with the kinematics shown live.

`SliderSession` holds the simulated scene and turns slider values into motion:

  Cartesian mode: sliders set the gripper's target pose (x, y, z, yaw, gripper). The
    project's own damped-least-squares IK (robot/ik.py) solves for the seven joints, and
    the joint PD servos follow. A further slider sets the elbow swivel angle, a second task
    solved in the null space of the Jacobian: the elbow swings while the gripper stays put.
    (Biasing a single joint does not do this near the home pose: axes 1 and 3 are then almost
    parallel and the IK simply trades one for the other. Measured: 0.5 deg of swivel.)
  Joint mode: sliders set the seven joint angles directly (forward kinematics only).

`SliderSession.state()` returns everything the interface shows: the modified-DH table with
the live joint angles, every link transform, their product down to the tool, the check
against MuJoCo, the Jacobian and its singular values, the last IK solve iteration by
iteration, and the one joint angle the Panda's geometry gives in closed form (the elbow,
`elbow_from_distance`).

Why only one closed-form angle? A pose has six numbers and the arm has seven joints, so
there is a one-parameter family of solutions (the elbow can swivel about the shoulder-wrist
line). The elbow angle is the exception: the shoulder-wrist distance depends on joint 4
alone, so the law of cosines gives it exactly. The interface shows that equation with live
numbers, and the swivel angle that the redundancy leaves free.

No HTTP and no rendering here (see scripts/slider_control.py); everything is testable.
"""

from __future__ import annotations

import numpy as np

from ari.envs.pick_place import PickPlaceEnv
from ari.robot.ik import JOINT_LIMITS, Q_HOME, pose_error, solve_ik
from ari.robot.kinematics import DH, FLANGE_D, HAND_YAW, _dh_link, joint_frames, tcp_pose_and_jacobian
from ari.robot.model import TCP_OFFSET, arm_dof_indices
from ari.robot.transforms import make_T, rot_x, rot_z, trans

GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0
DAMPING = 0.05                      # lambda in the IK update, as used by the controller
WORKSPACE = {"x": (0.25, 0.75), "y": (-0.40, 0.40), "z": (0.015, 0.65), "yaw": (-np.pi, np.pi)}


# ---------------------------------------------------------------- the elbow, in closed form
def _elbow_constants() -> dict:
    """Constants of  d^2 = A - 2 R cos(q4 - delta)  with d = |shoulder - wrist|.

    Shoulder S = origin of frame 2 (= frame 1, on the base axis at 0.333 m).
    Wrist    W = origin of frame 5 (where axes 5 and 6 meet).
    Joint 4 rotates frame 4 about its own z-axis, which passes through the elbow. Written in
    frame 4, W sits at a fixed point w (it does not depend on q5), and S sits at
    RotZ(-q4) p for a fixed point p (it does not depend on q1..q3). So
        d^2 = |w - RotZ(-q4) p|^2 = |w|^2 + |p|^2 - 2 w_z p_z - 2 R cos(q4 - delta)
    with R cos(delta) = w_x p_x + w_y p_y and R sin(delta) = w_x p_y - w_y p_x.
    """
    a3, al3 = DH[3, 0], DH[3, 2]
    a4, d5, al4 = DH[4, 0], DH[4, 1], DH[4, 2]
    d3 = DH[2, 1]
    w = np.array([a4, 0.0, 0.0]) + rot_x(al4) @ np.array([0.0, 0.0, d5])     # W in frame 4
    s3 = np.array([0.0, 0.0, -d3])                                             # S in frame 3
    p = (trans(-a3, 0, 0) @ make_T(rot_x(-al3), np.zeros(3)) @ np.append(s3, 1.0))[:3]
    c = w[0] * p[0] + w[1] * p[1]
    s = w[0] * p[1] - w[1] * p[0]
    return {"A": float(w @ w + p @ p - 2 * w[2] * p[2]), "R": float(np.hypot(c, s)),
            "delta": float(np.arctan2(s, c)), "w": w.tolist(), "p": p.tolist()}


ELBOW = _elbow_constants()


def shoulder_wrist_distance(q: np.ndarray) -> float:
    F = joint_frames(q)
    return float(np.linalg.norm(F[4][:3, 3] - F[1][:3, 3]))


def elbow_from_distance(d: float) -> dict:
    """q4 from the shoulder-wrist distance alone: q4 = delta +/- arccos((A - d^2) / 2R).

    Both roots are exact. The arm is fully stretched at q4 = delta - pi (about -26.8 deg),
    and on either side of it two elbow angles give the same distance. Joint 4's range
    [-176, -4] deg contains both roots when the elbow is nearly straight (q4 above about
    -49 deg); then the distance alone cannot say which one the arm is in, and `ambiguous` is True.
    """
    c = (ELBOW["A"] - d * d) / (2 * ELBOW["R"])
    c_clipped = float(np.clip(c, -1.0, 1.0))
    roots = [ELBOW["delta"] + np.arccos(c_clipped), ELBOW["delta"] - np.arccos(c_clipped)]
    roots = [float((r + np.pi) % (2 * np.pi) - np.pi) for r in roots]
    lo, hi = JOINT_LIMITS[3]
    inside = [r for r in roots if lo - 1e-9 <= r <= hi + 1e-9]
    return {"cos_term": float(c), "roots": roots, "roots_in_range": inside, "ambiguous": len(inside) > 1,
            "q4_straight": float((ELBOW["delta"] - np.pi + np.pi) % (2 * np.pi) - np.pi)}


def swivel_angle(q: np.ndarray) -> float:
    """Angle of the elbow around the shoulder-wrist line, measured from the vertical plane
    through that line (0 = elbow straight up). This is the freedom the 7th joint adds."""
    F = joint_frames(q)
    S, E, W = F[1][:3, 3], F[3][:3, 3], F[4][:3, 3]
    u = (W - S) / np.linalg.norm(W - S)
    e = E - S
    e_perp = e - (e @ u) * u
    ref = np.array([0.0, 0.0, 1.0]) - u[2] * u
    if np.linalg.norm(ref) < 1e-9:                  # shoulder-wrist line vertical: pick x instead
        ref = np.array([1.0, 0.0, 0.0]) - u[0] * u
    ref /= np.linalg.norm(ref)
    return float(np.arctan2(u @ np.cross(ref, e_perp), ref @ e_perp))


# ---------------------------------------------------------------- the session
def _target_T(x, y, z, yaw) -> np.ndarray:
    """Gripper pointing straight down, rotated by yaw about the vertical (as in all the policies)."""
    R = rot_z(yaw) @ np.diag([1.0, -1.0, -1.0])
    return make_T(R, np.array([x, y, z]))


def _yaw_of(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


class SliderSession:
    def __init__(self, seed: int = 0):
        self.env = PickPlaceEnv()
        self.arm_dofs = arm_dof_indices(self.env.model)
        self.substeps = max(1, int(round(0.01 / self.env.model.opt.timestep)))
        self.reset(seed)

    # ---- commands ----
    def reset(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.env.reset(self.seed)
        self.q_cmd = Q_HOME.copy()
        self.gripper = GRIPPER_OPEN
        self.mode = "cartesian"
        T = self.env.tcp_pose()
        self.target = {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]), "yaw": _yaw_of(T)}
        self.swivel_target = None          # degrees; None = let the IK choose (pull toward home)
        self.last_ik = {"iterations": 0, "trace": [], "success": True, "pos_error_mm": 0.0, "rot_error_deg": 0.0,
                        "e": [0.0] * 6, "dq": [0.0] * 7}

    def set_cartesian(self, x=None, y=None, z=None, yaw=None, gripper=None, swivel=None, free_swivel=False) -> None:
        self.mode = "cartesian"
        for k, v in (("x", x), ("y", y), ("z", z), ("yaw", yaw)):
            if v is not None:
                lo, hi = WORKSPACE[k]
                self.target[k] = float(np.clip(v, lo, hi))
        if free_swivel:
            self.swivel_target = None
        elif swivel is not None:
            self.swivel_target = float(np.clip(swivel, -120.0, 120.0))
        if gripper is not None:
            self.gripper = float(np.clip(gripper, GRIPPER_CLOSED, GRIPPER_OPEN))
        self._solve()

    def set_joints(self, q=None, gripper=None) -> None:
        self.mode = "joint"
        if q is not None:
            self.q_cmd = np.clip(np.asarray(q, dtype=float), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        if gripper is not None:
            self.gripper = float(np.clip(gripper, GRIPPER_CLOSED, GRIPPER_OPEN))
        T = tcp_pose_and_jacobian(self.q_cmd)[0]     # keep the Cartesian sliders in step
        self.target.update(x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3]), yaw=_yaw_of(T))

    def _swivel_step(self, q: np.ndarray, gain: float = 0.5) -> np.ndarray:
        """One null-space step toward the swivel target: dq = gain * N * grad(psi) * (psi* - psi).
        N = I - J+ J removes every part of the step that would move the gripper."""
        psi = swivel_angle(q)
        err = (np.radians(self.swivel_target) - psi + np.pi) % (2 * np.pi) - np.pi
        g = np.empty(7)
        for i in range(7):                                  # gradient of psi by central differences
            h = np.zeros(7)
            h[i] = 1e-5
            g[i] = (swivel_angle(q + h) - swivel_angle(q - h)) / 2e-5
        _, J = tcp_pose_and_jacobian(q)
        N = np.eye(7) - np.linalg.pinv(J) @ J
        step = N @ g
        nrm = step @ g
        if nrm < 1e-9:                                      # swivel not controllable here (singular)
            return q
        return np.clip(q + gain * step * err / nrm, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    def _solve(self) -> None:
        """IK from the last command (warm start). With a swivel target, the IK's own pull toward the
        home posture is switched off and the swivel task is solved in the null space instead."""
        T_goal = _target_T(**self.target)
        q0 = self.q_cmd.copy()
        free = self.swivel_target is None
        if not free:
            for _ in range(5):
                q0 = self._swivel_step(q0)
        trace: list = []
        res = solve_ik(T_goal, q_init=q0, max_iters=60, nullspace_gain=0.05 if free else 0.0, trace=trace)
        T, J = tcp_pose_and_jacobian(res.q)
        e = pose_error(T_goal, T)
        dq = J.T @ np.linalg.solve(J @ J.T + DAMPING**2 * np.eye(6), e)
        self.q_cmd = res.q
        self.last_ik = {"iterations": res.iterations, "trace": [[1000 * p, np.degrees(r)] for p, r in trace],
                        "success": bool(res.success), "pos_error_mm": 1000 * res.pos_error,
                        "rot_error_deg": float(np.degrees(res.rot_error)), "e": e.tolist(), "dq": dq.tolist()}

    # ---- physics ----
    def step(self, control_periods: int = 1) -> None:
        """Advance 10 ms per control period: PD servos toward q_cmd, gravity compensated."""
        m, d = self.env.model, self.env.data
        import mujoco

        for _ in range(control_periods):
            if self.mode == "cartesian":
                self._settle_nullspace()
            d.ctrl[:7] = self.q_cmd
            d.ctrl[7] = self.gripper
            for _ in range(self.substeps):
                d.qfrc_applied[self.arm_dofs] = d.qfrc_bias[self.arm_dofs]
                mujoco.mj_step(m, d)

    def _settle_nullspace(self) -> None:
        """Between slider moves, keep turning the elbow toward the swivel target (a few degrees per
        10 ms, so the motion is smooth), re-solving IK so the gripper stays where it is."""
        if self.swivel_target is None:
            return
        if abs(np.degrees(swivel_angle(self.q_cmd)) - self.swivel_target) < 0.2:
            return
        q = self._swivel_step(self.q_cmd, gain=0.15)
        res = solve_ik(_target_T(**self.target), q_init=q, max_iters=10, nullspace_gain=0.0)
        self.q_cmd = res.q

    # ---- what the interface shows ----
    def q_measured(self) -> np.ndarray:
        return self.env.data.qpos[self.env.arm_qpos].copy()

    def state(self) -> dict:
        import mujoco

        # After mj_step, MuJoCo's body positions describe the state BEFORE the last integration
        # step (0.6 um behind the joints while moving). Recompute them from the current joints.
        mujoco.mj_kinematics(self.env.model, self.env.data)
        q = self.q_measured()
        frames = joint_frames(q)
        links = [_dh_link(i, q[i]) for i in range(7)]
        T7_F = trans(0, 0, FLANGE_D) @ make_T(rot_z(HAND_YAW), np.zeros(3))
        T_F_TCP = trans(0, 0, TCP_OFFSET)
        T_fk = frames[-1] @ T7_F @ T_F_TCP
        T_mj = self.env.tcp_pose()
        _, J = tcp_pose_and_jacobian(q)
        sv = np.linalg.svd(J, compute_uv=False)
        d_sw = shoulder_wrist_distance(q)
        elbow = elbow_from_distance(d_sw)
        cube = self.env.cube_pose()
        tgt = self.env.target_pos()
        T_goal = _target_T(**self.target)
        lim = JOINT_LIMITS
        return {
            "mode": self.mode, "seed": self.seed, "time": float(self.env.data.time),
            "target": {**self.target, "gripper": self.gripper, "swivel": self.swivel_target},
            "q": q.tolist(), "q_cmd": self.q_cmd.tolist(), "limits": lim.tolist(),
            "limit_margin_deg": np.degrees(np.minimum(q - lim[:, 0], lim[:, 1] - q)).tolist(),
            "dh": [{"i": i + 1, "a": float(DH[i, 0]), "alpha": float(DH[i, 2]), "d": float(DH[i, 1]),
                    "theta": float(q[i])} for i in range(7)],
            "links": [L.tolist() for L in links],
            "flange": T7_F.tolist(), "tcp_offset": T_F_TCP.tolist(),
            "frames": [F.tolist() for F in frames],
            "T_fk": T_fk.tolist(), "T_mujoco": T_mj.tolist(),
            "fk_vs_mujoco_um": float(1e6 * np.linalg.norm(T_fk[:3, 3] - T_mj[:3, 3])),
            "tcp_error_to_target_mm": float(1000 * np.linalg.norm(T_mj[:3, 3] - T_goal[:3, 3])),
            "jacobian": J.tolist(), "singular_values": sv.tolist(),
            "manipulability": float(np.prod(sv)), "condition": float(sv[0] / max(sv[-1], 1e-12)),
            "ik": self.last_ik, "damping": DAMPING,
            "elbow": {**ELBOW, "d": d_sw, **elbow, "q4_actual": float(q[3])},
            "swivel_deg": float(np.degrees(swivel_angle(q))),
            "finger_width_mm": 1000 * self.env.finger_width(),
            "cube": {"xyz": cube[:3, 3].tolist(), "yaw_deg": float(np.degrees(self.env.cube_yaw())),
                     "tcp_to_cube_mm": float(1000 * np.linalg.norm(T_mj[:3, 3] - cube[:3, 3]))},
            "goal": {"xyz": tgt.tolist(), "cube_to_goal_mm": float(1000 * np.linalg.norm(cube[:2, 3] - tgt[:2]))},
        }
