"""Inverse kinematics: find joint angles that put the TCP at a target pose.

Method: damped least squares (also called Levenberg-Marquardt IK). At each
iteration we measure the 6D pose error e (position + orientation), and solve

    dq = J^T (J J^T + lambda^2 I)^-1 e

which is the joint step that best reduces e while keeping |dq| small.
- lambda (damping) trades accuracy for stability near singularities, where
  J J^T becomes nearly non-invertible and plain pseudo-inverse steps explode.
- The Panda has 7 joints but a pose has only 6 numbers, so there are infinitely
  many solutions. We add a small "null-space" step that pulls joints toward a
  preferred posture without disturbing the TCP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ari.robot.kinematics import tcp_jacobian, tcp_pose
from ari.robot.transforms import rot_to_axis_angle

# Panda joint limits (rad), identical to the Menagerie model's.
JOINT_LIMITS = np.array(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ]
)
Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    pos_error: float   # metres
    rot_error: float   # radians


def pose_error(T_target: np.ndarray, T_current: np.ndarray) -> np.ndarray:
    """6D error [position (m); orientation as rotation vector (rad)] in the world frame."""
    dp = T_target[:3, 3] - T_current[:3, 3]
    dR = T_target[:3, :3] @ T_current[:3, :3].T
    return np.concatenate([dp, rot_to_axis_angle(dR)])


def solve_ik(
    T_target: np.ndarray,
    q_init: np.ndarray = Q_HOME,
    *,
    damping: float = 0.05,
    step_scale: float = 1.0,
    max_iters: int = 200,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    nullspace_gain: float = 0.05,
    q_preferred: np.ndarray = Q_HOME,
) -> IKResult:
    """Damped-least-squares IK for the Panda TCP. Experiment with the keyword arguments."""
    q = np.array(q_init, dtype=float)
    for it in range(1, max_iters + 1):
        e = pose_error(T_target, tcp_pose(q))
        pos_err, rot_err = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if pos_err < pos_tol and rot_err < rot_tol:
            return IKResult(q, True, it, pos_err, rot_err)

        J = tcp_jacobian(q)
        JJt = J @ J.T + damping**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)

        # Null-space projection: N = I - J^+ J removes any part of a joint motion
        # that would move the TCP, so this term only reshapes the arm's posture.
        # Note: J^+ must be the *exact* pseudo-inverse. Using the damped one here
        # leaks posture motion into the task and IK stalls at ~0.5 mm error
        # (a real bug found during development; see docs/devlog.md).
        N = np.eye(7) - np.linalg.pinv(J) @ J
        dq += nullspace_gain * N @ (q_preferred - q)

        q = np.clip(q + step_scale * dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    e = pose_error(T_target, tcp_pose(q))
    return IKResult(q, False, max_iters, np.linalg.norm(e[:3]), np.linalg.norm(e[3:]))
