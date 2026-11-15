"""Forward kinematics and the Jacobian of the Panda, written from scratch.

This module deliberately does NOT use MuJoCo. It uses Franka's published
modified Denavit-Hartenberg (Craig convention) parameters, so comparing its
output with MuJoCo's (see tests/test_kinematics.py) is an independent check:
two different descriptions of the same robot must agree.

Modified DH: the transform from frame i-1 to frame i is

    T_{i-1,i} = RotX(alpha_{i-1}) * TransX(a_{i-1}) * RotZ(theta_i) * TransZ(d_i)

where theta_i is the joint angle q_i.
"""

from __future__ import annotations

import numpy as np

from ari.robot.model import TCP_OFFSET
from ari.robot.transforms import make_T, rot_x, rot_z, trans

# Franka Emika Panda, modified DH parameters (a in metres, alpha in radians, d in metres).
#          a_{i-1}   d_i     alpha_{i-1}
DH = np.array(
    [
        [0.0,      0.333,  0.0],
        [0.0,      0.0,   -np.pi / 2],
        [0.0,      0.316,  np.pi / 2],
        [0.0825,   0.0,    np.pi / 2],
        [-0.0825,  0.384, -np.pi / 2],
        [0.0,      0.0,    np.pi / 2],
        [0.088,    0.0,    np.pi / 2],
    ]
)
FLANGE_D = 0.107          # flange sits 0.107 m along joint-7's z-axis
HAND_YAW = -np.pi / 4     # the hand is mounted rotated -45 deg about the flange z-axis


def _dh_link(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_x(alpha)
    T = T @ trans(a, 0, 0)
    T = T @ make_T(rot_z(theta), np.zeros(3))
    T = T @ trans(0, 0, d)
    return T


def joint_frames(q: np.ndarray) -> list[np.ndarray]:
    """World-frame transforms of joint frames 1..7 (frame i rotates about its z-axis by q_i)."""
    q = np.asarray(q, dtype=float)
    assert q.shape == (7,), f"expected 7 arm joint angles, got {q.shape}"
    T = np.eye(4)
    frames = []
    for (a, d, alpha), theta in zip(DH, q):
        T = T @ _dh_link(a, d, alpha, theta)
        frames.append(T.copy())
    return frames


def hand_pose(q: np.ndarray) -> np.ndarray:
    """4x4 world pose of the hand body (matches MuJoCo's 'hand' body frame)."""
    T7 = joint_frames(q)[-1]
    return T7 @ trans(0, 0, FLANGE_D) @ make_T(rot_z(HAND_YAW), np.zeros(3))


def tcp_pose(q: np.ndarray) -> np.ndarray:
    """4x4 world pose of the tool centre point (between the fingertips)."""
    return hand_pose(q) @ trans(0, 0, TCP_OFFSET)


def tcp_jacobian(q: np.ndarray) -> np.ndarray:
    """6x7 geometric Jacobian of the TCP: [v; w] = J @ qdot, in the world frame.

    For a revolute joint i with unit axis z_i through point o_i, moving at rate
    qdot_i, the TCP at p moves with
        linear velocity  = z_i x (p - o_i) * qdot_i
        angular velocity = z_i * qdot_i
    Stacking those columns gives the Jacobian.
    """
    frames = joint_frames(q)
    p = tcp_pose(q)[:3, 3]
    J = np.zeros((6, 7))
    for i, T in enumerate(frames):
        z, o = T[:3, 2], T[:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J
