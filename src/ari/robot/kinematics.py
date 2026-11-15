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
from ari.robot.transforms import make_T, rot_z, trans

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


_CA, _SA = np.cos(DH[:, 2]), np.sin(DH[:, 2])


def _dh_link(i: int, theta: float) -> np.ndarray:
    """Closed form of RotX(alpha) TransX(a) RotZ(theta) TransZ(d) for link i.
    (Multiplying the four matrices out by hand; ~5x faster than four matrix products,
    which matters once reinforcement learning calls this millions of times.)"""
    a, d = DH[i, 0], DH[i, 1]
    ca, sa, ct, st = _CA[i], _SA[i], np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0.0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0.0, 0.0, 0.0, 1.0]])


def joint_frames(q: np.ndarray) -> list[np.ndarray]:
    """World-frame transforms of joint frames 1..7 (frame i rotates about its z-axis by q_i)."""
    q = np.asarray(q, dtype=float)
    assert q.shape == (7,), f"expected 7 arm joint angles, got {q.shape}"
    T = np.eye(4)
    frames = []
    for i in range(7):
        T = T @ _dh_link(i, q[i])
        frames.append(T)
    return frames


_T7_TO_HAND = trans(0, 0, FLANGE_D) @ make_T(rot_z(HAND_YAW), np.zeros(3))
_T7_TO_TCP = _T7_TO_HAND @ trans(0, 0, TCP_OFFSET)


def tcp_pose_and_jacobian(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TCP pose and 6x7 Jacobian from ONE pass over the joint frames (used by IK)."""
    frames = joint_frames(q)
    T = frames[-1] @ _T7_TO_TCP
    p = T[:3, 3]
    J = np.empty((6, 7))
    for i, F in enumerate(frames):
        z = F[:3, 2]
        J[:3, i] = np.cross(z, p - F[:3, 3])
        J[3:, i] = z
    return T, J


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
