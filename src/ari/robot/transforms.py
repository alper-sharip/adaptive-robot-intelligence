"""Small rigid-body transform helpers (4x4 homogeneous matrices).

A homogeneous transform T packs a rotation R (3x3) and translation p (3,) as

    T = [[R, p],
         [0, 1]]

so that a point x expressed in frame B maps to frame A by  x_A = T_AB @ [x_B, 1].
Chaining frames is just matrix multiplication:  T_AC = T_AB @ T_BC.
"""

from __future__ import annotations

import numpy as np


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def trans(x: float, y: float, z: float) -> np.ndarray:
    return make_T(np.eye(3), np.array([x, y, z]))


def rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), the 'log map' of SO(3).

    Used for orientation error: the rotation vector of R_target @ R_current.T is
    the smallest rotation that takes the current orientation to the target.
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Near 180 degrees the formula below divides by ~0; use the diagonal instead.
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        # Fix signs using off-diagonal terms.
        if axis[0] > 1e-6:
            axis[1] = np.copysign(axis[1], R[0, 1])
            axis[2] = np.copysign(axis[2], R[0, 2])
        elif axis[1] > 1e-6:
            axis[2] = np.copysign(axis[2], R[1, 2])
        return axis / np.linalg.norm(axis) * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(theta)) * theta


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def axis_angle_to_rot(w: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) -> rotation matrix: Rodrigues' formula, the 'exp map'.

    Inverse of rot_to_axis_angle. Together they let us interpolate rotations:
    R(s) = R0 @ exp(s * log(R0^T R1)) turns smoothly from R0 (s=0) to R1 (s=1)
    about a single fixed axis. This is the matrix form of quaternion slerp.
    """
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
